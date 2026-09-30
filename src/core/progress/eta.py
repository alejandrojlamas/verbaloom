"""Server-authoritative timing and ETA calculations for long document jobs.

The browser polls more often than chunks finish.  Estimating from the time
between two polls therefore produces wildly optimistic ETAs: most of the time
spent inside the previous chunk is discarded.  This module instead measures
active worker intervals, keeps a conservative per-phase throughput, and
returns an uncertainty range once enough work has completed.
"""

from __future__ import annotations

import time
from typing import Any

_MIN_SAMPLE_UNITS = 3
_TERMINAL_STATUSES = {"completed", "error", "failed", "partial", "interrupted", "rate_limited"}
_WAITING_STATUSES = {"pricing_wait", "provider_wait"}
_FINAL_JOB_PHASES = {"audit", "repair", "assemble", "publish"}
_TIMING_CHECKPOINT_KEY = "eta_timing"
_TIMING_CHECKPOINT_FIELDS = (
    "active_elapsed_seconds",
    "eta_finalization_reserve_seconds",
    "current_phase",
    "enable_refinement",
    "refine_only",
    "phase",
    "job_phase",
    "_eta_phase_key",
    "_eta_phase_started_elapsed",
    "_eta_phase_completed_baseline",
    "_eta_rate_observed_completed",
    "_eta_rate_seconds_per_unit",
    "_eta_fallback_rate_seconds_per_unit",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _phase_key(stats: dict[str, Any]) -> str:
    phase = str(stats.get("phase") or "").strip().lower()
    phase_number = _as_int(stats.get("current_phase"), 1)
    if stats.get("refine_only") or phase_number == 2 or phase == "refining":
        return "refine"
    return "translate"


def timing_checkpoint_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Return the small, JSON-safe timing state needed after a restart."""
    return {
        key: stats[key]
        for key in _TIMING_CHECKPOINT_FIELDS
        if key in stats and isinstance(stats[key], (bool, int, float, str))
    }


def restore_timing_checkpoint(stats: dict[str, Any]) -> dict[str, Any]:
    """Expand persisted timing state into a legacy progress dictionary."""
    current = dict(stats or {})
    timing = current.pop(_TIMING_CHECKPOINT_KEY, None)
    if isinstance(timing, dict):
        current.update({
            key: timing[key]
            for key in _TIMING_CHECKPOINT_FIELDS
            if key in timing and isinstance(timing[key], (bool, int, float, str))
        })
    return current


def begin_active_timing(
    stats: dict[str, Any] | None,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    """Begin a worker interval without counting time spent paused or waiting."""
    current = dict(stats or {})
    started_at = float(now if now is not None else time.time())
    accumulated = max(0.0, _as_float(current.get("active_elapsed_seconds")))
    completed = max(0, _as_int(current.get("completed_chunks")))
    current.update({
        "active_elapsed_before_run": accumulated,
        "active_run_started_at": started_at,
        "active_run_completed_baseline": completed,
        "active_elapsed_seconds": accumulated,
        "elapsed_time": accumulated,
        "elapsed_seconds": accumulated,
        "eta_seconds": None,
        "eta_lower_seconds": None,
        "eta_upper_seconds": None,
        "eta_status": "calculating",
        # A request can hang before its first provider callback.  Give the
        # watchdog and the UI a fresh activity baseline immediately.
        "last_activity_at": started_at,
    })
    return current


def apply_active_timing(
    stats: dict[str, Any],
    *,
    now: float | None = None,
    advance: bool = True,
    status: str | None = None,
) -> dict[str, Any]:
    """Return a timing snapshot with a conservative, phase-aware ETA.

    ``advance=False`` freezes elapsed time for provider/pricing waits and
    terminal states.  Internal ``_eta_*`` fields are deliberately retained in
    the stats dictionary so a pause/resume can preserve the learned rate.
    """
    current = dict(stats or {})
    timestamp = float(now if now is not None else time.time())
    normalized_status = str(status or "").strip().lower()

    base = max(0.0, _as_float(current.get("active_elapsed_before_run")))
    run_started = _as_float(current.get("active_run_started_at"), timestamp)
    run_elapsed = max(0.0, timestamp - run_started) if advance else 0.0
    if advance:
        active_elapsed = base + run_elapsed
    else:
        active_elapsed = max(
            base,
            _as_float(current.get("active_elapsed_seconds")),
            _as_float(current.get("elapsed_time")),
        )

    current.update({
        "active_elapsed_seconds": active_elapsed,
        "elapsed_time": active_elapsed,
        "elapsed_seconds": active_elapsed,
    })

    if normalized_status in _WAITING_STATUSES:
        current.update({
            "eta_seconds": None,
            "eta_lower_seconds": None,
            "eta_upper_seconds": None,
            "eta_status": "waiting",
        })
        return current

    if normalized_status in _TERMINAL_STATUSES:
        completed_status = normalized_status == "completed"
        current.update({
            "eta_seconds": 0.0 if completed_status else None,
            "eta_lower_seconds": 0.0 if completed_status else None,
            "eta_upper_seconds": 0.0 if completed_status else None,
            "eta_status": "complete" if completed_status else "paused",
        })
        return current

    completed = max(0, _as_int(current.get("completed_chunks")))
    total = max(0, _as_int(current.get("total_chunks")))
    phase_key = _phase_key(current)
    prior_phase_key = str(current.get("_eta_phase_key") or "")
    prior_active_elapsed = max(
        0.0,
        _as_float(stats.get("active_elapsed_seconds")),
        _as_float(stats.get("elapsed_time")),
    )

    if prior_phase_key != phase_key:
        # On a phase transition the current phase's counters restart.  Use the
        # last emitted active time as the new baseline so translation time is
        # never charged to refinement (or vice versa).
        baseline_completed = (
            max(0, _as_int(current.get("active_run_completed_baseline")))
            if not prior_phase_key
            else 0
        )
        current["_eta_phase_key"] = phase_key
        current["_eta_phase_started_elapsed"] = prior_active_elapsed
        current["_eta_phase_completed_baseline"] = baseline_completed
        current["_eta_rate_observed_completed"] = baseline_completed
        if prior_phase_key:
            current["_eta_fallback_rate_seconds_per_unit"] = _as_float(
                current.get("_eta_rate_seconds_per_unit")
            )
            current.pop("_eta_rate_seconds_per_unit", None)

    phase_started_elapsed = max(
        0.0,
        _as_float(current.get("_eta_phase_started_elapsed"), base),
    )
    phase_baseline_completed = max(
        0,
        _as_int(
            current.get("_eta_phase_completed_baseline"),
            _as_int(current.get("active_run_completed_baseline")),
        ),
    )
    observed_units = max(0, completed - phase_baseline_completed)
    phase_elapsed = max(0.0, active_elapsed - phase_started_elapsed)
    measured_rate = phase_elapsed / observed_units if observed_units > 0 else 0.0

    learned_rate = max(0.0, _as_float(current.get("_eta_rate_seconds_per_unit")))
    last_rate_completed = max(
        phase_baseline_completed,
        _as_int(current.get("_eta_rate_observed_completed"), phase_baseline_completed),
    )
    if measured_rate > 0 and completed > last_rate_completed:
        if learned_rate <= 0:
            learned_rate = measured_rate
        elif measured_rate >= learned_rate:
            # Slowdowns matter quickly; speedups must prove themselves over
            # several chunks before the estimate becomes more optimistic.
            learned_rate = (learned_rate * 0.5) + (measured_rate * 0.5)
        else:
            learned_rate = (learned_rate * 0.85) + (measured_rate * 0.15)
        current["_eta_rate_seconds_per_unit"] = learned_rate
        current["_eta_rate_observed_completed"] = completed

    fallback_rate = max(
        0.0,
        _as_float(current.get("_eta_fallback_rate_seconds_per_unit")),
    )
    # While one chunk is taking longer than usual, the cumulative measured
    # rate rises even before it completes.  Taking the maximum prevents a
    # stalled request from leaving a frozen, implausibly short ETA.
    effective_rate = max(learned_rate, measured_rate, fallback_rate)

    job_phase = str(current.get("job_phase") or "").strip().lower()
    if total > 0 and completed >= total:
        current.update({
            "eta_seconds": None,
            "eta_lower_seconds": None,
            "eta_upper_seconds": None,
            "eta_status": "finalizing",
        })
        return current
    if job_phase in _FINAL_JOB_PHASES:
        current.update({
            "eta_seconds": None,
            "eta_lower_seconds": None,
            "eta_upper_seconds": None,
            "eta_status": "finalizing",
        })
        return current

    has_enough_evidence = observed_units >= _MIN_SAMPLE_UNITS or (
        learned_rate > 0 and last_rate_completed > phase_baseline_completed
    )
    if total <= 0 or completed < 0 or effective_rate <= 0 or not has_enough_evidence:
        current.update({
            "eta_seconds": None,
            "eta_lower_seconds": None,
            "eta_upper_seconds": None,
            "eta_status": "calculating",
        })
        return current

    remaining_units = max(0, total - completed)
    if phase_key == "translate" and current.get("enable_refinement"):
        # A separate refine-after pass resets its counters, but it is real work
        # that the old ETA omitted entirely.
        remaining_units += total

    work_seconds = effective_rate * remaining_units
    token_eta = max(0.0, _as_float(current.get("estimated_remaining_seconds")))
    if token_eta > 0 and phase_key == "translate":
        work_seconds = max(work_seconds, token_eta)

    reserve_floor = max(
        0.0,
        _as_float(current.get("eta_finalization_reserve_seconds")),
    )
    reserve = max(reserve_floor, min(600.0, work_seconds * 0.05))
    estimate = max(0.0, work_seconds + reserve)

    progress_ratio = completed / total if total else 0.0
    if observed_units >= 20 and progress_ratio >= 0.15:
        confidence = "high"
        lower_factor, upper_factor = 0.90, 1.25
    elif observed_units >= 5:
        confidence = "medium"
        lower_factor, upper_factor = 0.80, 1.45
    else:
        confidence = "low"
        lower_factor, upper_factor = 0.70, 1.75

    lower = max(0.0, (work_seconds * lower_factor) + (reserve * 0.5))
    upper = max(estimate, (work_seconds * upper_factor) + (reserve * 1.5))
    current.update({
        "eta_seconds": estimate,
        "eta_lower_seconds": lower,
        "eta_upper_seconds": upper,
        "eta_confidence": confidence,
        "eta_status": "estimated",
        "eta_basis": "server_active_phase_throughput",
        "eta_sample_units": observed_units,
    })
    return current
