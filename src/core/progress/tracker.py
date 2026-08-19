"""The single authority for progress totals, phase, and percent math.

``ProgressTracker`` is deliberately ignorant of file formats. A pipeline:

1. constructs a tracker (declaring whether a refine phase will follow),
2. calls :meth:`begin_phase` with the unit count for the phase about to run,
3. calls :meth:`record_unit` once per processed unit,
4. emits :meth:`snapshot` to whoever reports progress.

The segment math lives in :func:`global_percent` (shared with the legacy
bridge): a single model where translation fills ``[0, split]`` and refinement
fills ``[split, 100]``, replacing the old scattered "double ``total_chunks``
when refining" trick plus client-side 0–50/50–100 remapping. Per-phase totals
may differ (e.g. SRT refines per subtitle but translates per block) — each
phase is scaled into its own segment independently, so that divergence no
longer breaks the bar.

This tracker is the intended driver for the engines, but it is **not** yet
wired into the pipeline: production reaches the same math through
:func:`snapshot_from_legacy_stats`. Today the tracker backs the unit tests and
documents the target contract; see ``progress/__init__.py`` for the wiring
status.
"""

from __future__ import annotations

import time
from typing import Callable

from .snapshot import Phase, ProgressSnapshot, global_percent

_DEFAULT_SPLIT = 50.0


class ProgressTracker:
    def __init__(
        self,
        *,
        refinement_enabled: bool = False,
        refine_split: float = _DEFAULT_SPLIT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < refine_split < 100.0:
            raise ValueError("refine_split must be strictly between 0 and 100")
        self._refinement_enabled = refinement_enabled
        self._split = refine_split
        self._clock = clock
        self._start = clock()

        self._phase = Phase.TRANSLATING
        self._total = 0
        self._done = 0
        self._succeeded = 0
        self._failed = 0
        # Monotonic floor: the bar is guaranteed never to regress, by
        # construction, regardless of how callers drive the tracker.
        self._max_percent = 0.0

    # -- driving the tracker -------------------------------------------------

    def begin_phase(self, phase: Phase, total_units: int) -> None:
        """Start a phase with its unit count, resetting per-phase counters."""
        if total_units < 0:
            raise ValueError("total_units must be non-negative")
        if phase is Phase.REFINING and not self._refinement_enabled:
            raise ValueError(
                "cannot begin REFINING phase on a tracker created with "
                "refinement_enabled=False"
            )
        self._phase = phase
        self._total = total_units
        self._done = 0
        self._succeeded = 0
        self._failed = 0

    def record_unit(self, *, succeeded: bool = True, count: int = 1) -> None:
        """Record ``count`` processed units in the current phase."""
        if count < 0:
            raise ValueError("count must be non-negative")
        self._done += count
        if succeeded:
            self._succeeded += count
        else:
            self._failed += count

    # -- reading the tracker -------------------------------------------------

    def _raw_percent(self) -> float:
        return global_percent(
            self._phase,
            self._done,
            self._total,
            self._refinement_enabled,
            self._split,
        )

    def snapshot(self) -> ProgressSnapshot:
        """Produce the current immutable snapshot (monotonic percent)."""
        pct = max(self._max_percent, self._raw_percent())
        self._max_percent = pct

        elapsed = self._clock() - self._start
        eta = None
        if 0.0 < pct < 100.0:
            eta = elapsed * (100.0 - pct) / pct

        return ProgressSnapshot(
            phase=self._phase,
            units_done=self._done,
            units_total=self._total,
            units_succeeded=self._succeeded,
            units_failed=self._failed,
            percent=pct,
            refinement_enabled=self._refinement_enabled,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )
