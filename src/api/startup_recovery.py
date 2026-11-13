"""Safe startup recovery for long-running translation jobs.

Only work that was still ``queued``, ``running`` or ``validating`` in a previous server
session is eligible. Manual pauses, credit exhaustion, partial quality results,
and ordinary historical errors remain visible for explicit user action.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.api.resume_schedule import resume_schedule_from_config
from src.api.services.path_validator import PathValidator
from src.config import NIM_API_ENDPOINT, OPENAI_API_ENDPOINT
from src.core.recovery_policy import (
    recovery_budget,
    recovery_counter,
    recovery_limit,
    recovery_stop_reason,
    reserve_recovery,
)
from src.utils.provider_security import sanitize_restored_endpoint_credentials

_KEYED_ENDPOINT_DEFAULTS = {
    "openai": OPENAI_API_ENDPOINT,
    "nim": NIM_API_ENDPOINT,
}


@dataclass
class StartupRecoveryReport:
    reset_count: int = 0
    restored_count: int = 0
    resumed_job_ids: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)


def build_startup_resume_config(
    checkpoint_data: dict[str, Any] | None,
    checkpoint_manager: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Build a resume config without reviving a deliberate pause."""
    if not checkpoint_data:
        return None, "checkpoint missing"

    job = checkpoint_data.get("job") or {}
    config = copy.deepcopy(job.get("config") or {})
    if config.get("_manual_pause_requested"):
        return None, "manual pause requested"
    if config.get("_provider_action_required") or config.get("_automatic_recovery_stop"):
        return None, "action required before automatic recovery"
    if str(job.get("status") or "").strip().lower() == "completed":
        return None, "job already completed"

    if config.get("text") and str(config.get("file_type") or "").lower() == "txt":
        pass
    else:
        preserved_path = str(config.get("preserved_input_path") or "").strip()
        if not preserved_path:
            fallback = checkpoint_manager.get_preserved_input_path(
                str(job.get("translation_id") or "")
            )
            preserved_path = str(fallback or "").strip()
        if not preserved_path:
            return None, "preserved source missing"
        checkpoint_uploads = getattr(checkpoint_manager, "uploads_dir", None)
        if checkpoint_uploads:
            translation_id = str(job.get("translation_id") or "")
            translation_id_ok, _ = PathValidator.validate_filename(translation_id)
            if not translation_id_ok:
                return None, "invalid checkpoint job identifier"
            try:
                preserved_file = PathValidator.resolve_managed_file(
                    preserved_path,
                    [Path(checkpoint_uploads) / translation_id],
                )
            except (ValueError, FileNotFoundError):
                return None, "preserved source outside managed checkpoint storage"
        else:
            # Compatibility for lightweight embedders and test doubles.  The
            # production checkpoint manager always exposes ``uploads_dir``.
            preserved_file = Path(preserved_path)
            if not preserved_file.is_file():
                return None, "preserved source missing"
        config["file_path"] = str(preserved_file)

    sanitize_restored_endpoint_credentials(config, _KEYED_ENDPOINT_DEFAULTS)
    # A persisted wait has already reserved its handoff; restarting its waiter
    # is not another failed attempt. A dead running worker does consume one.
    if resume_schedule_from_config(config) is None:
        total = (job.get("progress") or {}).get("total_chunks", 0)
        reason = recovery_stop_reason(config, total)
        if reason:
            return None, reason
        index = recovery_limit(checkpoint_data.get("resume_from_index"), 0)
        last_index = recovery_counter(config.get("_startup_recovery_last_index"))
        stuck_count = (
            1 if index > last_index else
            recovery_counter(config.get("_startup_recovery_stuck_count")) + 1
        )
        maximum = recovery_limit(config.get("max_startup_recoveries"), 3)
        if stuck_count > maximum:
            return None, "startup_recovery_exhausted"
        config = reserve_recovery(config, "startup", total)
        assert config is not None
        config.update({
            "_startup_recovery_last_index": max(index, last_index),
            "_startup_recovery_stuck_count": stuck_count,
            "max_startup_recoveries": maximum,
        })
    config["is_resume"] = True
    config["resume_from_index"] = max(
        0,
        int(checkpoint_data.get("resume_from_index") or 0),
    )
    config["_startup_auto_resume"] = True
    return config, ""


def restore_jobs_after_restart(
    state_manager: Any,
    start_job: Callable[[str, dict[str, Any]], None],
    *,
    enabled: bool = True,
    max_auto_resumes: int = 1,
) -> tuple[StartupRecoveryReport, list[dict[str, Any]]]:
    """Restore resumable jobs and restart the most recent stale active job.

    The application intentionally runs at most one automatic recovery at once.
    Any additional stale jobs remain safely resumable in the UI instead of
    creating concurrent long-book requests after a machine restart.
    """
    checkpoint_manager = state_manager.checkpoint_manager
    stale_ids = checkpoint_manager.get_stale_active_job_ids()
    report = StartupRecoveryReport()
    report.reset_count = checkpoint_manager.reset_running_jobs_on_startup()
    resumable_jobs = state_manager.get_resumable_jobs()

    jobs_by_id = {
        str(job.get("translation_id") or ""): job
        for job in resumable_jobs
        if job.get("translation_id")
    }
    for translation_id in jobs_by_id:
        if state_manager.restore_job_from_checkpoint(translation_id):
            report.restored_count += 1

    if not enabled or max_auto_resumes <= 0:
        return report, resumable_jobs

    for translation_id in stale_ids:
        if len(report.resumed_job_ids) >= max_auto_resumes:
            report.skipped[translation_id] = "another stale job is already recovering"
            continue
        if translation_id not in jobs_by_id:
            report.skipped[translation_id] = "job is no longer resumable"
            continue
        checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
        config, reason = build_startup_resume_config(
            checkpoint_data,
            checkpoint_manager,
        )
        if config is None:
            report.skipped[translation_id] = reason
            if reason in {"automatic_recovery_budget_exhausted", "startup_recovery_exhausted"}:
                job = (checkpoint_data or {}).get("job") or {}
                stopped_config = copy.deepcopy(job.get("config") or {})
                stopped_config["_automatic_recovery_stop"] = {
                    "reason": reason,
                    "scope": "startup",
                    "budget": recovery_budget(stopped_config, (job.get("progress") or {}).get("total_chunks", 0)),
                }
                checkpoint_manager.update_job_config(translation_id, stopped_config)
                state_manager.set_translation_field(translation_id, "config", stopped_config)
                state_manager.set_translation_field(translation_id, "pause_reason", reason)
            continue

        # Persist the reservation before marking active or starting a worker.
        if not checkpoint_manager.update_job_config(translation_id, config):
            report.skipped[translation_id] = "could not persist recovery reservation"
            continue
        state_manager.set_translation_field(translation_id, "config", config)
        state_manager.set_translation_field(translation_id, "interrupted", False)
        state_manager.set_translation_field(translation_id, "status", "running")
        checkpoint_manager.mark_running(translation_id)
        start_job(translation_id, config)
        report.resumed_job_ids.append(translation_id)

    return report, resumable_jobs
