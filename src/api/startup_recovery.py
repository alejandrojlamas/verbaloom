"""Safe startup recovery for long-running translation jobs.

Only work that was still ``running`` or ``validating`` in a previous server
session is eligible. Manual pauses, credit exhaustion, partial quality results,
and ordinary historical errors remain visible for explicit user action.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.api.services.path_validator import PathValidator
from src.utils.provider_security import sanitize_restored_endpoint_credentials
from src.config import NIM_API_ENDPOINT, OPENAI_API_ENDPOINT


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
            continue

        state_manager.set_translation_field(translation_id, "config", config)
        state_manager.set_translation_field(translation_id, "interrupted", False)
        state_manager.set_translation_field(translation_id, "status", "running")
        checkpoint_manager.mark_running(translation_id)
        checkpoint_manager.update_job_config(translation_id, config)
        start_job(translation_id, config)
        report.resumed_job_ids.append(translation_id)

    return report, resumable_jobs
