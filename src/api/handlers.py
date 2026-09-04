"""
Translation job handlers and processing logic
"""
import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.api.safe_payloads import client_safe_log_entry
from src.api.services.path_validator import PathValidator
from src.config import AUTO_PAUSE_ON_RATE_LIMIT, RATE_LIMIT_AUTO_RESUME_DELAY
from src.core.adapters import refine_file, translate_file
from src.core.audiobook_sanitizer import (
    audiobook_profile_enabled,
    sanitize_for_audiobook,
)
from src.core.editorial_quality import editorial_report_path
from src.core.epub.audiobook_companion import create_structured_audiobook_epub
from src.core.epub.metadata_localizer import (
    infer_epub_title_page,
    localize_epub_metadata,
)
from src.core.epub.missing_block_repair import repair_epub_missing_blocks_with_llm
from src.core.epub.publication_gate import audit_epub_publication
from src.core.fidelity_supervisor import fidelity_report_path
from src.core.final_artifact_audit import audit_and_clean_final_artifact
from src.core.final_source_sample_audit import audit_final_output_against_source_samples
from src.core.job_engine import JobEngine, JobPhase
from src.core.job_runtime_config import (
    configure_editorial_guard_options as _configure_editorial_guard_options,
)
from src.core.job_runtime_config import (
    ensure_layout_sanitizer_options as _ensure_layout_sanitizer_options,
)
from src.core.job_runtime_config import (
    resume_requires_layout_sanitizer_restart as _resume_requires_layout_sanitizer_restart,
)
from src.core.job_runtime_config import usage_process_type as _usage_process_type
from src.core.job_runtime_config import (
    uses_text_first_pipeline as _uses_text_first_pipeline,
)
from src.core.job_runtime_config import (
    with_source_guard_refs as _with_source_guard_refs,
)
from src.core.literary_continuity import literary_continuity_report_path
from src.core.llm import OpenRouterProvider
from src.core.llm.exceptions import (
    DeepSeekPeakPricingError,
    InsufficientCreditsError,
    RateLimitError,
)
from src.core.llm.factory import create_llm_provider
from src.core.llm.request_deadline import await_llm_call
from src.core.output_formats import (
    build_epub_structure_candidates,
    convert_output_file,
    ensure_output_extension,
    extract_epub_toc_titles,
    extract_readable_text,
    format_extension,
    native_output_format,
    normalize_output_format,
    requested_format_for_job,
    write_text_as_output,
)
from src.core.progress import snapshot_from_legacy_stats
from src.core.quality_assurance import run_quality_assurance
from src.core.usage import set_usage_context
from src.persistence.checkpoint_reconcile import checkpoint_progress_snapshot
from src.tts.tts_config import TTSConfig
from src.utils.custom_instructions import is_safe_filename, load_custom_instructions
from src.utils.file_utils import (
    find_partial_output_paths,
    generate_tts_for_translation,
    get_partial_output_path,
    get_unique_output_path,
)
from src.utils.json_extraction import loads_first_json_object
from src.utils.notifier import EVENT_FAILURE, EVENT_INTERRUPTION, EVENT_SUCCESS, notify
from src.utils.unified_logger import LogType, setup_web_logger

from .websocket import emit_update


_AUTOMATIC_RECOVERY_INSTRUCTION = """AUTOMATIC FAILED-CHUNK RECOVERY
The previous attempt did not produce a candidate that passed the local quality
contract. Translate the complete source unit into the requested target language.
Preserve every fact, name, number, paragraph boundary and structural placeholder.
Do not return source-language prose, a summary, an excerpt, or an explanation.
""".strip()

_FAILED_CHUNK_RECOVERY_POLICY_VERSION = 2
_DEFAULT_MAX_STUCK_RECOVERIES = 2
_DEFAULT_MAX_FINALIZATION_RECOVERIES = 2
_DEFAULT_MAX_WORKER_RECOVERIES = 2
_DEFAULT_MAX_RATE_LIMIT_AUTO_RESUMES = 5


def _strip_legacy_automatic_recovery_config(
    config: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    """Remove the legacy retry metadata that changed EPUB prompt fingerprints.

    Recovery state belongs to the worker, not to the editorial prompt.  Older
    workers injected an instruction and a cycle counter into ``prompt_options``;
    every cycle therefore invalidated the current XHTML checkpoint and could
    restart an already translated file from chunk zero.
    """
    cleaned = dict(config or {})
    prompt_options = dict(cleaned.get("prompt_options") or {})
    changed = False

    for key in ("automatic_failure_recovery", "automatic_failure_recovery_cycle"):
        if key in prompt_options:
            prompt_options.pop(key, None)
            changed = True

    custom_instructions = str(prompt_options.get("custom_instructions") or "")
    if _AUTOMATIC_RECOVERY_INSTRUCTION in custom_instructions:
        custom_instructions = custom_instructions.replace(
            _AUTOMATIC_RECOVERY_INSTRUCTION,
            "",
        ).strip()
        if custom_instructions:
            prompt_options["custom_instructions"] = custom_instructions
        else:
            prompt_options.pop("custom_instructions", None)
        changed = True

    try:
        previous_policy = int(
            cleaned.get("_failed_chunk_recovery_policy_version") or 0
        )
    except (TypeError, ValueError):
        previous_policy = 0
    if changed and previous_policy < _FAILED_CHUNK_RECOVERY_POLICY_VERSION:
        for key in (
            "_failure_recovery_cycle",
            "_failure_recovery_stuck_count",
            "_failure_recovery_last_index",
            "_failure_recovery_last_completed",
        ):
            cleaned.pop(key, None)

    cleaned["prompt_options"] = prompt_options
    cleaned["_failed_chunk_recovery_policy_version"] = (
        _FAILED_CHUNK_RECOVERY_POLICY_VERSION
    )
    return cleaned, changed


def _config_flag(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _begin_active_run_stats(
    stats: Dict[str, Any] | None,
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Start a processing-time interval without counting paused wall time."""
    current = dict(stats or {})
    started_at = float(now if now is not None else time.time())
    try:
        accumulated = max(0.0, float(current.get("active_elapsed_seconds") or 0.0))
    except (TypeError, ValueError):
        accumulated = 0.0
    try:
        completed = max(0, int(current.get("completed_chunks") or 0))
    except (TypeError, ValueError):
        completed = 0
    current.update({
        "active_elapsed_before_run": accumulated,
        "active_run_started_at": started_at,
        "active_run_completed_baseline": completed,
        "active_elapsed_seconds": accumulated,
        "elapsed_time": accumulated,
        "elapsed_seconds": accumulated,
        "eta_seconds": None,
    })
    return current


def _apply_active_timing(
    stats: Dict[str, Any],
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Attach active elapsed time and a throughput-based ETA to one snapshot."""
    current = dict(stats or {})
    timestamp = float(now if now is not None else time.time())
    try:
        base = max(0.0, float(current.get("active_elapsed_before_run") or 0.0))
        run_started = float(current.get("active_run_started_at") or timestamp)
    except (TypeError, ValueError):
        base = 0.0
        run_started = timestamp
    run_elapsed = max(0.0, timestamp - run_started)
    active_elapsed = base + run_elapsed

    try:
        completed = max(0, int(current.get("completed_chunks") or 0))
        baseline = max(0, int(current.get("active_run_completed_baseline") or 0))
        total = max(0, int(current.get("total_chunks") or 0))
    except (TypeError, ValueError):
        completed = baseline = total = 0
    run_completed = max(0, completed - baseline)
    eta = None
    if run_completed > 0 and total > completed and run_elapsed > 0:
        eta = (run_elapsed / run_completed) * (total - completed)

    current.update({
        "active_elapsed_seconds": active_elapsed,
        "elapsed_time": active_elapsed,
        "elapsed_seconds": active_elapsed,
        "eta_seconds": eta,
    })
    return current


def _live_activity_label(event: str, message: str = "", data: Any = None) -> str:
    """Return a compact user-facing status for long intra-chunk work."""
    data_type = data.get("type") if isinstance(data, dict) else ""
    text = f"{event} {data_type} {message}".lower()
    if "failed_chunk_auto_recovery" in text:
        return "Reanudación automática del fragmento fallido programada"
    if "target_language" in text and ("repair" in text or "retry" in text):
        return "Reparando el idioma del fragmento"
    if "fidelity" in text or "combined_quality_audit" in text:
        return "Auditando fidelidad contra la fuente"
    if "profile_repair" in text:
        return "Reparando el perfil editorial"
    if "profile_audit" in text:
        return "Auditando el perfil editorial"
    if "phase2" in text or "alignment" in text or "placeholder" in text:
        return "Reconstruyendo estructura y marcadores"
    if "retry" in text or "translation_attempt" in text:
        return "Reintentando el fragmento automáticamente"
    if "checkpoint" in text:
        return "Guardando avance recuperable"
    if "refinement" in text:
        return "Aplicando revisión editorial"
    if "llm_request" in text or "sending request" in text:
        return "Consultando el modelo"
    if "llm_response" in text or "response received" in text:
        return "Validando la respuesta del modelo"
    return ""


def _build_failed_chunk_recovery_plan(
    config: Dict[str, Any],
    checkpoint_data: Dict[str, Any] | None,
    stats: Dict[str, Any],
) -> Dict[str, Any] | None:
    """Build one bounded, non-recursive retry for an unresolved quality unit."""
    if not checkpoint_data or not _config_flag(config.get("auto_recover_failed_chunks"), True):
        return None
    config, _ = _strip_legacy_automatic_recovery_config(config)
    checkpoint_progress = (
        (checkpoint_data.get("job") or {}).get("progress") or {}
    )
    checkpoint_failures = max(
        int(checkpoint_progress.get("failed_chunks") or 0),
        len(checkpoint_data.get("failed_chunk_indices") or []),
    )
    # Assembly/publication errors need their own repair path. Only keep cycling
    # the translation worker when the checkpoint proves that a chunk is pending.
    if checkpoint_failures <= 0:
        return None
    try:
        resume_index = max(0, int(checkpoint_data.get("resume_from_index") or 0))
        completed = max(0, int(stats.get("completed_chunks") or 0))
    except (TypeError, ValueError):
        return None

    try:
        previous_index = int(config.get("_failure_recovery_last_index"))
        previous_completed = int(config.get("_failure_recovery_last_completed"))
        same_position = previous_index == resume_index and previous_completed == completed
    except (TypeError, ValueError):
        same_position = False
    stuck_count = (
        int(config.get("_failure_recovery_stuck_count") or 0) + 1
        if same_position else 1
    )
    try:
        configured_max = config.get("max_stuck_failed_chunk_recoveries")
        if configured_max is None:
            configured_max = _DEFAULT_MAX_STUCK_RECOVERIES
        max_stuck_recoveries = max(0, int(configured_max))
    except (TypeError, ValueError):
        max_stuck_recoveries = _DEFAULT_MAX_STUCK_RECOVERIES
    if stuck_count > max_stuck_recoveries:
        return None

    try:
        cycle = int(config.get("_failure_recovery_cycle") or 0) + 1
    except (TypeError, ValueError):
        cycle = 1
    delay_steps = (2, 10)
    delay_seconds = delay_steps[min(stuck_count - 1, len(delay_steps) - 1)]

    new_config = dict(config)
    new_config.update({
        "is_resume": True,
        "resume_from_index": resume_index,
        "auto_recover_failed_chunks": True,
        "max_stuck_failed_chunk_recoveries": max_stuck_recoveries,
        "_failure_recovery_cycle": cycle,
        "_failure_recovery_stuck_count": stuck_count,
        "_failure_recovery_last_index": resume_index,
        "_failure_recovery_last_completed": completed,
    })
    return {
        "config": new_config,
        "scope": "chunks",
        "resume_index": resume_index,
        "completed_chunks": completed,
        "cycle": cycle,
        "stuck_count": stuck_count,
        "max_stuck_recoveries": max_stuck_recoveries,
        "delay_seconds": delay_seconds,
    }


def _build_finalization_recovery_plan(
    config: Dict[str, Any],
    checkpoint_data: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Retry assembly and whole-book gates without retranslating completed chunks."""
    if (
        not checkpoint_data
        or not checkpoint_data.get("checkpoint_complete")
        or not _config_flag(config.get("auto_recover_finalization"), True)
        or config.get("_manual_pause_requested")
    ):
        return None

    checkpoint_progress = (
        (checkpoint_data.get("job") or {}).get("progress") or {}
    )
    checkpoint_failures = max(
        int(checkpoint_progress.get("failed_chunks") or 0),
        len(checkpoint_data.get("failed_chunk_indices") or []),
    )
    if checkpoint_failures > 0:
        return None

    try:
        configured_max = config.get("max_stuck_finalization_recoveries")
        if configured_max is None:
            configured_max = _DEFAULT_MAX_FINALIZATION_RECOVERIES
        max_recoveries = max(0, int(configured_max))
        cycle = int(config.get("_finalization_recovery_cycle") or 0) + 1
        resume_index = max(0, int(checkpoint_data.get("resume_from_index") or 0))
    except (TypeError, ValueError):
        return None
    if cycle > max_recoveries:
        return None

    delay_steps = (2, 10)
    delay_seconds = delay_steps[min(cycle - 1, len(delay_steps) - 1)]
    new_config = dict(config)
    new_config.update({
        "is_resume": True,
        "resume_from_index": resume_index,
        "auto_recover_finalization": True,
        "max_stuck_finalization_recoveries": max_recoveries,
        "_finalization_recovery_cycle": cycle,
    })
    return {
        "config": new_config,
        "scope": "finalization",
        "resume_index": resume_index,
        "cycle": cycle,
        "stuck_count": cycle,
        "max_stuck_recoveries": max_recoveries,
        "delay_seconds": delay_seconds,
    }


def _finalization_recovery_exhausted_stats(
    stats: Dict[str, Any],
    *,
    now: float | None = None,
) -> Dict[str, Any]:
    """Expose a bounded finalization stop without calling it a failed chunk."""
    current = dict(stats or {})
    current.update({
        "live_status": (
            "Texto completo guardado; la validación final requiere revisión"
        ),
        "live_status_kind": "recoverable",
        "live_activity_event": "finalization_recovery_exhausted",
        "last_activity_at": float(now if now is not None else time.time()),
    })
    return current


def _build_worker_exception_recovery_plan(
    config: Dict[str, Any],
    checkpoint_data: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Recover a crashed worker from durable progress, with a strict ceiling."""
    if (
        not checkpoint_data
        or not _config_flag(config.get("auto_recover_worker_errors"), True)
        or config.get("_manual_pause_requested")
    ):
        return None
    progress = ((checkpoint_data.get("job") or {}).get("progress") or {})
    chunks = checkpoint_data.get("chunks") or []
    if int(progress.get("total_chunks") or 0) <= 0 and not chunks:
        return None
    try:
        configured_max = config.get("max_worker_error_recoveries")
        if configured_max is None:
            configured_max = _DEFAULT_MAX_WORKER_RECOVERIES
        max_recoveries = max(0, int(configured_max))
        cycle = int(config.get("_worker_exception_recovery_cycle") or 0) + 1
        resume_index = max(0, int(checkpoint_data.get("resume_from_index") or 0))
    except (TypeError, ValueError):
        return None
    try:
        previous_index = int(config.get("_worker_exception_last_index"))
        previous_stuck = max(
            0,
            int(config.get("_worker_exception_stuck_count") or 0),
        )
    except (TypeError, ValueError):
        previous_index = None
        previous_stuck = 0
    stuck_count = previous_stuck + 1 if previous_index == resume_index else 1
    if stuck_count > max_recoveries:
        return None

    delay_steps = (3, 15)
    delay_seconds = delay_steps[min(cycle - 1, len(delay_steps) - 1)]
    new_config = dict(config)
    new_config.update({
        "is_resume": True,
        "resume_from_index": resume_index,
        "auto_recover_worker_errors": True,
        "max_worker_error_recoveries": max_recoveries,
        "_worker_exception_recovery_cycle": cycle,
        "_worker_exception_last_index": resume_index,
        "_worker_exception_stuck_count": stuck_count,
    })
    return {
        "config": new_config,
        "scope": "worker",
        "resume_index": resume_index,
        "cycle": cycle,
        "stuck_count": stuck_count,
        "max_stuck_recoveries": max_recoveries,
        "delay_seconds": delay_seconds,
    }


def _build_rate_limit_auto_resume_plan(
    config: Dict[str, Any],
    *,
    resume_index: int,
) -> Dict[str, Any]:
    """Return a durable, bounded provider-throttle recovery decision."""
    try:
        max_resumes = max(
            0,
            int(
                config.get("max_rate_limit_auto_resumes")
                if config.get("max_rate_limit_auto_resumes") is not None
                else _DEFAULT_MAX_RATE_LIMIT_AUTO_RESUMES
            ),
        )
    except (TypeError, ValueError):
        max_resumes = _DEFAULT_MAX_RATE_LIMIT_AUTO_RESUMES
    try:
        last_resume_index = int(config.get("_auto_resume_last_index"))
    except (TypeError, ValueError):
        last_resume_index = None
    try:
        previous_stuck_count = max(0, int(config.get("_auto_resume_stuck_count") or 0))
    except (TypeError, ValueError):
        previous_stuck_count = 0
    stuck_count = (
        previous_stuck_count + 1
        if last_resume_index == resume_index
        else 1
    )
    new_config = dict(config)
    new_config.update({
        "is_resume": True,
        "resume_from_index": resume_index,
        "_auto_resume_last_index": resume_index,
        "_auto_resume_stuck_count": stuck_count,
        "max_rate_limit_auto_resumes": max_resumes,
    })
    return {
        "allowed": stuck_count <= max_resumes,
        "config": new_config,
        "stuck_count": stuck_count,
        "max_resumes": max_resumes,
        "resume_index": resume_index,
    }


def _build_pricing_auto_resume_plan(
    config: Dict[str, Any],
    *,
    resume_index: int,
) -> Dict[str, Any]:
    """Build an unbounded-by-429-budget resume at a known pricing boundary."""
    new_config = dict(config)
    new_config.update({
        "is_resume": True,
        "resume_from_index": int(resume_index),
    })
    new_config.pop("_pricing_pause_until_utc", None)
    new_config.pop("_pricing_pause_timezone", None)
    return {
        "allowed": True,
        "config": new_config,
        "stuck_count": 0,
        "max_resumes": 0,
        "resume_index": int(resume_index),
    }


def _failed_chunk_recovery_exhausted_stats(
    stats: Dict[str, Any],
    *,
    failed_chunks: int,
    now: float | None = None,
) -> Dict[str, Any]:
    """Make a bounded recovery stop explicit instead of leaving stale activity."""
    current = dict(stats or {})
    failed = max(1, int(failed_chunks or 0))
    current.update({
        "failed_chunks": failed,
        "live_status": (
            f"Avance guardado; {failed} fragmento(s) requieren reparación"
        ),
        "live_status_kind": "recoverable",
        "live_activity_event": "failed_chunk_recovery_exhausted",
        "last_activity_at": float(now if now is not None else time.time()),
    })
    return current


def _schedule_failed_chunk_recovery(
    translation_id: str,
    plan: Dict[str, Any],
    state_manager: Any,
    output_dir: str,
    socketio: Any,
) -> bool:
    """Schedule a fresh recovery worker without growing the worker stack."""
    if not state_manager.exists(translation_id):
        return False
    if state_manager.get_translation_field(translation_id, 'recovery_scheduled'):
        return True

    state_manager.set_translation_field(translation_id, 'recovery_scheduled', True)
    state_manager.set_translation_field(translation_id, 'config', plan['config'])

    def _restart() -> None:
        if not state_manager.exists(translation_id):
            return
        if state_manager.get_translation_field(translation_id, 'interrupted'):
            state_manager.set_translation_field(translation_id, 'recovery_scheduled', False)
            return
        if str(state_manager.get_translation_field(translation_id, 'status') or '') != 'running':
            state_manager.set_translation_field(translation_id, 'recovery_scheduled', False)
            return
        state_manager.set_translation_field(translation_id, 'recovery_scheduled', False)
        start_translation_job(
            translation_id,
            plan['config'],
            state_manager,
            output_dir,
            socketio,
        )

    timer = threading.Timer(float(plan['delay_seconds']), _restart)
    timer.daemon = True
    timer.start()
    return True


def _notification_context(config, translation_id, elapsed_time, error=None):
    """Build the context dict passed to webhook notifications."""
    ctx = {
        'translation_id': translation_id,
        'file': config.get('original_filename') or config.get('input_filename') or config.get('file_path'),
        'output': config.get('output_filename'),
        'duration_seconds': elapsed_time,
        'provider': config.get('llm_provider'),
        'model': config.get('model'),
        'source_lang': config.get('source_language'),
        'target_lang': config.get('target_language'),
    }
    if error:
        ctx['error'] = error
    return ctx


def _job_has_unresolved_work(
    *,
    operation_success: bool,
    current_status: str,
    failed_chunks: int,
) -> bool:
    """Keep partial native results resumable through final publication.

    Some format adapters can write a useful partial artifact before returning
    ``False``. A later checkpoint snapshot may legitimately report zero failed
    chunks, but that must never erase the adapter's failure or an already-set
    partial status.
    """
    return (
        not operation_success
        or str(current_status or "").strip().lower() == "partial"
        or int(failed_chunks or 0) > 0
    )


def _job_is_ready_for_final_audits(
    *,
    operation_success: bool,
    current_status: str,
    failed_chunks: int,
) -> bool:
    """Return whether a complete artifact is ready for whole-book checks.

    Native adapters can assemble a useful partial artifact after an early
    failure. Auditing that copy as a completed book turns every untouched
    source unit into a false failure and hides the resumable chunk that
    actually needs attention.
    """
    status = str(current_status or "").strip().lower()
    return (
        bool(operation_success)
        and status not in {
            "error",
            "interrupted",
            "interrupted_before_save",
            "partial",
            "rate_limited",
        }
        and int(failed_chunks or 0) == 0
    )


def _final_source_sample_diagnostics(
    stats: Dict[str, Any] | None,
    report: Any,
) -> Dict[str, Any]:
    """Record bounded sample findings without deciding publication status.

    The source-sample pass is deliberately diagnostic. Complete publication
    gates and format QA remain authoritative because they inspect the whole
    artifact and understand explicit exclusions and reconstructed structure.
    """
    current = dict(stats or {})
    issues = list(getattr(report, "issues", None) or [])
    current.update({
        "final_source_sample_clean": bool(getattr(report, "clean", False)),
        "final_source_sample_warning_count": int(
            getattr(report, "warning_count", 0) or 0
        ),
        "final_source_sample_error_count": int(
            getattr(report, "error_count", 0) or 0
        ),
        "final_source_sample_issue_codes": [
            str(getattr(issue, "code", "") or "")
            for issue in issues
            if str(getattr(issue, "code", "") or "")
        ],
    })
    return current


def _resolve_job_output_path(
    config: Dict[str, Any],
    output_dir: str,
    *,
    is_resume: bool,
) -> str:
    """Choose one stable output path for every lifecycle of a job.

    New jobs still avoid overwriting unrelated files. Resumed jobs reuse the
    path selected on their first run so repeated pauses do not grow filename
    suffixes or leave several indistinguishable partial artifacts.
    """
    stable_name = str(config.get('_job_output_filename') or '').strip()
    requested_name = stable_name or str(config.get('output_filename') or '').strip()
    valid_name, error = PathValidator.validate_filename(requested_name)
    if not valid_name:
        raise ValueError(f"Unsafe output filename: {error}")
    if Path(requested_name).suffix.casefold() not in {
        '.txt',
        '.srt',
        '.epub',
        '.docx',
        '.pdf',
    }:
        raise ValueError("Unsafe output filename: unsupported output extension")
    tentative = os.path.join(output_dir, requested_name)
    resolved = tentative if is_resume and stable_name else get_unique_output_path(tentative)
    actual_name = os.path.basename(resolved)
    config['_job_output_filename'] = actual_name
    config['output_filename'] = actual_name
    return resolved


def _canonicalize_published_epub_stats(
    stats: Dict[str, Any],
    publication_report: Any,
) -> Dict[str, Any]:
    """Replace stale phase counters after whole-EPUB coverage is proven.

    Native EPUB checkpoints persist one row per XHTML file while the live UI
    tracks inner text chunks. A failed publication attempt used to overwrite
    those real chunk counters with DOM-unit failures. Once the whole-book gate
    passes, its proof plus the accumulated native metrics is authoritative.
    """
    result = dict(stats or {})
    if not publication_report or not bool(getattr(publication_report, 'publishable', False)):
        return result
    accumulated = dict(result.get('epub_accumulated_stats') or {})
    processed = max(
        int(accumulated.get('processed_chunks') or 0),
        int(result.get('processed_chunks') or 0),
    )
    if processed <= 0:
        return result
    result.update({
        'total_chunks': processed,
        'completed_chunks': processed,
        'failed_chunks': 0,
        'checkpoint_failed_chunks': 0,
        'processed_chunks': processed,
        'units_total': processed,
        'units_done': processed,
        'units_succeeded': processed,
        'units_failed': 0,
        'percent': 100.0,
        'quality_degraded': False,
    })
    return result


def _create_audiobook_companion_outputs(
    *,
    output_path: str,
    output_format: str,
    target_language: str,
    output_dir: str,
) -> dict[str, Any]:
    """Create audiobook-specific companion files for the final artifact."""
    final_path = Path(output_path)
    text = extract_readable_text(final_path)
    artifact = sanitize_for_audiobook(
        text,
        target_language=target_language,
        title=final_path.stem,
    )
    if not artifact.text.strip():
        return {"files": [], "report": artifact.report.to_dict()}

    base_stem = final_path.stem
    output_root = Path(output_dir)
    files: list[str] = []

    txt_path = Path(get_unique_output_path(str(output_root / f"{base_stem} (Audiobook).txt")))
    write_text_as_output(artifact.text, txt_path, "txt")
    files.append(txt_path.name)

    if normalize_output_format(output_format) == "epub":
        epub_path = Path(get_unique_output_path(str(output_root / f"{base_stem} (Audiobook).epub")))
        if final_path.suffix.casefold() == ".epub":
            structured_report = create_structured_audiobook_epub(
                final_path,
                epub_path,
                title_suffix="Audiobook",
            )
            artifact.report.structured_epub_preserved = True
            artifact.report.images_preserved = structured_report.image_count
            artifact.report.image_placements_preserved = structured_report.image_placements
            artifact.report.visual_captions_preserved = structured_report.captions_preserved
            artifact.report.cover_preserved = structured_report.cover_preserved
            artifact.report.spine_preserved = structured_report.spine_preserved
            artifact.report.xhtml_preserved = structured_report.xhtml_preserved
            artifact.report.epub_mimetype_valid = structured_report.mimetype_valid
        else:
            write_text_as_output(artifact.text, epub_path, "epub")
        files.append(epub_path.name)

    report_path = Path(get_unique_output_path(str(output_root / f"{base_stem} (Audiobook report).json")))
    report_path.write_text(
        json.dumps(artifact.report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    files.append(report_path.name)
    return {"files": files, "report": artifact.report.to_dict(), "summary": artifact.report.summary()}


def _persist_runtime_checkpoint_config(
    checkpoint_manager: Any,
    translation_id: str,
    config: Dict[str, Any],
    log_callback=None,
) -> bool:
    """Persist resolved runtime config so resume uses the same guards/profile."""
    try:
        saved = checkpoint_manager.update_job_config(translation_id, config)
    except Exception as exc:
        if log_callback:
            log_callback(
                "checkpoint_config_persist_error",
                f"⚠️ Could not persist runtime checkpoint config: {exc}"
            )
        return False
    if not saved and log_callback:
        log_callback(
            "checkpoint_config_persist_missing",
            "⚠️ Could not persist runtime checkpoint config; checkpoint was not found."
        )
    return saved


def _extract_json_object(text: str) -> dict | None:
    return loads_first_json_object(text)


def _final_source_sample_audit_enabled(prompt_options: Dict[str, Any]) -> bool:
    raw = str((prompt_options or {}).get('final_source_sample_audit', 'on')).strip().lower()
    return raw not in {'', '0', 'false', 'off', 'disabled', 'none'}


def _glossary_purpose_from_config(config: Dict[str, Any]) -> str:
    prompt_options = config.get('prompt_options') or {}
    if prompt_options.get('text_transform_mode'):
        return 'transformation'
    if config.get('refine_only'):
        return 'refinement'
    return 'translation'


def _glossary_filter_config(prompt_options: Dict[str, Any], purpose: str):
    from src.core.glossary import GlossaryConfig

    explicit_config = (prompt_options or {}).get('glossary_config')
    if explicit_config is not None:
        if isinstance(explicit_config, dict):
            allowed = {
                'max_entries',
                'case_sensitive',
                'accent_insensitive',
                'warn_on_cap',
            }
            return GlossaryConfig(
                **{
                    key: value
                    for key, value in explicit_config.items()
                    if key in allowed
                }
            )
        return explicit_config
    if purpose in {'refinement', 'transformation'}:
        return GlossaryConfig(case_sensitive=False, accent_insensitive=True)
    return GlossaryConfig()


def _glossary_match_summary(config: Dict[str, Any], input_path: str) -> Dict[str, Any] | None:
    prompt_options = config.get('prompt_options') or {}
    terms = prompt_options.get('glossary_terms') or {}
    if not terms or not input_path:
        return None

    from src.core.glossary import filter_glossary_for_purpose

    purpose = _glossary_purpose_from_config(config)
    scan_config = _glossary_filter_config(prompt_options, purpose)
    text = extract_readable_text(input_path)
    filtered, capped = filter_glossary_for_purpose(
        text,
        terms,
        scan_config,
        purpose,
    )
    return {
        'purpose': purpose,
        'total_terms': len(terms),
        'matched_terms': len(filtered),
        'capped': capped,
    }


def _profile_glossary_match_summary(config: Dict[str, Any], input_path: str) -> Dict[str, Any] | None:
    prompt_options = config.get('prompt_options') or {}
    if not prompt_options or prompt_options.get('use_profile_glossary') is False:
        return None
    if not prompt_options.get('profile_id'):
        return None

    from src.core.book_profiles import profile_glossary_match_summary

    purpose = _glossary_purpose_from_config(config)
    text = extract_readable_text(input_path)
    return profile_glossary_match_summary(
        text,
        prompt_options,
        purpose=purpose,
    )


def _known_input_readable_characters(config: Dict[str, Any]) -> int:
    prompt_options = config.get('prompt_options') or {}
    candidates = (
        prompt_options.get('_input_readable_characters'),
        config.get('input_readable_characters'),
        config.get('readable_characters'),
    )
    for value in candidates:
        try:
            count = int(value)
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return 0


def _ensure_engine_readable_input(
    config: Dict[str, Any],
    input_path: str,
    log_callback=None,
) -> int:
    """Verify the worker is not about to run a zero-text job.

    Upload and /api/translate normally compute ``_input_readable_characters``.
    This worker-level check is the final backstop for older clients, resumes,
    and direct callers that bypassed that route guard.
    """
    known_count = _known_input_readable_characters(config)
    if known_count > 0:
        if log_callback:
            log_callback(
                "input_readability",
                f"📄 Input readable text: {known_count:,} characters."
            )
        return known_count

    label = str(config.get('file_type') or "file").upper()
    try:
        readable_text = extract_readable_text(input_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not extract readable text from this {label} file before starting: {exc}"
        ) from exc

    readable_count = len((readable_text or "").strip())
    if readable_count <= 0:
        raise RuntimeError(
            f"This {label} file has no readable text. If it is a scanned PDF or "
            "image-only document, run OCR first and upload the OCR text/PDF."
        )

    config.setdefault('prompt_options', {})['_input_readable_characters'] = readable_count
    if log_callback:
        log_callback(
            "input_readability",
            f"📄 Input readable text: {readable_count:,} characters."
        )
    return readable_count


async def _epub_structure_hints_for_conversion(
    config: Dict[str, Any],
    source_output_path: str,
    final_output_path: str,
    input_reference_path: str | None,
    log_callback,
) -> dict | None:
    """Ask the configured LLM to vet EPUB heading candidates, cheaply."""
    prompt_options = config.get('prompt_options') or {}
    mode = str(prompt_options.get('epub_structure_llm_mode', 'auto')).lower()
    if mode in {'off', 'false', '0', 'disabled'}:
        return None

    provider_name = (config.get('llm_provider') or 'ollama').lower()
    if mode == 'auto' and provider_name == 'ollama':
        # Local models vary widely in JSON discipline. Keep auto deterministic
        # unless the user explicitly enables this for Ollama.
        return None

    try:
        text = extract_readable_text(source_output_path)
        candidates = build_epub_structure_candidates(
            text,
            Path(final_output_path).stem,
            max_preview_chars=120,
        )
    except Exception as exc:
        if log_callback:
            log_callback("epub_structure_candidate_error", f"⚠️ Could not build EPUB heading candidates: {exc}")
        return None

    if len(candidates) < 3:
        return None

    max_candidates = int(prompt_options.get('epub_structure_llm_max_candidates') or 700)
    if len(candidates) > max_candidates:
        if log_callback:
            log_callback(
                "epub_structure_llm_skipped",
                f"ℹ️ EPUB structure LLM skipped: {len(candidates)} candidates exceeds the {max_candidates} cap."
            )
        return None

    source_toc_titles = []
    if input_reference_path:
        try:
            source_toc_titles = extract_epub_toc_titles(input_reference_path, limit=120)
        except Exception:
            source_toc_titles = []

    model_name = (
        prompt_options.get('epub_structure_model')
        or config.get('model')
        or ('deepseek-v4-pro' if provider_name == 'deepseek' else None)
    )

    candidate_lines = []
    for item in candidates:
        title = str(item.get("title") or "").replace("\n", " ")
        preview = str(item.get("preview") or "").replace("\n", " ")
        candidate_lines.append(
            f"{item['index']}. title={title!r}; paragraphs={item['paragraph_count']}; starts={preview!r}"
        )

    toc_block = "\n".join(f"- {title}" for title in source_toc_titles[:120]) or "(none)"
    prompt = (
        "You are vetting EPUB navigation headings for a Spanish book export. "
        "Do not rewrite the book. Return JSON only.\n\n"
        "Goal: choose which candidate headings should become EPUB table-of-contents entries.\n"
        "Keep: real chapter/story/section/article titles, front matter headings, major parts.\n"
        "Reject: author names alone, letter closings, greetings, body sentences, OCR noise, "
        "formula fragments, and lines that only look like headings because a date appears inside prose.\n"
        "Always include candidate 1 unless it is pure OCR junk.\n\n"
        "Return exactly this shape:\n"
        '{"keep_indices":[1,2,3],"warnings":["short note if useful"]}\n\n'
        "SOURCE EPUB TOC, weak OCR reference when available:\n"
        f"{toc_block}\n\n"
        "CANDIDATES:\n"
        + "\n".join(candidate_lines)
    )
    system_prompt = (
        "You are a strict editorial production assistant. "
        "You classify headings only and output valid JSON only."
    )

    llm = None
    try:
        llm = create_llm_provider(
            provider_name,
            model=model_name,
            api_endpoint=config.get('llm_api_endpoint'),
            openai_api_key=config.get('openai_api_key', ''),
            openrouter_api_key=config.get('openrouter_api_key', ''),
            gemini_api_key=config.get('gemini_api_key', ''),
            mistral_api_key=config.get('mistral_api_key', ''),
            deepseek_api_key=config.get('deepseek_api_key', ''),
            poe_api_key=config.get('poe_api_key', ''),
            nim_api_key=config.get('nim_api_key', ''),
            context_window=config.get('context_window', 2048),
            log_callback=log_callback,
        )
        if log_callback:
            log_callback(
                "epub_structure_llm_start",
                f"🧭 Vetting {len(candidates)} EPUB heading candidates with {provider_name} / {model_name}."
            )
        response = await await_llm_call(
            llm.generate,
            prompt,
            provider=llm,
            request_timeout=90,
            deadline=90,
            system_prompt=system_prompt,
        )
        data = _extract_json_object(response.content if response else "")
        if not data:
            if log_callback:
                log_callback("epub_structure_llm_invalid", "⚠️ EPUB structure LLM returned invalid JSON; using rules only.")
            return None

        keep_indices = []
        for value in data.get("keep_indices") or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= len(candidates):
                keep_indices.append(index)

        if len(keep_indices) < 2 and len(candidates) > 8:
            if log_callback:
                log_callback("epub_structure_llm_rejected", "⚠️ EPUB structure LLM kept too few headings; using rules only.")
            return None

        prompt_tokens = getattr(response, "prompt_tokens", 0) if response else 0
        completion_tokens = getattr(response, "completion_tokens", 0) if response else 0
        if log_callback:
            log_callback(
                "epub_structure_llm_done",
                f"🧭 EPUB structure guard kept {len(set(keep_indices))}/{len(candidates)} headings "
                f"({prompt_tokens + completion_tokens} tokens)."
            )
        return {
            "keep_indices": sorted(set(keep_indices)),
            "source": "llm",
            "warnings": data.get("warnings") if isinstance(data.get("warnings"), list) else [],
        }
    except Exception as exc:
        if log_callback:
            log_callback("epub_structure_llm_error", f"⚠️ EPUB structure LLM failed; using rules only: {exc}")
        return None
    finally:
        if llm is not None:
            try:
                await llm.close()
            except Exception:
                pass


def run_translation_async_wrapper(translation_id, config, state_manager, output_dir, socketio):
    """
    Wrapper for running translation in async context
    
    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(perform_actual_translation(translation_id, config, state_manager, output_dir, socketio))
    except Exception as e:
        error_msg = f"Uncaught major error in translation wrapper {translation_id}: {str(e)}"
        # This is the last-resort handler: anything that raises before or
        # outside perform_actual_translation's own protected try-block (or a
        # second exception raised from within one of its except branches)
        # lands here. Mark the DB row first and unconditionally, so the job
        # never gets stuck at status='running' in SQLite while its worker
        # thread is actually dead -- that desync used to hide the job from
        # get_resumable_jobs() until a full server restart.
        try:
            state_manager.get_checkpoint_manager().mark_error(translation_id)
        except Exception:
            pass
        if state_manager.exists(translation_id):
            state_manager.set_translation_field(translation_id, 'status', 'error')
            state_manager.set_translation_field(translation_id, 'error', error_msg)
            state_manager.append_log(
                translation_id,
                f"[{datetime.now().strftime('%H:%M:%S')}] CRITICAL WRAPPER ERROR: {error_msg}",
            )
            emit_update(socketio, translation_id, {'error': error_msg, 'status': 'error', 'log': f"CRITICAL WRAPPER ERROR: {error_msg}"}, state_manager)
    finally:
        loop.close()


async def perform_actual_translation(translation_id, config, state_manager, output_dir, socketio):
    """
    Perform the actual translation job
    
    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    if not state_manager.exists(translation_id):
        return

    config, legacy_recovery_config_removed = (
        _strip_legacy_automatic_recovery_config(config)
    )
    state_manager.set_translation_field(translation_id, "config", config)
    active_stats = _begin_active_run_stats(
        state_manager.get_translation_field(translation_id, "stats") or {}
    )
    state_manager.set_translation_field(translation_id, "stats", active_stats)

    set_usage_context(
        translation_id=translation_id,
        process_id=translation_id,
        process_type=_usage_process_type(config),
        book_name=config.get('original_filename') or config.get('input_filename') or config.get('output_filename'),
        input_filename=config.get('input_filename') or config.get('original_filename') or '',
        output_filename=config.get('output_filename') or '',
        provider=config.get('llm_provider'),
        model=config.get('model'),
    )

    prompt_options_for_operation = config.get('prompt_options') or {}
    transform_mode_for_operation = str(
        config.get('text_transform_mode')
        or prompt_options_for_operation.get('text_transform_mode')
        or ''
    ).strip()
    transform_label_for_operation = str(
        config.get('text_transform_label')
        or prompt_options_for_operation.get('text_transform_label')
        or ''
    ).strip()
    operation_for_logging = str(
        config.get('operation')
        or ('transform' if transform_mode_for_operation else 'translate')
    ).strip().lower()
    is_transform_job = operation_for_logging == 'transform' or bool(transform_mode_for_operation)
    job_start_label = (
        f"Transformation task started by worker"
        f"{f': {transform_label_for_operation}' if transform_label_for_operation else ''}."
        if is_transform_job
        else "Translation task started by worker."
    )

    state_manager.set_translation_field(translation_id, 'status', 'running')
    state_manager.set_translation_field(translation_id, 'recovery_scheduled', False)
    emit_update(socketio, translation_id, {'status': 'running', 'log': job_start_label}, state_manager)

    def should_interrupt_current_task():
        if state_manager.exists(translation_id) and state_manager.get_translation_field(translation_id, 'interrupted'):
            _log_message_callback("interruption_check", f"Interruption signal detected for job {translation_id}. Halting processing.")
            return True
        return False

    # Setup unified logger for web interface
    def web_callback(log_entry):
        """Callback for WebSocket emission"""
        emit_update(socketio, translation_id, {'log': log_entry['message'], 'log_entry': log_entry}, state_manager)
    
    def storage_callback(log_entry):
        """Callback for storing logs (capped, see TranslationStateManager.append_log)"""
        state_manager.append_log(translation_id, client_safe_log_entry(log_entry))
    
    logger = setup_web_logger(web_callback, storage_callback)
    
    def _log_message_callback(message_key_from_translate_module, message_content="", data=None):
        """Legacy callback wrapper for backward compatibility"""
        # Skip debug messages for web interface
        if message_key_from_translate_module in ["llm_prompt_debug", "llm_raw_response_preview"]:
            return
        
        # Handle structured data from new logging system
        if data and isinstance(data, dict):
            log_type = data.get('type')
            if log_type == 'llm_request':
                logger.debug("LLM Request", LogType.LLM_REQUEST, data)
            elif log_type == 'llm_response':
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("LLM Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'refinement_request':
                # Refinement uses same log type as LLM request for UI display
                logger.debug("Refinement Request", LogType.LLM_REQUEST, data)
            elif log_type == 'refinement_response':
                # Refinement uses same log type as LLM response for UI display
                # Use INFO level to ensure translation preview works even when DEBUG_MODE=false
                logger.info("Refinement Response", LogType.LLM_RESPONSE, data)
            elif log_type == 'prompt_context':
                def _compact_glossary_context(value):
                    if not isinstance(value, dict):
                        return {}
                    return {
                        key: value.get(key)
                        for key in ('matched_terms', 'total_terms', 'rendered_terms', 'capped', 'purpose')
                        if value.get(key) is not None
                    }

                _update_prompt_context_stats('last_chunk', {
                    'chunk_sequence': int(data.get('chunk_sequence') or 0),
                    'profile_glossary': _compact_glossary_context(data.get('profile_glossary')),
                    'manual_glossary': _compact_glossary_context(data.get('manual_glossary')),
                })
                logger.info(message_content, data=data)
            elif log_type == 'progress':
                logger.info("Progress Update", LogType.PROGRESS, data)
            else:
                logger.info(message_content, data=data)
        else:
            # Map specific message patterns to appropriate log types
            if "error" in message_key_from_translate_module.lower():
                logger.error(message_content)
            elif "warning" in message_key_from_translate_module.lower():
                logger.warning(message_content)
            else:
                logger.info(message_content)

        _publish_live_activity(
            message_key_from_translate_module,
            message_content,
            data,
        )

    def _record_job_phase(event):
        """Store framework-neutral engine phase metadata in the job state."""
        if not state_manager.exists(translation_id):
            return
        state_manager.set_translation_field(translation_id, 'job_phase', event.phase.value)
        state_manager.set_translation_field(translation_id, 'job_phase_status', event.status.value)
        state_manager.set_translation_field(translation_id, 'job_phase_event', event.to_dict())

    engine = JobEngine(job_id=translation_id, on_phase_event=_record_job_phase)
    engine.mark_phase(JobPhase.PREPARE, "Initializing job runtime.")

    # Single progress-emit seam. The translate_file → refine_file orchestration
    # still runs two independent engine-side trackers, but the *workflow phase*
    # is now owned here and passed explicitly per phase (TRANSLATING vs
    # REFINING) by whichever call is driving the emit — replacing the old
    # mutable `_workflow_meta` side-channel. A monotonic floor on the canonical
    # `percent` guarantees the bar never regresses across the phase boundary,
    # which is what previously required a manual counter reset at the
    # transition.
    existing_progress_stats = dict(
        state_manager.get_translation_field(translation_id, 'stats') or {}
    )
    try:
        existing_percent = float(
            existing_progress_stats.get('percent')
            or existing_progress_stats.get('progress_percent')
            or state_manager.get_translation_field(translation_id, 'progress')
            or 0.0
        )
    except (TypeError, ValueError):
        existing_percent = 0.0
    _progress_floor = {'value': max(0.0, min(existing_percent, 100.0))}

    def _store_and_emit_stats(stats_update: Dict[str, Any]) -> Dict[str, Any]:
        """Persist one fresh canonical snapshot and emit that exact snapshot."""
        if not state_manager.exists(translation_id):
            return {}
        prior_stats = dict(
            state_manager.get_translation_field(translation_id, 'stats') or {}
        )
        safe_update = dict(stats_update or {})
        try:
            prior_phase = int(prior_stats.get('current_phase') or 1)
        except (TypeError, ValueError):
            prior_phase = 1
        try:
            incoming_phase = int(safe_update.get('current_phase') or prior_phase)
        except (TypeError, ValueError):
            incoming_phase = prior_phase
        if incoming_phase == prior_phase and 'completed_chunks' in safe_update:
            try:
                safe_update['completed_chunks'] = max(
                    int(prior_stats.get('completed_chunks') or 0),
                    int(safe_update.get('completed_chunks') or 0),
                )
            except (TypeError, ValueError):
                pass
        state_manager.update_stats(translation_id, safe_update)
        current_stats = dict(
            state_manager.get_translation_field(translation_id, 'stats') or {}
        )
        now = time.time()
        current_stats = _apply_active_timing(current_stats, now=now)
        snapshot = snapshot_from_legacy_stats(current_stats).to_dict()
        if snapshot['percent'] < _progress_floor['value']:
            snapshot['percent'] = _progress_floor['value']
        else:
            _progress_floor['value'] = snapshot['percent']
        current_stats.update(snapshot)
        state_manager.set_translation_field(translation_id, 'stats', current_stats)
        # Keep the detailed endpoint and the summary endpoint on the same
        # canonical percentage. Restored jobs historically left this field at 0.
        state_manager.set_translation_field(
            translation_id,
            'progress',
            snapshot['percent'],
        )
        emit_update(socketio, translation_id, {'stats': current_stats}, state_manager)
        return current_stats

    def _publish_live_activity(event: str, message: str = "", data: Any = None) -> None:
        label = _live_activity_label(event, message, data)
        if not label or not state_manager.exists(translation_id):
            return
        status = str(
            state_manager.get_translation_field(translation_id, 'status') or ''
        ).strip().lower()
        if status not in {'running', 'queued', 'rate_limited'}:
            return
        _store_and_emit_stats({
            'live_status': label,
            'live_status_kind': 'active',
            'live_activity_event': str(event or ''),
            'last_activity_at': time.time(),
        })

    if legacy_recovery_config_removed:
        _log_message_callback(
            "legacy_failed_chunk_recovery_migrated",
            "Se retiró metadata de una recuperación antigua que invalidaba "
            "checkpoints; el avance traducido se conservará.",
        )

    def _emit_progress(new_stats_dict, phase_meta):
        if not state_manager.exists(translation_id):
            return
        prompt_options_for_progress = config.get('prompt_options') or {}
        transform_mode = (
            config.get('text_transform_mode')
            or prompt_options_for_progress.get('text_transform_mode')
            or ''
        )
        transform_label = (
            config.get('text_transform_label')
            or prompt_options_for_progress.get('text_transform_label')
            or ''
        )
        operation = config.get('operation') or ('transform' if transform_mode else '')
        operation_meta = {
            'operation': operation,
            'text_transform_mode': transform_mode,
            'text_transform_label': transform_label,
            'output_filename': config.get('output_filename'),
        }
        completed = int(new_stats_dict.get('completed_chunks') or 0)
        total = int(new_stats_dict.get('total_chunks') or 0)
        live_status = (
            f"Avance guardado: {completed}/{total} fragmentos"
            if total > 0 else "Avance guardado; continuando"
        )
        current_stats = _store_and_emit_stats({
            **new_stats_dict,
            **phase_meta,
            **operation_meta,
            'live_status': live_status,
            'live_status_kind': 'active',
            'last_activity_at': time.time(),
        })

        # Update logger progress for CLI display
        completed = current_stats.get('completed_chunks', 0)
        total = current_stats.get('total_chunks', 0)
        if total > 0:
            logger.update_progress(completed, total)

    def _translate_stats_callback(new_stats_dict):
        # Phase 1. enable_refinement advertises the two-phase bar up-front when
        # a refine-after pass will follow, so phase 1 maps to the [0, 50] band.
        inline_refinement = (config.get('prompt_options') or {}).get('inline_refinement')
        _emit_progress(new_stats_dict, {
            'enable_refinement': bool(config.get('refine_after')) and not inline_refinement,
            'current_phase': 1,
            'inline_refinement': bool(inline_refinement),
        })

    def _refine_after_stats_callback(new_stats_dict):
        # Phase 2 of a refine-after workflow: maps to the [50, 100] band.
        _emit_progress(new_stats_dict, {'enable_refinement': True, 'current_phase': 2})

    def _refine_only_stats_callback(new_stats_dict):
        # Single-phase refine-only: the whole bar is the refinement pass.
        _emit_progress(new_stats_dict, {
            'enable_refinement': False, 'refine_only': True, 'current_phase': 1,
        })

    def _finalize_stats_callback(new_stats_dict):
        # Finalization pushes (e.g. final elapsed_time) must not re-assert a
        # phase; the stored stats already carry the terminal phase/percent.
        _emit_progress(new_stats_dict, {})

    def _openrouter_cost_callback(cost_data):
        """Update OpenRouter cost in state. No emit: this callback runs on the
        provider's HTTP response thread, and a cross-thread emit can overtake
        the main loop's stats emit on the wire (showing a stale snapshot and
        rolling the progress bar backward). The cost is picked up by the next
        chunk's stats_callback, which is the same thread that owns progress."""
        if state_manager.exists(translation_id):
            state_manager.update_stats(translation_id, {
                'openrouter_cost': cost_data['session_cost'],
                'openrouter_prompt_tokens': cost_data['total_prompt_tokens'],
                'openrouter_completion_tokens': cost_data['total_completion_tokens']
            })

    def _update_prompt_context_stats(key: str, payload: Dict[str, Any]) -> None:
        if not state_manager.exists(translation_id):
            return
        stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        prompt_context = dict(stats.get('prompt_context') or {})
        prompt_context[key] = payload
        _store_and_emit_stats({'prompt_context': prompt_context})

    # Setup OpenRouter cost callback if using OpenRouter provider
    if config.get('llm_provider') == 'openrouter':
        OpenRouterProvider.reset_session_cost()
        OpenRouterProvider.set_cost_callback(_openrouter_cost_callback)

    # Get checkpoint manager and handle resume
    engine.mark_phase(JobPhase.PREPARE, "Resolving checkpoint, sanitizer, glossary, and output defaults.")
    checkpoint_manager = state_manager.get_checkpoint_manager()
    resume_from_index = config.get('resume_from_index', 0)
    is_resume = config.get('is_resume', False)

    if is_resume and _resume_requires_layout_sanitizer_restart(config):
        try:
            checkpoint_manager.delete_checkpoint(translation_id)
        except Exception:
            pass
        resume_from_index = 0
        config['resume_from_index'] = 0
        config['is_resume'] = False
        is_resume = False
        _log_message_callback(
            "layout_sanitizer_checkpoint_restart",
            "📄 Previous PDF/DOCX checkpoint used an older extraction pipeline; "
            "restarting from the beginning with layout sanitization."
        )

    _ensure_layout_sanitizer_options(config)
    prompt_options = config.setdefault('prompt_options', {})
    prompt_options.setdefault('strict_quality_assurance', True)
    config.setdefault('quality_assurance', {})
    config['quality_assurance'].setdefault('strict', True)

    def _discard_unstarted_checkpoint(reason: str) -> None:
        try:
            checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
            chunks = checkpoint_data.get('chunks', []) if checkpoint_data else []
            total = 0
            if checkpoint_data:
                total = int(
                    checkpoint_data.get('job', {})
                    .get('progress', {})
                    .get('total_chunks') or 0
                )
            if total <= 0 and not chunks:
                checkpoint_manager.delete_checkpoint(translation_id)
                _log_message_callback(
                    "checkpoint_unstarted_removed",
                    f"Removed unstarted checkpoint: {reason}"
                )
        except Exception as cleanup_error:
            _log_message_callback(
                "checkpoint_unstarted_cleanup_error",
                f"Could not remove unstarted checkpoint: {cleanup_error}"
            )

    # Snapshot the active glossary into prompt_options BEFORE persisting the
    # job, so the snapshot survives resume even if the source glossary is
    # later edited or deleted. On resume, the snapshot is already in the
    # restored config and we skip the reload.
    if not is_resume:
        glossary_id = config.get('prompt_options', {}).get('glossary_id')
        if glossary_id and not config.get('prompt_options', {}).get('glossary_terms'):
            try:
                from src.api.translation_state import get_state_manager
                store = get_state_manager().get_glossary_store()
                glossary = store.get_glossary(int(glossary_id))
                if glossary:
                    complete_terms = glossary.terms_dict
                    if not complete_terms:
                        config.setdefault('prompt_options', {})['glossary_empty_name'] = glossary.name
                        config.setdefault('prompt_options', {})['glossary_empty_total_terms'] = len(glossary.terms)
                if glossary and glossary.terms_dict:
                    if 'prompt_options' not in config:
                        config['prompt_options'] = {}
                    config['prompt_options']['glossary_terms'] = glossary.terms_dict
                    config['prompt_options']['glossary_name'] = glossary.name
                    metadata = {}
                    for term in glossary.terms:
                        if term.category:
                            metadata[term.source_term] = {'category': term.category}
                    if metadata:
                        config['prompt_options']['glossary_term_metadata'] = metadata
            except Exception as e:
                # Non-fatal: log later once the logger is wired in.
                config.setdefault('prompt_options', {})['glossary_load_error'] = str(e)

    automatic_recovery_plan = None
    try:
        requested_output_format = normalize_output_format(config.get('output_format', 'auto'))
        config['output_format'] = requested_output_format
        final_output_format = requested_format_for_job(
            config['file_type'],
            requested_output_format,
        )
        # ``auto`` still resolves to a concrete native format. Always force
        # that safe extension before any adapter writes the file; otherwise a
        # caller could name a text result ``payload.cmd`` and later invoke the
        # local open-file route on Windows.
        config['output_filename'] = ensure_output_extension(
            config['output_filename'],
            final_output_format,
        )
        if config.get('_job_output_filename'):
            config['_job_output_filename'] = ensure_output_extension(
                str(config['_job_output_filename']),
                final_output_format,
            )

        # Create checkpoint for new jobs (not for resumed jobs)
        if not is_resume:
            file_type = config['file_type']
            input_file_path = config.get('file_path')
            checkpoint_manager.start_job(
                translation_id,
                file_type,
                config,
                input_file_path
            )

        # PHASE 2: Configuration validation is now handled by AdaptiveContextManager during translation

        # New jobs avoid unrelated files; resumes reuse this job's stable path.
        requested_output_filename = config['output_filename']
        final_output_filepath_on_server = _resolve_job_output_path(
            config,
            output_dir,
            is_resume=bool(is_resume),
        )

        # Update config with the actual filename (may have been modified)
        actual_output_filename = os.path.basename(final_output_filepath_on_server)
        if actual_output_filename != requested_output_filename:
            _log_message_callback("output_filename_modified",
                f"ℹ️ Output filename modified to avoid overwriting: "
                f"{requested_output_filename} → {actual_output_filename}")

        text_first_pipeline = _uses_text_first_pipeline(config)
        if text_first_pipeline:
            config.setdefault('prompt_options', {})['text_first_pipeline'] = True
        native_format = 'txt' if text_first_pipeline else native_output_format(config['file_type'])
        temp_engine_output_path = None
        output_filepath_on_server = final_output_filepath_on_server
        audiobook_companion_files: list[str] = []
        if final_output_format != native_format:
            native_suffix = format_extension(native_format, f".{native_format}")
            with tempfile.NamedTemporaryFile(
                prefix=f"tmp_{translation_id}_",
                suffix=native_suffix,
                delete=False,
                dir=output_dir,
            ) as tmp_output:
                temp_engine_output_path = tmp_output.name
            output_filepath_on_server = temp_engine_output_path

        # Log start with unified logger. Transform jobs share the same worker
        # as translation jobs, but their operation metadata must remain visible
        # in the logs and UI so they are not mistaken for normal translation.
        logger.info(
            "Transformation Started" if is_transform_job else "Translation Started",
            LogType.TRANSLATION_START,
            {
            'source_lang': config['source_language'],
            'target_lang': config['target_language'],
            'file_type': config['file_type'].upper(),
            'model': config['model'],
            'translation_id': translation_id,
            'output_file': config['output_filename'],
            'api_endpoint': config['llm_api_endpoint'],
            'chunk_size': config.get('chunk_size', 'default'),
            'operation': operation_for_logging,
            'text_transform_mode': transform_mode_for_operation,
            'text_transform_label': transform_label_for_operation,
        })

        # Make the resumed portion's model/provider explicit in the job log so a
        # resume that switched model/provider (issue #183) is auditable.
        if is_resume:
            _log_message_callback(
                "resume_model_info",
                f"↻ Resuming from chunk {resume_from_index} using "
                f"{config.get('llm_provider', 'ollama')} / {config['model']}."
            )

        engine.mark_phase(JobPhase.INGEST, "Resolving input source for the engine.")
        input_path_for_translate_module = config.get('file_path')

        # Handle special case for TXT with inline text content (no file upload)
        temp_txt_file_path = None
        if config['file_type'] == 'txt' and 'text' in config and input_path_for_translate_module is None:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt", dir=output_dir) as tmp_f:
                tmp_f.write(config['text'])
                temp_txt_file_path = tmp_f.name
            input_path_for_translate_module = temp_txt_file_path

        # Validate input file path
        if not input_path_for_translate_module:
            _log_message_callback("error_no_path", f"❌ {config['file_type'].upper()} translation requires a file path from upload.")
            raise Exception(f"{config['file_type'].upper()} translation requires a file_path.")

        try:
            _ensure_engine_readable_input(
                config,
                input_path_for_translate_module,
                _log_message_callback,
            )
        except RuntimeError:
            _discard_unstarted_checkpoint("input has no readable text")
            raise

        # Read custom instruction file if specified
        custom_instruction_file = config.get('prompt_options', {}).get('custom_instruction_file', '')

        translation_instructions = None
        refinement_instructions = None

        if custom_instruction_file:
            project_root = Path(os.getcwd())
            custom_instructions_dir = project_root / 'Custom_Instructions'

            if not is_safe_filename(custom_instruction_file):
                _log_message_callback(
                    "custom_instructions_invalid",
                    f"⚠️ Custom instructions file name '{custom_instruction_file}' is invalid "
                    f"(allowed: alphanumeric, underscore, hyphen, dot; must end in .txt, .yaml, or .yml). "
                    f"Translation will proceed without it."
                )
            else:
                try:
                    loaded = load_custom_instructions(
                        custom_instruction_file, custom_instructions_dir
                    )
                    translation_instructions = loaded.get('translation')
                    refinement_instructions = loaded.get('refinement')

                    if translation_instructions or refinement_instructions:
                        phases = []
                        if translation_instructions:
                            phases.append('translation')
                        if refinement_instructions:
                            phases.append('refinement')
                        _log_message_callback(
                            "custom_instructions",
                            f"📝 Loaded custom instructions: {custom_instruction_file} "
                            f"(phases: {', '.join(phases)})"
                        )
                    else:
                        _log_message_callback(
                            "custom_instructions_empty",
                            f"⚠️ Custom instructions file '{custom_instruction_file}' is empty. "
                            f"Translation will proceed without it."
                        )
                except FileNotFoundError:
                    _log_message_callback(
                        "custom_instructions_missing",
                        f"⚠️ Custom instructions file '{custom_instruction_file}' was selected "
                        f"but not found in {custom_instructions_dir}. Translation will proceed "
                        f"without it."
                    )
                except (ValueError, Exception) as e:
                    _log_message_callback(
                        "custom_instructions_error",
                        f"⚠️ Failed to load custom instructions '{custom_instruction_file}': {e}. "
                        f"Translation will proceed without it."
                    )

        # Inject phase-specific custom instructions into prompt_options
        if translation_instructions or refinement_instructions:
            if 'prompt_options' not in config:
                config['prompt_options'] = {}
            if translation_instructions:
                config['prompt_options']['custom_instructions'] = translation_instructions
            if refinement_instructions:
                config['prompt_options']['refinement_instructions'] = refinement_instructions

        _configure_editorial_guard_options(config)
        corrected_profile = str(
            config.get('prompt_options', {}).get('_profile_scope_corrected_from') or ''
        ).strip()
        if corrected_profile:
            replacement_profile = str(
                config.get('prompt_options', {}).get('profile_id') or ''
            ).strip()
            if replacement_profile:
                _log_message_callback(
                    "book_profile_scope_corrected",
                    f"🔒 El perfil '{corrected_profile}' pertenece a otra obra; "
                    f"se sustituyó por '{replacement_profile}'.",
                )
            else:
                _log_message_callback(
                    "book_profile_scope_removed",
                    f"🔒 El perfil '{corrected_profile}' pertenece a otra obra y no se aplicará.",
                )
        _persist_runtime_checkpoint_config(
            checkpoint_manager,
            translation_id,
            config,
            _log_message_callback,
        )
        active_profile_id = str(config.get('prompt_options', {}).get('profile_id') or '').strip()
        if active_profile_id:
            _update_prompt_context_stats('active_profile', {
                'profile_id': active_profile_id,
                'editorial_mode': config.get('prompt_options', {}).get('editorial_mode') or '',
                'use_profile_glossary': bool(config.get('prompt_options', {}).get('use_profile_glossary', True)),
            })
            _log_message_callback(
                "active_book_profile",
                f"📚 Active editorial profile: {active_profile_id}."
            )
            try:
                profile_glossary_summary = _profile_glossary_match_summary(
                    config,
                    input_path_for_translate_module,
                )
            except Exception as exc:
                profile_glossary_summary = None
                _log_message_callback(
                    "profile_glossary_coverage_unavailable",
                    f"⚠️ Could not compute active profile glossary coverage: {exc}"
                )
            if profile_glossary_summary:
                _update_prompt_context_stats('profile_glossary', profile_glossary_summary)
                matched_terms = profile_glossary_summary['matched_terms']
                total_terms = profile_glossary_summary['total_terms']
                purpose = profile_glossary_summary['purpose']
                capped_note = " (capped)" if profile_glossary_summary.get('capped') else ""
                rendered_terms = profile_glossary_summary.get('rendered_terms', matched_terms)
                if matched_terms:
                    render_note = (
                        f"; {rendered_terms} will be rendered per chunk cap"
                        if rendered_terms != matched_terms else ""
                    )
                    _log_message_callback(
                        "profile_glossary_coverage",
                        f"🔎 Active profile glossary coverage for {purpose}: "
                        f"{matched_terms}/{total_terms} approved prompt entries match this input{render_note}{capped_note}."
                    )
                else:
                    _log_message_callback(
                        "profile_glossary_no_matches",
                        f"⚠️ Active profile '{active_profile_id}' has {total_terms} approved prompt entries, "
                        f"but none match this input for {purpose}. It will still apply editorial profile instructions, "
                        "but its glossary will not affect prompts unless matching terms appear."
                    )

        # Surface glossary load result (snapshot was taken earlier, before start_job).
        glossary_terms_snapshot = config.get('prompt_options', {}).get('glossary_terms')
        glossary_load_error = config.get('prompt_options', {}).pop('glossary_load_error', None)
        glossary_empty_name = config.get('prompt_options', {}).pop('glossary_empty_name', None)
        glossary_empty_total_terms = config.get('prompt_options', {}).pop('glossary_empty_total_terms', 0)
        if glossary_terms_snapshot:
            glossary_name = config.get('prompt_options', {}).get('glossary_name', '?')
            _log_message_callback(
                "glossary_loaded",
                f"📖 Loaded glossary '{glossary_name}' ({len(glossary_terms_snapshot)} terms)"
            )
            try:
                glossary_summary = _glossary_match_summary(
                    config,
                    input_path_for_translate_module,
                )
            except Exception as exc:
                glossary_summary = None
                _log_message_callback(
                    "glossary_coverage_unavailable",
                    f"⚠️ Could not compute glossary coverage for this input: {exc}"
                )
            if glossary_summary:
                _update_prompt_context_stats('manual_glossary', glossary_summary)
                matched_terms = glossary_summary['matched_terms']
                total_terms = glossary_summary['total_terms']
                purpose = glossary_summary['purpose']
                capped_note = " (capped)" if glossary_summary.get('capped') else ""
                if matched_terms:
                    _log_message_callback(
                        "glossary_coverage",
                        f"🔎 Glossary coverage for {purpose}: {matched_terms}/{total_terms} terms match this input{capped_note}."
                    )
                else:
                    _log_message_callback(
                        "glossary_no_matches",
                        f"⚠️ Glossary '{glossary_name}' has {total_terms} complete term(s), but none match this input for {purpose}. "
                        "It will not affect prompts unless matching terms appear in the text."
                    )
        elif glossary_load_error:
            _log_message_callback(
                "glossary_error",
                f"⚠️ Could not load glossary: {glossary_load_error}"
            )
        elif glossary_empty_name:
            suffix = (
                "no terms"
                if not glossary_empty_total_terms
                else f"{glossary_empty_total_terms} draft term(s), but no complete source → target pairs"
            )
            _log_message_callback(
                "glossary_empty",
                f"⚠️ Selected glossary '{glossary_empty_name}' has {suffix}. "
                "It will not affect prompts until it has complete source → target terms."
            )

        operation_success = False
        if config.get('refine_only'):
            _log_message_callback(
                "refine_only_mode",
                "✨ Refine-only mode: skipping translation, polishing the input file as-is."
            )
            src_lang = config.get('source_language')
            tgt_lang = config.get('target_language')
            if src_lang and tgt_lang and src_lang != tgt_lang:
                _log_message_callback(
                    "refine_only_lang_mismatch",
                    f"⚠️ source_language ({src_lang}) ≠ target_language ({tgt_lang}). "
                    f"Refinement is monolingual; the file will be polished as {tgt_lang}."
                )
            operation_success = await engine.run_phase(
                JobPhase.REFINE,
                refine_file,
                input_filepath=input_path_for_translate_module,
                output_filepath=output_filepath_on_server,
                target_language=config['target_language'],
                model_name=config['model'],
                llm_provider=config.get('llm_provider', 'ollama'),
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=_log_message_callback,
                stats_callback=_refine_only_stats_callback,
                check_interruption_callback=should_interrupt_current_task,
                resume_from_index=resume_from_index,
                llm_api_endpoint=config['llm_api_endpoint'],
                gemini_api_key=config.get('gemini_api_key', ''),
                openai_api_key=config.get('openai_api_key', ''),
                openrouter_api_key=config.get('openrouter_api_key', ''),
                mistral_api_key=config.get('mistral_api_key', ''),
                deepseek_api_key=config.get('deepseek_api_key', ''),
                poe_api_key=config.get('poe_api_key', ''),
                nim_api_key=config.get('nim_api_key', ''),
                context_window=config.get('context_window', 2048),
                auto_adjust_context=config.get('auto_adjust_context', True),
                max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                prompt_options=config.get('prompt_options', {}),
                message="Running refine-only pass.",
                metadata={'refine_only': True},
            )
        else:
            # The translate-phase callback advertises the two-phase workflow
            # up-front (via enable_refinement) when a refine-after pass will
            # follow, so the UI renders the phase bar from the start of phase 1.
            operation_success = await engine.run_phase(
                JobPhase.TRANSLATE,
                translate_file,
                input_filepath=input_path_for_translate_module,
                output_filepath=output_filepath_on_server,
                source_language=config['source_language'],
                target_language=config['target_language'],
                model_name=config['model'],
                llm_provider=config.get('llm_provider', 'ollama'),
                checkpoint_manager=checkpoint_manager,
                translation_id=translation_id,
                log_callback=_log_message_callback,
                stats_callback=_translate_stats_callback,
                check_interruption_callback=should_interrupt_current_task,
                resume_from_index=resume_from_index,
                llm_api_endpoint=config['llm_api_endpoint'],
                gemini_api_key=config.get('gemini_api_key', ''),
                openai_api_key=config.get('openai_api_key', ''),
                openrouter_api_key=config.get('openrouter_api_key', ''),
                mistral_api_key=config.get('mistral_api_key', ''),
                deepseek_api_key=config.get('deepseek_api_key', ''),
                poe_api_key=config.get('poe_api_key', ''),
                nim_api_key=config.get('nim_api_key', ''),
                context_window=config.get('context_window', 2048),
                auto_adjust_context=config.get('auto_adjust_context', True),
                min_chunk_size=config.get('min_chunk_size', 5),
                max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                max_attempts=config.get('max_attempts', 2),
                prompt_options=config.get('prompt_options', {}),
                bilingual_output=config.get('bilingual_output', False),
                message="Running translation pass.",
                metadata={'refine_after': bool(config.get('refine_after'))},
            )

            current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            if (
                not operation_success
                and not state_manager.get_translation_field(translation_id, 'interrupted')
                and state_manager.get_translation_field(translation_id, 'status')
                    not in ('error', 'partial', 'rate_limited')
                and int(current_stats.get('completed_chunks') or 0) == 0
                and int(current_stats.get('failed_chunks') or 0) == 0
                and not os.path.exists(output_filepath_on_server)
            ):
                _discard_unstarted_checkpoint("translation produced no chunks")
                raise RuntimeError(
                    f"{config['file_type'].upper()} processing did not start. "
                    "No readable text or translation chunks were produced."
                )

            if (
                not operation_success
                and not state_manager.get_translation_field(translation_id, 'interrupted')
                and state_manager.get_translation_field(translation_id, 'status')
                    not in ('error', 'rate_limited')
            ):
                # A native translator may still write a useful partial artifact
                # before returning False.  Never let its mere existence promote
                # the job to completed or trigger a chained refinement pass.
                current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                failed_count = max(1, int(current_stats.get('failed_chunks') or 0))
                state_manager.update_stats(translation_id, {
                    **current_stats,
                    'failed_chunks': failed_count,
                    'checkpoint_failed_chunks': failed_count,
                    'quality_degraded': True,
                })
                state_manager.set_translation_field(translation_id, 'status', 'partial')
                checkpoint_manager.mark_partial(translation_id)
                _log_message_callback(
                    "translation_native_partial",
                    "⚠️ La fase de traducción no completó todos los bloques. "
                    "El archivo queda parcial y reanudable; se omite la revisión encadenada.",
                )

            post_translate_checkpoint = checkpoint_progress_snapshot(
                checkpoint_manager,
                translation_id,
            )
            if (
                post_translate_checkpoint
                and post_translate_checkpoint.get('unresolved')
                and not state_manager.get_translation_field(translation_id, 'interrupted')
                and state_manager.get_translation_field(translation_id, 'status')
                    not in ('error', 'rate_limited')
            ):
                failed_count = int(post_translate_checkpoint.get('failed_chunks') or 0)
                _translate_stats_callback({
                    'total_chunks': post_translate_checkpoint.get('total_chunks', 0),
                    'completed_chunks': post_translate_checkpoint.get('completed_chunks', 0),
                    'failed_chunks': failed_count,
                    'checkpoint_failed_chunks': failed_count,
                })
                state_manager.set_translation_field(translation_id, 'status', 'partial')
                _log_message_callback(
                    "translation_checkpoint_unresolved",
                    "⚠️ Translation phase left "
                    f"{failed_count} unresolved checkpoint chunk(s). "
                    "Skipping chained editorial review until resume repairs them."
                )

            # Optional chained refinement pass on the translated output.
            should_refine_after = (
                config.get('refine_after')
                and not (config.get('prompt_options') or {}).get('inline_refinement')
                and os.path.exists(output_filepath_on_server)
                and not state_manager.get_translation_field(translation_id, 'interrupted')
                and state_manager.get_translation_field(translation_id, 'status')
                    not in ('error', 'partial', 'rate_limited')
            )
            if should_refine_after:
                # Phase 2. No manual counter reset is needed: the refine-phase
                # callback always tags emits as phase 2 with the refine engine's
                # own counters (starting at 0), so a stale phase-1 count can
                # never pair with phase 2, and the monotonic floor in
                # _emit_progress keeps the bar from regressing across the
                # transition.
                refine_after_prompt_options = _with_source_guard_refs(
                    config.get('prompt_options', {}),
                    checkpoint_manager,
                    translation_id,
                    _log_message_callback,
                )
                _log_message_callback(
                    "refine_after_start",
                    "✨ Translation done — running refinement pass on the output."
                )
                await engine.run_phase(
                    JobPhase.REFINE,
                    refine_file,
                    input_filepath=output_filepath_on_server,
                    output_filepath=output_filepath_on_server,
                    target_language=config['target_language'],
                    model_name=config['model'],
                    llm_provider=config.get('llm_provider', 'ollama'),
                    checkpoint_manager=checkpoint_manager,
                    translation_id=translation_id,
                    log_callback=_log_message_callback,
                    stats_callback=_refine_after_stats_callback,
                    check_interruption_callback=should_interrupt_current_task,
                    resume_from_index=0,
                    llm_api_endpoint=config['llm_api_endpoint'],
                    gemini_api_key=config.get('gemini_api_key', ''),
                    openai_api_key=config.get('openai_api_key', ''),
                    openrouter_api_key=config.get('openrouter_api_key', ''),
                    mistral_api_key=config.get('mistral_api_key', ''),
                    deepseek_api_key=config.get('deepseek_api_key', ''),
                    poe_api_key=config.get('poe_api_key', ''),
                    nim_api_key=config.get('nim_api_key', ''),
                    context_window=config.get('context_window', 2048),
                    auto_adjust_context=config.get('auto_adjust_context', True),
                    max_tokens_per_chunk=config.get('max_tokens_per_chunk'),
                    prompt_options=refine_after_prompt_options,
                    message="Running chained refinement pass.",
                    metadata={'refine_after': True},
                )

        current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        if (
            config.get('refine_only')
            and not operation_success
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and state_manager.get_translation_field(translation_id, 'status')
                not in ('error', 'partial', 'rate_limited')
            and int(current_stats.get('completed_chunks') or 0) == 0
            and int(current_stats.get('failed_chunks') or 0) == 0
            and not os.path.exists(output_filepath_on_server)
        ):
            _discard_unstarted_checkpoint("refinement produced no chunks")
            raise RuntimeError(
                f"{config['file_type'].upper()} refinement did not start. "
                "No readable text or refinement chunks were produced."
            )

        # If an EPUB translation was paused, the file was saved with a `[partial NN%]`
        # prefix. Re-point the tracking variables to the actual file on disk so the
        # download endpoint and UI list the right name.
        if (config['file_type'] == 'epub'
                and state_manager.get_translation_field(translation_id, 'interrupted')
                and not os.path.exists(output_filepath_on_server)):
            candidates = find_partial_output_paths(output_filepath_on_server)
            if candidates:
                # Pick the most recently written one if several exist
                actual = max(candidates, key=lambda p: os.path.getmtime(p))
                output_filepath_on_server = actual
                config['output_filename'] = os.path.basename(actual)
                _log_message_callback("output_marked_partial",
                    f"💾 Partial EPUB saved as: {config['output_filename']}")

        if final_output_format != native_format and os.path.exists(output_filepath_on_server):
            report_moves = [
                (editorial_report_path(output_filepath_on_server), editorial_report_path(final_output_filepath_on_server)),
                (fidelity_report_path(output_filepath_on_server), fidelity_report_path(final_output_filepath_on_server)),
                (
                    literary_continuity_report_path(output_filepath_on_server),
                    literary_continuity_report_path(final_output_filepath_on_server),
                ),
            ]
            structure_hints = None
            if final_output_format == 'epub':
                structure_hints = await _epub_structure_hints_for_conversion(
                    config,
                    output_filepath_on_server,
                    final_output_filepath_on_server,
                    input_path_for_translate_module,
                    _log_message_callback,
                )
            await engine.run_phase(
                JobPhase.ASSEMBLE,
                convert_output_file,
                source_path=output_filepath_on_server,
                destination_path=final_output_filepath_on_server,
                output_format=final_output_format,
                structure_hints=structure_hints,
                message=f"Converting final output to {final_output_format}.",
                metadata={'output_format': final_output_format},
            )
            if temp_engine_output_path and os.path.exists(temp_engine_output_path):
                try:
                    native_suffix = format_extension(native_format, f".{native_format}")
                    native_backup_name = (
                        f"{Path(final_output_filepath_on_server).stem} "
                        f"[native {native_format}]{native_suffix}"
                    )
                    native_backup_path = get_unique_output_path(
                        os.path.join(output_dir, native_backup_name)
                    )
                    os.replace(temp_engine_output_path, native_backup_path)
                    _log_message_callback(
                        "native_output_preserved",
                        f"Preserved native {native_format.upper()} output for re-export: "
                        f"{os.path.basename(native_backup_path)}"
                    )
                except OSError:
                    pass
            output_filepath_on_server = final_output_filepath_on_server
            config['output_filename'] = os.path.basename(final_output_filepath_on_server)
            for source_report_path, final_report_path in report_moves:
                if source_report_path.exists():
                    final_report_path = Path(get_unique_output_path(str(final_report_path)))
                    source_report_path.replace(final_report_path)
            _log_message_callback(
                "output_format_converted",
                f"Converted final output to {final_output_format.upper()}: {config['output_filename']}"
            )

        source_is_epub = Path(input_path_for_translate_module or '').suffix.lower() == '.epub'
        output_is_epub = final_output_format == 'epub' and Path(output_filepath_on_server or '').suffix.lower() == '.epub'

        def _ready_for_final_audits() -> bool:
            current_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            return _job_is_ready_for_final_audits(
                operation_success=operation_success,
                current_status=(
                    state_manager.get_translation_field(translation_id, 'status') or ''
                ),
                failed_chunks=current_stats.get('failed_chunks') or 0,
            )

        if (
            source_is_epub
            and output_is_epub
            and os.path.exists(output_filepath_on_server)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _ready_for_final_audits()
        ):
            try:
                metadata_options = config.get('prompt_options') or {}
                metadata_title = str(metadata_options.get('target_metadata_title') or '').strip()
                metadata_subtitle = str(metadata_options.get('target_metadata_subtitle') or '').strip()
                if not metadata_title:
                    metadata_title, inferred_subtitle = infer_epub_title_page(output_filepath_on_server)
                    metadata_subtitle = metadata_subtitle or inferred_subtitle
                if metadata_title:
                    metadata_report = await engine.run_phase(
                        JobPhase.ASSEMBLE,
                        localize_epub_metadata,
                        input_path_for_translate_module,
                        output_filepath_on_server,
                        target_language=config.get('target_language') or '',
                        title=metadata_title,
                        subtitle=metadata_subtitle,
                        message="Localizing EPUB metadata and navigation.",
                        metadata={'output_format': 'epub', 'scope': 'metadata_navigation'},
                    )
                    _log_message_callback(
                        "epub_metadata_localized",
                        f"Metadatos EPUB localizados: {metadata_report.changed_files} archivo(s); "
                        f"título «{metadata_title}».",
                    )
            except Exception as metadata_error:
                _log_message_callback(
                    "epub_metadata_localization_error",
                    f"⛔ No se pudieron localizar los metadatos EPUB: {metadata_error}",
                )
                operation_success = False
                state_manager.set_translation_field(translation_id, 'status', 'partial')

        if (
            os.path.exists(output_filepath_on_server)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _ready_for_final_audits()
        ):
            try:
                artifact_report = await engine.run_phase(
                    JobPhase.AUDIT,
                    audit_and_clean_final_artifact,
                    output_filepath_on_server,
                    output_format=final_output_format,
                    write_report=True,
                    source_epub_path=(
                        input_path_for_translate_module
                        if final_output_format == 'epub'
                        and Path(input_path_for_translate_module or '').suffix.lower() == '.epub'
                        else None
                    ),
                    prompt_options=config.get('prompt_options') or {},
                    target_language=config.get('target_language') or '',
                    message="Running final artifact hygiene audit.",
                    metadata={'output_format': final_output_format},
                )
                if artifact_report.changed:
                    _log_message_callback(
                        "final_artifact_hygiene",
                        f"🧹 Chequeo final limpió {artifact_report.summary()}."
                    )
                elif artifact_report.unresolved_findings:
                    _log_message_callback(
                        "final_artifact_hygiene_warning",
                        f"⚠️ Chequeo final encontró {len(artifact_report.unresolved_findings)} advertencia(s)."
                    )
                elif artifact_report.reading_quality_warnings:
                    _log_message_callback(
                        "final_artifact_readability_warning",
                        f"⚠️ Chequeo final encontró {len(artifact_report.reading_quality_warnings)} alerta(s) de lectura."
                    )
            except Exception as artifact_error:
                _log_message_callback(
                    "final_artifact_hygiene_error",
                    f"⚠️ Chequeo final de artefactos no pudo ejecutarse: {artifact_error}"
                )
                operation_success = False
                state_manager.set_translation_field(translation_id, 'status', 'partial')

        if (
            source_is_epub
            and output_is_epub
            and os.path.exists(output_filepath_on_server)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _ready_for_final_audits()
        ):
            try:
                missing_block_report = await engine.run_phase(
                    JobPhase.REPAIR,
                    repair_epub_missing_blocks_with_llm,
                    input_path_for_translate_module,
                    output_filepath_on_server,
                    config=config,
                    log_callback=_log_message_callback,
                    max_blocks=int(
                        (config.get('prompt_options') or {}).get(
                            'max_final_missing_block_repairs',
                            16,
                        )
                    ),
                    timeout=min(int(config.get('request_timeout') or 180), 180),
                    message="Repairing source-proven empty EPUB blocks.",
                    metadata={'output_format': 'epub', 'scope': 'missing_text_blocks'},
                )
                if missing_block_report.repaired:
                    _log_message_callback(
                        "final_missing_blocks_repaired",
                        "🩹 Reparación final restauró "
                        f"{missing_block_report.repaired}/{missing_block_report.found} "
                        "bloque(s) omitido(s) y volvió a validarlos.",
                    )
                if missing_block_report.remaining or missing_block_report.errors:
                    _log_message_callback(
                        "final_missing_blocks_unresolved",
                        "⚠️ Reparación final conserva "
                        f"{missing_block_report.remaining} bloque(s) vacío(s) sin resolver; "
                        "el gate de publicación decidirá el resultado.",
                    )
            except Exception as missing_block_error:
                _log_message_callback(
                    "final_missing_blocks_error",
                    "⚠️ No se pudo ejecutar la reparación selectiva de bloques vacíos: "
                    f"{missing_block_error}",
                )

        if (
            os.path.exists(output_filepath_on_server)
            and input_path_for_translate_module
            and os.path.exists(input_path_for_translate_module)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _final_source_sample_audit_enabled(config.get('prompt_options') or {})
            and _ready_for_final_audits()
        ):
            try:
                source_sample_report = await engine.run_phase(
                    JobPhase.AUDIT,
                    audit_final_output_against_source_samples,
                    input_path_for_translate_module,
                    output_filepath_on_server,
                    source_language=config.get('source_language') or '',
                    target_language=config.get('target_language') or '',
                    write_report=True,
                    message="Running final source-aware sample audit.",
                    metadata={'output_format': final_output_format},
                )
                if not source_sample_report.clean:
                    audit_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                    state_manager.update_stats(
                        translation_id,
                        _final_source_sample_diagnostics(audit_stats, source_sample_report),
                    )
                    _log_message_callback(
                        "final_source_sample_audit_warning",
                        f"⚠️ Chequeo final contra fuente encontró {len(source_sample_report.issues)} alerta(s) en muestras."
                    )
                    if source_sample_report.error_count:
                        _log_message_callback(
                            "final_source_sample_advisory",
                            "ℹ️ Las alertas de muestra son diagnósticas; el gate completo "
                            "y la QA de formato decidirán si el archivo puede publicarse.",
                        )
                else:
                    audit_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                    state_manager.update_stats(
                        translation_id,
                        _final_source_sample_diagnostics(audit_stats, source_sample_report),
                    )
                    _log_message_callback(
                        "final_source_sample_audit",
                        "✅ Chequeo final contra fuente no encontró alertas en muestras."
                    )
            except Exception as source_sample_error:
                _log_message_callback(
                    "final_source_sample_audit_error",
                    f"⚠️ Chequeo final contra fuente no pudo ejecutarse: {source_sample_error}"
                )

        publication_report = None
        quality_run = None
        strict_epub_publication = bool(
            final_output_format == 'epub'
            and str(config.get('file_type') or '').lower() == 'epub'
            and Path(input_path_for_translate_module or '').suffix.lower() == '.epub'
            and not config.get('bilingual_output')
            and not (config.get('prompt_options') or {}).get('plain_text_mode')
            and (config.get('prompt_options') or {}).get('strict_epub_publication_gate', True) is not False
        )
        if (
            strict_epub_publication
            and os.path.exists(output_filepath_on_server)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _ready_for_final_audits()
        ):
            try:
                publication_report = await engine.run_phase(
                    JobPhase.AUDIT,
                    audit_epub_publication,
                    input_path_for_translate_module,
                    output_filepath_on_server,
                    source_language=config.get('source_language') or '',
                    target_language=config.get('target_language') or '',
                    message="Running strict whole-EPUB publication gate.",
                    metadata={'output_format': 'epub', 'scope': 'whole_epub'},
                )
                publication_report.write_json(
                    Path(output_filepath_on_server).with_name(
                        f"{Path(output_filepath_on_server).stem} - publication gate.json"
                    )
                )
                if not publication_report.publishable:
                    operation_success = False
                    audit_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                    failed_count = max(
                        1,
                        int(audit_stats.get('failed_chunks') or 0),
                        int(publication_report.source_units - publication_report.audited_units),
                    )
                    state_manager.update_stats(translation_id, {
                        **audit_stats,
                        'failed_chunks': failed_count,
                        'checkpoint_failed_chunks': failed_count,
                        'quality_degraded': True,
                    })
                    state_manager.set_translation_field(translation_id, 'status', 'partial')
                    checkpoint_manager.mark_partial(translation_id)
                    _log_message_callback(
                        "epub_publication_gate_failed",
                        "⛔ El EPUB ensamblado no cumple el contrato completo; "
                        "queda parcial y no se publicará como completado: "
                        + "; ".join(publication_report.errors[:4]),
                    )
                else:
                    stats = _canonicalize_published_epub_stats(
                        state_manager.get_translation_field(translation_id, 'stats') or stats,
                        publication_report,
                    )
                    state_manager.update_stats(translation_id, stats)
                    accumulated = dict(stats.get('epub_accumulated_stats') or {})
                    completed_chunks = int(stats.get('completed_chunks') or 0)
                    checkpoint_manager.update_progress(
                        translation_id,
                        current_chunk_index=max(-1, completed_chunks - 1),
                        total_chunks=int(stats.get('total_chunks') or 0),
                        completed_chunks=completed_chunks,
                        failed_chunks=0,
                        epub_accumulated_stats=accumulated or None,
                    )
                    _log_message_callback(
                        "epub_publication_gate_passed",
                        f"✅ EPUB validado: {publication_report.audited_units}/"
                        f"{publication_report.source_units} unidades AUDITED; EPUBCheck sin errores.",
                    )
            except Exception as publication_error:
                operation_success = False
                state_manager.set_translation_field(translation_id, 'status', 'partial')
                checkpoint_manager.mark_partial(translation_id)
                _log_message_callback(
                    "epub_publication_gate_error",
                    f"⛔ No se pudo validar el EPUB completo; publicación bloqueada: {publication_error}",
                )

        if (
            os.path.exists(output_filepath_on_server)
            and input_path_for_translate_module
            and os.path.exists(input_path_for_translate_module)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and _ready_for_final_audits()
        ):
            try:
                checkpoint_data_for_qa = checkpoint_manager.load_checkpoint(translation_id)
                quality_run = await engine.run_phase(
                    JobPhase.AUDIT,
                    run_quality_assurance,
                    source_path=input_path_for_translate_module,
                    output_path=output_filepath_on_server,
                    source_language=config.get('source_language') or '',
                    target_language=config.get('target_language') or '',
                    run_id=translation_id,
                    job_config=config,
                    checkpoint_data=checkpoint_data_for_qa,
                    publication_report=publication_report,
                    checkpoint_manager=checkpoint_manager,
                    message="Running universal whole-book quality gates.",
                    metadata={'output_format': final_output_format, 'scope': 'whole_book'},
                )
                config['quality_report_dir'] = str(quality_run.paths.root)
                config['quality_assurance_result'] = {
                    'status': quality_run.report.status.value,
                    'publishable': quality_run.publishable,
                    'report_dir': str(quality_run.paths.root),
                }
                _persist_runtime_checkpoint_config(
                    checkpoint_manager,
                    translation_id,
                    config,
                    _log_message_callback,
                )
                qa_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                state_manager.update_stats(translation_id, {
                    **qa_stats,
                    'quality_gate_status': quality_run.report.status.value,
                    'quality_report_dir': str(quality_run.paths.root),
                    'quality_repair_units': len(quality_run.report.repair_units),
                })
                if not quality_run.publishable:
                    operation_success = False
                    failed_count = max(
                        1,
                        int(qa_stats.get('failed_chunks') or 0),
                        len(quality_run.report.repair_units),
                    )
                    state_manager.update_stats(translation_id, {
                        **qa_stats,
                        'failed_chunks': failed_count,
                        'checkpoint_failed_chunks': max(
                            failed_count,
                            int(qa_stats.get('checkpoint_failed_chunks') or 0),
                        ),
                        'quality_degraded': True,
                        'quality_gate_status': quality_run.report.status.value,
                        'quality_report_dir': str(quality_run.paths.root),
                        'quality_repair_units': len(quality_run.report.repair_units),
                    })
                    state_manager.set_translation_field(translation_id, 'status', 'partial')
                    checkpoint_manager.mark_partial(translation_id)
                    _log_message_callback(
                        "universal_quality_gate_failed",
                        "⛔ La entrega no pasó los controles universales; "
                        f"{len(quality_run.report.repair_units)} unidad(es) quedaron en reparación selectiva. "
                        f"Reporte: {quality_run.paths.report_html}",
                    )
                else:
                    _log_message_callback(
                        "universal_quality_gate_passed",
                        "✅ La entrega pasó los 10 controles universales de calidad "
                        f"({quality_run.report.status.value}).",
                    )
            except Exception as quality_error:
                operation_success = False
                state_manager.set_translation_field(translation_id, 'status', 'partial')
                checkpoint_manager.mark_partial(translation_id)
                _log_message_callback(
                    "universal_quality_gate_error",
                    f"⛔ No se pudo demostrar la integridad del libro; publicación bloqueada: {quality_error}",
                )

        if (
            os.path.exists(output_filepath_on_server)
            and state_manager.get_translation_field(translation_id, 'status') == 'partial'
            and not Path(output_filepath_on_server).name.startswith('[partial] ')
        ):
            partial_output = get_partial_output_path(output_filepath_on_server)
            Path(output_filepath_on_server).replace(partial_output)
            output_filepath_on_server = partial_output
            config['output_filename'] = os.path.basename(partial_output)
            if quality_run is not None:
                quality_run.update_output_path(partial_output)
            _log_message_callback(
                "failed_publication_quarantined",
                f"📦 El artefacto no validado se guardó sólo como parcial: {config['output_filename']}",
            )

        if (
            os.path.exists(output_filepath_on_server)
            and not state_manager.get_translation_field(translation_id, 'interrupted')
            and state_manager.get_translation_field(translation_id, 'status') != 'partial'
            and audiobook_profile_enabled(config.get('prompt_options') or {})
        ):
            try:
                companion_result = await engine.run_phase(
                    JobPhase.ASSEMBLE,
                    _create_audiobook_companion_outputs,
                    output_path=output_filepath_on_server,
                    output_format=final_output_format,
                    target_language=config.get('target_language') or 'Spanish',
                    output_dir=output_dir,
                    message="Creating audiobook companion outputs.",
                    metadata={'profile_id': active_profile_id, 'audiobook': True},
                )
                audiobook_companion_files = list(companion_result.get('files') or [])
                summary = companion_result.get('summary') or 'no changes'
                if audiobook_companion_files:
                    _log_message_callback(
                        "audiobook_companions_created",
                        "🎧 Audiobook ready: "
                        f"{', '.join(audiobook_companion_files)} ({summary})."
                    )
            except Exception as audiobook_error:
                _log_message_callback(
                    "audiobook_companion_error",
                    f"⚠️ Could not create audiobook companion files: {audiobook_error}"
                )

        # Set result message based on file type
        file_type_upper = final_output_format.upper()
        if (
            os.path.exists(output_filepath_on_server)
            and state_manager.get_translation_field(translation_id, 'status')
                not in ['error', 'interrupted_before_save', 'partial']
        ):
            state_manager.set_translation_field(translation_id, 'result', f"[{file_type_upper} file translated - download to view]")
        else:
            state_manager.set_translation_field(translation_id, 'result', f"[{file_type_upper} file (partially) translated - content not loaded for preview or write failed]")

        # Clean up temporary text file if created
        if temp_txt_file_path and os.path.exists(temp_txt_file_path):
            os.remove(temp_txt_file_path)

        state_manager.set_translation_field(translation_id, 'output_filepath', output_filepath_on_server)

        stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        elapsed_time = _apply_active_timing(stats).get('elapsed_time', 0.0)
        _finalize_stats_callback({'elapsed_time': elapsed_time})

        engine.mark_phase(JobPhase.PUBLISH, "Publishing final job status.")
        final_status_payload = {
            'result': state_manager.get_translation_field(translation_id, 'result'),
            'output_filename': config['output_filename'],
            'output_dir': os.path.dirname(os.path.abspath(output_filepath_on_server)),
            'file_type': final_output_format,
            'companion_files': audiobook_companion_files,
        }

        if state_manager.get_translation_field(translation_id, 'interrupted'):
            state_manager.set_translation_field(translation_id, 'status', 'interrupted')
            _log_message_callback("summary_interrupted", f"🛑 Translation interrupted - partial result saved ({elapsed_time:.2f}s)")
            final_status_payload['status'] = 'interrupted'
            await asyncio.to_thread(notify, EVENT_INTERRUPTION,
                _notification_context(config, translation_id, elapsed_time))

            # Mark checkpoint as interrupted in database
            checkpoint_manager.mark_interrupted(translation_id)

            # Emit checkpoint_created event to trigger UI update
            socketio.emit('checkpoint_created', {
                'translation_id': translation_id,
                'status': 'interrupted',
                'message': 'Translation paused - checkpoint created'
            }, namespace='/')

            # DON'T clean up uploaded file on interruption - keep it for resume capability
            # The file will be preserved in the job-specific directory by checkpoint_manager
            # Only clean up if the preserved file exists (meaning backup was successful)
            preserved_path = config.get('preserved_input_path')
            if preserved_path and Path(preserved_path).exists():
                # Preserved file exists, we can safely delete the original upload
                if 'file_path' in config and config['file_path']:
                    uploaded_file_path = config['file_path']
                    upload_path = Path(uploaded_file_path)

                    if upload_path.exists() and upload_path != Path(preserved_path):
                        try:
                            # Only delete if it's in the uploads directory root (not in a job subdirectory)
                            uploads_dir = Path(output_dir) / 'uploads'
                            resolved_path = upload_path.resolve()

                            # Check if file is directly in uploads/ (not in a job subdirectory)
                            if resolved_path.parent.resolve() == uploads_dir.resolve():
                                upload_path.unlink()
                                _log_message_callback("cleanup_uploaded_file", f"🗑️ Cleaned up uploaded source file (preserved copy exists): {upload_path.name}")
                            else:
                                _log_message_callback("cleanup_skipped", f"ℹ️ Skipped cleanup - file is not in uploads root directory")
                        except Exception as e:
                            _log_message_callback("cleanup_error", f"⚠️ Could not delete uploaded file {upload_path.name}: {str(e)}")
                else:
                    _log_message_callback("cleanup_info", "ℹ️ Original upload file not found or already cleaned up")
            else:
                _log_message_callback("cleanup_skipped_no_preserve", "ℹ️ Skipped cleanup - preserved file not found, keeping original for resume")

        elif state_manager.get_translation_field(translation_id, 'status') != 'error':
            # Get stats for consolidated message
            final_stats = stats
            stats_summary = ""
            failed = 0
            status_before_publish = str(
                state_manager.get_translation_field(translation_id, 'status') or ''
            ).strip().lower()
            final_checkpoint = checkpoint_progress_snapshot(
                checkpoint_manager,
                translation_id,
            )
            if final_checkpoint:
                checkpoint_failed = int(final_checkpoint.get('failed_chunks') or 0)
                known_failed = int(final_stats.get('failed_chunks') or 0)
                final_stats = {
                    **final_stats,
                    'total_chunks': final_checkpoint.get('total_chunks', final_stats.get('total_chunks', 0)),
                    'completed_chunks': final_checkpoint.get('completed_chunks', final_stats.get('completed_chunks', 0)),
                    'failed_chunks': max(known_failed, checkpoint_failed),
                    'checkpoint_failed_chunks': max(
                        int(final_stats.get('checkpoint_failed_chunks') or 0),
                        checkpoint_failed,
                    ),
                }
                state_manager.update_stats(translation_id, final_stats)
            if (config['file_type'] in ('txt', 'srt', 'pdf')
                    or (config['file_type'] == 'epub' and final_stats.get('total_chunks', 0) > 0)):
                completed = final_stats.get('completed_chunks', 0)
                failed = final_stats.get('failed_chunks', 0)
                total = final_stats.get('total_chunks', 0)
                unit = 'subtitles' if config['file_type'] == 'srt' else 'chunks'
                stats_summary = f" | {completed}/{total} {unit}"
                if failed > 0:
                    stats_summary += f" ({failed} failed)"

            checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
            if (
                checkpoint_data
                and checkpoint_data.get('checkpoint_complete')
                and operation_success
                and status_before_publish != 'partial'
            ):
                total = checkpoint_data['job']['progress'].get('total_chunks', 0)
                completed = total
                failed = 0
                unit = 'subtitles' if config['file_type'] == 'srt' else 'chunks'
                stats_summary = f" | {completed}/{total} {unit}"
                state_manager.update_stats(translation_id, {
                    'total_chunks': total,
                    'completed_chunks': completed,
                    'failed_chunks': 0,
                })

            # If chunks remain in failed state after auto-retry, keep the job resumable
            # as 'partial' instead of marking it 'completed' and deleting the checkpoint.
            # The user can then resume to retry the failed chunks without re-running the file.
            unresolved = _job_has_unresolved_work(
                operation_success=operation_success,
                current_status=status_before_publish,
                failed_chunks=failed,
            )
            if unresolved:
                automatic_recovery_plan = _build_failed_chunk_recovery_plan(
                    config,
                    checkpoint_data,
                    final_stats,
                )
                if automatic_recovery_plan is None:
                    automatic_recovery_plan = _build_finalization_recovery_plan(
                        config,
                        checkpoint_data,
                    )
                recovery_scope = (
                    automatic_recovery_plan.get('scope')
                    if automatic_recovery_plan
                    else None
                )
                checkpoint_progress = (
                    ((checkpoint_data or {}).get('job') or {}).get('progress') or {}
                )
                checkpoint_chunk_failures = max(
                    int(checkpoint_progress.get('failed_chunks') or 0),
                    len((checkpoint_data or {}).get('failed_chunk_indices') or []),
                )
                finalization_only = bool(
                    checkpoint_data
                    and checkpoint_data.get('checkpoint_complete')
                    and checkpoint_chunk_failures == 0
                )
                if finalization_only:
                    failed = 0
                    final_stats = {
                        **final_stats,
                        'failed_chunks': 0,
                        'checkpoint_failed_chunks': 0,
                        'quality_degraded': True,
                    }
                else:
                    failed = max(1, int(failed or 0))
                    final_stats = {
                        **final_stats,
                        'failed_chunks': failed,
                        'checkpoint_failed_chunks': max(
                            failed,
                            int(final_stats.get('checkpoint_failed_chunks') or 0),
                        ),
                        'quality_degraded': True,
                    }
                if automatic_recovery_plan:
                    cycle = automatic_recovery_plan['cycle']
                    delay_seconds = automatic_recovery_plan['delay_seconds']
                    if recovery_scope == 'finalization':
                        live_status = (
                            "Reintentando ensamblado y validación final "
                            f"(ciclo {cycle})"
                        )
                        result_message = '[Automatic finalization recovery in progress]'
                        recovery_event = 'finalization_auto_recovery_scheduled'
                        recovery_log = (
                            "🔄 El texto ya está completo. El sistema repetirá sólo "
                            f"el ensamblado y los controles finales en {delay_seconds}s "
                            f"(ciclo {cycle}); no se retraducirán fragmentos."
                        )
                    else:
                        live_status = (
                            f"Corrigiendo {failed} fragmento(s) fallido(s) automáticamente "
                            f"(ciclo {cycle})"
                        )
                        result_message = '[Automatic failed-chunk recovery in progress]'
                        recovery_event = 'failed_chunk_auto_recovery_scheduled'
                        recovery_log = (
                            f"🔄 Se detectaron {failed} fragmento(s) sin resolver. "
                            f"El sistema continuará desde el checkpoint en {delay_seconds}s "
                            f"(ciclo {cycle}); no se requiere reanudación manual."
                        )
                    final_stats.update({
                        'failure_recovery_cycle': cycle,
                        'failure_recovery_stuck_count': automatic_recovery_plan['stuck_count'],
                        'live_status': live_status,
                        'live_status_kind': 'active',
                        'last_activity_at': time.time(),
                    })
                    state_manager.update_stats(translation_id, final_stats)
                    state_manager.set_translation_field(translation_id, 'status', 'running')
                    state_manager.set_translation_field(
                        translation_id,
                        'result',
                        result_message,
                    )
                    checkpoint_manager.mark_running(translation_id)
                    checkpoint_manager.update_job_config(
                        translation_id,
                        automatic_recovery_plan['config'],
                    )
                    _log_message_callback(
                        recovery_event,
                        recovery_log,
                    )
                    final_status_payload.update({
                        'status': 'running',
                        'recovering': True,
                        'recovery_scope': recovery_scope,
                        'recovery_cycle': cycle,
                        'retry_after': delay_seconds,
                        'result': state_manager.get_translation_field(translation_id, 'result'),
                    })
                else:
                    if finalization_only:
                        final_stats = _finalization_recovery_exhausted_stats(final_stats)
                        partial_message = (
                            "⚠️ El texto está completo, pero la validación final no "
                            "pudo demostrarse tras dos intentos; checkpoint conservado"
                        )
                    else:
                        final_stats = _failed_chunk_recovery_exhausted_stats(
                            final_stats,
                            failed_chunks=failed,
                        )
                        partial_message = (
                            f"⚠️ {'Transformation' if is_transform_job else 'Translation'} "
                            f"finished with {failed} failed chunk(s) in {elapsed_time:.2f}s"
                            f"{stats_summary} — checkpoint kept for retry"
                        )
                    state_manager.update_stats(translation_id, final_stats)
                    state_manager.set_translation_field(translation_id, 'status', 'partial')
                    _log_message_callback("summary_partial", partial_message)
                    final_status_payload['status'] = 'partial'
                    checkpoint_manager.mark_partial(translation_id)
                    # Skip cleanup_completed_job — we want the checkpoint to survive for retry.
            else:
                state_manager.set_translation_field(translation_id, 'status', 'completed')
                _log_message_callback(
                    "summary_completed",
                    f"✅ {'Transformation' if is_transform_job else 'Translation'} completed in {elapsed_time:.2f}s{stats_summary}",
                )
                final_status_payload['status'] = 'completed'
                await asyncio.to_thread(notify, EVENT_SUCCESS,
                    _notification_context(config, translation_id, elapsed_time))

                qa_settings = config.get('quality_assurance') or {}
                export_settings = qa_settings.get('export') or {}
                keep_intermediate = bool(
                    export_settings.get('keep_intermediate_files', True)
                )
                if keep_intermediate:
                    checkpoint_manager.mark_completed(translation_id, quality_gate_passed=True)
                    _log_message_callback(
                        "quality_checkpoint_preserved",
                        "Quality manifest and validated checkpoint preserved for traceability.",
                    )
                else:
                    checkpoint_manager.cleanup_completed_job(translation_id)

            # Clean up uploaded file if it exists and is in the uploads directory
            # On completion, we can safely delete the original upload file
            if automatic_recovery_plan is None and 'file_path' in config and config['file_path']:
                uploaded_file_path = config['file_path']
                # Convert to Path object for reliable path operations
                upload_path = Path(uploaded_file_path)

                # Check if file exists
                if upload_path.exists():
                    try:
                        # Only delete if it's in the uploads directory root (not in a job subdirectory)
                        uploads_dir = Path(output_dir) / 'uploads'
                        resolved_path = upload_path.resolve()

                        # Check if file is directly in uploads/ (not in a job subdirectory)
                        if resolved_path.parent.resolve() == uploads_dir.resolve():
                            upload_path.unlink()
                            # Removed verbose cleanup message - file cleanup is automatic
                    except Exception as e:
                        _log_message_callback("cleanup_error", f"⚠️ Could not delete uploaded file {upload_path.name}: {str(e)}")
        else:
            _log_message_callback("summary_error_final", f"❌ Translation finished with errors ({elapsed_time:.2f}s)")
            final_status_payload['status'] = 'error'
            final_status_payload['error'] = state_manager.get_translation_field(translation_id, 'error') or 'Unknown error during finalization.'
            await asyncio.to_thread(notify, EVENT_FAILURE,
                _notification_context(config, translation_id, elapsed_time,
                                      error=final_status_payload['error']))

        # Stats are now included in the consolidated completion message above

        # Log OpenRouter cost summary if applicable
        if config.get('llm_provider') == 'openrouter':
            cost = stats.get('openrouter_cost', 0.0)
            prompt_tokens = stats.get('openrouter_prompt_tokens', 0)
            completion_tokens = stats.get('openrouter_completion_tokens', 0)
            total_tokens = prompt_tokens + completion_tokens
            if cost > 0 or total_tokens > 0:
                _log_message_callback("openrouter_cost_final",
                    f"💰 OpenRouter Cost: ${cost:.4f} | Tokens: {total_tokens:,} ({prompt_tokens:,} prompt + {completion_tokens:,} completion)")
            # Clear the callback to avoid memory leaks
            OpenRouterProvider.set_cost_callback(None)

        # TTS Generation (if enabled and translation completed successfully)
        if config.get('tts_enabled') and final_status_payload.get('status') == 'completed':
            await _perform_tts_generation(
                translation_id,
                config,
                output_filepath_on_server,
                state_manager,
                socketio,
                _log_message_callback
            )

        # Attach final stats so the completion card can render its summary
        # (cost, tokens, failed chunks…). emit_update no longer auto-attaches.
        final_status_payload['stats'] = state_manager.get_translation_field(translation_id, 'stats') or {}
        emit_update(socketio, translation_id, final_status_payload, state_manager)

        if automatic_recovery_plan is not None:
            _schedule_failed_chunk_recovery(
                translation_id,
                automatic_recovery_plan,
                state_manager,
                output_dir,
                socketio,
            )
            return

        # Trigger file list refresh in the frontend if a file was saved
        if os.path.exists(output_filepath_on_server) and final_status_payload['status'] in ['completed', 'interrupted', 'partial']:
            socketio.emit('file_list_changed', {
                'reason': final_status_payload['status'],
                'filename': config.get('output_filename', 'unknown'),
                'companion_files': audiobook_companion_files,
            }, namespace='/')

    except RateLimitError as e:
        pricing_pause = isinstance(e, DeepSeekPeakPricingError)
        credits_exhausted = isinstance(e, InsufficientCreditsError) or not getattr(e, 'retryable', True)
        auto_pause = (
            False
            if pricing_pause
            else (
                True
                if credits_exhausted
                else config.get('auto_pause_on_rate_limit', AUTO_PAUSE_ON_RATE_LIMIT)
            )
        )
        retry_msg = f" Retry suggested after ~{e.retry_after}s." if e.retry_after else ""
        provider_name = e.provider or config.get('llm_provider', 'API')

        if not state_manager.exists(translation_id):
            return

        # Auto-resume mode keeps the job running: wait, then re-enter from the checkpoint.
        if not auto_pause:
            wait_seconds = e.retry_after or RATE_LIMIT_AUTO_RESUME_DELAY
            if pricing_pause:
                resume_local = getattr(e, 'next_available_at_local', '')
                wait_msg = (
                    "DeepSeek entró en horario de tarifa alta. No se enviarán "
                    f"tokens; reanudación automática en CDMX: {resume_local}."
                )
                wait_event = "deepseek_peak_pricing_wait"
                waiting_status = "pricing_wait"
                pause_reason = "deepseek_peak_pricing"
            else:
                wait_msg = (f"⏳ Rate limited by {provider_name}.{retry_msg} "
                            f"Auto-resume in {wait_seconds}s (auto-pause disabled).")
                wait_event = "rate_limit_auto_resume"
                waiting_status = "rate_limited"
                pause_reason = "rate_limited"
            _log_message_callback(wait_event, wait_msg)

            # A pricing wait is active, recoverable work rather than a terminal
            # provider error. Keep it visible across page refreshes.
            state_manager.set_translation_field(translation_id, 'status', waiting_status)
            state_manager.set_translation_field(translation_id, 'interrupted', False)
            state_manager.set_translation_field(translation_id, 'pause_reason', pause_reason)
            if pricing_pause:
                resume_at_utc = getattr(e, 'next_available_at_utc', '')
                resume_at_local = getattr(e, 'next_available_at_local', '')
                state_manager.set_translation_field(
                    translation_id, 'resume_at_utc', resume_at_utc
                )
                state_manager.set_translation_field(
                    translation_id, 'resume_at_local', resume_at_local
                )
                config = dict(config)
                config['_pricing_pause_until_utc'] = resume_at_utc
                config['_pricing_pause_timezone'] = getattr(
                    e, 'display_timezone', 'America/Mexico_City'
                )
                state_manager.set_translation_field(translation_id, 'config', config)
                checkpoint_manager.update_job_config(translation_id, config)
                paused_stats = _apply_active_timing(
                    state_manager.get_translation_field(translation_id, 'stats') or {}
                )
                paused_stats.update({
                    'active_elapsed_before_run': paused_stats.get(
                        'active_elapsed_seconds', 0.0
                    ),
                    'live_status': wait_msg,
                    'live_status_kind': 'scheduled_pause',
                    'live_activity_event': wait_event,
                    'pricing_resume_at_utc': resume_at_utc,
                    'pricing_resume_at_local': resume_at_local,
                    'eta_seconds': None,
                    'last_activity_at': time.time(),
                })
                state_manager.set_translation_field(
                    translation_id, 'stats', paused_stats
                )
            emit_update(socketio, translation_id, {
                'status': waiting_status,
                'log': wait_msg,
                'reason': pause_reason,
                'resume_at_utc': getattr(e, 'next_available_at_utc', None),
                'resume_at_local': getattr(e, 'next_available_at_local', None),
                'display_timezone': getattr(e, 'display_timezone', None),
                'stats': (
                    state_manager.get_translation_field(translation_id, 'stats') or {}
                    if pricing_pause else None
                ),
            }, state_manager)

            await asyncio.sleep(wait_seconds)

            # A manual interrupt wins over provider recovery and keeps its own
            # durable reason instead of being mislabeled as a rate-limit pause.
            if state_manager.get_translation_field(translation_id, 'interrupted'):
                _log_message_callback("rate_limit_auto_resume_cancelled",
                    "🛑 Auto-resume cancelled by user; checkpoint preserved.")
                state_manager.set_translation_field(
                    translation_id, 'status', 'interrupted'
                )
                state_manager.set_translation_field(
                    translation_id, 'pause_reason', 'manual'
                )
                checkpoint_manager.mark_interrupted(translation_id)
                emit_update(socketio, translation_id, {
                    'status': 'interrupted',
                    'reason': 'manual',
                    'log': "🛑 Reanudación automática cancelada por el usuario.",
                }, state_manager)
                return
            else:
                cp_data = checkpoint_manager.load_checkpoint(translation_id)
                if cp_data:
                    # Track consecutive auto-resume cycles that fail without
                    # advancing the checkpoint. The bounded budget prevents a
                    # throttled account from keeping a book "active" for hours.
                    resume_index = int(cp_data['resume_from_index'])
                    if pricing_pause:
                        # This wait has a deterministic end. It must not consume
                        # the bounded retry budget intended for repeated 429s.
                        rate_limit_plan = _build_pricing_auto_resume_plan(
                            config,
                            resume_index=resume_index,
                        )
                    else:
                        rate_limit_plan = _build_rate_limit_auto_resume_plan(
                            config,
                            resume_index=resume_index,
                        )
                    stuck_count = rate_limit_plan["stuck_count"]
                    max_rate_limit_resumes = rate_limit_plan["max_resumes"]
                    if not rate_limit_plan["allowed"]:
                        _log_message_callback(
                            "rate_limit_auto_resume_exhausted",
                            f"⏸️ El proveedor sigue limitando el trabajo en el fragmento "
                            f"{resume_index}. Se agotaron {max_rate_limit_resumes} "
                            "reanudaciones automáticas sin avance; el checkpoint queda "
                            "intacto para evitar un ciclo de horas.",
                        )
                    else:
                        new_config = rate_limit_plan["config"]
                        checkpoint_manager.mark_running(translation_id)
                        checkpoint_manager.update_job_config(translation_id, new_config)
                        state_manager.set_translation_field(translation_id, 'config', new_config)
                        state_manager.set_translation_field(translation_id, 'status', 'running')
                        state_manager.set_translation_field(translation_id, 'interrupted', False)
                        state_manager.set_translation_field(translation_id, 'pause_reason', None)
                        state_manager.set_translation_field(translation_id, 'resume_at_utc', None)
                        state_manager.set_translation_field(translation_id, 'resume_at_local', None)
                        emit_update(socketio, translation_id, {
                            'status': 'running',
                            'log': f"▶️ Auto-resuming from chunk {resume_index}..."
                        }, state_manager)
                        # Start a fresh worker instead of recursively awaiting this
                        # coroutine. Repeated throttling must not grow the stack.
                        start_translation_job(
                            translation_id, new_config, state_manager, output_dir, socketio
                        )
                        _log_message_callback(
                            "rate_limit_auto_resume_scheduled",
                            f"▶️ Reanudación programada desde el fragmento {resume_index}; "
                            "el worker anterior se cerrará limpiamente.",
                        )
                        return
                # No checkpoint available, fall through to the pause path below.
                _log_message_callback("rate_limit_no_checkpoint",
                    "⚠️ Auto-resume requested but no checkpoint found, falling back to pause.")

        if credits_exhausted:
            pause_msg = (
                f"💳 {provider_name} no tiene saldo suficiente. El trabajo quedó pausado "
                "en su último checkpoint; agrega saldo o configura otra clave y pulsa Reanudar."
            )
            pause_event = "provider_credits_exhausted"
            pause_reason = "insufficient_credits"
        else:
            pause_msg = (
                f"⏸️ Rate limited by {provider_name}.{retry_msg} "
                "Translation auto-paused, you can resume when ready."
            )
            pause_event = "rate_limit_auto_pause"
            pause_reason = "rate_limited"
        _log_message_callback(pause_event, pause_msg)

        state_manager.set_translation_field(translation_id, 'status', 'rate_limited')
        state_manager.set_translation_field(translation_id, 'interrupted', True)
        state_manager.set_translation_field(translation_id, 'pause_reason', pause_reason)
        checkpoint_manager.mark_interrupted(translation_id)

        stats = state_manager.get_translation_field(translation_id, 'stats') or {}
        elapsed_time = _apply_active_timing(stats).get('elapsed_time', 0.0)
        _finalize_stats_callback({'elapsed_time': elapsed_time})

        emit_update(socketio, translation_id, {
            'status': 'rate_limited',
            'log': pause_msg,
            'reason': pause_reason,
            'result': state_manager.get_translation_field(translation_id, 'result') or (
                "Translation paused (insufficient credits)"
                if credits_exhausted else "Translation paused (rate limited)"
            )
        }, state_manager)

        socketio.emit('checkpoint_created', {
            'translation_id': translation_id,
            'status': 'rate_limited',
            'reason': pause_reason,
            'message': pause_msg
        }, namespace='/')

        output_filepath = state_manager.get_translation_field(translation_id, 'output_filepath')
        if output_filepath and os.path.exists(output_filepath):
            socketio.emit('file_list_changed', {
                'reason': 'rate_limited',
                'filename': config.get('output_filename', 'unknown')
            }, namespace='/')

    except Exception as e:
        critical_error_msg = f"Critical error during translation task ({translation_id}): {str(e)}"
        _log_message_callback("critical_error_perform_task", critical_error_msg)
        import traceback
        tb_str = traceback.format_exc()
        _log_message_callback("critical_error_perform_task_traceback", tb_str)

        if state_manager.exists(translation_id):
            if state_manager.get_translation_field(translation_id, 'interrupted'):
                state_manager.set_translation_field(
                    translation_id, 'status', 'interrupted'
                )
                state_manager.set_translation_field(translation_id, 'error', None)
                checkpoint_manager.mark_interrupted(translation_id)
                emit_update(socketio, translation_id, {
                    'status': 'interrupted',
                    'reason': (
                        state_manager.get_translation_field(
                            translation_id, 'pause_reason'
                        ) or 'manual'
                    ),
                    'log': "🛑 Pausa confirmada; checkpoint preservado.",
                }, state_manager)
                return

            checkpoint_data = None
            try:
                stats = state_manager.get_translation_field(translation_id, 'stats') or {}
                checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
                chunks = checkpoint_data.get('chunks', []) if checkpoint_data else []
                total = 0
                if checkpoint_data:
                    total = int(
                        checkpoint_data.get('job', {})
                        .get('progress', {})
                        .get('total_chunks') or 0
                    )
                if total <= 0 and not chunks and int(stats.get('completed_chunks') or 0) == 0:
                    checkpoint_manager.delete_checkpoint(translation_id)
            except Exception:
                pass

            recovery_plan = _build_worker_exception_recovery_plan(
                config,
                checkpoint_data,
            )
            if recovery_plan is not None:
                cycle = recovery_plan["cycle"]
                delay_seconds = recovery_plan["delay_seconds"]
                recovery_stats = state_manager.get_translation_field(
                    translation_id, 'stats'
                ) or {}
                recovery_stats.update({
                    'live_status': (
                        "Recuperando el proceso desde el último checkpoint "
                        f"(ciclo {cycle})"
                    ),
                    'live_status_kind': 'active',
                    'live_activity_event': 'worker_exception_auto_recovery',
                    'last_activity_at': time.time(),
                    'worker_error_recovery_cycle': cycle,
                })
                state_manager.update_stats(translation_id, recovery_stats)
                state_manager.set_translation_field(
                    translation_id, 'config', recovery_plan['config']
                )
                state_manager.set_translation_field(translation_id, 'status', 'running')
                state_manager.set_translation_field(translation_id, 'error', None)
                checkpoint_manager.mark_running(translation_id)
                checkpoint_manager.update_job_config(
                    translation_id,
                    recovery_plan['config'],
                )
                _log_message_callback(
                    "worker_exception_auto_recovery_scheduled",
                    "🔄 El worker falló, pero el avance está íntegro. "
                    f"Se reanudará desde el checkpoint en {delay_seconds}s "
                    f"(intento {recovery_plan['stuck_count']}/"
                    f"{recovery_plan['max_stuck_recoveries']} en este punto).",
                )
                emit_update(socketio, translation_id, {
                    'status': 'running',
                    'recovering': True,
                    'recovery_scope': 'worker',
                    'recovery_cycle': cycle,
                    'retry_after': delay_seconds,
                    'stats': recovery_stats,
                }, state_manager)
                _schedule_failed_chunk_recovery(
                    translation_id,
                    recovery_plan,
                    state_manager,
                    output_dir,
                    socketio,
                )
                return

            state_manager.set_translation_field(translation_id, 'status', 'error')
            state_manager.set_translation_field(translation_id, 'error', critical_error_msg)
            # Unconditional: the DB row exists independently of whether we
            # could load its checkpoint payload just now (that read has its
            # own try/except above and may have failed transiently). Gating
            # this on `checkpoint_data` left the SQLite status at 'running'
            # whenever the load failed, desyncing it from the in-memory
            # 'error' set above and hiding the job from get_resumable_jobs().
            try:
                checkpoint_manager.mark_error(translation_id)
            except Exception:
                pass
            emit_update(socketio, translation_id, {
                'error': critical_error_msg,
                'status': 'error',
                'result': state_manager.get_translation_field(translation_id, 'result') or f"Translation failed: {critical_error_msg}"
            }, state_manager)

            stats = state_manager.get_translation_field(translation_id, 'stats') or {}
            elapsed_time = _apply_active_timing(stats).get('elapsed_time', 0.0)
            await asyncio.to_thread(notify, EVENT_FAILURE,
                _notification_context(config, translation_id, elapsed_time,
                                      error=critical_error_msg))


async def _perform_tts_generation(translation_id, config, output_filepath, state_manager, socketio, log_callback):
    """
    Perform TTS generation after successful translation.

    Args:
        translation_id: Translation job ID
        config: Translation configuration dict
        output_filepath: Path to the translated file
        state_manager: State manager instance
        socketio: SocketIO instance for WebSocket events
        log_callback: Logging callback function
    """
    try:
        log_callback("tts_phase_start", "🔊 Starting TTS audio generation...")

        # Emit TTS started event
        socketio.emit('tts_update', {
            'translation_id': translation_id,
            'status': 'started',
            'message': 'TTS generation started'
        }, namespace='/')

        # Reconstruct TTSConfig from dict
        tts_config_dict = config.get('tts_config', {})
        tts_config = TTSConfig(
            enabled=True,
            provider=tts_config_dict.get('provider', 'edge-tts'),
            voice=tts_config_dict.get('voice', ''),
            rate=tts_config_dict.get('rate', '+0%'),
            volume=tts_config_dict.get('volume', '+0%'),
            pitch=tts_config_dict.get('pitch', '+0Hz'),
            output_format=tts_config_dict.get('output_format', 'opus'),
            bitrate=tts_config_dict.get('bitrate', '64k'),
            sample_rate=tts_config_dict.get('sample_rate', 24000),
            chunk_size=tts_config_dict.get('chunk_size', 5000),
            pause_between_chunks=tts_config_dict.get('pause_between_chunks', 0.5)
        )

        target_language = config.get('target_language', '')

        # Create TTS progress callback
        def tts_progress_callback(current, total, message):
            progress_pct = int((current / total) * 100) if total > 0 else 0
            log_callback("tts_chunk_progress", f"🔊 TTS: {message}")
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'processing',
                'progress': progress_pct,
                'current_chunk': current,
                'total_chunks': total,
                'message': message
            }, namespace='/')

        # Generate TTS
        success, message, audio_path = await generate_tts_for_translation(
            translated_filepath=output_filepath,
            target_language=target_language,
            tts_config=tts_config,
            log_callback=log_callback,
            progress_callback=tts_progress_callback
        )

        if success:
            log_callback("tts_complete", f"✅ TTS audio generated: {os.path.basename(audio_path)}")

            # Store audio file path in state
            state_manager.set_translation_field(translation_id, 'audio_filepath', audio_path)
            state_manager.set_translation_field(translation_id, 'audio_filename', os.path.basename(audio_path))

            # Emit success event
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'completed',
                'progress': 100,
                'audio_filename': os.path.basename(audio_path),
                'message': 'TTS generation completed successfully'
            }, namespace='/')

            # Trigger file list refresh
            socketio.emit('file_list_changed', {
                'reason': 'tts_completed',
                'filename': os.path.basename(audio_path)
            }, namespace='/')

        else:
            log_callback("tts_failed", f"❌ TTS generation failed: {message}")
            socketio.emit('tts_update', {
                'translation_id': translation_id,
                'status': 'failed',
                'error': message,
                'message': f'TTS generation failed: {message}'
            }, namespace='/')

    except Exception as e:
        error_msg = f"TTS generation error: {str(e)}"
        log_callback("tts_error", f"❌ {error_msg}")
        socketio.emit('tts_update', {
            'translation_id': translation_id,
            'status': 'failed',
            'error': error_msg,
            'message': error_msg
        }, namespace='/')


def start_translation_job(translation_id, config, state_manager, output_dir, socketio):
    """
    Start a translation job in a separate thread

    Args:
        translation_id (str): Translation job ID
        config (dict): Translation configuration
        state_manager: State manager instance
        output_dir (str): Output directory path
        socketio: SocketIO instance
    """
    thread = threading.Thread(
        target=run_translation_async_wrapper,
        args=(translation_id, config, state_manager, output_dir, socketio)
    )
    thread.daemon = True
    thread.start()
