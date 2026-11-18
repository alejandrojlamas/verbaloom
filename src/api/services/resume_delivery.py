"""Find the published delivery that supersedes a recoverable job."""

from pathlib import Path

from src.api.services.path_validator import PathValidator
from src.core.quality_assurance.publication import artifact_sha256


_DELIVERY_FIELDS = (
    "file_type", "output_format", "source_language", "target_language",
    "operation", "text_transform_mode", "refine_only", "refine_after",
    "bilingual_output", "model", "llm_provider", "llm_api_endpoint",
    "pipeline", "tts_enabled", "tts_config",
)


def _same_delivery(left, right):
    if any(left.get(key) != right.get(key) for key in _DELIVERY_FIELDS):
        return False

    def prompt_policy(config):
        return {
            key: value for key, value in (config.get("prompt_options") or {}).items()
            if not key.startswith("_") and value is not None and value != ""
        }

    return prompt_policy(left) == prompt_policy(right)


def _managed_source(manager, output_root, job_id, config):
    job_id_valid, _ = PathValidator.validate_filename(job_id)
    if not job_id_valid:
        return None
    roots = [output_root / "uploads"]
    if getattr(manager, "uploads_dir", None):
        roots.append(Path(manager.uploads_dir) / job_id)
    try:
        return PathValidator.resolve_managed_file(config.get("preserved_input_path"), roots)
    except (OSError, TypeError, ValueError):
        return None


def _same_source(manager, output_root, job_id, old, later_id, new):
    # Pasted text has no upload and is persisted directly in the job config.
    if old.get("file_type") == "txt" and old.get("text") and not old.get("preserved_input_path"):
        return not new.get("preserved_input_path") and old["text"] == new.get("text")
    left = _managed_source(manager, output_root, job_id, old)
    right = _managed_source(manager, output_root, later_id, new)
    if left is None or right is None:
        return False
    try:
        if left.stat().st_size != right.stat().st_size:
            return False
        digest = artifact_sha256(left)
        return bool(digest) and digest == artifact_sha256(right)
    except OSError:
        return False


def find_superseding_delivery(manager, output_root, job):
    """Return safe metadata only when both jobs have the same source and work."""
    find_candidates = getattr(manager, "find_later_completed_jobs", None)
    if not callable(find_candidates):
        return None
    config = job.get("config") or {}
    requested = (
        config.get("_requested_output_filename")
        or config.get("_job_output_filename")
        or config.get("output_filename")
        or ""
    ).removeprefix("[partial] ")
    if not PathValidator.validate_filename(requested)[0]:
        return None
    job_id = str(job.get("translation_id") or "")
    for later in find_candidates(job_id, requested):
        later_config = later.get("config") or {}
        name = later_config.get("output_filename") or ""
        if not PathValidator.validate_filename(name)[0] or name.startswith("[partial] "):
            continue
        path = output_root / name
        if path.is_symlink() or not path.is_file() or not _same_delivery(config, later_config):
            continue
        if _same_source(manager, output_root, job_id, config, later["translation_id"], later_config):
            return {
                "translation_id": later["translation_id"],
                "output_filename": name,
                "completed_at": later.get("completed_at"),
            }
    return None
