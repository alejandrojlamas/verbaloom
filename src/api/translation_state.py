"""
Thread-safe translation state management
"""
import atexit
import threading
import time
import copy
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING
from src.persistence.checkpoint_manager import CheckpointManager

if TYPE_CHECKING:
    from src.core.glossary import GlossaryStore


_TRANSFORM_LABEL_TO_MODE = {
    "modernizar": ("modernize", "Modernizar"),
    "explicar": ("simplify", "Explicar"),
    "humanizar": ("humanize", "Humanizar"),
    "adaptar a mexicano": ("mexican_spanish", "Adaptar a mexicano"),
    # Keep recognizing legacy Spanish filenames while exposing the canonical
    # English label to current clients.
    "audiolibro": ("audiobook", "Audiobook"),
}


def _infer_transform_from_filename(filename: str) -> tuple[Optional[str], Optional[str]]:
    """Infer legacy transform metadata from generated filenames.

    Older or in-flight jobs may not have persisted ``text_transform_*`` in
    ``prompt_options`` even though the output filename carries the selected
    transform label, e.g. ``Book (Explicar).epub``. This keeps the UI honest
    without mutating the running job.
    """
    normalized = (filename or "").lower()
    for label, payload in _TRANSFORM_LABEL_TO_MODE.items():
        if f"({label}" in normalized:
            return payload
    return None, None


def _coerce_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_progress_values(data_progress: Any, stats: Dict[str, Any]) -> tuple[float, float]:
    """Return legacy progress and percent values for translation summaries.

    Paused/resumed jobs can have accurate completed/total counters while older
    in-memory ``progress`` fields remain zero or missing. The UI relies on both
    the legacy ``progress`` field and the newer ``progress_percent``/``percent``
    contract, so summaries derive a safe fallback from counters.
    """
    explicit_percent = _coerce_number(stats.get("percent"))
    progress_percent = _coerce_number(stats.get("progress_percent"))
    legacy_progress = _coerce_number(data_progress)

    total = _coerce_number(stats.get("total_chunks")) or 0.0
    completed = _coerce_number(stats.get("completed_chunks")) or 0.0
    failed = _coerce_number(stats.get("failed_chunks")) or 0.0
    derived = None
    if total > 0:
        derived = max(0.0, min(100.0, ((completed + failed) / total) * 100.0))

    percent = explicit_percent
    if percent is None:
        percent = progress_percent
    if percent is None:
        percent = derived
    if percent is None:
        percent = legacy_progress if legacy_progress is not None else 0.0
    percent = max(0.0, min(100.0, percent))

    progress = legacy_progress if legacy_progress is not None else percent
    if derived is not None and (progress <= 0.0 or progress < derived):
        progress = derived
    progress = max(0.0, min(100.0, progress))
    return progress, percent


def generate_server_session_id() -> str:
    """Generate a unique session ID for this server instance using timestamp."""
    import time
    return str(int(time.time()))


class TranslationStateManager:
    """Thread-safe manager for translation state"""

    # A long book can run for hours across thousands of chunks, each
    # producing several log lines (LLM request/response, progress, ...).
    # Every read of this list (get_translation, get_all_translations, the
    # summaries endpoint) does a copy.deepcopy of it under the same RLock
    # the translation thread uses for every progress update, so letting it
    # grow without bound turns into ever-slower polls that compete with the
    # worker thread for the same lock -- a progressive slowdown that can look
    # like the job "froze" on a long-running translation. Only the most
    # recent entries are needed for the live console; older ones are already
    # off-screen client-side and are not the source of truth for progress
    # (that lives in the checkpoint DB) or for translated content.
    MAX_LOGS_PER_TRANSLATION = 2000

    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None, server_session_id: Optional[str] = None):
        self._translations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()  # Use RLock to allow nested locking
        # Generate a unique session ID for this server instance
        self.server_session_id = server_session_id or generate_server_session_id()
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            server_session_id=self.server_session_id
        )
        self.glossary_store: Optional["GlossaryStore"] = None
        self._glossary_lock = threading.Lock()
    
    def create_translation(self, translation_id: str, config: Dict[str, Any]) -> None:
        """Create a new translation entry"""
        prompt_options = config.get('prompt_options') or {}
        text_transform_mode = (
            config.get('text_transform_mode')
            or prompt_options.get('text_transform_mode')
            or ''
        )
        text_transform_label = (
            config.get('text_transform_label')
            or prompt_options.get('text_transform_label')
            or ''
        )
        operation = config.get('operation') or ('transform' if text_transform_mode else 'translate')
        initial_log = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"{'Transformation' if operation == 'transform' else 'Translation'} {translation_id} queued."
        )
        with self._lock:
            self._translations[translation_id] = {
                'status': 'queued',
                'progress': 0,
                'stats': {
                    'start_time': time.time(),
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'failed_chunks': 0,
                    # OpenRouter cost tracking
                    'openrouter_cost': 0.0,
                    'openrouter_prompt_tokens': 0,
                    'openrouter_completion_tokens': 0,
                    # Operation metadata is part of the progress contract so a
                    # transform job does not look like a regular translation
                    # while the unified progress panel lives under the
                    # Translate tab.
                    'operation': operation,
                    'text_transform_mode': text_transform_mode,
                    'text_transform_label': text_transform_label,
                    'output_filename': config.get('output_filename')
                },
                'logs': [initial_log],
                'result': None,
                'config': config,
                'interrupted': False,
                'output_filepath': None
            }
    
    def update_translation(self, translation_id: str, updates: Dict[str, Any]) -> bool:
        """Update translation state safely"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            
            translation = self._translations[translation_id]
            
            # Handle nested updates for stats
            if 'stats' in updates and isinstance(updates['stats'], dict):
                if 'stats' not in translation:
                    translation['stats'] = {}
                translation['stats'].update(updates['stats'])
                updates = {k: v for k, v in updates.items() if k != 'stats'}
            
            # Handle logs append
            if 'log' in updates:
                if 'logs' not in translation:
                    translation['logs'] = []
                translation['logs'].append(updates['log'])
                self._cap_logs(translation)
                updates = {k: v for k, v in updates.items() if k != 'log'}
            
            # Update remaining fields
            translation.update(updates)
            return True
    
    def get_translation(self, translation_id: str) -> Optional[Dict[str, Any]]:
        """Get translation state safely"""
        with self._lock:
            if translation_id not in self._translations:
                return None
            # Return a deep copy to prevent external modification of nested objects
            return copy.deepcopy(self._translations[translation_id])
    
    def get_translation_field(self, translation_id: str, field: str, default=None):
        """Get a specific field from translation state"""
        with self._lock:
            if translation_id not in self._translations:
                return default
            return self._translations[translation_id].get(field, default)
    
    def set_translation_field(self, translation_id: str, field: str, value: Any) -> bool:
        """Set a specific field in translation state"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            self._translations[translation_id][field] = value
            return True
    
    def append_log(self, translation_id: str, log_entry: Any) -> bool:
        """Append a log entry to translation, capped to the most recent
        MAX_LOGS_PER_TRANSLATION entries (see class docstring note)."""
        with self._lock:
            if translation_id not in self._translations:
                return False
            translation = self._translations[translation_id]
            if 'logs' not in translation:
                translation['logs'] = []
            translation['logs'].append(log_entry)
            self._cap_logs(translation)
            return True

    def _cap_logs(self, translation: Dict[str, Any]) -> None:
        """Trim `translation['logs']` in place to the most recent entries.

        Must be called with `self._lock` already held.
        """
        logs = translation.get('logs')
        if logs is not None and len(logs) > self.MAX_LOGS_PER_TRANSLATION:
            del logs[: len(logs) - self.MAX_LOGS_PER_TRANSLATION]
    
    def update_stats(self, translation_id: str, stats_update: Dict[str, Any]) -> bool:
        """Update translation statistics"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            if 'stats' not in self._translations[translation_id]:
                self._translations[translation_id]['stats'] = {}
            self._translations[translation_id]['stats'].update(stats_update)
            return True
    
    def exists(self, translation_id: str) -> bool:
        """Check if translation exists"""
        with self._lock:
            return translation_id in self._translations
    
    def get_all_translations(self) -> Dict[str, Dict[str, Any]]:
        """Get all translations (returns a deep copy)"""
        with self._lock:
            return copy.deepcopy(self._translations)
    
    def get_translation_summaries(self) -> list:
        """Get summaries of all translations for listing"""
        with self._lock:
            summaries = []
            for tid, data in self._translations.items():
                config = data.get('config', {})
                prompt_options = config.get('prompt_options') or {}
                stats = data.get('stats', {})
                inferred_transform_mode, inferred_transform_label = _infer_transform_from_filename(
                    config.get('output_filename') or data.get('output_filename') or ""
                )
                text_transform_mode = (
                    config.get('text_transform_mode')
                    or prompt_options.get('text_transform_mode')
                    or inferred_transform_mode
                )
                text_transform_label = (
                    config.get('text_transform_label')
                    or prompt_options.get('text_transform_label')
                    or inferred_transform_label
                )
                summary_stats = {**stats}
                for restored_key in (
                    'total_chunks',
                    'completed_chunks',
                    'failed_chunks',
                    'progress_percent',
                    'percent',
                    'elapsed_time',
                    'start_time',
                    'current_phase',
                    'phase',
                    'enable_refinement',
                    'refine_only',
                ):
                    restored_value = data.get(restored_key)
                    current_value = summary_stats.get(restored_key)
                    if restored_value is not None and current_value in (None, '', 0, 0.0):
                        summary_stats[restored_key] = restored_value
                start_time = summary_stats.get('start_time')
                if data.get('status') in ('running', 'queued') and start_time:
                    elapsed_time = time.time() - start_time
                else:
                    elapsed_time = summary_stats.get('elapsed_time')
                progress_value, percent_value = _summary_progress_values(
                    data.get('progress'),
                    summary_stats,
                )
                summaries.append({
                    "translation_id": tid,
                    "status": data.get('status'),
                    "progress": progress_value,
                    "start_time": summary_stats.get('start_time'),
                    "elapsed_time": elapsed_time,
                    "output_filename": config.get('output_filename') or data.get('output_filename'),
                    "input_filename": config.get('input_filename') or data.get('input_filename'),
                    "file_type": config.get('file_type') or data.get('file_type', 'txt'),
                    "source_language": config.get('source_language') or data.get('source_language'),
                    "target_language": config.get('target_language') or data.get('target_language'),
                    "operation": "transform" if text_transform_mode else (
                        config.get("operation") or data.get("operation")
                    ),
                    "text_transform_mode": text_transform_mode,
                    "text_transform_label": text_transform_label,
                    "profile_id": prompt_options.get('profile_id') or data.get('profile_id'),
                    "pause_reason": data.get('pause_reason'),
                    "resume_at_utc": data.get('resume_at_utc'),
                    "resume_at_local": data.get('resume_at_local'),
                    # Include stats for UI restoration
                    "total_chunks": summary_stats.get('total_chunks', 0),
                    "completed_chunks": summary_stats.get('completed_chunks', 0),
                    "failed_chunks": summary_stats.get('failed_chunks', 0),
                    "progress_percent": (
                        summary_stats.get('progress_percent')
                        if _coerce_number(summary_stats.get('progress_percent')) is not None
                        else percent_value
                    ),
                    "current_phase": summary_stats.get('current_phase'),
                    "enable_refinement": summary_stats.get('enable_refinement', False),
                    "refine_only": summary_stats.get('refine_only', False),
                    # Canonical progress contract (Step 2 seam), alongside the
                    # legacy fields above for back-compat.
                    "percent": (
                        summary_stats.get('percent')
                        if _coerce_number(summary_stats.get('percent')) is not None
                        else percent_value
                    ),
                    "phase": summary_stats.get('phase'),
                    "live_status": summary_stats.get('live_status'),
                    "live_status_kind": summary_stats.get('live_status_kind'),
                    "live_activity_event": summary_stats.get('live_activity_event'),
                    "last_activity_at": summary_stats.get('last_activity_at'),
                    "failure_recovery_cycle": summary_stats.get('failure_recovery_cycle', 0),
                    "failure_recovery_stuck_count": summary_stats.get('failure_recovery_stuck_count', 0),
                    "elapsed_seconds": summary_stats.get('elapsed_seconds'),
                    "eta_seconds": summary_stats.get('eta_seconds'),
                    "prompt_context": summary_stats.get('prompt_context') or {},
                    "last_translation": data.get('last_translation')
                })
            return sorted(summaries, key=lambda x: x.get('start_time', 0), reverse=True)
    
    def is_interrupted(self, translation_id: str) -> bool:
        """Check if translation is interrupted"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            return self._translations[translation_id].get('interrupted', False)
    
    def set_interrupted(self, translation_id: str, interrupted: bool = True) -> bool:
        """Set interrupted flag for translation"""
        with self._lock:
            if translation_id not in self._translations:
                return False
            self._translations[translation_id]['interrupted'] = interrupted
            return True

    def get_resumable_jobs(self):
        """Get all jobs that can be resumed from database"""
        return self.checkpoint_manager.get_resumable_jobs()

    def restore_job_from_checkpoint(self, translation_id: str) -> bool:
        """
        Restore a job from checkpoint into in-memory state.

        Args:
            translation_id: Job identifier

        Returns:
            True if restored successfully
        """
        checkpoint_data = self.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return False

        job = checkpoint_data['job']
        restored_stats = copy.deepcopy(job['progress'])
        restored_progress, _ = _summary_progress_values(None, restored_stats)
        with self._lock:
            # Restore job into in-memory state
            # Use deepcopy for config to prevent mutation of stored config
            self._translations[translation_id] = {
                'status': 'paused',  # Will be set to 'running' when resumed
                'progress': restored_progress,
                'stats': restored_stats,
                'logs': [f"[{datetime.now().strftime('%H:%M:%S')}] Job restored from checkpoint."],
                'result': None,
                'config': copy.deepcopy(job['config']),
                'interrupted': False,
                'output_filepath': job['config'].get('output_filepath'),
                'resume_from_index': checkpoint_data['resume_from_index']
            }

        return True

    def delete_checkpoint(self, translation_id: str) -> bool:
        """
        Delete a checkpoint for a job.

        Args:
            translation_id: Job identifier

        Returns:
            True if deleted successfully
        """
        # Remove from in-memory state if exists
        with self._lock:
            if translation_id in self._translations:
                del self._translations[translation_id]

        # Delete from database
        return self.checkpoint_manager.delete_checkpoint(translation_id)

    def cleanup_completed_job(self, translation_id: str) -> bool:
        """
        Clean up a completed job (automatic cleanup).

        Args:
            translation_id: Job identifier

        Returns:
            True if cleaned up successfully
        """
        return self.checkpoint_manager.cleanup_completed_job(translation_id)

    def get_checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager instance"""
        return self.checkpoint_manager

    def get_glossary_store(self) -> "GlossaryStore":
        """Return the shared GlossaryStore, instantiating it on first use.

        A single store is shared by the glossary blueprint and the translation
        handler so we don't end up with multiple stores each leaking
        per-thread SQLite connections.
        """
        if self.glossary_store is not None:
            return self.glossary_store
        with self._glossary_lock:
            if self.glossary_store is None:
                from src.core.glossary import GlossaryStore
                self.glossary_store = GlossaryStore()
            return self.glossary_store

    def close_glossary_store(self) -> None:
        """Close every connection held by the shared GlossaryStore, if any."""
        with self._glossary_lock:
            store = self.glossary_store
            self.glossary_store = None
        if store is not None:
            try:
                store.close_all()
            except Exception:
                pass


# Process-wide instance. Keep it lazy so importing this module for tooling,
# static analysis, or tests never opens or migrates the production jobs DB.
_state_manager: Optional[TranslationStateManager] = None
_state_manager_lock = threading.Lock()


def get_state_manager() -> TranslationStateManager:
    """Get the global state manager instance"""
    global _state_manager
    if _state_manager is None:
        with _state_manager_lock:
            if _state_manager is None:
                _state_manager = TranslationStateManager()
    return _state_manager


def get_glossary_store() -> "GlossaryStore":
    """Return the process-wide shared GlossaryStore."""
    return get_state_manager().get_glossary_store()


@atexit.register
def _shutdown_glossary_store() -> None:
    """Close all GlossaryStore connections on interpreter shutdown."""
    manager = _state_manager
    if manager is None:
        return
    try:
        manager.close_glossary_store()
    except Exception:
        pass
