"""
Detects translation jobs whose worker thread has silently stopped making
progress.

Every auto-recovery path in ``src/api/handlers.py`` (worker-exception
recovery, rate-limit auto-pause, failed-chunk recovery) only reacts to
*exceptions*. A worker thread that blocks forever without ever raising --
a genuine hang, the same shape as the catastrophic-regex freeze already
fixed once in ``src/common/inline_markdown.py`` (see git history: "Prevent
preview and JSON parsing freezes"), or any future regression with the same
shape -- is invisible to all of it: nothing marks the job as failed, so it
sits at ``status='running'`` in both memory and SQLite until a human
notices and restarts the whole server.

This module closes that gap with a lightweight polling watchdog.  In a
managed installation it requests a process recycle while the durable job row
remains active; the service manager then starts a clean process and startup
recovery resumes the checkpoint.  Recycling the process is intentional:
Python cannot safely terminate one deadlocked thread, and starting a second
writer beside it can corrupt or regress progress.

Unmanaged embedders retain the conservative fallback: the stale job is marked
as an explicit resumable error rather than silently left "running" forever.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, List, Optional

from src.api.translation_state import TranslationStateManager
from src.config import JOB_WATCHDOG_CHECK_INTERVAL_SECONDS, JOB_WATCHDOG_STALE_SECONDS

# Only these statuses represent a worker thread that is expected to be
# actively making progress right now. 'queued'/'paused'/'rate_limited'/
# terminal states all have their own lifecycle and are not this
# watchdog's concern.
_ACTIVE_STATUS = "running"


class JobWatchdog:
    """Polls in-memory jobs for a stale ``last_activity_at`` and flags them."""

    def __init__(
        self,
        state_manager: TranslationStateManager,
        socketio=None,
        stale_after_seconds: float = JOB_WATCHDOG_STALE_SECONDS,
        check_interval_seconds: float = JOB_WATCHDOG_CHECK_INTERVAL_SECONDS,
        restart_callback: Optional[Callable[[str, str], None]] = None,
    ):
        self._state_manager = state_manager
        self._socketio = socketio
        self._stale_after_seconds = stale_after_seconds
        self._check_interval_seconds = check_interval_seconds
        self._restart_callback = restart_callback
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recovery_requested: set[str] = set()
        self._managed_restart_requested = False

    def start(self) -> None:
        """Start the background polling thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="job-watchdog"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop (does not block)."""
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        while not self._stop_event.wait(self._check_interval_seconds):
            try:
                self.check_once()
            except Exception as exc:  # pragma: no cover - defensive only
                print(f"⚠️ JobWatchdog check failed: {exc}")

    def check_once(self) -> List[str]:
        """Scan all in-memory jobs once; flag stale 'running' jobs.

        Returns the translation_ids that were flagged in this pass.
        """
        flagged: List[str] = []
        now = time.time()
        for translation_id, data in self._state_manager.get_all_translations().items():
            # One process recycle reconciles every durable row. Duplicate kill
            # requests only make ownership on startup ambiguous.
            if self._restart_callback is not None and self._managed_restart_requested:
                break
            if data.get("status") != _ACTIVE_STATUS:
                continue
            stats = data.get("stats") or {}
            last_activity_at = stats.get("last_activity_at")
            # No timestamp yet (job just started/resumed, before its first
            # chunk completed): nothing to compare against. Falling back to
            # a historical `start_time` would false-positive on every
            # resume of an old job, so we skip instead.
            if last_activity_at is None:
                continue
            try:
                age = now - float(last_activity_at)
            except (TypeError, ValueError):
                continue
            if age <= self._stale_after_seconds:
                continue
            if translation_id in self._recovery_requested:
                continue
            self._flag_stalled(translation_id, age)
            flagged.append(translation_id)
        return flagged

    def _flag_stalled(self, translation_id: str, stale_seconds: float) -> None:
        if self._restart_callback is not None:
            self._request_managed_recovery(translation_id, stale_seconds)
            return

        message = (
            f"⏱️ No progress detected for job {translation_id} in "
            f"{int(stale_seconds)}s (threshold {int(self._stale_after_seconds)}s). "
            "The worker thread may be stuck; marking the job as recoverable "
            "so it can be resumed from its last checkpoint. If this keeps "
            "happening on the same book, a server restart may still be "
            "needed to fully release the stuck thread."
        )
        print(f"⚠️ JobWatchdog: {message}")

        if not self._state_manager.exists(translation_id):
            return

        self._state_manager.set_translation_field(translation_id, "status", "error")
        self._state_manager.set_translation_field(translation_id, "error", message)
        self._state_manager.append_log(translation_id, f"[watchdog] {message}")

        try:
            self._state_manager.get_checkpoint_manager().mark_error(translation_id)
        except Exception:
            pass

        if self._socketio is not None:
            try:
                from src.api.websocket import emit_update

                emit_update(
                    self._socketio,
                    translation_id,
                    {"status": "error", "error": message, "log": message},
                    self._state_manager,
                )
            except Exception:
                pass

    def _request_managed_recovery(
        self,
        translation_id: str,
        stale_seconds: float,
    ) -> None:
        message = (
            f"No hubo actividad durante {int(stale_seconds)}s. El motor se "
            "reiniciará una vez y continuará desde el último checkpoint; "
            "no se iniciará un segundo worker en paralelo."
        )
        print(f"⚠️ JobWatchdog: {translation_id}: {message}")
        if not self._state_manager.exists(translation_id):
            return

        self._managed_restart_requested = True
        self._recovery_requested.add(translation_id)
        stats = dict(
            self._state_manager.get_translation_field(translation_id, "stats")
            or {}
        )
        stats.update({
            "live_status": "Reiniciando el motor y recuperando el checkpoint",
            "live_status_kind": "active",
            "live_activity_event": "watchdog_process_recovery",
            "last_activity_at": time.time(),
        })
        self._state_manager.update_stats(translation_id, stats)
        self._state_manager.set_translation_field(
            translation_id, "watchdog_recovery_requested", True
        )
        self._state_manager.set_translation_field(translation_id, "error", None)
        self._state_manager.append_log(
            translation_id,
            f"[watchdog] {message}",
        )
        try:
            # Keep the durable status active.  On restart the new server session
            # recognizes this row as stale and resumes it automatically.
            self._state_manager.get_checkpoint_manager().mark_running(translation_id)
        except Exception:
            pass

        if self._socketio is not None:
            try:
                from src.api.websocket import emit_update

                emit_update(
                    self._socketio,
                    translation_id,
                    {
                        "status": "running",
                        "recovering": True,
                        "recovery_scope": "process",
                        "stats": stats,
                        "log": message,
                    },
                    self._state_manager,
                )
            except Exception:
                pass

        try:
            self._restart_callback(translation_id, message)
        except Exception as exc:
            # A failed managed restart must not leave a permanently active row.
            self._managed_restart_requested = False
            self._recovery_requested.discard(translation_id)
            fallback = f"{message} No se pudo reiniciar el servicio: {exc}"
            self._state_manager.set_translation_field(
                translation_id, "status", "error"
            )
            self._state_manager.set_translation_field(
                translation_id, "error", fallback
            )
            try:
                self._state_manager.get_checkpoint_manager().mark_error(
                    translation_id
                )
            except Exception:
                pass
