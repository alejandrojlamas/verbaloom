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

This module closes that gap with a lightweight polling watchdog: it
periodically checks every in-memory job's ``last_activity_at`` timestamp
(already maintained by the progress-emit code in handlers.py) and, if a
*running* job has gone stale for longer than is ever legitimate for a
single chunk, marks it resumable through the same ``mark_error()`` path
the exception-based recovery code uses -- keeping memory and the SQLite
checkpoint in sync.

Honesty note: if the worker thread is truly deadlocked (not just slow),
Python cannot force-terminate it from another thread. This watchdog makes
the job visible and resumable immediately instead of leaving it silently
"running" forever, but the old thread may still be alive in the
background; a persistent recurrence of the same stall is a signal that a
full server restart is still warranted to actually free it.
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

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
    ):
        self._state_manager = state_manager
        self._socketio = socketio
        self._stale_after_seconds = stale_after_seconds
        self._check_interval_seconds = check_interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

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
            self._flag_stalled(translation_id, age)
            flagged.append(translation_id)
        return flagged

    def _flag_stalled(self, translation_id: str, stale_seconds: float) -> None:
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
