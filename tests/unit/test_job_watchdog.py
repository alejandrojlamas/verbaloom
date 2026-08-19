"""
Tests for JobWatchdog: detecting a translation worker thread that has
silently stopped making progress.

Every existing auto-recovery mechanism in src/api/handlers.py only reacts to
*exceptions*. A worker thread that blocks forever without ever raising (a
genuine hang, e.g. the historical catastrophic-regex freeze already fixed
once in src/common/inline_markdown.py, or any future regression with the
same shape) is invisible to all of it -- nothing marks the job as failed,
so it sits at status='running' in both memory and SQLite until the user
notices and restarts the whole server. JobWatchdog closes that gap by
periodically checking for stale `last_activity_at` timestamps on running
jobs and marking them resumable through the same mark_error() path used
by the exception-based recovery code.
"""
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from src.api.job_watchdog import JobWatchdog
from src.api.translation_state import TranslationStateManager
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def wired_state_manager():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_jobs.db"
    checkpoint_manager = CheckpointManager(db_path=str(db_path))
    checkpoint_manager.uploads_dir = Path(temp_dir) / "uploads"
    checkpoint_manager.uploads_dir.mkdir(parents=True, exist_ok=True)

    state_manager = TranslationStateManager(checkpoint_manager=checkpoint_manager)

    yield state_manager

    checkpoint_manager.close()
    shutil.rmtree(temp_dir, ignore_errors=True)


def _create_running_job(state_manager, translation_id, *, last_activity_at=None):
    config = {"input_filename": "book.txt", "output_filename": "book (Spanish).txt"}
    state_manager.create_translation(translation_id, config)
    state_manager.get_checkpoint_manager().start_job(translation_id, "txt", config)
    state_manager.set_translation_field(translation_id, "status", "running")
    stats = dict(state_manager.get_translation_field(translation_id, "stats") or {})
    if last_activity_at is not None:
        stats["last_activity_at"] = last_activity_at
    state_manager.update_stats(translation_id, stats)


def test_stale_running_job_is_flagged_and_synced_to_db(wired_state_manager):
    state_manager = wired_state_manager
    checkpoint_manager = state_manager.get_checkpoint_manager()
    translation_id = "trans_stale"
    _create_running_job(
        state_manager, translation_id, last_activity_at=time.time() - 10_000
    )

    watchdog = JobWatchdog(state_manager, stale_after_seconds=3600)
    flagged = watchdog.check_once()

    assert flagged == [translation_id]
    assert state_manager.get_translation_field(translation_id, "status") == "error"
    job_row = checkpoint_manager.db.get_job(translation_id)
    assert job_row["status"] == "error"
    resumable_ids = {job["translation_id"] for job in checkpoint_manager.get_resumable_jobs()}
    assert translation_id in resumable_ids


def test_recent_activity_job_is_not_flagged(wired_state_manager):
    state_manager = wired_state_manager
    translation_id = "trans_healthy"
    _create_running_job(
        state_manager, translation_id, last_activity_at=time.time() - 5
    )

    watchdog = JobWatchdog(state_manager, stale_after_seconds=3600)
    flagged = watchdog.check_once()

    assert flagged == []
    assert state_manager.get_translation_field(translation_id, "status") == "running"


def test_job_with_no_activity_timestamp_yet_is_not_flagged(wired_state_manager):
    """Right after start/resume, before the first chunk completes, there is
    no last_activity_at yet. Falling back to a stale historical start_time
    (e.g. a job resumed hours after it was first created) would false-
    positive on every resume; skipping is the safe choice."""
    state_manager = wired_state_manager
    translation_id = "trans_just_started"
    _create_running_job(state_manager, translation_id, last_activity_at=None)

    watchdog = JobWatchdog(state_manager, stale_after_seconds=3600)
    flagged = watchdog.check_once()

    assert flagged == []
    assert state_manager.get_translation_field(translation_id, "status") == "running"


def test_non_running_job_is_never_flagged(wired_state_manager):
    state_manager = wired_state_manager
    translation_id = "trans_paused"
    _create_running_job(
        state_manager, translation_id, last_activity_at=time.time() - 10_000
    )
    state_manager.set_translation_field(translation_id, "status", "paused")

    watchdog = JobWatchdog(state_manager, stale_after_seconds=3600)
    flagged = watchdog.check_once()

    assert flagged == []
    assert state_manager.get_translation_field(translation_id, "status") == "paused"


def test_start_stop_lifecycle_does_not_raise(wired_state_manager):
    watchdog = JobWatchdog(
        wired_state_manager, stale_after_seconds=3600, check_interval_seconds=0.05
    )
    watchdog.start()
    time.sleep(0.15)
    watchdog.stop()
    watchdog.join(timeout=2)
    assert not watchdog.is_alive()
