"""
Regression tests for the memory<->SQLite desync when a translation job dies
with an uncaught exception.

Before this fix, a job that failed before/outside the main protected
try-block in `perform_actual_translation` only had its status updated in the
in-memory TranslationStateManager. The SQLite `translation_jobs` row (the
source of truth for `/api/resumable`) stayed at `status='running'` forever,
so the job looked alive in the database while its worker thread was
actually dead -- invisible to the resume UI until a full server restart
(`reset_running_jobs_on_startup`) forced it back to `interrupted`.
"""
import shutil
import tempfile
from pathlib import Path

import pytest

from src.api import handlers
from src.api.translation_state import TranslationStateManager
from src.persistence.checkpoint_manager import CheckpointManager


class _FakeSocketIO:
    def emit(self, *args, **kwargs):
        pass


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


def test_uncaught_wrapper_exception_marks_job_resumable_in_db(wired_state_manager, monkeypatch):
    """A translation that blows up before the protected try-block must still
    end up queryable via get_resumable_jobs(), not just marked 'error' in
    the in-memory state manager."""
    state_manager = wired_state_manager
    checkpoint_manager = state_manager.get_checkpoint_manager()
    translation_id = "trans_wrapper_desync_test"
    config = {"input_filename": "book.txt", "output_filename": "book (Spanish).txt"}

    # Establish both sources of truth exactly like a real job start does.
    state_manager.create_translation(translation_id, config)
    assert checkpoint_manager.start_job(translation_id, "txt", config) is True

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated failure before the protected try-block")

    monkeypatch.setattr(handlers, "perform_actual_translation", _boom)

    handlers.run_translation_async_wrapper(
        translation_id, config, state_manager, "/tmp/does-not-matter", _FakeSocketIO()
    )

    # Memory side (already worked before the fix).
    assert state_manager.get_translation_field(translation_id, "status") == "error"

    # Database side: this is the part that used to stay 'running' forever.
    resumable_ids = {job["translation_id"] for job in checkpoint_manager.get_resumable_jobs()}
    assert translation_id in resumable_ids
    job_row = checkpoint_manager.db.get_job(translation_id)
    assert job_row["status"] == "error"
