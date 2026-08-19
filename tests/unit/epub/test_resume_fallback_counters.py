"""Regression tests for issue #180: fallback counters reset on resume.

The cross-file `accumulated_stats` in the EPUB pipeline used to be
re-initialized to a fresh `TranslationMetrics()` on every entry into
`_process_all_content_files`, so any token-alignment / Phase-3 fallbacks
that happened in previously translated files were lost when the job was
resumed from checkpoint. These tests cover the snapshot / restore wiring
end-to-end at the persistence layer.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
from lxml import etree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.translator import (
    _process_all_content_files,
    _restore_accumulated_stats,
    _save_checkpoint,
    _snapshot_accumulated_stats,
)
from src.core.epub.xhtml_translator import _resolved_translation_count
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def checkpoint_manager(tmp_path):
    """Isolated CheckpointManager backed by a temporary DB."""
    db_path = str(tmp_path / "jobs.db")
    mgr = CheckpointManager(db_path=db_path)
    yield mgr
    mgr.close()


def _make_populated_metrics() -> TranslationMetrics:
    metrics = TranslationMetrics()
    metrics.token_alignment_used = 7
    metrics.token_alignment_success = 5
    metrics.fallback_used = 3
    metrics.failed_chunks = 1
    metrics.placeholder_errors = 4
    metrics.processed_chunks = 20
    metrics.successful_first_try = 12
    metrics.successful_after_retry = 4
    metrics.retry_attempts = 9
    metrics.quality_warning_fired = True
    metrics.total_tokens_processed = 1500
    metrics.total_tokens_generated = 1400
    metrics.refinement_chunks_completed = 6
    return metrics


def test_snapshot_round_trip_preserves_cross_file_counters():
    metrics = _make_populated_metrics()
    snapshot = _snapshot_accumulated_stats(metrics)

    restored = TranslationMetrics()
    _restore_accumulated_stats(snapshot, restored)

    assert restored.token_alignment_used == 7
    assert restored.token_alignment_success == 5
    assert restored.fallback_used == 3
    assert restored.failed_chunks == 1
    assert restored.placeholder_errors == 4
    assert restored.processed_chunks == 20
    assert restored.successful_first_try == 12
    assert restored.successful_after_retry == 4
    assert restored.retry_attempts == 9
    assert restored.quality_warning_fired is True
    assert restored.total_tokens_processed == 1500
    assert restored.total_tokens_generated == 1400
    assert restored.refinement_chunks_completed == 6


def test_restore_on_empty_snapshot_is_a_noop():
    """No snapshot (legacy checkpoint) must leave the fresh metrics zeroed."""
    metrics = TranslationMetrics()
    _restore_accumulated_stats(None, metrics)
    _restore_accumulated_stats({}, metrics)

    assert metrics.token_alignment_used == 0
    assert metrics.fallback_used == 0
    assert metrics.placeholder_errors == 0


def test_begin_resume_attempt_clears_only_unresolved_failure_count():
    metrics = _make_populated_metrics()

    cleared = metrics.begin_resume_attempt()

    assert cleared == 1
    assert metrics.failed_chunks == 0
    assert metrics.retry_attempts == 9
    assert metrics.token_alignment_used == 7
    assert metrics.fallback_used == 3


def test_resolved_count_includes_alignment_recovered_chunks():
    metrics = TranslationMetrics()
    metrics.successful_first_try = 3
    metrics.token_alignment_success = 2
    metrics.processed_chunks = 5

    assert _resolved_translation_count(["a", "b", "c", "d", "e"], metrics) == 5


def test_metrics_round_trip_restores_processed_progress():
    metrics = _make_populated_metrics()

    restored = TranslationMetrics.from_dict(metrics.to_dict())

    assert restored.processed_chunks == 20


def test_save_checkpoint_persists_accumulated_stats(checkpoint_manager):
    translation_id = "trans_test_180"
    checkpoint_manager.start_job(translation_id, "epub", {"some": "config"})

    snapshot = _snapshot_accumulated_stats(_make_populated_metrics())
    checkpoint_manager.save_checkpoint(
        translation_id=translation_id,
        chunk_index=1,
        original_text="file1.xhtml",
        translated_text="file1.xhtml",
        chunk_data={"last_file": "file1.xhtml", "file_type": "epub_xhtml"},
        total_chunks=100,
        completed_chunks=20,
        failed_chunks=1,
        epub_accumulated_stats=snapshot,
    )

    job = checkpoint_manager.get_job(translation_id)
    persisted = job["progress"].get("epub_accumulated_stats")
    assert persisted is not None
    assert persisted["token_alignment_used"] == 7
    assert persisted["fallback_used"] == 3
    assert persisted["placeholder_errors"] == 4
    assert persisted["processed_chunks"] == 20
    assert persisted["quality_warning_fired"] is True


def test_subsequent_progress_update_keeps_accumulated_stats(checkpoint_manager):
    """save_xhtml_partial_state and similar callers update progress without
    touching epub_accumulated_stats — make sure the snapshot survives."""
    translation_id = "trans_test_180_b"
    checkpoint_manager.start_job(translation_id, "epub", {})

    checkpoint_manager.save_checkpoint(
        translation_id=translation_id,
        chunk_index=1,
        original_text="file1.xhtml",
        translated_text="file1.xhtml",
        total_chunks=100,
        completed_chunks=20,
        failed_chunks=0,
        epub_accumulated_stats=_snapshot_accumulated_stats(_make_populated_metrics()),
    )

    # Simulate save_xhtml_partial_state's progress nudge (no stats arg).
    checkpoint_manager.db.update_job_progress(
        translation_id=translation_id,
        completed_chunks=25,
    )

    job = checkpoint_manager.get_job(translation_id)
    persisted = job["progress"].get("epub_accumulated_stats")
    assert persisted is not None, "Progress update wiped out the fallback snapshot"
    assert persisted["fallback_used"] == 3
    assert job["progress"]["completed_chunks"] == 25


@pytest.mark.asyncio
async def test_file_checkpoint_cleans_matching_partial_and_stores_logical_progress(tmp_path):
    class RecordingCheckpoint:
        def __init__(self):
            self.deleted = []
            self.saved = []

        def save_epub_file(self, **_kwargs):
            return True

        def delete_xhtml_partial_state(self, translation_id, file_href):
            self.deleted.append((translation_id, file_href))
            return True

        def save_checkpoint(self, **kwargs):
            self.saved.append(kwargs)
            return True

    manager = RecordingCheckpoint()
    temp_dir = tmp_path / "book"
    file_path = temp_dir / "OEBPS" / "Text" / "001.xhtml"
    file_path.parent.mkdir(parents=True)
    root = etree.fromstring(
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body><p>Texto</p></body></html>'
    )

    await _save_checkpoint(
        manager,
        "job",
        2,
        "Text/001.xhtml",
        root,
        str(file_path),
        str(temp_dir),
        total_chunks=110,
        completed_chunks=21,
        failed_chunks=0,
    )

    assert manager.deleted == [("job", "Text/001.xhtml")]
    assert manager.saved[0]["chunk_data"] == {
        "last_file": "Text/001.xhtml",
        "file_type": "epub_xhtml",
        "logical_completed_chunks": 21,
        "logical_total_chunks": 110,
    }


@pytest.mark.asyncio
async def test_resume_keeps_original_chunk_denominator(monkeypatch, tmp_path):
    async def unexpected_precount(*_args, **_kwargs):
        raise AssertionError("resume must use the pristine-source chunk plan")

    class ResumeCheckpoint:
        def get_job(self, _translation_id):
            return {
                "progress": {
                    "total_chunks": 91,
                    "completed_chunks": 59,
                    "epub_accumulated_stats": {"processed_chunks": 59},
                }
            }

    monkeypatch.setattr(
        "src.core.epub.translator._precount_chunks",
        unexpected_precount,
    )
    emitted = []

    result = await _process_all_content_files(
        content_files=[f"part_{index}.xhtml" for index in range(8)],
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="German",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        max_attempts=1,
        context_manager=None,
        translation_id="resume-test",
        resume_from_index=5,
        checkpoint_manager=ResumeCheckpoint(),
        stats_callback=emitted.append,
        check_interruption_callback=lambda: True,
        precount_result=(91, [12, 11, 13, 12, 11, 10, 11, 11]),
    )

    assert result["total_chunks"] == 91
    assert result["completed_chunks"] == 59
    assert emitted[0]["total_chunks"] == 91
    assert emitted[0]["completed_chunks"] == 59


@pytest.mark.asyncio
async def test_partial_xhtml_resume_does_not_double_count_its_prefix(monkeypatch, tmp_path):
    async def unexpected_precount(*_args, **_kwargs):
        raise AssertionError("resume must use the supplied pristine chunk plan")

    class PartialResumeCheckpoint:
        def get_job(self, _translation_id):
            # Simulate the already-inflated database value left by the old bug.
            return {"progress": {"total_chunks": 121, "completed_chunks": 59}}

        def load_xhtml_partial_state(self, _translation_id, file_href):
            assert file_href == "main-2.xhtml"
            return SimpleNamespace(
                current_chunk_index=13,
                global_stats={"total_chunks": 121, "completed_chunks": 46},
            )

    monkeypatch.setattr(
        "src.core.epub.translator._precount_chunks",
        unexpected_precount,
    )
    emitted = []

    result = await _process_all_content_files(
        content_files=["main.xhtml", "main-1.xhtml", "main-2.xhtml"],
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=1400,
        max_attempts=1,
        context_manager=None,
        translation_id="partial-resume-test",
        resume_from_index=2,
        checkpoint_manager=PartialResumeCheckpoint(),
        stats_callback=emitted.append,
        check_interruption_callback=lambda: True,
        precount_result=(121, [21, 12, 15]),
    )

    assert emitted[0]["completed_chunks"] == 46
    assert result["completed_chunks"] == 46


@pytest.mark.asyncio
async def test_resume_ignores_stale_progress_when_partial_state_is_missing(monkeypatch, tmp_path):
    async def unexpected_precount(*_args, **_kwargs):
        raise AssertionError("resume must use the supplied pristine chunk plan")

    class ResumeCheckpoint:
        def get_job(self, _translation_id):
            # Eight chunks from the unfinished third file reached the global
            # checkpoint, but its per-XHTML state was invalidated.  Only the
            # two complete files are a valid baseline now.
            return {"progress": {"total_chunks": 48, "completed_chunks": 41}}

        def load_xhtml_partial_state(self, _translation_id, file_href):
            assert file_href == "main-2.xhtml"
            return None

    monkeypatch.setattr(
        "src.core.epub.translator._precount_chunks",
        unexpected_precount,
    )
    emitted = []

    result = await _process_all_content_files(
        content_files=["main.xhtml", "main-1.xhtml", "main-2.xhtml"],
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=1400,
        max_attempts=1,
        context_manager=None,
        translation_id="missing-partial-resume-test",
        resume_from_index=2,
        checkpoint_manager=ResumeCheckpoint(),
        stats_callback=emitted.append,
        check_interruption_callback=lambda: True,
        precount_result=(48, [17, 16, 15]),
    )

    assert emitted[0]["completed_chunks"] == 33
    assert result["completed_chunks"] == 33
