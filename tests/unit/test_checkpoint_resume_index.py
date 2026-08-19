"""Resume-index convention + backward-compat for load_checkpoint.

New checkpoints carry the 'resume_index_semantics' = 'completed' marker, so
current_chunk_index is the last completed unit for every format and resume is
always +1. Pre-migration checkpoints (no marker) must still resume correctly
via the legacy per-format branch (EPUB stored file_idx+1, TXT/SRT stored the
last completed chunk).
"""

import json

import pytest

from src.core.epub.xhtml_translation_state import XHTMLTranslationState
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def cm(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.uploads_dir = tmp_path / "uploads"
    manager.uploads_dir.mkdir()
    return manager


def _set_progress(manager, translation_id, progress):
    conn = manager.db._get_connection()
    conn.execute(
        "UPDATE translation_jobs SET progress = ? WHERE translation_id = ?",
        (json.dumps(progress), translation_id),
    )
    conn.commit()


def _save_xhtml_state(
    manager,
    *,
    translation_id,
    href,
    chunk_count,
    current,
    global_completed,
    total=20,
    failed=0,
):
    chunks = [
        {"text": f"source-{index}", "local_tag_map": {}, "global_indices": []}
        for index in range(chunk_count)
    ]
    state = XHTMLTranslationState(
        file_path=href,
        translation_id=translation_id,
        file_href=href,
        source_language="German",
        target_language="Spanish",
        model_name="test-model",
        max_tokens_per_chunk=1400,
        max_retries=3,
        chunks=chunks,
        global_tag_map={},
        placeholder_format=("[id", "]"),
        translated_chunks=[f"target-{index}" for index in range(current)],
        current_chunk_index=current,
        original_body_html="<body/>",
        doc_metadata={},
        stats={},
        created_at="2026-07-15T00:00:00Z",
        updated_at="2026-07-15T00:00:00Z",
        global_stats={
            "total_chunks": total,
            "completed_chunks": global_completed,
            "failed_chunks": failed,
        },
    )
    assert manager.save_xhtml_partial_state(translation_id, href, state)


@pytest.mark.parametrize("file_type", ["epub", "txt", "srt"])
def test_new_marked_checkpoint_resumes_at_next_unit(cm, file_type):
    cm.start_job("job", file_type, {}, None)
    # New convention: store the last completed unit index.
    for idx in range(3):
        cm.save_checkpoint(
            translation_id="job",
            chunk_index=idx,
            original_text=f"x{idx}",
            translated_text=f"y{idx}",
            total_chunks=10,
            completed_chunks=idx + 1,
            failed_chunks=0,
        )
    assert cm.load_checkpoint("job")["resume_from_index"] == 3


def test_resume_uses_real_chunk_prefix_when_progress_is_stale_low(cm):
    cm.start_job("job", "txt", {"output_filename": "book.txt"}, None)
    for idx in range(5):
        cm.save_checkpoint(
            translation_id="job",
            chunk_index=idx,
            original_text=f"x{idx}",
            translated_text=f"y{idx}",
            total_chunks=5,
            completed_chunks=idx + 1,
            failed_chunks=0,
        )
    _set_progress(cm, "job", {
        "current_chunk_index": 1,
        "total_chunks": 5,
        "completed_chunks": 2,
        "failed_chunks": 0,
        "resume_index_semantics": "completed",
    })
    cm.mark_interrupted("job")

    checkpoint = cm.load_checkpoint("job")

    assert checkpoint["resume_from_index"] == 5
    assert checkpoint["checkpoint_complete"] is True
    assert checkpoint["job"]["status"] == "completed"
    assert cm.get_resumable_jobs() == []


def test_resume_stops_at_first_missing_or_failed_chunk(cm):
    cm.start_job("job", "txt", {}, None)
    cm.save_checkpoint("job", 0, "x0", "y0", total_chunks=4, completed_chunks=1)
    cm.save_checkpoint("job", 1, "x1", "y1", total_chunks=4, completed_chunks=2)
    cm.save_checkpoint("job", 3, "x3", "y3", total_chunks=4, completed_chunks=3)

    assert cm.load_checkpoint("job")["resume_from_index"] == 2


def test_resumable_jobs_skip_unstarted_zero_chunk_checkpoint(cm):
    cm.start_job("job", "txt", {"output_filename": "book.txt"}, None)
    cm.mark_interrupted("job")

    checkpoint = cm.load_checkpoint("job")

    assert checkpoint["resume_from_index"] == 0
    assert checkpoint["checkpoint_complete"] is False
    assert cm.get_resumable_jobs() == []


def test_legacy_epub_checkpoint_without_marker(cm):
    # Pre-migration EPUB: current_chunk_index was file_idx+1 (the next file),
    # and there is no semantics marker.
    cm.start_job("job", "epub", {}, None)
    _set_progress(cm, "job", {
        "current_chunk_index": 3,  # = file_idx(2) + 1 under the old convention
        "total_chunks": 10,
        "completed_chunks": 3,
        "failed_chunks": 0,
    })
    # Legacy branch: EPUB must NOT add +1, so it resumes at file 3.
    assert cm.load_checkpoint("job")["resume_from_index"] == 3


def test_legacy_txt_checkpoint_without_marker(cm):
    cm.start_job("job", "txt", {}, None)
    _set_progress(cm, "job", {
        "current_chunk_index": 5,  # last completed chunk under the old convention
        "total_chunks": 10,
        "completed_chunks": 6,
        "failed_chunks": 0,
    })
    # Legacy branch: TXT adds +1.
    assert cm.load_checkpoint("job")["resume_from_index"] == 6


def test_resumable_native_epub_shows_logical_chunks_but_resumes_by_file(cm):
    cm.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    cm.save_checkpoint(
        "job", 0, "Text/001.xhtml", "Text/001.xhtml",
        chunk_data={
            "file_type": "epub_xhtml",
            "logical_completed_chunks": 2,
            "logical_total_chunks": 20,
        },
        total_chunks=20, completed_chunks=2,
    )
    cm.save_checkpoint(
        "job", 1, "Text/002.xhtml", "Text/002.xhtml",
        chunk_data={
            "file_type": "epub_xhtml",
            "logical_completed_chunks": 5,
            "logical_total_chunks": 20,
        },
        total_chunks=20, completed_chunks=5,
    )
    _save_xhtml_state(
        cm,
        translation_id="job",
        href="Text/003.xhtml",
        chunk_count=10,
        current=4,
        global_completed=9,
        failed=1,
    )
    cm.update_progress("job", completed_chunks=9, failed_chunks=1, status="partial")

    jobs = cm.get_resumable_jobs()

    assert len(jobs) == 1
    assert jobs[0]["resume_from_index"] == 2
    assert jobs[0]["progress"]["resume_from_index"] == 2
    assert jobs[0]["progress"]["completed_chunks"] == 9
    assert jobs[0]["progress"]["failed_chunks"] == 1
    assert jobs[0]["progress_percentage"] == 45


def test_epub_checkpoint_complete_uses_logical_chunk_counts_not_file_count(cm):
    """Real-world regression (Save the Cat Strikes Back, trans_1785043548685):
    a finished EPUB translation -- every file checkpointed, all 169 logical
    chunks completed, zero failures -- was stuck reporting
    checkpoint_complete=False forever, which in turn disabled
    _build_finalization_recovery_plan (it requires checkpoint_complete) and
    left the job unable to auto-recover from a whole-book finalization hiccup.

    Root cause: the EPUB `chunks` table checkpoints at the FILE level
    (chunk_index = XHTML file index, e.g. 0..28 for a 29-file book -- see
    src/core/epub/translator.py's `_save_checkpoint`), while `total_chunks`
    in the progress JSON is the much larger *logical* translation-unit count
    summed across all files (e.g. 169). `resume_from_index` (file count) can
    never reach `total_chunks` (logical count) for any real book with more
    than one logical chunk per file, so checkpoint_complete was structurally
    unable to ever be True for EPUB.
    """
    cm.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    # Only 3 files total, but 169 logical translation chunks across them --
    # exactly the shape that broke the old formula (resume_from_index=3
    # would never reach total_chunks=169).
    for file_idx, logical_completed in enumerate([60, 120, 169]):
        cm.save_checkpoint(
            "job", file_idx, f"Text/{file_idx:03d}.xhtml", f"Text/{file_idx:03d}.xhtml",
            chunk_data={
                "file_type": "epub_xhtml",
                "logical_completed_chunks": logical_completed,
                "logical_total_chunks": 169,
            },
            total_chunks=169, completed_chunks=logical_completed, failed_chunks=0,
        )

    checkpoint = cm.load_checkpoint("job")

    assert checkpoint["resume_from_index"] == 3  # file-level: all 3 files done
    assert checkpoint["job"]["progress"]["completed_chunks"] == 169  # logical: all done
    assert checkpoint["checkpoint_complete"] is True
    assert checkpoint["job"]["status"] == "completed"


def test_epub_checkpoint_not_complete_when_logical_chunks_remain(cm):
    """Sanity check for the fix above: an EPUB job with all checkpointed
    files finished but logical chunks still short of the total (mid-book)
    must NOT be reported as checkpoint_complete."""
    cm.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    cm.save_checkpoint(
        "job", 0, "Text/000.xhtml", "Text/000.xhtml",
        chunk_data={"file_type": "epub_xhtml", "logical_completed_chunks": 60, "logical_total_chunks": 169},
        total_chunks=169, completed_chunks=60, failed_chunks=0,
    )

    checkpoint = cm.load_checkpoint("job")

    assert checkpoint["checkpoint_complete"] is False
    assert checkpoint["job"]["status"] != "completed"


def test_epub_checkpoint_not_complete_when_logical_failures_remain(cm):
    """All logical chunks accounted for, but some marked failed: must not be
    reported complete even though the file-level checkpoint has no failed
    file rows (EPUB failures are tracked at the logical/progress level)."""
    cm.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    cm.save_checkpoint(
        "job", 0, "Text/000.xhtml", "Text/000.xhtml",
        chunk_data={"file_type": "epub_xhtml", "logical_completed_chunks": 165, "logical_total_chunks": 169},
        total_chunks=169, completed_chunks=165, failed_chunks=4,
    )

    checkpoint = cm.load_checkpoint("job")

    assert checkpoint["checkpoint_complete"] is False


def test_resumable_legacy_epub_reconstructs_progress_without_double_counting(cm):
    cm.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    cm.save_checkpoint(
        "job", 0, "Text/001.xhtml", "Text/001.xhtml",
        chunk_data={"file_type": "epub_xhtml"},
        total_chunks=20, completed_chunks=2,
    )
    cm.save_checkpoint(
        "job", 1, "Text/002.xhtml", "Text/002.xhtml",
        chunk_data={"file_type": "epub_xhtml"},
        total_chunks=20, completed_chunks=5,
    )
    _save_xhtml_state(
        cm, translation_id="job", href="Text/001.xhtml",
        chunk_count=2, current=2, global_completed=2,
    )
    _save_xhtml_state(
        cm, translation_id="job", href="Text/002.xhtml",
        chunk_count=3, current=3, global_completed=5,
    )
    _save_xhtml_state(
        cm, translation_id="job", href="Text/003.xhtml",
        chunk_count=10, current=4, global_completed=9, failed=1,
    )
    cm.update_progress("job", completed_chunks=99, failed_chunks=1, status="partial")

    job = cm.get_resumable_jobs()[0]

    assert job["resume_from_index"] == 2
    assert job["progress"]["completed_chunks"] == 9
    assert job["progress_percentage"] == 45
