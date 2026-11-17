import errno

from src.persistence.checkpoint_reconcile import checkpoint_progress_snapshot
from src.persistence.checkpoint_manager import CheckpointManager


def test_failed_xhtml_write_preserves_last_checkpoint_bytes(tmp_path, monkeypatch):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    href = "OEBPS/chapter.xhtml"
    assert manager.save_epub_file("job", href, b"<p>Previously saved.</p>")
    chapter = manager.uploads_dir / "job" / "translated_files" / href

    def disk_full(_fd):
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr("src.persistence.checkpoint_manager.os.fsync", disk_full)
    assert not manager.save_epub_file("job", href, b"<p>Replacement.</p>")
    assert chapter.read_bytes() == b"<p>Previously saved.</p>"
    assert list(chapter.parent.iterdir()) == [chapter]


def test_checkpoint_row_and_job_progress_roll_back_together(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    assert manager.start_job("job", "epub", {}, None)
    assert manager.save_checkpoint(
        "job", 0, "first", "saved", completed_chunks=1, total_chunks=2,
        translation_context={"chapter": "first"},
    )
    db = manager.db._get_connection()
    db.execute("""CREATE TRIGGER fail_progress BEFORE UPDATE OF progress ON translation_jobs
                  BEGIN SELECT RAISE(ABORT, 'could not write progress'); END""")
    assert not manager.save_checkpoint(
        "job", 1, "second", "new", completed_chunks=2, total_chunks=2,
        translation_context={"chapter": "second"},
    )
    interrupted = manager.load_checkpoint("job")
    assert [row["translated_text"] for row in interrupted["chunks"]] == ["saved"]
    assert interrupted["job"]["progress"]["completed_chunks"] == 1
    assert interrupted["translation_context"] == {"chapter": "first"}
    assert not interrupted["checkpoint_complete"]

    db.execute("DROP TRIGGER fail_progress")
    assert manager.save_checkpoint(
        "job", 1, "second", "new", completed_chunks=2, total_chunks=2,
        translation_context={"chapter": "second"},
    )
    assert manager.load_checkpoint("job")["checkpoint_complete"]


def test_text_first_epub_counts_missing_and_failed_chunk_rows(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job(
        "job",
        "epub",
        {
            "output_filename": "book.epub",
            "prompt_options": {"text_first_pipeline": True},
        },
        None,
    )
    manager.save_checkpoint("job", 0, "s0", "t0", total_chunks=4, completed_chunks=1)
    manager.save_checkpoint("job", 1, "s1", "t1", total_chunks=4, completed_chunks=2)
    manager.save_checkpoint("job", 2, "s2", None, total_chunks=4, completed_chunks=2, failed_chunks=1)
    manager.save_checkpoint("job", 3, "s3", "t3", total_chunks=4, completed_chunks=3, failed_chunks=1)

    snapshot = checkpoint_progress_snapshot(manager, "job")

    assert snapshot["rows_are_logical_chunks"] is True
    assert snapshot["total_chunks"] == 4
    assert snapshot["completed_chunks"] == 3
    assert snapshot["failed_chunks"] == 1
    assert snapshot["failed_chunk_indices"] == [2]
    assert snapshot["unresolved"] is True


def test_native_epub_file_checkpoints_do_not_expand_missing_text_chunks(tmp_path):
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    manager.start_job(
        "job",
        "epub",
        {"output_filename": "book.epub", "prompt_options": {}},
        None,
    )
    manager.save_checkpoint(
        "job",
        0,
        "chapter1.xhtml",
        "chapter1.xhtml",
        chunk_data={"file_type": "epub_xhtml"},
        total_chunks=20,
        completed_chunks=8,
        failed_chunks=0,
    )

    snapshot = checkpoint_progress_snapshot(manager, "job")

    assert snapshot["rows_are_logical_chunks"] is False
    assert snapshot["total_chunks"] == 20
    assert snapshot["failed_chunks"] == 0
    assert snapshot["unresolved"] is False
