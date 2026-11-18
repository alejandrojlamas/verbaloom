from src.persistence.checkpoint_manager import CheckpointManager


def test_find_later_completed_jobs_filters_by_state_age_and_exact_output(tmp_path):
    database = str(tmp_path / "jobs.db")
    uploads = tmp_path / "uploads"
    filename = "Título 100%_A.epub"
    manager = CheckpointManager(database, uploads_dir=uploads)
    fixtures = [
        ("older", "completed", {"output_filename": filename}, "00", "01"),
        ("same-second-older", "completed", {"output_filename": filename}, "02", "02"),
        ("original", "interrupted", {"_job_output_filename": filename}, "02", None),
        ("same-second-newer", "completed", {"output_filename": filename}, "02", "03"),
        ("legacy", "completed", {"output_filename": filename}, "03", "04"),
        ("stable-name", "completed", {"_job_output_filename": filename}, "04", "06"),
        ("not-published", "validating", {"output_filename": filename}, "04", None),
        ("different-name", "completed", {"output_filename": "Título 100AA.epub"}, "04", "07"),
        ("collision-suffix", "completed", {"output_filename": "Título 100%_A (1).epub"}, "05", "08"),
        ("numbered-publication", "completed", {
            "_requested_output_filename": filename,
            "_job_output_filename": "Título 100%_A (2).epub",
            "output_filename": "Título 100%_A (2).epub",
        }, "05", "09"),
        ("requested-with-suffix", "completed", {
            "_requested_output_filename": "Título 100%_A (3).epub",
            "output_filename": filename,
        }, "06", "10"),
    ]
    try:
        connection = manager.db._get_connection()
        for job_id, status, config, created, completed in fixtures:
            assert manager.start_job(job_id, "epub", config, status=status)
            connection.execute(
                "UPDATE translation_jobs SET created_at = ?, completed_at = ? WHERE translation_id = ?",
                (
                    f"2026-11-18 03:00:{created}",
                    f"2026-11-18 03:00:{completed}" if completed else None,
                    job_id,
                ),
            )
        connection.commit()
    finally:
        manager.close()

    restored = CheckpointManager(database, uploads_dir=uploads)
    try:
        rows = restored.find_later_completed_jobs("original", filename)
        assert [row["translation_id"] for row in rows] == [
            "numbered-publication", "stable-name", "legacy", "same-second-newer",
        ]
        assert rows[0]["config"]["output_filename"] == "Título 100%_A (2).epub"
        assert rows[1]["config"]["_job_output_filename"] == filename
        assert rows[1]["created_at"] == "2026-11-18 03:00:04"
        assert rows[1]["completed_at"] == "2026-11-18 03:00:06"
        assert restored.find_later_completed_jobs("missing", filename) == []
        assert restored.find_later_completed_jobs("original", "") == []
    finally:
        restored.close()
