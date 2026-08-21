import inspect

import pytest

from src.api.handlers import (
    _apply_active_timing,
    _begin_active_run_stats,
    _build_failed_chunk_recovery_plan,
    _build_finalization_recovery_plan,
    _build_rate_limit_auto_resume_plan,
    _build_worker_exception_recovery_plan,
    _canonicalize_published_epub_stats,
    _failed_chunk_recovery_exhausted_stats,
    _final_source_sample_diagnostics,
    _finalization_recovery_exhausted_stats,
    _job_has_unresolved_work,
    _job_is_ready_for_final_audits,
    _live_activity_label,
    _resolve_job_output_path,
    _strip_legacy_automatic_recovery_config,
    perform_actual_translation,
)


class _PublicationReport:
    def __init__(self, publishable):
        self.publishable = publishable


class _SampleIssue:
    def __init__(self, code):
        self.code = code


class _SampleReport:
    clean = False
    warning_count = 1
    error_count = 1
    issues = [_SampleIssue("source_language_residual_coverage")]


def test_published_epub_stats_use_native_processed_chunk_truth():
    stats = {
        "total_chunks": 160,
        "completed_chunks": 11,
        "failed_chunks": 3475,
        "checkpoint_failed_chunks": 3475,
        "quality_degraded": True,
        "epub_accumulated_stats": {"processed_chunks": 121, "failed_chunks": 0},
    }

    result = _canonicalize_published_epub_stats(stats, _PublicationReport(True))

    assert result["total_chunks"] == 121
    assert result["completed_chunks"] == 121
    assert result["failed_chunks"] == 0
    assert result["checkpoint_failed_chunks"] == 0
    assert result["units_done"] == 121
    assert result["units_failed"] == 0
    assert result["percent"] == 100.0
    assert result["quality_degraded"] is False


def test_failed_publication_does_not_rewrite_epub_stats():
    stats = {"total_chunks": 160, "failed_chunks": 4}

    assert _canonicalize_published_epub_stats(stats, _PublicationReport(False)) == stats


def test_native_adapter_failure_stays_partial_when_checkpoint_reports_zero_failures():
    assert _job_has_unresolved_work(
        operation_success=False,
        current_status="processing",
        failed_chunks=0,
    )


def test_existing_partial_status_survives_final_publication():
    assert _job_has_unresolved_work(
        operation_success=True,
        current_status="partial",
        failed_chunks=0,
    )


def test_successful_clean_job_can_complete():
    assert not _job_has_unresolved_work(
        operation_success=True,
        current_status="processing",
        failed_chunks=0,
    )


def test_partial_job_does_not_run_whole_book_audits():
    assert not _job_is_ready_for_final_audits(
        operation_success=False,
        current_status="partial",
        failed_chunks=1,
    )


def test_clean_job_runs_whole_book_audits():
    assert _job_is_ready_for_final_audits(
        operation_success=True,
        current_status="processing",
        failed_chunks=0,
    )


def test_source_sample_diagnostics_do_not_turn_completed_chunks_into_failures():
    stats = {
        "total_chunks": 214,
        "completed_chunks": 214,
        "failed_chunks": 0,
        "checkpoint_failed_chunks": 0,
    }

    result = _final_source_sample_diagnostics(stats, _SampleReport())

    assert result["completed_chunks"] == 214
    assert result["failed_chunks"] == 0
    assert result["checkpoint_failed_chunks"] == 0
    assert result["final_source_sample_error_count"] == 1
    assert result["final_source_sample_issue_codes"] == [
        "source_language_residual_coverage"
    ]


def test_resume_reuses_stable_job_output_path(tmp_path):
    existing = tmp_path / "book.epub"
    existing.write_bytes(b"partial")
    config = {
        "output_filename": "book (1).epub",
        "_job_output_filename": "book.epub",
    }

    resolved = _resolve_job_output_path(config, str(tmp_path), is_resume=True)

    assert resolved == str(existing)
    assert config["output_filename"] == "book.epub"


def test_new_job_still_avoids_unrelated_output(tmp_path):
    (tmp_path / "book.epub").write_bytes(b"existing")
    config = {"output_filename": "book.epub"}

    resolved = _resolve_job_output_path(config, str(tmp_path), is_resume=False)

    assert resolved == str(tmp_path / "book (1).epub")
    assert config["_job_output_filename"] == "book (1).epub"


def test_job_output_path_rejects_executable_extension(tmp_path):
    config = {"output_filename": "payload.cmd"}

    with pytest.raises(ValueError, match="unsupported output extension"):
        _resolve_job_output_path(config, str(tmp_path), is_resume=False)


def test_failed_chunk_builds_automatic_checkpoint_recovery_plan():
    checkpoint = {
        "resume_from_index": 7,
        "failed_chunk_indices": [7],
        "job": {"progress": {"failed_chunks": 1}},
    }

    plan = _build_failed_chunk_recovery_plan(
        {"prompt_options": {"custom_instructions": "Keep the author's voice."}},
        checkpoint,
        {"completed_chunks": 23},
    )

    assert plan is not None
    assert plan["resume_index"] == 7
    assert plan["cycle"] == 1
    assert plan["delay_seconds"] == 2
    assert plan["config"]["is_resume"] is True
    instructions = plan["config"]["prompt_options"]["custom_instructions"]
    assert "Keep the author's voice." in instructions
    assert "AUTOMATIC FAILED-CHUNK RECOVERY" not in instructions
    assert "automatic_failure_recovery_cycle" not in plan["config"]["prompt_options"]


def test_repeated_failed_chunk_recovery_is_bounded_without_prompt_mutation():
    checkpoint = {
        "resume_from_index": 7,
        "failed_chunk_indices": [7],
        "job": {"progress": {"failed_chunks": 1}},
    }
    first = _build_failed_chunk_recovery_plan({}, checkpoint, {"completed_chunks": 23})
    second = _build_failed_chunk_recovery_plan(
        first["config"], checkpoint, {"completed_chunks": 23}
    )
    exhausted = _build_failed_chunk_recovery_plan(
        second["config"], checkpoint, {"completed_chunks": 23}
    )

    assert second["stuck_count"] == 2
    assert second["delay_seconds"] == 10
    assert exhausted is None
    assert "AUTOMATIC FAILED-CHUNK RECOVERY" not in str(second["config"])


def test_exhausted_recovery_does_not_leave_dashboard_marked_active():
    stats = _failed_chunk_recovery_exhausted_stats(
        {
            "failed_chunks": 1,
            "live_status": "Avance guardado; continuando",
            "live_status_kind": "active",
        },
        failed_chunks=1,
        now=123.0,
    )

    assert stats["live_status_kind"] == "recoverable"
    assert stats["live_activity_event"] == "failed_chunk_recovery_exhausted"
    assert "requieren reparación" in stats["live_status"]
    assert stats["last_activity_at"] == 123.0


def test_legacy_recovery_prompt_is_removed_and_budget_is_reset_once():
    config = {
        "_failure_recovery_cycle": 58,
        "_failure_recovery_stuck_count": 2,
        "prompt_options": {
            "automatic_failure_recovery": True,
            "automatic_failure_recovery_cycle": 58,
            "custom_instructions": (
                "Keep the voice.\n\nAUTOMATIC FAILED-CHUNK RECOVERY\n"
                "The previous attempt did not produce a candidate that passed the local quality\n"
                "contract. Translate the complete source unit into the requested target language.\n"
                "Preserve every fact, name, number, paragraph boundary and structural placeholder.\n"
                "Do not return source-language prose, a summary, an excerpt, or an explanation."
            ),
        },
    }

    cleaned, changed = _strip_legacy_automatic_recovery_config(config)

    assert changed is True
    assert cleaned["prompt_options"]["custom_instructions"] == "Keep the voice."
    assert "automatic_failure_recovery" not in cleaned["prompt_options"]
    assert "_failure_recovery_cycle" not in cleaned


def test_legacy_migration_notice_runs_after_live_activity_publisher_exists():
    source = inspect.getsource(perform_actual_translation)

    assert source.index("def _publish_live_activity") < source.index(
        "if legacy_recovery_config_removed:"
    )


def test_active_timing_excludes_paused_wall_time_and_estimates_remaining_work():
    started = _begin_active_run_stats(
        {"completed_chunks": 40, "total_chunks": 100},
        now=1_000.0,
    )
    current = {**started, "completed_chunks": 50}

    timed = _apply_active_timing(current, now=1_100.0)

    assert timed["elapsed_seconds"] == 100.0
    assert timed["eta_seconds"] == 500.0

    resumed = _begin_active_run_stats(timed, now=9_000.0)
    resumed["completed_chunks"] = 55
    timed_again = _apply_active_timing(resumed, now=9_050.0)
    assert timed_again["elapsed_seconds"] == 150.0
    assert timed_again["eta_seconds"] == 450.0


def test_non_chunk_publication_failure_does_not_loop_translation_worker():
    checkpoint = {
        "resume_from_index": 10,
        "failed_chunk_indices": [],
        "job": {"progress": {"failed_chunks": 0}},
    }

    assert _build_failed_chunk_recovery_plan({}, checkpoint, {"completed_chunks": 10}) is None


def test_complete_checkpoint_retries_only_finalization():
    checkpoint = {
        "checkpoint_complete": True,
        "resume_from_index": 10,
        "failed_chunk_indices": [],
        "job": {"progress": {"total_chunks": 10, "failed_chunks": 0}},
    }

    plan = _build_finalization_recovery_plan({}, checkpoint)

    assert plan is not None
    assert plan["scope"] == "finalization"
    assert plan["resume_index"] == 10
    assert plan["delay_seconds"] == 2
    assert plan["config"]["is_resume"] is True
    assert plan["config"]["resume_from_index"] == 10


def test_finalization_recovery_is_bounded_and_never_handles_failed_chunks():
    complete = {
        "checkpoint_complete": True,
        "resume_from_index": 10,
        "failed_chunk_indices": [],
        "job": {"progress": {"total_chunks": 10, "failed_chunks": 0}},
    }
    first = _build_finalization_recovery_plan({}, complete)
    second = _build_finalization_recovery_plan(first["config"], complete)
    exhausted = _build_finalization_recovery_plan(second["config"], complete)

    assert second["cycle"] == 2
    assert second["delay_seconds"] == 10
    assert exhausted is None

    failed_chunk_checkpoint = {
        **complete,
        "failed_chunk_indices": [7],
        "job": {"progress": {"total_chunks": 10, "failed_chunks": 1}},
    }
    assert _build_finalization_recovery_plan({}, failed_chunk_checkpoint) is None


def test_manual_pause_disables_automatic_finalization_recovery():
    checkpoint = {
        "checkpoint_complete": True,
        "resume_from_index": 10,
        "failed_chunk_indices": [],
        "job": {"progress": {"total_chunks": 10, "failed_chunks": 0}},
    }

    assert _build_finalization_recovery_plan(
        {"_manual_pause_requested": True},
        checkpoint,
    ) is None


def test_real_epub_checkpoint_now_offers_finalization_recovery(tmp_path):
    """Integration regression for the real incident (Save the Cat Strikes
    Back, trans_1785043548685): a fully-translated EPUB -- every file
    checkpointed, all logical chunks completed, zero failures -- must feed
    a checkpoint_complete=True checkpoint into
    _build_finalization_recovery_plan(), not just in a hand-built dict but
    through the real persistence layer that produced the incident.

    Before the checkpoint_manager.load_checkpoint() unit-mismatch fix, this
    would have loaded checkpoint_complete=False here too, and
    _build_finalization_recovery_plan would have returned None -- exactly
    the dead end that left the real job stuck at 'partial' with no
    automatic path forward.
    """
    from src.persistence.checkpoint_manager import CheckpointManager

    cm = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    cm.uploads_dir = tmp_path / "uploads"
    cm.uploads_dir.mkdir()
    cm.start_job(
        "book_job", "epub",
        {"output_filename": "book.epub", "file_path": "book.epub"},
        None,
    )
    # 3 files, 169 logical chunks total -- same shape as the real incident.
    for file_idx, logical_completed in enumerate([60, 120, 169]):
        cm.save_checkpoint(
            "book_job", file_idx, f"Text/{file_idx:03d}.xhtml", f"Text/{file_idx:03d}.xhtml",
            chunk_data={
                "file_type": "epub_xhtml",
                "logical_completed_chunks": logical_completed,
                "logical_total_chunks": 169,
            },
            total_chunks=169, completed_chunks=logical_completed, failed_chunks=0,
        )

    checkpoint = cm.load_checkpoint("book_job")
    assert checkpoint["checkpoint_complete"] is True  # the fix

    plan = _build_finalization_recovery_plan({}, checkpoint)

    assert plan is not None
    assert plan["scope"] == "finalization"
    cm.close()


def test_exhausted_finalization_keeps_chunk_failure_counter_clean():
    stats = _finalization_recovery_exhausted_stats(
        {"failed_chunks": 0, "checkpoint_failed_chunks": 0},
        now=123.0,
    )

    assert stats["failed_chunks"] == 0
    assert stats["checkpoint_failed_chunks"] == 0
    assert stats["live_status_kind"] == "recoverable"
    assert stats["live_activity_event"] == "finalization_recovery_exhausted"
    assert "Texto completo guardado" in stats["live_status"]
    assert stats["last_activity_at"] == 123.0


def test_worker_exception_recovery_requires_durable_progress_and_is_bounded():
    checkpoint = {
        "resume_from_index": 4,
        "job": {"progress": {"total_chunks": 10, "completed_chunks": 4}},
        "chunks": [],
    }
    first = _build_worker_exception_recovery_plan({}, checkpoint)
    second = _build_worker_exception_recovery_plan(first["config"], checkpoint)
    exhausted = _build_worker_exception_recovery_plan(second["config"], checkpoint)

    assert first["scope"] == "worker"
    assert first["resume_index"] == 4
    assert first["delay_seconds"] == 3
    assert second["delay_seconds"] == 15
    assert exhausted is None
    assert _build_worker_exception_recovery_plan(
        {}, {"resume_from_index": 0, "job": {"progress": {}}, "chunks": []}
    ) is None


def test_worker_exception_budget_resets_after_checkpoint_progress():
    first_checkpoint = {
        "resume_from_index": 4,
        "job": {"progress": {"total_chunks": 10, "completed_chunks": 4}},
        "chunks": [],
    }
    later_checkpoint = {
        "resume_from_index": 6,
        "job": {"progress": {"total_chunks": 10, "completed_chunks": 6}},
        "chunks": [],
    }
    first = _build_worker_exception_recovery_plan({}, first_checkpoint)
    progressed = _build_worker_exception_recovery_plan(
        first["config"],
        later_checkpoint,
    )

    assert progressed["cycle"] == 2
    assert progressed["stuck_count"] == 1


def test_rate_limit_auto_resume_stops_after_configured_ceiling():
    config = {"max_rate_limit_auto_resumes": 2}
    first = _build_rate_limit_auto_resume_plan(config, resume_index=7)
    second = _build_rate_limit_auto_resume_plan(first["config"], resume_index=7)
    exhausted = _build_rate_limit_auto_resume_plan(second["config"], resume_index=7)

    assert first["allowed"] is True
    assert second["allowed"] is True
    assert exhausted["allowed"] is False
    assert exhausted["stuck_count"] == 3
    assert exhausted["max_resumes"] == 2


def test_rate_limit_progress_resets_stuck_counter():
    first = _build_rate_limit_auto_resume_plan({}, resume_index=7)
    progressed = _build_rate_limit_auto_resume_plan(
        first["config"],
        resume_index=8,
    )

    assert progressed["allowed"] is True
    assert progressed["stuck_count"] == 1


def test_live_activity_labels_repair_and_model_work():
    assert _live_activity_label("failed_chunk_auto_recovery_scheduled") == (
        "Reanudación automática del fragmento fallido programada"
    )
    assert _live_activity_label("target_language_gate_repair_request") == (
        "Reparando el idioma del fragmento"
    )
    assert _live_activity_label("translation_attempt") == (
        "Reintentando el fragmento automáticamente"
    )
    assert _live_activity_label("response", data={"type": "llm_response"}) == (
        "Validando la respuesta del modelo"
    )
