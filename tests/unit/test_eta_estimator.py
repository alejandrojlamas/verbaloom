from pathlib import Path

from src.core.progress.eta import (
    apply_active_timing,
    begin_active_timing,
    restore_timing_checkpoint,
    timing_checkpoint_from_stats,
)


def test_eta_uses_full_active_interval_not_time_since_last_poll():
    stats = begin_active_timing(
        {"completed_chunks": 0, "total_chunks": 100},
        now=1_000.0,
    )
    stats["completed_chunks"] = 1
    stats = apply_active_timing(stats, now=1_030.0)
    assert stats["eta_status"] == "calculating"

    # Several UI polls or live-status events can happen while the next units
    # are still running. None may move the timing baseline forward.
    stats = apply_active_timing(stats, now=1_080.0)
    stats["completed_chunks"] = 3
    stats = apply_active_timing(stats, now=1_090.0)

    assert stats["eta_status"] == "estimated"
    assert stats["eta_sample_units"] == 3
    # 90 active seconds / 3 units, not 10 seconds since the last poll.
    assert stats["eta_seconds"] >= (97 * 30)


def test_eta_includes_separate_refinement_pass():
    stats = begin_active_timing(
        {
            "completed_chunks": 0,
            "total_chunks": 100,
            "enable_refinement": True,
            "current_phase": 1,
        },
        now=0.0,
    )
    stats["completed_chunks"] = 50
    stats = apply_active_timing(stats, now=500.0)

    # 50 translation units plus the expected 100-unit refine pass remain.
    assert stats["eta_seconds"] >= 1_500.0


def test_eta_freezes_during_provider_wait():
    stats = begin_active_timing(
        {"completed_chunks": 0, "total_chunks": 20},
        now=100.0,
    )
    stats["completed_chunks"] = 5
    stats = apply_active_timing(stats, now=150.0)
    frozen = apply_active_timing(
        stats,
        now=10_000.0,
        advance=False,
        status="pricing_wait",
    )

    assert frozen["elapsed_time"] == 50.0
    assert frozen["eta_seconds"] is None
    assert frozen["eta_status"] == "waiting"


def test_resume_excludes_paused_wall_time_and_preserves_rate():
    stats = begin_active_timing(
        {"completed_chunks": 0, "total_chunks": 20},
        now=100.0,
    )
    stats["completed_chunks"] = 5
    stats = apply_active_timing(stats, now=150.0)

    resumed = begin_active_timing(stats, now=10_000.0)
    resumed["completed_chunks"] = 10
    resumed = apply_active_timing(resumed, now=10_050.0)

    assert resumed["elapsed_time"] == 100.0
    assert resumed["eta_status"] == "estimated"
    assert resumed["eta_seconds"] < 200.0


def test_timing_checkpoint_preserves_rate_without_stale_clock_fields():
    stats = begin_active_timing(
        {"completed_chunks": 0, "total_chunks": 20},
        now=100.0,
    )
    stats["completed_chunks"] = 5
    stats = apply_active_timing(stats, now=150.0)

    payload = timing_checkpoint_from_stats(stats)
    restored = restore_timing_checkpoint({
        "completed_chunks": 5,
        "total_chunks": 20,
        "eta_timing": payload,
    })

    assert restored["active_elapsed_seconds"] == 50.0
    assert restored["_eta_rate_seconds_per_unit"] == 10.0
    assert "active_run_started_at" not in payload
    assert "eta_seconds" not in payload
    assert "eta_timing" not in restored


def test_completed_chunks_show_finalizing_until_job_is_terminal():
    stats = begin_active_timing(
        {"completed_chunks": 0, "total_chunks": 3},
        now=0.0,
    )
    stats["completed_chunks"] = 3
    active = apply_active_timing(stats, now=30.0, status="running")
    done = apply_active_timing(active, now=30.0, advance=False, status="completed")

    assert active["eta_status"] == "finalizing"
    assert active["eta_seconds"] is None
    assert done["eta_status"] == "complete"
    assert done["eta_seconds"] == 0.0


def test_frontend_uses_server_eta_instead_of_poll_intervals():
    source = (
        Path(__file__).parents[2]
        / "src/web/static/js/translation/progress-manager.js"
    ).read_text()

    assert "stats.eta_seconds" in source
    assert "stats.eta_lower_seconds" in source
    assert "chunkCompletionTimes" not in source
    assert "lastElapsedTime" not in source
