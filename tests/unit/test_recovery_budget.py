"""Automatic recovery must be finite across checkpoints and server restarts."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from src.api import handlers
from src.api.startup_recovery import build_startup_resume_config
from src.api.translation_state import TranslationStateManager
from src.core.llm.exceptions import RateLimitError
from src.core.recovery_policy import recovery_budget, reserve_recovery
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    checkpoints = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoints)
    config = {
        "file_type": "txt",
        "text": "Private fixture.",
        "output_filename": "fixture.txt",
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "prompt_options": {},
        "auto_pause_on_rate_limit": False,
    }
    checkpoints.start_job("fixture", "txt", config)
    checkpoints.update_progress("fixture", total_chunks=10)
    state.create_translation("fixture", config)
    monkeypatch.setattr(handlers, "_claim_translation_worker", lambda _id, c, _s: c)
    monkeypatch.setattr(handlers, "JobEngine", Mock())
    monkeypatch.setattr(handlers, "emit_update", Mock())
    monkeypatch.setattr(handlers, "notify", Mock())
    try:
        yield config, checkpoints, state
    finally:
        checkpoints.close()


def checkpoint(index=4, config=None):
    return {
        "resume_from_index": index,
        "chunks": [],
        "job": {
            "config": config or {"file_type": "txt", "text": "Private fixture."},
            "progress": {"total_chunks": 10, "completed_chunks": index},
        },
    }


def test_transient_worker_recovery_obeys_explicit_limit():
    config = {"max_worker_error_recoveries": 1}
    first = handlers._build_worker_exception_recovery_plan(
        config, checkpoint(), TimeoutError("unavailable")
    )
    assert first is not None
    assert (
        handlers._build_worker_exception_recovery_plan(
            first["config"], checkpoint(), TimeoutError("unavailable")
        )
        is None
    )


def test_alternating_failures_share_persistent_job_budget():
    config = {"max_automatic_recoveries": 2}
    worker = handlers._build_worker_exception_recovery_plan(
        config, checkpoint(), TimeoutError()
    )
    rate = handlers._build_rate_limit_auto_resume_plan(worker["config"], resume_index=5)
    assert rate["allowed"]
    assert (
        handlers._build_worker_exception_recovery_plan(
            rate["config"], checkpoint(6), TimeoutError()
        )
        is None
    )
    assert config == {"max_automatic_recoveries": 2}


def test_rate_limit_obeys_limit_even_after_progress():
    config = {"max_automatic_recoveries": 1}
    first = handlers._build_rate_limit_auto_resume_plan(config, resume_index=4)
    second = handlers._build_rate_limit_auto_resume_plan(
        first["config"], resume_index=5
    )
    assert first["allowed"]
    assert not second["allowed"]
    assert second["reason"] == "automatic_recovery_budget_exhausted"


def test_manual_pause_disables_all_recovery_plans():
    config = {"_manual_pause_requested": True}
    cp = checkpoint()
    cp["failed_chunk_indices"] = [4]
    assert handlers._build_failed_chunk_recovery_plan(config, cp, {}) is None
    assert not handlers._build_rate_limit_auto_resume_plan(config, resume_index=4)[
        "allowed"
    ]


def test_retry_after_is_not_shortened():
    assert handlers._provider_wait_seconds({}, 86400) >= 86400


def test_startup_cannot_reset_exhausted_recovery_budget():
    config = {
        "file_type": "txt",
        "text": "Private fixture.",
        "max_automatic_recoveries": 2,
        "_worker_exception_recovery_cycle": 2,
    }
    restored, reason = build_startup_resume_config(checkpoint(config=config), Mock())
    assert restored is None
    assert reason == "automatic_recovery_budget_exhausted"


@pytest.mark.parametrize("delay", [86400, float("inf"), float("nan"), -1, "bad hint"])
def test_unusable_provider_delay_stops_instead_of_retrying_early(delay):
    plan = handlers._build_rate_limit_auto_resume_plan(
        {}, resume_index=4, suggested_seconds=delay
    )
    assert not plan["allowed"]
    assert plan["reason"] == "provider_retry_after_requires_review"
    assert "_rate_limit_recovery_cycle" not in plan["config"]


def test_wait_budget_is_cumulative_despite_checkpoint_progress():
    first = handlers._build_rate_limit_auto_resume_plan(
        {"max_provider_wait_total_seconds": 100}, resume_index=4
    )
    assert first["allowed"]
    second = handlers._build_rate_limit_auto_resume_plan(
        first["config"], resume_index=5
    )
    assert not second["allowed"]
    assert second["reason"] == "provider_wait_budget_exhausted"


@pytest.mark.parametrize(
    "scope", ["worker", "rate_limit", "chunks", "finalization", "startup"]
)
def test_recovery_budget_scales_with_source_and_survives_scope_change(scope):
    original = {"model": "deepseek-flash"}
    reserved = reserve_recovery(original, scope, total_units=100)
    assert recovery_budget(reserved) == {"used": 1, "limit": 200}
    second = reserve_recovery(reserved, "worker", total_units=1)
    assert recovery_budget(second) == {"used": 2, "limit": 200}
    assert original == {"model": "deepseek-flash"}


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), 1.5, True, "bad"])
def test_corrupt_counter_never_replenishes_budget(bad):
    assert reserve_recovery({"_worker_exception_recovery_cycle": bad}, "worker") is None


@pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), 1.5, True, "bad"])
def test_corrupt_stall_counter_is_not_restarted_as_zero(bad):
    assert not handlers._build_rate_limit_auto_resume_plan(
        {"_auto_resume_last_index": 4, "_auto_resume_stuck_count": bad}, resume_index=4
    )["allowed"]
    assert (
        handlers._build_worker_exception_recovery_plan(
            {"_worker_exception_last_index": 4, "_worker_exception_stuck_count": bad},
            checkpoint(),
            TimeoutError(),
        )
        is None
    )


def test_backwards_checkpoint_does_not_reset_stalled_counter():
    config = {"max_rate_limit_auto_resumes": 2}
    first = handlers._build_rate_limit_auto_resume_plan(config, resume_index=4)
    second = handlers._build_rate_limit_auto_resume_plan(
        first["config"], resume_index=3
    )
    third = handlers._build_rate_limit_auto_resume_plan(
        second["config"], resume_index=4
    )
    assert not third["allowed"]
    assert third["reason"] == "rate_limit_recovery_exhausted"


def test_pricing_wait_neither_reserves_nor_resets_recovery_allowance():
    config = {"max_automatic_recoveries": 1, "_worker_exception_recovery_cycle": 1}
    plan = handlers._build_pricing_auto_resume_plan(config, resume_index=4)
    assert plan["allowed"]
    assert recovery_budget(plan["config"]) == {"used": 1, "limit": 1}


def test_restarted_wait_uses_existing_reservation():
    config = {
        "file_type": "txt",
        "text": "Private fixture.",
        "max_automatic_recoveries": 1,
        "_rate_limit_recovery_cycle": 1,
        "_scheduled_resume_at_epoch": 2_000_000_000,
        "_scheduled_resume_status": "provider_wait",
        "_scheduled_resume_reason": "rate_limited",
    }
    restored, reason = build_startup_resume_config(checkpoint(config=config), Mock())
    assert not reason
    assert recovery_budget(restored) == {"used": 1, "limit": 1}
    assert restored["_scheduled_resume_at_epoch"] == 2_000_000_000


def test_worker_crash_restart_loop_is_finite():
    cp = checkpoint()
    for _ in range(3):
        restored, reason = build_startup_resume_config(cp, Mock())
        assert not reason
        cp["job"]["config"] = restored
    restored, reason = build_startup_resume_config(cp, Mock())
    assert restored is None
    assert reason == "startup_recovery_exhausted"


def test_startup_progress_renews_point_allowance_not_job_allowance():
    cp = checkpoint()
    for _ in range(3):
        restored, reason = build_startup_resume_config(cp, Mock())
        assert not reason
        cp["job"]["config"] = restored
    cp["resume_from_index"] = 5
    restored, reason = build_startup_resume_config(cp, Mock())
    assert not reason
    assert restored["_startup_recovery_stuck_count"] == 1
    assert restored["_startup_recovery_cycle"] == 4


@pytest.mark.parametrize("manual", [False, True])
def test_failed_reservation_or_manual_pause_cannot_launch_recovery(
    tmp_path, monkeypatch, manual
):
    checkpoints = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoints)
    config = {"file_type": "txt", "_manual_pause_requested": manual}
    checkpoints.start_job("fixture", "txt", config)
    state.create_translation("fixture", config)
    monkeypatch.setattr(checkpoints, "update_job_config", Mock(return_value=False))
    thread = Mock(
        side_effect=AssertionError("No recovery thread without durable reservation")
    )
    monkeypatch.setattr(handlers.threading, "Thread", thread)
    monkeypatch.setattr(handlers, "emit_update", Mock())
    plan = {"config": config, "delay_seconds": 1, "scope": "worker"}
    try:
        assert not handlers._schedule_failed_chunk_recovery(
            "fixture", plan, state, str(tmp_path), Mock()
        )
        thread.assert_not_called()
        if not manual:
            assert (
                state.get_translation_field("fixture", "pause_reason")
                == "checkpoint_write_failed"
            )
            assert checkpoints.load_checkpoint("fixture")["job"]["status"] == "error"
        else:
            checkpoints.update_job_config.assert_not_called()
    finally:
        checkpoints.close()


@pytest.mark.asyncio
async def test_rate_checkpoint_write_failure_prevents_wait_and_handoff(
    worker_case, tmp_path, monkeypatch
):
    config, checkpoints, state = worker_case
    monkeypatch.setattr(checkpoints, "update_job_config", Mock(return_value=False))
    monkeypatch.setattr(
        handlers, "normalize_output_format", Mock(side_effect=RateLimitError("Busy"))
    )
    wait = Mock(side_effect=AssertionError("No wait without reservation"))
    handoff = Mock(side_effect=AssertionError("No retry without reservation"))
    monkeypatch.setattr(handlers, "_interruptible_provider_wait", wait)
    monkeypatch.setattr(handlers, "start_translation_job", handoff)
    await handlers.perform_actual_translation(
        "fixture", config, state, str(tmp_path), Mock()
    )
    assert (
        state.get_translation_field("fixture", "pause_reason")
        == "checkpoint_write_failed"
    )
    assert checkpoints.load_checkpoint("fixture")["job"]["status"] == "error"
    wait.assert_not_called()
    handoff.assert_not_called()


@pytest.mark.asyncio
async def test_worker_exhaustion_remains_visible_after_reload(
    worker_case, tmp_path, monkeypatch
):
    config, checkpoints, state = worker_case
    config.update(
        {
            "max_worker_error_recoveries": 1,
            "_worker_exception_recovery_cycle": 1,
            "_worker_exception_last_index": 0,
            "_worker_exception_stuck_count": 1,
        }
    )
    monkeypatch.setattr(
        handlers,
        "normalize_output_format",
        Mock(side_effect=TimeoutError("Provider timeout")),
    )
    schedule = Mock(side_effect=AssertionError("No new retry after exhaustion"))
    monkeypatch.setattr(handlers, "_schedule_failed_chunk_recovery", schedule)
    await handlers.perform_actual_translation(
        "fixture", config, state, str(tmp_path), Mock()
    )
    saved = checkpoints.load_checkpoint("fixture")
    assert saved["job"]["status"] == "error"
    assert saved["job"]["config"]["_worker_exception_recovery_cycle"] == 1
    assert (
        state.get_translation_field("fixture", "stats")["live_status_kind"]
        == "recoverable"
    )
    restored = TranslationStateManager(checkpoints)
    assert restored.restore_job_from_checkpoint("fixture")
    assert (
        restored.get_translation_field("fixture", "pause_reason")
        == "worker_recovery_exhausted"
    )
    schedule.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("Failed"), RateLimitError("Busy")])
async def test_old_worker_error_cannot_demote_queued_explicit_resume(
    worker_case, tmp_path, monkeypatch, failure
):
    config, checkpoints, state = worker_case

    def replaced_then_failed(_format):
        replacement = dict(config, _explicit_resume_requested=True)
        state.set_translation_field("fixture", "config", replacement)
        state.set_translation_field("fixture", "interrupted", True)
        state.set_translation_field("fixture", "status", "queued")
        checkpoints.update_job_config("fixture", replacement)
        checkpoints.mark_running("fixture")
        raise failure

    monkeypatch.setattr(handlers, "normalize_output_format", replaced_then_failed)
    await handlers.perform_actual_translation(
        "fixture", config, state, str(tmp_path), Mock()
    )
    assert state.get_translation_field("fixture", "status") == "queued"
    assert checkpoints.load_checkpoint("fixture")["job"]["status"] == "running"


@pytest.mark.asyncio
async def test_rate_retry_is_persisted_before_wait_and_not_counted_twice(
    tmp_path, monkeypatch
):
    checkpoints = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoints)
    config = {
        "file_type": "txt",
        "text": "Private fixture.",
        "output_filename": "fixture.txt",
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "prompt_options": {},
        "auto_pause_on_rate_limit": False,
        "max_rate_limit_auto_resumes": 1,
    }
    checkpoints.start_job("fixture", "txt", config)
    checkpoints.update_progress("fixture", total_chunks=10)
    state.create_translation("fixture", config)
    monkeypatch.setattr(handlers, "_claim_translation_worker", lambda _id, c, _s: c)
    monkeypatch.setattr(handlers, "JobEngine", Mock())
    monkeypatch.setattr(handlers, "emit_update", Mock())
    monkeypatch.setattr(
        handlers,
        "normalize_output_format",
        Mock(
            side_effect=RateLimitError("Throttled", provider="deepseek", retry_after=1)
        ),
    )
    handoffs = []
    monkeypatch.setattr(
        handlers,
        "start_translation_job",
        lambda _id, c, *_a, **_k: handoffs.append(deepcopy(c)),
    )

    async def waited(*_args, **_kwargs):
        durable = checkpoints.load_checkpoint("fixture")["job"]["config"]
        assert durable["_rate_limit_recovery_cycle"] == 1
        assert durable["_auto_resume_stuck_count"] == 1
        assert durable["_scheduled_resume_at_epoch"] > 0
        return True

    monkeypatch.setattr(handlers, "_interruptible_provider_wait", waited)
    try:
        await handlers.perform_actual_translation(
            "fixture", config, state, str(tmp_path), Mock()
        )
        assert len(handoffs) == 1
        assert handoffs[0]["_rate_limit_recovery_cycle"] == 1
        assert "_scheduled_resume_at_epoch" not in handoffs[0]
        wait = Mock(side_effect=AssertionError("Exhausted retry must not wait"))
        monkeypatch.setattr(handlers, "_interruptible_provider_wait", wait)
        await handlers.perform_actual_translation(
            "fixture", handoffs[0], state, str(tmp_path), Mock()
        )
        assert len(handoffs) == 1
        assert (
            state.get_translation_field("fixture", "pause_reason")
            == "rate_limit_recovery_exhausted"
        )
        assert checkpoints.load_checkpoint("fixture")["job"]["status"] == "interrupted"
        reloaded = TranslationStateManager(checkpoints)
        assert reloaded.restore_job_from_checkpoint("fixture")
        assert (
            reloaded.get_translation_field("fixture", "pause_reason")
            == "rate_limit_recovery_exhausted"
        )
        assert (
            reloaded.get_translation_field("fixture", "config")[
                "_rate_limit_recovery_cycle"
            ]
            == 1
        )
        wait.assert_not_called()
    finally:
        checkpoints.close()
