"""Permanent provider stops preserve the job and never schedule a paid retry."""

from unittest.mock import Mock

import pytest

from src.api import handlers
from src.api.translation_state import TranslationStateManager
from src.core.llm.exceptions import ProviderActionRequiredError
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "provider_refusal",
        "authentication_error",
        "invalid_request",
        "model_unavailable",
    ],
)
async def test_action_required_is_persisted_without_auto_resume(
    tmp_path, monkeypatch, reason
):
    checkpoints = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoints)
    config = {
        "file_type": "txt",
        "output_filename": "fixture.txt",
        "llm_provider": "openai",
        "model": "gpt-6-luna",
        "auto_pause_on_rate_limit": False,
        "prompt_options": {},
    }
    checkpoints.start_job("fixture", "txt", config)
    state.create_translation("fixture", config)
    monkeypatch.setattr(
        handlers, "_claim_translation_worker", lambda _id, config, _state: config
    )

    async def ready(_id, config, _state, _socket):
        return config, True

    monkeypatch.setattr(handlers, "_honor_persisted_resume_schedule", ready)
    monkeypatch.setattr(handlers, "JobEngine", Mock())
    emitted = []
    monkeypatch.setattr(
        handlers,
        "emit_update",
        lambda _socket, _id, payload, _state: emitted.append(payload),
    )
    stop = ProviderActionRequiredError(
        "Provider needs attention.", reason=reason, provider="openai"
    )
    monkeypatch.setattr(handlers, "normalize_output_format", Mock(side_effect=stop))
    wait = Mock(side_effect=AssertionError("Permanent failures must not wait/retry"))
    monkeypatch.setattr(handlers, "_interruptible_provider_wait", wait)
    await handlers.perform_actual_translation(
        "fixture", config, state, str(tmp_path), Mock()
    )
    assert state.get_translation_field("fixture", "status") == "interrupted"
    assert state.get_translation_field("fixture", "pause_reason") == reason
    saved = checkpoints.load_checkpoint("fixture")
    assert saved["job"]["config"]["_provider_action_required"] == reason
    assert saved["job"]["status"] == "interrupted"
    assert not saved["job"]["config"].get("_scheduled_resume_at_epoch")
    assert emitted[-1]["result"].startswith("Provider needs attention.")
    wait.assert_not_called()
    checkpoints.close()


@pytest.mark.asyncio
async def test_worker_activates_saved_budget_and_restores_outer_context(
    tmp_path, monkeypatch
):
    from src.core.pipeline_config import apply_pipeline, pipeline_preset
    from src.core.pipeline_runtime import active_pipeline
    from src.core.usage.budget import SpendBudget
    from src.core.usage.context import (
        get_usage_context,
        reset_usage_context,
        set_usage_context,
    )
    from src.core.usage.store import TokenUsageStore

    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.store.default_usage_store", lambda: store)
    config = {
        "file_type": "txt",
        "output_filename": "fixture.txt",
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "deepseek_api_key": "YOUR_API_KEY_HERE",
        "prompt_options": {},
        "auto_pause_on_rate_limit": False,
    }
    policy = pipeline_preset("compatible", config)
    policy["budget"] = {"limit_usd": "5", "max_calls": 1}
    apply_pipeline(config, policy)
    checkpoints = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(checkpoints)
    checkpoints.start_job("fixture", "txt", config)
    state.create_translation("fixture", config)
    monkeypatch.setattr(
        handlers, "_claim_translation_worker", lambda _id, config, _state: config
    )

    async def ready(_id, config, _state, _socket):
        return config, True

    monkeypatch.setattr(handlers, "_honor_persisted_resume_schedule", ready)
    monkeypatch.setattr(handlers, "emit_update", Mock())
    monkeypatch.setattr(handlers, "JobEngine", Mock())
    budget = SpendBudget(store)

    def at_first_stage(_format):
        runtime = active_pipeline.get()
        assert runtime.config == config["pipeline"]
        assert runtime.budget_scope_ids == ("job:fixture",)
        budget.reserve(runtime.budget_scope_ids, upper_usd="1")
        budget.reserve(runtime.budget_scope_ids, upper_usd="1")

    monkeypatch.setattr(handlers, "normalize_output_format", at_first_stage)
    wait = Mock(
        side_effect=AssertionError("Budget exhaustion must not automatically retry")
    )
    monkeypatch.setattr(handlers, "_interruptible_provider_wait", wait)
    outer = set_usage_context(process_id="outer-task")
    try:
        await handlers.perform_actual_translation(
            "fixture", config, state, str(tmp_path), Mock()
        )
        assert active_pipeline.get() is None
        assert get_usage_context()["process_id"] == "outer-task"
        assert (
            state.get_translation_field("fixture", "pause_reason")
            == "call_budget_exhausted"
        )
        assert budget.snapshot("job:fixture")["calls"] == 1
        assert (
            checkpoints.load_checkpoint("fixture")["job"]["config"]["pipeline"]
            == config["pipeline"]
        )
        wait.assert_not_called()
    finally:
        reset_usage_context(outer)
        checkpoints.close()
