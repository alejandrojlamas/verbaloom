"""Role routing must be explicit, task-local, and credential-isolated."""

import asyncio
import json
from copy import deepcopy

import pytest

from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ProviderActionRequiredError
from src.core.llm.request_deadline import await_llm_call
from src.core.pipeline_config import apply_pipeline, pipeline_preset, validate_pipeline
from src.core.pipeline_runtime import (
    activate_pipeline,
    active_pipeline,
    close_pipeline,
    request_role,
)
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


def baseline():
    return {
        "llm_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "prompt_options": {
            "fidelity_supervisor_mode": "always",
            "quality_alert_model": "deepseek-flash",
            "target_locale": "es-ES",
        },
    }


def test_presets_change_only_roles_not_editorial_content():
    config = baseline()
    original = deepcopy(config)
    a = pipeline_preset("compatible", config)
    b = pipeline_preset("openai_equivalent", config)
    assert a["roles"]["translate"]["model"] == "deepseek-v4-pro"
    assert b["roles"]["translate"]["model"] == "gpt-6.1-terra"
    assert b["roles"]["repair"]["model"] == "gpt-6-luna"
    assert config == original
    apply_pipeline(config, b)
    assert config["prompt_options"] == original["prompt_options"]
    assert config["llm_api_endpoint"] == "https://api.openai.com/v1/chat/completions"


@pytest.mark.parametrize(
    "change",
    [
        {"version": 99},
        {"api_key": "YOUR_API_KEY_HERE"},
        {"review_policy": "skip_all"},
        {"preset": "optimized"},
        {"roles": {}},
        {"budget": {"limit_usd": "NaN", "max_calls": 10}},
    ],
)
def test_invalid_or_secret_config_is_rejected(change):
    with pytest.raises(ValueError):
        validate_pipeline({**pipeline_preset("compatible", baseline()), **change})


@pytest.mark.asyncio
async def test_mixed_roles_keep_provider_credentials_separate_and_record_actual_model(
    tmp_path, monkeypatch
):
    store = TokenUsageStore(tmp_path / "usage.db")
    policy = pipeline_preset("compatible", baseline())
    policy["roles"]["audit"] = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "reasoning_effort": "low",
        "max_output_tokens": 4096,
    }
    created = []

    class FakeProvider:
        def __init__(self, provider, options):
            self.model = options["model"]
            self.provider = provider
            self.options = options
            self.closed = False

        async def generate(self, prompt, **kwargs):
            await asyncio.sleep(0)
            return LLMResponse(
                content=self.provider, prompt_tokens=3, completion_tokens=1
            )

        async def close(self):
            self.closed = True

    def factory(provider, **options):
        raw = FakeProvider(provider, options)
        created.append(raw)
        return UsageTrackingProvider(raw, provider)

    monkeypatch.setattr("src.core.llm.factory.create_llm_provider", factory)
    legacy = FakeProvider("deepseek", {"model": "legacy-must-not-run"})
    tracked = UsageTrackingProvider(legacy, "deepseek")
    token = activate_pipeline(
        policy,
        job_id="fixture",
        store=store,
        credentials={
            "deepseek": "YOUR_DEEPSEEK_KEY_HERE",
            "openai": "YOUR_OPENAI_KEY_HERE",
        },
    )
    try:
        translated, audited = await asyncio.gather(
            await_llm_call(
                tracked.generate, "source", provider=tracked, role="translate"
            ),
            await_llm_call(tracked.generate, "audit", provider=tracked, role="audit"),
        )
        assert translated.content == "deepseek" and audited.content == "openai"
        assert request_role.get() is None
        assert {p.provider: p.options["api_key"] for p in created} == {
            "deepseek": "YOUR_DEEPSEEK_KEY_HERE",
            "openai": "YOUR_OPENAI_KEY_HERE",
        }
        for p in created:
            assert p.provider in p.options["api_endpoint"]
        events = store.events()
        assert {(e["provider"], e["model"], e["phase"]) for e in events} == {
            ("deepseek", "deepseek-v4-pro", "translate"),
            ("openai", "gpt-6-luna", "audit"),
        }
        assert "YOUR_" not in json.dumps(events)
        assert legacy.model == "legacy-must-not-run"
    finally:
        await close_pipeline(token)
    assert all(p.closed for p in created)
    assert active_pipeline.get() is None


@pytest.mark.asyncio
async def test_untagged_request_fails_before_provider_creation(tmp_path, monkeypatch):
    policy = pipeline_preset("compatible", baseline())
    token = activate_pipeline(
        policy, job_id="fixture", credentials={"deepseek": "YOUR_API_KEY_HERE"}
    )
    try:
        with pytest.raises(ProviderActionRequiredError) as error:
            await UsageTrackingProvider(object(), "deepseek").generate("fixture")
        assert error.value.reason == "pipeline_role_missing"
    finally:
        await close_pipeline(token)


@pytest.mark.asyncio
async def test_budget_and_roles_survive_resume_without_reading_changed_global_defaults(
    tmp_path,
):
    store = TokenUsageStore(tmp_path / "usage.db")
    policy = pipeline_preset("compatible", baseline())
    policy["budget"] = {"limit_usd": "5", "max_calls": 100}
    from src.core.usage.budget import SpendBudget

    for _ in range(2):
        token = activate_pipeline(
            policy,
            job_id="same-job",
            credentials={"deepseek": "YOUR_API_KEY_HERE"},
            store=store,
        )
        try:
            assert active_pipeline.get().budget_scope_ids == ("job:same-job",)
            SpendBudget(store).reserve(["job:same-job"], upper_usd="1")
        finally:
            await close_pipeline(token)
    assert SpendBudget(store).snapshot("job:same-job")["calls"] == 2


def test_missing_secondary_credential_is_detected_before_work():
    policy = pipeline_preset("openai_equivalent", baseline())
    with pytest.raises(ValueError, match="openai"):
        activate_pipeline(
            policy, job_id="fixture", credentials={"deepseek": "YOUR_API_KEY_HERE"}
        )


@pytest.mark.asyncio
async def test_cleanup_closes_every_transport_and_restores_context():
    from src.core.pipeline_runtime import PipelineRuntime

    closed = []

    class Provider:
        def __init__(self, name):
            self.name = name

        async def close(self):
            closed.append(self.name)
            if self.name == "first":
                raise RuntimeError("fixture close failure")

    runtime = PipelineRuntime({}, {})
    runtime.providers = {"first": Provider("first"), "second": Provider("second")}
    token = active_pipeline.set(runtime)
    await close_pipeline(token)
    assert closed == ["first", "second"]
    assert active_pipeline.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(content="partial", was_truncated=True),
        LLMResponse(content="refused", status="refused"),
        LLMResponse(content="unfinished", status="incomplete"),
    ],
)
async def test_legacy_client_convenience_method_cannot_approve_incomplete_response(
    response,
):
    from src.core.llm_client import LLMClient

    class Client(LLMClient):
        async def make_request(self, prompt, model=None):
            assert request_role.get() == "translate"
            return response

        def extract_translation(self, response):
            pytest.fail("Incomplete text must not be extracted as a translation")

    client = Client()
    assert await client.translate_text("fixture") is None
    assert request_role.get() is None
