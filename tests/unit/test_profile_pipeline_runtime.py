"""Profile HTTP input and execution use the same role/budget contracts as jobs."""

import json
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from flask import Flask

from src.api.blueprints import profile_routes as routes
from src.core.llm.request_deadline import await_llm_call
from src.core.pipeline_config import pipeline_preset
from src.core.pipeline_runtime import active_pipeline
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.context import get_usage_context
from src.core.usage.store import TokenUsageStore


def policy():
    result = pipeline_preset(
        "compatible", {"llm_provider": "deepseek", "model": "deepseek-flash"}
    )
    result["roles"]["profile_review"] = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "reasoning_effort": "low",
        "max_output_tokens": 4096,
    }
    result["budget"] = {"limit_usd": "5", "max_calls": 2}
    return result


def collect(pipeline):
    app = Flask(__name__)
    with app.test_request_context(
        json={
            "text": "The inventor did not agree with the proposed design. " * 20,
            "source_name": "fixture.txt",
            "source_language": "English",
            "language": "Spanish",
            "target_locale": "es-ES",
            "profile_id": "fixture-profile",
            "pipeline": pipeline,
            "deepseek_api_key": "YOUR_DEEPSEEK_KEY_HERE",
            "openai_api_key": "YOUR_OPENAI_KEY_HERE",
        }
    ):
        return routes._collect_profile_prep_payload()


@pytest.mark.parametrize("multipart", [False, True])
def test_profile_defaults_to_flow_b_unless_model_choice_is_explicit(monkeypatch, multipart):
    monkeypatch.setenv("VERBALOOM_DEFAULT_PIPELINE", "openai_equivalent")
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY_HERE")
    app = Flask(__name__)
    data = {"text": "The inventor approved the proposed design.", "source_language": "English"}
    for extra, models in (
        ({}, ("openai", "gpt-6-luna", "gpt-6.1-terra")),
        ({"provider": "deepseek"}, ("deepseek", "deepseek-flash", "deepseek-v4-pro")),
    ):
        request_data = {**data, **extra}
        kwargs = (
            {"data": {**request_data, "files": (BytesIO(data["text"].encode()), "fixture.txt")}}
            if multipart else {"json": request_data}
        )
        with app.test_request_context(**kwargs):
            result = routes._collect_profile_prep_payload()
        assert (result["provider_type"], result["model"], result["review_model"]) == models
        assert ("pipeline" in result) == (not extra)


@pytest.fixture
def simulated(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.store.default_usage_store", lambda: store)
    monkeypatch.setattr(
        "src.core.usage.tracking_provider.default_usage_store", lambda: store
    )
    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.get_deepseek_pricing_status",
        lambda: SimpleNamespace(disabled=False),
    )
    calls = []

    class Client:
        async def post(self, url, **kwargs):
            calls.append((url, kwargs))
            assert SpendBudget(store).snapshot("job:fixture-prep")["reserved_usd"] > 0
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "model": kwargs["json"]["model"],
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "{}"}}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                },
            )

    async def get_client(self):
        return Client()

    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.DeepSeekProvider._get_client", get_client
    )
    monkeypatch.setattr(
        "src.core.llm.providers.openai.OpenAICompatibleProvider._get_client", get_client
    )

    async def prepare(text, **kwargs):
        assert kwargs["target_locale"] == "es-ES"
        assert kwargs["term_review_provider_name"] == "openai"
        for field, role in (
            ("llm_provider", "profile_prepare"),
            ("term_review_provider", "profile_review"),
        ):
            provider = kwargs[field]
            await await_llm_call(provider.generate, text, provider=provider, role=role)
        return SimpleNamespace(to_dict=lambda: {"profile_id": "fixture-profile"})

    monkeypatch.setattr(routes, "prepare_book_profile_from_text", prepare)
    payload = collect(policy())
    payload["prep_id"] = "fixture-prep"
    return payload, store, calls


def test_profile_execution_routes_each_role_and_budget_survives_retry(simulated):
    payload, store, calls = simulated
    routes._run_profile_preparation(payload)
    assert active_pipeline.get() is None
    assert get_usage_context() == {}
    assert [(c[0], c[1]["headers"]["Authorization"]) for c in calls] == [
        ("https://api.deepseek.com/chat/completions", "Bearer YOUR_DEEPSEEK_KEY_HERE"),
        ("https://api.openai.com/v1/chat/completions", "Bearer YOUR_OPENAI_KEY_HERE"),
    ]
    assert calls[1][1]["json"]["reasoning_effort"] == "low"
    events = store.events()
    assert {e["phase"] for e in events} == {"profile_prepare", "profile_review"}
    assert {e["process_id"] for e in events} == {"fixture-profile"}
    assert "YOUR_" not in json.dumps(events)
    with pytest.raises(BudgetLimitError):
        routes._run_profile_preparation(payload)
    assert len(calls) == 2
    assert SpendBudget(store).snapshot("job:fixture-prep")["calls"] == 2
    assert active_pipeline.get() is None


@pytest.mark.asyncio
async def test_profile_worker_thread_keeps_usage_context(simulated):
    payload, store, _ = simulated
    routes._run_profile_preparation(payload)
    assert {e["process_id"] for e in store.events()} == {"fixture-profile"}
    assert active_pipeline.get() is None
    assert get_usage_context() == {}


def test_job_exposes_budget_stop_not_false_completion(simulated):
    payload, _store, calls = simulated
    payload["pipeline"]["budget"]["max_calls"] = 1
    prep_id = payload["prep_id"]
    routes._PROFILE_PREP_JOBS[prep_id] = routes._new_profile_prep_job(prep_id, payload)
    try:
        routes._run_profile_prep_job(prep_id, payload)
        snapshot = routes._profile_prep_job_snapshot(prep_id)
        assert snapshot["status"] == "failed"
        assert snapshot["reason"] == "call_budget_exhausted"
        assert snapshot["retryable"] is False
        assert snapshot["result"] is None
        assert snapshot["pipeline"] == payload["pipeline"]
        assert "YOUR_" not in json.dumps(snapshot)
        assert len(calls) == 1
    finally:
        routes._PROFILE_PREP_JOBS.pop(prep_id, None)


def test_profile_fingerprint_includes_effective_role_configuration():
    payload = collect(policy())
    changed = deepcopy(payload)
    changed["pipeline"]["roles"]["profile_review"]["reasoning_effort"] = "none"
    assert routes._profile_prep_request_fingerprint(
        payload
    ) != routes._profile_prep_request_fingerprint(changed)


def test_profile_only_requires_credentials_for_its_execution_roles(monkeypatch):
    pipeline = policy()
    pipeline["roles"]["profile_review"] = deepcopy(pipeline["roles"]["profile_prepare"])
    pipeline["roles"]["translate"] = {
        "provider": "openai",
        "model": "gpt-6.1-terra",
        "reasoning_effort": "none",
        "max_output_tokens": 8192,
    }
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    payload = collect(pipeline)
    assert "openai_api_key" not in payload
    assert payload["review_provider_type"] == "deepseek"
