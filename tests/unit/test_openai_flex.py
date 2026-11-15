"""Opt-in Flex keeps provider errors, deadlines and spend reservations intact."""

import asyncio
import json
from collections import deque
from unittest.mock import AsyncMock

import httpx
import pytest

import src.config as config
from src.core.llm.exceptions import (
    ContextOverflowError,
    InsufficientCreditsError,
    ProviderActionRequiredError,
    RateLimitError,
)
from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.llm.providers.openai_contract import text_parameters
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.context import reset_usage_context, set_usage_context
from src.core.usage.request_budget import request_bound
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


OFFICIAL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-6.1-terra"


def ok(tier="default"):
    return 200, {
        "model": MODEL,
        "service_tier": tier,
        "choices": [{"finish_reason": "stop", "message": {"content": "Accepted."}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
    }


def error(status, code, message="Rejected"):
    return status, {"error": {"code": code, "message": message}}


@pytest.fixture
def make(monkeypatch):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "default")
    monkeypatch.setattr(config, "OPENAI_FLEX_COOLDOWN_SECONDS", 600)
    monkeypatch.setattr("src.core.llm.providers.openai.asyncio.sleep", AsyncMock())

    def build(events, *, endpoint=OFFICIAL, api_key=None, model=MODEL):
        calls, queue = [], deque(events)

        class Client:
            async def post(self, url, **kwargs):
                calls.append(kwargs)
                event = queue.popleft()
                if isinstance(event, BaseException):
                    raise event
                status, body = event
                return httpx.Response(status, json=body, request=httpx.Request("POST", url))

        provider = OpenAICompatibleProvider(endpoint, model, api_key=api_key)
        monkeypatch.setattr(provider, "_get_client", AsyncMock(return_value=Client()))
        return provider, calls

    return build


@pytest.mark.asyncio
async def test_default_request_and_timeout_are_unchanged(make):
    provider, calls = make([ok()])
    await provider.generate("Unit", system_prompt="Instructions", timeout=900)
    assert calls[0]["timeout"] == 900
    assert calls[0]["json"] == {
        "model": MODEL,
        "messages": [{"role": "system", "content": "Instructions"},
                     {"role": "user", "content": "Unit"}],
        "stream": False,
        **text_parameters(MODEL, effort="none", max_output_tokens=8192),
    }


@pytest.mark.asyncio
async def test_explicit_capacity_error_falls_back_with_accurate_attempt_records(make, monkeypatch):
    from src.core.llm.attempts import attempt_observer

    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    provider, calls = make([error(429, "resource_unavailable"), ok(), ok()])
    records = []
    token = attempt_observer.set(lambda **event: records.append(event))
    try:
        assert (await provider.generate("Unit", timeout=900)).content == "Accepted."
    finally:
        attempt_observer.reset(token)
    assert [call["json"]["service_tier"] for call in calls] == ["flex", "default"]
    assert [call["timeout"] for call in calls] == [900, 900]
    assert [row["metadata"]["parameters"]["service_tier"] for row in records] == ["flex", "default"]
    assert [row["metadata"]["attempt"] for row in records] == [1, 2]
    assert records[0]["metadata"]["reason"] == "flex_unavailable"
    assert not records[0]["metadata"]["billing_uncertain"]
    assert provider._retry_manager._circuit_breaker._failure_count == 0
    await provider.generate("Next unit")
    assert calls[-1]["json"]["service_tier"] == "default"
    provider._flex_blocked_until = 0
    assert provider._editorial_service_tier() == "flex"


@pytest.mark.parametrize("api_key", [None, "YOUR_API_KEY_HERE"])
@pytest.mark.asyncio
async def test_real_rate_limit_never_falls_back_to_standard(make, monkeypatch, api_key):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    rotation = AsyncMock(side_effect=RateLimitError("Rate limited", provider="openai"))
    monkeypatch.setattr("src.core.llm.providers.openai.handle_rate_limit", rotation)
    provider, calls = make([error(429, "rate_limit_exceeded")], api_key=api_key)
    with pytest.raises(RateLimitError):
        await provider.generate("Unit")
    assert len(calls) == 1
    assert provider._flex_blocked_until == 0
    assert rotation.await_count == (1 if api_key else 0)


@pytest.mark.parametrize("status, code, expected", [
    (400, "unsupported_parameter", ProviderActionRequiredError),
    (400, "context_length_exceeded", ContextOverflowError),
    (401, "invalid_api_key", ProviderActionRequiredError),
    (429, "insufficient_quota", InsufficientCreditsError),
    (429, "unknown_capacity_error", RateLimitError),
])
@pytest.mark.asyncio
async def test_unrelated_errors_are_never_retried_as_capacity(make, monkeypatch, status, code, expected):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    provider, calls = make([error(status, code)])
    with pytest.raises(expected):
        await provider.generate("Unit")
    assert len(calls) == 1
    assert provider._flex_blocked_until == 0


@pytest.mark.asyncio
async def test_fallback_cannot_exceed_configured_attempt_limit(make, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    monkeypatch.setattr("src.core.llm.providers.openai.MAX_TRANSLATION_ATTEMPTS", 1)
    provider, calls = make([error(429, "resource_unavailable")])
    with pytest.raises(RateLimitError, match="attempt limit"):
        await provider.generate("Unit")
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_custom_and_legacy_models_do_not_inherit_flex(make, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    for endpoint, model in [("http://127.0.0.1:8080/v1", MODEL), (OFFICIAL, "legacy-model")]:
        provider, calls = make([ok()], endpoint=endpoint, model=model)
        await provider.generate("Unit")
        assert "service_tier" not in calls[0]["json"]


def test_flex_retains_full_standard_reservation():
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Unit"}],
               "stream": False, **text_parameters(MODEL, effort="none", max_output_tokens=8192)}
    standard = request_bound("openai", OFFICIAL, payload)
    flex = request_bound("openai", OFFICIAL, {**payload, "service_tier": "flex"})
    assert flex == standard
    assert float(flex[0]) == pytest.approx(2.68644)
    assert flex[1]["bound_basis"] == "full_context_window"


@pytest.mark.asyncio
async def test_invalid_tier_is_rejected_before_network(make, monkeypatch):
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "auto")
    provider, calls = make([])
    with pytest.raises(ValueError, match="service tier"):
        await provider.generate("Unit")
    assert not calls


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    return store, SpendBudget(store)


@pytest.mark.asyncio
async def test_fallback_obeys_per_job_call_limit_before_second_send(make, monkeypatch, ledger):
    store, budget = ledger
    budget.configure("job", limit_usd="10", max_calls=1)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    provider, calls = make([error(429, "resource_unavailable")])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(BudgetLimitError) as raised:
            await UsageTrackingProvider(provider, "openai").generate("Unit")
    finally:
        reset_usage_context(token)
    assert raised.value.reason == "call_budget_exhausted"
    assert len(calls) == len(store.events()) == 1
    assert budget.snapshot("job")["reserved_usd"] == 0


@pytest.mark.asyncio
async def test_nonbillable_capacity_release_makes_room_for_budgeted_fallback(make, monkeypatch, ledger):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=2)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    provider, calls = make([error(429, "resource_unavailable"), ok()])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        result = await UsageTrackingProvider(provider, "openai").generate("Unit")
    finally:
        reset_usage_context(token)
    assert result.content == "Accepted."
    assert len(calls) == len(store.events()) == 2
    snapshot = budget.snapshot("job")
    assert snapshot["reserved_usd"] == snapshot["uncertain_calls"] == 0
    assert snapshot["settled_usd"] == pytest.approx(0.0015)
    events = store.events()
    rejected = next(e for e in events if e["status"] == "error")
    assert rejected["total_cost_usd"] == 0
    assert json.loads(rejected["metadata"])["cost_basis"] == "unbilled_rejection"


@pytest.mark.parametrize("extra, nested", [
    ({"usage": {"prompt_tokens": 1000, "completion_tokens": 100}}, False),
    ({"usage": {}}, False),
    ({"choices": [{"message": {"content": "Unexpected generated text"}}]}, False),
    ({"output": [{"content": "Unexpected generated text"}]}, False),
    ({"content": "Unexpected generated text"}, False),
    ({"usage": {"prompt_tokens": 1000}}, True),
    ({"message": {"content": "Unexpected nested result"}}, True),
])
@pytest.mark.asyncio
async def test_provider_preserves_allowance_for_capacity_body_with_result_evidence(
    make, monkeypatch, ledger, extra, nested
):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=2)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    body = {"error": {"code": "resource_unavailable", "message": "Unavailable"}}
    (body["error"] if nested else body).update(extra)
    provider, calls = make([(429, body)])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(ProviderActionRequiredError) as raised:
            await UsageTrackingProvider(provider, "openai").generate("Unit")
    finally:
        reset_usage_context(token)
    assert raised.value.reason == "flex_capacity_evidence_conflict"
    assert len(calls) == len(store.events()) == 1
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2.68644)
    assert budget.snapshot("job")["uncertain_calls"] == 1
    assert budget.snapshot("job")["settled_usd"] == 0
    assert provider._flex_blocked_until == 0
    metadata = json.loads(store.events()[0]["metadata"])
    assert metadata["billing_uncertain"] is True
    assert metadata["reason"] == "flex_capacity_evidence_conflict"
    assert metadata["cost_basis"] == "unknown"


@pytest.mark.parametrize("change", [
    {"http_status": 200},
    {"http_status": 500},
    {"reason": "rate_limit_exceeded"},
    {"error": "ambiguous_timeout"},
    {"parameters": {"service_tier": "default"}},
    {"billing_uncertain": True},
    {"billing_uncertain": None},
    {"raw_usage": {"prompt_tokens": 0}},
])
def test_only_exact_capacity_evidence_releases_reservation(ledger, change):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=1)
    reservation = budget.reserve(["job"], upper_usd="2.68644")
    metadata = {
        "scope": "http_attempt", "http_status": 429,
        "error": "http_error", "reason": "flex_unavailable",
        "parameters": {"service_tier": "flex"}, "billing_uncertain": False,
        "budget_reservation_id": reservation, **change,
    }
    store.record_call(provider="openai", model=MODEL, prompt="Unit", status="error", metadata=metadata)
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2.68644)
    assert budget.snapshot("job")["uncertain_calls"] == 1
    assert json.loads(store.events()[0]["metadata"])["cost_basis"] == "unknown"


@pytest.mark.parametrize("provider, model", [("openai-compatible", MODEL), ("openai", "other-model")])
def test_capacity_evidence_does_not_apply_to_other_providers_or_models(ledger, provider, model):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=1)
    reservation = budget.reserve(["job"], upper_usd="2.68644")
    store.record_call(provider=provider, model=model, prompt="Unit", status="error", metadata={
        "scope": "http_attempt", "http_status": 429,
        "error": "http_error", "reason": "flex_unavailable",
        "parameters": {"service_tier": "flex"}, "billing_uncertain": False,
        "budget_reservation_id": reservation,
    })
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2.68644)


def test_reported_usage_is_not_erased_by_capacity_metadata(ledger):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=1)
    reservation = budget.reserve(["job"], upper_usd="2.68644")
    store.record_call(provider="openai", model=MODEL, prompt="Unit", status="error",
                      prompt_tokens=1000, completion_tokens=100, metadata={
        "scope": "http_attempt", "http_status": 429,
        "error": "http_error", "reason": "flex_unavailable",
        "parameters": {"service_tier": "flex"}, "billing_uncertain": False,
        "raw_usage": {"prompt_tokens": 1000, "completion_tokens": 100},
        "usage_observed": True, "budget_reservation_id": reservation,
    })
    assert budget.snapshot("job")["settled_usd"] == pytest.approx(0.00075)
    assert json.loads(store.events()[0]["metadata"])["cost_basis"] == "provider_usage"


@pytest.mark.parametrize("extra", [
    {"prompt_cache_hit_tokens": 100},
    {"prompt_cache_miss_tokens": 100},
    {"response_content": "Unexpected output"},
])
def test_contradictory_content_or_cache_tokens_keep_allowance(ledger, extra):
    store, budget = ledger
    budget.configure("job", limit_usd="3", max_calls=1)
    reservation = budget.reserve(["job"], upper_usd="2.68644")
    store.record_call(provider="openai", model=MODEL, prompt="Unit", status="error", metadata={
        "scope": "http_attempt", "http_status": 429,
        "error": "http_error", "reason": "flex_unavailable",
        "parameters": {"service_tier": "flex"}, "billing_uncertain": False,
        "budget_reservation_id": reservation,
    }, **extra)
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2.68644)


@pytest.mark.parametrize("actual_tier, expected_cost", [("flex", 0.00075), ("default", 0.0015)])
@pytest.mark.asyncio
async def test_settlement_uses_response_tier(make, monkeypatch, ledger, actual_tier, expected_cost):
    store, budget = ledger
    budget.configure("job", limit_usd="10", max_calls=2)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    provider, calls = make([ok(actual_tier)])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        await UsageTrackingProvider(provider, "openai").generate("Unit")
    finally:
        reset_usage_context(token)
    assert calls[0]["json"]["service_tier"] == "flex"
    assert budget.snapshot("job")["settled_usd"] == pytest.approx(expected_cost)
    assert budget.snapshot("job")["reserved_usd"] == 0
    metadata = json.loads(store.events()[0]["metadata"])
    assert metadata["service_tier"] == actual_tier
    assert metadata["parameters"]["service_tier"] == "flex"


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.asyncio
async def test_timeout_and_cancellation_keep_uncertain_allowance_without_tier_fallback(make, monkeypatch, ledger, cancelled):
    store, budget = ledger
    budget.configure("job", limit_usd="10", max_calls=2)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    monkeypatch.setattr("src.core.llm.providers.openai.MAX_TRANSLATION_ATTEMPTS", 1)
    event = asyncio.CancelledError() if cancelled else httpx.ReadTimeout("No response")
    provider, calls = make([event])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        if cancelled:
            with pytest.raises(asyncio.CancelledError):
                await UsageTrackingProvider(provider, "openai").generate("Unit")
        else:
            assert await UsageTrackingProvider(provider, "openai").generate("Unit") is None
    finally:
        reset_usage_context(token)
    assert len(calls) == len(store.events()) == 1
    assert calls[0]["json"]["service_tier"] == "flex"
    assert budget.snapshot("job")["uncertain_calls"] == 1
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2.68644)
    assert provider._flex_blocked_until == 0


@pytest.mark.asyncio
async def test_existing_timeout_retry_keeps_flex_and_both_uncertain_reservations(make, monkeypatch, ledger):
    store, budget = ledger
    budget.configure("job", limit_usd="10", max_calls=2)
    monkeypatch.setattr(config, "OPENAI_SERVICE_TIER", "flex")
    monkeypatch.setattr("src.core.llm.providers.openai.MAX_TRANSLATION_ATTEMPTS", 2)
    provider, calls = make([httpx.ReadTimeout("No response"), httpx.ReadTimeout("Still unknown")])
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        assert await UsageTrackingProvider(provider, "openai").generate("Unit") is None
    finally:
        reset_usage_context(token)
    assert [call["json"]["service_tier"] for call in calls] == ["flex", "flex"]
    assert len(store.events()) == 2
    assert budget.snapshot("job")["uncertain_calls"] == 2
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(2 * 2.68644)
