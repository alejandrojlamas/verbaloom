"""HTTP fault injection exercises the actual tracking proxy and provider."""

import asyncio
import json

import httpx
import pytest

from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.llm.providers.openai_contract import text_parameters
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.context import reset_usage_context, set_usage_context
from src.core.usage.request_budget import request_bound
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


def payload(model="gpt-6-luna"):
    return {
        "model": model,
        "messages": [{"role": "user", "content": "fixture"}],
        "stream": False,
        **text_parameters(model, effort="none", max_output_tokens=8192),
    }


def test_full_window_reservation_covers_writes_and_reasoning():
    amount, metadata = request_bound(
        "openai", "https://api.openai.com/v1/chat/completions", payload()
    )
    assert float(amount) == pytest.approx(0.268644)
    assert metadata["bound_basis"] == "full_context_window"
    amount, _ = request_bound(
        "openai", "https://api.openai.com/v1/chat/completions", payload("gpt-6.1-terra")
    )
    assert float(amount) == pytest.approx(2.68644)


@pytest.mark.parametrize(
    "change",
    [
        {"service_tier": "auto"},
        {"max_completion_tokens": None},
        {"tools": []},
        {"n": 2},
        {"prompt_cache_options": {"ttl": "12h"}},
        {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
        {"model": "gpt-6-luna-unknown"},
    ],
)
def test_unbounded_contracts_fail_closed(change):
    with pytest.raises(BudgetLimitError):
        request_bound(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            {**payload(), **change},
        )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr(
        "src.core.usage.tracking_provider.default_usage_store", lambda: store
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr("src.core.llm.providers.openai.asyncio.sleep", no_sleep)
    return store, SpendBudget(store)


def tracked(monkeypatch, post):
    class Client:
        async def post(self, url, **kwargs):
            return await post(url, **kwargs)

    provider = OpenAICompatibleProvider("https://api.openai.com/v1", "gpt-6-luna")

    async def client():
        return Client()

    monkeypatch.setattr(provider, "_get_client", client)
    return UsageTrackingProvider(provider, "openai")


@pytest.mark.asyncio
async def test_timeout_retry_cannot_bypass_spend_limit(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="0.3", max_calls=10)
    calls = []

    async def post(url, **kwargs):
        calls.append(kwargs)
        assert budget.snapshot("job")["reserved_usd"] > 0  # before bytes leave
        raise httpx.ReadTimeout("fixture timeout")

    provider = tracked(monkeypatch, post)
    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(BudgetLimitError) as error:
            await provider.generate("fixture")
        assert error.value.reason == "budget_exhausted"
    finally:
        reset_usage_context(token)
    assert len(calls) == 1
    assert len(store.events()) == 1
    assert budget.snapshot("job")["uncertain_calls"] == 1


@pytest.mark.asyncio
async def test_success_settles_cost_once_and_records_parameters(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="0.3", max_calls=2)

    async def post(url, **kwargs):
        assert budget.snapshot("job")["reserved_usd"] > 0
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "service_tier": "default",
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 900,
                    },
                },
            },
        )

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        await tracked(monkeypatch, post).generate("fixture")
    finally:
        reset_usage_context(token)
    snapshot = budget.snapshot("job")
    assert snapshot["reserved_usd"] == 0
    assert snapshot["settled_usd"] == pytest.approx(0.0001725)
    assert snapshot["calls"] == len(store.events()) == 1
    assert (
        json.loads(store.events()[0]["metadata"])["parameters"]["service_tier"]
        == "default"
    )


@pytest.mark.asyncio
async def test_cancelled_request_keeps_allowance(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="1", max_calls=10)

    async def post(url, **kwargs):
        raise asyncio.CancelledError()

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(asyncio.CancelledError):
            await tracked(monkeypatch, post).generate("fixture")
    finally:
        reset_usage_context(token)
    assert budget.snapshot("job")["uncertain_calls"] == 1
    assert len(store.events()) == 1


@pytest.mark.asyncio
async def test_unsupported_provider_never_reaches_network(setup):
    _, budget = setup
    budget.configure("job", limit_usd="1", max_calls=10)

    class Unsupported:
        model = "legacy"

        async def generate(self, *args, **kwargs):
            raise AssertionError("Must not be called")

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(BudgetLimitError, match="does not support"):
            await UsageTrackingProvider(Unsupported(), "deepseek").generate("fixture")
    finally:
        reset_usage_context(token)
    assert budget.snapshot("job")["calls"] == 0


@pytest.mark.asyncio
async def test_cancellation_wins_over_a_usage_write_failure(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="1", max_calls=10)

    def fail_record(**kwargs):
        raise OSError("fixture ledger unavailable")

    monkeypatch.setattr(store, "record_call", fail_record)

    async def post(url, **kwargs):
        raise asyncio.CancelledError()

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(asyncio.CancelledError):
            await tracked(monkeypatch, post).generate("fixture")
    finally:
        reset_usage_context(token)
    assert budget.snapshot("job")["reserved_usd"] > 0
    assert len(store.events()) == 0


@pytest.mark.asyncio
async def test_success_with_ledger_failure_stops_before_any_retry(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="1", max_calls=10)
    calls = []

    def fail_record(**kwargs):
        raise OSError("fixture ledger unavailable")

    monkeypatch.setattr(store, "record_call", fail_record)

    async def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(BudgetLimitError) as error:
            await tracked(monkeypatch, post).generate("fixture")
        assert error.value.reason == "budget_ledger_unavailable"
    finally:
        reset_usage_context(token)
    assert len(calls) == 1
    assert budget.snapshot("job")["reserved_usd"] > 0
