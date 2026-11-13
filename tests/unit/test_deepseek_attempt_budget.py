"""Regression cases for the DeepSeek HTTP boundary, without paid requests."""

import asyncio

import httpx
import pytest

from src.core.llm.exceptions import ProviderActionRequiredError, RateLimitError
from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.context import reset_usage_context, set_usage_context
from src.core.usage.request_budget import request_bound
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr(
        "src.core.usage.tracking_provider.default_usage_store", lambda: store
    )
    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.get_deepseek_pricing_status",
        lambda: type("Pricing", (), {"disabled": False})(),
    )

    async def no_sleep(_):
        pass

    monkeypatch.setattr("src.core.llm.providers.deepseek.asyncio.sleep", no_sleep)
    return store, SpendBudget(store)


def tracked(monkeypatch, post, **options):
    provider = DeepSeekProvider("YOUR_API_KEY_HERE", model="deepseek-flash", **options)

    class Client:
        async def post(self, url, **kwargs):
            return await post(url, **kwargs)

    async def client():
        return Client()

    monkeypatch.setattr(provider, "_get_client", client)
    return UsageTrackingProvider(provider, "deepseek")


@pytest.mark.asyncio
async def test_exhausted_429_keeps_retry_after_and_does_not_become_network_failure(
    setup, monkeypatch
):
    store, _ = setup
    calls = []

    async def post(url, **kwargs):
        calls.append(kwargs)
        return httpx.Response(
            429,
            json={"error": {"message": "rate limit"}},
            request=httpx.Request("POST", url),
        )

    async def exhausted(*args, **kwargs):
        raise RateLimitError("fixture", provider="deepseek", retry_after=77)

    monkeypatch.setattr("src.core.llm.providers.deepseek.handle_rate_limit", exhausted)
    provider = tracked(monkeypatch, post)
    with pytest.raises(RateLimitError) as error:
        await provider.generate("fixture")
    assert error.value.retry_after == 77
    assert len(calls) == len(store.events()) == 1
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "authentication_error"),
        (403, "permission_denied"),
        (404, "model_unavailable"),
        (422, "invalid_request"),
    ],
)
async def test_permanent_failures_do_not_retry(setup, monkeypatch, status, reason):
    store, _ = setup

    async def post(url, **kwargs):
        return httpx.Response(
            status,
            request=httpx.Request("POST", url),
            json={"error": {"message": "fixture"}},
        )

    with pytest.raises(ProviderActionRequiredError) as error:
        await tracked(monkeypatch, post).generate("fixture")
    assert error.value.reason == reason
    assert len(store.events()) == 1


@pytest.mark.asyncio
async def test_empty_billed_attempts_are_charged_and_call_limit_is_shared(
    setup, monkeypatch
):
    store, budget = setup
    budget.configure("experiment", limit_usd="5", max_calls=1)

    async def post(url, **kwargs):
        assert kwargs["json"]["max_tokens"] == 8192
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": ""}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            },
        )

    token = set_usage_context(budget_scope_ids=["experiment"])
    try:
        with pytest.raises(BudgetLimitError) as error:
            await tracked(monkeypatch, post, max_output_tokens=8192).generate("fixture")
        assert error.value.reason == "call_budget_exhausted"
    finally:
        reset_usage_context(token)
    assert len(store.events()) == 1
    assert budget.snapshot("experiment")["settled_usd"] > 0
    assert budget.snapshot("experiment")["reserved_usd"] == 0


@pytest.mark.asyncio
async def test_cancel_keeps_durable_reservation(setup, monkeypatch):
    store, budget = setup
    budget.configure("job", limit_usd="5", max_calls=10)

    async def post(url, **kwargs):
        raise asyncio.CancelledError()

    token = set_usage_context(budget_scope_ids=["job"])
    try:
        with pytest.raises(asyncio.CancelledError):
            await tracked(monkeypatch, post, max_output_tokens=8192).generate("fixture")
    finally:
        reset_usage_context(token)
    assert budget.snapshot("job")["uncertain_calls"] == 1
    assert len(store.events()) == 1


def test_deepseek_bound_uses_peak_prices_and_explicit_output_limit():
    payload = {
        "model": "deepseek-flash",
        "messages": [{"role": "user", "content": "fixture"}],
        "stream": False,
        "max_tokens": 8192,
    }
    amount, metadata = request_bound(
        "deepseek", "https://api.deepseek.com/chat/completions", payload
    )
    assert float(amount) == pytest.approx(0.3244032)
    assert metadata["bound_basis"] == "full_context_window_peak"
    del payload["max_tokens"]
    with pytest.raises(BudgetLimitError):
        request_bound("deepseek", "https://api.deepseek.com/chat/completions", payload)
