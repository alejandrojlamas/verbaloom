from __future__ import annotations

from collections import deque

import httpx
import pytest

import src.core.llm.providers.deepseek as deepseek_module
from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.exceptions import (
    ContentRiskError,
    DeepSeekPeakPricingError,
    InsufficientCreditsError,
    RateLimitError,
)


class _ScriptedClient:
    def __init__(self, events):
        self.events = deque(events)
        self.calls = 0
        self.last_kwargs = None

    async def post(self, *args, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        event = self.events.popleft()
        if isinstance(event, Exception):
            raise event
        return event

    async def get(self, *args, **kwargs):
        return await self.post(*args, **kwargs)


def _response(status: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.deepseek.test/chat/completions")
    return httpx.Response(
        status,
        request=request,
        json=payload or {"error": {"message": f"HTTP {status}"}},
    )


def _success() -> httpx.Response:
    return _response(
        200,
        {
            "choices": [{"message": {"content": "Traducción completa."}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
        },
    )


@pytest.mark.asyncio
async def test_deepseek_peak_guard_prevents_network_request(monkeypatch):
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.com/chat/completions",
    )
    client_requested = False

    async def forbidden_client():
        nonlocal client_requested
        client_requested = True
        raise AssertionError("network client must not be created during peak pricing")

    monkeypatch.setattr(provider, "_get_client", forbidden_client)
    monkeypatch.setattr(
        deepseek_module,
        "get_deepseek_pricing_status",
        lambda: type("Pricing", (), {
            "disabled": True,
            "seconds_until_available": 321,
            "next_available_at_utc": "2026-09-03T04:00:00+00:00",
            "next_available_at_local": "2026-09-02T22:00:00-06:00",
            "display_timezone": "America/Mexico_City",
            "source_url": "https://api-docs.deepseek.com/quick_start/pricing/",
        })(),
    )

    with pytest.raises(DeepSeekPeakPricingError) as exc_info:
        await provider.generate("Translate this complete unit.")

    assert client_requested is False
    assert exc_info.value.retry_after == 323
    assert exc_info.value.provider == "deepseek"
    assert exc_info.value.next_available_at_local.endswith("-06:00")


@pytest.mark.asyncio
async def test_deepseek_peak_guard_does_not_restrict_custom_gateway(monkeypatch):
    client = _ScriptedClient([_success()])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(
        deepseek_module,
        "get_deepseek_pricing_status",
        lambda: (_ for _ in ()).throw(AssertionError("custom gateway must bypass guard")),
    )

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_deepseek_exposes_provider_length_truncation(monkeypatch):
    client = _ScriptedClient([
        _response(
            200,
            {
                "choices": [{
                    "message": {"content": "<TRANSLATION>Salida incompleta"},
                    "finish_reason": "length",
                }],
                "usage": {"prompt_tokens": 50, "completion_tokens": 128},
            },
        )
    ])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.was_truncated is True


@pytest.mark.asyncio
async def test_deepseek_flash_disables_default_thinking_and_uses_exact_usage(monkeypatch):
    client = _ScriptedClient([
        _response(
            200,
            {
                "choices": [{"message": {"content": "Traducción completa."}}],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                    "prompt_tokens_details": {"cached_tokens": 5},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )
    ])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-flash",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    result = await provider.generate("Translate this complete unit.")

    assert client.last_kwargs["json"]["thinking"] == {"type": "disabled"}
    assert result.total_tokens == 26
    assert result.context_used == 26
    assert result.prompt_cache_hit_tokens == 5
    assert result.prompt_cache_miss_tokens == 12
    assert result.reasoning_tokens == 0


def test_deepseek_fallback_catalog_contains_only_current_models():
    provider = DeepSeekProvider(api_key="test-key")

    models = provider._get_fallback_models()

    assert [model["id"] for model in models] == [
        "deepseek-flash",
        "deepseek-v4-pro",
    ]
    assert models[0]["name"] == "DeepSeek V4.1 Flash"


def test_deepseek_usage_total_cannot_be_lower_than_its_components():
    usage = DeepSeekProvider._usage_counts({
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 5,
    })

    assert usage["total"] == 11


@pytest.mark.parametrize(
    "legacy_model",
    [
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek-chat",
        "deepseek-reasoner",
    ],
)
def test_deepseek_migrates_retired_model_ids(legacy_model):
    provider = DeepSeekProvider(api_key="test-key", model=legacy_model)

    assert provider.model == "deepseek-flash"


def test_deepseek_custom_gateway_preserves_its_model_ids():
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-reasoner",
        api_endpoint="https://private-gateway.example/chat/completions",
    )

    assert provider.model == "deepseek-reasoner"


@pytest.mark.asyncio
async def test_deepseek_live_catalog_filters_retired_aliases(monkeypatch):
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://api.deepseek.test/models"),
        json={
            "data": [
                {"id": "deepseek-chat"},
                {"id": "deepseek-v4-pro"},
                {"id": "deepseek-flash"},
                {"id": "deepseek-reasoner"},
            ]
        },
    )
    client = _ScriptedClient([response])
    provider = DeepSeekProvider(api_key="test-key")

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    models = await provider.get_available_models()

    assert [model["id"] for model in models] == [
        "deepseek-flash",
        "deepseek-v4-pro",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("first_event", ["timeout", 429, 503])
async def test_deepseek_retries_timeout_rate_limit_and_server_errors(
    monkeypatch,
    first_event,
):
    request = httpx.Request("POST", "https://api.deepseek.test/chat/completions")
    event = (
        httpx.ReadTimeout("provider timeout", request=request)
        if first_event == "timeout"
        else _response(first_event)
    )
    client = _ScriptedClient([event, _success()])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    async def no_wait_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(deepseek_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(deepseek_module, "handle_rate_limit", no_wait_rate_limit)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_deepseek_insufficient_credits_pauses_without_retry(monkeypatch):
    client = _ScriptedClient([_response(402)])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    with pytest.raises(InsufficientCreditsError) as exc_info:
        await provider.generate("Translate this complete unit.")

    assert isinstance(exc_info.value, RateLimitError)
    assert exc_info.value.retryable is False
    assert exc_info.value.provider == "deepseek"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_deepseek_401_fails_fast_without_wasting_a_retry(monkeypatch):
    """An invalid API key is a permanent failure -- retrying the identical
    request cannot fix it. Before this fix, the 401 branch raised a bare
    ValueError that fell into the generic `except Exception` handler and
    was retried like a transient network error, burning the one extra
    attempt MAX_TRANSLATION_ATTEMPTS=2 budgets for a request that could
    never succeed."""
    client = _ScriptedClient([_response(401), _success()])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(deepseek_module.asyncio, "sleep", no_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is None
    # Only the first (failing) call should have been made -- no retry burned
    # on a request that can never succeed.
    assert client.calls == 1


@pytest.mark.asyncio
async def test_deepseek_empty_choices_is_retried_not_silently_dropped(monkeypatch):
    """A response with an empty/missing `choices` array used to `return None`
    directly, skipping the retry loop entirely even though attempts remained
    -- unlike every other transient failure branch (timeout, 5xx, JSON decode
    error), which all retry. Malformed/empty responses can be transient
    (provider hiccup) and deserve the same retry treatment."""
    empty_choices_response = _response(200, {"choices": [], "usage": {}})
    client = _ScriptedClient([empty_choices_response, _success()])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(deepseek_module.asyncio, "sleep", no_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_deepseek_content_risk_is_not_retried_with_identical_payload(monkeypatch):
    client = _ScriptedClient([
        _response(400, {"error": {"message": "Content Exists Risk"}}),
    ])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    with pytest.raises(ContentRiskError) as exc_info:
        await provider.generate("Translate this complete unit.")

    assert exc_info.value.provider == "deepseek"
    assert exc_info.value.retryable_with_smaller_unit is True
    assert client.calls == 1


@pytest.mark.asyncio
async def test_deepseek_circuit_breaker_fails_fast_without_network_call(monkeypatch):
    """Once the shared RetryManager's circuit breaker has opened (a
    sustained run of plain connectivity failures on this provider
    instance), further calls must fail immediately without even attempting
    the network request -- the whole point of a circuit breaker is to stop
    hammering a provider that is known to be down."""
    client = _ScriptedClient([_success()])  # would succeed if ever called
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    provider._retry_manager._circuit_breaker.failure_threshold = 1
    provider._retry_manager.record_attempt_result(False)
    assert provider._retry_manager.get_circuit_state() == "open"

    result = await provider.generate("Translate this complete unit.")

    assert result is None
    assert client.calls == 0


@pytest.mark.asyncio
async def test_deepseek_transient_failure_records_circuit_breaker_failure(monkeypatch):
    request = httpx.Request("POST", "https://api.deepseek.test/chat/completions")
    client = _ScriptedClient([httpx.ReadTimeout("provider timeout", request=request)])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(deepseek_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(deepseek_module, "MAX_TRANSLATION_ATTEMPTS", 1)

    result = await provider.generate("Translate this complete unit.")

    assert result is None
    assert provider._retry_manager.get_circuit_state() == "closed"
    # One recorded failure out of a threshold of 5 -- not yet open, but
    # the failure must have been counted.
    assert provider._retry_manager._circuit_breaker._failure_count == 1


@pytest.mark.asyncio
async def test_deepseek_success_records_circuit_breaker_success(monkeypatch):
    client = _ScriptedClient([_success()])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    # Pre-seed a failure count that a success should decay (closed-state
    # behaviour of CircuitBreaker.record_success()).
    provider._retry_manager._circuit_breaker._failure_count = 2

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager._circuit_breaker._failure_count == 1


@pytest.mark.asyncio
async def test_deepseek_content_risk_does_not_trip_circuit_breaker(monkeypatch):
    """A content-policy refusal is a per-request/per-account condition, not
    a signal that DeepSeek's connectivity is unhealthy -- it must not count
    against the circuit breaker, or an author writing about a violent scene
    could accidentally trip fail-fast for the rest of the book."""
    client = _ScriptedClient([
        _response(400, {"error": {"message": "Content Exists Risk"}}),
    ])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    with pytest.raises(ContentRiskError):
        await provider.generate("Translate this complete unit.")

    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_deepseek_insufficient_credits_does_not_trip_circuit_breaker(monkeypatch):
    """A billing block is not a connectivity problem either -- it must not
    trip the circuit breaker, which exists to protect against network/
    provider-health issues, not account state."""
    client = _ScriptedClient([_response(402)])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    with pytest.raises(InsufficientCreditsError):
        await provider.generate("Translate this complete unit.")

    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_deepseek_401_does_not_trip_circuit_breaker(monkeypatch):
    """An invalid API key is a credential problem, not a provider-health
    signal -- must not count against the circuit breaker."""
    client = _ScriptedClient([_response(401)])
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    result = await provider.generate("Translate this complete unit.")

    assert result is None
    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0
