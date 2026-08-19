"""
Regression tests for the OpenRouter provider's shared circuit breaker +
backoff wiring, plus the 401/empty-choices fixes applied in the same pass.
"""
from __future__ import annotations

from collections import deque

import httpx
import pytest

import src.core.llm.providers.openrouter as openrouter_module
from src.core.llm.providers.openrouter import OpenRouterProvider
from src.core.llm.exceptions import ContextOverflowError


class _ScriptedClient:
    def __init__(self, events):
        self.events = deque(events)
        self.calls = 0

    async def post(self, *args, **kwargs):
        self.calls += 1
        event = self.events.popleft()
        if isinstance(event, Exception):
            raise event
        return event


def _response(status: int, payload: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
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
            "cost": 0.0001,
        },
    )


def _provider():
    return OpenRouterProvider(api_key="test-key", model="anthropic/claude-test")


@pytest.mark.asyncio
async def test_openrouter_401_fails_fast_without_wasting_a_retry(monkeypatch):
    client = _ScriptedClient([_response(401), _success()])
    provider = _provider()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    result = await provider.generate("Translate this complete unit.")

    assert result is None
    assert client.calls == 1
    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_openrouter_empty_choices_is_retried_not_silently_dropped(monkeypatch):
    empty_choices_response = _response(200, {"choices": [], "usage": {}})
    client = _ScriptedClient([empty_choices_response, _success()])
    provider = _provider()

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openrouter_module.asyncio, "sleep", no_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2


@pytest.mark.asyncio
async def test_openrouter_circuit_breaker_fails_fast_without_network_call(monkeypatch):
    client = _ScriptedClient([_success()])
    provider = _provider()

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
async def test_openrouter_timeout_retries_with_backoff_and_recovers(monkeypatch):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    client = _ScriptedClient([httpx.ReadTimeout("provider timeout", request=request), _success()])
    provider = _provider()

    async def get_client():
        return client

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openrouter_module.asyncio, "sleep", fake_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert client.calls == 2
    assert len(sleeps) == 1
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_openrouter_context_overflow_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedClient([
        _response(400, {"error": {"message": "maximum context length exceeded"}}),
    ])
    provider = _provider()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)

    with pytest.raises(ContextOverflowError):
        await provider.generate("Translate this complete unit.")

    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_openrouter_rate_limit_continue_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedClient([_response(429), _success()])
    provider = _provider()

    async def get_client():
        return client

    async def no_wait_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openrouter_module, "handle_rate_limit", no_wait_rate_limit)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0
