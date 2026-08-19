"""
Regression tests for the Gemini provider's shared circuit breaker +
backoff wiring (see src/core/llm/base.py's LLMProvider.__init__ and
src/core/adapters/retry_manager.py).
"""
from __future__ import annotations

from collections import deque

import httpx
import pytest

import src.core.llm.providers.gemini as gemini_module
from src.core.llm.providers.gemini import GeminiProvider
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
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent")
    return httpx.Response(
        status,
        request=request,
        json=payload or {"error": {"message": f"HTTP {status}"}},
    )


def _success() -> httpx.Response:
    return _response(
        200,
        {
            "candidates": [{
                "content": {"parts": [{"text": "Traducción completa."}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 3},
        },
    )


def _provider():
    return GeminiProvider(api_key="test-key", model="gemini-test")


@pytest.mark.asyncio
async def test_gemini_circuit_breaker_fails_fast_without_network_call(monkeypatch):
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
async def test_gemini_timeout_retries_with_backoff_and_recovers(monkeypatch):
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent")
    client = _ScriptedClient([httpx.ReadTimeout("provider timeout", request=request), _success()])
    provider = _provider()

    async def get_client():
        return client

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(gemini_module.asyncio, "sleep", fake_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2
    assert len(sleeps) == 1
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_gemini_success_records_circuit_breaker_success(monkeypatch):
    client = _ScriptedClient([_success()])
    provider = _provider()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    provider._retry_manager._circuit_breaker._failure_count = 2

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager._circuit_breaker._failure_count == 1


@pytest.mark.asyncio
async def test_gemini_context_overflow_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedClient([
        _response(400, {"error": {"message": "Request payload size exceeds the limit: token limit"}}),
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
async def test_gemini_rate_limit_continue_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedClient([_response(429), _success()])
    provider = _provider()

    async def get_client():
        return client

    async def no_wait_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(gemini_module, "handle_rate_limit", no_wait_rate_limit)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0
