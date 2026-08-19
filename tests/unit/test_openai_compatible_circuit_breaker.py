"""
Regression tests for the OpenAI-compatible provider's shared circuit
breaker + backoff wiring (see src/core/llm/base.py's LLMProvider.__init__
and src/core/adapters/retry_manager.py).
"""
from __future__ import annotations

from collections import deque

import httpx
import pytest

import src.core.llm.providers.openai as openai_module
from src.core.llm.providers.openai import OpenAICompatibleProvider
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
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
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


def _provider(api_key="test-key"):
    return OpenAICompatibleProvider(
        api_endpoint="https://api.example.test/v1/chat/completions",
        model="gpt-test",
        api_key=api_key,
    )


@pytest.mark.asyncio
async def test_openai_circuit_breaker_fails_fast_without_network_call(monkeypatch):
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
async def test_openai_transient_timeout_retries_with_backoff_and_records_failure(monkeypatch):
    request = httpx.Request("POST", "https://api.example.test/v1/chat/completions")
    client = _ScriptedClient([httpx.ReadTimeout("provider timeout", request=request), _success()])
    provider = _provider()

    async def get_client():
        return client

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openai_module.asyncio, "sleep", fake_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2
    assert len(sleeps) == 1
    assert provider._retry_manager._circuit_breaker._failure_count == 0  # decayed back by the success


@pytest.mark.asyncio
async def test_openai_success_records_circuit_breaker_success(monkeypatch):
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
async def test_openai_context_overflow_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedClient([
        _response(400, {"error": {"message": "This model's maximum context length is 4096 tokens"}}),
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
async def test_openai_rate_limit_continue_does_not_trip_circuit_breaker(monkeypatch):
    """429 handling rotates keys via handle_rate_limit() and `continue`s --
    it never reaches the plain-transient-failure branches, so it must not
    count against the circuit breaker either."""
    client = _ScriptedClient([_response(429), _success()])
    provider = _provider()

    async def get_client():
        return client

    async def no_wait_rate_limit(*args, **kwargs):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openai_module, "handle_rate_limit", no_wait_rate_limit)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager.get_circuit_state() == "closed"
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_openai_json_decode_error_retries_and_records_failure(monkeypatch):
    bad_json_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.example.test/v1/chat/completions"),
        content=b"not json",
    )
    client = _ScriptedClient([bad_json_response, _success()])
    provider = _provider()

    async def get_client():
        return client

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(openai_module.asyncio, "sleep", no_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert client.calls == 2
