"""
Regression tests for the Ollama provider's shared circuit breaker +
backoff wiring (see src/core/llm/base.py's LLMProvider.__init__ and
src/core/adapters/retry_manager.py).

Ollama streams responses via client.stream(...), so the scripted client
here is a bit different from the plain client.post(...) providers: it
returns an async context manager per call, mimicking httpx's streaming API.
"""
from __future__ import annotations

import json
from collections import deque

import httpx
import pytest

import src.core.llm.providers.ollama as ollama_module
from src.core.llm.providers.ollama import OllamaProvider
from src.core.llm.exceptions import ContextOverflowError


class _FakeStreamResponse:
    def __init__(self, lines, status_code=200, error_message="boom"):
        self._lines = lines
        self.status_code = status_code
        self._error_message = error_message

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://ollama.test/api/chat")
            response = httpx.Response(
                self.status_code, request=request, json={"error": self._error_message}
            )
            raise httpx.HTTPStatusError("error", request=request, response=response)

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        pass


class _FakeStreamContextManager:
    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc

    async def __aenter__(self):
        if self._raise_exc:
            raise self._raise_exc
        return self._response

    async def __aexit__(self, *args):
        return False


class _ScriptedStreamClient:
    def __init__(self, events):
        self.events = deque(events)
        self.calls = 0

    def stream(self, method, url, json=None, timeout=None):
        self.calls += 1
        event = self.events.popleft()
        if isinstance(event, Exception):
            return _FakeStreamContextManager(raise_exc=event)
        return _FakeStreamContextManager(response=event)


def _success_response() -> _FakeStreamResponse:
    line = json.dumps({
        "message": {"content": "Traducción completa."},
        "prompt_eval_count": 8,
        "eval_count": 3,
        "done": True,
    })
    return _FakeStreamResponse([line])


def _provider():
    return OllamaProvider(
        api_endpoint="http://ollama.test/api/chat",
        model="llama-test",
    )


@pytest.mark.asyncio
async def test_ollama_circuit_breaker_fails_fast_without_network_call(monkeypatch):
    client = _ScriptedStreamClient([_success_response()])
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
async def test_ollama_timeout_retries_and_records_failure_then_succeeds(monkeypatch):
    client = _ScriptedStreamClient([
        httpx.ReadTimeout("provider timeout", request=httpx.Request("POST", "http://ollama.test/api/chat")),
        _success_response(),
    ])
    provider = _provider()

    async def get_client():
        return client

    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(ollama_module.asyncio, "sleep", fake_sleep)

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert result.content == "Traducción completa."
    assert client.calls == 2
    assert len(sleeps) == 1
    # The success decayed the one recorded failure back down.
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
async def test_ollama_context_overflow_does_not_trip_circuit_breaker(monkeypatch):
    client = _ScriptedStreamClient([
        _FakeStreamResponse(
            [], status_code=400, error_message="context length exceeded, please reduce the length"
        ),
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
async def test_ollama_success_records_circuit_breaker_success(monkeypatch):
    client = _ScriptedStreamClient([_success_response()])
    provider = _provider()

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    provider._retry_manager._circuit_breaker._failure_count = 2

    result = await provider.generate("Translate this complete unit.")

    assert result is not None
    assert provider._retry_manager._circuit_breaker._failure_count == 1
