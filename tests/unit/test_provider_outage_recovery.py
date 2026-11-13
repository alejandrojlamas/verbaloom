"""A provider outage must wait and resume, never fail a healthy unit."""

from __future__ import annotations

import asyncio
from collections import deque

import httpx
import pytest

import src.core.llm.providers.deepseek as deepseek_module
from src.api import handlers
from src.common.provider_outage import (
    note_transport_outage,
    outage_recovery_enabled,
    provider_outage_recovery,
)
from src.core.adapters.retry_manager import RetryManager
from src.core.llm.exceptions import (
    DeepSeekPeakPricingError,
    InsufficientCreditsError,
    ProviderActionRequiredError,
    ProviderUnavailableError,
    RateLimitError,
)
from src.core.llm.providers.deepseek import DeepSeekProvider
from src.core.llm.request_deadline import await_llm_call


class _Provider:
    provider_name = "deepseek"

    def __init__(self, behavior):
        self.behavior = behavior

    async def generate(self, prompt, timeout=0, system_prompt=None):
        return await self.behavior()


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
    request = httpx.Request("POST", "https://api.deepseek.test/chat/completions")
    return httpx.Response(
        status,
        request=request,
        json=payload or {"error": {"message": f"HTTP {status}"}},
    )


async def _no_sleep(_seconds):
    return None


def _deepseek(monkeypatch, events):
    client = _ScriptedClient(events)
    provider = DeepSeekProvider(
        api_key="test-key",
        model="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.test/chat/completions",
    )

    async def get_client():
        return client

    monkeypatch.setattr(provider, "_get_client", get_client)
    monkeypatch.setattr(deepseek_module.asyncio, "sleep", _no_sleep)
    return provider, client


# --------------------------------------------------------------------------
# The seam: await_llm_call
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_outage_is_a_missing_response_outside_the_job_scope():
    async def unreachable():
        note_transport_outage("timeout")
        return None

    provider = _Provider(unreachable)

    assert outage_recovery_enabled() is False
    assert await await_llm_call(provider.generate, "hi", provider=provider) is None


@pytest.mark.asyncio
async def test_outage_becomes_a_retryable_provider_stop_inside_the_job_scope():
    async def unreachable():
        note_transport_outage("timeout")
        return None

    provider = _Provider(unreachable)

    with provider_outage_recovery():
        with pytest.raises(ProviderUnavailableError) as raised:
            await await_llm_call(provider.generate, "hi", provider=provider)

    assert isinstance(raised.value, RateLimitError)
    assert raised.value.retryable is True
    assert raised.value.reason == "timeout"
    assert raised.value.provider == "deepseek"
    assert outage_recovery_enabled() is False


@pytest.mark.asyncio
async def test_bad_model_answer_stays_a_missing_candidate_inside_the_job_scope():
    async def empty_answer():
        return None

    provider = _Provider(empty_answer)

    with provider_outage_recovery():
        assert await await_llm_call(provider.generate, "hi", provider=provider) is None


@pytest.mark.asyncio
async def test_recovered_outage_returns_the_response():
    async def flaky():
        note_transport_outage("timeout")
        return "answer"

    provider = _Provider(flaky)

    with provider_outage_recovery():
        assert await await_llm_call(provider.generate, "hi", provider=provider) == "answer"


@pytest.mark.asyncio
async def test_total_deadline_is_an_outage_inside_the_job_scope():
    async def hangs():
        await asyncio.sleep(10)

    provider = _Provider(hangs)

    with provider_outage_recovery():
        with pytest.raises(ProviderUnavailableError) as raised:
            await await_llm_call(
                provider.generate, "hi", provider=provider, deadline=0.01,
            )

    assert raised.value.reason == "operation_deadline"


@pytest.mark.asyncio
async def test_outage_of_one_operation_does_not_leak_into_the_next():
    calls = []

    async def first_unreachable_then_empty():
        calls.append(1)
        if len(calls) == 1:
            note_transport_outage("timeout")
        return None

    provider = _Provider(first_unreachable_then_empty)

    with provider_outage_recovery():
        with pytest.raises(ProviderUnavailableError):
            await await_llm_call(provider.generate, "hi", provider=provider)
        assert await await_llm_call(provider.generate, "hi", provider=provider) is None


# --------------------------------------------------------------------------
# Who reports an outage
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_open_circuit_breaker_is_reported_as_outage():
    manager = RetryManager(enable_circuit_breaker=True)
    for _ in range(5):
        manager.record_attempt_result(False)

    async def fail_fast():
        assert manager.circuit_allows_attempt() is False
        return None

    provider = _Provider(fail_fast)

    with provider_outage_recovery():
        with pytest.raises(ProviderUnavailableError) as raised:
            await await_llm_call(provider.generate, "hi", provider=provider)

    assert raised.value.reason == "circuit_open"


@pytest.mark.asyncio
async def test_deepseek_timeouts_are_an_outage(monkeypatch):
    request = httpx.Request("POST", "https://api.deepseek.test/chat/completions")
    provider, client = _deepseek(monkeypatch, [
        httpx.ReadTimeout("slow", request=request),
        httpx.ReadTimeout("slow", request=request),
        httpx.ReadTimeout("slow", request=request),
    ])

    # Legacy contract outside a web job: no response.
    assert await provider.generate("Translate this unit.") is None

    provider, client = _deepseek(monkeypatch, [
        httpx.ReadTimeout("slow", request=request),
        httpx.ConnectError("refused", request=request),
        _response(503),
    ])
    with provider_outage_recovery():
        with pytest.raises(ProviderUnavailableError) as raised:
            await await_llm_call(
                provider.generate, "Translate this unit.", provider=provider,
            )

    assert client.calls == 3
    assert raised.value.reason == "http_error"


@pytest.mark.asyncio
async def test_deepseek_malformed_answers_are_not_an_outage(monkeypatch):
    empty = _response(200, {"choices": [], "usage": {}})
    provider, client = _deepseek(monkeypatch, [empty, empty, empty])

    with provider_outage_recovery():
        result = await await_llm_call(
            provider.generate, "Translate this unit.", provider=provider,
        )

    assert result is None
    assert client.calls == 3


@pytest.mark.asyncio
async def test_deepseek_refusal_is_never_reported_as_outage(monkeypatch):
    provider, _client = _deepseek(monkeypatch, [
        _response(400, {"error": {"message": "Content Exists Risk"}}),
    ])

    with provider_outage_recovery():
        with pytest.raises(ProviderActionRequiredError) as raised:
            await await_llm_call(
                provider.generate, "Translate this unit.", provider=provider,
            )

    assert raised.value.reason == "provider_refusal"
    assert not isinstance(raised.value, ProviderUnavailableError)


# --------------------------------------------------------------------------
# The job boundary
# --------------------------------------------------------------------------

def test_outage_always_resumes_even_when_rate_limits_pause_the_job():
    outage = ProviderUnavailableError("down", reason="timeout", provider="deepseek")
    manual = {"auto_pause_on_rate_limit": True}

    assert handlers._provider_stop_pauses_job(outage, manual) is False
    assert handlers._provider_stop_pauses_job(RateLimitError("429"), manual) is True
    assert handlers._provider_stop_pauses_job(
        RateLimitError("429"), {"auto_pause_on_rate_limit": False},
    ) is False


def test_stops_that_waiting_cannot_fix_still_pause_the_job():
    resume = {"auto_pause_on_rate_limit": False}

    assert handlers._provider_stop_pauses_job(
        InsufficientCreditsError("no balance", provider="deepseek"), resume,
    ) is True
    assert handlers._provider_stop_pauses_job(
        ProviderActionRequiredError("refused", reason="provider_refusal", provider="deepseek"),
        resume,
    ) is True
    pricing = DeepSeekPeakPricingError(
        retry_after=60,
        next_available_at_utc="",
        next_available_at_local="",
        display_timezone="America/Mexico_City",
        source_url="",
    )
    assert handlers._provider_stop_pauses_job(pricing, {"auto_pause_on_rate_limit": True}) is False


def test_outage_wait_uses_the_finite_provider_recovery_budget():
    config = {"max_rate_limit_auto_resumes": 2}

    first = handlers._build_rate_limit_auto_resume_plan(
        config, resume_index=4, total_units=40,
    )
    second = handlers._build_rate_limit_auto_resume_plan(
        first["config"], resume_index=4, total_units=40,
    )
    third = handlers._build_rate_limit_auto_resume_plan(
        second["config"], resume_index=4, total_units=40,
    )

    assert first["allowed"] and second["allowed"]
    assert first["delay_seconds"] >= 60
    assert third["allowed"] is False
    assert third["reason"] == "rate_limit_recovery_exhausted"


@pytest.mark.asyncio
async def test_phase_wrapper_enables_recovery_only_for_the_phase():
    seen = []

    async def phase(value):
        seen.append(outage_recovery_enabled())
        return value

    result = await handlers._with_provider_outage_recovery(phase)("done")

    assert result == "done"
    assert seen == [True]
    assert outage_recovery_enabled() is False
    assert handlers._with_provider_outage_recovery(lambda: outage_recovery_enabled())


# --------------------------------------------------------------------------
# Provider-wait ceilings
# --------------------------------------------------------------------------

def _waits_until_stop(config, *, resume_index=4):
    waits = []
    while True:
        plan = handlers._build_rate_limit_auto_resume_plan(
            config, resume_index=resume_index, total_units=400,
        )
        if not plan["allowed"]:
            return waits, plan
        waits.append(plan["delay_seconds"])
        config = plan["config"]


def test_default_ceilings_ride_out_a_long_outage():
    waits, stop = _waits_until_stop({})

    assert len(waits) == 12
    assert stop["reason"] == "rate_limit_recovery_exhausted"
    # 60 s, 180 s, 600 s, then 30 min each: more than four hours in total.
    assert sum(waits) > 4 * 3600
    assert max(waits) <= 1800 * 1.2


def test_ceilings_frozen_by_an_older_release_are_upgraded_once():
    legacy = {
        "max_rate_limit_auto_resumes": 5,
        "max_provider_wait_total_seconds": 14400,
        "_auto_resume_last_index": 4,
        "_auto_resume_stuck_count": 5,
        "_provider_wait_total_seconds": 14000.0,
    }

    plan = handlers._build_rate_limit_auto_resume_plan(
        legacy, resume_index=4, total_units=400,
    )

    assert plan["allowed"] is True
    assert plan["config"]["max_rate_limit_auto_resumes"] == 12
    assert plan["config"]["max_provider_wait_total_seconds"] == 86400
    assert plan["config"]["_provider_wait_policy_version"] == 2


def test_other_saved_ceilings_are_obeyed():
    waits, stop = _waits_until_stop({"max_rate_limit_auto_resumes": 3})
    assert len(waits) == 3

    waits, stop = _waits_until_stop(
        {"max_provider_wait_total_seconds": 100}, resume_index=4,
    )
    # One 60 s wait fits; the next one would exceed the saved total.
    assert len(waits) == 1
    assert stop["reason"] == "provider_wait_budget_exhausted"

    # A ceiling saved under the current policy version is never rewritten.
    current = {"max_rate_limit_auto_resumes": 5, "_provider_wait_policy_version": 2}
    waits, _stop = _waits_until_stop(current)
    assert len(waits) == 5
