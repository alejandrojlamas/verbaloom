from __future__ import annotations

import asyncio

import pytest

from src.core.llm.request_deadline import (
    await_llm_call,
    operation_deadline_seconds,
)


class DeepSeekLikeProvider:
    provider_name = "deepseek"

    async def generate(self, prompt, timeout=0, system_prompt=None):
        await asyncio.sleep(10)


class LegacyProvider:
    async def generate(self, prompt, system_prompt=None):
        return prompt, system_prompt


@pytest.mark.asyncio
async def test_total_deadline_cancels_provider_retry_coroutine():
    provider = DeepSeekLikeProvider()

    with pytest.raises(asyncio.TimeoutError):
        await await_llm_call(
            provider.generate,
            "hello",
            provider=provider,
            request_timeout=30,
            deadline=0.01,
            system_prompt="system",
        )


@pytest.mark.asyncio
async def test_legacy_provider_without_timeout_keyword_still_works():
    provider = LegacyProvider()

    result = await await_llm_call(
        provider.generate,
        "hello",
        provider=provider,
        request_timeout=30,
        system_prompt="system",
    )

    assert result == ("hello", "system")


def test_cloud_operation_budget_is_not_multiplied_by_request_timeout(monkeypatch):
    provider = DeepSeekLikeProvider()
    monkeypatch.setattr(
        "src.core.llm.request_deadline.LLM_CLOUD_OPERATION_TIMEOUT",
        45,
    )

    assert operation_deadline_seconds(provider, request_timeout=900) == 45

