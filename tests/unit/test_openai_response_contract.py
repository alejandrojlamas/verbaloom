"""HTTP-boundary regressions; all provider traffic is simulated."""

import asyncio
import json

import httpx
import pytest

from src.core.llm.exceptions import (
    InsufficientCreditsError,
    ProviderActionRequiredError,
)
from src.core.llm.providers.openai import OpenAICompatibleProvider
from src.core.usage.store import TokenUsageStore
from src.core.usage.tracking_provider import UsageTrackingProvider


@pytest.fixture
def provider_factory(monkeypatch):
    async def no_sleep(_seconds):
        pass

    monkeypatch.setattr("src.core.llm.providers.openai.asyncio.sleep", no_sleep)

    def make(body, *, status=200, endpoint="https://api.openai.com/v1/chat/completions", **options):
        calls = []

        class Client:
            async def post(self, url, **kwargs):
                calls.append(kwargs)
                return httpx.Response(status, json=body, headers={"x-request-id": "request-fixture"},
                                      request=httpx.Request("POST", url))

        provider = OpenAICompatibleProvider(endpoint, "gpt-6-luna", **options)

        async def client():
            return Client()

        monkeypatch.setattr(provider, "_get_client", client)
        return provider, calls

    return make


def completion(reason="stop", content="<Translated>Complete.</Translated>"):
    return {"id": "completion-fixture", "model": "gpt-6-luna", "choices": [
        {"finish_reason": reason, "message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140,
                  "prompt_tokens_details": {"cached_tokens": 30},
                  "completion_tokens_details": {"reasoning_tokens": 10}}}


@pytest.mark.asyncio
async def test_length_finish_is_not_a_complete_translation(provider_factory):
    provider, calls = provider_factory(completion("length"))
    result = await provider.generate("Source")
    assert result.was_truncated is True
    assert result.finish_reason == "length"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_preserves_usage_and_request_identity(provider_factory):
    provider, _ = provider_factory(completion())
    result = await provider.generate("Source")
    assert result.prompt_cache_hit_tokens == 30
    assert result.prompt_cache_miss_tokens == 70
    assert result.reasoning_tokens == 10
    assert result.completion_tokens == 40  # reasoning is a subset, not an addition
    assert result.total_tokens == 140
    assert result.raw_usage == completion()["usage"]
    assert result.request_id == "request-fixture"
    assert result.response_id == "completion-fixture"
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_official_luna_defaults_to_explicit_none(provider_factory):
    provider, calls = provider_factory(completion())
    await provider.generate("Source")
    payload = calls[0]["json"]
    assert payload["reasoning_effort"] == "none"
    assert payload["max_completion_tokens"] > 0
    assert "thinking" not in payload


@pytest.mark.asyncio
async def test_low_reasoning_omits_sampling_parameters(provider_factory):
    provider, calls = provider_factory(completion(), reasoning_effort="low", max_output_tokens=2048)
    await provider.generate("Source", temperature=0.2)
    assert calls[0]["json"]["reasoning_effort"] == "low"
    assert calls[0]["json"]["max_completion_tokens"] == 2048
    assert "temperature" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_none_reasoning_accepts_temperature(provider_factory):
    provider, calls = provider_factory(completion(), reasoning_effort="none")
    await provider.generate("Source", temperature=0.2)
    assert calls[0]["json"]["temperature"] == 0.2


@pytest.mark.asyncio
@pytest.mark.parametrize("status,code", [(401, "invalid_api_key"), (403, "permission_denied"),
    (404, "model_not_found"), (400, "unsupported_parameter"), (429, "insufficient_quota")])
async def test_permanent_errors_do_not_retry(provider_factory, status, code):
    provider, calls = provider_factory({"error": {"code": code, "message": "Invalid configuration"}},
                                       status=status)
    with pytest.raises((InsufficientCreditsError, ProviderActionRequiredError)) as raised:
        await provider.generate("Source")
    assert raised.value.retryable is False
    assert len(calls) == 1
    assert provider._retry_manager._circuit_breaker._failure_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,refusal", [("content_filter", None), ("stop", "Provider declined")])
async def test_refusal_is_not_translated_or_retried(provider_factory, reason, refusal):
    body = completion(reason, None)
    body["choices"][0]["message"]["refusal"] = refusal
    provider, calls = provider_factory(body)
    with pytest.raises(ProviderActionRequiredError) as raised:
        await provider.generate("Source")
    assert raised.value.retryable is False
    assert raised.value.reason == "provider_refusal"
    assert raised.value.response.raw_usage == body["usage"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_success_is_not_approved(provider_factory):
    provider, calls = provider_factory(completion(content=""))
    assert await provider.generate("Source") is None
    assert len(calls) <= 3


@pytest.mark.asyncio
async def test_official_response_requires_finish_reason(provider_factory):
    provider, _ = provider_factory(completion(reason=None))
    assert await provider.generate("Source") is None


@pytest.mark.asyncio
async def test_translate_text_does_not_extract_a_truncated_response(provider_factory):
    provider, _ = provider_factory(completion("length"))
    assert await provider.translate_text("Source") is None


@pytest.mark.asyncio
async def test_documented_context_is_available_without_network(provider_factory):
    provider, calls = provider_factory(completion())
    assert await provider.get_model_context_size() == 1_050_000
    assert calls == []


def test_official_endpoint_requires_exact_https_origin():
    provider = OpenAICompatibleProvider("https://api.openai.com.attacker.invalid/v1", "gpt-6-luna")
    assert provider._is_official_openai_endpoint() is False
    provider = OpenAICompatibleProvider("http://api.openai.com/v1", "gpt-6-luna")
    assert provider._is_official_openai_endpoint() is False


@pytest.mark.asyncio
async def test_compatible_server_does_not_receive_gpt6_parameters(provider_factory):
    provider, calls = provider_factory(completion(), endpoint="http://127.0.0.1:8080/v1")
    await provider.generate("Source")
    assert "reasoning_effort" not in calls[0]["json"]
    assert "max_completion_tokens" not in calls[0]["json"]


@pytest.mark.asyncio
async def test_each_empty_billed_attempt_is_recorded_once(provider_factory, tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    provider, calls = provider_factory(completion(content=""))
    assert await UsageTrackingProvider(provider, "openai").generate("Private fixture source") is None
    events = store.events()
    assert len(events) == len(calls) == 3
    assert sum(row["prompt_tokens"] for row in events) == 300
    assert sum(row["completion_tokens"] for row in events) == 120
    metadata = [json.loads(row["metadata"]) for row in events]
    assert sorted(item["attempt"] for item in metadata) == [1, 2, 3]
    assert len({item["operation_id"] for item in metadata}) == 1
    assert all(item["scope"] == "http_attempt" for item in metadata)
    assert "Private fixture source" not in str(events)


@pytest.mark.asyncio
async def test_usage_survives_a_malformed_choice(provider_factory, tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    body = completion()
    body["choices"] = []
    provider, calls = provider_factory(body)
    assert await UsageTrackingProvider(provider, "openai").generate("fixture") is None
    events = store.events()
    assert len(events) == len(calls) == 3
    assert sum(row["total_tokens"] for row in events) == 420
    assert all(json.loads(row["metadata"])["cost_basis"] == "provider_usage" for row in events)


@pytest.mark.asyncio
async def test_refused_usage_is_not_lost_or_double_counted(provider_factory, tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    provider, _ = provider_factory(completion("content_filter", ""))
    with pytest.raises(ProviderActionRequiredError):
        await UsageTrackingProvider(provider, "openai").generate("Source")
    events = store.events()
    assert len(events) == 1
    assert events[0]["status"] == "refused"
    assert events[0]["total_tokens"] == 140
    metadata = json.loads(events[0]["metadata"])
    assert metadata["reason"] == "provider_refusal"
    assert metadata["raw_usage"] == completion()["usage"]


@pytest.mark.asyncio
async def test_cancelled_request_records_unknown_billing(tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    started = asyncio.Event()
    class Client:
        async def post(self, *args, **kwargs):
            started.set()
            await asyncio.Future()
    provider = OpenAICompatibleProvider("https://api.openai.com/v1", "gpt-6-luna")
    async def client():
        return Client()
    monkeypatch.setattr(provider, "_get_client", client)
    task = asyncio.create_task(UsageTrackingProvider(provider, "openai").generate("Source"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    events = store.events()
    assert len(events) == 1
    metadata = json.loads(events[0]["metadata"])
    assert metadata["billing_uncertain"] is True
    assert metadata["usage_unknown"] is True
    assert metadata["error"] == "cancelled_in_flight"


@pytest.mark.asyncio
async def test_concurrent_calls_keep_attempt_context_separate(provider_factory, tmp_path, monkeypatch):
    store = TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr("src.core.usage.tracking_provider.default_usage_store", lambda: store)
    first, _ = provider_factory(completion())
    second, _ = provider_factory(completion())
    await asyncio.gather(UsageTrackingProvider(first, "openai").generate("Source one"),
                         UsageTrackingProvider(second, "openai").generate("Source two"))
    events = store.events()
    assert len(events) == 2
    assert len({json.loads(row["metadata"])["operation_id"] for row in events}) == 2
