"""Preflight uses the existing client, official host, and no generation calls."""

import httpx
import pytest

from src.core.llm.preflight import preflight_openai
from src.core.llm.providers.openai import OpenAICompatibleProvider


@pytest.mark.asyncio
async def test_missing_key_does_not_contact_provider(monkeypatch):
    async def forbidden(_self):
        raise AssertionError("No credentials must mean no network")
    monkeypatch.setattr(OpenAICompatibleProvider, "_get_client", forbidden)
    report = await preflight_openai(api_key="")
    assert report["status"] == "missing_credentials"
    assert report["balance"] is None
    assert report["generation_tested"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(200, "models_visible"), (401, "not_ready"), (404, "not_ready")])
async def test_preflight_only_gets_official_model_metadata(monkeypatch, status, expected):
    calls = []
    async def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == "api.openai.com"
        return httpx.Response(status, json={"id": request.url.path.split("/")[-1]})
    async def client(self):
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return self._client
    monkeypatch.setattr(OpenAICompatibleProvider, "_get_client", client)
    report = await preflight_openai(api_key="YOUR_API_KEY_HERE")
    assert report["status"] == expected
    assert len(calls) == 2
    assert "YOUR_API_KEY_HERE" not in str(report)
    assert report["balance"] is None
    assert report["generation_tested"] is False
