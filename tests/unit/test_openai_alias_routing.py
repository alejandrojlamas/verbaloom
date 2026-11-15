"""Legacy reviewer aliases must stay inside the selected provider boundary."""

import httpx
import pytest

from src.core.llm.factory import create_llm_provider
from src.core.llm_client import LLMClient


OFFICIAL = "https://api.openai.com/v1/chat/completions"
COMPATIBLE = "http://127.0.0.1:8080/v1/chat/completions"


@pytest.mark.parametrize("named, expected", [
    ("deepseek-flash", "gpt-6-luna"),
    ("deepseek-v4-flash", "gpt-6-luna"),
    ("deepseek-v4-pro", "gpt-6.1-terra"),
    ("gpt-6-luna", "gpt-6-luna"),
    ("another-model", "another-model"),
])
@pytest.mark.parametrize("endpoint", [OFFICIAL, COMPATIBLE])
def test_initial_model_alias_requires_official_endpoint(named, expected, endpoint):
    provider = create_llm_provider("openai", api_endpoint=endpoint, model=named)

    assert provider.model == (expected if endpoint == OFFICIAL else named)


@pytest.mark.asyncio
@pytest.mark.parametrize("named, expected", [
    ("deepseek-flash", "gpt-6-luna"),
    ("deepseek-v4-flash", "gpt-6-luna"),
    ("deepseek-v4-pro", "gpt-6.1-terra"),
])
@pytest.mark.parametrize("endpoint", [OFFICIAL, COMPATIBLE])
async def test_reviewer_alias_matches_actual_request(monkeypatch, named, expected, endpoint):
    client = LLMClient(provider_type="openai", api_endpoint=endpoint,
                       model="gpt-6.1-terra")
    provider = client._get_provider()
    calls = []

    class Transport:
        async def post(self, url, **kwargs):
            payload = kwargs["json"]
            calls.append(payload)
            return httpx.Response(200, json={
                "model": payload["model"],
                "choices": [{"finish_reason": "stop", "message": {"content": "Accepted."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }, request=httpx.Request("POST", url))

    async def transport():
        return Transport()

    monkeypatch.setattr(provider._wrapped, "_get_client", transport)
    response = await client.make_request("Audit this unit.", named)

    assert response.content == "Accepted."
    assert len(calls) == 1
    assert calls[0]["model"] == (expected if endpoint == OFFICIAL else named)
    assert provider.model == calls[0]["model"]
