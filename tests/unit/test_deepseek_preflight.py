"""The settings access check must not treat offline discovery as authentication."""

import httpx
import pytest

from src.core.llm.preflight import preflight_deepseek
from src.core.llm.providers.deepseek import DeepSeekProvider


@pytest.fixture
def transport(monkeypatch):
    calls, clients = [], []

    def install(response=None, error=None):
        async def handle(request):
            calls.append(request)
            assert request.method == "GET"
            assert str(request.url) == DeepSeekProvider.MODELS_URL
            assert request.headers["Authorization"] == "Bearer YOUR_API_KEY_HERE"
            if error:
                raise error
            return response

        async def client(self):
            self._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            clients.append(self._client)
            return self._client

        monkeypatch.setattr(DeepSeekProvider, "_get_client", client)

    yield install, calls, clients
    assert all(client.is_closed for client in clients)


@pytest.mark.asyncio
async def test_missing_key_never_opens_client(transport):
    install, calls, clients = transport
    install(error=AssertionError("No credentials must mean no provider IO"))
    result = await preflight_deepseek(api_key=" ", models=["deepseek-flash"])
    assert result["status"] == "missing_credentials"
    assert not calls and not clients


@pytest.mark.asyncio
async def test_configured_alias_remains_visible_and_no_generation_or_balance_claim(transport):
    install, calls, _ = transport
    install(httpx.Response(200, json={"data": [{"id": "deepseek-flash"}, {"id": "deepseek-v4-pro"}]}))
    result = await preflight_deepseek(api_key="YOUR_API_KEY_HERE", models=["deepseek-v4-flash", "deepseek-v4-pro"])
    assert result["status"] == "models_visible"
    assert result["models"][0] == {"model": "deepseek-v4-flash", "resolved_model": "deepseek-flash", "status": "model_visible"}
    assert len(calls) == 1
    assert result["generation_tested"] is False and result["balance"] is None
    assert "YOUR_API_KEY_HERE" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(401, "authentication_error"), (403, "permission_denied"),
                                             (429, "rate_limited"), (500, "provider_error")])
async def test_failure_never_returns_fallback_models(transport, status, expected):
    install, calls, _ = transport
    install(httpx.Response(status, json={"error": {"message": "Do not echo provider bodies"}}))
    result = await preflight_deepseek(api_key="YOUR_API_KEY_HERE", models=["deepseek-flash"])
    assert result["status"] == expected
    assert result["models"] == [] and len(calls) == 1
    assert "Do not echo" not in str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [None, [], {}, {"data": None}, {"data": [{}]}, {"data": [{"id": []}]}])
async def test_malformed_catalog_cannot_approve_access(transport, payload):
    install, _, _ = transport
    install(httpx.Response(200, json=payload))
    result = await preflight_deepseek(api_key="YOUR_API_KEY_HERE", models=["deepseek-flash"])
    assert result["status"] == "invalid_response"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectError("No network"), TimeoutError("Bound reached")])
async def test_unavailable_provider_is_bounded_and_not_a_false_positive(transport, error):
    install, calls, _ = transport
    install(error=error)
    result = await preflight_deepseek(api_key="YOUR_API_KEY_HERE", models=["deepseek-flash"])
    assert result["status"] == "connection_error" and len(calls) == 1


@pytest.mark.asyncio
async def test_partial_catalog_is_not_full_role_access(transport):
    install, _, _ = transport
    install(httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]}))
    result = await preflight_deepseek(api_key="YOUR_API_KEY_HERE", models=["deepseek-flash", "deepseek-v4-pro"])
    assert result["status"] == "not_ready"
    assert result["models"][-1]["status"] == "model_unavailable"
