import requests
from flask import Flask
from types import SimpleNamespace

from src.api.blueprints import config_routes
from src.api.blueprints.config_routes import create_config_blueprint
from src.api.blueprints.provider_model_routes import ProviderModelCatalog


class FakeResponse:
    def __init__(self, status_code, payload, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def _client():
    app = Flask(__name__)
    app.register_blueprint(create_config_blueprint(server_session_id=1234567890))
    return app.test_client()


def test_deepseek_model_listing_reports_missing_key_without_network(monkeypatch):
    monkeypatch.setattr(config_routes._config, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(config_routes._config, "DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with _client() as client:
        payload = client.post("/api/models", json={"provider": "deepseek"}).get_json()

    assert payload == {
        "models": [],
        "model_names": [],
        "default": "deepseek-v4-pro",
        "status": "api_key_missing",
        "count": 0,
        "error": (
            "DeepSeek API key is required. Set DEEPSEEK_API_KEY "
            "environment variable or pass api_key parameter."
        ),
    }


def test_cloud_catalog_uses_configured_default_and_provider_contract(monkeypatch):
    from src.core import llm

    calls = {}

    class FakeDeepSeekProvider:
        def __init__(self, api_key):
            calls["api_key"] = api_key

        async def get_available_models(self):
            return [
                {"id": "deepseek-chat", "name": "Chat"},
                {"id": "deepseek-v4-pro", "name": "Pro"},
            ]

    monkeypatch.setattr(llm, "DeepSeekProvider", FakeDeepSeekProvider)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    catalog = ProviderModelCatalog(
        SimpleNamespace(
            DEEPSEEK_API_KEY="configured-key",
            DEEPSEEK_MODEL="deepseek-v4-pro",
        )
    )

    payload = catalog.list_models("deepseek")

    assert calls["api_key"] == "configured-key"
    assert payload["status"] == "deepseek_connected"
    assert payload["model_names"] == ["deepseek-chat", "deepseek-v4-pro"]
    assert payload["default"] == "deepseek-v4-pro"


def test_gemini_missing_key_keeps_legacy_error_shape(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    catalog = ProviderModelCatalog(
        SimpleNamespace(GEMINI_API_KEY="", GEMINI_MODEL="")
    )

    payload = catalog.list_models("gemini")

    assert payload["status"] == "api_key_missing"
    assert payload["default"] == "gemini-2.0-flash"
    assert "model_names" not in payload


def test_openai_compatible_listing_filters_non_chat_models_and_sorts(monkeypatch):
    requested = {}

    def fake_get(url, *, headers, timeout):
        requested.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(
            200,
            {
                "data": [
                    {"id": "zeta-chat", "owned_by": "local"},
                    {"id": "text-embedding-3-small", "owned_by": "local"},
                    {"id": "whisper-1", "owned_by": "local"},
                    {"id": "alpha-chat", "owned_by": "local"},
                ]
            },
        )

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(config_routes._config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config_routes._config, "DEFAULT_MODEL", "not-on-server")

    with _client() as client:
        payload = client.post(
            "/api/models",
            json={
                "provider": "openai",
                "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            },
        ).get_json()

    assert requested["url"] == "http://127.0.0.1:1234/v1/models"
    assert requested["timeout"] == 10
    assert payload["status"] == "openai_connected"
    assert payload["model_names"] == ["alpha-chat", "zeta-chat"]
    assert payload["default"] == "alpha-chat"


def test_openai_custom_endpoint_failure_never_returns_cloud_fallback(monkeypatch):
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(503, {}, text="local server unavailable"),
    )

    with _client() as client:
        payload = client.post(
            "/api/models",
            json={
                "provider": "openai",
                "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            },
        ).get_json()

    assert payload["status"] == "openai_error"
    assert payload["models"] == []
    assert payload["default"] is None
    assert payload["endpoint"] == "http://127.0.0.1:1234/v1"


def test_openai_custom_endpoint_never_receives_saved_key(monkeypatch):
    requested = {}

    def fake_get(url, *, headers, timeout):
        requested.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(200, {"data": [{"id": "local-chat"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("OPENAI_API_KEY", "saved-environment-secret")

    with _client() as client:
        payload = client.post(
            "/api/models",
            json={
                "provider": "openai",
                "api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
            },
        ).get_json()

    assert payload["status"] == "openai_connected"
    assert "Authorization" not in requested["headers"]


def test_openai_custom_endpoint_rejects_environment_key_sentinel(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "saved-environment-secret")

    with _client() as client:
        payload = client.post(
            "/api/models",
            json={
                "provider": "openai",
                "api_key": "__USE_ENV__",
                "api_endpoint": "https://gateway.example.test/v1/chat/completions",
            },
        ).get_json()

    assert payload["status"] == "unsafe_api_key_routing"
    assert payload["models"] == []


def test_nim_listing_filters_non_chat_and_base_models(monkeypatch):
    monkeypatch.setattr(config_routes._config, "NIM_API_KEY", "key-one,key-two")
    monkeypatch.setenv("NIM_API_KEY", "key-one,key-two")
    monkeypatch.setattr(
        config_routes._config,
        "NIM_API_ENDPOINT",
        "https://integrate.api.nvidia.com/v1/chat/completions",
    )
    monkeypatch.setattr(config_routes._config, "NIM_MODEL", "missing-default")
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: FakeResponse(
            200,
            {
                "data": [
                    {"id": "nvidia/embed-v1"},
                    {"id": "google/gemma-2b"},
                    {"id": "vendor/codegemma-chat"},
                    {"id": "vendor/general-chat", "owned_by": "vendor"},
                ]
            },
        ),
    )

    with _client() as client:
        payload = client.post(
            "/api/models",
            json={"provider": "nim", "api_key": "__USE_ENV__"},
        ).get_json()

    assert payload["status"] == "nim_connected"
    assert payload["model_names"] == ["vendor/general-chat"]
    assert payload["default"] == "vendor/general-chat"


def test_nim_custom_configured_endpoint_never_receives_saved_key(monkeypatch):
    requested = {}

    def fake_get(url, *, headers, timeout):
        requested.update(url=url, headers=headers, timeout=timeout)
        return FakeResponse(200, {"data": [{"id": "vendor/general-chat"}]})

    custom_endpoint = "https://gateway.example.test/v1/chat/completions"
    monkeypatch.setattr(config_routes._config, "NIM_API_KEY", "saved-config-secret")
    monkeypatch.setenv("NIM_API_KEY", "saved-environment-secret")
    monkeypatch.setattr(config_routes._config, "NIM_API_ENDPOINT", custom_endpoint)
    monkeypatch.setattr(config_routes._config, "NIM_MODEL", "vendor/general-chat")
    monkeypatch.setattr(requests, "get", fake_get)

    with _client() as client:
        payload = client.post("/api/models", json={"provider": "nim"}).get_json()

    assert payload["status"] == "api_key_missing"
    assert requested == {}


def test_ollama_listing_normalizes_generate_endpoint_to_tags(monkeypatch):
    requested = {}

    def fake_get(url, *, timeout):
        requested.update(url=url, timeout=timeout)
        return FakeResponse(200, {"models": [{"name": "qwen:latest"}, {"name": "llama:latest"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(config_routes._config, "DEFAULT_MODEL", "llama:latest")

    with _client() as client:
        payload = client.get(
            "/api/models?provider=ollama&api_endpoint=http://127.0.0.1:11434/api/generate"
        ).get_json()

    assert requested["url"] == "http://127.0.0.1:11434/api/tags"
    assert payload == {
        "models": ["qwen:latest", "llama:latest"],
        "default": "llama:latest",
        "status": "ollama_connected",
        "count": 2,
    }
