"""Provider model discovery and the ``/api/models`` route."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import requests
from flask import Blueprint, jsonify, request

import src.config as default_config
from src.core.deepseek_pricing import get_deepseek_pricing_status
from src.core.llm.base import normalize_api_keys
from src.utils.provider_security import (
    EndpointCredentialError,
    resolve_api_key_for_endpoint,
)


@dataclass(frozen=True)
class CloudProviderSpec:
    """Configuration for providers exposing ``get_available_models``."""

    env_var: str
    config_key_attr: str
    config_model_attr: str
    fallback_model: str
    status_prefix: str
    display_name: str
    api_key_missing_message: str
    model_name_field: str = "id"
    include_model_names_on_error: bool = True
    get_models_kwargs: dict[str, Any] = field(default_factory=dict)


OPENROUTER_SPEC = CloudProviderSpec(
    env_var="OPENROUTER_API_KEY",
    config_key_attr="OPENROUTER_API_KEY",
    config_model_attr="OPENROUTER_MODEL",
    fallback_model="anthropic/claude-sonnet-4",
    status_prefix="openrouter",
    display_name="OpenRouter",
    api_key_missing_message=(
        "OpenRouter API key is required. Set OPENROUTER_API_KEY "
        "environment variable or pass api_key parameter."
    ),
    get_models_kwargs={"text_only": True},
)

MISTRAL_SPEC = CloudProviderSpec(
    env_var="MISTRAL_API_KEY",
    config_key_attr="MISTRAL_API_KEY",
    config_model_attr="MISTRAL_MODEL",
    fallback_model="mistral-large-latest",
    status_prefix="mistral",
    display_name="Mistral",
    api_key_missing_message=(
        "Mistral API key is required. Set MISTRAL_API_KEY "
        "environment variable or pass api_key parameter."
    ),
)

DEEPSEEK_SPEC = CloudProviderSpec(
    env_var="DEEPSEEK_API_KEY",
    config_key_attr="DEEPSEEK_API_KEY",
    config_model_attr="DEEPSEEK_MODEL",
    fallback_model="deepseek-v4-pro",
    status_prefix="deepseek",
    display_name="DeepSeek",
    api_key_missing_message=(
        "DeepSeek API key is required. Set DEEPSEEK_API_KEY "
        "environment variable or pass api_key parameter."
    ),
)

POE_SPEC = CloudProviderSpec(
    env_var="POE_API_KEY",
    config_key_attr="POE_API_KEY",
    config_model_attr="POE_MODEL",
    fallback_model="Claude-Sonnet-4",
    status_prefix="poe",
    display_name="Poe",
    api_key_missing_message="Poe API key is required. Get your key at https://poe.com/api_key",
)

GEMINI_SPEC = CloudProviderSpec(
    env_var="GEMINI_API_KEY",
    config_key_attr="GEMINI_API_KEY",
    config_model_attr="GEMINI_MODEL",
    fallback_model="gemini-2.0-flash",
    status_prefix="gemini",
    display_name="Gemini",
    api_key_missing_message=(
        "Gemini API key is required. Set GEMINI_API_KEY "
        "environment variable or pass api_key parameter."
    ),
    model_name_field="name",
    include_model_names_on_error=False,
)

NIM_NON_CHAT_KEYWORDS = (
    "embed",
    "rerank",
    "bge",
    "arctic-embed",
    "vision",
    "vlm",
    "-vl-",
    "-vl",
    "clip",
    "neva",
    "vila",
    "fuyu",
    "deplot",
    "paligemma",
    "kosmos",
    "multimodal",
    "cosmos",
    "streampetr",
    "starcoder",
    "codellama",
    "codegemma",
    "usdcode",
    "coder",
    "codestral",
    "code-instruct",
    "guard",
    "safety",
    "shield",
    "whisper",
    "parakeet",
    "canary",
    "fastpitch",
    "gliner",
    "parse",
    "reward",
    "mathstral",
)

NIM_BASE_MODELS = {
    "google/gemma-2b",
    "google/gemma-7b",
    "google/recurrentgemma-2b",
    "nvidia/mistral-nemo-minitron-8b-base",
    "mistralai/mixtral-8x22b-v0.1",
}

OPENAI_STATIC_MODELS = [
    {"id": "gpt-4o", "name": "GPT-4o (Latest)"},
    {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
    {"id": "gpt-4-turbo", "name": "GPT-4 Turbo"},
    {"id": "gpt-4", "name": "GPT-4"},
    {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"},
]


class ProviderModelCatalog:
    """List models without coupling provider details to the Flask blueprint."""

    def __init__(
        self,
        config=default_config,
        *,
        http_get: Callable[..., Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._http_get = http_get or requests.get
        self._logger = logger or logging.getLogger(__name__)

    def list_models(
        self,
        provider: str,
        *,
        provided_api_key: str | None = None,
        api_endpoint: str | None = None,
    ) -> dict:
        """Dispatch model discovery while preserving provider response shapes."""
        if provider == "gemini":
            from src.core.llm import GeminiProvider

            return self._fetch_cloud_models(GEMINI_SPEC, GeminiProvider, provided_api_key)
        if provider == "openrouter":
            from src.core.llm import OpenRouterProvider

            return self._fetch_cloud_models(
                OPENROUTER_SPEC,
                OpenRouterProvider,
                provided_api_key,
            )
        if provider == "mistral":
            from src.core.llm import MistralProvider

            return self._fetch_cloud_models(MISTRAL_SPEC, MistralProvider, provided_api_key)
        if provider == "deepseek":
            from src.core.llm import DeepSeekProvider

            return self._fetch_cloud_models(DEEPSEEK_SPEC, DeepSeekProvider, provided_api_key)
        if provider == "poe":
            from src.core.llm.providers.poe import PoeProvider

            return self._fetch_cloud_models(POE_SPEC, PoeProvider, provided_api_key)
        if provider == "nim":
            return self._get_nim_models(provided_api_key)
        if provider == "openai":
            return self._get_openai_models(provided_api_key, api_endpoint)
        return self._get_ollama_models(api_endpoint)

    def _resolve_api_key(
        self,
        provided_key: str | None,
        env_var_name: str,
        config_default: str | None,
    ) -> str | None:
        if provided_key and provided_key != "__USE_ENV__":
            return provided_key
        return os.getenv(env_var_name, config_default)

    @staticmethod
    def _first_key(raw: str | None) -> str | None:
        keys = normalize_api_keys(raw)
        return keys[0] if keys else None

    def _fetch_cloud_models(
        self,
        spec: CloudProviderSpec,
        provider_class,
        provided_api_key: str | None,
    ) -> dict:
        api_key = self._resolve_api_key(
            provided_api_key,
            spec.env_var,
            getattr(self._config, spec.config_key_attr),
        )
        configured_default = getattr(self._config, spec.config_model_attr)
        default_model = configured_default or spec.fallback_model

        def error_body(status: str, message: str) -> dict:
            body = {
                "models": [],
                "default": default_model,
                "status": status,
                "count": 0,
                "error": message,
            }
            if spec.include_model_names_on_error:
                body["model_names"] = []
            return body

        if not api_key:
            return error_body("api_key_missing", spec.api_key_missing_message)

        try:
            provider = provider_class(api_key=api_key)
            models = asyncio.run(
                provider.get_available_models(**dict(spec.get_models_kwargs))
            )
            if not models:
                return error_body(
                    f"{spec.status_prefix}_error",
                    f"Failed to retrieve {spec.display_name} models",
                )

            model_names = [model[spec.model_name_field] for model in models]
            resolved_default = default_model
            if resolved_default not in model_names and model_names:
                resolved_default = model_names[0]
            return {
                "models": models,
                "model_names": model_names,
                "default": resolved_default,
                "status": f"{spec.status_prefix}_connected",
                "count": len(models),
            }
        except Exception as exc:
            return error_body(
                f"{spec.status_prefix}_error",
                f"Error connecting to {spec.display_name} API: {exc}",
            )

    def _get_nim_models(self, provided_api_key: str | None) -> dict:
        endpoint = self._config.NIM_API_ENDPOINT
        try:
            resolved_key, _source = resolve_api_key_for_endpoint(
                provided_api_key,
                "NIM_API_KEY",
                endpoint=endpoint,
                default_endpoint=endpoint,
            )
        except EndpointCredentialError as exc:
            return {
                "models": [],
                "model_names": [],
                "default": self._config.NIM_MODEL or "meta/llama-3.1-8b-instruct",
                "status": "unsafe_api_key_routing",
                "count": 0,
                "error": str(exc),
            }
        api_key = self._first_key(resolved_key)
        default_model = self._config.NIM_MODEL or "meta/llama-3.1-8b-instruct"
        if not api_key:
            return {
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "api_key_missing",
                "count": 0,
                "error": "NVIDIA NIM API key is required. Get your key at https://build.nvidia.com/",
            }

        try:
            base_url = endpoint.replace(
                "/chat/completions", ""
            ).rstrip("/")
            response = self._http_get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                models = []
                for raw_model in models_data:
                    model_id = raw_model.get("id", "")
                    model_lower = model_id.lower()
                    if any(keyword in model_lower for keyword in NIM_NON_CHAT_KEYWORDS):
                        continue
                    if model_id in NIM_BASE_MODELS:
                        continue
                    models.append(
                        {
                            "id": model_id,
                            "name": model_id,
                            "owned_by": raw_model.get("owned_by", "nvidia"),
                        }
                    )
                models.sort(key=lambda model: model["name"].lower())
                if models:
                    model_ids = [model["id"] for model in models]
                    if default_model not in model_ids:
                        default_model = model_ids[0]
                    return {
                        "models": models,
                        "model_names": model_ids,
                        "default": default_model,
                        "status": "nim_connected",
                        "count": len(models),
                    }

            return {
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": (
                    "Failed to retrieve NVIDIA NIM models "
                    f"(HTTP {response.status_code})"
                ),
            }
        except requests.exceptions.ConnectionError:
            return {
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": (
                    "Could not connect to NVIDIA NIM API. "
                    "Check your internet connection."
                ),
            }
        except Exception as exc:
            return {
                "models": [],
                "model_names": [],
                "default": default_model,
                "status": "nim_error",
                "count": 0,
                "error": f"Error connecting to NVIDIA NIM API: {exc}",
            }

    def _get_openai_models(
        self,
        provided_api_key: str | None,
        api_endpoint: str | None,
    ) -> dict:
        default_endpoint = getattr(
            self._config,
            "OPENAI_API_ENDPOINT",
            "https://api.openai.com/v1/chat/completions",
        )
        try:
            resolved_key, _source = resolve_api_key_for_endpoint(
                provided_api_key,
                "OPENAI_API_KEY",
                endpoint=api_endpoint or default_endpoint,
                default_endpoint=default_endpoint,
            )
        except EndpointCredentialError as exc:
            return {
                "models": [],
                "model_names": [],
                "default": None,
                "status": "unsafe_api_key_routing",
                "count": 0,
                "error": str(exc),
            }
        api_key = self._first_key(resolved_key)
        base_url = (
            api_endpoint.replace("/chat/completions", "").rstrip("/")
            if api_endpoint
            else "https://api.openai.com/v1"
        )
        is_official_openai = urlparse(base_url).hostname == "api.openai.com"
        fetch_error = None
        models_url = f"{base_url}/models"

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            response = self._http_get(models_url, headers=headers, timeout=10)
            if response.status_code == 200:
                models_data = response.json().get("data", [])
                models = []
                for raw_model in models_data:
                    model_id = raw_model.get("id", "")
                    if "embedding" in model_id.lower() or "whisper" in model_id.lower():
                        continue
                    models.append(
                        {
                            "id": model_id,
                            "name": model_id,
                            "owned_by": raw_model.get("owned_by", "unknown"),
                        }
                    )
                models.sort(key=lambda model: model["name"].lower())
                if models:
                    model_ids = [model["id"] for model in models]
                    default_model = (
                        self._config.DEFAULT_MODEL
                        if self._config.DEFAULT_MODEL in model_ids
                        else model_ids[0]
                    )
                    return {
                        "models": models,
                        "model_names": model_ids,
                        "default": default_model,
                        "status": "openai_connected",
                        "count": len(models),
                    }
                fetch_error = "Endpoint returned no models (HTTP 200, empty data)"
            else:
                fetch_error = f"HTTP {response.status_code} from {models_url}"
                if response.text:
                    fetch_error += f": {response.text[:200]}"
        except requests.exceptions.SSLError as exc:
            fetch_error = (
                f"SSL error ({exc}). If this is a local server, "
                "use http:// instead of https://"
            )
            self._logger.warning(
                "OpenAI-compatible models fetch SSL error at %s: %s",
                base_url,
                exc,
            )
        except requests.exceptions.ConnectionError as exc:
            fetch_error = f"Could not connect to {base_url} ({exc})"
            self._logger.warning(
                "OpenAI-compatible models fetch connection error at %s: %s",
                base_url,
                exc,
            )
        except Exception as exc:
            fetch_error = f"{type(exc).__name__}: {exc}"
            self._logger.warning(
                "OpenAI-compatible models fetch failed at %s: %s",
                base_url,
                exc,
            )

        if not is_official_openai:
            return {
                "models": [],
                "model_names": [],
                "default": None,
                "status": "openai_error",
                "count": 0,
                "endpoint": base_url,
                "error": fetch_error or f"Could not list models at {models_url}",
            }

        model_ids = [model["id"] for model in OPENAI_STATIC_MODELS]
        fallback_default = (
            self._config.DEFAULT_MODEL
            if self._config.DEFAULT_MODEL in model_ids
            else "gpt-4o"
        )
        return {
            "models": OPENAI_STATIC_MODELS,
            "model_names": model_ids,
            "default": fallback_default,
            "status": "openai_static",
            "count": len(OPENAI_STATIC_MODELS),
            "error": fetch_error,
        }

    def _get_ollama_models(self, api_endpoint: str | None) -> dict:
        ollama_base_from_ui = api_endpoint or self._config.API_ENDPOINT
        tags_url = ""
        try:
            parsed = urlparse(ollama_base_from_ui)
            path = parsed.path or "/"
            if "/api/" in path:
                base_path = path.split("/api/")[0]
                base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"
            else:
                base_url = f"{parsed.scheme}://{parsed.netloc}"
            tags_url = f"{base_url}/api/tags"
            response = self._http_get(tags_url, timeout=10)
            if response.status_code == 200:
                model_names = [
                    model.get("name")
                    for model in response.json().get("models", [])
                    if model.get("name")
                ]
                default_model = (
                    self._config.DEFAULT_MODEL
                    if self._config.DEFAULT_MODEL in model_names
                    else model_names[0]
                    if model_names
                    else self._config.DEFAULT_MODEL
                )
                return {
                    "models": model_names,
                    "default": default_model,
                    "status": "ollama_connected",
                    "count": len(model_names),
                }
        except requests.exceptions.ConnectionError:
            print(f"Connection refused to {tags_url}. Is Ollama running?")
        except requests.exceptions.Timeout:
            print(f"Timeout connecting to {tags_url} (10s)")
        except requests.exceptions.RequestException as exc:
            print(f"Could not connect to Ollama at {ollama_base_from_ui}: {exc}")
        except Exception as exc:
            print(f"Error retrieving models from {ollama_base_from_ui}: {exc}")

        return {
            "models": [],
            "default": self._config.DEFAULT_MODEL,
            "status": "ollama_offline_or_error",
            "count": 0,
            "error": (
                f"Ollama is not accessible at {ollama_base_from_ui} or an error "
                "occurred. Verify that Ollama is running ('ollama serve') and "
                "the endpoint is correct."
            ),
        }


def register_provider_model_routes(
    bp: Blueprint,
    *,
    config=default_config,
    logger: logging.Logger | None = None,
) -> ProviderModelCatalog:
    """Register model discovery and return its reusable catalog service."""
    catalog = ProviderModelCatalog(config, logger=logger)

    @bp.route("/api/providers/deepseek/availability", methods=["GET"])
    def get_deepseek_availability():
        """Expose the server-authoritative pricing window in CDMX time."""
        response = jsonify(get_deepseek_pricing_status().to_dict())
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @bp.route("/api/models", methods=["GET", "POST"])
    def get_available_models():
        if request.method == "POST":
            data = request.get_json() or {}
        else:
            data = {}
        provider = (
            data.get("provider", "ollama")
            if request.method == "POST"
            else request.args.get("provider", "ollama")
        )
        api_key = (
            data.get("api_key")
            if request.method == "POST"
            else request.args.get("api_key")
        )
        if provider == "openai":
            default_endpoint = "https://api.openai.com/v1/chat/completions"
            api_endpoint = (
                data.get("api_endpoint", default_endpoint)
                if request.method == "POST"
                else request.args.get("api_endpoint", default_endpoint)
            )
        else:
            # The legacy Ollama path reads only the query string, including
            # for POST requests. Keep that contract for unknown providers too.
            api_endpoint = request.args.get("api_endpoint", config.API_ENDPOINT)
        return jsonify(
            catalog.list_models(
                provider,
                provided_api_key=api_key,
                api_endpoint=api_endpoint,
            )
        )

    return catalog
