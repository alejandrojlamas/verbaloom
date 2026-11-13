"""Non-billable official provider credential/model visibility preflight."""

import asyncio

import httpx

from .providers.deepseek import DeepSeekProvider
from .providers.openai import OpenAICompatibleProvider
from .providers.openai_contract import EDITORIAL_MODELS, text_parameters

OFFICIAL_OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"


async def preflight_deepseek(*, api_key, models):
    """Check the live catalog without the discovery UI's offline fallback."""
    report = {"provider": "deepseek", "generation_tested": False, "balance": None, "models": []}
    if not api_key or not str(api_key).strip():
        return {**report, "status": "missing_credentials"}
    provider = DeepSeekProvider(api_key=api_key)
    try:
        client = await provider._get_client()
        response = await asyncio.wait_for(client.get(
            provider.MODELS_URL, headers={"Authorization": f"Bearer {provider.api_key}"},
            timeout=httpx.Timeout(10, connect=5),
        ), timeout=12)
        if response.status_code != 200:
            status = {401: "authentication_error", 403: "permission_denied", 429: "rate_limited"}.get(response.status_code, "provider_error")
            return {**report, "status": status}
        available = {item["id"] for item in response.json()["data"]}
        for model in models:
            resolved = provider.LEGACY_MODEL_ALIASES.get(model, model)
            report["models"].append({"model": model, "resolved_model": resolved,
                                     "status": "model_visible" if resolved in available else "model_unavailable"})
        return {**report, "status": "models_visible" if report["models"] and all(
            item["status"] == "model_visible" for item in report["models"]) else "not_ready"}
    except (httpx.HTTPError, TimeoutError):
        return {**report, "status": "connection_error"}
    except (ValueError, KeyError, TypeError):
        return {**report, "status": "invalid_response"}
    finally:
        await provider.close()


async def preflight_openai(*, api_key, models=None, reasoning_effort="none", max_output_tokens=8192):
    """Check model visibility without generating text or claiming billing access.

    A successful GET does not prove that a completion can be billed or that the
    account has credit. A bounded generation probe still belongs to the pilot's
    shared budget, never to this non-billable check.
    """
    models = list(EDITORIAL_MODELS if models is None else models)
    report = {"provider": "openai", "endpoint": OFFICIAL_OPENAI_ENDPOINT,
              "generation_tested": False, "balance": None, "models": []}
    for model in models:
        if model not in EDITORIAL_MODELS:
            raise ValueError("Preflight accepts only the requested editorial models")
        text_parameters(model, effort=reasoning_effort, max_output_tokens=max_output_tokens)
    if not api_key or not str(api_key).strip():
        return {**report, "status": "missing_credentials"}
    provider = OpenAICompatibleProvider(OFFICIAL_OPENAI_ENDPOINT, models[0] if models else "gpt-6-luna",
                                         api_key=api_key, provider_name="openai")
    try:
        client = await provider._get_client()
        for model in models:
            try:
                response = await asyncio.wait_for(client.get(
                    f"https://api.openai.com/v1/models/{model}",
                    headers={"Authorization": f"Bearer {provider.api_key}"},
                    timeout=httpx.Timeout(10, connect=5),
                ), timeout=12)
                status = {200: "model_visible", 401: "authentication_error", 403: "permission_denied",
                          404: "model_unavailable", 429: "rate_limited"}.get(response.status_code, "provider_error")
                if response.status_code == 200 and response.json().get("id") != model:
                    status = "invalid_response"
                report["models"].append({"model": model, "status": status,
                                         "http_status": response.status_code})
            except (httpx.HTTPError, TimeoutError):
                report["models"].append({"model": model, "status": "connection_error"})
            except (ValueError, AttributeError):
                report["models"].append({"model": model, "status": "invalid_response"})
    finally:
        await provider.close()
    report["status"] = "models_visible" if report["models"] and all(
        item["status"] == "model_visible" for item in report["models"]) else "not_ready"
    return report
