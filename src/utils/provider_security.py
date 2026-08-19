"""Credential boundaries for user-selectable LLM endpoints."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from src.utils.branding import env_value


# A configurable endpoint is not automatically trusted just because it was
# loaded at startup. Otherwise changing OPENAI_API_ENDPOINT in ``.env`` and
# restarting would silently turn an arbitrary host into a credential sink.
# Additional origins require the explicit VERBALOOM_TRUSTED_KEY_ENDPOINTS opt-in.
BUILTIN_TRUSTED_KEY_ENDPOINT_ORIGINS = {
    "https://api.openai.com",
    "https://integrate.api.nvidia.com",
}


class EndpointCredentialError(ValueError):
    """Raised when a saved credential would cross an untrusted endpoint boundary."""


def endpoint_origin(endpoint: str | None) -> str:
    """Return a normalized HTTP(S) origin, or an empty string for invalid URLs."""
    parsed = urlparse(str(endpoint or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    host = parsed.hostname.casefold()
    try:
        port = parsed.port
    except ValueError:
        return ""
    default_port = 443 if parsed.scheme == "https" else 80
    suffix = f":{port}" if port and port != default_port else ""
    return f"{parsed.scheme.casefold()}://{host}{suffix}"


def trusted_key_endpoint_origins(raw: str | None = None) -> set[str]:
    """Parse the explicit custom-endpoint credential allowlist."""
    value = env_value("TRUSTED_KEY_ENDPOINTS", "") if raw is None else raw
    return {
        origin
        for item in str(value or "").split(",")
        if (origin := endpoint_origin(item))
    }


def endpoint_allows_saved_key(
    endpoint: str | None,
    default_endpoint: str | None,
    *,
    trusted_origins: set[str] | None = None,
) -> bool:
    """Return whether an endpoint may receive a key loaded from local settings."""
    requested_origin = endpoint_origin(endpoint)
    default_origin = endpoint_origin(default_endpoint)
    if not requested_origin:
        return not endpoint
    if (
        default_origin
        and requested_origin == default_origin
        and requested_origin in BUILTIN_TRUSTED_KEY_ENDPOINT_ORIGINS
    ):
        return True
    allowed = trusted_key_endpoint_origins() if trusted_origins is None else trusted_origins
    return requested_origin in allowed


def resolve_api_key_for_endpoint(
    value: object,
    env_var_name: str,
    *,
    endpoint: str | None,
    default_endpoint: str | None,
) -> tuple[str, str]:
    """Resolve a request key without leaking saved credentials to a custom host.

    An actual key in the request is considered explicit. Empty values use the
    environment only for the configured provider endpoint (or an origin listed
    in ``VERBALOOM_TRUSTED_KEY_ENDPOINTS``). ``__USE_ENV__`` is rejected for any
    other custom origin so the caller receives a visible error instead of a
    silent credential downgrade.
    """
    raw = "" if value is None else str(value)
    if raw and raw != "__USE_ENV__":
        return raw, "explicit"

    if endpoint_allows_saved_key(endpoint, default_endpoint):
        key = os.getenv(env_var_name, "")
        return key, "environment" if key else "none"

    if raw == "__USE_ENV__":
        raise EndpointCredentialError(
            "Saved API keys cannot be used with an untrusted custom endpoint. "
            "Provide the key explicitly for this request or add the endpoint "
            "origin to VERBALOOM_TRUSTED_KEY_ENDPOINTS."
        )
    return "", "none"


def sanitize_restored_endpoint_credentials(
    config: dict,
    default_endpoints: dict[str, str],
    *,
    keyed_custom_providers: tuple[str, ...] = ("openai", "nim"),
) -> None:
    """Remove ambiguous saved keys before a persisted job is resumed.

    New jobs record whether a key was explicitly supplied. Older checkpoints
    do not, so a key attached to an untrusted endpoint must be treated as an
    implicitly resolved local secret and stripped. Explicit request keys and
    allowlisted endpoints continue to work.
    """
    provider = str(config.get("llm_provider") or "ollama").lower()
    if provider not in keyed_custom_providers:
        return
    default_endpoint = default_endpoints.get(provider, "")
    endpoint = config.get("llm_api_endpoint") or default_endpoint
    sources = dict(config.get("_credential_sources") or {})
    if (
        not endpoint_allows_saved_key(endpoint, default_endpoint)
        and sources.get(provider) != "explicit"
    ):
        config[f"{provider}_api_key"] = ""
        sources[provider] = "none"
        config["_credential_sources"] = sources
