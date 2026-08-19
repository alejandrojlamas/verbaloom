"""Client-safe payload helpers for REST and WebSocket responses."""

from __future__ import annotations

import copy
from typing import Any


_SENSITIVE_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "password",
)

_SENSITIVE_TOKEN_KEYS = {
    "token",
    "access_token",
    "refresh_token",
    "auth_token",
    "bearer_token",
}

_INTERNAL_RUNTIME_KEYS = {
    "_fidelity_report",
    "_editorial_quality_report",
    "_candidate_results",
    "_source_guard_refs",
    "_source_guard_refs_loaded",
}

_VERBOSE_LOG_DATA_KEYS = {
    "prompt",
    "system_prompt",
    "user_prompt",
    "response",
    "raw_response",
}

MAX_CLIENT_PREVIEW_CHARS = 8000


def redact_for_client(value: Any) -> Any:
    """Return a client-safe copy of nested runtime data.

    Runtime configs and logs can carry resolved provider credentials, full
    prompts, full responses, and internal Python objects. UI status surfaces
    only need metadata and short previews, so keep those payloads out of the
    browser and in-memory status logs.
    """
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if key_text in _INTERNAL_RUNTIME_KEYS:
                continue
            if (
                any(fragment in key_lower for fragment in _SENSITIVE_KEY_FRAGMENTS)
                or key_lower in _SENSITIVE_TOKEN_KEYS
                or (key_lower.endswith("_token") and not key_lower.endswith("_tokens"))
            ):
                safe[key_text] = "[redacted]" if item else ""
                continue
            if key_text in _VERBOSE_LOG_DATA_KEYS:
                safe[key_text] = _omitted_marker(item)
                continue
            safe[key_text] = redact_for_client(item)
        return safe
    if isinstance(value, list):
        return [redact_for_client(item) for item in value]
    return value


def client_safe_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return redact_for_client(copy.deepcopy(config or {}))


def client_safe_logs(logs: list[Any] | None, limit: int = 100) -> list[Any]:
    return redact_for_client(copy.deepcopy((logs or [])[-limit:]))


def client_safe_log_entry(log_entry: dict[str, Any] | None) -> dict[str, Any]:
    return redact_for_client(copy.deepcopy(log_entry or {}))


def trim_client_preview(text: str | None, limit: int = MAX_CLIENT_PREVIEW_CHARS) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "\n\n[preview truncated]"


def _omitted_marker(value: Any) -> Any:
    if not value:
        return value
    if isinstance(value, str) and value.startswith("[omitted ") and value.endswith(" chars]"):
        return value
    return f"[omitted {len(str(value))} chars]"
