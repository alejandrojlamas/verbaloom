"""Context labels for token usage events.

LLM providers are created in several places: translation jobs, profile
preparation, samples and future agents. ContextVars let the factory attach the
current book/job metadata to every provider call without threading parameters
through every prompt builder.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any


_USAGE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("tbl_usage_context", default={})


def get_usage_context() -> dict[str, Any]:
    return dict(_USAGE_CONTEXT.get() or {})


def set_usage_context(**values: Any) -> Token:
    current = get_usage_context()
    current.update({key: value for key, value in values.items() if value is not None})
    return _USAGE_CONTEXT.set(current)


def reset_usage_context(token: Token) -> None:
    _USAGE_CONTEXT.reset(token)
