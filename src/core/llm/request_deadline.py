"""Bound complete LLM operations independently of provider retry internals."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from src.config import (
    LLM_CLOUD_OPERATION_TIMEOUT,
    LLM_LOCAL_OPERATION_TIMEOUT,
    REQUEST_TIMEOUT,
)


_CLOUD_PROVIDER_HINTS = {
    "deepseek",
    "gemini",
    "litellm",
    "mistral",
    "nim",
    "openai",
    "openrouter",
    "poe",
}


def _accepts_keyword(call: Callable[..., Any], keyword: str) -> bool:
    try:
        parameters = inspect.signature(call).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or parameter.name == keyword
        for parameter in parameters
    )


def provider_kind(provider: Any) -> str:
    explicit = (
        getattr(provider, "provider_type", "")
        or getattr(provider, "provider_name", "")
    )
    if explicit:
        return str(explicit).strip().lower()
    return type(provider).__name__.replace("Provider", "").strip().lower()


def operation_deadline_seconds(
    provider: Any,
    *,
    request_timeout: int | float | None = None,
    deadline: int | float | None = None,
) -> float:
    """Resolve one total deadline for all retries inside a provider call."""
    if deadline is not None:
        return max(0.01, float(deadline))
    request_budget = max(
        0.01,
        float(request_timeout if request_timeout is not None else REQUEST_TIMEOUT),
    )
    kind = provider_kind(provider)
    operation_budget = (
        LLM_CLOUD_OPERATION_TIMEOUT
        if any(hint in kind for hint in _CLOUD_PROVIDER_HINTS)
        else LLM_LOCAL_OPERATION_TIMEOUT
    )
    return max(0.01, min(request_budget, float(operation_budget)))


async def await_llm_call(
    call: Callable[..., Awaitable[Any]],
    *args: Any,
    provider: Any,
    request_timeout: int | float | None = None,
    deadline: int | float | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async LLM call within one total wall-clock budget.

    Providers may use ``request_timeout`` independently for every network
    retry. ``asyncio.wait_for`` wraps the complete coroutine, preventing those
    retries from multiplying a nominal timeout into a multi-hour job stall.
    Unsupported optional keywords are removed for lightweight test/custom
    providers while preserving real exceptions raised inside the call.
    """
    request_budget = (
        request_timeout if request_timeout is not None else REQUEST_TIMEOUT
    )
    filtered = dict(kwargs)
    if _accepts_keyword(call, "timeout"):
        filtered["timeout"] = request_budget
    else:
        filtered.pop("timeout", None)
    for optional in ("system_prompt", "temperature"):
        if optional in filtered and not _accepts_keyword(call, optional):
            filtered.pop(optional, None)

    total_budget = operation_deadline_seconds(
        provider,
        request_timeout=request_budget,
        deadline=deadline,
    )
    return await asyncio.wait_for(
        call(*args, **filtered),
        timeout=total_budget,
    )

