"""Bound complete LLM operations independently of provider retry internals."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from src.common.provider_outage import (
    collect_transport_outage,
    outage_recovery_enabled,
)
from src.config import (
    LLM_CLOUD_OPERATION_TIMEOUT,
    LLM_LOCAL_OPERATION_TIMEOUT,
    REQUEST_TIMEOUT,
)

from .exceptions import ProviderUnavailableError


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
    role: str | None = None,
    **kwargs: Any,
) -> Any:
    """Execute an async LLM call within one total wall-clock budget.

    Providers may use ``request_timeout`` independently for every network
    retry. ``asyncio.wait_for`` wraps the complete coroutine, preventing those
    retries from multiplying a nominal timeout into a multi-hour job stall.
    Unsupported optional keywords are removed for lightweight test/custom
    providers while preserving real exceptions raised inside the call.

    Inside ``provider_outage_recovery`` an operation that ended without a
    response because the provider was unreachable (or because the total
    deadline elapsed) raises ``ProviderUnavailableError`` instead of returning
    ``None`` or ``asyncio.TimeoutError``. Callers then see a retryable provider
    stop rather than a missing candidate. Outside that scope nothing changes.
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
    from src.core.pipeline_runtime import request_role
    token = request_role.set(role) if role else None
    try:
        if not outage_recovery_enabled():
            return await asyncio.wait_for(call(*args, **filtered), timeout=total_budget)
        with collect_transport_outage() as outage:
            try:
                result = await asyncio.wait_for(
                    call(*args, **filtered),
                    timeout=total_budget,
                )
            except asyncio.TimeoutError:
                raise ProviderUnavailableError(
                    f"The provider did not answer within {total_budget:.0f}s.",
                    reason="operation_deadline",
                    provider=provider_kind(provider),
                ) from None
        if result is None and outage.get("reason"):
            raise ProviderUnavailableError(
                "The provider could not be reached after its bounded retries "
                f"({outage['reason']}).",
                reason=outage["reason"],
                provider=provider_kind(provider),
            )
        return result
    finally:
        if token is not None:
            request_role.reset(token)

