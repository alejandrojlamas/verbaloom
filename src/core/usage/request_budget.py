"""Fail-closed bounds for provider routes with a verified text-only contract."""

from decimal import Decimal

from src.core.llm.providers.openai_contract import (
    EDITORIAL_MODELS,
    MAX_CONTEXT_TOKENS,
    MAX_OUTPUT_TOKENS,
)
from src.core.pricing.pricing_data import (
    DEEPSEEK_PRICING_EFFECTIVE_AT,
    DEEPSEEK_PRICING_TIERS,
    OPENAI_PRICING_VERSION,
    get_openai_editorial_pricing,
)
from src.utils.provider_security import endpoint_origin

from .budget import BudgetLimitError


def request_bound(provider, endpoint, payload):
    """Reserve a documented full-window bound, not a chars/token estimate.

    Deliberately conservative: the pilot can serialize Terra requests within
    USD 5. Unknown contracts/tier/tool charges fail before network activity.
    """
    if provider == "deepseek":
        return _deepseek_bound(endpoint, payload)
    allowed = {
        "model",
        "messages",
        "stream",
        "reasoning_effort",
        "max_completion_tokens",
        "temperature",
        "service_tier",
        "prompt_cache_options",
    }
    messages = payload.get("messages")
    output = payload.get("max_completion_tokens")
    if (
        provider != "openai"
        or endpoint_origin(endpoint) != "https://api.openai.com"
        or payload.get("model") not in EDITORIAL_MODELS
        or set(payload) - allowed
        or payload.get("service_tier") != "default"
        or payload.get("prompt_cache_options") != {"ttl": "30m"}
        or payload.get("stream") is not False
        or type(output) is not int
        or not 1 <= output <= MAX_OUTPUT_TOKENS
        or not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in {"user", "system", "developer"}
            or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise BudgetLimitError(
            "No verified spend bound for this provider request",
            reason="budget_configuration",
        )
    rates = get_openai_editorial_pricing(
        payload["model"], prompt_tokens=MAX_CONTEXT_TOKENS
    )
    upper = (
        Decimal(str(rates["input_cache_write_30m"])) * MAX_CONTEXT_TOKENS
        + Decimal(str(rates["output"])) * output
    ) / 1_000_000
    return upper, {
        "pricing_version": OPENAI_PRICING_VERSION,
        "pricing_snapshot": rates,
        "bound_basis": "full_context_window",
        "input_token_bound": MAX_CONTEXT_TOKENS,
        "output_token_bound": output,
        "provider": provider,
        "model": payload["model"],
    }


def _deepseek_bound(endpoint, payload):
    messages = payload.get("messages")
    output = payload.get("max_tokens")
    allowed = {"model", "messages", "stream", "thinking", "max_tokens", "temperature"}
    if (
        endpoint_origin(endpoint) != "https://api.deepseek.com"
        or payload.get("model") not in {"deepseek-flash", "deepseek-v4-pro"}
        or set(payload) - allowed
        or payload.get("stream") is not False
        or type(output) is not int
        or not 1 <= output <= 384_000
        or not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in {"user", "system"}
            or not isinstance(m["content"], str)
            for m in messages
        )
    ):
        raise BudgetLimitError(
            "No verified spend bound for this DeepSeek request",
            reason="budget_configuration",
        )
    rates = DEEPSEEK_PRICING_TIERS["peak"][payload["model"]]
    upper = (
        Decimal(str(rates["input"])) * 1_048_576
        + Decimal(str(rates["output"])) * output
    ) / 1_000_000
    return upper, {
        "pricing_version": f"deepseek_official_{DEEPSEEK_PRICING_EFFECTIVE_AT}_peak",
        "pricing_snapshot": rates,
        "bound_basis": "full_context_window_peak",
        "input_token_bound": 1_048_576,
        "output_token_bound": output,
        "provider": "deepseek",
        "model": payload["model"],
    }
