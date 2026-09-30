"""Shared cache-aware token cost calculation."""

from __future__ import annotations

from typing import Mapping


def calculate_usage_cost(
    pricing: Mapping[str, float],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
) -> tuple[float, float, float]:
    """Return input, output, and total USD cost for provider-reported usage.

    Any prompt tokens not covered by a provider cache breakdown are billed at
    the cache-miss rate. This keeps partial/legacy responses conservative.
    """

    prompt = max(0, int(prompt_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    cache_hit = max(0, int(prompt_cache_hit_tokens or 0))
    cache_miss = max(0, int(prompt_cache_miss_tokens or 0))

    input_rate = float(pricing.get("input_cache_miss", pricing.get("input", 0.0)) or 0.0)
    cache_hit_rate = float(pricing.get("input_cache_hit", input_rate) or 0.0)
    output_rate = float(pricing.get("output", 0.0) or 0.0)

    # Provider docs define prompt = cache hit + cache miss. Be defensive when
    # compatible gateways omit one field or report an inconsistent breakdown.
    cache_hit = min(cache_hit, prompt)
    cache_miss = min(cache_miss, max(0, prompt - cache_hit))
    unreported_input = max(0, prompt - cache_hit - cache_miss)

    input_cost = (
        cache_hit * cache_hit_rate
        + (cache_miss + unreported_input) * input_rate
    ) / 1_000_000
    output_cost = completion * output_rate / 1_000_000
    return input_cost, output_cost, input_cost + output_cost
