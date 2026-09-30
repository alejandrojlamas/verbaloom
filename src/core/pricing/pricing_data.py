"""
Default pricing data for providers without a public pricing API.

Prices are in USD per 1 million tokens.
Users can override these values via the UI (sent in /api/cost/estimate payload).

Last updated: 2026-09-29
Sources: official provider documentation pages.
"""

from __future__ import annotations

LAST_UPDATED = "2026-09-29"
DEEPSEEK_PRICING_EFFECTIVE_AT = "2026-09-10T04:00:00+00:00"


def _deepseek_rates(*, flash: tuple[float, float, float], pro: tuple[float, float, float]) -> dict:
    """Build the official DeepSeek table, including resume-only aliases.

    Tuple order is cache-hit input, cache-miss input, output. The canonical
    public IDs are ``deepseek-flash`` and ``deepseek-v4-pro``. Legacy IDs stay
    priced so old checkpoints remain auditable while new jobs use the current
    names.
    """

    def entry(rates: tuple[float, float, float]) -> dict:
        cache_hit, cache_miss, output = rates
        return {
            "input": cache_miss,
            "input_cache_hit": cache_hit,
            "input_cache_miss": cache_miss,
            "output": output,
        }

    flash_entry = entry(flash)
    pro_entry = entry(pro)
    return {
        "deepseek-flash": {**flash_entry, "note": "DeepSeek V4.1 Flash"},
        "deepseek-v4-pro": {**pro_entry, "note": "DeepSeek V4 Pro"},
        "deepseek-v4-flash": {**flash_entry, "note": "Legacy alias for deepseek-flash"},
        "deepseek-v4-flash-vision-exp": {
            **flash_entry,
            "note": "Retired alias routed to deepseek-flash",
        },
        "deepseek-chat": {**flash_entry, "note": "Retired compatibility alias"},
        "deepseek-reasoner": {**flash_entry, "note": "Retired compatibility alias"},
    }


DEEPSEEK_PRICING_TIERS = {
    "off_peak": _deepseek_rates(
        flash=(0.003, 0.15, 0.60),
        pro=(0.022, 0.66, 1.98),
    ),
    "peak": _deepseek_rates(
        flash=(0.006, 0.30, 1.20),
        pro=(0.044, 1.32, 3.96),
    ),
}

DEFAULT_PRICING = {
    "gemini": {
        "gemini-2.5-pro":        {"input": 1.25, "output": 10.00, "note": "Standard tier (<=200K context)"},
        "gemini-2.5-pro-large":  {"input": 2.50, "output": 15.00, "note": "Extended tier (>200K context)"},
        "gemini-2.5-flash":      {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        "gemini-2.0-flash":      {"input": 0.10, "output": 0.40, "note": "Deprecated 2026-06-01"},
        "gemini-1.5-pro":        {"input": 1.25, "output": 5.00, "note": "Legacy"},
        "gemini-1.5-flash":      {"input": 0.075, "output": 0.30, "note": "Legacy"},
    },
    "openai": {
        "gpt-4o":           {"input": 2.50,  "output": 10.00},
        "gpt-4o-mini":      {"input": 0.15,  "output": 0.60},
        "gpt-4.1":          {"input": 2.00,  "output": 8.00},
        "gpt-4.1-mini":     {"input": 0.40,  "output": 1.60},
        "gpt-4.1-nano":     {"input": 0.10,  "output": 0.40},
        "gpt-4-turbo":      {"input": 10.00, "output": 30.00, "note": "Legacy"},
        "gpt-4":            {"input": 30.00, "output": 60.00, "note": "Legacy"},
        "gpt-3.5-turbo":    {"input": 0.50,  "output": 1.50,  "note": "Legacy"},
        "o1":               {"input": 15.00, "output": 60.00, "note": "Reasoning model"},
        "o1-mini":          {"input": 3.00,  "output": 12.00, "note": "Reasoning model"},
        "o3-mini":          {"input": 1.10,  "output": 4.40,  "note": "Reasoning model"},
    },
    # The app waits through peak windows by default, so the static table used
    # for future-job estimates is the off-peak table. Completed calls select
    # the actual tier from their UTC timestamp in the usage ledger.
    "deepseek": DEEPSEEK_PRICING_TIERS["off_peak"],
    "mistral": {
        "mistral-large-latest":  {"input": 2.00, "output": 6.00},
        "mistral-large-2411":    {"input": 2.00, "output": 6.00},
        "mistral-medium-latest": {"input": 0.40, "output": 2.00},
        "mistral-medium-3":      {"input": 0.40, "output": 2.00},
        "mistral-small-latest":  {"input": 0.20, "output": 0.60},
        "mistral-small-3":       {"input": 0.10, "output": 0.30},
        "ministral-8b-latest":   {"input": 0.10, "output": 0.10},
        "ministral-3b-latest":   {"input": 0.04, "output": 0.04},
        "codestral-latest":      {"input": 0.30, "output": 0.90},
        "pixtral-large-latest":  {"input": 2.00, "output": 6.00},
    },
    "nim": {
        # NVIDIA NIM is mostly free-credits via build.nvidia.com.
        # These values are pay-as-you-go reference prices.
        "meta/llama-3.1-8b-instruct":   {"input": 0.04, "output": 0.04},
        "meta/llama-3.1-70b-instruct":  {"input": 0.40, "output": 0.40},
        "meta/llama-3.1-405b-instruct": {"input": 1.20, "output": 1.20},
        "deepseek-ai/deepseek-v3":      {"input": 0.27, "output": 1.10},
        "deepseek-ai/deepseek-r1":      {"input": 0.55, "output": 2.19},
    },
}


def get_default_pricing(
    provider: str,
    model: str,
    *,
    pricing_tier: str | None = None,
) -> dict | None:
    """
    Return {input, output} prices per 1M tokens for the given provider/model.

    Returns None if no default pricing is known.
    Lookup is case-insensitive and tolerates minor variants in model names.
    """
    provider_name = provider.lower()
    if provider_name == "deepseek":
        tier = str(pricing_tier or "off_peak").strip().lower().replace("-", "_")
        provider_data = DEEPSEEK_PRICING_TIERS.get(tier)
        if provider_data is None:
            provider_data = DEEPSEEK_PRICING_TIERS["off_peak"]
    else:
        provider_data = DEFAULT_PRICING.get(provider_name)
    if not provider_data:
        return None

    if model in provider_data:
        return _strip_note(provider_data[model])

    model_lower = model.lower()
    for known_model, pricing in provider_data.items():
        if known_model.lower() == model_lower:
            return _strip_note(pricing)

    for known_model, pricing in provider_data.items():
        if known_model.lower() in model_lower or model_lower in known_model.lower():
            return _strip_note(pricing)

    return None


def _strip_note(entry: dict) -> dict:
    result = {"input": entry["input"], "output": entry["output"]}
    if "input_cache_hit" in entry:
        result["input_cache_hit"] = entry["input_cache_hit"]
    if "input_cache_miss" in entry:
        result["input_cache_miss"] = entry["input_cache_miss"]
    return result
