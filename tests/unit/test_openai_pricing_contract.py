"""Verified pricing categories, not real paid measurements."""

import json

import pytest

from src.core.llm.providers.openai_contract import normalize_completion
from src.core.pricing import calculate_usage_cost, get_default_pricing
from src.core.pricing.pricing_data import get_openai_editorial_pricing
from src.core.usage.store import TokenUsageStore


def test_write_categories_replace_ordinary_input():
    rates = get_default_pricing("openai", "gpt-6.1-terra")
    result = calculate_usage_cost(
        rates,
        prompt_tokens=12000,
        completion_tokens=1000,
        prompt_cache_hit_tokens=4000,
        cache_write_30m_tokens=3000,
        cache_write_12h_tokens=4000,
    )
    assert result == pytest.approx((0.01495, 0.005, 0.01995))


@pytest.mark.parametrize(
    "tier,multiplier",
    [("default", 1), ("flex", 0.5), ("batch", 0.5), ("priority", 2), ("fast", 2)],
)
def test_tiers_and_long_context_are_applied_to_whole_request(tier, multiplier):
    short = get_openai_editorial_pricing(
        "gpt-6-luna", prompt_tokens=272000, service_tier=tier
    )
    long = get_openai_editorial_pricing(
        "gpt-6-luna", prompt_tokens=272001, service_tier=tier
    )
    assert short["input"] == pytest.approx(0.1 * multiplier)
    assert long["input"] == pytest.approx(short["input"] * 2)
    assert long["output"] == pytest.approx(short["output"] * 1.5)
    assert long["input_cache_write_30m"] == pytest.approx(
        short["input_cache_write_30m"] * 2
    )


def test_unknown_model_or_tier_is_not_priced_as_nearest_model():
    assert get_default_pricing("openai", "gpt-6-luna-unknown") is None
    assert get_openai_editorial_pricing("gpt-6-luna", service_tier="auto") is None


def test_normalizer_does_not_double_count_writes_or_reasoning():
    result = normalize_completion(
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            "service_tier": "priority",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "cache_write_tokens": 60,
                },
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        }
    )
    assert (
        result.prompt_cache_hit_tokens,
        result.prompt_cache_miss_tokens,
        result.prompt_cache_write_tokens,
    ) == (30, 10, 60)
    assert result.total_tokens == 150
    assert result.service_tier == "priority"


def test_ledger_saves_rate_snapshot_and_real_tier(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")
    store.record_call(
        provider="openai",
        model="gpt-6-luna",
        prompt="fixture",
        prompt_tokens=10000,
        completion_tokens=1000,
        prompt_cache_hit_tokens=4000,
        metadata={
            "service_tier": "priority",
            "usage_observed": True,
            "parameters": {
                "service_tier": "default",
                "prompt_cache_options": {"ttl": "30m"},
            },
            "raw_usage": {"prompt_tokens_details": {"cache_write_tokens": 5000}},
        },
    )
    event = store.events()[0]
    assert event["total_cost_usd"] == pytest.approx(0.00253)
    meta = json.loads(event["metadata"])
    assert meta["pricing_snapshot"]["input"] == 0.2
    assert meta["cost_basis"] == "provider_usage"
    assert meta["cash_paid_usd"] is None
    assert "2026-11-12" in event["pricing_source"]
