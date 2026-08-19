from src.core.pricing import get_default_pricing
from src.core.pricing.estimator import CostEstimator


def test_deepseek_v4_pro_pricing_uses_current_cache_prices():
    pricing = get_default_pricing("deepseek", "deepseek-v4-pro")

    assert pricing["input"] == 0.435
    assert pricing["input_cache_hit"] == 0.003625
    assert pricing["input_cache_miss"] == 0.435
    assert pricing["output"] == 0.87


def test_deepseek_estimate_reports_cache_aware_range():
    pricing = get_default_pricing("deepseek", "deepseek-v4-pro")
    estimator = CostEstimator(
        provider="deepseek",
        model="deepseek-v4-pro",
        pricing=pricing,
        max_tokens_per_chunk=10,
    )

    text = "\n\n".join(
        f"Paragraph {idx} with enough repeated words to become its own chunk."
        for idx in range(30)
    )
    result = estimator.estimate(text, "Spanish", "Spanish")

    assert result["n_chunks"] > 1
    assert result["input_cost_min"] < result["input_cost"]
    assert result["total_cost_min"] < result["total_cost_max"]
    assert result["pricing_used"]["input_cache_hit_per_million"] == 0.003625
    assert result["cache_assumption"]["min_uses_cache_hits"] is True
