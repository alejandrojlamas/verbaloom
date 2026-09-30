from src.core.pricing import get_default_pricing
from src.core.pricing.estimator import CostEstimator


def test_deepseek_v4_pro_pricing_uses_current_cache_prices():
    pricing = get_default_pricing("deepseek", "deepseek-v4-pro")

    assert pricing["input"] == 0.66
    assert pricing["input_cache_hit"] == 0.022
    assert pricing["input_cache_miss"] == 0.66
    assert pricing["output"] == 1.98


def test_deepseek_flash_and_legacy_alias_share_current_off_peak_price():
    current = get_default_pricing("deepseek", "deepseek-flash")
    legacy = get_default_pricing("deepseek", "deepseek-v4-flash")

    assert current == legacy
    assert current == {
        "input": 0.15,
        "input_cache_hit": 0.003,
        "input_cache_miss": 0.15,
        "output": 0.60,
    }


def test_deepseek_peak_price_is_selected_explicitly():
    pricing = get_default_pricing(
        "deepseek",
        "deepseek-v4-pro",
        pricing_tier="peak",
    )

    assert pricing == {
        "input": 1.32,
        "input_cache_hit": 0.044,
        "input_cache_miss": 1.32,
        "output": 3.96,
    }


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
    assert result["pricing_used"]["input_cache_hit_per_million"] == 0.022
    assert result["cache_assumption"]["min_uses_cache_hits"] is True
    assert result["input_tokens"] == result["input_tokens_per_pass"]


def test_multi_pass_estimate_reports_total_tokens_across_all_passes():
    pricing = get_default_pricing("deepseek", "deepseek-flash")
    estimator = CostEstimator("deepseek", "deepseek-flash", pricing, max_tokens_per_chunk=20)

    result = estimator.estimate(
        "A sufficiently long sentence for token estimation. " * 20,
        "English",
        "Spanish",
        options={"refine": True, "text_cleanup": True},
    )

    assert result["passes"] == 3
    assert result["input_tokens"] == result["input_tokens_per_pass"] * 3
    assert result["estimated_output_tokens_min"] > result["main_text_tokens"]
    assert result["token_count_source"] == "tokenizer_estimate"
