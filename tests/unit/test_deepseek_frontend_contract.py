"""Frontend contracts for current DeepSeek models and cost estimates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROVIDER_MANAGER = ROOT / "src/web/static/js/providers/provider-manager.js"
SETTINGS_MANAGER = ROOT / "src/web/static/js/core/settings-manager.js"
COST_ESTIMATOR = ROOT / "src/web/static/js/providers/cost-estimator.js"
PRICING_MANAGER = ROOT / "src/web/static/js/providers/deepseek-pricing-manager.js"
TEMPLATE = ROOT / "src/web/templates/translation_interface.html"


def _deepseek_fallback_block() -> str:
    source = PROVIDER_MANAGER.read_text(encoding="utf-8")
    start = source.index("const DEEPSEEK_FALLBACK_MODELS")
    end = source.index("\n];", start) + 3
    return source[start:end]


def test_deepseek_picker_exposes_only_current_public_models():
    block = _deepseek_fallback_block()

    assert "deepseek-flash" in block
    assert "deepseek-v4-pro" in block
    assert "deepseek-v4-flash'" not in block
    assert "deepseek-chat" not in block
    assert "deepseek-reasoner" not in block


def test_saved_retired_models_are_migrated_before_selection():
    source = SETTINGS_MANAGER.read_text(encoding="utf-8")

    for legacy_model in (
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
        "deepseek-chat",
        "deepseek-reasoner",
    ):
        assert f"'{legacy_model}'" in source
    assert "? 'deepseek-flash'" in source


def test_default_deepseek_prices_are_resolved_server_side():
    source = COST_ESTIMATOR.read_text(encoding="utf-8")

    assert "if (ctx.source !== 'default_table') payload.pricing = ctx.pricing" in source
    assert "pricing_context" in source
    assert "cost_pricing_tier_off_peak" in source
    assert "cost_pricing_waits_off_peak" in source
    assert 'role="dialog"' in source
    assert 'aria-modal="true"' in source


def test_icon_only_usage_refresh_control_has_an_accessible_name():
    template = TEMPLATE.read_text(encoding="utf-8")
    button_start = template.index('id="usageRefreshBtn"')
    button_end = template.index("</button>", button_start)
    button = template[button_start:button_end]

    assert 'aria-label="Refresh"' in button
    assert "aria-label:common:refresh" in button


def test_pricing_boundary_invalidates_cached_cost_estimates():
    estimator = COST_ESTIMATOR.read_text(encoding="utf-8")
    manager = PRICING_MANAGER.read_text(encoding="utf-8")

    assert "deepseekPricingChanged" in manager
    assert "deepseekPricingChanged" in estimator
    assert "providers.deepseekPricing" in estimator
