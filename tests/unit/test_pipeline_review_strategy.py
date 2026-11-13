"""A/B comparisons change roles, not provider-dependent editorial policy."""

from copy import deepcopy

import pytest

from src.core.job_runtime_config import configure_editorial_guard_options
from src.core.pipeline_config import apply_pipeline, pipeline_preset, validate_pipeline


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"refine": True, "audit_entire_book": True},
        {"refine": True, "permit_unaudited_segments": False},
        {"refine": False, "fidelity_supervisor_mode": "always"},
    ],
)
@pytest.mark.parametrize("model", ["deepseek-flash", "deepseek-v4-pro"])
def test_model_substitution_preserves_effective_guard_policy(options, model):
    baseline = {
        "llm_provider": "deepseek",
        "model": model,
        "file_type": "epub",
        "source_language": "Czech",
        "target_language": "Spanish",
        "prompt_options": {"target_locale": "es-ES", **options},
    }
    original = deepcopy(baseline)
    variants = []
    for preset in ("compatible", "openai_equivalent"):
        config = deepcopy(baseline)
        apply_pipeline(config, pipeline_preset(preset, baseline))
        resolved = configure_editorial_guard_options(config)
        variants.append(deepcopy(resolved))
        # Re-entering a saved job must not change the strategy.
        assert configure_editorial_guard_options(config) == resolved
        assert resolved["target_locale"] == "es-ES"
        assert resolved["_source_language"] == "Czech"
    assert variants[0] == variants[1]
    assert baseline == original


def test_flash_preset_uses_actual_pro_audit_default():
    baseline = {"llm_provider": "deepseek", "model": "deepseek-flash"}
    a = pipeline_preset("compatible", baseline)
    b = pipeline_preset("openai_equivalent", baseline)
    assert a["roles"]["audit"]["model"] == "deepseek-v4-pro"
    assert b["roles"]["audit"]["model"] == "gpt-6.1-terra"
    assert (
        a["review_reference"]
        == b["review_reference"]
        == {
            "provider": "deepseek",
            "model": "deepseek-flash",
        }
    )


@pytest.mark.parametrize(
    "reference",
    [
        [],
        {"provider": "deepseek"},
        {"provider": "openai", "model": "deepseek-flash"},
        {"provider": ["deepseek"], "model": "deepseek-flash"},
        {
            "provider": "deepseek",
            "model": "deepseek-flash",
            "api_key": "YOUR_API_KEY_HERE",
        },
    ],
)
def test_review_reference_is_validated_and_cannot_carry_credentials(reference):
    policy = pipeline_preset("compatible", {})
    policy["review_reference"] = reference
    with pytest.raises(ValueError):
        validate_pipeline(policy)


def test_reports_use_effective_role_models_without_opening_transports():
    from types import SimpleNamespace

    from src.core.fidelity_supervisor import resolve_fidelity_auditor_model
    from src.core.pipeline_runtime import active_pipeline
    from src.core.quality_guard import (
        _resolve_quality_alert_model,
        _resolve_source_aware_editorial_guard_model,
    )
    from src.core.translator import (
        _resolve_profile_audit_model,
        _resolve_profile_repair_model,
    )

    policy = pipeline_preset(
        "openai_equivalent", {"llm_provider": "deepseek", "model": "deepseek-flash"}
    )
    token = active_pipeline.set(SimpleNamespace(config=policy))
    try:
        for resolver in (
            resolve_fidelity_auditor_model,
            _resolve_source_aware_editorial_guard_model,
            _resolve_profile_audit_model,
        ):
            assert resolver("deepseek-flash", {}) == policy["roles"]["audit"]["model"]
        for resolver in (_resolve_quality_alert_model, _resolve_profile_repair_model):
            assert resolver("deepseek-flash", {}) == policy["roles"]["repair"]["model"]
    finally:
        active_pipeline.reset(token)
