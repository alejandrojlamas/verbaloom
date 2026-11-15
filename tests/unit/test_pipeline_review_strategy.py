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


# --------------------------------------------------------------------------
# The cost-aware review policy is defined for the official OpenAI models too.
# --------------------------------------------------------------------------

OFFICIAL_OPENAI = "https://api.openai.com/v1/chat/completions"
STRICT_EPUB = {"refine": True, "audit_entire_book": True}


def _openai_job(model, *, endpoint=OFFICIAL_OPENAI, options=None):
    return {
        "llm_provider": "openai",
        "model": model,
        "llm_api_endpoint": endpoint,
        "file_type": "epub",
        "source_language": "English",
        "target_language": "Spanish",
        "prompt_options": dict(options or {}),
    }


def _deepseek_job(model, options=None):
    return {
        "llm_provider": "deepseek",
        "model": model,
        "file_type": "epub",
        "source_language": "English",
        "target_language": "Spanish",
        "prompt_options": dict(options or {}),
    }


def _policy(resolved, *, without=()):
    """Guard policy with model names removed, to compare across providers."""
    return {
        key: value
        for key, value in resolved.items()
        if not key.endswith("_model") and key not in without
    }


@pytest.mark.parametrize("options", [{}, STRICT_EPUB, {"refine": False}])
@pytest.mark.parametrize(
    "openai_model, deepseek_model",
    [("gpt-6.1-terra", "deepseek-v4-pro"), ("gpt-6-luna", "deepseek-flash")],
)
def test_openai_jobs_get_the_same_review_policy_as_deepseek(
    options, openai_model, deepseek_model
):
    openai = configure_editorial_guard_options(_openai_job(openai_model, options=options))
    deepseek = configure_editorial_guard_options(_deepseek_job(deepseek_model, options))

    assert _policy(openai) == _policy(deepseek)
    # The strong reviewer is always Terra, never a DeepSeek model name.
    reviewers = {key: value for key, value in openai.items() if key.endswith("_model")}
    assert reviewers
    assert "deepseek" not in " ".join(map(str, reviewers.values()))
    assert openai["source_aware_editorial_guard_model"] == "gpt-6.1-terra"
    assert openai["fidelity_supervisor_model"] == "gpt-6.1-terra"


def test_luna_does_the_bounded_rewrites_and_terra_the_final_audit():
    resolved = configure_editorial_guard_options(
        _openai_job("gpt-6-luna", options=STRICT_EPUB)
    )

    assert resolved["quality_alert_model"] == "gpt-6-luna"
    assert resolved["_quality_alert_model_policy"] == "primary_then_strict_final_audit"
    assert resolved["source_aware_editorial_guard_model"] == "gpt-6.1-terra"


def test_terra_reviews_its_own_strict_job():
    resolved = configure_editorial_guard_options(
        _openai_job("gpt-6.1-terra", options=STRICT_EPUB)
    )

    assert resolved["quality_alert_model"] == "gpt-6.1-terra"
    assert "_quality_alert_model_policy" not in resolved


@pytest.mark.parametrize(
    "model, endpoint",
    [
        # An OpenAI-compatible local server is not the verified model family.
        ("gpt-6.1-terra", "http://127.0.0.1:1234/v1/chat/completions"),
        ("local-model", "http://127.0.0.1:1234/v1/chat/completions"),
        # Unverified models on the official host keep their previous behavior.
        ("gpt-4o", OFFICIAL_OPENAI),
    ],
)
def test_other_openai_compatible_jobs_keep_their_previous_policy(model, endpoint):
    resolved = configure_editorial_guard_options(
        _openai_job(model, endpoint=endpoint, options=STRICT_EPUB)
    )

    assert not any(key.endswith("_model") for key in resolved)
    assert "fidelity_supervisor_retry" not in resolved


def test_leftover_deepseek_model_name_still_gets_the_openai_policy():
    resolved = configure_editorial_guard_options(
        _openai_job("deepseek-flash", options=STRICT_EPUB)
    )

    assert resolved["quality_alert_model"] == "gpt-6-luna"
    assert resolved["source_aware_editorial_guard_model"] == "gpt-6.1-terra"


def test_openai_pipeline_sets_strong_auditor_without_legacy_endpoint():
    baseline = {"llm_provider": "openai", "model": "gpt-6-luna"}
    policy = pipeline_preset("compatible", baseline)

    assert policy["roles"]["translate"]["model"] == "gpt-6-luna"
    assert policy["roles"]["audit"]["model"] == "gpt-6.1-terra"
    assert baseline == {"llm_provider": "openai", "model": "gpt-6-luna"}


def test_validated_openai_reference_controls_saved_pipeline_policy():
    baseline = _openai_job("gpt-6-luna", options=STRICT_EPUB)
    apply_pipeline(baseline, pipeline_preset("compatible", baseline))
    # Explicit role routing uses official endpoints, even if a legacy field
    # was absent from the checkpoint. Re-entry preserves the same policy.
    baseline.pop("llm_api_endpoint", None)
    first = deepcopy(configure_editorial_guard_options(baseline))

    assert first["quality_alert_model"] == "gpt-6-luna"
    assert first["fidelity_supervisor_model"] == "gpt-6.1-terra"
    assert configure_editorial_guard_options(baseline) == first


def test_unvalidated_reference_cannot_declare_local_endpoint_official():
    baseline = _openai_job("gpt-6-luna", endpoint="http://127.0.0.1:1234/v1/chat/completions")
    baseline["pipeline"] = {
        "review_reference": {"provider": "openai", "model": "gpt-6-luna"}
    }

    with pytest.raises(ValueError):
        configure_editorial_guard_options(baseline)
