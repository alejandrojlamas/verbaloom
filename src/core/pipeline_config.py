"""Serializable model-role configuration, independent of HTTP and Flask."""

from copy import deepcopy
from decimal import Decimal, InvalidOperation

ROLES = (
    "profile_prepare",
    "profile_review",
    "translate",
    "refine",
    "audit",
    "repair",
    "structure_repair",
)
ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}
MODELS = {
    "openai": {"gpt-6-luna", "gpt-6.1-terra"},
    "deepseek": {"deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"},
}
PRESETS = (
    "compatible", "openai_equivalent", "deepseek_source_audited", "openai_source_audited",
)


def validate_pipeline(value):
    """Reject unsupported/secret fields; return a fully explicit JSON snapshot."""
    if not isinstance(value, dict) or set(value) - {
        "version",
        "preset",
        "review_policy",
        "epub_repair_policy",
        "review_reference",
        "roles",
        "budget",
    }:
        raise ValueError("Invalid pipeline configuration fields")
    if (
        type(value.get("version")) is not int
        or value["version"] != 1
        or not isinstance(value.get("review_policy"), str)
        or value.get("review_policy") not in {"compatible", "source_audited_v1"}
    ):
        raise ValueError("Unsupported pipeline version or review policy")
    if not isinstance(value.get("preset"), str) or value["preset"] not in {
        "custom",
        "compatible",
        "openai_equivalent",
        "deepseek_source_audited",
        "openai_source_audited",
    }:
        raise ValueError("Unsupported pipeline preset")
    roles = value.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise ValueError("Every pipeline role must be configured explicitly")
    for name, role in roles.items():
        if not isinstance(role, dict) or set(role) != {
            "provider",
            "model",
            "reasoning_effort",
            "max_output_tokens",
        }:
            raise ValueError(f"Invalid fields for pipeline role {name}")
        provider = role["provider"]
        if (
            not isinstance(provider, str)
            or provider not in MODELS
            or not isinstance(role["model"], str)
            or role["model"] not in MODELS[provider]
        ):
            raise ValueError(f"Unsupported provider/model for pipeline role {name}")
        efforts = {"none", "low"} if provider == "openai" else {"none"}
        if (
            not isinstance(role["reasoning_effort"], str)
            or role["reasoning_effort"] not in efforts
        ):
            raise ValueError(f"Unsupported reasoning effort for pipeline role {name}")
        limit = role["max_output_tokens"]
        if type(limit) is not int or not 256 <= limit <= (
            128_000 if provider == "openai" else 384_000
        ):
            raise ValueError(f"Invalid output allowance for pipeline role {name}")
    budget = value.get("budget")
    if budget is not None:
        if not isinstance(budget, dict) or set(budget) != {"limit_usd", "max_calls"}:
            raise ValueError("Budget requires a gross USD cap and HTTP attempt cap")
        try:
            limit = Decimal(str(budget["limit_usd"]))
            if not limit.is_finite() or not 0 < limit <= 1_000_000:
                raise ValueError("Invalid gross USD budget")
        except InvalidOperation as exc:
            raise ValueError("Invalid gross USD budget") from exc
        if (
            type(budget["max_calls"]) is not int
            or not 1 <= budget["max_calls"] <= 1_000_000
        ):
            raise ValueError("Invalid HTTP attempt budget")
    result = deepcopy(value)
    if not isinstance(
        result.get("epub_repair_policy", "compatible"), str
    ) or result.get("epub_repair_policy", "compatible") not in {
        "compatible",
        "localized_v1",
    }:
        raise ValueError("Unsupported EPUB repair policy")
    result.setdefault("epub_repair_policy", "compatible")
    reference = result.get(
        "review_reference",
        {
            "provider": roles["translate"]["provider"],
            "model": roles["translate"]["model"],
        },
    )
    if (
        not isinstance(reference, dict)
        or set(reference) != {"provider", "model"}
        or not isinstance(reference["provider"], str)
        or reference["provider"] not in MODELS
        or not isinstance(reference["model"], str)
        or reference["model"] not in MODELS[reference["provider"]]
    ):
        raise ValueError("Invalid legacy review reference")
    result["review_reference"] = reference
    result.setdefault("budget", None)
    if budget is not None:
        result["budget"]["limit_usd"] = str(limit)
    return result


def pipeline_preset(name, config):
    """A/B preserve strategy; C/D test source-audited translation without universal rewrite."""
    if name not in {
        "compatible",
        "openai_equivalent",
        "deepseek_source_audited",
        "openai_source_audited",
    }:
        raise ValueError("Preset is not implemented")
    from .job_runtime_config import configure_editorial_guard_options

    reference = (
        validate_pipeline(config["pipeline"])["review_reference"]
        if config.get("pipeline")
        else {
            "provider": config.get("llm_provider") or "deepseek",
            "model": config.get("model") or "deepseek-v4-pro",
        }
    )
    provider, model = reference["provider"], reference["model"]
    if name == "deepseek_source_audited" and provider != "deepseek":
        raise ValueError("The DeepSeek strategy preset requires a DeepSeek baseline")
    baseline = deepcopy(config)
    baseline.pop("pipeline", None)
    baseline.update(llm_provider=provider, model=model)
    options = configure_editorial_guard_options(baseline)
    audit = (
        options.get("fidelity_supervisor_model")
        or options.get("profile_audit_model")
        or model
    )
    repair = (
        options.get("profile_repair_model")
        or options.get("quality_alert_model")
        or model
    )
    defaults = {
        "profile_prepare": "deepseek-flash" if provider == "deepseek" else "gpt-6-luna",
        "profile_review": audit,
        "translate": model,
        "refine": model,
        "audit": audit,
        "repair": repair,
        "structure_repair": "deepseek-flash"
        if provider == "deepseek"
        else "gpt-6-luna",
    }
    roles = {}
    for role, selected in defaults.items():
        target = provider
        if name in {"openai_equivalent", "openai_source_audited"}:
            target = "openai"
            selected = {
                "deepseek-flash": "gpt-6-luna",
                "deepseek-v4-flash": "gpt-6-luna",
                "deepseek-v4-pro": "gpt-6.1-terra",
            }.get(selected, selected)
        roles[role] = {
            "provider": target,
            "model": selected,
            "reasoning_effort": "none",
            "max_output_tokens": 8192,
        }
    return validate_pipeline(
        {
            "version": 1,
            "preset": name,
            "review_policy": "source_audited_v1"
            if name.endswith("source_audited")
            else "compatible",
            "epub_repair_policy": "localized_v1"
            if name.endswith("source_audited")
            else "compatible",
            "review_reference": reference,
            "roles": roles,
        }
    )


def apply_pipeline(config, pipeline):
    """Persist a validated snapshot; never modify editorial/profile decisions."""
    snapshot = validate_pipeline(pipeline)
    primary = snapshot["roles"]["refine" if config.get("refine_only") else "translate"]
    config["pipeline"] = snapshot
    config["llm_provider"] = primary["provider"]
    config["model"] = primary["model"]
    config["llm_api_endpoint"] = ENDPOINTS[primary["provider"]]
    return snapshot
