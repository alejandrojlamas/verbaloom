"""Experimental review choices, separate from provider/model assignments.

The first ablation removes universal rewriting, not semantic coverage. Mechanical
calibration missed meaning and voice defects, so sampled coverage is not a preset.
"""

from __future__ import annotations

STRATEGY = "source_audited_v1"
OPTION = "experimental_review_strategy"


def apply_review_strategy(config: dict, *, file_type: str | None = None) -> dict:
    """Resolve an explicit pipeline strategy without changing requested settings."""
    options = config.setdefault("prompt_options", {})
    pipeline = config.get("pipeline") or {}
    if pipeline.get("review_policy") != STRATEGY:
        options.pop(OPTION, None)
        return options
    kind = str(file_type or config.get("file_type") or "").lower()
    transform = config.get("text_transform_mode") or options.get("text_transform_mode")
    if config.get("refine_only"):
        reason = "explicit_refinement"
    elif transform or str(config.get("operation", "")).lower() == "transform":
        reason = "transformation_contract"
    elif str(options.get("profile_strength", "")).lower() in {"strict", "full", "llm"}:
        reason = "strict_profile"
    else:
        reason = "source_audited_translation"
    enabled = reason == "source_audited_translation"
    options[OPTION] = {
        "version": 1,
        "name": STRATEGY,
        "enabled": enabled,
        "reason": reason,
        "semantic_coverage": "full",
        "skip_universal_refinement": enabled,
        "final_unit_audit": enabled
        and kind in {"epub", "docx"}
        and not options.get("text_first_pipeline_active")
        and not options.get("plain_text_mode"),
    }
    return options


def bind_runtime_strategy(options: dict | None, *, file_type: str) -> dict:
    """Bind low-level format entrypoints to the same task-local configuration."""
    from .pipeline_runtime import active_pipeline

    result = dict(options or {})
    runtime = active_pipeline.get()
    if runtime is not None:
        prior = result.get(OPTION) or {}
        apply_review_strategy(
            {
                "pipeline": runtime.config,
                "prompt_options": result,
                "file_type": file_type,
                "refine_only": prior.get("reason") == "explicit_refinement",
                "operation": "transform"
                if prior.get("reason") == "transformation_contract"
                else "translate",
            }
        )
    return result


def source_audited(options) -> bool:
    policy = (options or {}).get(OPTION) or {}
    return policy.get("name") == STRATEGY and policy.get("enabled") is True


def skip_universal_refinement(options) -> bool:
    return source_audited(options) and (
        options[OPTION].get("skip_universal_refinement") is True
    )


def needs_final_unit_audit(options) -> bool:
    return source_audited(options) and options[OPTION].get("final_unit_audit") is True


def refinement_plan(config: dict) -> dict[str, bool]:
    options = config.get("prompt_options") or {}
    if config.get("pipeline") is not None:
        options = apply_review_strategy({**config, "prompt_options": dict(options)})
    skip = skip_universal_refinement(options)
    inline = bool(options.get("inline_refinement")) and not skip
    return {
        "inline_refinement": inline,
        "refine_after": bool(config.get("refine_after")) and not inline and not skip,
    }


def semantic_audit_reason(decision, options) -> str:
    if not source_audited(options):
        return ""
    if not decision.source_chars or not decision.candidate_chars:
        return "source_audited:empty_or_structural"
    if needs_final_unit_audit(options) and not str(decision.phase).startswith("final_"):
        if decision.rejections and not str(decision.phase).startswith(
            ("refinement", "inline_refinement")
        ):
            return "source_audited:hard_check_escalation"
        return "source_audited:deferred_to_final"
    return "source_audited:complete_semantic_coverage"


def valid_saved_semantic_approval(
    source: str, candidate: str, assessment: dict | None
) -> bool:
    from .fidelity_supervisor import _hash_text

    value = assessment if isinstance(assessment, dict) else {}
    return (
        value.get("accepted") is True
        and isinstance(value.get("judge_decision"), str)
        and value.get("judge_decision") in {"pass", "warn"}
        and bool(value.get("judge_model"))
        and value.get("source_hash") == _hash_text(source)
        and value.get("candidate_hash") == _hash_text(candidate)
    )
