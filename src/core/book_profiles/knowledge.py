"""Profile-scoped editorial knowledge base summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.core.editorial_knowledge import build_editorial_knowledge_base

from .artifacts import editorial_artifact_counts, editorial_artifact_saturation
from .models import BookProfile, ProfileGlossaryEntry


_PRESERVE_POLICIES = {"preserve_exact", "preserve", "do_not_translate", "keep_source"}
_TRANSLATE_POLICIES = {"translate", "translate_consistently", "canonical_translation"}


@dataclass(frozen=True)
class ProfileKnowledgeBase:
    profile_id: str
    name: str
    target_locale: str = ""
    source_name: str = ""
    generated_profile: bool = False
    profile_goal: str = ""
    glossary: dict[str, Any] = field(default_factory=dict)
    editorial_map: dict[str, Any] = field(default_factory=dict)
    signal_index: dict[str, Any] = field(default_factory=dict)
    prompt_readiness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "target_locale": self.target_locale,
            "source_name": self.source_name,
            "generated_profile": self.generated_profile,
            "profile_goal": self.profile_goal,
            "glossary": self.glossary,
            "editorial_map": self.editorial_map,
            "signal_index": self.signal_index,
            "prompt_readiness": self.prompt_readiness,
        }


def build_profile_knowledge_base(profile: BookProfile) -> ProfileKnowledgeBase:
    """Build a UI/API friendly summary of one profile's editorial knowledge."""

    entries = list(profile.glossary_entries)
    approved = [entry for entry in entries if entry.approved]
    pending = [entry for entry in entries if entry.pending]
    rejected = [entry for entry in entries if entry.rejected]
    translations = [entry for entry in approved if _is_translation_entry(entry)]
    preserve = [entry for entry in approved if _is_preserve_entry(entry)]
    source_equals_target = [
        entry for entry in approved
        if entry.target and entry.source.casefold() == entry.target.casefold()
    ]
    by_type = Counter(entry.entry_type or "term" for entry in entries)
    saturation = editorial_artifact_saturation(profile.editorial_artifacts)
    saturated_buckets = [
        key for key, value in saturation.items()
        if value.get("saturated")
    ]
    artifact_counts = editorial_artifact_counts(profile.editorial_artifacts)

    glossary = {
        "total": len(entries),
        "approved": len(approved),
        "pending": len(pending),
        "rejected": len(rejected),
        "translated_terms": len(translations),
        "preserve_terms": len(preserve),
        "source_equals_target_approved": len(source_equals_target),
        "by_type": dict(sorted(by_type.items())),
        "translation_candidates": [_entry_summary(entry) for entry in translations[:80]],
        "preserve_candidates": [_entry_summary(entry) for entry in preserve[:80]],
        "pending_review": [_entry_summary(entry) for entry in pending[:120]],
        "source_equals_target": [_entry_summary(entry) for entry in source_equals_target[:80]],
    }
    editorial_knowledge = build_editorial_knowledge_base(profile=profile)
    editorial_map = {
        "counts": artifact_counts,
        "saturation": saturation,
        "saturated_buckets": saturated_buckets,
    }
    signal_index = _signal_index_summary(profile.editorial_signal_index)
    prompt_readiness = _prompt_readiness(
        approved=approved,
        pending=pending,
        translations=translations,
        source_equals_target=source_equals_target,
        saturated_buckets=saturated_buckets,
        artifact_counts=artifact_counts,
        signal_index=signal_index,
    )
    return ProfileKnowledgeBase(
        profile_id=profile.profile_id,
        name=profile.name,
        target_locale=profile.target_locale,
        source_name=str(profile.raw_config.get("source_name") or ""),
        generated_profile=bool(profile.raw_config.get("generated_profile")),
        profile_goal=_profile_goal(profile),
        glossary=glossary,
        editorial_map=editorial_map,
        signal_index=signal_index,
        prompt_readiness={
            **prompt_readiness,
            "editorial_knowledge": editorial_knowledge.prompt_diagnostics(),
        },
    )


def _entry_summary(entry: ProfileGlossaryEntry) -> dict[str, Any]:
    data = {
        "source": entry.source,
        "target": entry.target,
        "target_options": list(entry.target_options),
        "type": entry.entry_type,
        "status": entry.status,
        "confidence": entry.confidence,
        "occurrences": entry.occurrences,
        "translation_policy": entry.translation_policy,
        "injection_policy": entry.injection_policy,
        "review_status": entry.review_status,
        "source_file": entry.source_file,
    }
    return {key: value for key, value in data.items() if value not in ("", [], 0, 0.0)}


def _is_translation_entry(entry: ProfileGlossaryEntry) -> bool:
    policy = (entry.translation_policy or entry.injection_policy or "").strip().lower()
    if policy in _TRANSLATE_POLICIES:
        return True
    return bool(entry.target and entry.source.casefold() != entry.target.casefold())


def _is_preserve_entry(entry: ProfileGlossaryEntry) -> bool:
    policy = (entry.translation_policy or entry.injection_policy or "").strip().lower()
    if policy in _PRESERVE_POLICIES:
        return True
    if entry.target and entry.source.casefold() == entry.target.casefold():
        return True
    return not entry.target and entry.approved


def _profile_goal(profile: BookProfile) -> str:
    rules = profile.raw_config.get("business_rules")
    if isinstance(rules, dict) and rules.get("goal"):
        return str(rules["goal"])
    return str(profile.raw_config.get("profile_goal") or profile.raw_config.get("goal") or "")


def _prompt_readiness(
    *,
    approved: list[ProfileGlossaryEntry],
    pending: list[ProfileGlossaryEntry],
    translations: list[ProfileGlossaryEntry],
    source_equals_target: list[ProfileGlossaryEntry],
    saturated_buckets: list[str],
    artifact_counts: dict[str, int],
    signal_index: dict[str, Any],
) -> dict[str, Any]:
    warnings: list[str] = []
    if not approved and not pending and not any(artifact_counts.values()):
        warnings.append("empty_profile")
    if len(source_equals_target) > max(8, len(translations) * 2):
        warnings.append("many_preserve_exact_entries")
    if pending:
        warnings.append("pending_review_items")
    if saturated_buckets:
        warnings.append("editorial_map_saturated")
    if signal_index.get("risk_flags"):
        warnings.extend(
            flag for flag in signal_index.get("risk_flags", [])
            if flag not in warnings
        )
    return {
        "ready": "empty_profile" not in warnings,
        "warnings": warnings,
        "recommended_action": _recommended_action(warnings),
    }


def _signal_index_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        return {
            "available": False,
            "risk_flags": [],
        }
    buckets = {}
    for key in (
        "local_candidates",
        "reviewed_candidates",
        "approved_entries",
        "pending_suggestions",
        "llm_suggestions",
    ):
        bucket = value.get(key) if isinstance(value.get(key), dict) else {}
        buckets[key] = {
            "total": int(bucket.get("total") or 0),
            "stored_items": int(bucket.get("stored_items") or 0),
            "items_truncated": int(bucket.get("items_truncated") or 0),
            "by_kind": dict(bucket.get("by_kind") or {}),
        }
    return {
        "available": True,
        "version": value.get("version") or "",
        "source_stats": dict(value.get("source_stats") or {}),
        "coverage": dict(value.get("coverage") or {}),
        "risk_flags": list(value.get("risk_flags") or []),
        "buckets": buckets,
    }


def _recommended_action(warnings: list[str]) -> str:
    if "empty_profile" in warnings:
        return "prepare_or_import_profile_knowledge"
    if "editorial_map_saturated" in warnings:
        return "refresh_profile_with_full_signal_index"
    if "many_preserve_exact_entries" in warnings:
        return "review_source_equals_target_entries"
    if "pending_review_items" in warnings:
        return "review_pending_glossary_suggestions"
    if "preserve_exact_dominates_translation_terms" in warnings:
        return "review_source_equals_target_entries"
    if "many_unclassified_local_candidates" in warnings:
        return "refresh_or_expand_profile_review"
    return "ready"
