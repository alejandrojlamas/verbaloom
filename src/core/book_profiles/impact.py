"""Pre-run impact preview for book editorial profiles."""

from __future__ import annotations

from typing import Any

from src.core.editorial_knowledge_compiler import match_profile_entry_in_text

from .knowledge import build_profile_knowledge_base
from .models import BookProfile, ProfileGlossaryEntry
from .rendering import profile_glossary_match_summary


def build_profile_impact_preview(
    profile: BookProfile,
    text: str,
    *,
    purpose: str = "translation",
    max_terms: int = 80,
) -> dict[str, Any]:
    """Return a compact explanation of what a profile would affect."""
    value = text or ""
    summary = profile_glossary_match_summary(
        value,
        {
            "editorial_mode": "book_profile",
            "profile_id": profile.profile_id,
            "use_profile_glossary": True,
        },
        purpose=purpose,
    ) or {}
    approved_matches = [
        entry
        for entry in profile.approved_entries
        if _entry_matches(value, entry, purpose=purpose)
    ]
    pending_matches = [
        entry
        for entry in profile.pending_entries
        if _entry_matches(value, entry, purpose=purpose)
    ]
    translate_terms = [
        entry for entry in approved_matches
        if _entry_action(entry) == "translate"
    ]
    preserve_terms = [
        entry for entry in approved_matches
        if _entry_action(entry) == "preserve"
    ]
    kb = build_profile_knowledge_base(profile).to_dict()
    risks = _impact_risks(kb, approved_matches, pending_matches, summary)
    return {
        "profile": {
            "profile_id": profile.profile_id,
            "name": profile.name,
            "target_locale": profile.target_locale,
        },
        "purpose": purpose,
        "input_characters": len(value),
        "glossary": {
            "approved_total": profile.approved_count,
            "pending_total": profile.pending_count,
            "matched_approved": len(approved_matches),
            "matched_pending": len(pending_matches),
            "matched_for_prompt": int(summary.get("matched_terms") or 0),
            "rendered_for_prompt": int(summary.get("rendered_terms") or 0),
            "capped": bool(summary.get("capped")),
        },
        "terms_to_translate": [_entry_preview(entry) for entry in translate_terms[:max_terms]],
        "terms_to_preserve": [_entry_preview(entry) for entry in preserve_terms[:max_terms]],
        "pending_suggestions": [_entry_preview(entry) for entry in pending_matches[:max_terms]],
        "display_limits": {
            "max_terms_per_group": max_terms,
            "translate_truncated": max(0, len(translate_terms) - max_terms),
            "preserve_truncated": max(0, len(preserve_terms) - max_terms),
            "pending_truncated": max(0, len(pending_matches) - max_terms),
        },
        "risks": risks,
        "knowledge_summary": kb,
    }


def _entry_action(entry: ProfileGlossaryEntry) -> str:
    policy = (entry.translation_policy or entry.injection_policy or "").casefold()
    if "preserve" in policy:
        return "preserve"
    target = (entry.target or entry.render_target or "").strip()
    if target and target.casefold() != entry.source.casefold():
        return "translate"
    if entry.target_options:
        return "translate"
    return "preserve"


def _entry_preview(entry: ProfileGlossaryEntry) -> dict[str, Any]:
    return {
        "source": entry.source,
        "target": entry.target,
        "render_target": entry.render_target,
        "type": entry.entry_type,
        "status": entry.status,
        "confidence": entry.confidence,
        "translation_policy": entry.translation_policy,
        "injection_policy": entry.injection_policy,
        "rationale": entry.rationale,
    }


def _impact_risks(
    knowledge: dict[str, Any],
    approved_matches: list[ProfileGlossaryEntry],
    pending_matches: list[ProfileGlossaryEntry],
    prompt_summary: dict[str, Any],
) -> list[str]:
    risks: list[str] = []
    readiness = knowledge.get("prompt_readiness") or {}
    risks.extend(str(item) for item in readiness.get("warnings") or [] if str(item).strip())
    signal_index = knowledge.get("signal_index") or {}
    risks.extend(str(item) for item in signal_index.get("risk_flags") or [] if str(item).strip())
    if pending_matches:
        risks.append(f"{len(pending_matches)} matched profile suggestion(s) are still pending review.")
    if prompt_summary.get("capped"):
        risks.append("Matched glossary entries exceed the per-chunk prompt cap; only the highest-priority entries will be injected.")
    source_equals_target = [
        entry.source for entry in approved_matches
        if entry.target and entry.source.casefold() == entry.target.casefold()
        and _entry_action(entry) != "preserve"
    ]
    if source_equals_target:
        risks.append(
            "Some approved entries have source == target but are not marked preserve_exact: "
            + ", ".join(source_equals_target[:8])
        )
    return _unique(risks)


def _entry_matches(
    text: str,
    entry: ProfileGlossaryEntry,
    *,
    purpose: str,
) -> bool:
    """Use the same source/target-side relevance contract as prompt injection."""
    return match_profile_entry_in_text(entry, text, purpose=purpose)


def _unique(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
