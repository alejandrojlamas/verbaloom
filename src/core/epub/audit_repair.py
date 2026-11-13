"""Pure dependency and comparison rules for the existing EPUB unit checkpoint."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

from .unit_contract import OPERATIONAL_PROMPT_OPTIONS, prompt_versions, text_sha256


def bounded_count(value: Any) -> int:
    """Malformed saved counters exhaust their allowance, never replenish it."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 2**63 - 1
    return value


def repair_limit(options: Mapping[str, Any]) -> int:
    if options.get("repair_until_pass") is False:
        return 0
    value = options.get("epub_audit_repair_rounds", options.get("max_repair_rounds", 2))
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else 0
    )


def contract_hash(
    record,
    chunk,
    options,
    *,
    source_language,
    target_language,
    model_name,
    provider,
    placeholder_format,
) -> str:
    # Limits govern future reservations, not the meaning of a saved judgment.
    excluded = OPERATIONAL_PROMPT_OPTIONS | {
        "_fidelity_adjudication_context",
        "_candidate_results",
        "_fidelity_report",
        "_editorial_quality_report",
        "_combined_quality_audit_records",
        "_last_target_language_gate_rejection",
    }

    def dependencies(value):
        if isinstance(value, Mapping):
            return {
                str(k): dependencies(v)
                for k, v in value.items()
                if str(k) not in excluded
                and not any(
                    s in str(k).lower() for s in ("api_key", "secret", "credential")
                )
            }
        if isinstance(value, (tuple, list)):
            return [dependencies(v) for v in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    profile = None
    if options.get("profile_id"):
        from src.core.book_profiles.loader import load_book_profile

        loaded = load_book_profile(str(options["profile_id"]))
        if loaded is not None:
            profile = asdict(loaded)
    payload = {
        "version": 1,
        "unit_id": record.get("unit_id"),
        "source": chunk.get("text"),
        "source_language": source_language,
        "target_language": target_language,
        "model": model_name,
        "provider": provider,
        "tags": chunk.get("local_tag_map"),
        "indices": chunk.get("global_indices"),
        "placeholder_format": placeholder_format,
        "versions": prompt_versions(),
        "stages": record.get("stage_fingerprints"),
        "options": options,
        "profile": profile,
    }
    return text_sha256(
        json.dumps(dependencies(payload), sort_keys=True, ensure_ascii=False)
    )


def new_candidate(text: str, *, origin: str, base_hash: str = "") -> dict:
    return {
        "text": text,
        "hash": text_sha256(text),
        "origin": origin,
        "base_hash": base_hash,
        "assessments": {},
        "audit_calls": {},
    }


def strictly_better(previous, candidate) -> bool:
    """Never use confidence, style scores or recency as an improvement signal.

    Acceptance remains the existing hard/semantic contract. For two rejected
    candidates, require a strict reduction in identified objective defects with
    no newly reported defect. Paraphrased findings are conservatively incomparable.
    """
    if previous.accepted:
        return False
    if candidate.accepted:
        return True

    def findings(decision):
        objective = set()
        for name in (
            "judge_missing_from_source",
            "judge_added_not_in_source",
            "judge_changed_facts",
            "judge_censored_or_softened",
        ):
            objective.update(
                (name, " ".join(text.split()).casefold())
                for text in getattr(decision, name)
                if text.strip()
            )
        local = {
            (i.code, i.detail or i.message)
            for i in decision.rejections
            if i.code != "fidelity_judge_reject"
        }
        general = {("judge_issue", text) for text in decision.judge_issues}
        return objective | local, objective | local | general

    old_objective, old_all = findings(previous)
    new_objective, new_all = findings(candidate)
    return new_objective < old_objective and new_all <= old_all
