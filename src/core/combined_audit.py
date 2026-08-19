"""Combined quality audit contract for alerted chunk candidates.

The translation pipeline has three independent reviewers: editorial quality,
source fidelity, and active book profile compliance.  This module lets alerted
chunks ask one structured reviewer for all three decisions, then reuses the
existing per-axis parsers and decision mergers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from src.prompts.security import UNTRUSTED_BOOK_CONTENT_SECTION
from src.utils.json_extraction import extract_tagged_payload, loads_first_json_object

from .book_profiles.audit import ProfileAuditResult
from .editorial_quality import QualityDecision
from .fidelity_supervisor import FidelityDecision

COMBINED_AUDIT_TAG_IN = "<COMBINED_QUALITY_AUDIT_JSON>"
COMBINED_AUDIT_TAG_OUT = "</COMBINED_QUALITY_AUDIT_JSON>"


@dataclass(frozen=True)
class CombinedAuditPrompt:
    system: str
    user: str


def combined_quality_audit_enabled(prompt_options: Optional[Mapping[str, Any]]) -> bool:
    """Return whether alerted chunks may use the combined quality auditor."""

    if prompt_options and prompt_options.get("combined_quality_audit") is False:
        return False
    return True


def build_combined_quality_audit_prompt(
    *,
    source_text: str,
    draft_text: str,
    candidate_text: str,
    source_language: str = "",
    target_language: str = "",
    section: str = "",
    phase: str = "refinement",
    editorial_decision: Optional[QualityDecision] = None,
    fidelity_decision: Optional[FidelityDecision] = None,
    profile_result: Optional[ProfileAuditResult] = None,
    profile_id: str = "",
    profile_policy: str = "",
    profile_audit_prompt: str = "",
    glossary_block: str = "",
) -> CombinedAuditPrompt:
    """Build one JSON audit prompt covering editorial, fidelity, and profile axes."""

    local_payload = {
        "editorial": _quality_decision_payload(editorial_decision),
        "fidelity": _fidelity_decision_payload(fidelity_decision),
        "profile": _profile_result_payload(profile_result),
    }

    system = f"""You are an independent bilingual quality auditor for long-form book processing.

{UNTRUSTED_BOOK_CONTENT_SECTION}

Audit the candidate against the source in three axes at once:
1. editorial: should the refined candidate replace the draft?
2. fidelity: does the candidate preserve all source content without additions, omissions, censorship, or fact changes?
3. profile: does the candidate satisfy the active book profile and approved glossary?

Do not rewrite the text. Do not improve it. Return only JSON wrapped in {COMBINED_AUDIT_TAG_IN} and {COMBINED_AUDIT_TAG_OUT}.

Use the local prechecks as hints, not as final truth. If the source is visibly damaged by OCR or pagination,
judge semantic content and meaningful structure rather than preserving extraction damage. For same-language
modernization or transformation, accept safe syntax and diction updates, but reject changes to speaker/addressee,
grammatical person, facts, chronology, relationships, intentional ambiguity, censorship, or unsupported explanation.

JSON schema:
{{
  "editorial": {{
    "decision": "accept" | "reject" | "repair" | "flag",
    "confidence": 0.0,
    "reason": "short reason",
    "issues": ["short issue names"],
    "missing_from_source": ["meaningful source items absent from candidate"],
    "added_not_in_source": ["candidate meanings unsupported by source"],
    "weird_symbols": ["reader-visible strange symbols"],
    "structure_score": 0.0
  }},
  "fidelity": {{
    "verdict": "pass" | "warn" | "fail" | "repair_needed",
    "confidence": 0.0,
    "reason": "short reason",
    "issues": ["short issue labels"],
    "missing_from_source": ["source meanings absent from candidate"],
    "added_not_in_source": ["candidate meanings unsupported by source"],
    "changed_facts": ["numbers, names, dates, chronology, causality changed"],
    "censored_or_softened": ["source content softened, sanitized, or censored"],
    "structure_issues": ["paragraph/list/title/order problems"],
    "evidence_source": ["short source excerpts"],
    "evidence_candidate": ["short candidate excerpts"]
  }},
  "profile": {{
    "scores": {{}},
    "overall_decision": "pass" | "warn" | "fail",
    "issues": [
      {{
        "type": "editorial_consistency",
        "severity": "low" | "medium" | "high",
        "source_excerpt": "",
        "candidate_excerpt": "",
        "reason": "",
        "suggested_fix": "",
        "glossary_action": "none"
      }}
    ],
    "glossary_suggestions": [],
    "summary": "short profile summary"
  }},
  "repair": {{
    "needed": false,
    "focus": [
      {{"excerpt": "", "axis": "editorial|fidelity|profile", "instruction": ""}}
    ]
  }},
  "worst": "pass" | "warn" | "fail"
}}"""

    profile_context = ""
    if profile_id or profile_policy or profile_audit_prompt or glossary_block:
        profile_context = f"""
# ACTIVE PROFILE
{profile_id or '(none)'}

# PROFILE POLICY
{profile_policy or '(No profile policy text was provided.)'}

# PROFILE AUDIT RUBRIC
{profile_audit_prompt or '(Use general profile compliance: voice, locale, glossary, and isolation.)'}

# APPROVED PROFILE GLOSSARY
{glossary_block or '(No matching approved entries for this chunk.)'}
""".strip()

    user = f"""# AUDIT CONTEXT
Phase: {phase or 'refinement'}
Section: {section or 'Document'}
Source language: {source_language or 'unknown'}
Target language: {target_language or 'unknown'}

{profile_context}

# LOCAL PRECHECKS
{json.dumps(local_payload, ensure_ascii=False)}

# ORIGINAL SOURCE
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

# INITIAL DRAFT
<DRAFT_TEXT>
{draft_text}
</DRAFT_TEXT>

# CANDIDATE
<CANDIDATE_TEXT>
{candidate_text}
</CANDIDATE_TEXT>

Return the combined JSON assessment now."""

    return CombinedAuditPrompt(system=system.strip(), user=user.strip())


def parse_combined_quality_audit_response(text: str) -> Optional[dict[str, Any]]:
    """Parse and normalize a combined quality audit JSON payload."""

    if not text:
        return None
    payload = (
        extract_tagged_payload(text, COMBINED_AUDIT_TAG_IN, COMBINED_AUDIT_TAG_OUT)
        or text
    )
    parsed = loads_first_json_object(payload)
    if not isinstance(parsed, Mapping):
        return None

    editorial = editorial_assessment_from_combined(parsed)
    fidelity = fidelity_assessment_from_combined(parsed)
    profile = profile_payload_from_combined(parsed)
    if not any((editorial, fidelity, profile)):
        return None

    normalized = dict(parsed)
    if editorial:
        normalized["editorial"] = editorial
    if fidelity:
        normalized["fidelity"] = fidelity
    if profile:
        normalized["profile"] = profile
    worst = str(normalized.get("worst") or "").strip().lower()
    if worst not in {"pass", "warn", "fail"}:
        worst = _worst_from_axes(editorial, fidelity, profile)
    normalized["worst"] = worst
    return normalized


def editorial_assessment_from_combined(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    raw = payload.get("editorial")
    if not isinstance(raw, Mapping):
        return None
    decision = str(raw.get("decision") or raw.get("verdict") or "").strip().lower()
    decision = {
        "pass": "accept",
        "warn": "flag",
        "fail": "reject",
        "repair_needed": "repair",
    }.get(decision, decision)
    if decision not in {"accept", "reject", "repair", "flag"}:
        decision = "flag"
    return {
        "decision": decision,
        "confidence": _float(raw.get("confidence"), 0.0, 1.0),
        "reason": str(raw.get("reason") or raw.get("summary") or "").strip(),
        "issues": _string_list(raw.get("issues")),
        "missing_from_source": _string_list(raw.get("missing_from_source")),
        "added_not_in_source": _string_list(raw.get("added_not_in_source")),
        "weird_symbols": _string_list(raw.get("weird_symbols")),
        "structure_score": _optional_float(raw.get("structure_score")),
    }


def fidelity_assessment_from_combined(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    raw = payload.get("fidelity")
    if not isinstance(raw, Mapping):
        return None
    verdict = str(raw.get("verdict") or raw.get("decision") or "").strip().lower()
    verdict = {
        "accept": "pass",
        "reject": "fail",
        "repair": "repair_needed",
        "flag": "warn",
    }.get(verdict, verdict)
    if verdict not in {"pass", "warn", "fail", "repair_needed"}:
        verdict = "warn"
    return {
        "verdict": verdict,
        "confidence": _float(raw.get("confidence"), 0.0, 1.0),
        "reason": str(raw.get("reason") or raw.get("summary") or "").strip(),
        "issues": _string_list(raw.get("issues")),
        "missing_from_source": _string_list(raw.get("missing_from_source")),
        "added_not_in_source": _string_list(raw.get("added_not_in_source")),
        "changed_facts": _string_list(raw.get("changed_facts")),
        "censored_or_softened": _string_list(raw.get("censored_or_softened")),
        "structure_issues": _string_list(raw.get("structure_issues")),
        "evidence_source": _string_list(raw.get("evidence_source")),
        "evidence_candidate": _string_list(raw.get("evidence_candidate")),
    }


def profile_payload_from_combined(payload: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    raw = payload.get("profile")
    if not isinstance(raw, Mapping):
        return None
    result = dict(raw)
    decision = str(
        result.get("overall_decision") or result.get("decision") or result.get("verdict") or ""
    ).strip().lower()
    if decision not in {"pass", "warn", "fail"}:
        decision = "warn"
    result["overall_decision"] = decision
    result.setdefault("scores", {})
    result.setdefault("issues", [])
    result.setdefault("glossary_suggestions", [])
    result.setdefault("summary", "")
    return result


def _quality_decision_payload(decision: Optional[QualityDecision]) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "accepted": decision.accepted,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "detail": issue.detail,
            }
            for issue in decision.issues
        ],
    }


def _fidelity_decision_payload(decision: Optional[FidelityDecision]) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "accepted": decision.accepted,
        "issues": [
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "detail": issue.detail,
            }
            for issue in decision.issues
        ],
    }


def _profile_result_payload(result: Optional[ProfileAuditResult]) -> dict[str, Any]:
    if result is None:
        return {}
    return result.to_dict()


def _worst_from_axes(
    editorial: Optional[Mapping[str, Any]],
    fidelity: Optional[Mapping[str, Any]],
    profile: Optional[Mapping[str, Any]],
) -> str:
    rank = 0
    if editorial:
        rank = max(rank, {"accept": 0, "flag": 1, "repair": 1, "reject": 2}.get(str(editorial.get("decision")), 1))
    if fidelity:
        rank = max(rank, {"pass": 0, "warn": 1, "repair_needed": 1, "fail": 2}.get(str(fidelity.get("verdict")), 1))
    if profile:
        rank = max(rank, {"pass": 0, "warn": 1, "fail": 2}.get(str(profile.get("overall_decision")), 1))
    return ("pass", "warn", "fail")[rank]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
