"""Profile-aware editorial audit helpers.

The LLM auditor prompt lives in the active profile; this module provides the
token-free precheck and report primitives that make profile isolation auditable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from src.core.editorial_knowledge import build_editorial_knowledge_base
from src.core.locale_quality import (
    count_spanish_modernization_residue,
    format_spanish_modernization_residue,
)
from src.prompts.security import UNTRUSTED_BOOK_CONTENT_SECTION
from src.utils.json_extraction import extract_tagged_payload, loads_first_json_object

from .detectors import contains_profile_term, detector_hits
from .loader import BookProfileError, load_book_profile
from .models import BookProfile, ProfileDetector, ProfileGlossaryEntry

PROFILE_AUDIT_TAG_IN = "<PROFILE_AUDIT_JSON>"
PROFILE_AUDIT_TAG_OUT = "</PROFILE_AUDIT_JSON>"


@dataclass(frozen=True)
class ProfileAuditIssue:
    issue_type: str
    severity: str
    source_excerpt: str = ""
    candidate_excerpt: str = ""
    reason: str = ""
    suggested_fix: str = ""
    glossary_action: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.issue_type,
            "severity": self.severity,
            "source_excerpt": self.source_excerpt,
            "candidate_excerpt": self.candidate_excerpt,
            "reason": self.reason,
            "suggested_fix": self.suggested_fix,
            "glossary_action": self.glossary_action,
        }


@dataclass
class ProfileAuditResult:
    profile_id: str
    scores: dict[str, float]
    overall_decision: str
    issues: list[ProfileAuditIssue] = field(default_factory=list)
    glossary_suggestions: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "scores": self.scores,
            "overall_decision": self.overall_decision,
            "issues": [issue.to_dict() for issue in self.issues],
            "glossary_suggestions": self.glossary_suggestions,
            "summary": self.summary,
        }


SCORE_FIELDS = (
    "content_fidelity",
    "facts_names_order",
    "editorial_consistency",
    "voice_differentiation",
    "no_censorship_summary_omission",
    "glossary_compliance",
    "profile_isolation",
)

PROFILE_GOAL_SCORE_FIELDS: dict[str, tuple[str, ...]] = {
    "faithful_translation": (
        "terminology_translation",
        "name_consistency",
        "target_locale_naturalness",
        "register_consistency",
        "authorial_voice",
    ),
    "academic_translation": (
        "terminology_translation",
        "conceptual_precision",
        "citation_and_apparatus_handling",
        "table_formula_handling",
    ),
    "audiobook": (
        "listenability",
        "note_intrusion_control",
        "caption_integration",
        "oral_flow",
    ),
    "explanatory_rewrite": (
        "conceptual_clarity",
        "university_level_depth",
        "plain_language_explanation",
        "interpretive_fidelity",
    ),
    "modernization": (
        "orthographic_modernization",
        "syntactic_modernization",
        "contemporary_naturalness",
        "authorial_voice",
        "period_language_removal",
        "voice_register_preservation",
    ),
    "literary_polish": (
        "authorial_voice",
        "target_locale_naturalness",
        "rhythm_and_flow",
        "register_consistency",
    ),
}

AUDIT_DIMENSION_PRESETS: dict[str, tuple[str, ...]] = {
    # ``editorial_full`` means all dimensions relevant to the active goal. It
    # intentionally carries no work- or locale-specific fields.
    "editorial_full": (),
    "translation_audiobook_full": (
        *PROFILE_GOAL_SCORE_FIELDS["faithful_translation"],
        *PROFILE_GOAL_SCORE_FIELDS["audiobook"],
    ),
}

CRITICAL_SCORE_FIELDS = {
    "content_fidelity",
    "facts_names_order",
    "no_censorship_summary_omission",
    "profile_isolation",
}

CRITICAL_ISSUE_TYPES = {
    "fidelity_error",
    "profile_contamination",
    "hardcoded_translation_risk",
}


def build_profile_audit_prompt(
    source_text: str,
    candidate_text: str,
    *,
    profile: BookProfile,
    glossary_block: str = "",
    local_issues: Optional[list[ProfileAuditIssue]] = None,
) -> tuple[str, str]:
    prompt_text = profile.prompt_texts.get("audit") or _DEFAULT_AUDIT_PROMPT
    score_fields = score_fields_for_profile(profile)
    local_payload = [issue.to_dict() for issue in (local_issues or [])]
    system = f"""{prompt_text}

{UNTRUSTED_BOOK_CONTENT_SECTION}

Profile-specific score dimensions to include in the JSON scores object:
{", ".join(score_fields)}

Return only valid JSON wrapped in {PROFILE_AUDIT_TAG_IN} and {PROFILE_AUDIT_TAG_OUT}.
Do not include markdown or commentary outside the tags.
"""
    user = f"""# ACTIVE PROFILE
{profile.profile_id}

# APPROVED PROFILE GLOSSARY
{glossary_block or '(No matching approved entries for this chunk.)'}

# LOCAL PRECHECK ISSUES
{json.dumps(local_payload, ensure_ascii=False)}

# SOURCE
{source_text}

# CANDIDATE
{candidate_text}
"""
    return system.strip(), user.strip()


def parse_profile_audit_response(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    payload = (
        extract_tagged_payload(text, PROFILE_AUDIT_TAG_IN, PROFILE_AUDIT_TAG_OUT)
        or text
    )
    return loads_first_json_object(payload)


def profile_audit_result_from_payload(
    profile_id: str,
    payload: Mapping[str, Any],
    *,
    local_issues: Optional[list[ProfileAuditIssue]] = None,
    min_score: float = 8.5,
    score_fields: Optional[tuple[str, ...]] = None,
) -> ProfileAuditResult:
    active_score_fields = score_fields or SCORE_FIELDS
    scores = _default_scores(active_score_fields)
    raw_scores = payload.get("scores") or {}
    if isinstance(raw_scores, Mapping):
        for field in active_score_fields:
            if field not in raw_scores:
                continue
            try:
                scores[field] = max(0.0, min(10.0, float(raw_scores[field])))
            except (TypeError, ValueError):
                continue

    issues = list(local_issues or [])
    for raw in payload.get("issues") or []:
        if not isinstance(raw, Mapping):
            if isinstance(raw, str) and raw.strip():
                issues.append(ProfileAuditIssue(
                    issue_type="editorial_consistency",
                    severity="medium",
                    reason=raw.strip(),
                ))
            continue
        issue_type = str(raw.get("type") or "editorial_consistency").strip()
        severity = str(raw.get("severity") or "medium").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "medium"
        issues.append(ProfileAuditIssue(
            issue_type=issue_type,
            severity=severity,
            source_excerpt=str(raw.get("source_excerpt") or raw.get("source") or raw.get("original_excerpt") or ""),
            candidate_excerpt=str(raw.get("candidate_excerpt") or raw.get("candidate") or raw.get("modernized_excerpt") or ""),
            reason=str(raw.get("reason") or raw.get("rationale") or raw.get("explanation") or ""),
            suggested_fix=str(raw.get("suggested_fix") or raw.get("recommendation") or raw.get("fix") or ""),
            glossary_action=str(raw.get("glossary_action") or "none"),
        ))

    for issue in local_issues or []:
        _apply_issue_penalty(scores, issue)

    _append_score_floor_issues(
        scores,
        issues,
        min_score=min_score,
    )

    computed_decision = _decision_from_scores(scores, issues, min_score)
    reported_decision = str(payload.get("overall_decision") or "").strip().lower()
    if reported_decision not in {"pass", "warn", "fail"}:
        reported_decision = computed_decision
    decision = _worse_decision(computed_decision, reported_decision)

    raw_suggestions = payload.get("glossary_suggestions") or []
    suggestions = [
        dict(item) for item in raw_suggestions
        if isinstance(item, Mapping)
    ]

    return ProfileAuditResult(
        profile_id=profile_id,
        scores=scores,
        overall_decision=decision,
        issues=issues,
        glossary_suggestions=suggestions,
        summary=str(payload.get("summary") or ""),
    )


def score_fields_for_profile(profile: BookProfile) -> tuple[str, ...]:
    """Return base, goal, preset, and explicitly declared dimensions."""

    goal = _profile_goal(profile)
    fields = list(SCORE_FIELDS)
    for field in PROFILE_GOAL_SCORE_FIELDS.get(goal, ()):
        if field not in fields:
            fields.append(field)
    configured = profile.raw_config.get("audit_score_fields")
    if configured is None:
        configured = profile.raw_config.get("audit_dimensions")
    declared: list[str] = []
    if isinstance(configured, (list, tuple)):
        declared = [str(item).strip() for item in configured]
    elif isinstance(configured, str):
        preset = configured.strip().lower()
        declared = list(AUDIT_DIMENSION_PRESETS.get(preset, ()))
        if not declared and "," in configured:
            declared = [part.strip() for part in configured.split(",")]
    for field in declared:
        normalized = re.sub(r"[^a-z0-9_]+", "_", field.casefold()).strip("_")
        if normalized and normalized not in fields:
            fields.append(normalized)
    return tuple(fields)


def _profile_goal(profile: BookProfile) -> str:
    rules = profile.raw_config.get("business_rules")
    if isinstance(rules, Mapping) and rules.get("goal"):
        return str(rules["goal"]).strip()
    return str(
        profile.raw_config.get("profile_goal")
        or profile.raw_config.get("goal")
        or ""
    ).strip()


def score_repair_guidance(
    scores: Mapping[str, float],
    *,
    min_score: float,
) -> list[str]:
    """Convert low audit dimensions into actionable, profile-agnostic guidance."""
    return [
        _score_guidance(field)
        for field, value in scores.items()
        if value < min_score
    ]


def _append_score_floor_issues(
    scores: Mapping[str, float],
    issues: list[ProfileAuditIssue],
    *,
    min_score: float,
) -> None:
    existing_reasons = {
        (issue.issue_type, issue.reason)
        for issue in issues
        if issue.reason
    }
    for field, value in scores.items():
        if value >= min_score:
            continue
        line = _score_guidance(field)
        issue_type = _issue_type_for_score(field)
        if (issue_type, line) in existing_reasons:
            continue
        # Scores below 7 in fidelity-critical dimensions are severe failures.
        # Low style dimensions should drive repair, not automatic fallback to
        # the unmodernized source text.
        severity = (
            "high"
            if issue_type in CRITICAL_ISSUE_TYPES and value < 7.0
            else "medium"
        )
        issues.append(ProfileAuditIssue(
            issue_type=issue_type,
            severity=severity,
            reason=line,
            suggested_fix=line,
            glossary_action="none",
        ))


def _score_guidance(field: str) -> str:
    guidance = {
        "content_fidelity": "Fidelidad baja: revisa contra la fuente y restaura contenido omitido, agregado o cambiado.",
        "facts_names_order": "Hechos/nombres/orden bajos: conserva nombres, objetos, acciones, números y secuencia narrativa.",
        "orthographic_modernization": "Modernización ortográfica baja: actualiza grafías antiguas superficiales permitidas por el perfil.",
        "syntactic_modernization": "Modernización sintáctica baja: reestructura oraciones con andamiaje antiguo; no basta cambiar palabras.",
        "contemporary_naturalness": "Naturalidad contemporánea baja: mejora ritmo, orden y puntuación para lectura actual sin aplanar la voz.",
        "editorial_consistency": "Consistencia editorial baja: aplica de forma uniforme el perfil activo y el glosario aprobado.",
        "voice_differentiation": "Diferenciación de voces baja: separa narrador, protagonistas y voces secundarias según el perfil activo.",
        "no_censorship_summary_omission": "No omisión/censura bajo: no resumas, no suavices, no moralices y no elimines ambigüedades.",
        "glossary_compliance": "Cumplimiento de glosario bajo: aplica las entradas aprobadas del perfil activo cuando correspondan.",
        "profile_isolation": "Aislamiento de perfil bajo: elimina reglas o tonos que pertenezcan a otros perfiles.",
        "terminology_translation": "Terminología baja: traduce los términos traducibles y conserva de forma consistente solo los que el perfil marque.",
        "name_consistency": "Consistencia de nombres baja: usa las formas canónicas del perfil sin inventar variantes.",
        "target_locale_naturalness": "Naturalidad regional baja: ajusta la prosa al locale objetivo sin introducir regionalismos ajenos al perfil.",
        "register_consistency": "Registro inconsistente: conserva el nivel, tono y relación entre voces a lo largo del texto.",
        "authorial_voice": "Voz autoral baja: recupera el ritmo, la ironía y los contrastes definidos por el perfil activo.",
        "conceptual_precision": "Precisión conceptual baja: restaura relaciones técnicas y matices presentes en la fuente.",
        "citation_and_apparatus_handling": "Aparato crítico bajo: conserva y reconstruye citas, notas y referencias según la política del documento.",
        "table_formula_handling": "Tablas o fórmulas deficientes: conserva datos, relaciones y estructura reconstruible.",
        "listenability": "Escucha poco natural: mejora respiración, puntuación y continuidad sin alterar contenido.",
        "note_intrusion_control": "Notas intrusivas: mueve o reformula llamadas según la política de audiolibro.",
        "caption_integration": "Pies de imagen deficientes: integra solo los informativos de forma clara y fiel.",
        "oral_flow": "Flujo oral bajo: corrige cortes y estructuras difíciles de escuchar sin resumir.",
        "conceptual_clarity": "Claridad conceptual baja: explica relaciones complejas sin perder precisión.",
        "university_level_depth": "Profundidad insuficiente: conserva un nivel universitario y evita simplificación infantil.",
        "plain_language_explanation": "Explicación poco accesible: usa lenguaje común sin borrar conceptos necesarios.",
        "interpretive_fidelity": "Fidelidad interpretativa baja: distingue explicación de información añadida y no cambies la tesis.",
        "period_language_removal": "Lenguaje de época residual: moderniza obstáculos temporales definidos por el perfil.",
        "voice_register_preservation": "Voz o registro debilitados: conserva personalidad y contraste aunque modernices la forma.",
        "rhythm_and_flow": "Ritmo editorial bajo: mejora transiciones, respiración y continuidad sin reescribir hechos.",
    }
    return guidance.get(
        field,
        f"Dimensión {field} baja: corrige únicamente según la política y el glosario del perfil activo.",
    )


def _issue_type_for_score(field: str) -> str:
    if field in {"content_fidelity", "facts_names_order", "interpretive_fidelity"}:
        return "fidelity_error"
    if field in {"orthographic_modernization", "syntactic_modernization", "period_language_removal"}:
        return "syntactic_modernization_error"
    if field in {"contemporary_naturalness", "target_locale_naturalness", "rhythm_and_flow", "oral_flow"}:
        return "contemporary_naturalness_error"
    if field in {"authorial_voice", "voice_register_preservation"}:
        return "authorial_voice_loss"
    if field == "voice_differentiation":
        return "voice_differentiation_error"
    if field in {"glossary_compliance", "terminology_translation", "name_consistency"}:
        return "glossary_error"
    if field == "profile_isolation":
        return "profile_contamination"
    return "editorial_consistency"


def run_profile_precheck(
    source_text: str,
    candidate_text: str,
    *,
    profile_id: str,
) -> ProfileAuditResult:
    profile = load_book_profile(profile_id)
    issues: list[ProfileAuditIssue] = []
    _check_profile_detectors(source_text, candidate_text, profile.detectors, issues)
    _check_glossary_compliance(source_text, candidate_text, profile.approved_entries, issues)
    _check_profile_isolation(candidate_text, profile, issues)
    _check_modernization_residue(source_text, candidate_text, profile, issues)

    scores = _default_scores(score_fields_for_profile(profile))
    for issue in issues:
        _apply_issue_penalty(scores, issue)
    decision = _decision_from_scores(scores, issues, profile.min_dimension_score)
    return ProfileAuditResult(
        profile_id=profile.profile_id,
        scores=scores,
        overall_decision=decision,
        issues=issues,
        summary=f"{len(issues)} profile issue(s) found by local precheck.",
    )


def summarize_profile_audit_records(records: list[Mapping[str, Any]]) -> str:
    if not records:
        return "Sin auditorias de perfil registradas.\n"
    totals: dict[str, list[float]] = {}
    decisions: dict[str, int] = {}
    issues_by_type: dict[str, int] = {}
    for record in records:
        decision = str(record.get("overall_decision") or "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
        for key, value in (record.get("scores") or {}).items():
            try:
                totals.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                continue
        for issue in record.get("issues") or []:
            if isinstance(issue, Mapping):
                issue_type = str(issue.get("type") or "unknown")
                issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1

    lines = ["# Reporte editorial por perfil", ""]
    lines.append("## Decisiones")
    for key, count in sorted(decisions.items()):
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## Promedios"])
    for key, values in sorted(totals.items()):
        if values:
            lines.append(f"- {key}: {sum(values) / len(values):.2f}")
    if issues_by_type:
        lines.extend(["", "## Incidencias"])
        for key, count in sorted(issues_by_type.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"- {key}: {count}")
    return "\n".join(lines).rstrip() + "\n"


def _check_profile_detectors(
    source_text: str,
    candidate_text: str,
    detectors: tuple[ProfileDetector, ...],
    issues: list[ProfileAuditIssue],
) -> None:
    for hit in detector_hits(source_text, candidate_text, detectors):
        issues.append(ProfileAuditIssue(
            issue_type=hit.code,
            severity=hit.severity,
            source_excerpt=hit.excerpt if hit.applies_to == "source" else "",
            candidate_excerpt=hit.excerpt if hit.applies_to != "source" else "",
            reason=hit.message,
            glossary_action="revise_existing",
        ))


def _check_glossary_compliance(
    source_text: str,
    candidate_text: str,
    entries: tuple[ProfileGlossaryEntry, ...],
    issues: list[ProfileAuditIssue],
) -> None:
    profile_like = type("_ProfileLike", (), {"glossary_entries": entries})()
    knowledge = build_editorial_knowledge_base(profile=profile_like)
    validation = knowledge.validate_candidate(
        source_text,
        candidate_text,
        purpose="translation",
    )
    for entry in entries:
        if entry.forbidden_default and contains_profile_term(candidate_text, entry.forbidden_default):
            issues.append(ProfileAuditIssue(
                issue_type="treatment_error",
                severity="high",
                source_excerpt=entry.source,
                candidate_excerpt=entry.forbidden_default,
                reason=f"Candidate used forbidden default for active profile entry: {entry.source}",
                suggested_fix=entry.decision_rule,
                glossary_action="approve_existing",
            ))
    for issue in validation.issues:
        severity = "high" if issue.severity == "reject" else "medium"
        if issue.code == "editorial_preserve_term_missing":
            issue_type = "preserve_term_loss"
        else:
            issue_type = "glossary_error"
        issues.append(ProfileAuditIssue(
            issue_type=issue_type,
            severity=severity,
            source_excerpt=issue.detail.split(" -> ", 1)[0] if issue.detail else "",
            candidate_excerpt="",
            reason=issue.message,
            suggested_fix=issue.detail,
            glossary_action="approve_existing",
        ))


def _check_profile_isolation(
    candidate_text: str,
    profile: BookProfile,
    issues: list[ProfileAuditIssue],
) -> None:
    if profile.allow_cross_profile_glossary:
        return
    loaded = profile.raw_config.get("loaded_glossaries") or []
    foreign = [
        item for item in loaded
        if isinstance(item, str)
        and item not in {"common", profile.profile_id}
    ]
    if foreign:
        issues.append(ProfileAuditIssue(
            issue_type="profile_contamination",
            severity="high",
            candidate_excerpt=_snippet(candidate_text, 0, min(len(candidate_text), 80)),
            reason="Profile configuration declares foreign glossary loading while cross-profile glossary is disabled.",
            suggested_fix="Remove foreign glossary ids from loaded_glossaries or enable explicit cross-profile loading.",
            glossary_action="none",
        ))


def _check_modernization_residue(
    source_text: str,
    candidate_text: str,
    profile: BookProfile,
    issues: list[ProfileAuditIssue],
) -> None:
    if str(profile.modernization_strength or "").strip().lower() not in {"high", "strong", "aggressive"}:
        return
    if not (profile.target_locale or "").lower().startswith("es"):
        return
    source_counts = count_spanish_modernization_residue(source_text)
    if not source_counts:
        return
    candidate_counts = count_spanish_modernization_residue(candidate_text)
    total = sum(candidate_counts.values())
    if total <= 0:
        return
    source_total = max(1, sum(source_counts.values()))
    ratio = total / source_total
    severity = "high" if total >= 6 or ratio >= 0.75 else "medium"
    issues.append(ProfileAuditIssue(
        issue_type="modernization_residue",
        severity=severity,
        source_excerpt=_snippet(source_text, 0, min(len(source_text), 160)),
        candidate_excerpt=_snippet(candidate_text, 0, min(len(candidate_text), 160)),
        reason=(
            "Residual old-Spanish/Peninsular forms remain after high-strength "
            f"modernization: {format_spanish_modernization_residue(candidate_counts)}"
        ),
        suggested_fix=(
            "Rewrite obsolete surface grammar into current editorial Spanish while "
            "preserving content, relationships, order, and voice."
        ),
        glossary_action="none",
    ))


def _snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    value = re.sub(r"\s+", " ", text[left:right]).strip()
    if left > 0:
        value = "..." + value
    if right < len(text):
        value += "..."
    return value


def _default_scores(score_fields: tuple[str, ...] = SCORE_FIELDS) -> dict[str, float]:
    return {field: 10.0 for field in score_fields}


def _apply_issue_penalty(scores: dict[str, float], issue: ProfileAuditIssue) -> None:
    penalty = {"low": 0.5, "medium": 1.5, "high": 3.5}.get(issue.severity, 1.0)
    target_map = {
        "treatment_error": ("glossary_compliance", "voice_differentiation"),
        "glossary_error": ("glossary_compliance", "editorial_consistency"),
        "preserve_term_loss": ("glossary_compliance", "facts_names_order"),
        "profile_contamination": ("profile_isolation",),
        "modernization_residue": (
            "orthographic_modernization",
            "syntactic_modernization",
            "contemporary_naturalness",
        ),
        "syntactic_modernization_error": ("syntactic_modernization", "contemporary_naturalness"),
        "contemporary_naturalness_error": (
            "contemporary_naturalness",
            "target_locale_naturalness",
        ),
        "authorial_voice_loss": ("authorial_voice",),
    }
    targets = target_map.get(issue.issue_type)
    if targets is None and issue.issue_type.endswith("_voice_loss"):
        targets = tuple(
            field
            for field in scores
            if field == "authorial_voice" or field.endswith("_voice")
        )
    if targets is None and issue.issue_type.endswith("_style_error"):
        targets = tuple(
            field
            for field in scores
            if "locale" in field or "naturalness" in field or field.endswith("_style")
        )
    if not targets:
        targets = ("editorial_consistency",)
    for target in targets:
        if target in scores:
            scores[target] = max(0.0, scores[target] - penalty)


def _decision_from_scores(
    scores: Mapping[str, float],
    issues: list[ProfileAuditIssue],
    min_score: float,
) -> str:
    if any(issue.severity == "high" for issue in issues):
        return "fail"
    if any(float(scores.get(field, 10.0)) < 7.0 for field in CRITICAL_SCORE_FIELDS):
        return "fail"
    if any(value < min_score for value in scores.values()):
        return "warn"
    if any(issue.severity == "medium" for issue in issues):
        return "warn"
    return "pass"


def _worse_decision(left: str, right: str) -> str:
    rank = {"pass": 0, "warn": 1, "fail": 2}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


_DEFAULT_AUDIT_PROMPT = """Audit the candidate against the source and the active editorial profile.

Evaluate only the score dimensions requested below. Preserve source content,
facts, names, order, voice, and register unless the active profile explicitly
requires a transformation. Apply only the active profile and its approved
glossary; do not import assumptions from another book, locale, or workflow.

Return JSON with scores, overall_decision, issues, glossary_suggestions, and summary."""
