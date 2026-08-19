"""Token-free quality guard for editorial refinement.

The guard compares a draft translation against its refined version using local
heuristics only. It is deliberately conservative: when the refinement looks
unsafe, the caller should keep the draft and record the reason in the report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional

from src.core.locale_quality import (
    count_mexican_spanish_issues,
    format_mexican_spanish_issues,
    is_mexican_spanish_target,
)
from src.core.text_transform import is_faithful_modernize, modernize_guard_findings
from src.utils.text_encoding import mojibake_score, remove_artifact_glyphs
from src.utils.json_extraction import extract_tagged_payload, loads_first_json_object


_ARTIFACT_RE = re.compile(
    r"[\ufffc\ufffd\u2580-\u259f\u25a0-\u25a1\u25aa-\u25ac\u25ae-\u25b0"
    r"\u25fb-\u25fe\u2b1b-\u2b1c]"
)
_PLACEHOLDER_RE = re.compile(r"\[id\d+\]", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(?:\d{1,4}(?:\s*[,;]\s*\d{1,4})*)\]")
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d+(?:[.,]\d+)?(?:\s*(?:x|X|×|\*)\s*10\^?-?\d+)?|10\^?-?\d+)(?![\w])"
)
_INLINE_CODE_RE = re.compile(r"`[^`\n]{1,160}`")
_LATEX_RE = re.compile(r"\${1,2}[^$\n]{1,240}\${1,2}")
_ASTERISK_OMISSION_RE = re.compile(r"(?:\*\s*){2,}")
_MARKDOWN_EMPHASIS_RE = re.compile(r"(?<!\w)_[^_\n]{1,120}_(?!\w)")
_VARIABLE_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9{}]+|\^-?\d+(?:[.,]\d+)?)\b"
)
_GREEK_TOKEN_RE = re.compile(r"[\u0370-\u03ff][\w{}^_.-]*")
_FORMULA_LINE_RE = re.compile(r"(?:[=<>+\-*/^_{}]|[\u0370-\u03ff]|10\^)")
_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"(?:chapter|cap[ií]tulo|section|secci[oó]n|part|parte)\s+[\wIVXLCDMivxlcdm.-]+.*|"
    r"\d+(?:\.\d+)*\s+\S.*|"
    r"[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÜÑáéíóúüñ'’.,:;() -]{0,90}"
    r")$",
    re.IGNORECASE,
)
EDITORIAL_GUARD_TAG_IN = "<EDITORIAL_GUARD_JSON>"
EDITORIAL_GUARD_TAG_OUT = "</EDITORIAL_GUARD_JSON>"

_FORCE_REJECT_CODES = {
    "empty_refinement",
    "artifact_glyphs_added",
    "mojibake_regression",
    "placeholder_mismatch",
    "mexican_spanish_regression",
}


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message: str
    detail: str = ""


@dataclass
class QualityDecision:
    chunk_index: int
    section: str
    accepted: bool
    issues: list[QualityIssue] = field(default_factory=list)
    local_accepted: bool = True
    draft_chars: int = 0
    refined_chars: int = 0
    source_chars: int = 0
    draft_paragraphs: int = 0
    refined_paragraphs: int = 0
    source_paragraphs: int = 0
    source_available: bool = False
    draft_snippet: str = ""
    refined_snippet: str = ""
    source_snippet: str = ""
    judge_model: str = ""
    judge_decision: str = ""
    judge_confidence: float = 0.0
    judge_reason: str = ""
    judge_issues: list[str] = field(default_factory=list)
    judge_missing_from_source: list[str] = field(default_factory=list)
    judge_added_not_in_source: list[str] = field(default_factory=list)
    judge_weird_symbols: list[str] = field(default_factory=list)
    structure_score: Optional[float] = None

    @property
    def warnings(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def rejections(self) -> list[QualityIssue]:
        return [issue for issue in self.issues if issue.severity == "reject"]


def soften_decision_for_modernize(
    decision: QualityDecision,
    hard_reject_codes: frozenset,
) -> QualityDecision:
    """Downgrade non-corruption rejects to warnings for modernize jobs.

    In same-language modernization, rejecting a candidate often means keeping
    the archaic/source text. That is only safer for actual corruption: empty
    output, mojibake, placeholder/marker damage, weird symbol injection, or a
    source-aware judge rejection. Length/style/paragraph drift should be
    reported and repaired, not silently reverted.
    """
    softened: list[QualityIssue] = []
    changed = False
    for issue in decision.issues:
        if issue.severity == "reject" and issue.code not in hard_reject_codes:
            softened.append(QualityIssue(
                code=issue.code,
                severity="warning",
                message=issue.message,
                detail=(
                    f"{issue.detail} [softened for modernize]"
                    if issue.detail else "[softened for modernize]"
                ),
            ))
            changed = True
        else:
            softened.append(issue)
    if changed:
        decision.issues = softened
        decision.accepted = not any(
            issue.severity == "reject" for issue in decision.issues
        )
    return decision


class EditorialQualityReport:
    """Compact report grouped by detected section/chapter."""

    def __init__(
        self,
        document_name: str = "",
        target_language: str = "",
        profile_summary: Optional[Mapping[str, Any]] = None,
    ):
        self.document_name = document_name
        self.target_language = target_language
        self.records: list[QualityDecision] = []
        self.profile_summary = dict(profile_summary or {})

    def add(self, decision: QualityDecision) -> None:
        self.records.append(decision)

    def has_records(self) -> bool:
        return bool(self.records)

    def summary_counts(self) -> dict[str, int]:
        accepted = sum(1 for r in self.records if r.accepted)
        rejected = len(self.records) - accepted
        warnings = sum(1 for r in self.records if r.warnings)
        return {
            "chunks_reviewed": len(self.records),
            "accepted": accepted,
            "rejected": rejected,
            "warnings": warnings,
        }

    def to_markdown(self, *, max_examples_per_section: int = 20) -> str:
        counts = self.summary_counts()
        lines = [
            "# Reporte de calidad editorial",
            "",
            f"- Documento: {self.document_name or 'Documento'}",
            f"- Idioma: {self.target_language or 'N/D'}",
            f"- Chunks revisados: {counts['chunks_reviewed']}",
            f"- Refinamientos aceptados: {counts['accepted']}",
            f"- Refinamientos rechazados: {counts['rejected']}",
            f"- Chunks con dudas/advertencias: {counts['warnings']}",
            "",
            "Este reporte combina heuristicas locales con juez fuente-aware cuando la fuente esta disponible.",
            "",
        ]

        if self.profile_summary:
            lines.extend([
                "## Perfil editorial activo",
                "",
                f"- Perfil: {self.profile_summary.get('profile_id') or 'N/D'}",
                f"- Nombre: {self.profile_summary.get('profile_name') or 'N/D'}",
                f"- Locale objetivo: {self.profile_summary.get('target_locale') or 'N/D'}",
                f"- Entradas aprobadas del glosario: {self.profile_summary.get('approved_glossary_entries', 0)}",
                f"- Sugerencias pendientes: {self.profile_summary.get('pending_glossary_suggestions', 0)}",
                f"- Glosario comun permitido: {self.profile_summary.get('allow_common_glossary', False)}",
                f"- Glosarios de otros perfiles permitidos: {self.profile_summary.get('allow_cross_profile_glossary', False)}",
                "",
            ])

        judge_records = [r for r in self.records if r.judge_decision]
        if judge_records:
            judge_counts = Counter(r.judge_decision for r in judge_records)
            lines.extend([
                "## Juez fuente-aware",
                "",
                f"- Chunks evaluados con fuente: {len(judge_records)}",
            ])
            for decision_name, count in judge_counts.most_common():
                lines.append(f"- {decision_name}: {count}")
            lines.append("")

        issue_counts = Counter(issue.code for rec in self.records for issue in rec.issues)
        if issue_counts:
            lines.extend(["## Motivos detectados", ""])
            for code, count in issue_counts.most_common():
                lines.append(f"- {code}: {count}")
            lines.append("")

        by_section: dict[str, list[QualityDecision]] = defaultdict(list)
        for record in self.records:
            by_section[record.section or "Documento"].append(record)

        for section, records in by_section.items():
            section_counts = {
                "accepted": sum(1 for r in records if r.accepted),
                "rejected": sum(1 for r in records if not r.accepted),
                "warnings": sum(1 for r in records if r.warnings),
            }
            lines.extend([
                f"## {section}",
                "",
                f"- Chunks: {len(records)}",
                f"- Aceptados: {section_counts['accepted']}",
                f"- Rechazados: {section_counts['rejected']}",
                f"- Con dudas: {section_counts['warnings']}",
                "",
            ])

            interesting = [
                r for r in records
                if (not r.accepted) or r.warnings or _meaningful_change(r)
            ][:max_examples_per_section]

            if not interesting:
                lines.extend(["Sin incidencias destacables.", ""])
                continue

            for record in interesting:
                state = "aceptado" if record.accepted else "rechazado"
                lines.append(f"### Chunk {record.chunk_index} ({state})")
                lines.append(
                    f"- Longitud: {record.draft_chars} -> {record.refined_chars} caracteres; "
                    f"parrafos: {record.draft_paragraphs} -> {record.refined_paragraphs}"
                )
                for issue in record.issues:
                    label = "rechazo" if issue.severity == "reject" else "duda"
                    detail = f" ({issue.detail})" if issue.detail else ""
                    lines.append(f"- {label}: {issue.message}{detail}")
                if record.judge_decision:
                    judge_bits = [
                        f"decision={record.judge_decision}",
                        f"confianza={record.judge_confidence:.2f}",
                    ]
                    if record.judge_model:
                        judge_bits.append(f"modelo={record.judge_model}")
                    lines.append(f"- juez fuente-aware: {', '.join(judge_bits)}")
                    if record.judge_reason:
                        lines.append(f"- razon del juez: {record.judge_reason}")
                    if record.judge_missing_from_source:
                        lines.append(
                            "- posible perdida vs fuente: "
                            + ", ".join(record.judge_missing_from_source[:8])
                        )
                    if record.judge_added_not_in_source:
                        lines.append(
                            "- posible agregado no fuente: "
                            + ", ".join(record.judge_added_not_in_source[:8])
                        )
                    if record.judge_weird_symbols:
                        lines.append(
                            "- simbolos raros: "
                            + ", ".join(record.judge_weird_symbols[:8])
                        )
                if record.source_snippet:
                    lines.append(f"- Fuente: {record.source_snippet}")
                if record.draft_snippet:
                    lines.append(f"- Antes: {record.draft_snippet}")
                if record.refined_snippet:
                    lines.append(f"- Despues: {record.refined_snippet}")
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


def editorial_report_path(output_filepath: str | Path) -> Path:
    path = Path(output_filepath)
    return path.with_name(f"{path.stem} - reporte editorial.md")


def assess_refinement(
    draft_text: str,
    refined_text: str,
    *,
    chunk_index: int,
    section: str = "Documento",
    source_text: str = "",
    glossary_terms: Any = None,
    target_language: str = "",
    source_language: str = "",
    prompt_options: Optional[Mapping] = None,
) -> QualityDecision:
    """Compare draft vs refined text and decide whether refinement is safe."""
    source = source_text or ""
    draft = draft_text or ""
    raw_refined = refined_text or ""
    refined = remove_artifact_glyphs(raw_refined)

    issues: list[QualityIssue] = []
    source_chars = _content_chars(source)
    draft_chars = _content_chars(draft)
    refined_chars = _content_chars(refined)
    source_paragraphs = _paragraph_count(source)
    draft_paragraphs = _paragraph_count(draft)
    refined_paragraphs = _paragraph_count(refined)

    if draft.strip() and not refined.strip():
        issues.append(QualityIssue("empty_refinement", "reject", "El refinamiento quedo vacio"))

    draft_artifacts = _artifact_count(draft)
    refined_artifacts = _artifact_count(raw_refined)
    if refined_artifacts > draft_artifacts:
        issues.append(QualityIssue(
            "artifact_glyphs_added",
            "reject",
            "El refinamiento agrego glifos raros/cuadros",
            f"{draft_artifacts} -> {refined_artifacts}",
        ))
    elif refined_artifacts > 0:
        issues.append(QualityIssue(
            "artifact_glyphs_remaining",
            "warning",
            "Quedan glifos raros/cuadros en el texto refinado",
            str(refined_artifacts),
        ))

    if mojibake_score(raw_refined) > mojibake_score(draft):
        issues.append(QualityIssue("mojibake_regression", "reject", "El refinamiento empeoro la codificacion"))

    _check_placeholders(draft, refined, issues)
    _check_length(draft_chars, refined_chars, issues, prompt_options=prompt_options)
    _check_paragraphs(
        draft_paragraphs,
        refined_paragraphs,
        issues,
        prompt_options=prompt_options,
    )
    _check_asterisk_omissions(draft, refined, issues)
    _check_dialogue_opening_style(draft, refined, issues)
    _check_missing_tokens(
        "numbers_lost",
        "Se perdieron numeros relevantes",
        _extract_numbers(draft),
        _extract_numbers(refined),
        issues,
        reject_threshold=0.30,
    )
    _check_missing_tokens(
        "citations_lost",
        "Se perdieron citas/referencias",
        _extract_citations(draft),
        _extract_citations(refined),
        issues,
        reject_threshold=0.0,
    )
    _check_missing_tokens(
        "formula_fragments_lost",
        "Se perdieron fragmentos de formula o variables",
        _extract_formula_fragments(draft),
        _extract_formula_fragments(refined),
        issues,
        reject_threshold=0.25,
    )
    _check_glossary_terms(draft, refined, glossary_terms, issues)
    _check_mexican_spanish_locale(
        draft,
        refined,
        target_language=target_language,
        prompt_options=prompt_options,
        issues=issues,
    )
    for finding in modernize_guard_findings(
        draft,
        refined,
        prompt_options=prompt_options,
    ):
        issues.append(QualityIssue(
            finding["code"],
            finding["severity"],
            finding["message"],
            finding.get("detail", ""),
        ))

    accepted = not any(issue.severity == "reject" for issue in issues)
    return QualityDecision(
        chunk_index=chunk_index,
        section=section or "Documento",
        accepted=accepted,
        issues=issues,
        local_accepted=accepted,
        source_chars=source_chars,
        draft_chars=draft_chars,
        refined_chars=refined_chars,
        source_paragraphs=source_paragraphs,
        draft_paragraphs=draft_paragraphs,
        refined_paragraphs=refined_paragraphs,
        source_available=bool(source.strip()),
        source_snippet=_snippet(source),
        draft_snippet=_snippet(draft),
        refined_snippet=_snippet(refined),
    )


@dataclass(frozen=True)
class EditorialGuardPrompt:
    system: str
    user: str


def build_source_aware_guard_prompt(
    source_text: str,
    draft_text: str,
    refined_text: str,
    *,
    source_language: str = "",
    target_language: str = "",
    section: str = "",
    local_decision: Optional[QualityDecision] = None,
) -> EditorialGuardPrompt:
    """Build a compact source/draft/refined assessment prompt."""
    local_issues = []
    if local_decision is not None:
        for issue in local_decision.issues:
            local_issues.append({
                "code": issue.code,
                "severity": issue.severity,
                "detail": issue.detail,
            })

    system_prompt = f"""You are a bilingual editorial quality controller.

Your job is to decide whether a refined {target_language or 'target-language'} text should replace an initial translation.
You must compare THREE texts: the original source, the initial translation, and the refined candidate.

Assessment rules:
- Accept a refinement that removes meaningless OCR/scanner garbage, mojibake, duplicated symbols, bad spacing, or collapsed layout while preserving the real source content.
- Reject a refinement that loses real source content, adds unsupported content, changes facts, invents numbers, damages names, drops formulas/citations, collapses meaningful structure, or adds weird symbols.
- For same-language transformations, accept safe wording updates but reject semantic drift: changed speaker/addressee relations, grammatical person, quoted meaning, narrative tone that carries meaning, or intentional ambiguity.
- Do not protect obvious OCR trash just because it existed in the initial translation.
- Do protect names, dates, numbers, formulas, citations, headings, list items, and paragraph-level structure when they are meaningful in the source.
- If the source itself contains unreadable OCR noise, judge whether the refinement handled it conservatively.

Output ONLY valid JSON wrapped in {EDITORIAL_GUARD_TAG_IN} and {EDITORIAL_GUARD_TAG_OUT}.
Do not include markdown or explanation outside the tags.

JSON schema:
{{
  "decision": "accept" | "reject" | "repair" | "flag",
  "confidence": 0.0,
  "reason": "short reason",
  "issues": ["short issue names"],
  "missing_from_source": ["meaningful source items absent from refined text"],
  "added_not_in_source": ["meaningful refined items unsupported by source"],
  "weird_symbols": ["weird symbols present in refined text"],
  "structure_score": 0.0
}}"""

    user_prompt = f"""# SECTION
{section or 'Documento'}

# LANGUAGES
Source: {source_language or 'unknown'}
Target: {target_language or 'unknown'}

# LOCAL HEURISTIC ISSUES
{json.dumps(local_issues, ensure_ascii=False)}

# ORIGINAL SOURCE
<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

# INITIAL TRANSLATION
<DRAFT_TRANSLATION>
{draft_text}
</DRAFT_TRANSLATION>

# REFINED CANDIDATE
<REFINED_CANDIDATE>
{refined_text}
</REFINED_CANDIDATE>

Return the JSON assessment now."""

    return EditorialGuardPrompt(system=system_prompt.strip(), user=user_prompt.strip())


def parse_source_aware_guard_response(response_text: str) -> Optional[dict[str, Any]]:
    """Parse the JSON verdict from the source-aware guard response."""
    if not response_text:
        return None

    body = (
        extract_tagged_payload(
            response_text,
            EDITORIAL_GUARD_TAG_IN,
            EDITORIAL_GUARD_TAG_OUT,
        )
        or response_text
    )
    parsed = loads_first_json_object(body)
    if parsed is None:
        return None

    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject", "repair", "flag"}:
        decision = "flag"
    return {
        "decision": decision,
        "confidence": _coerce_confidence(parsed.get("confidence")),
        "reason": str(parsed.get("reason") or "").strip(),
        "issues": _coerce_string_list(parsed.get("issues")),
        "missing_from_source": _coerce_string_list(parsed.get("missing_from_source")),
        "added_not_in_source": _coerce_string_list(parsed.get("added_not_in_source")),
        "weird_symbols": _coerce_string_list(parsed.get("weird_symbols")),
        "structure_score": _coerce_optional_float(parsed.get("structure_score")),
    }


def apply_source_aware_guard_assessment(
    decision: QualityDecision,
    assessment: Mapping[str, Any],
    *,
    model: str = "",
    min_confidence: float = 0.65,
) -> QualityDecision:
    """Merge a source-aware LLM verdict into the local quality decision."""
    verdict = str(assessment.get("decision") or "flag").lower()
    confidence = _coerce_confidence(assessment.get("confidence"))
    decision.judge_model = model or ""
    decision.judge_decision = verdict
    decision.judge_confidence = confidence
    decision.judge_reason = str(assessment.get("reason") or "").strip()
    decision.judge_issues = _coerce_string_list(assessment.get("issues"))
    decision.judge_missing_from_source = _coerce_string_list(
        assessment.get("missing_from_source")
    )
    decision.judge_added_not_in_source = _coerce_string_list(
        assessment.get("added_not_in_source")
    )
    decision.judge_weird_symbols = _coerce_string_list(assessment.get("weird_symbols"))
    decision.structure_score = _coerce_optional_float(assessment.get("structure_score"))

    local_force_reject = any(
        issue.severity == "reject" and issue.code in _FORCE_REJECT_CODES
        for issue in decision.issues
    )

    if decision.judge_weird_symbols:
        decision.issues.append(QualityIssue(
            "source_aware_weird_symbols",
            "reject",
            "El juez fuente-aware detecto simbolos raros en la revision",
            ", ".join(decision.judge_weird_symbols[:8]),
        ))
        decision.accepted = False
        return decision

    if local_force_reject:
        decision.accepted = False
        return decision

    high_confidence = confidence >= min_confidence
    if verdict == "accept" and high_confidence:
        if not decision.local_accepted:
            decision.issues = [
                QualityIssue(issue.code, "warning", issue.message, issue.detail)
                if issue.severity == "reject" else issue
                for issue in decision.issues
            ]
            decision.issues.append(QualityIssue(
                "source_aware_override_accept",
                "warning",
                "El juez fuente-aware acepto la revision pese a alertas locales",
                decision.judge_reason,
            ))
        decision.accepted = True
        return decision

    if verdict in {"reject", "repair"} and high_confidence:
        code = "source_aware_judge_reject" if verdict == "reject" else "source_aware_judge_repair"
        message = (
            "El juez fuente-aware rechazo la revision frente a la fuente"
            if verdict == "reject"
            else "El juez fuente-aware pide reparacion antes de aceptar la revision"
        )
        decision.issues.append(QualityIssue(code, "reject", message, decision.judge_reason))
        decision.accepted = False
        return decision

    if verdict == "flag" or not high_confidence:
        decision.issues.append(QualityIssue(
            "source_aware_judge_flag",
            "warning",
            "El juez fuente-aware marco dudas o baja confianza",
            decision.judge_reason,
        ))

    decision.accepted = not any(issue.severity == "reject" for issue in decision.issues)
    return decision


def infer_section_title(
    text: str,
    *,
    context_before: str = "",
    fallback: str = "Documento",
) -> str:
    """Infer a compact section/chapter label from nearby text."""
    candidates = []
    for source in (text, context_before):
        candidates.extend([line.strip() for line in (source or "").splitlines() if line.strip()])

    for line in candidates[:16]:
        clean = _clean_heading(line)
        if clean:
            return clean
    return fallback or "Documento"


def _content_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _paragraph_count(text: str) -> int:
    return len([p for p in re.split(r"\n\s*\n+", text or "") if p.strip()])


def _artifact_count(text: str) -> int:
    return len(_ARTIFACT_RE.findall(text or ""))


def _check_placeholders(draft: str, refined: str, issues: list[QualityIssue]) -> None:
    before = Counter(_PLACEHOLDER_RE.findall(draft or ""))
    after = Counter(_PLACEHOLDER_RE.findall(refined or ""))
    if before != after:
        issues.append(QualityIssue(
            "placeholder_mismatch",
            "reject",
            "Cambian placeholders o marcadores estructurales",
            f"{sum(before.values())} -> {sum(after.values())}",
        ))


def _check_length(
    draft_chars: int,
    refined_chars: int,
    issues: list[QualityIssue],
    *,
    prompt_options: Optional[Mapping] = None,
) -> None:
    if draft_chars < 120 or refined_chars == 0:
        return
    ratio = refined_chars / max(1, draft_chars)
    if is_faithful_modernize(prompt_options):
        if draft_chars >= 500 and (ratio < 0.45 or ratio > 2.20):
            issues.append(QualityIssue(
                "length_regression",
                "reject",
                "Cambio de longitud extremo incluso para modernizacion",
                f"ratio={ratio:.2f}",
            ))
        elif ratio < 0.60 or ratio > 1.90:
            issues.append(QualityIssue(
                "length_warning",
                "warning",
                "Cambio de longitud inusual para modernizacion",
                f"ratio={ratio:.2f}",
            ))
        return
    if draft_chars >= 500 and (ratio < 0.70 or ratio > 1.60):
        issues.append(QualityIssue(
            "length_regression",
            "reject",
            "Cambio de longitud demasiado grande para una revision editorial",
            f"ratio={ratio:.2f}",
        ))
    elif ratio < 0.55 or ratio > 2.0:
        issues.append(QualityIssue(
            "length_warning",
            "warning",
            "Cambio de longitud inusual",
            f"ratio={ratio:.2f}",
        ))


def _check_paragraphs(
    draft_count: int,
    refined_count: int,
    issues: list[QualityIssue],
    *,
    prompt_options: Optional[Mapping] = None,
) -> None:
    reject_collapse = not is_faithful_modernize(prompt_options)
    if draft_count >= 3 and refined_count <= 1:
        issues.append(QualityIssue(
            "paragraph_collapse",
            "reject" if reject_collapse else "warning",
            "El refinamiento colapso varios parrafos en un bloque",
            f"{draft_count} -> {refined_count}",
        ))
    elif draft_count >= 5 and refined_count < max(2, int(draft_count * 0.5)):
        issues.append(QualityIssue(
            "paragraph_loss",
            "warning",
            "Se redujeron muchos saltos de parrafo",
            f"{draft_count} -> {refined_count}",
        ))


def _check_asterisk_omissions(draft: str, refined: str, issues: list[QualityIssue]) -> None:
    before = len(_ASTERISK_OMISSION_RE.findall(draft or ""))
    after = len(_ASTERISK_OMISSION_RE.findall(refined or ""))
    if after < before:
        issues.append(QualityIssue(
            "asterisk_omission_lost",
            "reject",
            "Se perdieron marcadores de omision o separadores con asteriscos",
            f"{before} -> {after}",
        ))


def _check_dialogue_opening_style(draft: str, refined: str, issues: list[QualityIssue]) -> None:
    draft_start = (draft or "").lstrip()
    refined_start = (refined or "").lstrip()
    if draft_start.startswith("—") and refined_start.startswith(("“", '"', "«")):
        issues.append(QualityIssue(
            "dialogue_opening_style_regression",
            "reject",
            "La revision cambio una raya de dialogo inicial por comillas",
        ))


def _check_missing_tokens(
    code: str,
    message: str,
    draft_tokens: set[str],
    refined_tokens: set[str],
    issues: list[QualityIssue],
    *,
    reject_threshold: float,
) -> None:
    if not draft_tokens:
        return
    missing = sorted(draft_tokens - refined_tokens)
    if not missing:
        return
    missing_ratio = len(missing) / len(draft_tokens)
    severity = "reject" if missing_ratio > reject_threshold else "warning"
    issues.append(QualityIssue(
        code,
        severity,
        message,
        ", ".join(missing[:8]),
    ))


def _check_glossary_terms(
    draft: str,
    refined: str,
    glossary_terms: Any,
    issues: list[QualityIssue],
) -> None:
    terms = _target_terms_from_glossary(glossary_terms)
    if not terms:
        return
    draft_norm = _casefold(draft)
    refined_norm = _casefold(refined)
    missing = [
        term for term in terms
        if _casefold(term) in draft_norm and _casefold(term) not in refined_norm
    ]
    if missing:
        issues.append(QualityIssue(
            "glossary_terms_lost",
            "reject",
            "Se perdieron terminos del glosario presentes en el borrador",
            ", ".join(missing[:8]),
        ))


def _check_mexican_spanish_locale(
    draft: str,
    refined: str,
    *,
    target_language: str,
    prompt_options: Optional[Mapping],
    issues: list[QualityIssue],
) -> None:
    if not is_mexican_spanish_target(target_language, prompt_options):
        return

    draft_counts = count_mexican_spanish_issues(draft)
    refined_counts = count_mexican_spanish_issues(refined)
    draft_total = sum(draft_counts.values())
    refined_total = sum(refined_counts.values())

    if refined_total > draft_total:
        issues.append(QualityIssue(
            "mexican_spanish_regression",
            "reject",
            "La revision agrego formas no mexicanas/peninsulares",
            f"{format_mexican_spanish_issues(draft_counts)} -> {format_mexican_spanish_issues(refined_counts)}",
        ))
    elif refined_total > 0:
        issues.append(QualityIssue(
            "mexican_spanish_remaining",
            "warning",
            "Quedan formas no mexicanas/peninsulares en el texto refinado",
            format_mexican_spanish_issues(refined_counts),
        ))


def _extract_numbers(text: str) -> set[str]:
    return {_normalize_number(m.group(0)) for m in _NUMBER_RE.finditer(text or "")}


def _extract_citations(text: str) -> set[str]:
    return {_normalize_fragment(m.group(0)) for m in _CITATION_RE.finditer(text or "")}


def _extract_formula_fragments(text: str) -> set[str]:
    fragments: set[str] = set()
    source = _PLACEHOLDER_RE.sub("", text or "")
    source = _MARKDOWN_EMPHASIS_RE.sub("", source)
    for regex in (_INLINE_CODE_RE, _LATEX_RE, _VARIABLE_RE, _GREEK_TOKEN_RE):
        for match in regex.finditer(source):
            normalized = _normalize_fragment(match.group(0))
            if len(normalized) >= 2:
                fragments.add(normalized)
    for line in source.splitlines():
        stripped = line.strip()
        if 6 <= len(stripped) <= 180 and _FORMULA_LINE_RE.search(stripped):
            operators = sum(1 for ch in stripped if ch in "=<>+-*/^_{}")
            if operators >= 2:
                fragments.add(_normalize_fragment(stripped))
    return fragments


def _normalize_number(token: str) -> str:
    token = re.sub(r"\s+", "", token or "")
    token = token.replace("×", "x").replace("*", "x").lower()
    token = re.sub(r"(?<=\d),(?=\d)", ".", token)
    return token


def _normalize_fragment(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    return normalized


def _casefold(text: str) -> str:
    return (text or "").casefold()


def _target_terms_from_glossary(glossary_terms: Any) -> list[str]:
    if not glossary_terms:
        return []
    raw_terms: Iterable[Any]
    if isinstance(glossary_terms, Mapping):
        raw_terms = glossary_terms.values()
    elif isinstance(glossary_terms, list):
        raw_terms = glossary_terms
    else:
        return []

    terms: list[str] = []
    for item in raw_terms:
        if isinstance(item, str):
            candidate = item
        elif isinstance(item, Mapping):
            candidate = (
                item.get("target")
                or item.get("translated_term")
                or item.get("translation")
                or ""
            )
        else:
            candidate = ""
        candidate = str(candidate).strip()
        if len(candidate) >= 3:
            terms.append(candidate)
    return sorted(set(terms), key=str.casefold)


def _clean_heading(line: str) -> Optional[str]:
    line = re.sub(r"\s+", " ", remove_artifact_glyphs(line or "")).strip()
    if not line or len(line) > 110:
        return None
    if line.endswith((".", "?", "!", ";")):
        return None
    if _FORMULA_LINE_RE.search(line) and "=" in line:
        return None
    words = line.split()
    if not 1 <= len(words) <= 12:
        return None
    if not _HEADING_RE.match(line):
        return None
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", line):
        return line
    alpha_words = [w for w in words if re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", w)]
    if not alpha_words:
        return None
    uppercase_ratio = sum(1 for w in alpha_words if w[:1].isupper()) / len(alpha_words)
    if uppercase_ratio >= 0.5 or len(words) <= 4:
        return line
    return None


def _snippet(text: str, limit: int = 260) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _meaningful_change(record: QualityDecision) -> bool:
    if not record.accepted:
        return True
    if record.draft_chars == 0:
        return False
    ratio = abs(record.refined_chars - record.draft_chars) / max(1, record.draft_chars)
    return ratio >= 0.08


def _coerce_confidence(value: Any) -> float:
    parsed = _coerce_optional_float(value)
    if parsed is None:
        return 0.0
    return max(0.0, min(1.0, parsed))


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text[:160])
    return result[:20]
