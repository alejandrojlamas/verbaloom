"""Source-fidelity supervision for translation and editorial refinement.

This module is intentionally independent from the editorial quality guard. The
editorial guard asks "is the text better written?"; this supervisor asks "does
the candidate still say what the source says?"
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from src.config import temperature_for_phase
from src.core.candidate_result import CandidateResult, record_candidate_result
from src.core.document_structure import (
    DocumentBlockClassifier,
    is_comma_delimited_bibliographic_record,
)
from src.core.language_evidence import untranslated_source_pronouns
from src.core.llm.exceptions import ContentRiskError
from src.core.llm.request_deadline import await_llm_call
from src.prompts.security import UNTRUSTED_BOOK_CONTENT_SECTION
from src.utils.json_extraction import extract_tagged_payload, loads_first_json_object
from src.utils.proper_names import (
    extract_symbol_bearing_names,
    missing_symbol_bearing_names,
)
from src.utils.text_encoding import mojibake_score

FIDELITY_AUDIT_TAG_IN = "<FIDELITY_AUDIT_JSON>"
FIDELITY_AUDIT_TAG_OUT = "</FIDELITY_AUDIT_JSON>"

_OFF_VALUES = {"", "none", "off", "false", "disabled"}
_SAME_VALUES = {"same", "primary", "main", "default"}
_LLM_MODES = {"alerted", "sampled", "always", "strict", "strict_full"}
_AUTO_LANGUAGE_KEYS = {
    "auto",
    "autodetect",
    "autodetectar",
    "detect",
    "detectar automaticamente",
    "detectar automáticamente",
    "detectar",
}
_FORCE_REJECT_CODES = {
    "empty_candidate",
    "artifact_glyphs_added",
    "mojibake_regression",
    "placeholder_mismatch",
    "target_language_missing",
    "target_script_mismatch",
    "source_language_residual",
    "symbol_bearing_names_lost",
    "untranslated_source",
}

_ARTIFACT_RE = re.compile(
    r"[\ufffc\ufffd\u2580-\u259f\u25a0-\u25a1\u25aa-\u25ac\u25ae-\u25b0"
    r"\u25fb-\u25fe\u2b1b-\u2b1c]"
)
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d+(?:[.,]\d+)?(?:\s*(?:x|X|×|\*)\s*10\^?-?\d+)?|10\^?-?\d+)(?![\w])"
)
_TIME_NUMBER_RE = re.compile(
    r"(?<![\w])(?P<hour>[01]?\d|2[0-3])[.:](?P<minute>[0-5]\d)"
    r"(?=\s*(?:[ap]\s*\.?\s*m\s*\.?|h(?:oras?)?|hours?|hrs?))",
    re.IGNORECASE,
)
_DECADE_NUMBER_RE = re.compile(r"(?<![\w])(\d{3,4})s\b", re.IGNORECASE)
_ENGLISH_OCR_I_AFTER_CUE_RE = re.compile(
    r"\bthe\s+rest\s+(?P<token>1)(?=\s+suppose\b)",
    re.IGNORECASE,
)
_ENGLISH_OCR_I_TOKEN_RE = re.compile(r"(?<![\w])(?P<token>1)(?![\w])")
_ENGLISH_OCR_I_ADVERBS = frozenset({
    "almost", "already", "also", "always", "even", "ever", "hardly",
    "immediately", "just", "merely", "never", "often", "only", "perhaps",
    "really", "simply", "sometimes", "still", "then", "usually",
})
_ENGLISH_OCR_I_VERBS = frozenset({
    "accept", "accepted", "admire", "admired", "agree", "agreed", "am",
    "ask", "asked", "believe", "believed", "can", "choose", "chose",
    "could", "did", "do", "expect", "expected", "fear", "feared", "feel",
    "felt", "find", "found", "gave", "go", "got", "had", "have", "hear",
    "heard", "hope", "hoped", "know", "knew", "learn", "learned", "left",
    "look", "looked", "made", "may", "mean", "meant", "might", "must",
    "need", "needed", "notice", "noticed", "prefer", "preferred", "promise",
    "promised", "read", "refuse", "refused", "remember", "remembered", "say",
    "said", "saw", "see", "seek", "shall", "shook", "should", "suppose",
    "supposed", "take", "tell", "think", "thought", "told", "understand",
    "understood", "want", "wanted", "warm", "was", "watch", "watched", "went",
    "will", "wish", "wished", "would", "write", "wrote",
})
_ENGLISH_OCR_I_STRONG_VERBS = frozenset({
    "accepted", "admired", "agreed", "asked", "believed", "chose", "expected",
    "feared", "felt", "found", "gave", "got", "heard", "hoped", "knew",
    "learned", "left", "looked", "made", "meant", "needed", "noticed",
    "preferred", "promised", "refused", "remembered", "said", "saw", "shook",
    "supposed", "told", "understood", "wanted", "watched", "went", "wished",
    "wrote",
})
_ENGLISH_OCR_I_CLAUSE_CUES = frozenset({
    "after", "although", "and", "as", "because", "before", "but", "if",
    "or", "since", "so", "than", "then", "though", "till", "unless", "until",
    "when", "whenever", "where", "whereas", "while", "yet",
})
_ENGLISH_OCR_I_QUANTITY_CUES = frozenset({
    "act", "approximately", "book", "chapter", "exactly", "figure", "had",
    "has", "have", "item", "line", "no", "number", "only", "page", "part",
    "route", "scene", "section", "table", "total", "volume", "with",
})
_ENGLISH_OCR_I_NOMINAL_VERBS = frozenset({
    "can", "hope", "need", "promise", "thought", "watch", "will", "wish",
})
_ENGLISH_OCR_I_NOMINAL_FOLLOWERS = frozenset({"about", "for", "of", "to"})
_CITATION_RE = re.compile(r"\[(?:\d{1,4}(?:\s*[,;]\s*\d{1,4})*)\]")
_PLACEHOLDER_PATTERNS = (
    re.compile(r"\[id\d+\]", re.IGNORECASE),
)
_FORMULA_RE = re.compile(
    r"(?:\b[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9{}]+|\^-?\d+(?:[.,]\d+)?)\b|"
    r"[\u0370-\u03ff][\w{}^_.-]*|[=<>+\-*/^{}])"
)
_PROPER_NOUN_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’.-]{2,}(?:\s+"
    r"[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’.-]{2,}){0,3}\b"
)
_WORD_RE = re.compile(r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+", re.UNICODE)
_SPANISH_MARKER_RE = re.compile(
    r"\b(?:el|la|los|las|un|una|unos|unas|de|del|que|en|para|con|por|se|"
    r"no|su|sus|al|es|son|era|fue|fueron|est[aá]|estaba|hab[ií]a|pero|"
    r"pues|cuando|entonces|tambi[eé]n|m[aá]s|menos|como|desde|hasta|sobre)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKER_RE = re.compile(
    r"\b(?:the|and|of|to|in|that|is|was|for|with|as|on|by|from|this|it|"
    r"be|are|were|or|an|at|which|no|not|have|has|had|but|they|their)\b",
    re.IGNORECASE,
)
_FRENCH_MARKER_RE = re.compile(
    r"\b(?:le|la|les|des|du|de|et|que|qui|dans|pour|avec|sur|est|sont|"
    r"une|un|ce|cette|par|pas|plus|mais|comme)\b",
    re.IGNORECASE,
)
_SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "latin": re.compile(r"[A-Za-z\u00c0-\u024f]"),
    "greek": re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]"),
    "cyrillic": re.compile(r"[\u0400-\u052f]"),
    "cjk": re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "hangul": re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
}
_LANGUAGE_SCRIPT_ALIASES: dict[str, str] = {
    "spanish": "latin",
    "espanol": "latin",
    "español": "latin",
    "es": "latin",
    "english": "latin",
    "en": "latin",
    "french": "latin",
    "frances": "latin",
    "francés": "latin",
    "fr": "latin",
    "german": "latin",
    "aleman": "latin",
    "alemán": "latin",
    "de": "latin",
    "italian": "latin",
    "italiano": "latin",
    "it": "latin",
    "portuguese": "latin",
    "portugues": "latin",
    "portugués": "latin",
    "pt": "latin",
    "dutch": "latin",
    "neerlandes": "latin",
    "neerlandés": "latin",
    "nl": "latin",
    "czech": "latin",
    "checo": "latin",
    "cs": "latin",
    "polish": "latin",
    "polaco": "latin",
    "pl": "latin",
    "greek": "greek",
    "griego": "greek",
    "el": "greek",
    "russian": "cyrillic",
    "ruso": "cyrillic",
    "ru": "cyrillic",
    "ukrainian": "cyrillic",
    "ucraniano": "cyrillic",
    "uk": "cyrillic",
    "bulgarian": "cyrillic",
    "bulgaro": "cyrillic",
    "búlgaro": "cyrillic",
    "bg": "cyrillic",
    "chinese": "cjk",
    "chino": "cjk",
    "zh": "cjk",
    "japanese": "cjk",
    "japones": "cjk",
    "japonés": "cjk",
    "ja": "cjk",
    "korean": "hangul",
    "coreano": "hangul",
    "ko": "hangul",
    "arabic": "arabic",
    "arabe": "arabic",
    "árabe": "arabic",
    "ar": "arabic",
    "hebrew": "hebrew",
    "hebreo": "hebrew",
    "he": "hebrew",
    "hindi": "devanagari",
    "hi": "devanagari",
    "sanskrit": "devanagari",
    "sanscrito": "devanagari",
    "sánscrito": "devanagari",
}
_SENSITIVE_RE = re.compile(
    r"\b("
    r"kill|killed|murder|massacre|rape|sex|sexual|slave|slavery|torture|"
    r"prisoner|execution|suicide|blood|war|religion|god|christ|jew|muslim|"
    r"politic|king|queen|empire|colonial|race|racial|nazi|communist|"
    r"matar|muerto|asesin|masacre|violaci[oó]n|sexo|sexual|esclav|tortura|"
    r"prisioner|ejecuci[oó]n|suicidio|sangre|guerra|religi[oó]n|dios|cristo|"
    r"jud[ií]o|musulm[aá]n|pol[ií]tic|rey|reina|imperio|colonial|raza|racial"
    r")\b",
    re.IGNORECASE,
)
_ISOLATED_PAGE_NUMBER_RE = re.compile(r"^\s*(?:\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)
_PDF_TOC_GLUE_RE = re.compile(r"\d{2,4}\d+\.\d+")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_CHUNK_BOUNDARY_END_RE = re.compile(
    r"(?:\b(?:a|al|de|del|el|en|la|las|lo|los|que|un|una|y|o|the|of|to|a|an|"
    r"and|or|that|which|who|so|with|for)\s*|-)$",
    re.IGNORECASE,
)
_OCR_BOUNDARY_FAILURE_RE = re.compile(
    r"\b("
    r"page[_ -]?number\w*|trailing[_ -]?number\w*|page[_ -]?marker\w*|pagination|paginaci[oó]n|"
    r"n[uú]mero de p[aá]gina|numero de pagina|ocr|glued|concatenat|"
    r"proper[_ -]?noun\w*|proper[_ -]?name\w*|transliteration|conjoined[_ -]?phrase\w*|"
    r"table|row|data|tabla|fila|dato|damaged|broken|incomplete|"
    r"split across chunks|chunk boundary|boundary|continuation|"
    r"truncat\w*|incomplete[_ -]?sentence|sentence[_ -]?incomplete|"
    r"rest of sentence after|source ends with"
    r")\b",
    re.IGNORECASE,
)
_COMMON_CAPITALIZED = {
    "The", "This", "That", "These", "Those", "A", "An", "And", "But", "For",
    "Nor", "Or", "So", "Yet", "If", "In", "On", "At", "By", "To", "From",
    "Chapter", "Section", "Part", "Book", "Table", "Figure", "El", "La", "Los",
    "Las", "Un", "Una", "En", "Por", "Para", "Con", "Sin", "Capitulo",
    "Capítulo", "Seccion", "Sección", "Parte", "Libro",
}


@dataclass(frozen=True)
class FidelityIssue:
    code: str
    severity: str
    message: str
    detail: str = ""


@dataclass
class FidelityDecision:
    chunk_index: int
    phase: str
    section: str
    accepted: bool
    issues: list[FidelityIssue] = field(default_factory=list)
    local_accepted: bool = True
    source_chars: int = 0
    candidate_chars: int = 0
    source_paragraphs: int = 0
    candidate_paragraphs: int = 0
    source_hash: str = ""
    candidate_hash: str = ""
    source_snippet: str = ""
    candidate_snippet: str = ""
    judge_model: str = ""
    judge_provider: str = ""
    judge_decision: str = ""
    judge_confidence: float = 0.0
    judge_reason: str = ""
    judge_issues: list[str] = field(default_factory=list)
    judge_missing_from_source: list[str] = field(default_factory=list)
    judge_added_not_in_source: list[str] = field(default_factory=list)
    judge_changed_facts: list[str] = field(default_factory=list)
    judge_censored_or_softened: list[str] = field(default_factory=list)
    judge_evidence_source: list[str] = field(default_factory=list)
    judge_evidence_candidate: list[str] = field(default_factory=list)
    independence: str = "none"

    @property
    def warnings(self) -> list[FidelityIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def rejections(self) -> list[FidelityIssue]:
        return [issue for issue in self.issues if issue.severity == "reject"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "phase": self.phase,
            "section": self.section,
            "accepted": self.accepted,
            "issues": [issue.__dict__ for issue in self.issues],
            "local_accepted": self.local_accepted,
            "source_chars": self.source_chars,
            "candidate_chars": self.candidate_chars,
            "source_paragraphs": self.source_paragraphs,
            "candidate_paragraphs": self.candidate_paragraphs,
            "source_hash": self.source_hash,
            "candidate_hash": self.candidate_hash,
            "source_snippet": self.source_snippet,
            "candidate_snippet": self.candidate_snippet,
            "judge_model": self.judge_model,
            "judge_provider": self.judge_provider,
            "judge_decision": self.judge_decision,
            "judge_confidence": self.judge_confidence,
            "judge_reason": self.judge_reason,
            "judge_issues": self.judge_issues,
            "judge_missing_from_source": self.judge_missing_from_source,
            "judge_added_not_in_source": self.judge_added_not_in_source,
            "judge_changed_facts": self.judge_changed_facts,
            "judge_censored_or_softened": self.judge_censored_or_softened,
            "judge_evidence_source": self.judge_evidence_source,
            "judge_evidence_candidate": self.judge_evidence_candidate,
            "independence": self.independence,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FidelityDecision":
        issues = [
            FidelityIssue(
                code=str(item.get("code", "")),
                severity=str(item.get("severity", "")),
                message=str(item.get("message", "")),
                detail=str(item.get("detail", "")),
            )
            for item in data.get("issues", []) if isinstance(item, Mapping)
        ]
        return cls(
            chunk_index=int(data.get("chunk_index") or 0),
            phase=str(data.get("phase") or ""),
            section=str(data.get("section") or "Documento"),
            accepted=bool(data.get("accepted", True)),
            issues=issues,
            local_accepted=bool(data.get("local_accepted", True)),
            source_chars=int(data.get("source_chars") or 0),
            candidate_chars=int(data.get("candidate_chars") or 0),
            source_paragraphs=int(data.get("source_paragraphs") or 0),
            candidate_paragraphs=int(data.get("candidate_paragraphs") or 0),
            source_hash=str(data.get("source_hash") or ""),
            candidate_hash=str(data.get("candidate_hash") or ""),
            source_snippet=str(data.get("source_snippet") or ""),
            candidate_snippet=str(data.get("candidate_snippet") or ""),
            judge_model=str(data.get("judge_model") or ""),
            judge_provider=str(data.get("judge_provider") or ""),
            judge_decision=str(data.get("judge_decision") or ""),
            judge_confidence=float(data.get("judge_confidence") or 0.0),
            judge_reason=str(data.get("judge_reason") or ""),
            judge_issues=_list_of_str(data.get("judge_issues")),
            judge_missing_from_source=_list_of_str(data.get("judge_missing_from_source")),
            judge_added_not_in_source=_list_of_str(data.get("judge_added_not_in_source")),
            judge_changed_facts=_list_of_str(data.get("judge_changed_facts")),
            judge_censored_or_softened=_list_of_str(data.get("judge_censored_or_softened")),
            judge_evidence_source=_list_of_str(data.get("judge_evidence_source")),
            judge_evidence_candidate=_list_of_str(data.get("judge_evidence_candidate")),
            independence=str(data.get("independence") or "none"),
        )


class FidelityReport:
    """Compact, auditable report grouped by phase and section."""

    def __init__(
        self,
        document_name: str = "",
        source_language: str = "",
        target_language: str = "",
        translator_model: str = "",
        auditor_model: str = "",
        translator_provider: str = "",
        auditor_provider: str = "",
    ):
        self.document_name = document_name
        self.source_language = source_language
        self.target_language = target_language
        self.translator_model = translator_model
        self.auditor_model = auditor_model
        self.translator_provider = translator_provider
        self.auditor_provider = auditor_provider
        self.records: list[FidelityDecision] = []

    def add(self, decision: FidelityDecision) -> None:
        self.records.append(decision)

    def has_records(self) -> bool:
        return bool(self.records)

    def summary_counts(self) -> dict[str, int]:
        rejected = sum(1 for record in self.records if not record.accepted)
        warnings = sum(1 for record in self.records if record.warnings)
        judged = sum(1 for record in self.records if record.judge_decision)
        return {
            "chunks_reviewed": len(self.records),
            "accepted": len(self.records) - rejected,
            "rejected": rejected,
            "warnings": warnings,
            "llm_judged": judged,
        }

    def to_markdown(self, *, max_examples_per_section: int = 20) -> str:
        counts = self.summary_counts()
        independence_counts = Counter(r.independence for r in self.records if r.independence)
        lines = [
            "# Reporte de fidelidad al original",
            "",
            f"- Documento: {self.document_name or 'Documento'}",
            f"- Fuente: {self.source_language or 'N/D'}",
            f"- Destino: {self.target_language or 'N/D'}",
            f"- Traductor: {_format_model(self.translator_provider, self.translator_model)}",
            f"- Auditor: {_format_model(self.auditor_provider, self.auditor_model)}",
            f"- Chunks supervisados: {counts['chunks_reviewed']}",
            f"- Aprobados: {counts['accepted']}",
            f"- Rechazados: {counts['rejected']}",
            f"- Con advertencias: {counts['warnings']}",
            f"- Evaluados por juez LLM: {counts['llm_judged']}",
            "",
            "Este reporte audita si la traduccion o revision conserva el contenido fuente. "
            "No evalua belleza editorial salvo cuando afecta fidelidad.",
            "",
        ]
        if independence_counts:
            lines.extend(["## Independencia del auditor", ""])
            for level, count in independence_counts.most_common():
                lines.append(f"- {level}: {count}")
            lines.append("")

        issue_counts = Counter(issue.code for rec in self.records for issue in rec.issues)
        if issue_counts:
            lines.extend(["## Alertas detectadas", ""])
            for code, count in issue_counts.most_common():
                lines.append(f"- {code}: {count}")
            lines.append("")

        grouped: dict[tuple[str, str], list[FidelityDecision]] = defaultdict(list)
        for record in self.records:
            grouped[(record.phase or "fase", record.section or "Documento")].append(record)

        for (phase, section), records in grouped.items():
            rejected = sum(1 for r in records if not r.accepted)
            warnings = sum(1 for r in records if r.warnings)
            lines.extend([
                f"## {phase}: {section}",
                "",
                f"- Chunks: {len(records)}",
                f"- Rechazados: {rejected}",
                f"- Con advertencias: {warnings}",
                "",
            ])
            interesting = [
                r for r in records
                if (not r.accepted) or r.warnings or r.judge_decision in {"warn", "fail", "repair_needed"}
            ][:max_examples_per_section]
            if not interesting:
                lines.extend(["Sin incidencias destacables.", ""])
                continue
            for record in interesting:
                state = "aprobado" if record.accepted else "rechazado"
                lines.append(f"### Chunk {record.chunk_index} ({state})")
                lines.append(
                    f"- Longitud: fuente {record.source_chars} chars, candidato {record.candidate_chars} chars; "
                    f"parrafos {record.source_paragraphs} -> {record.candidate_paragraphs}"
                )
                for issue in record.issues:
                    label = "rechazo" if issue.severity == "reject" else "advertencia"
                    detail = f" ({issue.detail})" if issue.detail else ""
                    lines.append(f"- {label}: {issue.message}{detail}")
                if record.judge_decision:
                    lines.append(
                        "- juez de fidelidad: "
                        f"decision={record.judge_decision}, "
                        f"confianza={record.judge_confidence:.2f}, "
                        f"modelo={record.judge_model or 'N/D'}, "
                        f"independencia={record.independence}"
                    )
                    if record.judge_reason:
                        lines.append(f"- razon: {record.judge_reason}")
                    if record.judge_missing_from_source:
                        lines.append(
                            "- posible omision: "
                            + ", ".join(record.judge_missing_from_source[:8])
                        )
                    if record.judge_added_not_in_source:
                        lines.append(
                            "- posible agregado: "
                            + ", ".join(record.judge_added_not_in_source[:8])
                        )
                    if record.judge_changed_facts:
                        lines.append(
                            "- datos alterados: "
                            + ", ".join(record.judge_changed_facts[:8])
                        )
                    if record.judge_censored_or_softened:
                        lines.append(
                            "- posible suavizado/censura: "
                            + ", ".join(record.judge_censored_or_softened[:8])
                        )
                if record.source_snippet:
                    lines.append(f"- Fuente: {record.source_snippet}")
                if record.candidate_snippet:
                    lines.append(f"- Candidato: {record.candidate_snippet}")
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def write(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


def fidelity_report_path(output_filepath: str | Path) -> Path:
    path = Path(output_filepath)
    return path.with_name(f"{path.stem} - reporte fidelidad.md")


def ensure_fidelity_report(
    prompt_options: Optional[dict[str, Any]],
    *,
    document_name: str = "",
    source_language: str = "",
    target_language: str = "",
    translator_model: str = "",
    translator_provider: str = "",
    auditor_model: str = "",
    auditor_provider: str = "",
) -> Optional[FidelityReport]:
    if prompt_options is None or not fidelity_supervisor_enabled(prompt_options):
        return None
    existing = prompt_options.get("_fidelity_report")
    if existing is not None and hasattr(existing, "add"):
        return existing
    report = FidelityReport(
        document_name=document_name,
        source_language=source_language,
        target_language=target_language,
        translator_model=translator_model,
        auditor_model=auditor_model or resolve_fidelity_auditor_model(translator_model, prompt_options),
        translator_provider=translator_provider,
        auditor_provider=auditor_provider or translator_provider,
    )
    prompt_options["_fidelity_report"] = report
    return report


def fidelity_supervisor_enabled(prompt_options: Optional[Mapping[str, Any]]) -> bool:
    options = prompt_options or {}
    explicit_flag = options.get("fidelity_supervisor")
    if explicit_flag is False:
        return False
    if explicit_flag is True:
        return fidelity_supervisor_mode(options) != "off"
    if "fidelity_supervisor_mode" not in options:
        return False
    return fidelity_supervisor_mode(options) != "off"


def fidelity_supervisor_mode(prompt_options: Optional[Mapping[str, Any]]) -> str:
    options = prompt_options or {}
    raw = str(options.get("fidelity_supervisor_mode") or "alerted").strip().lower()
    if raw in _OFF_VALUES:
        return "off"
    if raw in {"local", "local_only"}:
        return "local"
    if raw in {"strict_full", "full"}:
        return "strict_full"
    if raw not in {"alerted", "sampled", "always", "strict"}:
        return "alerted"
    return raw


def resolve_fidelity_auditor_model(
    primary_model: str,
    prompt_options: Optional[Mapping[str, Any]],
) -> str:
    options = prompt_options or {}
    configured = (
        options.get("fidelity_supervisor_model")
        or options.get("source_fidelity_model")
        or options.get("fidelity_auditor_model")
    )
    if configured is not None:
        value = str(configured).strip()
        normalized = value.lower()
        if normalized in _OFF_VALUES or normalized in _SAME_VALUES:
            return primary_model
        return value
    if "deepseek" in (primary_model or "").lower():
        return "deepseek-v4-pro"
    return primary_model


def infer_auditor_independence(
    *,
    translator_provider: str = "",
    translator_model: str = "",
    auditor_provider: str = "",
    auditor_model: str = "",
    llm_judge_used: bool = False,
) -> str:
    if not llm_judge_used:
        return "local"
    t_provider = (translator_provider or "").strip().lower()
    a_provider = (auditor_provider or t_provider).strip().lower()
    t_model = (translator_model or "").strip().lower()
    a_model = (auditor_model or "").strip().lower()
    if a_provider and t_provider and a_provider != t_provider:
        return "strong"
    if a_model and t_model and a_model != t_model:
        return "medium"
    return "weak"


def target_language_gate_issues(
    source_text: str,
    candidate_text: str,
    *,
    source_language: str = "",
    target_language: str = "",
    phase: str = "",
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> list[FidelityIssue]:
    """Return hard local issues when a candidate is clearly not in the target language.

    The gate is deliberately conservative. It does not reject borrowed terms,
    names, formulas, citations, or technical vocabulary; it only blocks chunks
    that are dominated by the wrong script/language or are effectively the
    source text returned unchanged when a translation is expected.
    """
    del phase  # Reserved for future per-phase thresholds.
    if not _target_language_gate_enabled(prompt_options):
        return []

    source = source_text or ""
    candidate = candidate_text or ""
    if not source.strip() or not candidate.strip():
        return []

    semantic_source = _clean_structural_language_text(source)
    semantic_candidate = _clean_structural_language_text(candidate)
    source_words = _alpha_word_count(semantic_source)
    candidate_words = _alpha_word_count(semantic_candidate)
    if source_words < 3 or candidate_words < 3:
        return []

    issues: list[FidelityIssue] = []
    target_script = _script_for_language(target_language)
    source_script = _script_for_language(source_language)
    source_profile = _script_profile(semantic_source)
    candidate_profile = _script_profile(semantic_candidate)
    candidate_dominant_script, candidate_dominant_ratio = _dominant_script(candidate_profile)
    source_dominant_script, source_dominant_ratio = _dominant_script(source_profile)
    source_norm = _normalize_text(semantic_source)
    candidate_norm = _normalize_text(semantic_candidate)
    similarity = _text_similarity(source_norm, candidate_norm)
    translation_expected = _translation_expected(
        source_language,
        target_language,
        source_script=source_script,
        target_script=target_script,
        source_dominant_script=source_dominant_script,
        candidate_dominant_script=candidate_dominant_script,
    )
    preservable_index_echo = (
        translation_expected
        and _looks_like_preservable_name_index_echo(
            source,
            candidate,
            prompt_options=prompt_options,
        )
    )
    preservable_bibliographic_echo = (
        translation_expected
        and _looks_like_preservable_bibliographic_echo(
            source,
            candidate,
            prompt_options=prompt_options,
        )
    )
    critical_apparatus_target_evidence = (
        translation_expected
        and _translated_critical_apparatus_has_target_evidence(
            source,
            candidate,
            target_language=target_language,
            prompt_options=prompt_options,
        )
    )

    if target_script and candidate_dominant_script:
        target_ratio = candidate_profile.get(target_script, 0.0)
        if (
            candidate_dominant_script != target_script
            and candidate_dominant_ratio >= 0.35
            and target_ratio <= 0.25
            and candidate_words >= 12
        ):
            issues.append(FidelityIssue(
                "target_script_mismatch",
                "reject",
                "El candidato parece estar escrito en un script distinto al idioma destino",
                f"destino={target_language or 'N/D'} script={target_script}; "
                f"candidato={candidate_dominant_script} {candidate_dominant_ratio:.2f}",
            ))

    if preservable_index_echo or preservable_bibliographic_echo:
        return _unique_issues(issues)

    if (
        translation_expected
        and source_norm == candidate_norm
        and _content_chars(semantic_source) > 20
        and not critical_apparatus_target_evidence
    ):
        issues.append(FidelityIssue(
            "untranslated_source",
            "reject",
            "El candidato parece conservar la fuente sin traducir",
        ))
    elif (
        translation_expected
        and similarity >= 0.93
        and _content_chars(semantic_source) > 80
        and source_dominant_ratio >= 0.55
        and not critical_apparatus_target_evidence
    ):
        issues.append(FidelityIssue(
            "untranslated_source",
            "reject",
            "El candidato es practicamente identico a la fuente",
            f"similitud={similarity:.2f}",
        ))

    exact_echo_issue = None
    if not critical_apparatus_target_evidence:
        exact_echo_issue = _exact_latin_echo_issue(
            source,
            candidate,
            source_language=source_language,
            target_language=target_language,
        )
    if exact_echo_issue:
        issues.append(exact_echo_issue)

    language_issue = _latin_language_gate_issue(
        source,
        candidate,
        source_language=source_language,
        target_language=target_language,
        translation_expected=translation_expected,
        similarity=similarity,
        prompt_options=prompt_options,
    )
    if language_issue:
        issues.append(language_issue)

    detected_language_issue = _same_script_language_mismatch_issue(
        candidate,
        source=source,
        source_language=source_language,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    if detected_language_issue:
        issues.append(detected_language_issue)

    residual_issue = _source_language_residual_issue(
        source,
        candidate,
        source_language=source_language,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    if residual_issue:
        issues.append(residual_issue)

    short_residuals = untranslated_source_pronouns(
        _strip_language_neutral_literals(source),
        _strip_language_neutral_literals(candidate),
        source_language=source_language,
        target_language=target_language,
    )
    if short_residuals:
        issues.append(FidelityIssue(
            "source_language_residual",
            "reject",
            "El candidato conserva pronombres breves del idioma fuente dentro de la traducción",
            f"tipo=short_source_pronoun; fuente={source_language}; "
            f"destino={target_language}; "
            "residuos="
            + "; ".join(f"{token} (1.00)" for token in short_residuals[:3]),
        ))

    return _unique_issues(issues)


def _source_language_residual_issue(
    source: str,
    candidate: str,
    *,
    source_language: str,
    target_language: str,
    prompt_options: Optional[Mapping[str, Any]],
) -> Optional[FidelityIssue]:
    """Detect copied source-language words inside an otherwise translated chunk.

    Whole-chunk language detection misses isolated leftovers in languages that
    share the Latin script. This check only considers words copied verbatim
    from the source, excludes explicit preserve decisions, and requires the
    token itself to be identified as the known source language with strong
    confidence. It does not contain book-specific vocabulary.
    """
    source_key = _language_key(source_language)
    target_key = _language_key(target_language)
    if (
        not source_key
        or source_key in _AUTO_LANGUAGE_KEYS
        or not target_key
        or target_key in _AUTO_LANGUAGE_KEYS
        or source_key == target_key
        or _script_for_language(source_language) != "latin"
        or _script_for_language(target_language) != "latin"
    ):
        return None

    source_code = _language_code(source_key)
    if not source_code:
        return None
    preserved, required_translation, required_phrases = (
        _language_gate_glossary_policy(prompt_options, source_text=source)
    )
    analysis_source = _strip_language_neutral_literals(source)
    analysis_candidate = _strip_language_neutral_literals(candidate)
    copied_phrases = _copied_source_language_phrases(
        analysis_source,
        analysis_candidate,
        source_code=source_code,
        target_key=target_key,
        preserved=preserved,
    )
    stylized_sound_effects: list[tuple[str, float]] = []
    ambiguous_short_phrases: list[tuple[str, float]] = []
    ambiguous_vocatives: list[tuple[str, float]] = []
    bibliographic_metadata: list[tuple[str, float]] = []
    if copied_phrases:
        bibliographic_metadata = [
            item
            for item in copied_phrases
            if _is_bibliographic_registry_overlap(
                item[0],
                source_text=analysis_source,
            )
        ]
        copied_phrases = [
            item
            for item in copied_phrases
            if item not in bibliographic_metadata
        ]
        stylized_sound_effects = [
            item
            for item in copied_phrases
            if _is_stylized_nonlexical_sound_effect(item[0])
        ]
        copied_phrases = [
            item
            for item in copied_phrases
            if not _is_stylized_nonlexical_sound_effect(item[0])
        ]
        ambiguous_vocatives = [
            item
            for item in copied_phrases
            if _is_ambiguous_vocative_with_target_function_tail(
                item[0],
                source_text=analysis_source,
                target_key=target_key,
            )
        ]
        copied_phrases = [
            item
            for item in copied_phrases
            if item not in ambiguous_vocatives
        ]
        ambiguous_short_phrases = [
            item
            for item in copied_phrases
            if _is_short_coordinated_residual(item[0], source_code=source_code)
        ]
        copied_phrases = [
            item
            for item in copied_phrases
            if not _is_short_coordinated_residual(item[0], source_code=source_code)
        ]
    if copied_phrases:
        examples = "; ".join(
            f"{phrase} ({confidence:.2f})"
            for phrase, confidence in copied_phrases[:3]
        )
        explicit_context = str(
            (prompt_options or {}).get("_document_block_context") or ""
        ).strip().casefold()
        inferred_block_type = ""
        if explicit_context not in {"critical_apparatus", "glossary"}:
            inferred_block_type, _policy, _confidence, _strategy, _notes = (
                DocumentBlockClassifier(source_type="text").classify_block(
                    source.splitlines()
                )
            )
        defer_to_auditor = bool(
            (
                explicit_context in {"critical_apparatus", "glossary"}
                or inferred_block_type == "glossary"
            )
            and _translated_critical_apparatus_has_target_evidence(
                source,
                candidate,
                target_language=target_language,
                prompt_options=prompt_options,
            )
        )
        return FidelityIssue(
            "source_language_residual",
            "warning" if defer_to_auditor else "reject",
            (
                "El bloque de referencia conserva identidad o notación que requiere "
                "revisión contextual"
                if defer_to_auditor
                else "El candidato conserva frases del idioma fuente que parecen traducibles"
            ),
            f"fuente={source_language}; destino={target_language}; residuos={examples}",
        )

    required_phrase_hits = [
        phrase
        for phrase in sorted(required_phrases, key=lambda value: (-len(value), value.casefold()))
        if _contains_profile_term(source, phrase)
        and _contains_profile_term(candidate, phrase)
    ]
    if required_phrase_hits:
        examples = "; ".join(required_phrase_hits[:3])
        return FidelityIssue(
            "source_language_residual",
            "reject",
            "El candidato conserva frases que el glosario exige traducir",
            f"fuente={source_language}; destino={target_language}; residuos={examples}",
        )

    source_tokens = _surface_token_map(analysis_source)
    candidate_tokens = _surface_token_map(analysis_candidate)
    overlap = set(source_tokens).intersection(candidate_tokens)
    if not overlap:
        return _ambiguous_source_phrase_warning(
            source_language,
            target_language,
            stylized_sound_effects
            + ambiguous_short_phrases
            + ambiguous_vocatives
            + bibliographic_metadata,
        )

    suspicious: list[tuple[str, float, bool]] = []
    for folded in sorted(overlap, key=lambda item: (-len(item), item))[:40]:
        surface = source_tokens[folded]
        if (
            folded in preserved
            or len(surface) < 7
            or surface.isupper()
            or any(char.isdigit() for char in surface)
            or _looks_like_proper_name_context(surface, analysis_source)
            or _looks_like_proper_name_context(surface, analysis_candidate)
        ):
            continue
        detected_code, confidence = _detect_token_language(surface)
        if detected_code != source_code:
            continue
        foreign_diacritic = _has_foreign_target_diacritic(surface, target_key)
        if (
            (len(surface) >= 10 and confidence >= 0.92)
            or (len(surface) >= 8 and confidence >= 0.97)
            or (foreign_diacritic and confidence >= 0.60)
        ):
            strong_evidence = (
                folded in required_translation
                or (
                    foreign_diacritic
                    and _has_article_like_leader(surface, analysis_candidate)
                )
            )
            suspicious.append((surface, confidence, strong_evidence))

    if not suspicious:
        return _ambiguous_source_phrase_warning(
            source_language,
            target_language,
            stylized_sound_effects
            + ambiguous_short_phrases
            + ambiguous_vocatives
            + bibliographic_metadata,
        )
    # A single same-script token is not enough evidence by itself: loanwords,
    # technical labels and genre terms are common in otherwise valid prose,
    # and ``langdetect`` is not a lexical dictionary. Reject one token only
    # when the glossary requires translation or target-language grammar marks
    # it as a common noun (for example, ``la Wirklichkeit``). Several isolated
    # lowercase guesses are useful audit evidence, but remain too ambiguous for
    # a hard rejection: cognates, place names and loanwords can all be
    # misclassified by a token-level language detector.
    strong_count = sum(1 for _token, _confidence, strong in suspicious if strong)
    lowercase_count = sum(
        1 for token, _confidence, _strong in suspicious if token[:1].islower()
    )
    if strong_count == 0 and lowercase_count < 2:
        return _ambiguous_source_phrase_warning(
            source_language,
            target_language,
            stylized_sound_effects
            + ambiguous_short_phrases
            + ambiguous_vocatives
            + bibliographic_metadata,
        )
    examples = ", ".join(
        f"{token} ({confidence:.2f})" for token, confidence, _strong in suspicious[:5]
    )
    return FidelityIssue(
        "source_language_residual",
        "reject" if strong_count else "warning",
        "El candidato conserva palabras del idioma fuente que parecen traducibles",
        f"fuente={source_language}; destino={target_language}; residuos={examples}",
    )


def _is_stylized_nonlexical_sound_effect(phrase: str) -> bool:
    """Recognize repeated, visibly stylized sound notation rather than prose.

    The exception is intentionally narrow: a repeated token and a hyphenated
    sound cluster must both be present. Ordinary dialogue, lyrics and quoted
    clauses therefore remain actionable source-language residue.
    """
    tokens = _ordered_surface_tokens(phrase)
    if not 2 <= len(tokens) <= 8:
        return False
    folded = [token.casefold() for token in tokens]
    repeated = any(count >= 2 for count in Counter(folded).values())
    hyphenated_cluster = any(
        token.count("-") >= 1
        and all(1 <= len(part) <= 8 for part in token.split("-") if part)
        for token in tokens
    )
    return repeated and hyphenated_cluster


_SOURCE_COORDINATORS = {
    "en": {"and", "or"},
    "de": {"und", "oder"},
    "fr": {"et", "ou"},
    "it": {"e", "o"},
    "pt": {"e", "ou"},
    "nl": {"en", "of"},
}
_BIBLIOGRAPHIC_REGISTRY_TOKENS = {
    "arxiv",
    "biorxiv",
    "crossref",
    "doi",
    "isbn",
    "issn",
    "jstor",
    "medrxiv",
    "pubmed",
    "ssrn",
    "zenodo",
}
_BIBLIOGRAPHIC_DESCRIPTOR_TOKENS = _BIBLIOGRAPHIC_REGISTRY_TOKENS | {
    "archive",
    "blog",
    "eprint",
    "journal",
    "manuscript",
    "online",
    "paper",
    "press",
    "preprint",
    "proceedings",
    "report",
    "repository",
    "review",
    "technical",
    "working",
}
_BIBLIOGRAPHIC_MEDIA_TOKENS = {
    "podcast",
    "soundcloud",
    "spotify",
    "vimeo",
    "youtube",
}
_BIBLIOGRAPHIC_MEASUREMENT_TOKENS = {
    "ed",
    "edition",
    "min",
    "mins",
    "minute",
    "minutes",
    "no",
    "number",
    "page",
    "pages",
    "pp",
    "sec",
    "secs",
    "second",
    "seconds",
    "vol",
    "volume",
}
_BIBLIOGRAPHIC_LOCATOR_RE = re.compile(
    r"(?:"
    r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|"
    r"\b(?:https?://|www\.)|"
    r"\b[\w.-]+\.(?:be|com|edu|gov|io|net|org)/|"
    r"\b(?:arxiv|doi|isbn|issn|jstor|pubmed|ssrn)\b"
    r")",
    re.IGNORECASE,
)


def _is_short_coordinated_residual(phrase: str, *, source_code: str) -> bool:
    """Defer ambiguous three-word binomials to the independent LLM judge.

    Expressions shaped like ``x and y`` are often untranslated prose, but they
    are also productive loan terms, genre labels and technical names in another
    language.  Treating them as a deterministic rejection can create an
    impossible retry loop.  This exception does not accept the phrase silently:
    it emits a warning so the existing alerted fidelity supervisor decides from
    the full source/candidate context.  Explicit glossary translation rules are
    still checked first and remain hard failures.
    """
    tokens = _ordered_surface_tokens(phrase)
    coordinators = _SOURCE_COORDINATORS.get(source_code, set())
    return bool(
        len(tokens) == 3
        and tokens[1].casefold() in coordinators
        and tokens[0][:1].islower()
        and tokens[2][:1].islower()
    )


def _is_bibliographic_registry_overlap(
    phrase: str,
    *,
    source_text: str,
) -> bool:
    """Defer compact citation metadata to the source-aware LLM auditor.

    Registry names and standard publication descriptors often remain unchanged
    in otherwise translated notes (for example, ``preprint, arXiv``). They are
    not enough evidence for a deterministic rejection. The exemption requires
    both a tightly controlled metadata vocabulary and a source block already
    classified as critical apparatus, so ordinary untranslated prose remains a
    hard failure.
    """
    values = _ordered_surface_tokens(phrase)
    tokens = [token.casefold() for token in values]
    if not 1 <= len(tokens) <= 64:
        return False

    if _is_explicit_published_title_overlap(values, source_text=source_text):
        return True

    # ``classify_block`` can label a short note locator as narrative even when
    # ``classify_text`` has already identified its quoted work title from the
    # citation tail. Trust the narrower title-range detector here; it excludes
    # social-status URLs and ordinary dialogue before returning a match.
    if _profile_phrase_matches_published_title(source_text, phrase):
        return True

    block_type, _policy, confidence, _strategy, _notes = (
        DocumentBlockClassifier(source_type="text").classify_block(
            str(source_text or "").splitlines()
        )
    )
    if block_type != "critical_apparatus" or confidence < 0.70:
        return False

    if _profile_term_is_nested_in_published_title(source_text, phrase):
        return True

    if _looks_like_bibliographic_identity_run(values):
        return True

    # Token alignment can attach one unchanged target-language cognate from
    # translated connective prose to the author/title run that follows it.
    # Example: Spanish ``mas popular`` contributes ``popular`` before
    # ``Helen Toner, Leaning into ...``.  Defer that citation-shaped suffix to
    # the source-aware judge instead of forcing an impossible rewrite loop.
    # The enclosing block is already verified as critical apparatus, and the
    # suffix must remain a strict title/identity run.
    if (
        len(values) >= 5
        and not values[0][:1].isupper()
        and _looks_like_bibliographic_identity_run(values[1:])
    ):
        return True

    metadata_tokens = (
        _BIBLIOGRAPHIC_DESCRIPTOR_TOKENS
        | _BIBLIOGRAPHIC_MEDIA_TOKENS
        | _BIBLIOGRAPHIC_MEASUREMENT_TOKENS
    )
    metadata_start: Optional[int] = None
    for index in range(len(tokens)):
        if tokens[index] not in metadata_tokens:
            continue
        suffix_values = values[index:]
        if all(
            token.casefold() in metadata_tokens
            or token[:1].isupper()
            or (len(token) >= 2 and token.isupper())
            for token in suffix_values
        ):
            metadata_start = index
            break
    if metadata_start is None:
        return _is_compact_citation_metadata_overlap(
            phrase,
            source_text=source_text,
        )

    title_prefix = values[:metadata_start]
    if not title_prefix:
        return True
    anchors = 0
    for token in title_prefix:
        folded = token.casefold()
        if folded in _WORK_TITLE_CONNECTORS or folded == "al":
            continue
        if token[:1].isupper() or (len(token) >= 2 and token.isupper()):
            anchors += 1
            continue
        return _is_compact_citation_metadata_overlap(
            phrase,
            source_text=source_text,
        )
    return anchors >= 2 or _is_compact_citation_metadata_overlap(
        phrase,
        source_text=source_text,
    )


def _looks_like_bibliographic_identity_run(tokens: list[str]) -> bool:
    """Recognize authors and published titles merged by inline EPUB markup.

    Semantic XHTML cleanup removes formatting tags before the final audit.
    Adjacent author/title spans can consequently appear as one copied phrase
    even though every lowercase word is only a title connector. This shape is
    safe to defer inside verified critical apparatus; ordinary source prose has
    lowercase lexical words and therefore remains a hard rejection.
    """
    values = [str(token or "").strip(".'’-") for token in tokens]
    values = [value for value in values if value]
    if not 4 <= len(values) <= 64:
        return False

    anchors = 0
    for value in values:
        folded = value.casefold()
        if folded in _WORK_TITLE_CONNECTORS or folded == "al":
            continue
        if (
            value[:1].isupper()
            or (len(value) >= 2 and value.isupper())
            # Brand and technical names such as xAI, iPhone, eBay and
            # arXiv can be identity-bearing title anchors despite beginning
            # with a lowercase letter.
            or any(char.isupper() for char in value[1:])
        ):
            anchors += 1
            continue
        return False
    return anchors >= 3


def _is_explicit_published_title_overlap(
    values: list[str],
    *,
    source_text: str,
) -> bool:
    """Recognize a title citation even when a one-line classifier says prose.

    Short endnotes often begin with a translated locator phrase (for example,
    ``Musk was expanding``) and contain one conventional author/title/outlet
    citation. Classifying that single line as narrative is reasonable, but the
    quoted title remains identity-bearing. Require an exact title-shaped token
    run plus nearby quotation punctuation and a year or immutable locator.
    """
    if not _looks_like_bibliographic_identity_run(values):
        return False

    folded = [value.casefold() for value in values]
    source_entries = _surface_tokens_with_spans(source_text)
    for index in range(0, len(source_entries) - len(folded) + 1):
        source_folded = [
            entry[0].casefold()
            for entry in source_entries[index:index + len(folded)]
        ]
        if source_folded != folded:
            continue
        start = source_entries[index][1]
        end = source_entries[index + len(folded) - 1][2]
        local = str(source_text or "")[max(0, start - 180):end + 260]
        if not _BIBLIOGRAPHIC_LOCATOR_RE.search(local):
            continue
        if not re.search(r"[“”\"«»]", local):
            continue
        if local.count(",") + local.count(";") < 2:
            continue
        return True
    return False


def _is_compact_citation_metadata_overlap(
    phrase: str,
    *,
    source_text: str,
) -> bool:
    """Recognize a compact metadata run inside a verified citation.

    Token alignment deliberately ignores punctuation and numbers, so a citation
    such as ``uploader, YouTube, 24 min.`` can appear to be copied source prose
    as ``uploader YouTube min``. This helper only defers that overlap when the
    same local source window contains a date or locator and citation-shaped
    punctuation. The result remains a warning for the independent LLM auditor;
    it is never accepted silently.
    """
    phrase_tokens = _ordered_surface_tokens(phrase)
    folded = [token.casefold() for token in phrase_tokens]
    if not 1 <= len(folded) <= 10:
        return False

    metadata_tokens = (
        _BIBLIOGRAPHIC_DESCRIPTOR_TOKENS
        | _BIBLIOGRAPHIC_MEDIA_TOKENS
        | _BIBLIOGRAPHIC_MEASUREMENT_TOKENS
    )
    if not set(folded).intersection(metadata_tokens):
        return False

    source_entries = _surface_tokens_with_spans(source_text)
    for index in range(0, len(source_entries) - len(folded) + 1):
        source_folded = [
            entry[0].casefold()
            for entry in source_entries[index:index + len(folded)]
        ]
        if source_folded != folded:
            continue
        start = source_entries[index][1]
        end = source_entries[index + len(folded) - 1][2]
        local = str(source_text or "")[max(0, start - 220):end + 220]
        if not _BIBLIOGRAPHIC_LOCATOR_RE.search(local):
            continue
        if sum(local.count(mark) for mark in (",", ";", ":", "“", "”", '"')) < 3:
            continue
        return True
    return False


def _ambiguous_source_phrase_warning(
    source_language: str,
    target_language: str,
    phrases: list[tuple[str, float]],
) -> Optional[FidelityIssue]:
    if not phrases:
        return None
    examples = "; ".join(
        f"{phrase} ({confidence:.2f})" for phrase, confidence in phrases[:3]
    )
    return FidelityIssue(
        "source_language_residual",
        "warning",
        "El candidato conserva una frase fuente breve que requiere juicio contextual",
        f"fuente={source_language}; destino={target_language}; residuos={examples}",
    )


def _ordered_surface_tokens(text: str) -> list[str]:
    return [
        match.group(0).strip(".'’-")
        for match in re.finditer(r"[^\W\d_][\w'’.-]*", text or "", flags=re.UNICODE)
        if match.group(0).strip(".'’-")
    ]


def _surface_tokens_with_spans(text: str) -> list[tuple[str, int, int]]:
    entries: list[tuple[str, int, int]] = []
    for match in re.finditer(
        r"[^\W\d_][\w'’.-]*",
        text or "",
        flags=re.UNICODE,
    ):
        raw = match.group(0)
        token = raw.strip(".'’-")
        if not token:
            continue
        offset = raw.find(token)
        start = match.start() + max(offset, 0)
        entries.append((token, start, start + len(token)))
    return entries


def _copied_source_language_phrases(
    source: str,
    candidate: str,
    *,
    source_code: str,
    target_key: str,
    preserved: set[str],
) -> list[tuple[str, float]]:
    """Find contiguous source prose copied into a same-script translation.

    Single-token language identification is deliberately conservative because
    names and German capitalized nouns are noisy. A copied sequence supplies
    much stronger evidence: it must contain at least three words, be detected
    as the known source language, and contain lowercase lexical material. This
    catches complete residual clauses while leaving publisher names, place
    lists, work titles and other proper-name sequences alone.
    """
    findings: list[tuple[str, float]] = []
    seen: set[str] = set()
    source_segments = _placeholder_bounded_segments(source)
    candidate_segments = _placeholder_bounded_segments(candidate)
    # Placeholder integrity has its own hard gate. Do not manufacture a
    # language-residue phrase by aligning text across missing HTML boundaries.
    if len(source_segments) != len(candidate_segments):
        return []

    for source_segment, candidate_segment in zip(source_segments, candidate_segments):
        source_tokens = _ordered_surface_tokens(source_segment)
        candidate_tokens = _ordered_surface_tokens(candidate_segment)
        if len(source_tokens) < 3 or len(candidate_tokens) < 3:
            continue
        source_folded = [token.casefold() for token in source_tokens]
        candidate_folded = [token.casefold() for token in candidate_tokens]
        matcher = difflib.SequenceMatcher(
            None,
            source_folded,
            candidate_folded,
            autojunk=False,
        )
        for block in matcher.get_matching_blocks():
            if block.size < 2:
                continue
            surfaces = candidate_tokens[block.b:block.b + block.size]
            folded = candidate_folded[block.b:block.b + block.size]
            phrase = " ".join(surfaces)
            dialogue_overlap = _looks_like_dialogue_source_overlap(source_segment, phrase)
            if block.size < 3 and not dialogue_overlap:
                continue
            if len(phrase) < (6 if dialogue_overlap else 12):
                continue
            if all(token in preserved for token in folded):
                continue
            if _is_target_language_function_word_sequence(folded, target_key):
                continue
            if _is_preserved_term_with_target_function_words(
                folded,
                target_key=target_key,
                preserved=preserved,
            ):
                continue
            if _is_intentional_source_language_literal(source_segment, phrase):
                continue
            if _is_honorific_name_sequence(surfaces):
                continue
            if _is_multiword_proper_name(
                surfaces,
                source_text=source_segment,
            ):
                continue
            if _looks_like_work_title_sequence(surfaces):
                continue
            if (
                len(surfaces) >= 2
                and all(token[:1].isupper() for token in surfaces)
            ):
                # A fully title-cased overlap such as ``John Cameron Swayze``
                # is a person, institution or work title, not untranslated
                # source prose. Actual copied dialogue still contains a
                # lowercase grammar word (for example ``Oh my God``).
                continue
            if (
                dialogue_overlap
                and len(surfaces) == 2
                and all(token[:1].isupper() for token in surfaces)
            ):
                # A two-token vocative such as a person's full name is not
                # enough evidence of untranslated prose by itself.
                continue

            # Proper names and title strings are usually title-cased. Require
            # substantive lowercase prose, not just an article or connector.
            lowercase_lexemes = [
                token
                for token in surfaces
                if token[:1].islower()
                and len(token) >= 4
                and token.casefold() not in _CAPITALIZED_CONTEXT_STOPWORDS
            ]
            short_lowercase_lexemes = [
                token
                for token in surfaces
                if token[:1].islower()
                and len(token) >= 3
                and token.casefold() not in _CAPITALIZED_CONTEXT_STOPWORDS
            ]
            # One shared cognate followed by a split proper name is not copied
            # source prose (for example, ``inexorable Ch*Tril``). Requiring two
            # substantive lowercase lexemes preserves detection of actual
            # residual phrases while avoiding expensive false repair loops.
            if (
                len(lowercase_lexemes) < 2
                and len(short_lowercase_lexemes) < 2
                and not dialogue_overlap
            ):
                continue
            detected_code, confidence = _detect_token_language(phrase)
            lowercase_tokens = [token for token in surfaces if token[:1].islower()]
            dialogue_grammar_evidence = dialogue_overlap and bool(lowercase_tokens)
            short_prose_evidence = len(short_lowercase_lexemes) >= 2
            target_code = _language_code(_language_key(target_key))
            if (
                len(surfaces) <= 4
                and (
                    detected_code == target_code
                    or _has_repeated_target_language_marker_evidence(
                        folded,
                        target_key,
                    )
                    or (
                        detected_code != source_code
                        and _has_distinct_target_language_marker(
                            folded,
                            target_key,
                        )
                    )
                )
            ):
                # A foreign-language source can deliberately contain a short
                # expression that is already valid in the target language.
                # Statistical detection is especially noisy for these tiny
                # spans, so explicit target grammar outweighs dialogue shape.
                continue
            if not dialogue_grammar_evidence:
                min_confidence = 0.80 if short_prose_evidence else 0.90
                if detected_code != source_code or confidence < min_confidence:
                    continue
            elif detected_code != source_code:
                # Very short interjections are routinely misclassified by
                # statistical language detectors. Exact source overlap plus a
                # lowercase grammar word inside dialogue is stronger evidence.
                confidence = max(confidence, 0.90)
            if detected_code == source_code and confidence < 0.80:
                continue
            normalized = phrase.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            findings.append((phrase, confidence))
    return findings


def _is_target_language_function_word_sequence(
    folded_tokens: list[str],
    target_key: str,
) -> bool:
    """Allow short expressions whose spelling is already valid in the target.

    Statistical language identification cannot distinguish shared function
    words such as a repeated ``no``. Requiring every token to be a known target
    marker avoids suppressing mixed phrases that still contain source prose.
    """
    if not folded_tokens:
        return False
    normalized = _language_key(target_key)
    if normalized in {"spanish", "espanol", "español", "es"}:
        marker = _SPANISH_MARKER_RE
    elif normalized in {"english", "en"}:
        marker = _ENGLISH_MARKER_RE
    elif normalized in {"french", "francais", "français", "fr"}:
        marker = _FRENCH_MARKER_RE
    else:
        return False
    return all(marker.fullmatch(token) is not None for token in folded_tokens)


def _is_preserved_term_with_target_function_words(
    folded_tokens: list[str],
    *,
    target_key: str,
    preserved: set[str],
) -> bool:
    """Honor canonical glossary terms inside target-language expressions.

    A phrase such as ``<canonical name>, no`` is copied from the source only
    because the name must remain stable and the particle is valid in both
    languages. Requiring every token to be either explicitly preserved or a
    target-language function word keeps this exception narrower than a generic
    proper-name heuristic.
    """
    if not folded_tokens or not preserved:
        return False
    has_preserved_term = False
    for token in folded_tokens:
        if token in preserved:
            has_preserved_term = True
            continue
        if _is_target_language_function_word_sequence([token], target_key):
            continue
        return False
    return has_preserved_term


def _is_ambiguous_vocative_with_target_function_tail(
    phrase: str,
    *,
    source_text: str,
    target_key: str,
) -> bool:
    """Recognize a short vocative whose remaining words fit the target.

    Without a glossary, a capitalized vocative is not enough evidence to call
    the phrase a proper name. It is also not enough evidence to stop a whole
    book. These cases go to the contextual auditor as warnings. The comma
    boundary and target-language tail deliberately exclude ordinary residuals
    such as ``No problem`` and longer copied clauses.
    """
    tokens = _ordered_surface_tokens(phrase)
    if not 2 <= len(tokens) <= 4:
        return False
    for split_at in range(1, len(tokens)):
        vocative = tokens[:split_at]
        tail = [token.casefold() for token in tokens[split_at:]]
        if not all(token[:1].isupper() for token in vocative):
            continue
        if _is_target_language_function_word_sequence(
            [token.casefold() for token in vocative],
            target_key,
        ):
            continue
        if not _is_target_language_function_word_sequence(tail, target_key):
            continue
        head_pattern = r"[^\w]+".join(re.escape(token) for token in vocative)
        tail_pattern = r"[^\w]+".join(
            re.escape(token) for token in tokens[split_at:]
        )
        if re.search(
            rf"(?<!\w){head_pattern}\s*,\s*{tail_pattern}(?!\w)",
            source_text or "",
            flags=re.IGNORECASE,
        ):
            return True
    return False


_DISTINCT_TARGET_LANGUAGE_MARKERS = {
    "spanish": {
        "al", "con", "cuando", "del", "desde", "el", "ella", "ellas",
        "ellos", "era", "es", "estaba", "está", "fue", "fueron", "había",
        "la", "las", "lo", "los", "más", "mucha", "muchas", "mucho",
        "muchos", "muy", "para", "pero", "por", "pues", "que", "se",
        "son", "sobre", "su", "sus", "también", "una", "unas", "uno",
        "unos",
    },
    "english": {
        "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "had", "has", "have", "is", "it", "of", "on", "that", "the",
        "their", "they", "this", "to", "was", "were", "which", "with",
    },
    "french": {
        "avec", "ce", "cette", "dans", "des", "du", "est", "et", "la",
        "le", "les", "mais", "par", "pas", "plus", "pour", "qui", "sont",
        "sur", "une",
    },
}


def _has_distinct_target_language_marker(
    folded_tokens: list[str],
    target_key: str,
) -> bool:
    """Return target-language grammar evidence inside a short copied span.

    Ambiguous particles such as ``no`` and bare prepositions shared broadly
    across languages are intentionally absent.  This is evidence for preserving
    a target-language insertion, not a general lexical allowlist.
    """
    normalized = _language_key(target_key)
    aliases = {
        "es": "spanish", "espanol": "spanish", "español": "spanish",
        "en": "english",
        "fr": "french", "francais": "french", "français": "french",
    }
    marker_key = aliases.get(normalized, normalized)
    markers = _DISTINCT_TARGET_LANGUAGE_MARKERS.get(marker_key, set())
    return any(token in markers for token in folded_tokens)


def _has_repeated_target_language_marker_evidence(
    folded_tokens: list[str],
    target_key: str,
) -> bool:
    """Recognize a short phrase that already carries target-language grammar.

    Statistical detectors are unreliable on telegraphic dialogue. Requiring at
    least two marker occurrences keeps one ambiguous loanword insufficient while
    accepting phrases such as repeated Spanish quantifiers.
    """
    normalized = _language_key(target_key)
    aliases = {
        "es": "spanish", "espanol": "spanish", "español": "spanish",
        "en": "english",
        "fr": "french", "francais": "french", "français": "french",
    }
    marker_key = aliases.get(normalized, normalized)
    markers = _DISTINCT_TARGET_LANGUAGE_MARKERS.get(marker_key, set())
    return sum(token in markers for token in folded_tokens) >= 2


def _looks_like_dialogue_source_overlap(source_segment: str, phrase: str) -> bool:
    """Return True for a short copied phrase used as dialogue or exclamation.

    Two-word interjections are too short for ordinary language detection and
    often look title-cased, so the proper-name guard used to let them through.
    Requiring source dialogue punctuation keeps this narrow: narrative mentions
    of people, publishers, works, and locations are unaffected.
    """
    if not source_segment or not phrase:
        return False
    words = _ordered_surface_tokens(phrase)
    if not 2 <= len(words) <= 7:
        return False
    tokens = _ordered_surface_tokens(phrase)
    escaped = r"[^\w]+".join(re.escape(token) for token in tokens)
    quoted_dialogue = re.search(
        rf"[\"'“”‘’—]\s*{escaped}\s*[,;:!?”\"'’—]",
        source_segment,
        flags=re.IGNORECASE,
    )
    if quoted_dialogue:
        return True

    # Unquoted exclamations can still be dialogue, but a comma is not enough
    # evidence at a segment boundary.  Comma-delimited scene headings can
    # otherwise look like two-word utterances after placeholder segmentation
    # and trigger an expensive false repair loop.
    unquoted_exclamation = re.search(
        rf"(?:^|[.!?]\s+){escaped}\s*[!?](?:\s|$)",
        source_segment,
        flags=re.IGNORECASE,
    )
    return bool(unquoted_exclamation)


_QUERY_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*(?:search|query|find|define|limits?|select|where)\s*:",
)
_QUERY_BOOLEAN_RE = re.compile(r"\b(?:AND|OR|NOT|NEAR)\b")
_METALANGUAGE_CUE_RE = re.compile(
    r"\b(?:letters?|strings?|words?|terms?|spelling|spellings|search(?:es|ing)?|"
    r"look\s+for|named|called)\b",
    re.IGNORECASE,
)


def _is_intentional_source_language_literal(source_segment: str, phrase: str) -> bool:
    """Recognize source-language material whose exact spelling is the content.

    Database queries, code-like Boolean expressions, and word-form examples
    are not untranslated prose. Translating their operands can falsify a scene
    that explains how a source-language search works. This exception is kept
    narrow: ordinary quoted prose and technical vocabulary still reach the
    residual-language gate.
    """
    source_value = source_segment or ""
    phrase_value = phrase or ""
    if (
        _QUERY_DIRECTIVE_RE.search(source_value)
        and len(_QUERY_BOOLEAN_RE.findall(source_value)) >= 2
    ):
        return True
    if not _METALANGUAGE_CUE_RE.search(source_value):
        return False

    words = [
        token.casefold()
        for token in re.findall(r"[^\W\d_]+", phrase_value, flags=re.UNICODE)
        if len(token) >= 4
    ]
    if len(words) < 4:
        return False
    prefix_counts = Counter(word[:4] for word in words)
    related_words = sum(count for count in prefix_counts.values() if count >= 2)
    return related_words >= 4


def _placeholder_bounded_segments(text: str) -> list[str]:
    """Return prose runs that cannot cross an EPUB structural placeholder."""
    return _PLACEHOLDER_PATTERNS[0].split(text or "")


def _surface_token_map(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in re.findall(r"[^\W\d_][\w'’.-]*", text or "", flags=re.UNICODE):
        folded = token.strip(".'’-").casefold()
        if folded:
            result.setdefault(folded, token.strip(".'’-"))
    return result


_ENTITY_CONTEXT_MARKERS = {
    "ag", "company", "corporation", "editorial", "foundation", "fundación",
    "gmbh", "inc", "institut", "institute", "ltd", "museum", "press",
    "publishing", "stiftung", "universidad", "universität", "university",
    "verlag",
}
_CAPITALIZED_CONTEXT_STOPWORDS = {
    "a", "an", "das", "de", "dem", "den", "der", "des", "die", "ein",
    "eine", "einer", "el", "la", "las", "le", "les", "los", "the", "un",
    "una", "une",
}
_PROPER_NAME_CONNECTORS = _CAPITALIZED_CONTEXT_STOPWORDS | {
    # Cross-language surname and entity particles. These are structural name
    # mechanics, not book-specific vocabulary. Keeping them separate from the
    # broader work-title connectors avoids exempting ordinary copied prose.
    "al", "ap", "ben", "bin", "da", "das", "de", "del", "della", "den",
    "der", "di", "do", "dos", "du", "ibn", "la", "le", "st", "van",
    "von", "y",
}
_WORK_TITLE_CONNECTORS = {
    "a", "an", "and", "as", "at", "but", "by", "da", "das", "de", "del",
    "der", "des", "di", "do", "dos", "du", "e", "el", "en", "et",
    "for", "from", "in", "into", "la", "las", "le", "les", "los", "of", "on",
    "or", "over", "para", "por", "the", "through", "to", "un", "una", "under",
    "une", "und", "van", "versus", "via", "von", "with", "without", "y",
}
_INDEX_TRANSLATABLE_QUALIFIERS = {
    "abbot", "admiral", "archbishop", "archduke", "baron", "baroness", "bishop",
    "captain", "colonel", "commander", "corporal", "count", "countess",
    "doctor", "duchess", "duke", "earl", "emperor", "empress", "father", "general",
    "king", "lady", "lieutenant", "lord", "major", "marshal", "midshipman",
    "mother", "prince", "princess", "professor", "queen", "reverend",
    "saint", "sergeant", "sir", "venture", "vice",
}
_COMMON_NOUN_DETERMINERS = {
    "a", "an", "das", "dem", "den", "der", "des", "die", "ein",
    "eine", "einem", "einen", "einer", "eines", "el", "la", "las",
    "le", "les", "los", "the", "un", "una", "une", "unos", "unas",
}
_PROPER_NAME_LEADERS = {
    # Generic honorifics, ranks and epithets.  These are language mechanics,
    # not book-specific translation rules.  They let the residual-language
    # gate distinguish "el pobre Algernon[id19]" from a copied German noun
    # even when an EPUB placeholder hides the following name context.
    "alte", "ancien", "ancienne", "captain", "capitán", "capitana",
    "doctor", "doctora", "don", "doña", "dr", "dra", "emperor",
    "emperador", "emperatriz", "father", "frau", "general", "herr",
    "joven", "junior", "kaiser", "king", "könig", "lady", "lord",
    "madame", "maestro", "maestra", "monsieur", "mother", "padre",
    "poor", "pobre", "prince", "princess", "princesa", "príncipe",
    "prof", "profesor", "profesora", "professor", "queen", "reina",
    "rey", "saint", "san", "santa", "señor", "señora", "sir", "sr",
    "sra", "st", "viejo", "vieja", "young",
}


def _is_honorific_name_sequence(tokens: list[str]) -> bool:
    """Recognize one or more honorific-led names, including initials.

    Historical prose often lists people as ``sir T. A. Bridges, sir John
    Browne``. A repeated lower-case honorific made the full overlap look like
    untranslated source prose even though every lexical item was a proper name.
    This parser accepts only complete name-shaped runs; any ordinary lower-case
    verb or noun still makes the sequence fail closed.
    """
    values = [str(token or "").strip(".'’-") for token in tokens]
    values = [value for value in values if value]
    if not 2 <= len(values) <= 24:
        return False

    index = 0
    groups = 0
    anchors = 0
    saw_leader = False
    while index < len(values):
        group_has_leader = values[index].casefold() in _PROPER_NAME_LEADERS
        if group_has_leader:
            saw_leader = True
            index += 1
        group_anchors = 0
        while index < len(values):
            value = values[index]
            folded = value.casefold()
            if folded in _PROPER_NAME_LEADERS and group_anchors > 0:
                break
            if folded in _PROPER_NAME_CONNECTORS and group_anchors > 0:
                next_anchor = index
                while (
                    next_anchor < len(values)
                    and values[next_anchor].casefold() in _PROPER_NAME_CONNECTORS
                ):
                    next_anchor += 1
                if next_anchor >= len(values):
                    return False
                following = values[next_anchor]
                if (
                    following.casefold() in _PROPER_NAME_LEADERS
                    or not following[:1].isupper()
                ):
                    return False
                index = next_anchor
                continue
            if value[:1].isupper() or (len(value) >= 2 and value.isupper()):
                group_anchors += 1
                anchors += 1
                index += 1
                continue
            return False
        if group_anchors == 0:
            return False
        if not group_has_leader and group_anchors < 2:
            return False
        groups += 1

    return saw_leader and anchors >= 1 and groups >= 1


def _is_multiword_proper_name(
    tokens: list[str],
    *,
    source_text: str = "",
) -> bool:
    """Recognize names whose internal connector is lower-case.

    Places, institutions, and titles routinely preserve connectors from a
    third language, for example ``Pongo das Mortes``. Statistical language
    detection overweights the connector and can misclassify the whole name as
    source-language prose. Require at least two title-cased lexical anchors
    and allow only known article/preposition connectors between them; ordinary
    copied clauses still contain an unapproved lower-case lexical word.
    """
    values = [str(token or "").strip(".'’-") for token in tokens]
    values = [value for value in values if value]
    if not 2 <= len(values) <= 8:
        return False

    anchors = [
        value
        for value in values
        if value[:1].isupper()
        and value.casefold() not in _CAPITALIZED_CONTEXT_STOPWORDS
        and len(value) >= 2
    ]
    if len(anchors) < 2:
        return False
    if not values[0][:1].isupper() or not values[-1][:1].isupper():
        return False
    if all(
        value[:1].isupper()
        or value.casefold() in _PROPER_NAME_CONNECTORS
        for value in values
    ):
        return True

    # Analytical indexes also contain organization names shaped like
    # ``Black in AI``: a capitalized anchor, a lower-case connector, and a
    # terminal acronym. Limit this extension to a comma-delimited entry whose
    # entire readable segment is the name, so a copied sentence that happens
    # to end in an acronym remains actionable source-language residue.
    source_value = str(source_text or "")
    source_tokens = _ordered_surface_tokens(source_value)
    phrase_pattern = re.compile(
        r"(?<!\w)"
        + r"\s+".join(re.escape(value) for value in values)
        + r"\s*,",
    )
    isolated_entry = (
        [value.casefold() for value in source_tokens]
        == [value.casefold() for value in values]
        and bool(re.search(r",\s*$", source_value))
    )
    dense_index_entry = bool(
        source_value.count(",") >= 4
        and not re.search(r"[.!?]", source_value)
        and phrase_pattern.search(source_value)
    )
    return bool(
        3 <= len(values) <= 6
        and 2 <= len(values[-1]) <= 8
        and values[-1].isupper()
        and values[0][:1].isupper()
        and all(
            value[:1].isupper()
            or value.casefold() in _WORK_TITLE_CONNECTORS
            for value in values
        )
        and (isolated_entry or dense_index_entry)
    )


def _looks_like_work_title_sequence(tokens: list[str]) -> bool:
    """Recognize a title-cased work name with lower-case connectors.

    EPUBs commonly isolate bibliography titles between placeholders. Long
    titles contain source-language articles and prepositions, so statistical
    language detection otherwise mistakes them for untranslated prose. This
    accepts only title-shaped sequences: every lexical word must be capitalized
    (or an all-caps acronym), with a small generic connector vocabulary.
    Sentence-case quotations and ordinary residual clauses remain actionable.
    """
    values = [str(token or "").strip(".'’-") for token in tokens]
    values = [value for value in values if value]
    if not 3 <= len(values) <= 28:
        return False

    anchors = 0
    for value in values:
        folded = value.casefold()
        if folded in _WORK_TITLE_CONNECTORS:
            continue
        if value[:1].isupper() or (len(value) >= 2 and value.isupper()):
            anchors += 1
            continue
        return False
    return anchors >= 3


def _looks_like_proper_name_context(surface: str, text: str) -> bool:
    """Recognize names inside short multi-token entity phrases.

    Capitalization alone is intentionally insufficient because German
    capitalizes common nouns. An adjacent organization marker or another
    non-article capitalized token supplies the missing contextual evidence.
    """
    if not surface[:1].isupper() or not text:
        return False
    tokens = list(re.finditer(r"[^\W\d_][\w'’.-]*", text, flags=re.UNICODE))
    target = surface.casefold()
    for index, match in enumerate(tokens):
        value = match.group(0).strip(".'’-")
        if value.casefold() != target:
            continue
        if index > 0:
            previous = tokens[index - 1].group(0).strip(".'’-").casefold()
            if previous in _PROPER_NAME_LEADERS:
                return True
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(tokens):
                continue
            neighbor_match = tokens[neighbor_index]
            between = text[
                min(match.end(), neighbor_match.end()):
                max(match.start(), neighbor_match.start())
            ]
            if re.search(r"[.!?;:\n]", between):
                continue
            neighbor = neighbor_match.group(0).strip(".'’-")
            folded = neighbor.casefold()
            if folded in _ENTITY_CONTEXT_MARKERS:
                return True
            if (
                neighbor[:1].isupper()
                and folded not in _CAPITALIZED_CONTEXT_STOPWORDS
                and len(neighbor) >= 3
            ):
                return True
    return False


def _has_article_like_leader(surface: str, text: str) -> bool:
    """Return whether a capitalized token is used as a common noun.

    German capitalizes common nouns, so capitalization cannot distinguish
    ``Zerstörung`` from ``Hölderlin``. In translated prose, an immediately
    preceding article is useful generic evidence for the former while names
    without such a determiner remain protected from single-token guesses.
    """
    if not surface or not text:
        return False
    tokens = list(re.finditer(r"[^\W\d_][\w'’.\-]*", text, flags=re.UNICODE))
    target = surface.casefold()
    for index, match in enumerate(tokens):
        value = match.group(0).strip(".'’-")
        if value.casefold() != target or index == 0:
            continue
        previous = tokens[index - 1].group(0).strip(".'’-").casefold()
        if previous in _COMMON_NOUN_DETERMINERS:
            return True
    return False


def _language_gate_glossary_policy(
    prompt_options: Optional[Mapping[str, Any]],
    *,
    source_text: str = "",
) -> tuple[set[str], set[str], set[str]]:
    """Load glossary language policy once for a candidate assessment.

    Returns preserved source tokens, one-token translation obligations, and
    multiword translation obligations.  Keeping this in one pass avoids
    repeatedly loading large generated book profiles for every quality gate.
    """
    options = prompt_options or {}
    preserved: set[str] = set()
    required_tokens: set[str] = set()
    required_phrases: set[str] = set()
    critical_apparatus_context = (
        str(options.get("_document_block_context") or "").strip().casefold()
        == "critical_apparatus"
    )

    def collect(pairs: Mapping[Any, Any]) -> None:
        for source, target in pairs.items():
            source_term = str(source or "").strip()
            target_text = str(target or "").strip()
            if not source_term or not target_text:
                continue
            if source_term.casefold() == target_text.casefold():
                preserved.update(_surface_token_map(source_term))
                continue
            if source_text and _profile_term_is_nested_in_published_title(
                source_text,
                source_term,
            ):
                # The glossary still informs the LLM contextually, but the
                # language gate must not demand a word-level rewrite inside an
                # authentic longer citation title.
                continue
            source_tokens = _ordered_surface_tokens(source_term)
            if (
                critical_apparatus_context
                and source_text
                and _contains_profile_term(source_text, source_term)
                and _looks_like_bibliographic_identity_run(source_tokens)
            ):
                # An approved prose translation must not rewrite an official
                # institution, publisher, journal, or cited-work identity in
                # apparatus metadata. The same glossary entry remains binding
                # outside this explicitly classified context.
                preserved.update(_surface_token_map(source_term))
                continue
            if len(source_tokens) == 1:
                required_tokens.add(source_tokens[0].casefold())
            elif len(source_tokens) > 1:
                required_phrases.add(source_term)

    manual = options.get("glossary_terms") or {}
    if isinstance(manual, Mapping):
        collect(manual)

    profile_id = str(options.get("profile_id") or "").strip()
    if not profile_id:
        return preserved, required_tokens, required_phrases
    try:
        from src.core.book_profiles.rendering import profile_terms_dict

        collect(profile_terms_dict(dict(options), source_text=source_text))
    except Exception:
        # A language gate must remain available even if a profile is damaged.
        pass
    return preserved, required_tokens, required_phrases


def _profile_term_is_nested_in_published_title(text: str, term: str) -> bool:
    try:
        from src.core.book_profiles.rendering import (
            profile_term_nested_in_published_title,
        )

        return profile_term_nested_in_published_title(text, term)
    except Exception:
        return False


def _profile_phrase_matches_published_title(text: str, phrase: str) -> bool:
    try:
        from src.core.book_profiles.rendering import (
            profile_phrase_matches_published_title,
        )

        return profile_phrase_matches_published_title(text, phrase)
    except Exception:
        return False


def _contains_profile_term(text: str, term: str) -> bool:
    """Match a glossary phrase as a complete term, not as a substring."""
    value = str(term or "").strip()
    if not value:
        return False
    pieces = re.split(r"([\s'’]+)", value)
    pattern_parts: list[str] = []
    for piece in pieces:
        if not piece:
            continue
        if piece.isspace():
            pattern_parts.append(r"\s+")
        elif all(char in "'’" for char in piece):
            pattern_parts.append(r"['’]+")
        else:
            pattern_parts.append(re.escape(piece))
    pattern = "".join(pattern_parts)
    return bool(re.search(rf"(?<!\w){pattern}(?!\w)", text or "", re.IGNORECASE))


def _same_script_language_mismatch_issue(
    candidate: str,
    *,
    source: str = "",
    source_language: str,
    target_language: str,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> Optional[FidelityIssue]:
    """Catch whole candidates dominated by the known source language.

    Marker lists cover only a few language pairs.  Full-text language
    detection is reliable enough for long candidates and avoids applying the
    unreliable single-token detector to names.
    """
    if _alpha_word_count(candidate) < 40:
        return None
    source_code = _language_code(_language_key(source_language))
    target_code = _language_code(_language_key(target_language))
    if not source_code or not target_code or source_code == target_code:
        return None
    if _translated_critical_apparatus_has_target_evidence(
        source,
        candidate,
        target_language=target_language,
        prompt_options=prompt_options,
    ):
        return None
    detected_code, confidence = _detect_token_language(
        re.sub(r"\[id\d+\]", " ", candidate or "", flags=re.IGNORECASE)
    )
    if detected_code != source_code or confidence < 0.96:
        return None
    return FidelityIssue(
        "target_language_missing",
        "reject",
        "El candidato sigue dominado por el idioma fuente",
        f"fuente={source_language}; destino={target_language}; detectado={detected_code}; confianza={confidence:.2f}",
    )


_STRUCTURAL_WRAPPER_LINE_RE = re.compile(
    r"^\s*\[\[\[/?(?:VERBALOOMBLOCK|TBLBLOCK|BLOCK)\d+\]\]\]\s*$",
    re.IGNORECASE,
)
_STRUCTURAL_WRAPPER_TOKEN_RE = re.compile(
    r"\[\[\[/?(?:VERBALOOMBLOCK|TBLBLOCK|BLOCK)\d+\]\]\]",
    re.IGNORECASE,
)
_LANGUAGE_NEUTRAL_LITERAL_RE = re.compile(
    r"(?ix)"
    r"(?<![\w@])(?:"
    r"(?:https?://|www\.)[^\s<>\[\]]+"
    r"|doi:\s*10\.\d{4,9}/[^\s<>\[\]]+"
    r"|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,24}(?::\d{2,5})?(?:/[^\s<>\[\]]*)?"
    r"|[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[a-z0-9-]+\.)+[a-z]{2,24}"
    r"|@[a-z0-9_](?:[a-z0-9_.-]{0,63})"
    r")"
)
_BIBLIOGRAPHIC_PUBLISHER_YEAR_RE = re.compile(
    r":\s*[^.;\n]{2,100},\s*(?:1[5-9]\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_JOURNAL_LOCATOR_RE = re.compile(
    r"\((?:1[5-9]\d{2}|20\d{2})\).*?"
    r"\b\d{1,4}\s*\(\d{1,4}\)\s*,\s*\d{1,5}\s*[–—-]\s*\d{1,5}\b",
    re.IGNORECASE | re.DOTALL,
)
_BIBLIOGRAPHIC_ENTRY_RE = re.compile(
    r"(?:^|[.!?]\s+)"
    r"[A-ZÀ-ÖØ-Þ][^.!?\n]{1,90}\.\s+"
    r"[A-ZÀ-ÖØ-Þ][^.!?\n]{2,220}\.\s+"
    r"[^.!?\n]{2,120},\s*(?:1[5-9]\d{2}|20\d{2})\b",
)
_BIBLIOGRAPHIC_CONNECTIVE_PROSE_RE = re.compile(
    r"(?ix)"
    r"(?:^\s*(?:chapter|cap[ií]tulo)\s+\w+)"
    r"|\b(?:see(?:\s+also)?|v[ée]ase(?:\s+tambi[ée]n)?|"
    r"quote\s+is\s+from|la\s+cita\s+es\s+de|"
    r"for\s+more\s+(?:suggested\s+)?reading|"
    r"para\s+m[aá]s\s+lecturas|visit(?:e)?\s+(?:https?://|www\.))\b"
)


def _translated_critical_apparatus_has_target_evidence(
    source: str,
    candidate: str,
    *,
    target_language: str,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Allow translated reference prose while preserving identity data.

    Bibliographies and lexical glossaries can remain statistically dominated
    by the source language after their definitions or connective prose have
    been translated because names, titles and pronunciation spellings are
    identity data. The source must classify as one of those reference blocks
    and the candidate must add several target-language function words. Exact
    or near-exact source echoes are still rejected by earlier fidelity checks.
    """
    structural_source = _clean_structural_language_text(
        source,
        strip_language_neutral_literals=False,
    )
    cleaned_source = _strip_language_neutral_literals(structural_source)
    cleaned_candidate = _clean_structural_language_text(candidate)
    if not cleaned_source or not cleaned_candidate:
        return False

    context = str(
        (prompt_options or {}).get("_document_block_context") or ""
    ).strip().casefold()
    block_type, confidence = "", 0.0
    reference_block_types = {"critical_apparatus", "glossary"}
    if context in reference_block_types:
        block_type, confidence = context, 1.0
    else:
        block_type, _policy, confidence, _strategy, _notes = (
            DocumentBlockClassifier(source_type="text").classify_block(
                structural_source.splitlines()
            )
        )
    if block_type not in reference_block_types or confidence < 0.70:
        return False

    marker = _target_language_marker_pattern(target_language)
    if marker is None:
        return False
    source_markers = _count_markers(marker, cleaned_source)
    candidate_markers = _count_markers(marker, cleaned_candidate)
    if block_type == "glossary":
        # A valid lexical entry may translate to only two words (for example,
        # ``grandfather of`` -> ``abuelo de``) while most of the record remains
        # a proper name and pronunciation key. One new target-language marker
        # is sufficient here because exact/near-exact echoes are rejected by
        # the earlier untranslated-source checks.
        return candidate_markers >= max(1, source_markers + 1)
    return candidate_markers >= max(3, source_markers + 2)


def _looks_like_preservable_bibliographic_echo(
    source: str,
    candidate: str,
    *,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Recognize an unchanged, citation-only critical-apparatus block.

    Published titles, authors, journals and publisher records are identity data
    and the translation prompt explicitly preserves them. A bibliography can
    therefore contain a complete block with no target-language prose at all.
    Requiring target-language markers in that block creates an impossible retry
    loop. This exemption requires explicit critical-apparatus context, a close
    source-aligned candidate, and publication-shape evidence in both texts.
    Small metadata localization such as ``and`` -> ``y`` or ``Rev. ed.`` ->
    ``ed. rev.`` remains valid. Connective note prose is deliberately excluded.
    """
    context = str(
        (prompt_options or {}).get("_document_block_context") or ""
    ).strip().casefold()
    if context != "critical_apparatus":
        return False

    cleaned_source = _clean_structural_language_text(
        source,
        strip_language_neutral_literals=False,
    )
    cleaned_candidate = _clean_structural_language_text(
        candidate,
        strip_language_neutral_literals=False,
    )
    if not cleaned_source or not cleaned_candidate:
        return False
    source_norm = _normalize_text(cleaned_source)
    candidate_norm = _normalize_text(cleaned_candidate)
    similarity = _text_similarity(source_norm, candidate_norm)
    source_words = _alpha_word_count(cleaned_source)
    candidate_words = _alpha_word_count(cleaned_candidate)
    word_ratio = candidate_words / max(1, source_words)
    if similarity < 0.88 or not 0.78 <= word_ratio <= 1.22:
        return False
    if _BIBLIOGRAPHIC_CONNECTIVE_PROSE_RE.search(cleaned_source):
        return False

    source_has_publication_record = bool(
        _BIBLIOGRAPHIC_PUBLISHER_YEAR_RE.search(cleaned_source)
        or _BIBLIOGRAPHIC_JOURNAL_LOCATOR_RE.search(cleaned_source)
        or _BIBLIOGRAPHIC_ENTRY_RE.search(cleaned_source)
        or is_comma_delimited_bibliographic_record(cleaned_source)
    )
    candidate_has_publication_record = bool(
        _BIBLIOGRAPHIC_PUBLISHER_YEAR_RE.search(cleaned_candidate)
        or _BIBLIOGRAPHIC_JOURNAL_LOCATOR_RE.search(cleaned_candidate)
        or _BIBLIOGRAPHIC_ENTRY_RE.search(cleaned_candidate)
        or is_comma_delimited_bibliographic_record(cleaned_candidate)
    )
    if not source_has_publication_record or not candidate_has_publication_record:
        return False

    source_year_count = len(re.findall(
        r"\b(?:1[5-9]\d{2}|20\d{2})\b",
        cleaned_source,
    ))
    candidate_year_count = len(re.findall(
        r"\b(?:1[5-9]\d{2}|20\d{2})\b",
        cleaned_candidate,
    ))
    punctuation_anchors = (
        cleaned_source.count(",")
        + cleaned_source.count(".")
        + cleaned_source.count(":")
    )
    return (
        source_year_count >= 1
        and candidate_year_count == source_year_count
        and punctuation_anchors >= 4
    )


def _strip_language_neutral_literals(text: str) -> str:
    """Remove URLs and email addresses from statistical language evidence."""
    decoded = html.unescape(text or "")
    return re.sub(
        r"\s+",
        " ",
        _LANGUAGE_NEUTRAL_LITERAL_RE.sub(" ", decoded),
    ).strip()


def _clean_structural_language_text(
    text: str,
    *,
    strip_language_neutral_literals: bool = True,
) -> str:
    lines = []
    # Recovery markers are normally emitted on isolated lines, but providers
    # may attach them to neighboring prose. They are pipeline metadata, never
    # language evidence.
    structural_text = _STRUCTURAL_WRAPPER_TOKEN_RE.sub(" ", text or "")
    for raw_line in structural_text.splitlines():
        if _STRUCTURAL_WRAPPER_LINE_RE.match(raw_line):
            continue
        lines.append(raw_line)
    value = "\n".join(lines)
    value = re.sub(r"\[id\d+\]", " ", value, flags=re.IGNORECASE)
    if strip_language_neutral_literals:
        value = _strip_language_neutral_literals(value)
    return value.strip()


def _looks_like_preservable_name_index_echo(
    source: str,
    candidate: str,
    *,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> bool:
    """Recognize structured name/catalog runs that should remain unchanged.

    Analytical indexes and table cells frequently contain nothing but person
    names, organizations, brands or work titles. Translating those entries
    would corrupt names, while treating the unchanged run as prose causes an
    expensive repair loop. The exemption requires structural evidence and
    overwhelmingly name/title-shaped tokens. Any substantive lower-case source
    lexeme keeps an exact echo subject to the normal language gate.
    """
    cleaned_source = _clean_structural_language_text(source)
    cleaned_candidate = _clean_structural_language_text(candidate)
    if not cleaned_source or not cleaned_candidate:
        return False

    source_norm = _normalize_text(cleaned_source)
    candidate_norm = _normalize_text(cleaned_candidate)
    tokens = _ordered_surface_tokens(cleaned_source)
    if not 3 <= len(tokens) <= 120:
        return False

    placeholder_count = len(_PLACEHOLDER_PATTERNS[0].findall(source or ""))
    wrapper_count = len(re.findall(
        r"\[\[\[(?:VERBALOOMBLOCK|TBLBLOCK|BLOCK)\d+\]\]\]",
        source or "",
        flags=re.IGNORECASE,
    ))
    line_count = len([
        line for line in cleaned_source.splitlines() if line.strip()
    ])
    comma_count = cleaned_source.count(",")
    structured_context = str(
        (prompt_options or {}).get("_document_block_context") or ""
    ).strip().casefold()
    quoted_translatable_tokens: set[str] = set()
    if structured_context in {"table", "catalog", "index"}:
        for quoted_span in re.findall(
            r"[\"\u201c]([^\"\u201d]{3,500})[\"\u201d]",
            cleaned_source,
        ):
            quoted_tokens = _ordered_surface_tokens(quoted_span)
            uppercase_words = [
                token
                for token in quoted_tokens
                if len(token) >= 2 and token.isupper()
            ]
            if len(uppercase_words) >= 3:
                quoted_translatable_tokens.update(
                    token.casefold()
                    for token in uppercase_words
                    if token.casefold() not in _WORK_TITLE_CONNECTORS
                )
    similarity = _text_similarity(source_norm, candidate_norm)
    if (
        similarity < 0.90
        and structured_context not in {"table", "catalog", "index"}
    ):
        return False
    structured_items = max(placeholder_count, wrapper_count, line_count)
    has_catalog_structure = (
        comma_count >= 2
        or structured_items >= 3
        or structured_context in {"table", "catalog", "index"}
    )
    if not has_catalog_structure:
        return False
    if (
        comma_count + structured_items < 4
        and structured_context not in {"table", "catalog", "index"}
    ):
        return False

    title_anchors = 0
    non_name_lexemes: list[str] = []
    for token in tokens:
        folded = token.casefold()
        if folded in quoted_translatable_tokens:
            non_name_lexemes.append(token)
            continue
        if (
            structured_context in {"table", "catalog", "index"}
            and folded in _INDEX_TRANSLATABLE_QUALIFIERS
        ):
            non_name_lexemes.append(token)
            continue
        if folded in _WORK_TITLE_CONNECTORS:
            continue
        is_short_acronym = 2 <= len(token) <= 5 and token.isupper()
        is_internal_capitalized_brand = (
            not token.isupper()
            and any(char.islower() for char in token)
            and any(char.isupper() for char in token[1:])
        )
        is_normally_capitalized = token[:1].isupper() and not (
            len(token) > 5 and token.isupper()
        )
        if (
            is_normally_capitalized
            or is_short_acronym
            or is_internal_capitalized_brand
        ):
            title_anchors += 1
            continue
        non_name_lexemes.append(token)

    if title_anchors < max(3, int(len(tokens) * 0.60)):
        return False
    if len(non_name_lexemes) > max(1, int(len(tokens) * 0.30)):
        return False

    # A prose sentence can contain several names; sentence punctuation and
    # substantive lower-case words keep it out of this narrow exemption.
    if (
        structured_context not in {"table", "catalog", "index"}
        and re.search(r"[!?;:]", cleaned_source)
    ):
        return False

    if source_norm == candidate_norm:
        if structured_context in {"table", "catalog", "index"}:
            return all(
                token.casefold() in _INDEX_TRANSLATABLE_QUALIFIERS
                for token in non_name_lexemes
            )
        return not non_name_lexemes

    # Mixed index runs may contain a few translatable subject entries among
    # many immutable names. Accept the high overall similarity only when each
    # substantive lower-case source lexeme disappeared from the candidate;
    # leaving even one such entry untouched keeps the normal gate active.
    candidate_tokens = {
        token.casefold() for token in _ordered_surface_tokens(cleaned_candidate)
    }
    if structured_context in {"table", "catalog", "index"}:
        candidate_token_count = len(
            _ordered_surface_tokens(cleaned_candidate)
        )
        candidate_token_ratio = candidate_token_count / max(1, len(tokens))
        source_anchor_tokens = {
            token.casefold()
            for token in tokens
            if token.casefold() not in _WORK_TITLE_CONNECTORS
            and token not in non_name_lexemes
        }
        preserved_anchor_ratio = (
            len(source_anchor_tokens & candidate_tokens)
            / max(1, len(source_anchor_tokens))
        )
        if (
            not 0.65 <= candidate_token_ratio <= 1.75
            or preserved_anchor_ratio < 0.85
        ):
            return False
    if structured_context in {"table", "catalog", "index"}:
        source_counter = Counter(
            token.casefold()
            for token in tokens
            if token.casefold() not in _WORK_TITLE_CONNECTORS
        )
        candidate_counter = Counter(
            token.casefold()
            for token in _ordered_surface_tokens(cleaned_candidate)
            if token.casefold() not in _WORK_TITLE_CONNECTORS
        )
        missing_source_tokens = source_counter - candidate_counter
        allowed_missing_tokens = {
            token.casefold() for token in non_name_lexemes
        }
        return (
            bool(missing_source_tokens)
            and set(missing_source_tokens) <= allowed_missing_tokens
            and quoted_translatable_tokens.isdisjoint(candidate_tokens)
            and _extract_numbers(cleaned_source) == _extract_numbers(cleaned_candidate)
        )
    return bool(non_name_lexemes) and all(
        token.casefold() not in candidate_tokens
        for token in non_name_lexemes
    )


def _target_language_marker_pattern(language: str) -> Optional[re.Pattern[str]]:
    key = _language_key(language)
    if key in {"spanish", "espanol", "español", "es"}:
        return _SPANISH_MARKER_RE
    if key in {"english", "en"}:
        return _ENGLISH_MARKER_RE
    if key in {"french", "francais", "français", "fr"}:
        return _FRENCH_MARKER_RE
    return None


def _detect_token_language(token: str) -> tuple[str, float]:
    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        matches = detect_langs(token)
        if not matches:
            return "", 0.0
        return str(matches[0].lang).lower(), float(matches[0].prob)
    except Exception:
        return "", 0.0


def _language_code(language_key: str) -> str:
    aliases = {
        "english": "en", "en": "en",
        "spanish": "es", "espanol": "es", "español": "es", "es": "es",
        "german": "de", "aleman": "de", "alemán": "de", "deutsch": "de", "de": "de",
        "french": "fr", "frances": "fr", "francés": "fr", "fr": "fr",
        "italian": "it", "italiano": "it", "it": "it",
        "portuguese": "pt", "portugues": "pt", "portugués": "pt", "pt": "pt",
        "dutch": "nl", "nederlands": "nl", "nl": "nl",
        "czech": "cs", "checo": "cs", "cs": "cs",
        "polish": "pl", "polaco": "pl", "pl": "pl",
    }
    return aliases.get(language_key, language_key if len(language_key) == 2 else "")


def _has_foreign_target_diacritic(token: str, target_key: str) -> bool:
    allowed = {
        "spanish": set("áéíóúüñ"), "espanol": set("áéíóúüñ"),
        "español": set("áéíóúüñ"), "es": set("áéíóúüñ"),
        "english": set(), "en": set(),
        "german": set("äöüß"), "de": set("äöüß"),
        "french": set("àâæçéèêëîïôœùûüÿ"), "fr": set("àâæçéèêëîïôœùûüÿ"),
    }.get(target_key)
    if allowed is None:
        return False
    for char in token.casefold():
        if not char.isalpha() or ord(char) < 128 or char in allowed:
            continue
        decomposed = unicodedata.normalize("NFD", char)
        if any(unicodedata.combining(part) for part in decomposed[1:]) or char not in allowed:
            return True
    return False


def assess_fidelity(
    source_text: str,
    candidate_text: str,
    *,
    chunk_index: int,
    phase: str,
    section: str = "Documento",
    source_language: str = "",
    target_language: str = "",
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> FidelityDecision:
    """Run deterministic source-vs-candidate checks without spending tokens."""
    source = source_text or ""
    candidate = candidate_text or ""
    issues: list[FidelityIssue] = []
    source_chars = _content_chars(source)
    candidate_chars = _content_chars(candidate)
    source_paragraphs = _paragraph_count(source)
    candidate_paragraphs = _paragraph_count(candidate)
    source_pdf_noise = _looks_like_pdf_extraction_noise(source)
    source_boundary_fragment = _looks_like_chunk_boundary_fragment(source)
    semantic_source = _clean_structural_language_text(source)
    semantic_candidate = _clean_structural_language_text(candidate)

    if source.strip() and not candidate.strip():
        issues.append(FidelityIssue("empty_candidate", "reject", "El candidato quedo vacio"))

    if (
        _content_chars(semantic_source) > 20
        and _alpha_word_count(semantic_source) >= 3
        and _normalize_text(semantic_source) == _normalize_text(semantic_candidate)
        and not _looks_like_preservable_name_index_echo(
            source,
            candidate,
            prompt_options=prompt_options,
        )
        and not _looks_like_preservable_bibliographic_echo(
            source,
            candidate,
            prompt_options=prompt_options,
        )
    ):
        if _different_languages(source_language, target_language):
            issues.append(FidelityIssue(
                "untranslated_source",
                "reject",
                "El candidato parece conservar la fuente sin traducir",
            ))

    for gate_issue in target_language_gate_issues(
        source,
        candidate,
        source_language=source_language,
        target_language=target_language,
        phase=phase,
        prompt_options=prompt_options,
    ):
        if not any(
            issue.code == gate_issue.code and issue.severity == gate_issue.severity
            for issue in issues
        ):
            issues.append(gate_issue)

    source_artifacts = _artifact_count(source)
    candidate_artifacts = _artifact_count(candidate)
    if candidate_artifacts > source_artifacts:
        issues.append(FidelityIssue(
            "artifact_glyphs_added",
            "reject",
            "El candidato agrego glifos raros/cuadros",
            f"{source_artifacts} -> {candidate_artifacts}",
        ))
    elif candidate_artifacts > 0:
        issues.append(FidelityIssue(
            "artifact_glyphs_present",
            "warning",
            "El candidato conserva glifos raros/cuadros",
            str(candidate_artifacts),
        ))

    if mojibake_score(candidate) > mojibake_score(source):
        issues.append(FidelityIssue(
            "mojibake_regression",
            "reject",
            "El candidato empeoro la codificacion",
        ))

    if source_pdf_noise:
        issues.append(FidelityIssue(
            "source_pdf_extraction_noise",
            "warning",
            "La fuente parece contener ruido de extraccion PDF/OCR",
        ))
    if source_boundary_fragment:
        issues.append(FidelityIssue(
            "chunk_boundary_fragment",
            "warning",
            "El chunk parece terminar a media frase o con paginacion",
        ))

    _check_missing_tokens(
        "numbers_lost",
        "Se perdieron numeros o cantidades presentes en la fuente",
        _extract_numbers(source, language=source_language),
        _extract_numbers(candidate, language=target_language),
        issues,
        reject_threshold=0.35,
        warning_only=source_pdf_noise or source_boundary_fragment,
    )
    _check_missing_tokens(
        "citations_lost",
        "Se perdieron citas o referencias presentes en la fuente",
        _extract_citations(source),
        _extract_citations(candidate),
        issues,
        reject_threshold=0.0,
    )
    _check_missing_tokens(
        "formula_fragments_lost",
        "Se perdieron fragmentos de formulas o variables",
        _extract_formula_fragments(source),
        _extract_formula_fragments(candidate),
        issues,
        reject_threshold=0.35,
        min_source_items=4,
    )
    _check_missing_tokens(
        "proper_nouns_lost",
        "Hay nombres propios de la fuente que no aparecen en el candidato",
        _extract_proper_nouns(source),
        _extract_proper_nouns(candidate),
        issues,
        reject_threshold=0.50,
        min_source_items=4,
        warning_only=True,
    )
    missing_symbol_names = missing_symbol_bearing_names(source, candidate)
    if missing_symbol_names:
        issues.append(FidelityIssue(
            "symbol_bearing_names_lost",
            "reject",
            "Se alteraron nombres o identificadores que contienen signos significativos",
            ", ".join(list(missing_symbol_names.elements())[:12]),
        ))
    _check_placeholders(source, candidate, issues)
    _check_length_ratio(source_chars, candidate_chars, issues)
    _check_paragraph_ratio(source_paragraphs, candidate_paragraphs, issues)

    if _SENSITIVE_RE.search(source):
        issues.append(FidelityIssue(
            "sensitive_content_needs_audit",
            "warning",
            "La fuente contiene contenido sensible que conviene auditar contra omision o suavizado",
        ))

    accepted = not any(issue.severity == "reject" for issue in issues)
    return FidelityDecision(
        chunk_index=chunk_index,
        phase=phase or "translation",
        section=section or "Documento",
        accepted=accepted,
        issues=issues,
        local_accepted=accepted,
        source_chars=source_chars,
        candidate_chars=candidate_chars,
        source_paragraphs=source_paragraphs,
        candidate_paragraphs=candidate_paragraphs,
        source_hash=_hash_text(source),
        candidate_hash=_hash_text(candidate),
        source_snippet=_snippet(source),
        candidate_snippet=_snippet(candidate),
        independence="local",
    )


@dataclass(frozen=True)
class FidelityAuditPrompt:
    system: str
    user: str


def _profile_fidelity_audit_context(
    source_text: str,
    candidate_text: str,
    prompt_options: Optional[Mapping[str, Any]],
    *,
    max_entries: int = 24,
) -> str:
    """Return compact, book-scoped terminology evidence for the auditor.

    Approved entries are authoritative. High-confidence pending entries are
    included only when the candidate already uses their proposed rendering;
    they can explain an established localization, but cannot make the pipeline
    apply an unapproved term. This keeps the fidelity judge from mistaking a
    canonical name or localized title for a changed fact without allowing one
    book's glossary to affect another.
    """
    options = prompt_options or {}
    profile_id = str(options.get("profile_id") or "").strip()
    if not profile_id:
        return ""
    try:
        from src.core.book_profiles.loader import load_book_profile
        from src.core.book_profiles.rendering import profile_terms_dict

        profile = load_book_profile(profile_id)
        approved_terms = profile_terms_dict(dict(options), source_text=source_text)
    except Exception:
        return ""
    if profile is None:
        return ""

    approved_missing: list[tuple[Any, str]] = []
    approved_present: list[tuple[Any, str]] = []
    pending: list[tuple[Any, str]] = []
    for entry in profile.glossary_entries:
        source = str(entry.source or "").strip()
        target = str(entry.target or "").strip() or source
        if not source or not target:
            continue
        source_matches = _contains_profile_term(source_text, source)
        candidate_matches = _contains_profile_term(candidate_text, target)
        if entry.approved and source_matches and source in approved_terms:
            destination = approved_present if candidate_matches else approved_missing
            destination.append((entry, target))
        elif (
            entry.pending
            and float(entry.confidence or 0.0) >= 0.90
            and source_matches
            and candidate_matches
        ):
            pending.append((entry, target))

    issue_context = json.dumps(
        options.get("_fidelity_adjudication_context") or {},
        ensure_ascii=False,
    ).casefold()

    def contextual_priority(item: tuple[Any, str]) -> tuple[int, int]:
        entry, target = item
        mentioned = int(
            bool(issue_context)
            and (
                str(entry.source or "").casefold() in issue_context
                or str(target or "").casefold() in issue_context
            )
        )
        return mentioned, len(str(entry.source or ""))

    approved_missing.sort(key=contextual_priority, reverse=True)
    pending.sort(key=contextual_priority, reverse=True)
    approved_present.sort(key=contextual_priority, reverse=True)

    # Put actionable violations and candidate-rendered localization hints ahead
    # of already-satisfied entries so the small prompt budget cannot hide the
    # evidence that triggered the audit.
    approved = approved_missing + approved_present
    ranked = approved_missing + pending + approved_present
    if not ranked:
        return ""
    ranked = ranked[:max(1, int(max_entries or 1))]

    lines = [
        "# ACTIVE PROFILE FIDELITY CONTEXT",
        f"Profile: {profile.profile_id}",
        (
            "APPROVED entries are authoritative book-scoped terminology. "
            "PENDING HINT entries are non-binding evidence only: they may explain "
            "a localization already present, but they cannot excuse omission, "
            "addition, damaged names, or changed meaning."
        ),
    ]
    approved_ids = {id(entry) for entry, _target in approved}
    for entry, target in ranked:
        if id(entry) in approved_ids:
            status = "APPROVED"
        else:
            status = f"PENDING HINT confidence={float(entry.confidence or 0.0):.2f}"
        policy = str(entry.translation_policy or entry.injection_policy or entry.entry_type or "term")
        lines.append(
            f'- {status} [{policy}]: "{entry.source}" -> "{target}"'
        )
    return "\n".join(lines)


def build_fidelity_audit_prompt(
    source_text: str,
    candidate_text: str,
    *,
    source_language: str = "",
    target_language: str = "",
    phase: str = "translation",
    section: str = "",
    local_decision: Optional[FidelityDecision] = None,
    prior_assessment: Optional[Mapping[str, Any]] = None,
    profile_context: str = "",
    document_context: str = "",
) -> FidelityAuditPrompt:
    local_issues = []
    if local_decision is not None:
        local_issues = [
            {
                "code": issue.code,
                "severity": issue.severity,
                "detail": issue.detail,
            }
            for issue in local_decision.issues
        ]

    adjudication_rules = ""
    if prior_assessment:
        adjudication_rules = """
This is an adjudication pass over a prior rejection. Treat PRIOR_AUDIT as an untrusted opinion, not as evidence.
Resolve contradictions between its verdict, reason, and structured fields. Do not repeat a finding merely because
the prior audit or local precheck asserted it. Return your own internally consistent verdict and fields.
"""

    system_prompt = f"""You are an independent bilingual fidelity auditor.

{UNTRUSTED_BOOK_CONTENT_SECTION}

Your only job is to compare a source passage with a candidate {target_language or 'target-language'} text.
Do not improve the translation. Do not rewrite. Audit whether the candidate preserves the source.

Pay special attention to omissions, additions, censorship, euphemistic softening, changed facts, changed numbers,
damaged names, changed chronology, lost causal links, lost formulas/citations, and structure loss.
Only put MATERIAL semantic losses in missing_from_source, added_not_in_source, changed_facts, or
censored_or_softened. A natural target-locale equivalent, regional vocabulary that preserves meaning,
minor punctuation preference, or stylistic alternative is not a material fidelity error; mention it only
in issues and do not reject the candidate for it. Preserve symbols that are internal to names exactly.
Standard target-language exonyms, localized spellings of historical names, institutions, missions, places,
and translated standalone title mentions are not changed facts when the referent remains unambiguous. An isolated obvious source
typo or OCR anomaly may be normalized when the correct form is established repeatedly by the same nearby passage;
record that normalization in issues, but do not put it in changed_facts or reject it. Never require the candidate
to reproduce extraction damage or an internally inconsistent typo merely to satisfy a token-count precheck.
In notes, references, and bibliographies, authentic published work titles, journal names, publisher names, author
names, platforms, and registry labels are identity-bearing citation data and may remain in their original language
when the surrounding explanatory prose is translated. Do not reject them as source-language residue or glossary
violations. A partly translated or mixed-language cited title is a defect: it must be restored completely or
translated completely according to an explicit book policy, never assembled word by word from glossary entries.
Quoted interview answers, speeches, social-media posts, letters, and messages are content rather than work titles;
require their prose to be translated even when a citation, platform name, date, handle, or URL follows the quotation.
Sensitive or offensive source content must be preserved faithfully in translation; do not approve omission or sanitization.
Profanity and idioms may use a natural target-locale equivalent when they preserve pragmatic force. Do not treat
nonliteral wording alone as censorship. However, adding a speaker identity, gender, anatomy, motive, or cultural fact
that the source leaves unspecified is an unsupported addition even when the replacement sounds equally forceful.
Dialogue, narration, and internal monologue are supposed to be translated into the target language. Translating them
does not by itself change narrative convention, voice, grammatical person, or factual content; compare their meaning,
speaker, intent, and pragmatic force instead of requiring source-language wording.
When an explicit measurement is localized, require the original numeric value and unit to remain present. A converted
value may be added in parentheses, but replacing the source measurement with a rounded conversion is a fidelity defect.
For same-language transformations such as modernization, do not require literal wording. Accept safe wording updates,
but reject changes that alter speaker/addressee relationships, grammatical person, dialogue intent, quoted meaning,
tone that carries narrative information, ambiguity that the source intentionally preserves, or the order of ideas.
If SOURCE is visibly damaged by PDF/OCR extraction (glued words, page numbers on isolated lines, table-of-contents
page numbers glued to headings, or broken line/page boundaries), judge semantic and structural fidelity, not literal
preservation of extraction damage. Chunks may split a sentence or word; do not fail solely because the final fragment
continues in the next chunk when the candidate preserves the same incomplete continuation.
Your verdict, reason, and structured arrays must agree. If the reason concludes that there is no material fidelity
error, use pass or warn and leave the material-error arrays empty.
{adjudication_rules}

Return ONLY valid JSON wrapped in {FIDELITY_AUDIT_TAG_IN} and {FIDELITY_AUDIT_TAG_OUT}.
Do not include markdown or explanation outside the tags.

JSON schema:
{{
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
}}"""

    prior_block = ""
    if prior_assessment:
        prior_block = (
            "# PRIOR AUDIT TO ADJUDICATE\n"
            + json.dumps(dict(prior_assessment), ensure_ascii=False)
            + "\n\n"
        )
    profile_block = f"{profile_context.strip()}\n\n" if profile_context.strip() else ""
    context = str(document_context or "").strip().casefold()
    document_policy_block = ""
    if context in {"index", "catalog"}:
        document_policy_block = """# STRUCTURED DOCUMENT POLICY
This passage is an analytical index or catalog. Mixed-language surface text is
expected and correct: preserve identity-bearing names, brands, institutions,
places, letter headings, and work titles, while translating common-noun
subjects, qualifiers, cross-references, and descriptions. Evaluate each entry
by meaning and identity. Never reject the candidate merely because preserved
names remain in the source language beside translated subject entries.

"""
    elif context == "table":
        document_policy_block = """# STRUCTURED DOCUMENT POLICY
This passage is a table. Preserve identity-bearing names and data while
translating headers, generic labels, and explanatory cells. Mixed-language
proper names are not source-language residue.

"""

    user_prompt = f"""# AUDIT CONTEXT
Phase: {phase or 'translation'}
Section: {section or 'Documento'}
Source language: {source_language or 'unknown'}
Target language: {target_language or 'unknown'}

{document_policy_block}{profile_block}# LOCAL PRECHECK
{json.dumps(local_issues, ensure_ascii=False)}

{prior_block}# SOURCE
{source_text}

# CANDIDATE
{candidate_text}

# TASK
Decide whether CANDIDATE faithfully preserves SOURCE in {target_language or 'the target language'}.
Do not judge literary polish unless it changes fidelity."""

    return FidelityAuditPrompt(system=system_prompt, user=user_prompt)


def _normalize_structured_index_audit_policy(
    source_text: str,
    candidate_text: str,
    assessment: Mapping[str, Any],
    *,
    document_context: str,
) -> dict[str, Any]:
    """Downgrade a judge that mistakes correct mixed-language index policy for loss."""
    normalized = dict(assessment)
    context = str(document_context or "").strip().casefold()
    verdict = str(normalized.get("verdict") or "").strip().casefold()
    if context not in {"index", "catalog"} or verdict not in {
        "fail",
        "repair_needed",
    }:
        return normalized
    if any(
        _list_of_str(normalized.get(key))
        for key in ("changed_facts", "censored_or_softened", "structure_issues")
    ):
        return normalized

    missing = _list_of_str(normalized.get("missing_from_source"))
    added = _list_of_str(normalized.get("added_not_in_source"))
    if not missing or len(missing) != len(added):
        return normalized
    if any(value not in (source_text or "") for value in missing):
        return normalized
    if any(value not in (candidate_text or "") for value in added):
        return normalized

    def is_common_subject(value: str) -> bool:
        tokens = _ordered_surface_tokens(value)
        return bool(tokens) and all(
            not token[:1].isupper()
            and not (len(token) >= 2 and token.isupper())
            for token in tokens
        )

    if not all(is_common_subject(value) for value in missing):
        return normalized

    audit_text = " ".join([
        str(normalized.get("reason") or ""),
        " ".join(_list_of_str(normalized.get("issues"))),
    ])
    policy_cue = re.search(
        r"\b(?:index|catalog|entries|translation policy|mixed.language|"
        r"inconsistent(?:ly)? translated|preserved proper)\b",
        audit_text,
        re.IGNORECASE,
    )
    semantic_error = re.search(
        r"\b(?:wrong meaning|meaning changed|mistranslat|semantic|omitted|"
        r"unrelated|reversed|contradict)\w*\b",
        audit_text,
        re.IGNORECASE,
    )
    if not policy_cue or semantic_error:
        return normalized

    normalized["verdict"] = "warn"
    normalized["missing_from_source"] = []
    normalized["added_not_in_source"] = []
    issues = _list_of_str(normalized.get("issues"))
    issues.append("structured_index_mixed_language_policy_is_expected")
    normalized["issues"] = list(dict.fromkeys(issues))
    normalized["reason"] = (
        "The candidate follows the analytical-index policy: identity-bearing "
        "entries remain exact while common subject entries are translated."
    )
    return normalized


def parse_fidelity_audit_response(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    payload = (
        extract_tagged_payload(text, FIDELITY_AUDIT_TAG_IN, FIDELITY_AUDIT_TAG_OUT)
        or text
    )
    parsed = loads_first_json_object(payload)
    if parsed is None:
        return None
    verdict = str(parsed.get("verdict") or parsed.get("decision") or "").strip().lower()
    if verdict not in {"pass", "warn", "fail", "repair_needed"}:
        return None
    parsed["verdict"] = verdict
    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    return parsed


def apply_fidelity_audit_assessment(
    decision: FidelityDecision,
    assessment: Mapping[str, Any],
    *,
    model: str,
    provider: str = "",
    primary_model: str = "",
    primary_provider: str = "",
) -> FidelityDecision:
    verdict = str(assessment.get("verdict") or "").strip().lower()
    confidence = _bounded_float(assessment.get("confidence"), 0.0, 1.0)

    decision.judge_model = model
    decision.judge_provider = provider
    decision.judge_decision = verdict
    decision.judge_confidence = confidence
    decision.judge_reason = str(assessment.get("reason") or "")
    decision.judge_issues = _list_of_str(assessment.get("issues"))
    decision.judge_missing_from_source = _list_of_str(assessment.get("missing_from_source"))
    decision.judge_added_not_in_source = _list_of_str(assessment.get("added_not_in_source"))
    decision.judge_changed_facts = _list_of_str(assessment.get("changed_facts"))
    decision.judge_censored_or_softened = _list_of_str(assessment.get("censored_or_softened"))
    decision.judge_evidence_source = _list_of_str(assessment.get("evidence_source"))
    decision.judge_evidence_candidate = _list_of_str(assessment.get("evidence_candidate"))
    decision.independence = infer_auditor_independence(
        translator_provider=primary_provider,
        translator_model=primary_model,
        auditor_provider=provider,
        auditor_model=model,
        llm_judge_used=True,
    )

    if (
        verdict in {"fail", "repair_needed"}
        and confidence >= 0.72
        and _audit_numeric_finding_contradicts_its_evidence(decision, assessment)
    ):
        decision.judge_decision = "warn"
        decision.judge_issues.append(
            "auditor_self_contradiction: numeric evidence is identical"
        )
        decision.judge_changed_facts = []
        decision.issues.append(FidelityIssue(
            "fidelity_judge_numeric_self_contradiction",
            "warning",
            "El juez reporto un cambio numerico que contradice su propia evidencia",
            decision.judge_reason,
        ))
        decision.accepted = not any(
            issue.severity == "reject" for issue in decision.issues
        )
        return decision

    if verdict == "pass" and confidence >= 0.88:
        force_rejections = [
            issue for issue in decision.rejections
            if issue.code in _FORCE_REJECT_CODES
        ]
        if force_rejections:
            decision.accepted = False
            return decision
        if decision.rejections:
            decision.issues = [
                issue for issue in decision.issues
                if issue.severity != "reject"
            ]
            decision.issues.append(FidelityIssue(
                "fidelity_judge_override_pass",
                "warning",
                "El juez LLM aprobo el candidato pese a alertas locales",
                decision.judge_reason,
            ))
        decision.accepted = True
        return decision

    if verdict == "warn" and confidence >= 0.72:
        material_findings = _material_fidelity_findings(assessment)
        if material_findings and not _audit_failure_is_ocr_or_boundary_only(decision, assessment):
            decision.issues.append(FidelityIssue(
                "fidelity_judge_reject",
                "reject",
                "El juez de fidelidad detecto perdida, agregado, cambio de datos o suavizado",
                decision.judge_reason or "; ".join(material_findings[:4]),
            ))
            decision.accepted = False
            return decision
        force_rejections = [
            issue for issue in decision.rejections
            if issue.code in _FORCE_REJECT_CODES
        ]
        if force_rejections:
            decision.accepted = False
            return decision
        if decision.rejections:
            decision.issues = [
                FidelityIssue(
                    issue.code,
                    "warning",
                    issue.message,
                    issue.detail,
                )
                if issue.severity == "reject"
                else issue
                for issue in decision.issues
            ]
            decision.issues.append(FidelityIssue(
                "fidelity_judge_override_warn",
                "warning",
                "El juez LLM redujo alertas locales a advertencias auditables",
                decision.judge_reason,
            ))
        decision.issues.append(FidelityIssue(
            "fidelity_judge_warning",
            "warning",
            "El juez de fidelidad marco dudas que requieren auditoria",
            decision.judge_reason,
        ))
        decision.accepted = True
        return decision

    if verdict in {"fail", "repair_needed"} and confidence >= 0.72:
        if _audit_failure_is_ocr_or_boundary_only(decision, assessment):
            if decision.rejections:
                decision.issues = [
                    FidelityIssue(
                        issue.code,
                        "warning",
                        issue.message,
                        issue.detail,
                    )
                    if issue.severity == "reject" and issue.code not in _FORCE_REJECT_CODES
                    else issue
                    for issue in decision.issues
                ]
            decision.issues.append(FidelityIssue(
                "fidelity_judge_ocr_boundary_warning",
                "warning",
                "El juez marco ruido de OCR/paginacion o corte de chunk; se conserva como alerta",
                decision.judge_reason,
            ))
            decision.accepted = not any(issue.severity == "reject" for issue in decision.issues)
            return decision
        decision.issues.append(FidelityIssue(
            "fidelity_judge_reject",
            "reject",
            "El juez de fidelidad detecto perdida, agregado, cambio de datos o suavizado",
            decision.judge_reason,
        ))
        decision.accepted = False
        return decision

    if verdict in {"warn", "fail", "repair_needed"}:
        decision.issues.append(FidelityIssue(
            "fidelity_judge_warning",
            "warning",
            "El juez de fidelidad marco dudas que requieren auditoria",
            decision.judge_reason,
        ))

    decision.accepted = not any(issue.severity == "reject" for issue in decision.issues)
    return decision


async def supervise_fidelity(
    source_text: str,
    candidate_text: str,
    *,
    chunk_index: int,
    phase: str,
    section: str = "Documento",
    source_language: str = "",
    target_language: str = "",
    primary_model: str = "",
    primary_provider: str = "",
    client: Any = None,
    prompt_options: Optional[Mapping[str, Any]] = None,
    log_callback=None,
) -> tuple[FidelityDecision, Any]:
    """Run local checks plus optional LLM audit, returning decision and response."""
    decision = assess_fidelity(
        source_text,
        candidate_text,
        chunk_index=chunk_index,
        phase=phase,
        section=section,
        source_language=source_language,
        target_language=target_language,
        prompt_options=prompt_options,
    )

    audit_response = None
    if _should_run_llm_audit(decision, prompt_options) and client is not None:
        auditor_model = resolve_fidelity_auditor_model(primary_model, prompt_options)
        auditor_provider = str((prompt_options or {}).get("fidelity_supervisor_provider") or primary_provider or "")
        profile_context = _profile_fidelity_audit_context(
            source_text,
            candidate_text,
            prompt_options,
        )
        prompt_pair = build_fidelity_audit_prompt(
            source_text,
            candidate_text,
            source_language=source_language,
            target_language=target_language,
            phase=phase,
            section=section,
            local_decision=decision,
            prior_assessment=(prompt_options or {}).get("_fidelity_adjudication_context"),
            profile_context=profile_context,
            document_context=str(
                (prompt_options or {}).get("_document_block_context") or ""
            ),
        )
        if log_callback:
            log_callback(
                "fidelity_supervisor_request",
                f"Fidelity supervisor for {phase} chunk {chunk_index} with {auditor_model}",
                data={
                    "type": "fidelity_supervisor_request",
                    "system_prompt": prompt_pair.system,
                    "user_prompt": prompt_pair.user,
                    "model": auditor_model,
                    "primary_model": primary_model,
                    "phase": phase,
                },
            )
        mode = str((prompt_options or {}).get("fidelity_supervisor_mode") or "").strip().lower()
        strict_structured_audit = mode == "strict_full" or phase.startswith("final_epub_unit_")
        configured_parse_retries = (prompt_options or {}).get(
            "fidelity_supervisor_invalid_json_retries"
        )
        try:
            parse_retries = int(configured_parse_retries)
        except (TypeError, ValueError):
            parse_retries = 2 if strict_structured_audit else 0
        parse_retries = max(0, min(parse_retries, 2))

        assessment = None
        content_filter_unavailable = False
        for parse_attempt in range(parse_retries + 1):
            if parse_attempt and log_callback:
                log_callback(
                    "fidelity_supervisor_parse_retry",
                    f"Retrying structured fidelity audit for {phase} chunk {chunk_index} "
                    f"after invalid JSON ({parse_attempt}/{parse_retries}).",
                )
            retry_system = prompt_pair.system
            if parse_attempt:
                retry_system += (
                    "\n\nSTRICT SERIALIZATION RETRY: the previous response could not be parsed. "
                    "Return one complete JSON object inside the requested tags. Use double-quoted "
                    "JSON keys and strings, no trailing commas, no markdown, and no prose outside "
                    "the tags. Keep reason and evidence concise so the object cannot be truncated."
                )
            try:
                audit_response = await _request_with_temporary_model(
                    client,
                    prompt_pair.user,
                    retry_system,
                    primary_model=primary_model,
                    auditor_model=auditor_model,
                    temperature=temperature_for_phase("fidelity_audit"),
                )
            except ContentRiskError:
                content_filter_unavailable = True
                decision.issues.append(FidelityIssue(
                    "fidelity_auditor_content_filter",
                    "warning",
                    "El proveedor impidio la auditoria LLM por su filtro de contenido",
                    "Se conservaron los guards deterministas y se registro la falta de juicio remoto.",
                ))
                if log_callback:
                    log_callback(
                        "fidelity_supervisor_content_filter",
                        f"Fidelity auditor unavailable for {phase} chunk {chunk_index}: "
                        "provider content filter; using deterministic checks.",
                    )
                break
            if audit_response:
                assessment = parse_fidelity_audit_response(audit_response.content)
            if assessment:
                break

        if assessment:
            assessment = _normalize_structured_index_audit_policy(
                source_text,
                candidate_text,
                assessment,
                document_context=str(
                    (prompt_options or {}).get("_document_block_context") or ""
                ),
            )
            decision = apply_fidelity_audit_assessment(
                decision,
                assessment,
                model=auditor_model,
                provider=auditor_provider,
                primary_model=primary_model,
                primary_provider=primary_provider,
            )
        elif log_callback and not content_filter_unavailable:
            log_callback(
                "fidelity_supervisor_parse_failed",
                f"Fidelity supervisor returned invalid JSON for {phase} chunk {chunk_index} "
                f"after {parse_retries + 1} attempt(s); using local checks.",
            )
        if log_callback:
            log_callback(
                "fidelity_supervisor_decision",
                f"Fidelity supervisor: {phase} chunk {chunk_index} -> "
                f"{'accepted' if decision.accepted else 'rejected'}"
                + (f" ({decision.judge_decision}, {decision.judge_confidence:.2f})" if decision.judge_decision else ""),
                data={
                    "type": "fidelity_supervisor_decision",
                    "accepted": decision.accepted,
                    "phase": phase,
                    "chunk_index": chunk_index,
                    "issues": [issue.code for issue in decision.issues],
                    "judge_decision": decision.judge_decision,
                    "confidence": decision.judge_confidence,
                    "independence": decision.independence,
                },
            )

    report = (prompt_options or {}).get("_fidelity_report")
    if report is not None and hasattr(report, "add"):
        report.add(decision)
    record_candidate_result(
        prompt_options,
        CandidateResult.from_fidelity_decision(
            decision,
            text=candidate_text,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            response=audit_response,
        ),
    )
    return decision, audit_response


def build_fidelity_retry_instructions(decision: FidelityDecision) -> str:
    issue_codes = ", ".join(issue.code for issue in decision.rejections or decision.warnings)
    judge_bits = []
    if decision.judge_missing_from_source:
        judge_bits.append("missing: " + "; ".join(decision.judge_missing_from_source[:5]))
    if decision.judge_added_not_in_source:
        judge_bits.append("unsupported additions: " + "; ".join(decision.judge_added_not_in_source[:5]))
    if decision.judge_changed_facts:
        judge_bits.append("changed facts: " + "; ".join(decision.judge_changed_facts[:5]))
    if decision.judge_censored_or_softened:
        judge_bits.append("softening/censorship: " + "; ".join(decision.judge_censored_or_softened[:5]))
    details = "\n".join(f"- {bit}" for bit in judge_bits)
    return f"""# FIDELITY RETRY

Your previous candidate failed a source-fidelity audit.
- Do not omit, summarize, sanitize, censor, soften, or add content.
- Preserve all names, numbers, dates, chronology, causal links, citations, formulas, and sensitive/offensive source content.
- In notes and bibliographies, preserve complete authentic cited-work titles and registry identifiers exactly.
  Translate surrounding prose, but never return a partly translated or mixed-language title.
- Translate quoted interviews, speeches, social-media posts, letters, and messages completely; they are prose,
  not cited-work titles, even when followed by a platform, date, handle, or URL.
- Preserve every explicit measurement's original numeric value and unit. You may add an exact localized conversion in parentheses, but never replace the source measurement with a rounded conversion.
- Translate faithfully into the target language while keeping natural prose.
- Preserve the pragmatic force of profanity and idioms with a natural target-locale equivalent; do not calque them word for word.
- Do not invent a speaker's identity, gender, anatomy, motive, or cultural detail when the source leaves it unspecified.
- Specific audit alerts: {issue_codes or 'source fidelity failure'}.
{details}
"""


def build_fidelity_retry_prompt_options(
    prompt_options: Optional[Mapping[str, Any]],
    decision: FidelityDecision,
) -> dict[str, Any]:
    options = dict(prompt_options or {})
    retry_instructions = build_fidelity_retry_instructions(decision)
    existing = str(options.get("custom_instructions") or "").strip()
    options["custom_instructions"] = (
        f"{existing}\n\n{retry_instructions}" if existing else retry_instructions
    )
    return options


def write_fidelity_report_from_options(
    output_filepath: str | Path,
    prompt_options: Optional[Mapping[str, Any]],
    *,
    log_callback=None,
) -> Optional[Path]:
    report = (prompt_options or {}).get("_fidelity_report") if prompt_options else None
    if report is None or not hasattr(report, "has_records") or not report.has_records():
        return None
    path = report.write(fidelity_report_path(output_filepath))
    if log_callback:
        counts = report.summary_counts()
        log_callback(
            "fidelity_report_saved",
            "Fidelity report saved: "
            f"{path.name} "
            f"({counts['accepted']} accepted, {counts['rejected']} rejected, "
            f"{counts['warnings']} warnings, {counts['llm_judged']} LLM-audited)."
        )
    return path


def _should_run_llm_audit(
    decision: FidelityDecision,
    prompt_options: Optional[Mapping[str, Any]],
) -> bool:
    if not fidelity_supervisor_enabled(prompt_options):
        return False
    mode = fidelity_supervisor_mode(prompt_options)
    if mode in {"off", "local"}:
        return False
    if not decision.source_chars or not decision.candidate_chars:
        return False
    if mode in {"always", "strict", "strict_full"}:
        return True
    if decision.rejections or decision.warnings:
        return True
    if mode == "sampled":
        options = prompt_options or {}
        rate = int(options.get("fidelity_supervisor_sample_rate") or 20)
        rate = max(1, rate)
        return decision.chunk_index == 1 or decision.chunk_index % rate == 0
    return False


async def _request_with_temporary_model(
    client: Any,
    prompt: str,
    system_prompt: str,
    *,
    primary_model: str,
    auditor_model: str,
    temperature: float | None = None,
) -> Any:
    if not client:
        return None
    if not auditor_model or auditor_model == primary_model or not hasattr(client, "make_request"):
        if hasattr(client, "generate"):
            return await await_llm_call(
                client.generate,
                prompt,
                provider=client,
                system_prompt=system_prompt,
                temperature=temperature,
            )
        return await await_llm_call(
            client.make_request,
            prompt,
            provider=client,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    provider = None
    original_model = None
    try:
        if hasattr(client, "_get_provider"):
            provider = client._get_provider()
            original_model = getattr(provider, "model", None)
        return await await_llm_call(
            client.make_request,
            prompt,
            auditor_model,
            provider=client,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    finally:
        if provider is not None and original_model:
            provider.model = original_model


def _check_missing_tokens(
    code: str,
    message: str,
    source_tokens: Counter[str],
    candidate_tokens: Counter[str],
    issues: list[FidelityIssue],
    *,
    reject_threshold: float,
    min_source_items: int = 1,
    warning_only: bool = False,
) -> None:
    total_source = sum(source_tokens.values())
    if total_source < min_source_items:
        return
    missing = source_tokens - candidate_tokens
    missing_total = sum(missing.values())
    if missing_total <= 0:
        return
    ratio = missing_total / max(1, total_source)
    detail = ", ".join(list(missing.elements())[:12])
    severity = "warning" if warning_only or ratio <= reject_threshold else "reject"
    issues.append(FidelityIssue(code, severity, message, detail))


def _check_placeholders(source: str, candidate: str, issues: list[FidelityIssue]) -> None:
    for pattern in _PLACEHOLDER_PATTERNS:
        source_items = Counter(pattern.findall(source or ""))
        if not source_items:
            continue
        candidate_items = Counter(pattern.findall(candidate or ""))
        if source_items != candidate_items:
            missing = source_items - candidate_items
            extra = candidate_items - source_items
            detail_bits = []
            if missing:
                detail_bits.append("missing: " + ", ".join(list(missing.elements())[:8]))
            if extra:
                detail_bits.append("extra: " + ", ".join(list(extra.elements())[:8]))
            issues.append(FidelityIssue(
                "placeholder_mismatch",
                "reject",
                "Los marcadores estructurales no coinciden con la fuente",
                "; ".join(detail_bits),
            ))
            return


def _check_length_ratio(source_chars: int, candidate_chars: int, issues: list[FidelityIssue]) -> None:
    if source_chars < 80 or candidate_chars <= 0:
        return
    ratio = candidate_chars / max(1, source_chars)
    if source_chars >= 120 and ratio < 0.35:
        issues.append(FidelityIssue(
            "severe_length_drop",
            "reject",
            "El candidato parece truncado respecto a la fuente",
            f"{ratio:.2f}x",
        ))
    elif source_chars >= 300 and ratio < 0.45:
        issues.append(FidelityIssue(
            "severe_length_drop",
            "reject",
            "El candidato es demasiado corto respecto a la fuente",
            f"{ratio:.2f}x",
        ))
    elif ratio < 0.58:
        issues.append(FidelityIssue(
            "length_drop",
            "warning",
            "El candidato parece mucho mas corto que la fuente",
            f"{ratio:.2f}x",
        ))
    elif ratio > 2.20:
        issues.append(FidelityIssue(
            "length_expansion",
            "warning",
            "El candidato parece mucho mas largo que la fuente",
            f"{ratio:.2f}x",
        ))


def _check_paragraph_ratio(source_paragraphs: int, candidate_paragraphs: int, issues: list[FidelityIssue]) -> None:
    if source_paragraphs >= 3 and candidate_paragraphs <= 1:
        issues.append(FidelityIssue(
            "paragraph_collapse",
            "warning",
            "La estructura de parrafos se colapso respecto a la fuente",
            f"{source_paragraphs} -> {candidate_paragraphs}",
        ))


def _extract_numbers(text: str, *, language: str = "") -> Counter[str]:
    value = text or ""
    ignored_ocr_spans = _probable_english_ocr_i_spans(value, language=language)
    time_matches = list(_TIME_NUMBER_RE.finditer(value))
    time_spans = [match.span() for match in time_matches]
    raw_tokens = [
        f"{int(match.group('hour'))}:{match.group('minute')}"
        for match in time_matches
    ]

    def overlaps_time(span: tuple[int, int]) -> bool:
        return any(span[0] < end and span[1] > start for start, end in time_spans)

    raw_tokens.extend(
        _normalize_number_token(match.group(0))
        for match in _NUMBER_RE.finditer(value)
        if match.span() not in ignored_ocr_spans and not overlaps_time(match.span())
    )
    raw_tokens.extend(match.group(1) for match in _DECADE_NUMBER_RE.finditer(value))
    ignored = _isolated_page_number_tokens(value)
    tokens: list[str] = []
    for token in raw_tokens:
        if ignored[token] > 0:
            ignored[token] -= 1
            continue
        tokens.append(token)
    return Counter(tokens)


def _probable_english_ocr_i_spans(text: str, *, language: str = "") -> set[tuple[int, int]]:
    """Locate isolated ``1`` glyphs that are almost certainly OCR for English ``I``.

    The guard remains conservative: a token is ignored only at a clause boundary
    (including comma-delimited historical prose), before a common first-person
    verb, and when the declared source language is English. Real quantities such
    as ``1 mile`` or ``1 person`` remain protected.
    """
    normalized_language = _language_key(language)
    if normalized_language not in {"english", "en"}:
        return set()
    value = text or ""
    spans = {
        match.span("token")
        for match in _ENGLISH_OCR_I_AFTER_CUE_RE.finditer(value)
    }
    for match in _ENGLISH_OCR_I_TOKEN_RE.finditer(value):
        token_span = match.span("token")
        if token_span in spans:
            continue

        tail = value[token_span[1]:]
        tail = re.sub(
            r"^(?:\s|\[id\d+\])*",
            "",
            tail,
            flags=re.IGNORECASE,
        )
        following_words = _ordered_surface_tokens(tail)[:4]
        word_index = 0
        while (
            word_index < len(following_words)
            and following_words[word_index].casefold() in _ENGLISH_OCR_I_ADVERBS
        ):
            word_index += 1
        if word_index >= len(following_words):
            continue
        verb = following_words[word_index].casefold()
        if verb not in _ENGLISH_OCR_I_VERBS:
            continue
        next_word = (
            following_words[word_index + 1].casefold()
            if word_index + 1 < len(following_words)
            else ""
        )
        if (
            verb in _ENGLISH_OCR_I_NOMINAL_VERBS
            and next_word in _ENGLISH_OCR_I_NOMINAL_FOLLOWERS
        ):
            continue

        prefix = value[:token_span[0]].rstrip()
        prefix_without_placeholders = re.sub(
            r"\[id\d+\]",
            " ",
            prefix,
            flags=re.IGNORECASE,
        )
        preceding_words = _ordered_surface_tokens(prefix_without_placeholders)
        previous_word = preceding_words[-1].casefold() if preceding_words else ""
        punctuation_boundary = bool(
            not prefix
            or unicodedata.category(prefix[-1]).startswith("P")
        )
        clause_cue = previous_word in _ENGLISH_OCR_I_CLAUSE_CUES
        strong_verb_context = bool(
            verb in _ENGLISH_OCR_I_STRONG_VERBS
            and previous_word not in _ENGLISH_OCR_I_QUANTITY_CUES
        )
        if punctuation_boundary or clause_cue or strong_verb_context:
            spans.add(token_span)
    return spans


def _extract_citations(text: str) -> Counter[str]:
    return Counter(_CITATION_RE.findall(text or ""))


def _extract_formula_fragments(text: str) -> Counter[str]:
    tokens = [
        token for token in _FORMULA_RE.findall(text or "")
        if _is_meaningful_formula_fragment(token)
    ]
    return Counter(tokens)


def _is_meaningful_formula_fragment(token: str) -> bool:
    """Return True for formula-like tokens without treating prose hyphens as math.

    Literary texts contain many hyphens and dashes. Counting every standalone
    hyphen as a formula fragment made the fidelity guard reject good prose
    translations before the LLM auditor could make a nuanced call.
    """
    value = (token or "").strip()
    if not value:
        return False
    if value in {"-", "+", "*", "/", "^", "_", "{", "}"}:
        return False
    if value in {"=", "<", ">"}:
        return True
    return any(ch in value for ch in "=<>+-*/^_{}") or bool(re.search(r"[\u0370-\u03ff]", value))


def _extract_proper_nouns(text: str) -> Counter[str]:
    items: list[str] = []
    for match in _PROPER_NOUN_RE.findall(text or ""):
        words = _WORD_RE.findall(match)
        if not words:
            continue
        if len(words) == 1 and words[0] in _COMMON_CAPITALIZED:
            continue
        item = " ".join(words)
        if item in _COMMON_CAPITALIZED:
            continue
        items.append(item)
    return Counter(items)


def _extract_symbol_bearing_names(text: str) -> Counter[str]:
    return extract_symbol_bearing_names(text)


def _artifact_count(text: str) -> int:
    return len(_ARTIFACT_RE.findall(text or ""))


def _normalize_number_token(token: str) -> str:
    return (token or "").strip().rstrip("sS").replace(",", ".")


def _isolated_page_number_tokens(text: str) -> Counter[str]:
    tokens: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not _ISOLATED_PAGE_NUMBER_RE.fullmatch(stripped):
            continue
        tokens.extend(_normalize_number_token(token) for token in _NUMBER_RE.findall(stripped))
    return Counter(tokens)


def _looks_like_pdf_extraction_noise(text: str) -> bool:
    value = text or ""
    if not value.strip():
        return False
    dot_leaders = len(re.findall(r"\.{5,}", value))
    glued_toc_numbers = len(_PDF_TOC_GLUE_RE.findall(value))
    isolated_page_lines = sum(
        1 for line in value.splitlines()
        if _ISOLATED_PAGE_NUMBER_RE.fullmatch(line.strip() or "")
    )
    long_glued_words = len(re.findall(r"\b[\wÁÉÍÓÚÜÑáéíóúüñ]{24,}\b", value, flags=re.UNICODE))
    return (
        dot_leaders >= 3
        or glued_toc_numbers >= 2
        or isolated_page_lines >= 1
        or long_glued_words >= 4
        or _looks_like_damaged_markdown_table(value)
    )


def _looks_like_damaged_markdown_table(text: str) -> bool:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    table_lines = [line for line in lines if line.lstrip().startswith("|")]
    if len(table_lines) < 3:
        return False
    if not any(_MARKDOWN_TABLE_SEPARATOR_RE.match(line) for line in table_lines):
        return False

    pipe_counts = [line.count("|") for line in table_lines]
    expected = max(pipe_counts)
    short_rows = sum(1 for count in pipe_counts if count < expected - 1)
    unterminated_rows = sum(1 for line in table_lines if not line.rstrip().endswith("|"))
    sparse_rows = sum(1 for line in table_lines if re.search(r"\|\s*\|", line))
    return unterminated_rows > 0 or short_rows > 0 or sparse_rows >= max(2, len(table_lines) // 2)


def _looks_like_chunk_boundary_fragment(text: str) -> bool:
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    if _CHUNK_BOUNDARY_END_RE.search(stripped):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return bool(lines and _ISOLATED_PAGE_NUMBER_RE.fullmatch(lines[-1]))


def _audit_failure_is_ocr_or_boundary_only(
    decision: FidelityDecision,
    assessment: Mapping[str, Any],
) -> bool:
    issue_codes = {issue.code for issue in decision.issues}
    if not issue_codes.intersection({"source_pdf_extraction_noise", "chunk_boundary_fragment"}):
        return False

    concerns: list[str] = []
    for key in (
        "issues",
        "missing_from_source",
        "added_not_in_source",
        "changed_facts",
        "structure_issues",
    ):
        concerns.extend(_list_of_str(assessment.get(key)))
    reason = str(assessment.get("reason") or "").strip()
    if reason:
        concerns.append(reason)
    if not concerns:
        return False
    return all(
        _OCR_BOUNDARY_FAILURE_RE.search(re.sub(r"[_-]+", " ", item or ""))
        for item in concerns
    )


def _material_fidelity_findings(assessment: Mapping[str, Any]) -> list[str]:
    """Return structured findings that cannot be downgraded by ``warn``.

    Judge models occasionally choose a lenient top-level verdict while still
    reporting concrete omissions, additions, altered facts, or censorship in
    their structured fields. Those fields are the stronger contract.
    """
    findings: list[str] = []
    reason = str(assessment.get("reason") or "")
    minor_only = bool(re.search(r"\b(?:minor|slight(?:ly)?)\b", reason, re.IGNORECASE))
    for key in (
        "missing_from_source",
        "added_not_in_source",
        "changed_facts",
        "censored_or_softened",
    ):
        for finding in _list_of_str(assessment.get(key)):
            if _explicitly_nonmaterial_judge_finding(finding, minor_only=minor_only):
                continue
            findings.append(finding)
    return findings


_NUMERIC_FACT_CLAIM_RE = re.compile(
    r"\b(?:date|year|number|amount|time|age|page|percent(?:age)?|"
    r"fecha|año|numero|número|cantidad|hora|edad|pagina|página|porcentaje)\b",
    re.IGNORECASE,
)


def _audit_numeric_finding_contradicts_its_evidence(
    decision: FidelityDecision,
    assessment: Mapping[str, Any],
) -> bool:
    """Detect a judge claim disproved by its own quoted numeric evidence.

    URLs often contain a publication or archive year different from the
    citation date. A judge can read the URL year as prose, claim that the date
    changed, and still quote source and candidate excerpts containing the exact
    same numbers. This narrow check only applies when deterministic fidelity
    checks passed, numeric/date change is the sole material finding, and both
    evidence sides contain identical numeric multisets.
    """
    if not decision.local_accepted:
        return False

    changed_facts = _list_of_str(assessment.get("changed_facts"))
    if not changed_facts:
        return False
    if any(
        _list_of_str(assessment.get(key))
        for key in (
            "missing_from_source",
            "added_not_in_source",
            "censored_or_softened",
            "structure_issues",
        )
    ):
        return False

    claim_text = " ".join(
        [str(assessment.get("reason") or ""), *changed_facts]
    )
    if not (_NUMERIC_FACT_CLAIM_RE.search(claim_text) or re.search(r"\d", claim_text)):
        return False

    source_evidence = "\n".join(
        _list_of_str(assessment.get("evidence_source"))
    )
    candidate_evidence = "\n".join(
        _list_of_str(assessment.get("evidence_candidate"))
    )
    if not source_evidence.strip() or not candidate_evidence.strip():
        return False

    source_numbers = _extract_numbers(source_evidence)
    candidate_numbers = _extract_numbers(candidate_evidence)
    return bool(source_numbers) and source_numbers == candidate_numbers


def _explicitly_nonmaterial_judge_finding(finding: str, *, minor_only: bool) -> bool:
    """Ignore judge notes that explicitly concede semantic equivalence.

    The structured fidelity fields remain authoritative for real omissions and
    changed facts. This narrow exception prevents contradictory judge output
    such as "playera is acceptable and not inaccurate" from becoming a hard
    rejection merely because it was placed in the wrong JSON array.
    """
    text = re.sub(r"\s+", " ", str(finding or "").strip()).casefold()
    if not text:
        return True
    explicit_equivalence = (
        "meaning is preserved",
        "meaning preserved",
        "preserves the meaning",
        "preserves meaning",
        "preserves the emotional intensity",
        "preserves emotional intensity",
        "emotional intensity is preserved",
        "natural equivalent",
        "semantically equivalent",
        "not inaccurate",
        "not softening",
        "no censorship detected",
        "localization choice",
        "localisation choice",
        "equally colloquial",
        "equally vulgar",
        "equally strong",
        "same pragmatic force",
        "same degree of profanity",
        "equivalent idiom",
        "pragmatic force is preserved",
        "pragmatic force remains",
        "does not materially alter",
        "does not constitute censorship",
    )
    if any(marker in text for marker in explicit_equivalence):
        return True
    if minor_only and any(marker in text for marker in (
        "nuance",
        "regional term",
        "less specific",
        "capitalization",
        "capitalisation",
        "reduplication",
        "stylistic",
        "idiom",
        "expletive",
        "profanity",
        "curse",
        "direct equivalent",
    )):
        return True
    return False


def _paragraph_count(text: str) -> int:
    paragraphs = [part for part in re.split(r"\n\s*\n", (text or "").strip()) if part.strip()]
    if len(paragraphs) > 1:
        return len(paragraphs)
    return 1 if (text or "").strip() else 0


def _content_chars(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def _target_language_gate_enabled(prompt_options: Optional[Mapping[str, Any]]) -> bool:
    options = prompt_options or {}
    if _truthy(options.get("disable_target_language_gate")):
        return False
    raw = str(options.get("target_language_gate") or options.get("language_gate") or "on").strip().lower()
    return raw not in _OFF_VALUES


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _language_key(language: str) -> str:
    value = (language or "").strip().lower()
    value = re.sub(r"\([^)]*\)", "", value).strip()
    value = value.replace("_", "-")
    value = re.split(r"\s+[-–—]\s+|-|,", value, maxsplit=1)[0].strip()
    if value.startswith("spanish") or value.startswith("espanol") or value.startswith("español"):
        return "spanish"
    if value.startswith("english") or value.startswith("ingles") or value.startswith("inglés"):
        return "english"
    if value.startswith("french") or value.startswith("frances") or value.startswith("francés"):
        return "french"
    if value.startswith("greek") or value.startswith("griego"):
        return "greek"
    return value


def _script_for_language(language: str) -> str:
    return _LANGUAGE_SCRIPT_ALIASES.get(_language_key(language), "")


def _script_profile(text: str) -> dict[str, float]:
    letters = [ch for ch in text or "" if ch.isalpha()]
    total = len(letters)
    if total == 0:
        return {}
    counts = {
        script: len(pattern.findall(text or ""))
        for script, pattern in _SCRIPT_PATTERNS.items()
    }
    return {script: count / total for script, count in counts.items() if count}


def _dominant_script(profile: Mapping[str, float]) -> tuple[str, float]:
    if not profile:
        return "", 0.0
    script, ratio = max(profile.items(), key=lambda item: item[1])
    return script, ratio


def _translation_expected(
    source_language: str,
    target_language: str,
    *,
    source_script: str,
    target_script: str,
    source_dominant_script: str,
    candidate_dominant_script: str,
) -> bool:
    source = _language_key(source_language)
    target = _language_key(target_language)
    if not target or target in _AUTO_LANGUAGE_KEYS:
        return False
    if source and source not in _AUTO_LANGUAGE_KEYS:
        if source != target:
            return True
        return False
    if target_script and source_dominant_script and source_dominant_script != target_script:
        return True
    return bool(target_script and candidate_dominant_script and candidate_dominant_script != target_script)


def _text_similarity(source_norm: str, candidate_norm: str) -> float:
    if not source_norm or not candidate_norm:
        return 0.0
    source_sample = source_norm[:3000]
    candidate_sample = candidate_norm[:3000]
    return difflib.SequenceMatcher(None, source_sample, candidate_sample).ratio()


def _count_markers(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def _exact_latin_echo_issue(
    source: str,
    candidate: str,
    *,
    source_language: str,
    target_language: str,
) -> Optional[FidelityIssue]:
    if _normalize_text(source) != _normalize_text(candidate):
        return None
    if _alpha_word_count(candidate) < 3:
        return None
    target = _language_key(target_language)
    source_key = _language_key(source_language)
    if source_key and source_key not in _AUTO_LANGUAGE_KEYS and source_key == target:
        return None

    if target in {"spanish", "es"}:
        english_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        spanish_markers = _count_markers(_SPANISH_MARKER_RE, candidate)
        if english_markers >= 1 and spanish_markers == 0:
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato parece ser un eco exacto en ingles pese a que el destino es espanol",
                f"marcadores_en={english_markers}; marcadores_es={spanish_markers}",
            )
    elif target in {"english", "en"}:
        spanish_markers = _count_markers(_SPANISH_MARKER_RE, candidate)
        english_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        if spanish_markers >= 2 and english_markers == 0:
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato parece ser un eco exacto en espanol pese a que el destino es ingles",
                f"marcadores_es={spanish_markers}; marcadores_en={english_markers}",
            )
    elif target in {"french", "fr"}:
        english_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        french_markers = _count_markers(_FRENCH_MARKER_RE, candidate)
        if english_markers >= 1 and french_markers == 0:
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato parece ser un eco exacto en ingles pese a que el destino es frances",
                f"marcadores_en={english_markers}; marcadores_fr={french_markers}",
            )
    return None


def _latin_language_gate_issue(
    source: str,
    candidate: str,
    *,
    source_language: str,
    target_language: str,
    translation_expected: bool,
    similarity: float,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> Optional[FidelityIssue]:
    target = _language_key(target_language)
    source_key = _language_key(source_language)
    candidate_words = _alpha_word_count(candidate)
    if candidate_words < 40:
        return None
    if _translated_critical_apparatus_has_target_evidence(
        source,
        candidate,
        target_language=target_language,
        prompt_options=prompt_options,
    ):
        return None

    if target in {"spanish", "espanol", "español", "es"}:
        target_markers = _count_markers(_SPANISH_MARKER_RE, candidate)
        english_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        source_english_markers = _count_markers(_ENGLISH_MARKER_RE, source)
        if (
            english_markers >= 10
            and target_markers <= max(3, english_markers // 5)
            and (
                translation_expected
                or source_key in {"", *_AUTO_LANGUAGE_KEYS}
                or source_english_markers >= 8
                or similarity >= 0.70
            )
        ):
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato conserva demasiado texto en ingles pese a que el destino es espanol",
                f"marcadores_en={english_markers}; marcadores_es={target_markers}",
            )
    elif target in {"english", "en"}:
        target_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        spanish_markers = _count_markers(_SPANISH_MARKER_RE, candidate)
        source_spanish_markers = _count_markers(_SPANISH_MARKER_RE, source)
        if (
            spanish_markers >= 10
            and target_markers <= max(3, spanish_markers // 5)
            and (
                translation_expected
                or source_key in {"", *_AUTO_LANGUAGE_KEYS}
                or source_spanish_markers >= 8
                or similarity >= 0.70
            )
        ):
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato conserva demasiado texto en espanol pese a que el destino es ingles",
                f"marcadores_es={spanish_markers}; marcadores_en={target_markers}",
            )
    elif target in {"french", "frances", "francés", "fr"}:
        target_markers = _count_markers(_FRENCH_MARKER_RE, candidate)
        english_markers = _count_markers(_ENGLISH_MARKER_RE, candidate)
        if (
            english_markers >= 10
            and target_markers <= max(3, english_markers // 5)
            and (translation_expected or source_key in {"", *_AUTO_LANGUAGE_KEYS})
        ):
            return FidelityIssue(
                "target_language_missing",
                "reject",
                "El candidato conserva demasiado texto en ingles pese a que el destino es frances",
                f"marcadores_en={english_markers}; marcadores_fr={target_markers}",
            )
    return None


def _unique_issues(issues: list[FidelityIssue]) -> list[FidelityIssue]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[FidelityIssue] = []
    for issue in issues:
        key = (issue.code, issue.severity, issue.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique


def _different_languages(source_language: str, target_language: str) -> bool:
    source = (source_language or "").strip().lower()
    target = (target_language or "").strip().lower()
    if not source or not target:
        return False
    if source in _AUTO_LANGUAGE_KEYS:
        return False
    return source != target


def _alpha_word_count(text: str) -> int:
    return sum(1 for word in _WORD_RE.findall(text or "") if any(ch.isalpha() for ch in word))


def _snippet(text: str, limit: int = 260) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _list_of_str(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _bounded_float(value: Any, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = low
    return max(low, min(high, number))


def _format_model(provider: str, model: str) -> str:
    if provider and model:
        return f"{provider} / {model}"
    return model or provider or "N/D"
