"""Deterministic guardrails for raw LLM-visible output.

The model-facing protocol uses tags, section headings, and instruction blocks
that must never become reader-visible text. This module keeps those checks in
one place so translation, transformation, refinement, and repair can share the
same cleanup and evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable

from src.core.candidate_result import CandidateIssue
from src.utils.text_encoding import clean_text_artifacts


_CODE_FENCE_RE = re.compile(r"^\s*```(?:[a-z0-9_-]+)?\s*$|^\s*```\s*$", re.IGNORECASE)
_PROTOCOL_TAG_RE = re.compile(
    r"(?:<|&lt;)\s*/?\s*(?:TRANSLATION(?:ATION)*|SOURCE|TARGET|INPUT|OUTPUT|TEXT|"
    r"TEXT_TO_TRANSLATE|TEXT_TO_TRANSFORM|DRAFT|CANDIDATE|NER_JSON|"
    r"FIDELITY_AUDIT_JSON|EDITORIAL_GUARD_JSON|PROFILE_AUDIT_JSON|"
    r"PROFILE_TERM_REVIEW_JSON|GLOSSARY_DISCOVERY_JSON|CORRECTED_TAG_IN|CORRECTED_TAG_OUT)"
    r"(?:\s+[^>]*?)?\s*(?:>|&gt;)",
    re.IGNORECASE,
)
_PROMPT_HEADING_RE = re.compile(
    r"^\s{0,3}#{1,6}\s*(?:"
    r"system|user|assistant|instructions?|rules?|requirements?|task|objective|"
    r"sistema|usuario|asistente|instrucciones?|reglas?|requisitos?|tarea|objetivo|"
    r"text to translate|text to transform|source text|target text|original source|"
    r"texto a traducir|texto a transformar|texto fuente|texto objetivo|fuente original|"
    r"candidate|draft|glossary|continuity|surrounding source context|"
    r"candidato|borrador|glosario|continuidad|contexto fuente circundante|"
    r"surrounding draft context|output format|return format|fidelity retry|"
    r"contexto de borrador circundante|formato de salida|formato de respuesta|reintento de fidelidad|"
    r"editorial guard|quality audit|repair instructions"
    r"|guard editorial|auditor[ií]a de calidad|instrucciones de reparaci[oó]n"
    r")\b",
    re.IGNORECASE,
)
_PROMPT_LINE_RE = re.compile(
    r"^\s*(?:"
    r"system\s*:|user\s*:|assistant\s*:|developer\s*:|"
    r"sistema\s*:|usuario\s*:|asistente\s*:|desarrollador\s*:|"
    r"you are (?:an?|the)\s+|your task is to\s+|"
    r"eres (?:un|una|el|la)\s+|tu tarea es\s+|"
    r"return only\b|respond only\b|do not translate this block\b|"
    r"devuelve solo\b|responde solo\b|no traduzcas este bloque\b|"
    r"do not output this context\b|do not include explanations\b|"
    r"no incluyas este contexto\b|no incluyas explicaciones\b|"
    r"here is (?:the|your)\s+(?:translation|rewriting|transformation)\b|"
    r"(?:aqu[ií]|ac[aá]) (?:est[aá]|tienes) (?:la|tu)\s+(?:traducci[oó]n|reescritura|transformaci[oó]n)\b|"
    r"as an ai\b|i cannot\b|i can't\b|i am unable\b"
    r"|como (?:ia|modelo de lenguaje)\b|no puedo\b"
    r")",
    re.IGNORECASE,
)
_META_PREFIX_WITH_CONTENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:sure|certainly|claro|por supuesto)[,! ]+\s*)?"
    r"(?:here is (?:the|your)|(?:aqu[ií]|ac[aá]) (?:est[aá]|tienes) (?:la|tu))\s+"
    r"(?:translation|rewriting|transformation|traducci[oó]n|reescritura|transformaci[oó]n)"
    r"\s*[:\-]\s*(?P<content>.+?)\s*$",
    re.IGNORECASE,
)
_INPUT_MARKER_RE = re.compile(
    r"\b(?:INPUT_TAG_IN|INPUT_TAG_OUT|TRANSLATE_TAG_IN|TRANSLATE_TAG_OUT|"
    r"BEGIN_(?:SOURCE|INPUT|TRANSLATION)|END_(?:SOURCE|INPUT|TRANSLATION))\b",
    re.IGNORECASE,
)
_READER_ARTIFACT_PREFIX_RE = re.compile(
    r"^\s*(?P<prefix>(?:(?:descripci[oó]n\s+de\s+(?:la\s+)?imagen|image\s+description)"
    r"\s*:\s*)+)(?P<content>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_WORD_RE = re.compile(r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?。！？]+")
_DIALOGUE_RE = re.compile(r"(^|\n)\s*(?:[-–—]|[\"“«])")


@dataclass(frozen=True)
class OutputGuardResult:
    text: str
    issues: list[CandidateIssue] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    changed: bool = False


def guard_llm_output(
    text: str,
    *,
    phase: str = "",
    style_reference: str = "",
    allow_meta_intro: bool = False,
) -> OutputGuardResult:
    """Clean protocol leaks and return deterministic quality evidence.

    The guard is intentionally conservative: it only removes standalone protocol
    lines, known wrapper tags, code fences, and meta-introductions that are not
    book content. Anything ambiguous is reported as an issue rather than
    aggressively rewritten.
    """

    original = text or ""
    raw_protocol_hits = _residual_protocol_hits(original)
    cleaned, prompt_leak_count, reader_artifact_count = _strip_prompt_leak_lines(
        clean_text_artifacts(original),
        allow_meta_intro,
    )
    second_pass, second_count, second_artifact_count = _strip_prompt_leak_lines(
        clean_text_artifacts(cleaned),
        allow_meta_intro,
    )
    idempotent = second_pass == cleaned
    if not idempotent:
        cleaned = second_pass
        prompt_leak_count += second_count
        reader_artifact_count += second_artifact_count
    if raw_protocol_hits:
        prompt_leak_count = max(prompt_leak_count, len(raw_protocol_hits))

    issues: list[CandidateIssue] = []
    if reader_artifact_count:
        issues.append(
            CandidateIssue(
                "reader_artifact_label_cleaned",
                "warning",
                "Reader-visible generated labels were removed from the candidate.",
                detail=f"removed_prefixes={reader_artifact_count}",
                source=phase or "llm_output_guard",
            )
        )
    if prompt_leak_count:
        issues.append(
            CandidateIssue(
                "llm_protocol_leak_cleaned",
                "warning",
                "Reader-visible prompt or wrapper protocol was removed from the candidate.",
                detail=f"removed_lines_or_tags={prompt_leak_count}",
                source=phase or "llm_output_guard",
            )
        )
    residual = _residual_protocol_hits(cleaned)
    if residual:
        issues.append(
            CandidateIssue(
                "llm_protocol_leak_residual",
                "reject",
                "The candidate still appears to contain prompt protocol or assistant meta text.",
                detail="; ".join(residual[:4]),
                source=phase or "llm_output_guard",
            )
        )
    if not idempotent:
        issues.append(
            CandidateIssue(
                "non_idempotent_output_cleanup",
                "warning",
                "Output cleanup needed more than one pass before stabilizing.",
                source=phase or "llm_output_guard",
            )
        )

    scores = _style_scores(cleaned, style_reference)
    if reader_artifact_count:
        scores["reader_artifact_labels_removed"] = float(reader_artifact_count)
    if prompt_leak_count:
        scores["prompt_protocol_items_removed"] = float(prompt_leak_count)
    drift = scores.get("style_drift", 0.0)
    if drift >= 0.50 and _word_count(cleaned) >= 80 and _word_count(style_reference) >= 80:
        issues.append(
            CandidateIssue(
                "possible_style_drift",
                "warning",
                "The candidate's sentence/paragraph/dialogue cadence differs sharply from adjacent accepted text.",
                detail=f"style_drift={drift:.2f}",
                source=phase or "llm_output_guard",
            )
        )

    return OutputGuardResult(
        text=cleaned,
        issues=issues,
        scores=scores,
        changed=cleaned != original,
    )


def _strip_prompt_leak_lines(text: str, allow_meta_intro: bool) -> tuple[str, int, int]:
    if not text:
        return text, 0, 0

    replaced = _PROTOCOL_TAG_RE.sub(" ", text)
    tag_removed = 1 if replaced != text else 0
    lines = replaced.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    removed = tag_removed
    artifact_prefixes = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        artifact_prefix = _READER_ARTIFACT_PREFIX_RE.match(stripped)
        if artifact_prefix:
            artifact_prefixes += len(re.findall(r":", artifact_prefix.group("prefix")))
            content = artifact_prefix.group("content").strip()
            if content:
                kept.append(content)
            continue
        if _CODE_FENCE_RE.match(stripped):
            removed += 1
            continue
        if _INPUT_MARKER_RE.search(stripped):
            removed += 1
            continue
        if _PROMPT_HEADING_RE.match(stripped):
            removed += 1
            continue
        meta_with_content = _META_PREFIX_WITH_CONTENT_RE.match(stripped)
        if meta_with_content:
            content = meta_with_content.group("content").strip()
            if content:
                kept.append(content)
            removed += 1
            continue
        if _PROMPT_LINE_RE.match(stripped):
            if allow_meta_intro and index == 0:
                kept.append(line)
            else:
                removed += 1
            continue
        kept.append(line)

    value = "\n".join(kept)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"[ \t]+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip(), removed, artifact_prefixes


def _residual_protocol_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern, label in (
        (_PROTOCOL_TAG_RE, "wrapper_tag"),
        (_INPUT_MARKER_RE, "input_marker"),
        (_PROMPT_HEADING_RE, "prompt_heading"),
        (_PROMPT_LINE_RE, "prompt_meta_line"),
    ):
        if pattern.search(text or ""):
            hits.append(label)
    return hits


def _style_scores(text: str, reference: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    current = _style_vector(text)
    scores.update({f"style_{key}": value for key, value in current.items()})
    if reference and text:
        ref = _style_vector(reference)
        scores["style_drift"] = _vector_distance(current, ref)
    return scores


def _style_vector(text: str) -> dict[str, float]:
    words = max(1, _word_count(text))
    paragraphs = [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    sentences = max(1, len(_SENTENCE_RE.findall(text or "")))
    return {
        "avg_sentence_words": min(80.0, words / sentences),
        "paragraphs_per_1000_words": min(30.0, len(paragraphs) * 1000.0 / words),
        "dialogue_markers_per_1000_words": min(30.0, len(_DIALOGUE_RE.findall(text or "")) * 1000.0 / words),
        "semicolon_colon_per_1000_words": min(40.0, (text.count(";") + text.count(":")) * 1000.0 / words),
    }


def _vector_distance(left: dict[str, float], right: dict[str, float]) -> float:
    weights = {
        "avg_sentence_words": 80.0,
        "paragraphs_per_1000_words": 30.0,
        "dialogue_markers_per_1000_words": 30.0,
        "semicolon_colon_per_1000_words": 40.0,
    }
    total = 0.0
    for key, scale in weights.items():
        total += ((left.get(key, 0.0) - right.get(key, 0.0)) / scale) ** 2
    return min(1.0, math.sqrt(total / max(1, len(weights))))


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def merge_guard_issues(*groups: Iterable[CandidateIssue]) -> list[CandidateIssue]:
    merged: list[CandidateIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for issue in group or ():
            key = (issue.code, issue.severity, issue.detail)
            if key in seen:
                continue
            seen.add(key)
            merged.append(issue)
    return merged
