"""Compile profile knowledge into a small, chunk-scoped prompt context.

Book-profile preparation can produce thousands of useful signals.  Those
signals should behave like an editorial database, not like a prompt dump.  This
module selects the entries that actually matter for one chunk and ranks them by
editorial value so translation, transformation, audit, and repair prompts stay
small and actionable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Mapping

from src.core.editorial_knowledge import contains_term, fold_text, normalize_policy


_TARGET_SIDE_PURPOSES = {
    "refine",
    "refinement",
    "transform",
    "transformation",
    "modernize",
    "same-language",
    "profile_audit",
    "profile_repair",
}
_TRANSLATE_POLICIES = {
    "translate",
    "translate_exact",
    "translate_contextual",
    "translate_consistently",
    "canonical_translation",
    "canonical_name",
    "contextual",
}
_PRESERVE_POLICIES = {
    "preserve",
    "preserve_exact",
    "do_not_translate",
    "keep_source",
    "source_form",
}
_TRANSLATABLE_TYPES = {
    "caption",
    "caption_pattern",
    "concept",
    "formula_caption",
    "glossary",
    "idiom",
    "key_term",
    "lexical_archaism",
    "method",
    "note_pattern",
    "orthographic_variant",
    "phrase",
    "reference_pattern",
    "syntax_pattern",
    "technical",
    "technical_term",
    "term",
}
_PRESERVE_TYPES = {
    "acronym",
    "canonical_proper_noun",
    "character",
    "location",
    "organization",
    "proper_noun",
    "title",
}
_ARTIFACT_KEYS = (
    "canonical_names",
    "characters_entities",
    "entities",
    "relationships",
    "technical_cultural_terms",
    "translatable_terms",
    "preserve_terms",
    "iconic_phrases",
    "blockers",
    "risks",
    "editorial_risks",
)
_ARTIFACT_HINT_STOP_WORDS = {
    "all",
    "america",
    "american",
    "and",
    "at",
    "britain",
    "british",
    "city",
    "country",
    "england",
    "english",
    "europe",
    "european",
    "for",
    "france",
    "french",
    "german",
    "germany",
    "india",
    "indian",
    "in",
    "mexican",
    "mexico",
    "of",
    "or",
    "spain",
    "spanish",
    "state",
    "states",
    "the",
    "to",
    "united",
}
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)?")


@dataclass(frozen=True)
class CompiledProfilePromptContext:
    """Prompt-ready subset of profile knowledge for a single chunk."""

    entries: tuple[Any, ...] = ()
    artifact_hints: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    matched_total: int = 0
    prompt_budget: int = 0

    @property
    def capped(self) -> bool:
        return self.matched_total > len(self.entries)


def compile_profile_prompt_context(
    profile: Any,
    chunk_text: str,
    entries: Iterable[Any],
    *,
    purpose: str = "translation",
    max_entries: int = 48,
    max_artifact_hints: int = 8,
    excluded_artifact_labels: Iterable[str] = (),
) -> CompiledProfilePromptContext:
    """Rank profile glossary entries and local artifact hints for one chunk.

    Args:
        profile: Loaded ``BookProfile`` or compatible object.
        chunk_text: Source/candidate text used to select relevant knowledge.
        entries: Already allowed and phase-matched profile glossary entries.
        purpose: Prompt phase. Refinement/transformation may match target-side
            terms because source wording may already be gone.
        max_entries: Hard prompt budget for glossary entries.
        max_artifact_hints: Hard prompt budget for broader editorial-map hints.
        excluded_artifact_labels: Folded labels rejected by the prompt-safety
            layer. This prevents stale editorial maps from reintroducing a
            generated preserve rule that the active glossary already withheld.
    """

    text = chunk_text or ""
    normalized_purpose = str(purpose or "translation").strip().lower() or "translation"
    scored: list[tuple[tuple[float, ...], Any]] = []
    warnings: list[str] = []
    source_equals_target = 0
    suspicious_preserve = 0

    for entry in entries or ():
        if not _source(entry):
            continue
        if _source_equals_target(entry):
            source_equals_target += 1
            if _looks_like_translatable_preserve(entry):
                suspicious_preserve += 1
                # Do not inject low-signal "preserve exact" entries that look
                # like ordinary terms.  They bias the model toward leaving the
                # source language untouched, which is the failure mode this
                # compiler is meant to prevent.
                continue
        scored.append((_entry_score(entry, normalized_purpose), entry))

    scored.sort(key=lambda item: item[0], reverse=True)
    budget = max(1, int(max_entries or 48))
    selected = tuple(entry for _score, entry in scored[:budget])

    if source_equals_target:
        warnings.append(
            f"{source_equals_target} source-equals-target entries matched; treat only explicit preserve decisions as binding."
        )
    if suspicious_preserve:
        warnings.append(
            f"{suspicious_preserve} low-signal preserve entries were withheld from this prompt."
        )
    if len(scored) > budget:
        warnings.append(
            f"{len(scored) - budget} matched profile entries were omitted to protect the prompt budget."
        )

    artifact_hints = _artifact_hints_for_chunk(
        getattr(profile, "editorial_artifacts", {}) or {},
        text,
        purpose=normalized_purpose,
        limit=max_artifact_hints,
        excluded_labels={fold_text(item) for item in excluded_artifact_labels if str(item).strip()},
    )
    return CompiledProfilePromptContext(
        entries=selected,
        artifact_hints=tuple(artifact_hints),
        warnings=tuple(warnings),
        matched_total=len(scored),
        prompt_budget=budget,
    )


def match_profile_entry_in_text(
    entry: Any,
    text: str,
    *,
    purpose: str = "translation",
) -> bool:
    """Return whether a profile entry is relevant to the current chunk."""

    if not text or not _source(entry):
        return False
    if contains_term(text, _source(entry)):
        return True
    if str(purpose or "").strip().lower() not in _TARGET_SIDE_PURPOSES:
        return False
    target = _target(entry)
    if target and contains_term(text, target):
        return True
    for option in _target_options(entry):
        if contains_term(text, option):
            return True
    return False


def _entry_score(entry: Any, purpose: str) -> tuple[float, ...]:
    policy = normalize_policy(_policy(entry))
    source = _source(entry)
    target = _target(entry)
    entry_type = normalize_policy(getattr(entry, "entry_type", "") or "term")
    words = _WORD_RE.findall(source)
    word_count = len(words)
    confidence = max(
        _float(getattr(entry, "confidence", 0.0)),
        _float(getattr(entry, "review_confidence", 0.0)),
    )
    occurrences = max(0, _int(getattr(entry, "occurrences", 0)))
    translated = bool(target and fold_text(target) != fold_text(source))
    preserve = policy in _PRESERVE_POLICIES or _source_equals_target(entry)
    decision_heavy = bool(
        getattr(entry, "decision_rule", "")
        or getattr(entry, "forbidden_default", "")
        or _target_options(entry)
    )

    if translated and policy in {"translate_exact", "canonical_translation", "canonical_name"}:
        tier = 100.0
    elif translated:
        tier = 90.0
    elif decision_heavy:
        tier = 82.0
    elif preserve and entry_type in _PRESERVE_TYPES and word_count >= 2:
        tier = 72.0
    elif preserve and entry_type in _PRESERVE_TYPES:
        tier = 58.0
    elif purpose in _TARGET_SIDE_PURPOSES and target:
        tier = 46.0
    else:
        tier = 20.0

    if _looks_like_translatable_preserve(entry):
        tier -= 45.0
    if _has_useful_notes(entry):
        tier += 5.0
    if normalize_policy(getattr(entry, "source_file", "")) in {"editorial_map", "derived_prompt_alias"}:
        tier -= 4.0

    return (
        tier,
        min(math.log1p(occurrences), 8.0),
        min(word_count, 10),
        confidence,
        min(len(source), 120) / 120.0,
        -1.0 * _source_noise_penalty(source),
    )


def _artifact_hints_for_chunk(
    artifacts: Mapping[str, Any],
    chunk_text: str,
    *,
    purpose: str,
    limit: int,
    excluded_labels: set[str] | None = None,
) -> list[str]:
    if not artifacts or not chunk_text:
        return []
    hints: list[tuple[tuple[float, ...], str]] = []
    seen: set[str] = set()
    excluded = excluded_labels or set()
    for key in _ARTIFACT_KEYS:
        values = artifacts.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, Mapping):
                continue
            label = _artifact_label(item)
            if (
                not label
                or fold_text(label) in excluded
                or not _artifact_label_allowed(label, item)
                or not contains_term(chunk_text, label)
            ):
                continue
            folded = fold_text(f"{key}:{label}")
            if folded in seen:
                continue
            seen.add(folded)
            score = _artifact_score(key, item)
            hints.append((score, _render_artifact_hint(key, item, label, purpose)))
    hints.sort(key=lambda item: item[0], reverse=True)
    return [hint for _score, hint in hints[: max(0, int(limit or 0))]]


def _artifact_score(key: str, item: Mapping[str, Any]) -> tuple[float, float, float]:
    key_weight = {
        "canonical_names": 100.0,
        "translatable_terms": 94.0,
        "technical_cultural_terms": 90.0,
        "characters_entities": 84.0,
        "entities": 80.0,
        "relationships": 76.0,
        "blockers": 72.0,
        "risks": 66.0,
        "editorial_risks": 66.0,
        "preserve_terms": 60.0,
        "iconic_phrases": 58.0,
    }.get(key, 40.0)
    return (
        key_weight,
        min(math.log1p(_int(item.get("occurrences"))), 8.0),
        _float(item.get("confidence")),
    )


def _render_artifact_hint(key: str, item: Mapping[str, Any], label: str, purpose: str) -> str:
    target = str(item.get("target") or item.get("canonical") or item.get("suggested_target") or "").strip()
    kind = str(item.get("type") or item.get("category") or key).strip()
    reason = str(item.get("reason") or item.get("rationale") or item.get("notes") or "").strip()
    if key in {"translatable_terms", "technical_cultural_terms"} and target and fold_text(target) != fold_text(label):
        head = f"{label} -> {target}"
    elif key in {"canonical_names", "characters_entities", "entities", "preserve_terms"}:
        head = f"{label}: keep/canonicalize according to the active profile context"
    else:
        head = f"{label}: {key.replace('_', ' ')}"
    if kind:
        head += f" [{kind}]"
    if reason:
        return f"- {head}. {reason[:180]}"
    return f"- {head}."


def _artifact_label(item: Mapping[str, Any]) -> str:
    for key in ("source", "name", "term", "phrase", "title", "canonical"):
        value = str(item.get(key) or "").strip()
        if value:
            return value
    return ""


def _artifact_label_allowed(label: str, item: Mapping[str, Any]) -> bool:
    words = _WORD_RE.findall(label or "")
    if not words or len(words) > 5:
        return False
    folded_words = {fold_text(word) for word in words}
    if folded_words & _ARTIFACT_HINT_STOP_WORDS:
        return False
    confidence = _float(item.get("confidence"))
    occurrences = _int(item.get("occurrences"))
    if confidence and confidence < 0.75:
        return False
    if occurrences and occurrences < 2:
        return False
    kind = normalize_policy(str(item.get("type") or item.get("category") or ""))
    if kind and kind not in (_PRESERVE_TYPES | _TRANSLATABLE_TYPES | {"relationship", "risk"}):
        return False
    return True


def _looks_like_translatable_preserve(entry: Any) -> bool:
    if not _source_equals_target(entry):
        return False
    entry_type = normalize_policy(getattr(entry, "entry_type", "") or "")
    if entry_type in _TRANSLATABLE_TYPES:
        return True
    source = _source(entry)
    words = _WORD_RE.findall(source)
    if len(words) >= 4:
        return True
    if words and all(word[:1].islower() for word in words):
        return True
    return False


def _source_equals_target(entry: Any) -> bool:
    source = _source(entry)
    target = _target(entry)
    return bool(source and target and fold_text(source) == fold_text(target))


def _has_useful_notes(entry: Any) -> bool:
    return bool(
        str(getattr(entry, "decision_rule", "") or "").strip()
        or str(getattr(entry, "rationale", "") or "").strip()
        or str(getattr(entry, "review_rationale", "") or "").strip()
        or getattr(entry, "examples", None)
    )


def _source_noise_penalty(source: str) -> float:
    words = _WORD_RE.findall(source or "")
    if not words:
        return 4.0
    penalty = 0.0
    if len(words) > 8:
        penalty += 4.0
    if any(len(word) == 1 for word in words):
        penalty += 1.0
    if source.count(",") + source.count(";") >= 2:
        penalty += 2.0
    return penalty


def _policy(entry: Any) -> str:
    return str(
        getattr(entry, "translation_policy", "")
        or getattr(entry, "injection_policy", "")
        or getattr(entry, "review_status", "")
        or ""
    )


def _source(entry: Any) -> str:
    return str(getattr(entry, "source", "") or "").strip()


def _target(entry: Any) -> str:
    return str(getattr(entry, "target", "") or getattr(entry, "render_target", "") or "").strip()


def _target_options(entry: Any) -> tuple[str, ...]:
    raw = getattr(entry, "target_options", ()) or ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item).strip() for item in raw if str(item).strip())


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
