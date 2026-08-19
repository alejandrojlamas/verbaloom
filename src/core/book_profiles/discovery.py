"""Glossary discovery for book-scoped editorial profiles."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Optional

import yaml

from src.utils.json_extraction import extract_tagged_payload, loads_first_json_value

from .loader import load_book_profile
from .models import BookProfile, ProfileGlossaryEntry
from .profile_goals import ProfileGoalRules, resolve_profile_goal


GLOSSARY_DISCOVERY_TAG_IN = "<GLOSSARY_DISCOVERY_JSON>"
GLOSSARY_DISCOVERY_TAG_OUT = "</GLOSSARY_DISCOVERY_JSON>"

_WORD_RE = re.compile(r"\b[^\W\d_][\wáéíóúüñÁÉÍÓÚÜÑ'-]{3,}\b", re.UNICODE)
_CAPITALIZED_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ'’-]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ][a-záéíóúüñ'’-]{2,}){0,3}\b"
)


@dataclass(frozen=True)
class GlossarySuggestion:
    source: str
    suggested_target: str = ""
    target_options: tuple[str, ...] = ()
    suggestion_type: str = "lexical_archaism"
    scope: str = ""
    status: str = "pending"
    confidence: float = 0.0
    rationale: str = ""
    examples: tuple[dict[str, str], ...] = ()
    risks: tuple[str, ...] = ()
    do_not_apply_if: tuple[str, ...] = ()
    candidate_for_common_glossary: bool = False

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "suggested_target": self.suggested_target,
            "target_options": list(self.target_options),
            "type": self.suggestion_type,
            "scope": self.scope,
            "status": self.status,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "examples": list(self.examples),
            "risks": list(self.risks),
            "do_not_apply_if": list(self.do_not_apply_if),
        }
        if self.candidate_for_common_glossary:
            data["candidate_for_common_glossary"] = True
        return data


def build_glossary_discovery_prompt(
    source_text: str,
    *,
    profile: BookProfile,
    candidate_text: str = "",
    chunk_index: int = 0,
    chunk_total: int = 0,
    coverage_mode: str = "sampled",
    goal_rules: ProfileGoalRules | None = None,
) -> tuple[str, str]:
    rules = goal_rules or resolve_profile_goal(
        "",
        transform_mode=str(profile.raw_config.get("transform_mode") or ""),
        source_name=str(profile.raw_config.get("source_name") or ""),
        profile_config=profile.raw_config,
    )
    prompt = profile.prompt_texts.get("glossary_discovery") or _DEFAULT_DISCOVERY_PROMPT
    audiobook_hint = _audiobook_discovery_hint(profile)
    system = f"""{prompt}

{rules.prompt_brief()}

Return only valid JSON wrapped in {GLOSSARY_DISCOVERY_TAG_IN} and {GLOSSARY_DISCOVERY_TAG_OUT}.
Do not include markdown or commentary outside the tags.
Return only useful glossary entries. Do not copy paragraphs, sentences, or whole passages.
The "source" field must be a short term, name, treatment, formula, or reusable phrase found verbatim in this chunk.
Prefer high-value suggestions over noisy volume. The right count depends on the active business goal; return fewer items for ordinary chunks and more only when the chunk contains genuinely useful signals.
Return at most 24 suggestions for this chunk. Keep rationale, risks, and examples concise.
{audiobook_hint}

You may also include a compact "editorial_map" object with these optional arrays:
entities, voices, chapters, translatable_terms, preserve_terms, risks,
canonical_names, blockers, characters_entities, sections, narrative_voices,
relationships, technical_cultural_terms, iconic_phrases, do_not_translate,
editorial_risks.
Keep each item short. Never copy full paragraphs or long passages into the map.
Keep at most 8 items in each editorial_map array for this chunk. Omit empty arrays.

Use exactly this top-level contract:
{{
  "suggestions": [
    {{
      "source": "verbatim short source term",
      "suggested_target": "recommended target form or empty string",
      "target_options": [],
      "type": "proper_noun | technical_term | concept | idiom | address_form | phrase | syntax_pattern",
      "confidence": 0.0,
      "rationale": "short reason",
      "examples": [],
      "risks": [],
      "do_not_apply_if": []
    }}
  ],
  "editorial_map": {{
    "translatable_terms": [{{"source": "short term", "target": "target form", "reason": "short reason"}}],
    "entities": [{{"name": "canonical name", "type": "person | place | organization | work"}}],
    "risks": [{{"label": "short risk", "severity": "low | medium | high"}}]
  }}
}}
Every editorial_map array item must be a JSON object, never a bare string.
Use empty arrays when there are no useful suggestions. Do not invent signals."""
    user = f"""# ACTIVE PROFILE
{profile.profile_id}

# BUSINESS GOAL
{rules.key}: {rules.glossary_goal}

# DISCOVERY COVERAGE
mode: {coverage_mode}
chunk: {chunk_index or '?'} / {chunk_total or '?'}

# SOURCE TEXT
{source_text}

# CANDIDATE TEXT
{candidate_text or '(not provided)'}
"""
    return system.strip(), user.strip()


def _audiobook_discovery_hint(profile: BookProfile) -> str:
    config = profile.raw_config.get("audiobook") or profile.raw_config.get("audio_sanitization") or {}
    enabled = bool(config.get("enabled") or config.get("generate_companion")) if isinstance(config, Mapping) else bool(config)
    if not enabled:
        return ""
    return """
For this audiobook profile, prioritize signals that improve faithful listening:
recurring film/art/media terms, names and titles, informative caption patterns,
credit-only caption patterns, note/reference patterns, and terms that should be
translated rather than preserved in English. Keep suggestions short and useful.
""".strip()


def parse_glossary_discovery_payload(text: str) -> dict[str, Any]:
    if not text:
        return {"suggestions": [], "editorial_map": {}, "valid_schema": False}
    payload = (
        extract_tagged_payload(
            text,
            GLOSSARY_DISCOVERY_TAG_IN,
            GLOSSARY_DISCOVERY_TAG_OUT,
        )
        or text
    )
    parsed = loads_first_json_value(payload)
    if isinstance(parsed, list):
        suggestions = parsed
        editorial_map: Any = {}
        valid_schema = all(isinstance(item, Mapping) for item in parsed)
    elif isinstance(parsed, Mapping):
        suggestions = parsed.get("suggestions")
        editorial_map = parsed.get("editorial_map")
        direct_map = {
            key: parsed.get(key)
            for key in _DISCOVERY_EDITORIAL_KEYS
            if key in parsed
        }
        if not isinstance(editorial_map, Mapping) and direct_map:
            editorial_map = direct_map
        valid_schema = (
            ("suggestions" in parsed and isinstance(suggestions, list))
            or ("editorial_map" in parsed and isinstance(editorial_map, Mapping))
            or any(isinstance(value, list) for value in direct_map.values())
        )
    else:
        return {"suggestions": [], "editorial_map": {}, "valid_schema": False}

    return {
        "suggestions": [dict(item) for item in suggestions or [] if isinstance(item, Mapping)],
        "editorial_map": _normalise_discovery_editorial_map(editorial_map),
        "valid_schema": bool(valid_schema),
    }


_DISCOVERY_EDITORIAL_KEYS = {
    "entities",
    "voices",
    "chapters",
    "translatable_terms",
    "preserve_terms",
    "risks",
    "canonical_names",
    "blockers",
    "characters_entities",
    "sections",
    "narrative_voices",
    "relationships",
    "technical_cultural_terms",
    "iconic_phrases",
    "do_not_translate",
    "editorial_risks",
}

_DISCOVERY_MAP_LABEL_FIELDS = {
    "entities": "name",
    "characters_entities": "name",
    "canonical_names": "name",
    "voices": "label",
    "narrative_voices": "label",
    "chapters": "title",
    "sections": "title",
    "translatable_terms": "source",
    "preserve_terms": "source",
    "blockers": "source",
    "do_not_translate": "source",
    "technical_cultural_terms": "term",
    "iconic_phrases": "phrase",
    "risks": "label",
    "editorial_risks": "label",
    "relationships": "label",
}


def _normalise_discovery_editorial_map(value: Any) -> dict[str, list[dict[str, Any]]]:
    """Accept strict object items and bounded compact string items from LLMs."""
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for key, raw_items in value.items():
        if key not in _DISCOVERY_EDITORIAL_KEYS or not isinstance(raw_items, list):
            continue
        items: list[dict[str, Any]] = []
        for raw in raw_items[:8]:
            if isinstance(raw, Mapping):
                item = dict(raw)
            elif isinstance(raw, str):
                label = re.sub(r"\s+", " ", raw).strip()
                if not label or len(label) > 160:
                    continue
                item = {_DISCOVERY_MAP_LABEL_FIELDS[key]: label}
            else:
                continue
            if item:
                items.append(item)
        if items:
            normalized[key] = items
    return normalized


def parse_glossary_discovery_response(text: str) -> list[dict[str, Any]]:
    return parse_glossary_discovery_payload(text)["suggestions"]


def suggest_glossary_entries(
    source_text: str,
    *,
    profile_id: str,
    max_suggestions: int = 80,
) -> list[GlossarySuggestion]:
    profile = load_book_profile(profile_id)
    existing = {entry.source.casefold() for entry in profile.glossary_entries}
    suggestions: list[GlossarySuggestion] = []

    for entry in profile.pending_entries:
        if len(suggestions) >= max_suggestions:
            break
        if _contains(source_text, entry.source):
            suggestions.append(_suggestion_from_entry(entry, profile.profile_id))

    candidates = _generic_recurrent_candidates(source_text)
    for source, count in candidates:
        if len(suggestions) >= max_suggestions:
            break
        key = source.casefold()
        if key in existing or any(s.source.casefold() == key for s in suggestions):
            continue
        suggestions.append(GlossarySuggestion(
            source=source,
            suggested_target="",
            suggestion_type="syntax_pattern" if " " in source else "lexical_archaism",
            scope=profile.profile_id,
            status="pending",
            confidence=min(0.84, 0.42 + count / 50),
            rationale=(
                "Recurrent form detected in this source. It needs profile-specific "
                "editorial review before becoming an approved modernization rule."
            ),
            examples=({"source_excerpt": _example(source_text, source), "recommended_modernization": ""},),
            risks=("Meaning may vary by context; do not auto-approve.",),
        ))
    return suggestions


def merge_pending_suggestions(
    profile_id: str,
    suggestions: list[GlossarySuggestion | Mapping[str, Any]],
) -> Path:
    profile = load_book_profile(profile_id)
    path = profile.root / "glossary" / "pending_suggestions.yml"
    payload = _read_yaml_mapping(path)
    existing = [
        item for item in payload.get("suggestions") or payload.get("entries") or []
        if isinstance(item, Mapping)
    ]
    seen = {str(item.get("source") or "").casefold() for item in existing}
    for suggestion in suggestions:
        data = suggestion.to_dict() if isinstance(suggestion, GlossarySuggestion) else dict(suggestion)
        source = str(data.get("source") or "").strip()
        if not source or source.casefold() in seen:
            continue
        data["status"] = "pending"
        data.setdefault("scope", profile.profile_id)
        existing.append(data)
        seen.add(source.casefold())
    path.write_text(
        yaml.safe_dump({"suggestions": existing}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def _suggestion_from_entry(entry: ProfileGlossaryEntry, profile_id: str) -> GlossarySuggestion:
    return GlossarySuggestion(
        source=entry.source,
        suggested_target=entry.target,
        target_options=entry.target_options,
        suggestion_type=entry.entry_type,
        scope=profile_id,
        status="pending",
        confidence=entry.confidence,
        rationale=entry.rationale,
        examples=tuple(example.__dict__ for example in entry.examples),
        do_not_apply_if=entry.do_not_apply_if,
    )


def _generic_recurrent_candidates(text: str) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for match in _WORD_RE.finditer(text or ""):
        word = match.group(0)
        folded = word.casefold()
        if len(folded) < 5 or folded in _COMMON_WORDS:
            continue
        if _looks_editorially_interesting(word):
            counts[word] += 1
    for match in _CAPITALIZED_RE.finditer(text or ""):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(value.split()) >= 2:
            counts[value] += 1
    return sorted(
        [(source, count) for source, count in counts.items() if count >= 2],
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    )


def _looks_editorially_interesting(word: str) -> bool:
    folded = word.casefold()
    if "'" in word or "’" in word:
        return True
    if folded.endswith(("ades", "edes", "istes", "edeslo", "allo", "ello")):
        return True
    if re.search(r"(?:[bcdfghjklmnpqrstvwxyz]{4,})", folded):
        return True
    return False


def _contains(text: str, source: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", text or "", re.I))


def _example(text: str, source: str) -> str:
    match = re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", text or "", re.I)
    if not match:
        return ""
    start = max(0, match.start() - 70)
    end = min(len(text), match.end() + 70)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.safe_load(text) if text.strip() else {}
    return dict(data) if isinstance(data, Mapping) else {}


_COMMON_WORDS = {
    "aunque", "cuando", "donde", "entonces", "habia", "había", "porque",
    "señor", "señora", "todos", "todas", "tener", "hacer", "decir",
    "capitulo", "capítulo",
}


_DEFAULT_DISCOVERY_PROMPT = """You are a lexical and editorial discovery agent for a book-scoped intralingual modernization.

Analyze the source text and propose terms, address forms, formulas, turns of
phrase, names, false proper nouns, iconic phrases, sensitive terms, and syntax
patterns that may need entries in the ACTIVE PROFILE glossary.

Useful glossary entries are compact reusable signals, not copied prose. Focus on:
- recurring named people, peoples, places, institutions, gods, titles, and works
- culturally or historically loaded terms whose rendering must stay consistent
- address forms and formulas that require contextual treatment
- archaic, regional, OCR-risk, or orthographic forms that may affect style
- phrases that should be preserved or handled consistently because they carry voice

Also return a compact editorial_map when useful:
- entities: recurring actors, peoples, places, organizations, works
- voices: narrator, speaker, register, or recurring stylistic signals
- chapters: chapter/part/front-matter titles visible in this chunk
- translatable_terms: technical/cultural/common terms likely requiring target-language rendering
- preserve_terms: names, formulas, quotations, acronyms, or culturally fixed terms to preserve
- canonical_names: names with canonical spelling or target form decisions
- blockers: items that should not be approved/injected automatically until reviewed
- risks: OCR noise, tables, formulas, notes, censorship-sensitive material, structural ambiguity
- characters_entities: recurring actors, peoples, places, organizations, works
- sections: chapter/part/front-matter titles visible in this chunk
- narrative_voices: narrator or speaker/register signals
- relationships: recurring participant pairs or social relations
- technical_cultural_terms: domain, historical, cultural, ritual, legal, academic, or scientific terms
- iconic_phrases: short phrases that should be preserved or handled consistently
- do_not_translate: names, formulas, quotations, or terms that should remain as written
- editorial_risks: OCR noise, tables, formulas, notes, censorship-sensitive material, structural ambiguity

Do not emit:
- full sentences or paragraphs
- generic common words
- one-off prose that will not recur
- duplicate variants already present in the same response
- explanations as target text

Do not propose global rules. Do not auto-approve. Return suggestions as pending
unless the caller has explicitly enabled a separate approval workflow."""
