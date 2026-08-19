"""Language-scoped lexical policy for deterministic glossary heuristics.

The generic glossary engine owns algorithms and structural categories. Words
that only make sense for a particular source language live in the adjacent
JSON resource so they cannot silently affect books in another language.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

from src.core.language_evidence import normalize_language_name


_POLICY_PATH = Path(__file__).with_name("lexical_policies.json")
_AUTO_LANGUAGE_VALUES = {"", "auto", "autodetect", "automatic", "unknown"}


@dataclass(frozen=True)
class LexicalPolicy:
    language: str
    connectors: frozenset[str] = frozenset()
    leading_articles: frozenset[str] = frozenset()
    single_title_stopwords: frozenset[str] = frozenset()
    titlecase_translatable_words: frozenset[str] = frozenset()
    fragment_noise_words: frozenset[str] = frozenset()
    acronym_stopwords: frozenset[str] = frozenset()
    tech_nouns: frozenset[str] = frozenset()
    tech_modifiers: frozenset[str] = frozenset()
    high_value_tech_phrases: frozenset[str] = frozenset()
    translatable_single_words: frozenset[str] = frozenset()
    translatable_phrase_hints: frozenset[str] = frozenset()
    leading_noise_words: frozenset[str] = frozenset()
    single_word_reject_noise: frozenset[str] = frozenset()
    source_equals_target_translatable_words: frozenset[str] = frozenset()
    trailing_fragment_words: frozenset[str] = frozenset()
    uppercase_word_noise: frozenset[str] = frozenset()
    common_noun_capitalization: bool = False


_SET_FIELDS = {
    item.name
    for item in fields(LexicalPolicy)
    if item.name not in {"language", "common_noun_capitalization"}
}


@lru_cache(maxsize=1)
def _load_policy_data() -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"common": {}}
    if not isinstance(raw, Mapping):
        return {"common": {}}
    return {
        str(key).strip().casefold(): dict(value)
        for key, value in raw.items()
        if isinstance(value, Mapping)
    }


def _language_key(value: str) -> str:
    raw = str(value or "").strip().casefold()
    if raw in _AUTO_LANGUAGE_VALUES:
        return ""
    normalized = normalize_language_name(raw)
    if normalized:
        return normalized
    return raw


@lru_cache(maxsize=32)
def resolve_lexical_policy(
    source_language: str = "",
    *,
    default_language: str = "",
) -> LexicalPolicy:
    """Return common policy plus the active source-language policy.

    Unknown or explicit auto-detection values receive only language-neutral
    data. Callers that historically assumed English can opt into that behavior
    with ``default_language="english"`` while live profile preparation always
    passes the actual detected source language.
    """
    key = _language_key(source_language)
    if not key:
        key = _language_key(default_language)

    data = _load_policy_data()
    common = data.get("common", {})
    scoped = data.get(key, {}) if key else {}
    values: dict[str, Any] = {"language": key or "common"}
    for field_name in _SET_FIELDS:
        merged = [
            *(common.get(field_name) or []),
            *(scoped.get(field_name) or []),
        ]
        values[field_name] = frozenset(
            str(item).strip().casefold()
            for item in merged
            if str(item).strip()
        )
    values["common_noun_capitalization"] = bool(
        scoped.get("common_noun_capitalization", False)
    )
    return LexicalPolicy(**values)


def clear_lexical_policy_cache() -> None:
    """Test/admin hook for reloading edited policy resources."""
    _load_policy_data.cache_clear()
    resolve_lexical_policy.cache_clear()
