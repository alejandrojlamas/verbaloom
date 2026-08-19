"""
Local glossary candidate extraction.

This module is deliberately deterministic and token-free. It scans the full
available book text (capped by the caller for memory) and proposes recurring
names, acronyms, identifiers, and domain phrases for review before they enter
the glossary.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.glossary.lexical_policy import LexicalPolicy, resolve_lexical_policy
from src.utils.proper_names import SYMBOL_BEARING_NAME_RE


_TOKEN_RE = re.compile(r"(?!\d)[^\W_][\w'-]*", re.UNICODE)
_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:[-_][A-Z0-9]+)*\b")
_IDENTIFIER_RE = re.compile(
    r"\b(?=[A-Za-z][A-Za-z0-9_'-]*\b)(?=[A-Za-z0-9_'-]*[_0-9])"
    r"[A-Za-z][A-Za-z0-9_'-]{1,}\b"
)
_CAMEL_RE = re.compile(r"\b[a-z]+(?:[A-Z][A-Za-z0-9]+)+\b")
_HYPHEN_RE = re.compile(r"\b[A-Za-z]{2,}(?:-[A-Za-z0-9]{2,})+\b")
_MAX_CONTEXTS = 3
_MAX_EXTRACTED_TERMS = 5000


@dataclass
class AutoGlossaryCandidate:
    """A candidate term proposed by the local extractor."""

    source: str
    target: str = ""
    category: str = "other"
    occurrences: int = 0
    score: float = 0.0
    confidence: float = 0.0
    extraction_method: str = "local"
    keep_source: bool = False
    needs_translation: bool = False
    contexts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "category": self.category,
            "occurrences": self.occurrences,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "extraction_method": self.extraction_method,
            "keep_source": self.keep_source,
            "needs_translation": self.needs_translation,
            "contexts": self.contexts,
        }


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _strip_wrapping_punct(value: str) -> str:
    return value.strip(" \t\r\n.,;:!?()[]{}<>\"'")


def _canonical_key(value: str, policy: LexicalPolicy) -> str:
    value = _normalize_space(_strip_wrapping_punct(value)).replace(" - ", "-")
    parts = value.split()
    while parts and parts[0].casefold() in policy.leading_articles:
        parts.pop(0)
    return " ".join(parts).casefold()


def _display_source(value: str, policy: LexicalPolicy) -> str:
    value = _normalize_space(_strip_wrapping_punct(value))
    parts = value.split()
    while len(parts) > 1 and parts[0].casefold() in policy.leading_articles:
        parts.pop(0)
    return " ".join(parts)


def _is_title_token(token: str, policy: LexicalPolicy) -> bool:
    if not token or token.casefold() in policy.connectors:
        return False
    first = token[0]
    if not first.isalpha() or not first.isupper():
        return False
    letters = [c for c in token if c.isalpha()]
    if len(letters) < 2:
        return False
    # All-caps belongs to the acronym pass, not title-name extraction.
    return not token.isupper()


def _context(text: str, start: int, end: int, radius: int = 80) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = _normalize_space(text[left:right])
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet += "..."
    return snippet


def _count_pattern(
    text: str,
    pattern: re.Pattern,
    category: str,
    *,
    policy: LexicalPolicy,
    min_length: int = 2,
) -> Dict[str, Tuple[int, List[str]]]:
    counts: Counter[str] = Counter()
    contexts: Dict[str, List[str]] = defaultdict(list)
    for match in pattern.finditer(text):
        raw = _display_source(match.group(0), policy)
        if category == "concept":
            raw = raw.lower()
        if len(raw) < min_length:
            continue
        if category == "acronym" and raw.casefold() in policy.acronym_stopwords:
            continue
        counts[raw] += 1
        if len(contexts[raw]) < _MAX_CONTEXTS:
            contexts[raw].append(_context(text, match.start(), match.end()))
    return {term: (count, contexts[term]) for term, count in counts.items()}


def _title_sequence_counts(
    text: str,
    policy: LexicalPolicy,
) -> Dict[str, Tuple[int, List[str]]]:
    tokens = [
        (m.group(0), m.start(), m.end())
        for m in _TOKEN_RE.finditer(text)
    ]
    counts: Counter[str] = Counter()
    spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    i = 0
    while i < len(tokens):
        tok, start, _end = tokens[i]
        if not _is_title_token(tok, policy):
            i += 1
            continue

        parts = [tok]
        last_end = tokens[i][2]
        j = i + 1
        title_tokens = 1
        while j < len(tokens) and len(parts) < 7:
            next_tok, next_start, next_end = tokens[j]
            gap = text[last_end:next_start]
            # An internal symbol can be part of a fictional or domain-specific
            # proper name (for example ``Ab*Cd``). Do not silently normalize it
            # into a space-separated title sequence; the dedicated extractor
            # below preserves the exact spelling.
            if "\n\n" in gap or len(gap) > 4 or re.search(r"[*\\]", gap):
                break
            if _is_title_token(next_tok, policy):
                parts.append(next_tok)
                title_tokens += 1
                last_end = next_end
                j += 1
                continue
            if (
                next_tok.casefold() in policy.connectors
                and j + 1 < len(tokens)
                and _is_title_token(tokens[j + 1][0], policy)
            ):
                parts.append(next_tok)
                last_end = next_end
                j += 1
                continue
            break

        source = _display_source(" ".join(parts), policy)
        if source:
            words = source.split()
            is_single = len(words) == 1
            if not (is_single and source.casefold() in policy.single_title_stopwords):
                # Single title-case words are noisy, so require stronger
                # evidence unless they look like named technical constructs.
                if not is_single or title_tokens > 1 or len(source) >= 5:
                    counts[source] += 1
                    if len(spans[source]) < _MAX_CONTEXTS:
                        spans[source].append((start, last_end))
        i += 1

    return {
        term: (count, [_context(text, start, end) for start, end in spans[term]])
        for term, count in counts.items()
    }


def _term_category(
    source: str,
    seed_category: str,
    policy: LexicalPolicy,
) -> str:
    if seed_category:
        return seed_category
    lowered = source.casefold()
    if _looks_like_fragment_noise(source, policy):
        return "other"
    if len(source.split()) == 1 and lowered in policy.titlecase_translatable_words:
        return "term"
    if source.isupper() and len(source) > 1:
        return "acronym"
    if SYMBOL_BEARING_NAME_RE.fullmatch(source):
        return "character"
    if "_" in source or any(ch.isdigit() for ch in source):
        return "technical"
    if "-" in source:
        return "technical"
    words = source.split()
    if len(words) > 1 and all(
        w[:1].isupper() or w.casefold() in policy.connectors
        for w in words
    ):
        if any(noun in lowered for noun in policy.tech_nouns):
            return "concept"
        return "character"
    if source[:1].isupper():
        if lowered in policy.tech_nouns:
            return "technical"
        return "character"
    return "technical"


def _target_policy(
    source: str,
    category: str,
    policy: LexicalPolicy,
) -> Tuple[str, bool, bool]:
    """Return (target, keep_source, needs_translation)."""
    if _looks_like_fragment_noise(source, policy):
        return "", False, True
    if (
        len(source.split()) == 1
        and source.casefold() in policy.titlecase_translatable_words
    ):
        return "", False, True
    if category in {"character", "location", "organization", "title", "item", "acronym"}:
        return source, True, False
    if category == "technical" and (
        source.isupper()
        or "_" in source
        or any(ch.isdigit() for ch in source)
        or _CAMEL_RE.search(source)
    ):
        return source, True, False
    return "", False, True


def _score(category: str, occurrences: int, source: str) -> float:
    weights = {
        "character": 1.25,
        "organization": 1.2,
        "location": 1.15,
        "acronym": 1.35,
        "technical": 1.1,
        "concept": 1.0,
        "title": 1.1,
        "item": 1.0,
    }
    length_bonus = min(len(source), 40) / 80.0
    return occurrences * weights.get(category, 1.0) + length_bonus


def _confidence(score: float, occurrences: int, category: str) -> float:
    base = 1.0 - math.exp(-max(occurrences, 0) / 4.0)
    if category in {"acronym", "technical"}:
        base += 0.08
    return max(0.15, min(0.98, base + min(score, 10) / 80.0))


def _merge_candidate(
    bucket: Dict[str, AutoGlossaryCandidate],
    source: str,
    occurrences: int,
    contexts: Sequence[str],
    policy: LexicalPolicy,
    seed_category: str = "",
) -> None:
    source = _display_source(source, policy)
    if not source or len(source) < 2:
        return
    if _looks_like_fragment_noise(source, policy):
        return
    key = _canonical_key(source, policy)
    if not key:
        return

    category = _term_category(source, seed_category, policy)
    target, keep_source, needs_translation = _target_policy(source, category, policy)
    score = _score(category, occurrences, source)
    candidate = AutoGlossaryCandidate(
        source=source,
        target=target,
        category=category,
        occurrences=occurrences,
        score=score,
        confidence=_confidence(score, occurrences, category),
        keep_source=keep_source,
        needs_translation=needs_translation,
        contexts=list(contexts[:_MAX_CONTEXTS]),
    )

    existing = bucket.get(key)
    if not existing or (candidate.score, len(candidate.source)) > (existing.score, len(existing.source)):
        bucket[key] = candidate


def extract_glossary_candidates(
    text: str,
    *,
    max_terms: int = 120,
    min_occurrences: int = 2,
    existing_sources: Optional[Iterable[str]] = None,
    source_language: str = "English",
) -> Tuple[List[Dict], List[str]]:
    """Extract token-free glossary candidates from full document text.

    Returns ``(candidate_dicts, warnings)``. Candidate dicts mirror the LLM
    NER endpoint shape and add occurrence/confidence metadata for review.
    """
    text = text or ""
    max_terms = max(1, min(int(max_terms or 120), _MAX_EXTRACTED_TERMS))
    min_occurrences = max(1, min(int(min_occurrences or 2), 20))
    warnings: List[str] = []

    if len(text.strip()) < 200:
        return [], ["Text is too short for reliable automatic glossary extraction."]

    policy = resolve_lexical_policy(source_language)
    bucket: Dict[str, AutoGlossaryCandidate] = {}

    sources = [
        (
            _count_pattern(
                text,
                SYMBOL_BEARING_NAME_RE,
                "character",
                policy=policy,
                min_length=4,
            ),
            "character",
        ),
        (_title_sequence_counts(text, policy), ""),
        (
            _count_pattern(
                text,
                _ACRONYM_RE,
                "acronym",
                policy=policy,
                min_length=2,
            ),
            "acronym",
        ),
        (
            _count_pattern(
                text,
                _IDENTIFIER_RE,
                "technical",
                policy=policy,
                min_length=3,
            ),
            "technical",
        ),
        (
            _count_pattern(
                text,
                _CAMEL_RE,
                "technical",
                policy=policy,
                min_length=4,
            ),
            "technical",
        ),
        (
            _count_pattern(
                text,
                _HYPHEN_RE,
                "technical",
                policy=policy,
                min_length=5,
            ),
            "technical",
        ),
    ]
    high_value_pattern = _literal_phrase_pattern(policy.high_value_tech_phrases)
    if high_value_pattern is not None:
        sources.append(
            (
                _count_pattern(
                    text,
                    high_value_pattern,
                    "concept",
                    policy=policy,
                    min_length=7,
                ),
                "concept",
            )
        )
    tech_pattern = _technical_phrase_pattern(policy)
    if tech_pattern is not None:
        sources.append(
            (
                _count_pattern(
                    text,
                    tech_pattern,
                    "concept",
                    policy=policy,
                    min_length=7,
                ),
                "concept",
            )
        )

    for found, seed_category in sources:
        for source, (occurrences, contexts) in found.items():
            if occurrences < min_occurrences:
                continue
            _merge_candidate(
                bucket,
                source,
                occurrences,
                contexts,
                policy,
                seed_category,
            )

    candidates = list(bucket.values())
    existing = {
        _canonical_key(x, policy)
        for x in (existing_sources or [])
        if x
    }
    for candidate in candidates:
        candidate_dict_key = _canonical_key(candidate.source, policy)
        setattr(candidate, "already_in_glossary", candidate_dict_key in existing)

    candidates.sort(key=lambda c: (c.score, c.occurrences, len(c.source)), reverse=True)
    if len(candidates) > max_terms:
        warnings.append(
            f"Local extractor found {len(candidates)} candidates; showing the top {max_terms}."
        )
        candidates = candidates[:max_terms]

    out: List[Dict] = []
    for c in candidates:
        data = c.to_dict()
        data["already_in_glossary"] = bool(getattr(c, "already_in_glossary", False))
        out.append(data)

    if not out:
        warnings.append(
            "No recurring terms met the local frequency threshold. Lower the minimum occurrences or use the LLM sample mode."
        )
    return out, warnings


def _source_words(value: str) -> list[str]:
    return [
        word.strip(" \t\r\n.,;:!?()[]{}<>\"'“”‘’").replace("’", "'").casefold()
        for word in re.split(r"\s+", str(value or "").strip())
        if word.strip(" \t\r\n.,;:!?()[]{}<>\"'“”‘’")
    ]


def _looks_like_fragment_noise(
    source: str,
    policy: LexicalPolicy,
) -> bool:
    words = _source_words(source)
    if len(words) < 2:
        return False
    if words[0] in policy.single_title_stopwords:
        return True
    if words[-1] in policy.fragment_noise_words:
        return True
    return sum(1 for word in words if word in policy.fragment_noise_words) >= 2


def _literal_phrase_pattern(phrases: frozenset[str]) -> re.Pattern[str] | None:
    if not phrases:
        return None
    alternatives = [
        re.escape(phrase).replace(r"\ ", r"\s+")
        for phrase in sorted(phrases, key=len, reverse=True)
    ]
    return re.compile(r"\b(?:" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def _technical_phrase_pattern(
    policy: LexicalPolicy,
) -> re.Pattern[str] | None:
    if not policy.tech_modifiers or not policy.tech_nouns:
        return None
    modifiers = "|".join(
        re.escape(item)
        for item in sorted(policy.tech_modifiers, key=len, reverse=True)
    )
    nouns = "|".join(
        re.escape(item)
        for item in sorted(policy.tech_nouns, key=len, reverse=True)
    )
    return re.compile(
        rf"\b(?:{modifiers})(?:[-\s]+[^\W\d_][\w-]{{2,}}){{0,3}}"
        rf"[-\s]+(?:{nouns})\b",
        re.IGNORECASE,
    )
