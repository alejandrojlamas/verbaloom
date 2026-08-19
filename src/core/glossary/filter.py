"""
Per-chunk glossary filtering.

Latin terms are matched with word boundaries (so "Fan" does not match "Fantasy").
CJK terms are matched as substrings (no word boundary concept in CJK scripts).
The filter returns only the subset of glossary entries that actually appear in
the chunk, sorted by source-term length (longest first) to handle overlaps.

When the per-chunk cap is hit, the kept subset is selected by occurrence count
(most frequent first, length as tiebreaker), then re-sorted by length for
output stability.

Source terms may declare alternative forms separated by '|' to handle inflected
languages (e.g. "Москва|Москве|Москвы|Москвой -> Moscou"). The filter matches
the entry if ANY of the alternatives appears in the chunk; occurrence counts
are summed across alternatives.

Matching is exact by default. Callers may opt into case-insensitive and
accent-insensitive matching through GlossaryConfig.
"""
import re
import unicodedata
from typing import Dict, List, Tuple

from src.core.glossary.models import GlossaryConfig

_CJK_RE = re.compile(r'[぀-ゟ゠-ヿ一-鿿가-힯㐀-䶿]')
_TARGET_SIDE_PURPOSES = {
    "refinement",
    "transform",
    "transformation",
    "modernize",
    "same-language",
}
_LATIN_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))


def _has_word_char_at_edge(term: str) -> bool:
    """True if the term starts or ends with a regex \\w character (Latin/digit/underscore)."""
    if not term:
        return False
    return bool(re.match(r'\w', term[0])) or bool(re.match(r'\w', term[-1]))


def _split_alternatives(source: str) -> List[str]:
    """Split a source term on '|' into non-empty stripped alternatives."""
    if "|" not in source:
        stripped = source.strip()
        return [stripped] if stripped else []
    return [a.strip() for a in source.split("|") if a.strip()]


def _max_alt_length(source: str) -> int:
    """Length used for sort: the longest alternative wins (overlap handling)."""
    alts = _split_alternatives(source)
    return max((len(a) for a in alts), default=0)


def _target_side_alternatives(target: str) -> List[str]:
    """Return conservative target-side variants for refinement/transform matching.

    Translation still scans source terms only. For refinement and same-language
    transforms, the current text may already contain an inflected target form
    such as "británica" for the glossary rendering "británico". These variants
    are matching triggers only; prompt rendering still shows the canonical
    right-hand form stored in the glossary.
    """
    target = (target or "").strip()
    if not target:
        return []

    variants = [target]
    words = _LATIN_WORD_RE.findall(target)
    if len(words) != 1:
        return variants

    word = words[0]
    prefix, suffix = target[: target.find(word)], target[target.find(word) + len(word):]
    lower = word.casefold()

    word_variants = {word}
    if lower.endswith(("o", "ó")) and len(word) > 2:
        stem = word[:-1]
        word_variants.update({stem + "a", stem + "os", stem + "as"})
    elif lower.endswith(("a", "e", "i", "í", "u", "ú")) and len(word) > 2:
        word_variants.add(word + "s")
    elif len(word) > 3 and not lower.endswith("s"):
        word_variants.add(word + "es")

    variants.extend(prefix + item + suffix for item in sorted(word_variants) if item != word)
    return variants


def _strip_diacritics(text: str) -> str:
    """Remove combining diacritics for accent-insensitive matching."""
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(char)
    )


def _normalize_for_match(text: str, config: GlossaryConfig) -> str:
    if config.accent_insensitive:
        text = _strip_diacritics(text)
    if not config.case_sensitive:
        text = text.casefold()
    return text


def _count_alternative(alt: str, haystack: str, config: GlossaryConfig) -> int:
    """Count occurrences of a single alternative form in the chunk."""
    needle = _normalize_for_match(alt, config)
    if not needle:
        return 0
    if _is_cjk(alt) or not _has_word_char_at_edge(needle):
        return haystack.count(needle)
    pattern = r'\b' + re.escape(needle) + r'\b'
    return len(re.findall(pattern, haystack))


def filter_glossary(
    chunk: str,
    glossary_terms: Dict[str, str],
    config: GlossaryConfig = None,
) -> Tuple[Dict[str, str], bool]:
    """
    Return only the glossary entries that appear in the chunk.

    Args:
        chunk: The source text to scan.
        glossary_terms: {source_term: translated_term}. A source_term may
            declare alternative inflected forms separated by '|'.
        config: GlossaryConfig (max_entries cap, case/accent sensitivity).

    Returns:
        (filtered_terms, capped) where filtered_terms preserves order
        (longest source first) and capped is True if the cap was hit.
    """
    if not chunk or not glossary_terms:
        return {}, False

    config = config or GlossaryConfig()

    # Sort by longest alternative descending so longer terms (e.g.
    # "Li Fanqing") are checked before shorter prefixes (e.g. "Li Fan").
    sorted_terms = sorted(
        glossary_terms.items(),
        key=lambda kv: _max_alt_length(kv[0]),
        reverse=True,
    )

    matched: List[Tuple[str, str, int]] = []  # (source, target, occurrence_count)
    haystack = _normalize_for_match(chunk, config)

    for source, target in sorted_terms:
        alternatives = _split_alternatives(source)
        if not alternatives:
            continue

        total_count = sum(
            _count_alternative(alt, haystack, config)
            for alt in alternatives
        )
        if total_count > 0:
            matched.append((source, target, total_count))

    capped = False
    if config.max_entries and len(matched) > config.max_entries:
        capped = True
        # When capping, keep the most frequent terms first (length as
        # tiebreaker so longer-and-rarer beats shorter-and-rarer). This
        # is more useful than the previous length-only cut, which could
        # drop a high-frequency 2-char CJK name in favor of 50 longer
        # but rarer entries.
        kept = set(
            (s, t) for s, t, _ in
            sorted(matched, key=lambda x: (x[2], _max_alt_length(x[0])), reverse=True)[: config.max_entries]
        )
        # Preserve the original length-desc order in the output so the
        # rendered block stays predictable for the LLM.
        matched = [(s, t, c) for s, t, c in matched if (s, t) in kept]

    return {s: t for s, t, _ in matched}, capped


def filter_glossary_for_purpose(
    chunk: str,
    glossary_terms: Dict[str, str],
    config: GlossaryConfig = None,
    purpose: str = "translation",
) -> Tuple[Dict[str, str], bool]:
    """Filter glossary terms using the matching semantics for a prompt phase.

    Translation scans the source side only. Refinement and same-language
    transformation also include a term when the current text already contains
    the right-hand form, because those phases often operate on text where the
    source term has already been translated or canonicalized.
    """
    config = config or GlossaryConfig()
    filtered, capped = filter_glossary(chunk, glossary_terms, config)

    normalized_purpose = (purpose or "translation").strip().lower()
    if normalized_purpose not in _TARGET_SIDE_PURPOSES:
        return filtered, capped

    target_scan_to_sources: Dict[str, List[str]] = {}
    source_to_target_scan: Dict[str, str] = {}
    for source, target in (glossary_terms or {}).items():
        clean_source = (source or "").strip()
        clean_target = (target or "").strip()
        if clean_source and clean_target:
            target_scan = "|".join(_target_side_alternatives(clean_target))
            if target_scan:
                target_scan_to_sources.setdefault(target_scan, []).append(source)
                source_to_target_scan[source] = target_scan
    if not target_scan_to_sources:
        return filtered, capped

    target_scan_terms = {target_scan: target_scan for target_scan in target_scan_to_sources}
    target_filtered, target_capped = filter_glossary(
        chunk,
        target_scan_terms,
        config,
    )
    capped = capped or target_capped
    if not target_filtered:
        return filtered, capped

    merged = dict(filtered)
    matched_target_scans = set(target_filtered)
    for source, target in (glossary_terms or {}).items():
        if source_to_target_scan.get(source) in matched_target_scans and source not in merged:
            merged[source] = target

    if config.max_entries and len(merged) > config.max_entries:
        capped = True
        merged = dict(list(merged.items())[: config.max_entries])

    return merged, capped
