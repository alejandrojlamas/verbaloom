"""Deterministic helpers for proper names with meaningful internal symbols."""

from __future__ import annotations

from collections import Counter, defaultdict
import re


SYMBOL_BEARING_NAME_RE = re.compile(
    r"(?<![\w])"
    r"[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,24}"
    r"(?:[*\\][A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{1,24})+"
    r"(?![\w])"
)


def extract_symbol_bearing_names(text: str) -> Counter[str]:
    """Return exact symbol-bearing names without normalizing punctuation."""
    return Counter(SYMBOL_BEARING_NAME_RE.findall(text or ""))


def missing_symbol_bearing_names(source_text: str, candidate_text: str) -> Counter[str]:
    """Return genuinely missing symbol-bearing names after OCR reconciliation.

    Exact spellings are consumed first. Remaining occurrences may then match a
    visually equivalent OCR spelling, currently a terminal uppercase ``I``
    versus lowercase ``l`` inside a name segment. The meaningful separators
    remain part of the key, so dropping ``*`` or ``\\`` is still a hard loss.
    """
    source_names = extract_symbol_bearing_names(source_text)
    candidate_names = extract_symbol_bearing_names(candidate_text)
    missing = source_names - candidate_names
    available = candidate_names - source_names
    if not missing or not available:
        return missing

    available_by_key: Counter[str] = Counter()
    for spelling, count in available.items():
        available_by_key[_ocr_fidelity_key(spelling)] += count

    unresolved: Counter[str] = Counter()
    for spelling, count in missing.items():
        key = _ocr_fidelity_key(spelling)
        reconciled = min(count, available_by_key[key])
        available_by_key[key] -= reconciled
        if count > reconciled:
            unresolved[spelling] = count - reconciled
    return unresolved


def restore_symbol_bearing_names(source_text: str, candidate_text: str) -> str:
    """Restore dropped internal symbols when the letter sequence is unambiguous.

    LLMs sometimes treat ``*`` or ``\\`` inside fictional names as formatting
    and emit the same letters without the separator. Replacement is safe only
    when one exact source spelling maps to the normalized letter sequence.
    """
    if not source_text or not candidate_text:
        return candidate_text

    by_letters: dict[str, set[str]] = defaultdict(set)
    for exact in extract_symbol_bearing_names(source_text):
        by_letters[_letters_only_key(exact)].add(exact)

    result = candidate_text
    for spellings in by_letters.values():
        if len(spellings) != 1:
            continue
        exact = next(iter(spellings))
        parts = [part for part in re.split(r"[*\\]", exact) if part]
        if len(parts) < 2:
            continue
        # Match only the damaged no-symbol/space-separated rendering. The exact
        # spelling itself does not match because the separator allows whitespace
        # but not ``*`` or ``\\``.
        pattern = re.compile(
            r"(?<!\w)" + r"\s*".join(re.escape(part) for part in parts) + r"(?!\w)",
            re.IGNORECASE,
        )
        result = pattern.sub(lambda _match, value=exact: value, result)
    return result


def _letters_only_key(value: str) -> str:
    return re.sub(r"[*\\\s]+", "", value or "").casefold()


def _ocr_fidelity_key(value: str) -> str:
    # OCR commonly confuses terminal uppercase I with lowercase l. Restricting
    # this equivalence to segment endings avoids broad lexical normalization.
    normalized = re.sub(r"I(?=$|[*\\])", "l", value or "")
    return normalized.casefold()
