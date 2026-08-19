"""Conservative deterministic repair of source invariants.

These repairs run before semantic quality gates.  They only restore exact
source identifiers when the candidate contains an unambiguous symbol-stripped
form, avoiding another paid LLM request for a mechanical defect.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Iterable

from .technical_identifiers import SYMBOLIC_IDENTIFIER_PATTERN


_AMBIGUOUS_SINGLE_LETTER_BASES = frozenset({"A", "E", "I", "O", "U", "Y"})
_SYMBOL_SUFFIX_RE = re.compile(r"(?:\+\+|#|\*)$")


@dataclass(frozen=True)
class SourceInvariantRepair:
    source: str
    replacement_count: int
    kind: str = "symbolic_identifier"


def repair_source_invariants(
    source_text: str,
    candidate_text: str,
) -> tuple[str, tuple[SourceInvariantRepair, ...]]:
    """Restore symbol-bearing identifiers when correspondence is unambiguous.

    Example: if the source contains five occurrences of ``Q*`` and the
    candidate contains one ``Q*`` plus four standalone ``Q`` tokens, the four
    stripped forms are restored.  Ambiguous prose letters and sources that also
    use the bare form are intentionally left unchanged.
    """

    source = str(source_text or "")
    candidate = str(candidate_text or "")
    if not source or not candidate:
        return candidate, ()

    source_identifiers = Counter(
        match.group(0)
        for match in SYMBOLIC_IDENTIFIER_PATTERN.finditer(source)
    )
    if not source_identifiers:
        return candidate, ()

    repaired = candidate
    repairs: list[SourceInvariantRepair] = []
    for identifier, source_count in sorted(
        source_identifiers.items(),
        key=lambda item: (-len(item[0]), item[0]),
    ):
        suffix_match = _SYMBOL_SUFFIX_RE.search(identifier)
        if suffix_match is None:
            continue
        base = identifier[: suffix_match.start()]
        if not _safe_to_restore_base(base):
            continue

        exact_pattern = _exact_identifier_pattern(identifier)
        bare_pattern = _bare_identifier_pattern(base)
        source_bare_count = len(bare_pattern.findall(source))
        if source_bare_count:
            continue

        candidate_exact_count = len(exact_pattern.findall(repaired))
        missing = source_count - candidate_exact_count
        if missing <= 0:
            continue

        bare_matches = list(bare_pattern.finditer(repaired))
        # Exact cardinality is the key safety constraint: never guess which
        # occurrence should carry the source symbol.
        if len(bare_matches) != missing:
            continue

        repaired, replaced = bare_pattern.subn(
            lambda _match, value=identifier: value,
            repaired,
            count=missing,
        )
        if replaced != missing:
            return candidate, ()
        repairs.append(
            SourceInvariantRepair(
                source=identifier,
                replacement_count=replaced,
            )
        )

    return repaired, tuple(repairs)


def _safe_to_restore_base(base: str) -> bool:
    if not base or len(base) > 40:
        return False
    if len(base) == 1 and base.upper() in _AMBIGUOUS_SINGLE_LETTER_BASES:
        return False
    return bool(re.fullmatch(r"[A-Z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*", base))


def _exact_identifier_pattern(identifier: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![\w*+#]){re.escape(identifier)}(?![\w*+#])"
    )


def _bare_identifier_pattern(base: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<![\w*+#]){re.escape(base)}(?![\w*+#])"
    )


def summarize_source_invariant_repairs(
    repairs: Iterable[SourceInvariantRepair],
) -> str:
    details = [
        f"{repair.source} x{repair.replacement_count}"
        for repair in repairs
    ]
    return ", ".join(details)
