"""Shared detection primitives for exact technical identifiers."""

from __future__ import annotations

import re


# The negative lookbehind avoids treating the closing side of ``*Title*`` as
# an identifier while still covering forms such as ``Q*-related``.
SYMBOLIC_IDENTIFIER_PATTERN = re.compile(
    r"(?<![\w*+#])"
    r"(?:[A-Z][A-Za-z0-9]{0,31}(?:[-.][A-Za-z0-9]+)*)"
    r"(?:\+\+|#|\*)"
    r"(?![\w*+#])"
)

# Explicit syntax avoids mistaking an ordinary missing sentence boundary such
# as ``journey.Afterward`` for a bare domain. These spans are data, not prose.
SPACING_LITERAL_PATTERN = re.compile(
    r"(?:https?|ftp)://[^\s<>\"“”«»)]+|\bwww\.[^\s<>\"“”«»)]+"
    r"|\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"
    r"|\b(?:doi(?:\.org/|:)\s*)?10\.\d{4,9}/[^\s<>\"“”«»)]+"
    r"|(?<!\w)(?:\.\.?/|/)[A-Z0-9._~!$&'()*+,;=:@%/\-]+",
    re.IGNORECASE,
)


def matches_outside_technical_literals(pattern: re.Pattern[str], text: str):
    """Yield prose findings without interpreting punctuation inside literals.

    Match containment, not overlap, keeps punctuation at a literal boundary
    eligible. A zero-width word boundary is excluded only strictly inside.
    """
    spans = [match.span() for match in SPACING_LITERAL_PATTERN.finditer(text)]
    for match in pattern.finditer(text):
        start, end = match.span()
        if any(
            (left < start < right) if start == end else (left <= start and end <= right)
            for left, right in spans
        ):
            continue
        yield match
