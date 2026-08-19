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
