"""Small deterministic style hints for cross-chunk continuity."""

from __future__ import annotations

import re
from typing import Mapping, Optional


_WORD_RE = re.compile(r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+", re.UNICODE)
_SENTENCE_RE = re.compile(r"[.!?。！？]+")
_DIALOGUE_RE = re.compile(r"(^|\n)\s*(?:[-–—]|[\"“«])")


def style_continuity_enabled(prompt_options: Optional[Mapping]) -> bool:
    """Return whether refinement prompts should include local style guidance."""
    if prompt_options and prompt_options.get("style_continuity") is False:
        return False
    return True


def build_style_continuity_hint(
    reference_text: str,
    *,
    prompt_options: Optional[Mapping] = None,
    min_reference_words: int = 60,
) -> str:
    """Build a compact, token-cheap style hint from adjacent accepted text.

    The hint never includes long source passages. It gives the LLM only local
    cadence targets, which helps reduce chunk-to-chunk register drift without
    spending tokens on another model call.
    """
    if not style_continuity_enabled(prompt_options):
        return ""
    reference = (reference_text or "").strip()
    if _word_count(reference) < min_reference_words:
        return ""

    stats = _style_stats(reference)
    paragraph_shape = _paragraph_shape_label(stats["paragraphs_per_1000_words"])
    sentence_shape = _sentence_shape_label(stats["avg_sentence_words"])
    dialogue_shape = "present" if stats["dialogue_markers_per_1000_words"] >= 2.0 else "low"
    punctuation_shape = _punctuation_shape_label(stats["semicolon_colon_per_1000_words"])

    return (
        "# STYLE CONTINUITY HINT\n"
        "Use this only to keep adjacent chunks stylistically coherent; do not copy facts or wording from context.\n"
        f"- Adjacent accepted cadence: {sentence_shape} sentences, {paragraph_shape} paragraphing, "
        f"{dialogue_shape} dialogue markers, {punctuation_shape} semicolon/colon cadence.\n"
        "- Keep the same register, paragraph density, and dialogue rhythm unless the current source chunk clearly changes voice.\n"
        "- Avoid sudden shifts into summary, bullet-like exposition, generic assistant phrasing, or a different regional register."
    )


def _style_stats(text: str) -> dict[str, float]:
    words = max(1, _word_count(text))
    paragraphs = [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    sentences = max(1, len(_SENTENCE_RE.findall(text or "")))
    return {
        "avg_sentence_words": min(80.0, words / sentences),
        "paragraphs_per_1000_words": min(30.0, len(paragraphs) * 1000.0 / words),
        "dialogue_markers_per_1000_words": min(30.0, len(_DIALOGUE_RE.findall(text or "")) * 1000.0 / words),
        "semicolon_colon_per_1000_words": min(40.0, (text.count(";") + text.count(":")) * 1000.0 / words),
    }


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _sentence_shape_label(avg_sentence_words: float) -> str:
    if avg_sentence_words < 11:
        return "short"
    if avg_sentence_words > 26:
        return "long"
    return "medium-length"


def _paragraph_shape_label(paragraphs_per_1000_words: float) -> str:
    if paragraphs_per_1000_words < 4:
        return "dense"
    if paragraphs_per_1000_words > 14:
        return "frequent"
    return "moderate"


def _punctuation_shape_label(semicolon_colon_per_1000_words: float) -> str:
    if semicolon_colon_per_1000_words < 3:
        return "light"
    if semicolon_colon_per_1000_words > 16:
        return "heavy"
    return "moderate"
