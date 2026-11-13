"""Structural partitioning for placeholder recovery.

Block-level placeholders are immutable separators. Only placeholders that live
inside one visible block may be aligned proportionally after an LLM drops them.
This prevents a valid-looking XHTML tree from receiving text from neighboring
paragraphs, headings, table cells, or captions.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
from typing import Dict, Literal

from src.common.placeholder_format import PlaceholderFormat

from .tag_classifier import TagClassifier


PartKind = Literal["content", "structure"]
_TAG_CLASSIFIER = TagClassifier()
_PARAGRAPH_OPEN = re.compile(r"<p(?:\s[^<>]*)?>\s*$", re.IGNORECASE)
_PARAGRAPH_CLOSE = re.compile(r"^\s*</p\s*>", re.IGNORECASE)


@dataclass(frozen=True)
class StructureSafePart:
    kind: PartKind
    text: str
    tag_map: Dict[str, str]


def is_structural_tag_group(tag_group: str) -> bool:
    return bool(
        _TAG_CLASSIFIER.is_block_opening_tag(tag_group)
        or _TAG_CLASSIFIER.is_block_closing_tag(tag_group)
    )


def structural_placeholders(tag_map: Dict[str, str]) -> list[str]:
    return [
        placeholder
        for placeholder, tag_group in tag_map.items()
        if is_structural_tag_group(tag_group)
    ]


def split_structure_safe_parts(
    text: str,
    tag_map: Dict[str, str],
) -> list[StructureSafePart]:
    """Split text around block-level placeholders while preserving exact order."""
    fmt = PlaceholderFormat.from_config()
    structural = set(structural_placeholders(tag_map))
    if not structural:
        inline_map = {
            placeholder: tag_map[placeholder]
            for _start, _end, placeholder, _index in fmt.find_all(text)
            if placeholder in tag_map
        }
        return [StructureSafePart("content", text, inline_map)]

    parts: list[StructureSafePart] = []
    cursor = 0
    for start, end, placeholder, _index in fmt.find_all(text):
        if placeholder not in structural:
            continue
        if start > cursor:
            content = text[cursor:start]
            inline_map = {
                found: tag_map[found]
                for _s, _e, found, _i in fmt.find_all(content)
                if found in tag_map and found not in structural
            }
            parts.append(StructureSafePart("content", content, inline_map))
        parts.append(
            StructureSafePart(
                "structure",
                placeholder,
                {placeholder: tag_map[placeholder]},
            )
        )
        cursor = end

    if cursor < len(text):
        content = text[cursor:]
        inline_map = {
            found: tag_map[found]
            for _s, _e, found, _i in fmt.find_all(content)
            if found in tag_map and found not in structural
        }
        parts.append(StructureSafePart("content", content, inline_map))
    return parts


def structure_signature(text: str, tag_map: Dict[str, str]) -> tuple[str, ...]:
    """Return the ordered block-boundary signature of a placeholder stream."""
    structural = set(structural_placeholders(tag_map))
    fmt = PlaceholderFormat.from_config()
    return tuple(
        placeholder
        for _start, _end, placeholder, _index in fmt.find_all(text)
        if placeholder in structural
    )


def refinement_preserves_paragraphs(
    draft: str, revision: str, tag_map: Dict[str, str]
) -> bool:
    """Check for extreme text movement across intact paragraph markers."""
    fmt = PlaceholderFormat.from_config()

    def paragraph_lengths(text: str) -> dict[tuple[str, str], int]:
        boundaries = [
            (start, end, placeholder)
            for start, end, placeholder, _index in fmt.find_all(text)
            if is_structural_tag_group(tag_map.get(placeholder, ""))
        ]
        return {
            (opening, closing): len(" ".join(unescape(fmt.remove_all(text[left:right])).split()))
            for (_, left, opening), (right, _, closing) in zip(boundaries, boundaries[1:])
            if _PARAGRAPH_OPEN.search(tag_map[opening]) and _PARAGRAPH_CLOSE.match(tag_map[closing])
        }

    if not tag_map:
        return True
    if structure_signature(draft, tag_map) != structure_signature(revision, tag_map):
        return False
    before, after = paragraph_lengths(draft), paragraph_lengths(revision)
    return all(
        size < 40 or 0.28 <= after.get(key, 0) / size <= 4.5
        for key, size in before.items()
    )
