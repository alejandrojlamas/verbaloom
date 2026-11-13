"""Reuse a paid model candidate whose placeholders were damaged.

Strict placeholder validation rejects a candidate when even one ``[idN]`` is
missing, duplicated or unknown. The prose of such a candidate is normally
complete; only its placeholder stream is defective. Requesting the whole unit
again discards output that was already paid for, and the block-by-block
structural recovery then translates the same text a further time.

This module separates what is provably intact from what is not:

* Block boundaries stay fixed. A missing block-level placeholder is restored
  only where the source proves its position (a chunk edge, or a neighbor with
  no readable content). Readable source blocks that the candidate merged are
  never split heuristically; they are reported as a damaged region.
* A block whose own inline placeholders are all present exactly once is kept
  byte for byte. Stray tokens that belong to no placeholder of that block are
  removed, because they have no legitimate position there.
* Every other block is a damaged region. The caller re-translates only that
  region. A deterministic last resort is offered for a single block: surviving
  placeholders stay where the model put them and only missing *formatting
  spans* (``<i>``, ``<span>``, ``<a>`` ...) are interpolated between the two
  nearest surviving anchors. A missing line break, image or protected literal
  is position-critical content and is never interpolated.

The result must still pass the caller's strict validator and every later
fidelity, language and audit gate. Salvage never approves a candidate.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.common.placeholder_format import PlaceholderFormat

from .placeholder_validator import PlaceholderValidator
from .structure_safe_fallback import structural_placeholders

# Unambiguous spelling variants of one placeholder: ``[ID3]``, ``[id 3]``,
# ``[id03]`` and the same token inside fullwidth or lenticular brackets.
_OPENING_BRACKETS = "[" + chr(0xFF3B) + chr(0x3010)
_CLOSING_BRACKETS = "]" + chr(0xFF3D) + chr(0x3011)
_VARIANT_RE = re.compile(
    "[" + re.escape(_OPENING_BRACKETS) + "]"
    + r"\s*id\s*(\d+)\s*"
    + "[" + re.escape(_CLOSING_BRACKETS) + "]",
    re.IGNORECASE,
)
# Anything that still looks like a placeholder after the well-formed ones were
# removed (``(id5)``, ``[id6``, ``id7]`` ...). Such debris would reach the
# reader as literal text, so a block containing it is never kept as is.
_DEBRIS_OPENING = re.escape(_OPENING_BRACKETS + "({<")
_DEBRIS_CLOSING = re.escape(_CLOSING_BRACKETS + ")}>")
_DEBRIS_RE = re.compile(
    "[" + _DEBRIS_OPENING + "]" + r"\s*/?\s*id\s*\d+\s*" + "[" + _DEBRIS_CLOSING + "]?"
    + r"|\bid\s*\d+\s*" + "[" + _DEBRIS_CLOSING + "]"
    + r"|\[\s*id\b",
    re.IGNORECASE,
)
# Space, no-break space, sentence punctuation (Latin, Spanish, CJK).
_WORD_BOUNDARY_CHARS = frozenset(
    " \t\n\r.,;:!?"
    + "".join(chr(code) for code in (
        0x00A0, 0x00A1, 0x00BF, 0x2026,
        0x3001, 0x3002, 0xFF0C, 0xFF1A, 0xFF1F, 0xFF01,
    ))
)
_INLINE_SPACE_CHARS = frozenset(" \t" + chr(0x00A0))
_CLOSING_PUNCTUATION = frozenset(".,;:!?" + chr(0x2026))
# Scripts written without spaces have no word boundary to snap to.
_UNSPACED_SCRIPT_START = 0x2E80

# Elements that only style or link running text. Moving one of their tags by a
# word changes typography, never content. Void elements (``br``, ``img``) are
# deliberately absent.
_FORMAT_TAGS = frozenset({
    "a", "abbr", "b", "bdi", "bdo", "big", "cite", "code", "dfn", "del", "em",
    "font", "i", "ins", "kbd", "mark", "q", "s", "samp", "small", "span",
    "strike", "strong", "sub", "sup", "tt", "u", "var",
})
_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z][\w:.-]*)[^<>]*?(/?)\s*>")


@dataclass(frozen=True)
class DamagedRegion:
    """Readable source blocks that the candidate merged, lost or corrupted."""

    candidate_region: int
    """Index of the plan region that must be replaced."""
    source_text: str
    """Exact source sub-stream of the region, block placeholders included."""
    readable_blocks: int
    interpolated: Optional[str] = None
    """Deterministic last-resort repair of the region, when one exists."""


@dataclass
class SalvagePlan:
    """Outcome of analysing one rejected candidate."""

    reason: str = ""
    """Why nothing could be reused. Empty when a plan exists."""
    regions: List[Optional[str]] = field(default_factory=list)
    """Reusable candidate regions; ``None`` marks a damaged region."""
    separators: List[str] = field(default_factory=list)
    """Block placeholders that survived in the candidate, in order."""
    damaged: List[DamagedRegion] = field(default_factory=list)
    repaired_blocks: int = 0
    restored_boundaries: int = 0
    readable_blocks: int = 0

    @property
    def usable(self) -> bool:
        return not self.reason

    @property
    def complete(self) -> bool:
        return self.usable and not self.damaged

    @property
    def damaged_share(self) -> float:
        """Share of readable blocks that still need a model request."""
        if not self.usable or self.readable_blocks <= 0:
            return 1.0
        damaged = sum(region.readable_blocks for region in self.damaged)
        return min(1.0, damaged / self.readable_blocks)

    def assemble(self, replacements: Optional[Dict[int, str]] = None) -> Optional[str]:
        """Join the regions, filling damaged ones from ``replacements``."""
        if not self.usable:
            return None
        parts: List[str] = []
        for index, region in enumerate(self.regions):
            if region is None:
                region = (replacements or {}).get(index)
                if region is None:
                    return None
            parts.append(region)
            if index < len(self.separators):
                parts.append(self.separators[index])
        return "".join(parts)


def placeholder_debris(text: str) -> bool:
    """Return whether malformed placeholder residue remains in ``text``."""
    fmt = PlaceholderFormat.from_config()
    return bool(_DEBRIS_RE.search(fmt.remove_all(text or "")))


def normalize_placeholder_variants(candidate_text: str, tag_map: Dict[str, str]) -> str:
    """Rewrite unambiguous spelling variants of an expected placeholder.

    ``[ID 3]`` can only mean ``[id3]``. A variant is rewritten when the
    canonical placeholder is expected and not already present, so the rewrite
    never creates a duplicate.
    """
    fmt = PlaceholderFormat.from_config()
    text = candidate_text or ""
    present = Counter(found for _s, _e, found, _i in fmt.find_all(text))

    def replace(match: "re.Match[str]") -> str:
        canonical = fmt.create(int(match.group(1)))
        if match.group(0) == canonical:
            return canonical
        if canonical not in tag_map or present[canonical]:
            return match.group(0)
        present[canonical] += 1
        return canonical

    return _VARIANT_RE.sub(replace, text)


def is_formatting_group(tag_group: str) -> bool:
    """Whether a placeholder holds nothing but inline formatting tags."""
    value = str(tag_group or "")
    if not value.strip():
        return False
    cursor = 0
    seen = False
    for match in _TAG_RE.finditer(value):
        if value[cursor:match.start()].strip():
            return False
        closing, name, self_closing = match.group(1), match.group(2), match.group(3)
        if self_closing and not closing:
            return False
        if name.lower() not in _FORMAT_TAGS:
            return False
        seen = True
        cursor = match.end()
    return seen and not value[cursor:].strip()


def _clean_offsets(text: str, fmt: PlaceholderFormat) -> Tuple[str, List[Tuple[str, int]]]:
    """Return text without placeholders plus every placeholder's clean offset."""
    clean: List[str] = []
    offsets: List[Tuple[str, int]] = []
    cursor = 0
    length = 0
    for start, end, placeholder, _index in fmt.find_all(text):
        clean.append(text[cursor:start])
        length += start - cursor
        offsets.append((placeholder, length))
        cursor = end
    clean.append(text[cursor:])
    return "".join(clean), offsets


def _drop_placeholders(text: str, fmt: PlaceholderFormat, keep: frozenset) -> str:
    """Remove every placeholder outside ``keep`` without leaving a stray space."""
    parts: List[str] = []
    cursor = 0
    for start, end, placeholder, _index in fmt.find_all(text):
        if placeholder in keep:
            continue
        piece = text[cursor:start]
        cursor = end
        before = text[start - 1:start]
        after = text[end:end + 1]
        if before and before in _INLINE_SPACE_CHARS:
            if after and after in _INLINE_SPACE_CHARS:
                cursor = end + 1
            elif (not after or after in _CLOSING_PUNCTUATION) and piece:
                piece = piece[:-1]
        parts.append(piece)
    parts.append(text[cursor:])
    return "".join(parts)


def _snap_to_word_boundary(text: str, position: int, low: int, high: int) -> int:
    """Move ``position`` to the nearest word boundary inside ``[low, high]``."""
    position = max(low, min(position, high))
    if position <= 0 or position >= len(text):
        return position
    if text[position] in _WORD_BOUNDARY_CHARS or text[position - 1] in _WORD_BOUNDARY_CHARS:
        return position
    if ord(text[position]) >= _UNSPACED_SCRIPT_START:
        return position
    left = position
    while left > low and text[left - 1] not in _WORD_BOUNDARY_CHARS:
        left -= 1
    right = position
    while right < high and text[right] not in _WORD_BOUNDARY_CHARS:
        right += 1
    left_ok = left == 0 or text[left - 1] in _WORD_BOUNDARY_CHARS
    right_ok = right == len(text) or text[right] in _WORD_BOUNDARY_CHARS
    if left_ok and right_ok:
        return left if position - left <= right - position else right
    if left_ok:
        return left
    if right_ok:
        return right
    return position


def _prefer_tag_side(text: str, position: int, tag_group: str, low: int, high: int) -> int:
    """Keep opening tags before a word and closing tags after one."""
    value = str(tag_group or "").lstrip()
    if value.startswith("</"):
        while position > low and text[position - 1] in _INLINE_SPACE_CHARS:
            position -= 1
    elif value.startswith("<"):
        while position < high and position < len(text) and text[position] in _INLINE_SPACE_CHARS:
            position += 1
    return max(low, min(position, high))


def _first_word_end(text: str, low: int, high: int) -> int:
    """End of the first word inside ``[low, high]``."""
    position = low
    while position < high and text[position] in _WORD_BOUNDARY_CHARS:
        position += 1
    while position < high and text[position] not in _WORD_BOUNDARY_CHARS:
        position += 1
    return position


def _last_word_start(text: str, low: int, high: int) -> int:
    """Start of the last word inside ``[low, high]``."""
    position = high
    while position > low and text[position - 1] in _WORD_BOUNDARY_CHARS:
        position -= 1
    while position > low and text[position - 1] not in _WORD_BOUNDARY_CHARS:
        position -= 1
    return position


def _mid_word_distance(clean_source: str, offset: int) -> Optional[int]:
    """Characters from the word start for a placeholder inside a word."""
    if not 0 < offset < len(clean_source):
        return None
    if not (clean_source[offset - 1].isalnum() and clean_source[offset].isalnum()):
        return None
    start = offset
    while start > 0 and clean_source[start - 1].isalnum():
        start -= 1
    return offset - start


def keep_intact_block(source_block: str, candidate_block: str) -> Optional[str]:
    """Return the candidate block when its own placeholders are provably intact.

    Every placeholder of the source block must appear exactly once. Tokens
    that belong to no placeholder of this block are removed. ``None`` means the
    block needs a repair.
    """
    fmt = PlaceholderFormat.from_config()
    expected = [found for _s, _e, found, _i in fmt.find_all(source_block)]
    counts = Counter(found for _s, _e, found, _i in fmt.find_all(candidate_block))
    if len(set(expected)) != len(expected):
        return None
    if any(counts[placeholder] != 1 for placeholder in expected):
        return None
    result = candidate_block
    if len(counts) != len(expected):
        result = _drop_placeholders(candidate_block, fmt, frozenset(expected))
        # Removing a token can splice its neighbors into a new one
        # (``[id[id3]1]``). Such a block is not provably intact.
        remaining = sorted(found for _s, _e, found, _i in fmt.find_all(result))
        if remaining != sorted(expected):
            return None
    clean = fmt.remove_all(result)
    if _DEBRIS_RE.search(clean):
        return None
    if fmt.remove_all(source_block).strip() and not clean.strip():
        return None
    return result


def realign_inline_placeholders(
    source_block: str,
    candidate_block: str,
    tag_map: Dict[str, str],
) -> Optional[str]:
    """Rebuild one block's inline placeholders around the surviving anchors.

    ``source_block`` and ``candidate_block`` contain no block-level
    placeholder. A placeholder the model kept exactly once is never moved.
    Returns ``None`` when the candidate cannot be reused.
    """
    intact = keep_intact_block(source_block, candidate_block)
    if intact is not None:
        return intact

    fmt = PlaceholderFormat.from_config()
    clean_source, source_offsets = _clean_offsets(source_block, fmt)
    expected = [placeholder for placeholder, _offset in source_offsets]
    if len(set(expected)) != len(expected) or any(p not in tag_map for p in expected):
        return None

    counts = Counter(found for _s, _e, found, _i in fmt.find_all(candidate_block))
    kept = frozenset(placeholder for placeholder in expected if counts[placeholder] == 1)
    pruned = _drop_placeholders(candidate_block, fmt, kept)
    clean_candidate, candidate_offsets = _clean_offsets(pruned, fmt)
    # Removing a token can splice its neighbors into a new one.
    if sorted(placeholder for placeholder, _offset in candidate_offsets) != sorted(kept):
        return None
    if _DEBRIS_RE.search(clean_candidate):
        return None
    if clean_source.strip() and not clean_candidate.strip():
        return None
    # Only formatting spans may be re-inserted without the model.
    if any(
        placeholder not in kept and not is_formatting_group(tag_map[placeholder])
        for placeholder in expected
    ):
        return None

    kept_offset = dict(candidate_offsets)
    source_order = {placeholder: index for index, placeholder in enumerate(expected)}
    source_length = len(clean_source)
    candidate_length = len(clean_candidate)

    # Survivors that keep the source order pin the interpolation. Every
    # re-inserted placeholder becomes an anchor for the ones after it.
    anchors: List[Tuple[int, int]] = [(0, 0)]
    for placeholder, source_offset in source_offsets:
        if placeholder in kept and kept_offset[placeholder] >= anchors[-1][1]:
            anchors.append((source_offset, kept_offset[placeholder]))
    anchors.append((source_length, candidate_length))

    placed: List[Tuple[int, int, str]] = [
        (offset, source_order[placeholder], placeholder)
        for placeholder, offset in candidate_offsets
    ]
    for placeholder, source_offset in source_offsets:
        if placeholder in kept:
            continue
        lower = max(anchor for anchor in anchors if anchor[0] <= source_offset)
        upper = min(anchor for anchor in anchors if anchor[0] >= source_offset)
        low, high = lower[1], max(lower[1], upper[1])
        span = upper[0] - lower[0]
        ratio = ((source_offset - lower[0]) / span) if span > 0 else 0.0
        raw = max(low, min(low + int(round(ratio * (high - low))), high))
        distance = _mid_word_distance(clean_source, source_offset)
        if distance is not None:
            # Drop caps and similar markup split a word on purpose.
            word_start = raw
            while word_start > low and clean_candidate[word_start - 1].isalnum():
                word_start -= 1
            word_end = word_start
            while word_end < high and clean_candidate[word_end].isalnum():
                word_end += 1
            position = min(word_start + distance, word_end)
        else:
            group = tag_map[placeholder]
            position = _snap_to_word_boundary(clean_candidate, raw, low, high)
            position = _prefer_tag_side(clean_candidate, position, group, low, high)
            # A span that wrapped text in the source may not come back empty.
            if group.lstrip().startswith("</"):
                if position <= low and clean_source[lower[0]:source_offset].strip():
                    position = _first_word_end(clean_candidate, low, high)
            elif position >= high and clean_source[source_offset:upper[0]].strip():
                position = _last_word_start(clean_candidate, low, high)
        position = max(low, min(position, high))
        anchors.append((source_offset, position))
        anchors.sort()
        placed.append((position, source_order[placeholder], placeholder))

    result = clean_candidate
    for position, _order, placeholder in sorted(placed, reverse=True):
        result = result[:position] + placeholder + result[position:]

    rebuilt = sorted(found for _s, _e, found, _i in fmt.find_all(result))
    if rebuilt != sorted(expected):
        return None
    return result


def plan_placeholder_salvage(
    source_text: str,
    candidate_text: str,
    tag_map: Dict[str, str],
    *,
    interpolate_inline: bool = False,
) -> SalvagePlan:
    """Analyse a rejected candidate against its source placeholder stream.

    With ``interpolate_inline`` a single block with missing formatting spans
    is repaired in place instead of being reported as damaged.
    """
    plan = SalvagePlan()
    if not (candidate_text or "").strip():
        plan.reason = "empty_candidate"
        return plan

    fmt = PlaceholderFormat.from_config()
    structural = set(structural_placeholders(tag_map))

    # Source: segments[j] is the content before separators[j]; the final
    # segment follows the last block placeholder.
    source_separators: List[str] = []
    segments: List[str] = []
    cursor = 0
    for start, end, placeholder, _index in fmt.find_all(source_text):
        if placeholder not in tag_map:
            plan.reason = "source_placeholder_unknown"
            return plan
        if placeholder not in structural:
            continue
        segments.append(source_text[cursor:start])
        source_separators.append(placeholder)
        cursor = end
    segments.append(source_text[cursor:])
    if len(set(source_separators)) != len(source_separators):
        plan.reason = "source_structure_ambiguous"
        return plan
    separator_index = {
        placeholder: index for index, placeholder in enumerate(source_separators)
    }
    readable = [bool(fmt.remove_all(segment).strip()) for segment in segments]
    plan.readable_blocks = sum(readable)

    # Candidate: split on the block placeholders that survived.
    candidate_regions: List[str] = []
    present: List[int] = []
    cursor = 0
    for start, end, placeholder, _index in fmt.find_all(candidate_text):
        if placeholder not in structural:
            continue
        index = separator_index.get(placeholder)
        if index is None:
            # The map names a block placeholder that the source stream lacks.
            plan.reason = "source_structure_inconsistent"
            return plan
        if index in present:
            plan.reason = "block_placeholder_duplicated"
            return plan
        if present and index < present[-1]:
            plan.reason = "block_placeholder_reordered"
            return plan
        candidate_regions.append(candidate_text[cursor:start])
        present.append(index)
        cursor = end
    candidate_regions.append(candidate_text[cursor:])

    def source_span(first: int, last: int) -> str:
        parts: List[str] = []
        for j in range(first, last + 1):
            parts.append(segments[j])
            if j < last:
                parts.append(source_separators[j])
        return "".join(parts)

    bounds: List[Tuple[int, int]] = []
    low = 0
    for index in present:
        bounds.append((low, index))
        low = index + 1
    bounds.append((low, len(segments) - 1))

    for region_index, ((first, last), content) in enumerate(zip(bounds, candidate_regions)):
        readable_segments = [j for j in range(first, last + 1) if readable[j]]
        missing_boundaries = last - first

        if not readable_segments:
            if fmt.remove_all(content).strip():
                plan.reason = "text_outside_source_blocks"
                return plan
            plan.regions.append(source_span(first, last))
            plan.restored_boundaries += missing_boundaries
            continue

        interpolated: Optional[str] = None
        if len(readable_segments) == 1:
            target = readable_segments[0]
            prefix = "".join(
                segments[j] + source_separators[j] for j in range(first, target)
            )
            suffix = "".join(
                source_separators[j] + segments[j + 1] for j in range(target, last)
            )
            block = keep_intact_block(segments[target], content)
            if block is None:
                inline_map = {
                    found: tag_map[found]
                    for _s, _e, found, _i in fmt.find_all(segments[target])
                }
                realigned = realign_inline_placeholders(
                    segments[target], content, inline_map,
                )
                if realigned is not None:
                    if interpolate_inline:
                        block = realigned
                    else:
                        interpolated = prefix + realigned + suffix
            if block is not None:
                if block != content:
                    plan.repaired_blocks += 1
                plan.regions.append(prefix + block + suffix)
                plan.restored_boundaries += missing_boundaries
                continue

        plan.regions.append(None)
        plan.damaged.append(DamagedRegion(
            candidate_region=region_index,
            source_text=source_span(first, last),
            readable_blocks=len(readable_segments),
            interpolated=interpolated,
        ))

    plan.separators = [source_separators[index] for index in present]
    return plan


def verify_salvaged_stream(
    source_text: str,
    result: Optional[str],
    tag_map: Dict[str, str],
) -> Optional[str]:
    """Accept a salvaged stream only when it is structurally identical."""
    if not result:
        return None
    is_valid, _error = PlaceholderValidator.validate_strict(result, tag_map)
    if not is_valid:
        return None
    fmt = PlaceholderFormat.from_config()
    structural = set(structural_placeholders(tag_map))

    def shape(text: str) -> Tuple[Tuple[str, ...], List[Tuple[str, ...]], List[bool]]:
        signature: List[str] = []
        inline: List[List[str]] = [[]]
        has_text: List[bool] = []
        cursor = 0
        block_text = ""
        for start, end, placeholder, _index in fmt.find_all(text):
            block_text += text[cursor:start]
            cursor = end
            if placeholder in structural:
                signature.append(placeholder)
                inline.append([])
                has_text.append(bool(block_text.strip()))
                block_text = ""
            else:
                inline[-1].append(placeholder)
        block_text += text[cursor:]
        has_text.append(bool(block_text.strip()))
        return tuple(signature), [tuple(sorted(block)) for block in inline], has_text

    source_signature, source_inline, source_has_text = shape(source_text)
    result_signature, result_inline, result_has_text = shape(result)
    if result_signature != source_signature:
        return None
    # An inline tag must stay inside the block that owns it.
    if result_inline != source_inline:
        return None
    # A readable source block may not come back empty, nor the reverse.
    if result_has_text != source_has_text:
        return None
    if placeholder_debris(result):
        return None
    return result


def salvage_placeholder_candidate(
    source_text: str,
    candidate_text: str,
    tag_map: Dict[str, str],
    *,
    interpolate_inline: bool = False,
) -> Optional[str]:
    """Return a valid placeholder stream built from ``candidate_text`` alone.

    ``None`` means the candidate needs a model request or cannot be reused.
    """
    candidate = normalize_placeholder_variants(candidate_text, tag_map)
    plan = plan_placeholder_salvage(
        source_text, candidate, tag_map, interpolate_inline=interpolate_inline,
    )
    if not plan.complete:
        return None
    return verify_salvaged_stream(source_text, plan.assemble(), tag_map)
