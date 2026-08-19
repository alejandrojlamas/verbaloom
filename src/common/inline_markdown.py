"""Shared inline-markdown codec for the text-first (plain text) pipeline.

The plain-text pipeline historically flattened every inline tag/run, so
bold, italic, and hyperlinks were destroyed on any EPUB/DOCX job routed
through it. This module encodes those three inline attributes as light
markdown that LLMs preserve naturally:

    **bold**   *italic*   ***bold italic***   [link text](url)

Both the DOCX extractor (python-docx runs) and the EPUB extractor (XHTML
inline tags) encode to this representation before translation, and decode
it back to runs/tags at rebuild time.

Design constraints:

- The decoder must be tolerant: LLMs occasionally drop or duplicate a
  marker. An unmatched marker is treated as literal text rather than
  corrupting the segmentation.
- Emphasis markers must hug non-space characters (CommonMark-style), so
  the encoder moves leading/trailing whitespace outside the markers.
- We deliberately do NOT escape stray ``*`` / ``[`` in source text: a
  backslash-escape convention tends to leak literal backslashes through
  the LLM. The conservative decoder keeps lone markers as literal text,
  which is the better failure mode for literary prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class InlineSegment:
    """One run of text with uniform inline formatting."""

    text: str
    bold: bool = False
    italic: bool = False
    href: Optional[str] = None

    @property
    def has_formatting(self) -> bool:
        return bool(self.bold or self.italic or self.href)


# --- Encoding (runs/tags -> markdown) ----------------------------------------


def _emphasis_wrap(text: str, bold: bool, italic: bool) -> str:
    """Wrap text with emphasis markers, keeping whitespace outside them."""
    if not text or not text.strip() or not (bold or italic):
        return text
    leading_len = len(text) - len(text.lstrip())
    trailing_len = len(text) - len(text.rstrip())
    leading = text[:leading_len]
    trailing = text[len(text) - trailing_len:] if trailing_len else ""
    core = text[leading_len: len(text) - trailing_len] if trailing_len else text[leading_len:]
    marker = "***" if (bold and italic) else ("**" if bold else "*")
    return f"{leading}{marker}{core}{marker}{trailing}"


def _merge_adjacent(segments: List[InlineSegment]) -> List[InlineSegment]:
    """Merge neighbours with identical formatting (Word splits runs arbitrarily)."""
    merged: List[InlineSegment] = []
    for seg in segments:
        if seg.text == "":
            continue
        if merged:
            prev = merged[-1]
            if (
                prev.bold == seg.bold
                and prev.italic == seg.italic
                and prev.href == seg.href
            ):
                merged[-1] = InlineSegment(
                    text=prev.text + seg.text,
                    bold=prev.bold,
                    italic=prev.italic,
                    href=prev.href,
                )
                continue
        merged.append(seg)
    return merged


def segments_to_markdown(segments: List[InlineSegment]) -> str:
    """Serialize formatted segments into inline markdown."""
    out: List[str] = []
    for seg in _merge_adjacent(list(segments or [])):
        text = seg.text
        if seg.href and text.strip():
            inner = _emphasis_wrap(text.strip(), seg.bold, seg.italic)
            leading_len = len(text) - len(text.lstrip())
            trailing_len = len(text) - len(text.rstrip())
            out.append(text[:leading_len])
            out.append(f"[{inner}]({seg.href})")
            if trailing_len:
                out.append(text[len(text) - trailing_len:])
        else:
            out.append(_emphasis_wrap(text, seg.bold, seg.italic))
    return "".join(out)


# --- Decoding (markdown -> segments) ------------------------------------------

# Emphasis: inner content must not start/end with whitespace and must be non-empty.
_BOLD_ITALIC_RE = re.compile(r"\*\*\*(?!\s)((?:[^*\n]|\*(?!\*\*))+?)(?<!\s)\*\*\*")
_BOLD_RE = re.compile(r"\*\*(?!\s)((?:[^*\n]|\*(?!\*))+?)(?<!\s)\*\*")
_ITALIC_RE = re.compile(r"\*(?![\s*])((?:[^*\n])+?)(?<![\s])\*(?!\*)")
_MAX_INLINE_LINK_LABEL_CHARS = 500
_MAX_INLINE_LINK_URL_CHARS = 2048


def _apply_flags(
    segments: List[InlineSegment],
    *,
    bold: Optional[bool] = None,
    italic: Optional[bool] = None,
    href: Optional[str] = None,
) -> List[InlineSegment]:
    out: List[InlineSegment] = []
    for seg in segments:
        out.append(InlineSegment(
            text=seg.text,
            bold=seg.bold if bold is None else (seg.bold or bold),
            italic=seg.italic if italic is None else (seg.italic or italic),
            href=seg.href if href is None else (seg.href or href),
        ))
    return out


def _parse_emphasis(text: str) -> List[InlineSegment]:
    """Parse ***x***, **x**, *x* iteratively; unmatched markers stay literal."""
    if not text:
        return []

    segments: List[InlineSegment] = [InlineSegment(text=text)]
    for pattern, flags in (
        (_BOLD_ITALIC_RE, {"bold": True, "italic": True}),
        (_BOLD_RE, {"bold": True}),
        (_ITALIC_RE, {"italic": True}),
    ):
        next_segments: List[InlineSegment] = []
        for segment in segments:
            value = segment.text
            if not value or "*" not in value:
                next_segments.append(segment)
                continue

            cursor = 0
            matched = False
            for match in pattern.finditer(value):
                matched = True
                if match.start() > cursor:
                    next_segments.append(InlineSegment(
                        text=value[cursor:match.start()],
                        bold=segment.bold,
                        italic=segment.italic,
                        href=segment.href,
                    ))
                next_segments.append(InlineSegment(
                    text=match.group(1),
                    bold=segment.bold or bool(flags.get("bold")),
                    italic=segment.italic or bool(flags.get("italic")),
                    href=segment.href,
                ))
                cursor = match.end()
            if not matched:
                next_segments.append(segment)
                continue
            if cursor < len(value):
                next_segments.append(InlineSegment(
                    text=value[cursor:],
                    bold=segment.bold,
                    italic=segment.italic,
                    href=segment.href,
                ))
        segments = next_segments
    return _merge_adjacent(segments)


def _find_markdown_link_at(text: str, start: int) -> tuple[int, int, str, str] | None:
    """Return (start, end, label, url) for a markdown link at ``start``.

    This intentionally avoids the old nested-bracket regex. Malformed PDF/OCR
    text can contain many stray ``[`` characters; a bounded scanner keeps
    preview cleanup linear and prevents WebSocket updates from freezing Flask.
    """
    if start >= len(text) or text[start] != "[":
        return None

    depth = 1
    idx = start + 1
    label_limit = min(len(text), start + _MAX_INLINE_LINK_LABEL_CHARS + 1)
    while idx < label_limit:
        char = text[idx]
        if char == "\n":
            return None
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        idx += 1
    else:
        return None

    label_end = idx
    if label_end + 1 >= len(text) or text[label_end + 1] != "(":
        return None

    url_start = label_end + 2
    url_limit = min(len(text), url_start + _MAX_INLINE_LINK_URL_CHARS)
    url_end = text.find(")", url_start, url_limit)
    if url_end < 0:
        return None

    url = text[url_start:url_end]
    if not url or any(char.isspace() or char in "()" for char in url):
        return None

    return start, url_end + 1, text[start + 1:label_end], url


def parse_inline_markdown(text: str) -> List[InlineSegment]:
    """Decode inline markdown into formatted segments (tolerant)."""
    if not text:
        return []
    if "*" not in text and "[" not in text:
        return [InlineSegment(text=text)]

    segments: List[InlineSegment] = []
    emit_cursor = 0
    search_pos = 0
    while search_pos < len(text):
        start = text.find("[", search_pos)
        if start < 0:
            break
        link = _find_markdown_link_at(text, start)
        if link is None:
            search_pos = start + 1
            continue
        link_start, link_end, link_text, url = link
        before = text[emit_cursor:link_start]
        if before:
            segments.extend(_parse_emphasis(before))
        segments.extend(_apply_flags(_parse_emphasis(link_text), href=url))
        emit_cursor = link_end
        search_pos = link_end
    tail = text[emit_cursor:]
    if tail:
        segments.extend(_parse_emphasis(tail))
    return _merge_adjacent(segments)


def strip_inline_markdown(text: str) -> str:
    """Plain-text view of an inline-markdown string (for audits/ratios)."""
    return "".join(seg.text for seg in parse_inline_markdown(text or ""))


def has_inline_markdown(text: str) -> bool:
    """Whether the string contains any decodable inline formatting."""
    if not text or ("*" not in text and "[" not in text):
        return False
    return any(seg.has_formatting for seg in parse_inline_markdown(text))


# Prompt section injected when the active job carries inline-markdown blocks.
INLINE_MARKDOWN_PROMPT_SECTION = """# INLINE FORMATTING MARKERS

The text uses lightweight markers for inline formatting:
- **bold**, *italic*, ***bold italic***
- [link text](url)

Rules:
- Keep each marker pair around the same words after translation/transformation.
- Translate the text INSIDE markers and link brackets; NEVER translate or alter URLs.
- Do not add, remove, or reorder markers; do not convert them to other styles.
- Markers must hug words: `**word**`, never `** word **`.""".strip()
