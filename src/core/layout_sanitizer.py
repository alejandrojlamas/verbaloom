"""Pre-LLM layout cleanup for PDF/DOCX style artifacts.

The translation model should see editorial content, not page furniture.  This
module removes pagination, repeated running headers/footers, page-break markers,
and inline style wrappers before chunking.  Output styling is rebuilt later by
the selected exporter (TXT/DOCX/PDF/EPUB).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from src.common.inline_markdown import parse_inline_markdown
from src.utils.text_encoding import remove_pdf_toc_dot_leaders, remove_source_link_artifacts


LAYOUT_SANITIZER_VERSION = "layout-sanitizer-v1"


_PAGE_NUM_RE = re.compile(
    r"""
    ^\s*
    (?:[-\u2013\u2014]\s*)?
    (?:
        (?:(?:page|pagina|pag\.?|p\.)\s*)?
        (?:\d{1,5}|[ivxlcdm]{1,10})
        (?:\s*(?:of|de|/)\s*(?:\d{1,5}|[ivxlcdm]{1,10}))?
    )
    (?:\s*[-\u2013\u2014])?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PAGE_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:page|pagina|pag\.?|p\.)\s*(?:\d{1,5}|[ivxlcdm]{1,10})(?:\s*(?:of|de|/)\s*\d{1,5})?\s*$",
    re.IGNORECASE,
)
_LEADING_TRAILING_PAGE_NUM_RE = re.compile(
    r"""
    (?:^\s*(?:[-\u2013\u2014]\s*)?(?:\d{1,5}|[ivxlcdm]{1,10})(?:\s*[-\u2013\u2014])?\s+)
    |
    (?:\s+(?:[-\u2013\u2014]\s*)?(?:\d{1,5}|[ivxlcdm]{1,10})(?:\s*[-\u2013\u2014])?\s*$)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DECORATIVE_RULE_RE = re.compile(r"^\s*[-_=*~•·\u2022\u00b7]{3,}\s*$")
_DOTTED_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,5}\.\s*$")
_FORM_FEED_RE = re.compile(r"\f+")
_HYPHENATED_LINE_BREAK_RE = re.compile(
    r"(?<=[A-Za-z\u00c0-\u024f])-\n(?=[a-z\u00df-\u024f])"
)
_LETTER_CLASS = r"A-Za-z\u00c0-\u024f"
_BOLD_ITALIC_RE = re.compile(rf"\*\*\*([^*\n]*[{_LETTER_CLASS}][^*\n]*)\*\*\*")
_BOLD_RE = re.compile(rf"\*\*([^*\n]*[{_LETTER_CLASS}][^*\n]*)\*\*")
_ITALIC_RE = re.compile(rf"(?<!\*)\*([^*\n]*[{_LETTER_CLASS}][^*\n]*)\*(?!\*)")
_UNDERLINE_RE = re.compile(rf"__([^_\n]*[{_LETTER_CLASS}][^_\n]*)__")


@dataclass
class LayoutSanitizerReport:
    source_type: str
    pages_seen: int = 0
    lines_before: int = 0
    lines_after: int = 0
    removed_page_artifacts: int = 0
    removed_repeated_margin_lines: int = 0
    removed_style_markers: int = 0
    dehyphenated_breaks: int = 0


def should_use_sanitized_text_pipeline(
    file_type: str,
    prompt_options: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether rich input should be flattened before LLM calls."""
    options = prompt_options or {}
    ft = str(file_type or "").strip().lower()
    if ft not in {"epub", "docx"}:
        return False
    if _truthy(options.get("preserve_source_formatting")):
        return False
    if ft == "docx":
        return True
    return bool(
        _truthy(options.get("plain_text_mode"))
        or options.get("text_first_pipeline") is True
    )


def sanitize_extracted_text(
    text: str,
    *,
    source_type: str = "text",
    strip_inline_style: bool = False,
) -> tuple[str, LayoutSanitizerReport]:
    """Sanitize already-extracted document text before chunking."""
    report = LayoutSanitizerReport(source_type=source_type)
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    report.lines_before = len(value.splitlines())

    value, page_breaks = _FORM_FEED_RE.subn("\n\n", value)
    report.removed_page_artifacts += page_breaks
    value = remove_pdf_toc_dot_leaders(value)
    before_source_link_lines = len(value.splitlines())
    value = remove_source_link_artifacts(value)
    after_source_link_lines = len(value.splitlines())
    if after_source_link_lines < before_source_link_lines:
        report.removed_page_artifacts += before_source_link_lines - after_source_link_lines

    if strip_inline_style:
        value, style_count = strip_inline_style_markers(value)
        report.removed_style_markers += style_count

    value, dehyphenated = _HYPHENATED_LINE_BREAK_RE.subn("", value)
    report.dehyphenated_breaks += dehyphenated

    lines: list[str] = []
    for raw in value.splitlines():
        line = _normalize_line(raw)
        if not line:
            lines.append("")
            continue
        if _is_page_artifact_line(line, margin=False):
            report.removed_page_artifacts += 1
            continue
        if source_type.lower() in {"epub", "pdf", "docx"} and _DOTTED_PAGE_NUMBER_RE.match(line):
            report.removed_page_artifacts += 1
            continue
        if _DECORATIVE_RULE_RE.match(line):
            report.removed_page_artifacts += 1
            continue
        lines.append(line)

    cleaned = _join_clean_lines(lines)
    cleaned = remove_pdf_toc_dot_leaders(cleaned)
    report.lines_after = len(cleaned.splitlines())
    return cleaned, report


def sanitize_extracted_pages(
    pages: Iterable[str],
    *,
    source_type: str = "pdf",
) -> tuple[str, LayoutSanitizerReport]:
    """Sanitize page-aware extraction, removing repeated margins safely."""
    raw_pages = [page or "" for page in pages]
    report = LayoutSanitizerReport(source_type=source_type, pages_seen=len(raw_pages))
    normalized_pages = [_page_lines(page) for page in raw_pages]
    report.lines_before = sum(len(lines) for lines in normalized_pages)

    repeated_margin_keys = _repeated_margin_keys(normalized_pages)
    cleaned_pages: list[str] = []

    for lines in normalized_pages:
        cleaned_lines: list[str] = []
        last_index = len(lines) - 1
        for index, line in enumerate(lines):
            in_margin = index <= 2 or index >= max(0, last_index - 2)
            key = _running_margin_key(line)
            if in_margin and _is_page_artifact_line(line, margin=True):
                report.removed_page_artifacts += 1
                continue
            if in_margin and key and key in repeated_margin_keys:
                report.removed_repeated_margin_lines += 1
                continue
            if _DECORATIVE_RULE_RE.match(line):
                report.removed_page_artifacts += 1
                continue
            cleaned_lines.append(line)
        page_text = _join_clean_lines(cleaned_lines)
        if page_text:
            cleaned_pages.append(page_text)

    cleaned = "\n\n".join(cleaned_pages)
    cleaned, text_report = sanitize_extracted_text(cleaned, source_type=source_type)
    report.removed_page_artifacts += text_report.removed_page_artifacts
    report.removed_style_markers += text_report.removed_style_markers
    report.dehyphenated_breaks += text_report.dehyphenated_breaks
    report.lines_after = len(cleaned.splitlines())
    return cleaned, report


def strip_inline_style_markers(text: str) -> tuple[str, int]:
    """Remove markdown wrappers used only to preserve source styling."""
    value = text or ""
    segments = parse_inline_markdown(value)
    total = sum(1 for segment in segments if segment.has_formatting)
    if total:
        value = "".join(segment.text for segment in segments)
    for pattern in (_BOLD_ITALIC_RE, _BOLD_RE, _ITALIC_RE, _UNDERLINE_RE):
        value, count = pattern.subn(lambda match: match.group(1), value)
        total += count
    return value, total


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _page_lines(page: str) -> list[str]:
    page = remove_pdf_toc_dot_leaders((page or "").replace("\r\n", "\n").replace("\r", "\n"))
    return [
        _normalize_line(line)
        for line in page.splitlines()
        if _normalize_line(line)
    ]


def _normalize_line(line: str) -> str:
    return re.sub(r"[ \t]+", " ", line or "").strip()


def _is_page_artifact_line(line: str, *, margin: bool = False) -> bool:
    if not line:
        return False
    if _PAGE_NUM_PREFIX_RE.match(line):
        return True
    return margin and bool(_PAGE_NUM_RE.match(line))


def _running_margin_key(line: str) -> str:
    cleaned = _normalize_line(line)
    if not cleaned or len(cleaned) > 120:
        return ""
    if _is_page_artifact_line(cleaned, margin=True):
        return ""
    cleaned = _LEADING_TRAILING_PAGE_NUM_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\u2013\u2014")
    if len(cleaned) < 4:
        return ""
    if sum(ch.isalpha() for ch in cleaned) < 3:
        return ""
    return re.sub(r"\W+", " ", cleaned.casefold()).strip()


def _repeated_margin_keys(pages: list[list[str]]) -> set[str]:
    if len(pages) < 2:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages:
        page_keys: set[str] = set()
        if not lines:
            continue
        candidates = lines[:3] + lines[-3:]
        for line in candidates:
            key = _running_margin_key(line)
            if key:
                page_keys.add(key)
        counter.update(page_keys)
    threshold = max(2, int(len(pages) * 0.2))
    return {key for key, count in counter.items() if count >= threshold}


def _join_clean_lines(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()
