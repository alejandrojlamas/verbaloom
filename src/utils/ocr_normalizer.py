"""Detect and normalize OCR-like scan artifacts in plain text.

The goal is conservative cleanup before refinement: fix mechanical scan/OCR
damage without rewriting style or dropping meaningful content.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from src.utils.text_encoding import remove_artifact_glyphs


_WORD_CHARS = r"A-Za-zÀ-ÖØ-öø-ÿĀ-ſƀ-ɏ"
_HYPHENATED_LINEBREAK_RE = re.compile(
    rf"([{_WORD_CHARS}])-\s*\n\s*([{_WORD_CHARS}])"
)
_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)
_BULLET_OR_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?;:)\]\"']\s*$")
_LOWER_OR_CONTINUATION_RE = re.compile(rf"^\s*(?:[a-zà-öø-ÿ]|\(|,|;|:)")
_SECTION_HEADING_RE = re.compile(
    rf"^\s*(?:\d+(?:\.\d+)*\s+)?[{_WORD_CHARS}0-9][{_WORD_CHARS}0-9 ,:;()'\"/–—-]{{0,95}}$"
)
_FORMULAISH_RE = re.compile(
    r"(?:[=<>≤≥±×÷*/^_{}]|β|α|γ|δ|ε|λ|μ|σ|Σ|∑|√|≈|≠|≤|≥)"
)

_LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "st",
    "\ufb06": "st",
}


@dataclass(frozen=True)
class OcrNormalizationResult:
    text: str
    is_likely_scan: bool
    changed: bool
    score: int
    reasons: tuple[str, ...]
    original_chars: int
    normalized_chars: int


def analyze_ocr_artifacts(text: str) -> tuple[bool, int, tuple[str, ...]]:
    """Return ``(is_likely_scan, score, reasons)`` for OCR/scan-like text."""
    if not text or not text.strip():
        return False, 0, ()

    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    nonempty = [line for line in lines if line.strip()]
    if len(nonempty) < 6:
        return False, 0, ()

    score = 0
    reasons: list[str] = []

    hyphen_breaks = len(_HYPHENATED_LINEBREAK_RE.findall(text))
    if hyphen_breaks >= 2:
        score += 3
        reasons.append("hyphenated_line_breaks")
    elif hyphen_breaks == 1:
        score += 1
        reasons.append("single_hyphenated_line_break")

    lengths = [len(line.strip()) for line in nonempty]
    short_wrapped = [n for n in lengths if 18 <= n <= 95]
    if len(short_wrapped) / len(lengths) >= 0.65:
        score += 2
        reasons.append("line_wrapped_paragraphs")

    non_terminal = [
        line for line in nonempty
        if len(line.strip()) >= 20
        and not _SENTENCE_END_RE.search(line.strip())
        and not _BULLET_OR_LIST_RE.search(line)
    ]
    if len(non_terminal) / len(nonempty) >= 0.45:
        score += 2
        reasons.append("many_mid_sentence_line_breaks")

    page_number_lines = [line for line in nonempty if _PAGE_NUMBER_RE.match(line)]
    if len(page_number_lines) >= 2:
        score += 1
        reasons.append("standalone_page_numbers")

    control_chars = sum(1 for ch in text if unicodedata.category(ch) in {"Cc", "Cf"} and ch not in "\n\t")
    if control_chars:
        score += 1
        reasons.append("control_characters")

    return score >= 3, score, tuple(reasons)


def normalize_ocr_text(text: str, *, force: bool = False) -> OcrNormalizationResult:
    """Normalize OCR/scan artifacts when detected or ``force`` is true."""
    original = text or ""
    is_likely_scan, score, reasons = analyze_ocr_artifacts(original)

    normalized = _normalize_unicode(original)
    should_apply_ocr_pass = force or is_likely_scan
    if should_apply_ocr_pass:
        normalized = _remove_repeated_page_number_lines(normalized)
        normalized = _HYPHENATED_LINEBREAK_RE.sub(r"\1\2", normalized)
        normalized = _join_wrapped_lines(normalized)

    normalized = _normalize_spacing(normalized)

    return OcrNormalizationResult(
        text=normalized,
        is_likely_scan=is_likely_scan,
        changed=normalized != original,
        score=score,
        reasons=reasons,
        original_chars=len(original),
        normalized_chars=len(normalized),
    )


def _normalize_unicode(text: str) -> str:
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = text.replace("\x0c", "\n\n")
    text = remove_artifact_glyphs(text)
    return text


def _remove_repeated_page_number_lines(text: str) -> str:
    lines = text.split("\n")
    page_like = [line.strip() for line in lines if _PAGE_NUMBER_RE.match(line)]
    if len(page_like) < 2:
        return text
    return "\n".join("" if _PAGE_NUMBER_RE.match(line) else line for line in lines)


def _join_wrapped_lines(text: str) -> str:
    blocks = re.split(r"(\n{2,})", text)
    out: list[str] = []
    for block in blocks:
        if not block or block.startswith("\n"):
            out.append(block)
            continue

        lines = [line.strip() for line in block.split("\n")]
        nonempty = [line for line in lines if line]
        if len(nonempty) < 3 or _looks_like_list_block(nonempty):
            out.append(block)
            continue

        joined: list[str] = []
        current = nonempty[0]
        for line in nonempty[1:]:
            if _should_keep_line_break(current, line):
                joined.append(current.strip())
                current = line
            else:
                current = f"{current.rstrip()} {line.lstrip()}"
        joined.append(current.strip())
        out.append("\n\n".join(joined))
    return "".join(out)


def _looks_like_list_block(lines: list[str]) -> bool:
    if not lines:
        return False
    listish = sum(1 for line in lines if _BULLET_OR_LIST_RE.search(line))
    return listish / len(lines) >= 0.35


def _should_keep_line_break(previous: str, current: str) -> bool:
    prev = previous.strip()
    cur = current.strip()
    if not prev or not cur:
        return True
    if _BULLET_OR_LIST_RE.search(cur) or _BULLET_OR_LIST_RE.search(prev):
        return True
    if _looks_like_heading_line(prev) and not _LOWER_OR_CONTINUATION_RE.search(cur):
        return True
    if _looks_like_formula_line(prev) or _looks_like_formula_line(cur):
        return True
    if len(prev) < 45 and _SENTENCE_END_RE.search(prev) and not _LOWER_OR_CONTINUATION_RE.search(cur):
        return True
    if prev.endswith(":") and not _LOWER_OR_CONTINUATION_RE.search(cur):
        return True
    return False


def _looks_like_heading_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 96:
        return False
    if _SENTENCE_END_RE.search(line) or _FORMULAISH_RE.search(line):
        return False
    words = line.split()
    if not 1 <= len(words) <= 10:
        return False
    if not _SECTION_HEADING_RE.match(line):
        return False
    # All-uppercase headings, title-case headings, and numbered headings from
    # extracted PDFs should remain separate from the paragraph that follows.
    if re.match(r"^\d+(?:\.\d+)*\s+\S+", line):
        return True
    alpha_words = [w for w in words if re.search(rf"[{_WORD_CHARS}]", w)]
    if not alpha_words:
        return False
    uppercase_ratio = sum(1 for w in alpha_words if w[:1].isupper()) / len(alpha_words)
    return uppercase_ratio >= 0.5 or (len(words) <= 4 and line[:1].isupper())


def _looks_like_formula_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 180:
        return False
    if not _FORMULAISH_RE.search(line):
        return False
    digits_or_ops = sum(1 for ch in line if ch.isdigit() or ch in "=<>+-*/^_{}().,")
    return digits_or_ops >= 3


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
