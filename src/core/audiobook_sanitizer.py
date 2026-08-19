"""Audiobook-oriented final artifact sanitization.

This layer is intentionally profile-scoped.  It does not change the generic
translation/refinement pipeline; it creates companion artifacts for profiles
that explicitly opt into audiobook cleanup.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Mapping

from src.core.book_profiles.rendering import active_profile
from src.core.document_structure import DocumentBlockClassifier


@dataclass
class AudiobookSanitizationReport:
    captions_integrated: int = 0
    captions_to_appendix: int = 0
    note_blocks_to_appendix: int = 0
    critical_blocks_to_appendix: int = 0
    toc_blocks_to_appendix: int = 0
    junk_blocks_removed: int = 0
    links_removed: int = 0
    note_calls_removed: int = 0
    blocks_kept: int = 0
    structured_epub_preserved: bool = False
    images_preserved: int = 0
    image_placements_preserved: int = 0
    visual_captions_preserved: int = 0
    cover_preserved: bool = False
    spine_preserved: bool = False
    xhtml_preserved: bool = False
    epub_mimetype_valid: bool = False
    appendix_sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        parts = []
        if self.captions_integrated:
            parts.append(f"{self.captions_integrated} pie(s) de imagen integrados")
        if self.note_blocks_to_appendix or self.critical_blocks_to_appendix:
            parts.append(
                f"{self.note_blocks_to_appendix + self.critical_blocks_to_appendix} bloque(s) de notas/referencias al apendice"
            )
        if self.links_removed:
            parts.append(f"{self.links_removed} link(s) retirados")
        if self.note_calls_removed:
            parts.append(f"{self.note_calls_removed} llamada(s) de nota retiradas")
        if self.junk_blocks_removed:
            parts.append(f"{self.junk_blocks_removed} bloque(s) basura retirados")
        if self.structured_epub_preserved:
            parts.append(
                f"EPUB estructurado con portada y {self.images_preserved} imagen(es) preservadas"
            )
        return ", ".join(parts) or "sin cambios"


@dataclass
class AudiobookArtifact:
    text: str
    main_text: str
    appendix_text: str
    report: AudiobookSanitizationReport


_FIGURE_CAPTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"figure|figura|fig\.|image|imagen|photo|foto|photograph|fotografia|fotografía|"
    r"caption|pie\s+de\s+(?:foto|imagen)|illustration|ilustraci[oó]n|"
    r"l[aá]mina|still|frame|poster|production\s+still"
    r")\s*(?:\d+[A-Za-z]?|[IVXLCDM]+)?\s*[:.\-–—]\s*(?P<body>.+)$",
    re.IGNORECASE,
)
_FIGURE_CONTEXT_RE = re.compile(
    r"\b(?:figure|figura|fig\.|image|imagen|photo|foto|caption|pie de foto|"
    r"still|frame|poster|production still)\b",
    re.IGNORECASE,
)
_CREDIT_ONLY_RE = re.compile(
    r"\b(?:courtesy of|copyright|©|all rights reserved|source:|fuente:|"
    r"archivo de|credit|cr[eé]dito|photograph by|foto de|image courtesy)\b",
    re.IGNORECASE,
)
_JUNK_SOURCE_RE = re.compile(
    r"\b(?:oceanofpdf|z-?library|1lib\.sk|bookzz|pdfdrive|libgen|vk\.com|t\.me)\b",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_RAW_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_NOTE_MARKDOWN_CALL_RE = re.compile(r"\s*\[\[\d{1,4}\]\]\([^)]+\)")
_DOUBLE_BRACKET_NOTE_RE = re.compile(r"\s*\[\[\d{1,4}\]\]")
_SINGLE_BRACKET_NOTE_RE = re.compile(r"(?<!\w)\s*\[\d{1,4}\](?!\w)")
_FOOTNOTE_TARGET_RE = re.compile(r"\b(?:notas?|notes?|footnotes?|endnotes?)\b|#nt\d+", re.IGNORECASE)
_STANDALONE_NUMBER_RE = re.compile(r"^\s*\d{1,5}\.?\s*$")


def audiobook_profile_enabled(prompt_options: Mapping[str, Any] | None) -> bool:
    """Return True when the active profile explicitly asks for audiobook output."""
    profile = active_profile(dict(prompt_options or {}))
    if profile is None:
        return False
    return _audio_config_enabled(profile.raw_config)


def sanitize_for_audiobook(
    text: str,
    *,
    target_language: str = "Spanish",
    title: str = "",
) -> AudiobookArtifact:
    """Create a TTS-friendly companion text while preserving translated content."""
    classifier = DocumentBlockClassifier(source_type="audiobook")
    report = AudiobookSanitizationReport()
    main_blocks: list[str] = []
    appendix: list[tuple[str, str]] = []

    for block_text in _iter_blocks(text):
        lines = block_text.splitlines()
        block_type, policy, _confidence, _strategy, _notes = classifier.classify_block(lines)
        cleaned_block, removed_links, removed_calls = _clean_inline_audio_artifacts(block_text)
        report.links_removed += removed_links
        report.note_calls_removed += removed_calls
        cleaned_block = cleaned_block.strip()

        if not cleaned_block:
            if policy == "exclude" or block_type in {"junk_link", "watermark", "header_footer"}:
                report.junk_blocks_removed += 1
            continue

        if _is_junk_or_page_block(cleaned_block):
            report.junk_blocks_removed += 1
            continue

        if _is_caption_block(cleaned_block, block_type=block_type):
            caption = _caption_body(cleaned_block)
            if _caption_should_be_integrated(caption):
                main_blocks.append(_render_caption_for_audio(caption, target_language=target_language))
                report.captions_integrated += 1
            else:
                appendix.append(("Pies de imagen y creditos", caption))
                report.captions_to_appendix += 1
            continue

        if block_type == "note":
            appendix.append(("Notas", cleaned_block))
            report.note_blocks_to_appendix += 1
            continue

        if block_type == "critical_apparatus":
            appendix.append(("Referencias", cleaned_block))
            report.critical_blocks_to_appendix += 1
            continue

        if block_type == "toc":
            appendix.append(("Indice", cleaned_block))
            report.toc_blocks_to_appendix += 1
            continue

        main_blocks.append(cleaned_block)
        report.blocks_kept += 1

    main_text = _join_blocks(main_blocks)
    appendix_text = _render_appendix(appendix, target_language=target_language)
    if appendix_text:
        report.appendix_sections = sorted({section for section, _value in appendix})
    full_text = _join_blocks([main_text, appendix_text] if appendix_text else [main_text])
    return AudiobookArtifact(
        text=full_text,
        main_text=main_text,
        appendix_text=appendix_text,
        report=report,
    )


def _audio_config_enabled(config: Mapping[str, Any]) -> bool:
    audio = config.get("audiobook") or config.get("audio_sanitization") or {}
    if isinstance(audio, Mapping):
        return bool(audio.get("enabled") or audio.get("generate_companion"))
    return bool(audio)


def _iter_blocks(text: str) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]
    if blocks:
        return blocks
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def _clean_inline_audio_artifacts(text: str) -> tuple[str, int, int]:
    value = text or ""
    links_removed = 0
    note_calls_removed = 0

    value, count = _NOTE_MARKDOWN_CALL_RE.subn("", value)
    note_calls_removed += count
    value, count = _DOUBLE_BRACKET_NOTE_RE.subn("", value)
    note_calls_removed += count
    value, count = _SINGLE_BRACKET_NOTE_RE.subn("", value)
    note_calls_removed += count

    def markdown_link_repl(match: re.Match) -> str:
        nonlocal links_removed
        label = (match.group(1) or "").strip()
        target = (match.group(2) or "").strip()
        links_removed += 1
        if _JUNK_SOURCE_RE.search(label) or _JUNK_SOURCE_RE.search(target):
            return ""
        if _FOOTNOTE_TARGET_RE.search(target):
            return label if not re.fullmatch(r"\[?\d{1,4}\]?", label) else ""
        return label

    value = _MARKDOWN_LINK_RE.sub(markdown_link_repl, value)

    def raw_url_repl(match: re.Match) -> str:
        nonlocal links_removed
        links_removed += 1
        return ""

    value = _RAW_URL_RE.sub(raw_url_repl, value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\(\s*\)", "", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip(), links_removed, note_calls_removed


def _is_junk_or_page_block(text: str) -> bool:
    stripped = re.sub(r"\s+", " ", text or "").strip()
    if not stripped:
        return True
    if _JUNK_SOURCE_RE.search(stripped):
        return True
    if _STANDALONE_NUMBER_RE.match(stripped):
        return True
    return False


def _is_caption_block(text: str, *, block_type: str) -> bool:
    if block_type == "figure_text":
        return True
    first_line = next((line.strip() for line in (text or "").splitlines() if line.strip()), "")
    if _FIGURE_CAPTION_PREFIX_RE.match(first_line):
        return True
    return bool(len(first_line) <= 260 and _FIGURE_CONTEXT_RE.search(first_line))


def _caption_body(text: str) -> str:
    value = re.sub(r"\s+", " ", text or "").strip()
    match = _FIGURE_CAPTION_PREFIX_RE.match(value)
    if match:
        value = (match.group("body") or "").strip()
    return value.strip(" -–—")


def _caption_should_be_integrated(caption: str) -> bool:
    text = (caption or "").strip()
    if len(text) < 18:
        return False
    if _CREDIT_ONLY_RE.search(text) and len(re.findall(r"\w+", text)) < 18:
        return False
    if _JUNK_SOURCE_RE.search(text):
        return False
    return True


def _render_caption_for_audio(caption: str, *, target_language: str) -> str:
    caption = caption.strip()
    if not caption:
        return ""
    if re.search(r"[.!?…]$", caption):
        sentence = caption
    else:
        sentence = f"{caption}."
    if str(target_language or "").casefold().startswith("spanish"):
        return f"Descripcion de imagen: {sentence}"
    return f"Image description: {sentence}"


def _render_appendix(items: list[tuple[str, str]], *, target_language: str) -> str:
    if not items:
        return ""
    if str(target_language or "").casefold().startswith("spanish"):
        heading = "Apendice de notas, referencias y creditos visuales"
    else:
        heading = "Appendix of notes, references, and visual credits"
    grouped: dict[str, list[str]] = {}
    for section, value in items:
        cleaned = value.strip()
        if cleaned:
            grouped.setdefault(section, []).append(cleaned)
    parts = [heading]
    for section in sorted(grouped):
        parts.append(section)
        parts.extend(_dedupe(grouped[section]))
    return _join_blocks(parts)


def _join_blocks(blocks: list[str]) -> str:
    return "\n\n".join(block.strip() for block in blocks if block and block.strip()).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value).casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
