"""Document-structure normalization for text-first pipelines.

This module keeps structural cleanup separate from book glossaries.  Glossaries
answer lexical/editorial questions; this layer answers layout questions before
chunking: "is this block a table/figure/formula, and how do we make it stable
enough for the LLM to preserve?"
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Iterable, Literal


DOCUMENT_STRUCTURE_VERSION = "document-structure-v1"

BlockType = Literal[
    "narrative",
    "title",
    "toc",
    "glossary",
    "note",
    "table",
    "formula",
    "critical_apparatus",
    "header_footer",
    "watermark",
    "junk_link",
    "figure_text",
]
BlockPolicy = Literal["translate", "preserve", "clean", "reconstruct", "exclude"]

_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:table|tabla|cuadro)\s+(\d+[A-Za-z]?)\s*[:.\-–—]\s*(.*)$",
    re.IGNORECASE,
)
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(?:figure|figura|fig\.)\s+\d+[A-Za-z]?\s*[:.\-–—]",
    re.IGNORECASE,
)
_SECTION_HEADING_RE = re.compile(
    r"^\s*\d+(?:\.\d+){0,3}\s+[A-ZÁÉÍÓÚÑ][^\n]{2,90}$"
)
_MATH_CELL_RE = re.compile(
    r"O\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)"
    r"|(?:\d+(?:\.\d+)?\s*[·×x]\s*10(?:\^?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+))"
    r"|(?<![\[\dA-Za-z])(?:\d+(?:\.\d+)?(?:K|M|B|%)?)(?![\]A-Za-z])",
    re.IGNORECASE,
)
_SPACED_DECIMAL_RE = re.compile(r"(?<=\d)\.\s+(?=\d)")
_SPACED_SCI_RE = re.compile(r"([·×x]\s*)10([+-]?\d{1,3})\b")
_SCI_WITH_SPACED_DECIMAL_RE = re.compile(
    r"(\d)\.\s+(\d)(\s*[·×x]\s*)10\s*([+-]?\d{1,3})\b"
)
_PIPE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_PUNCT_NO_SPACE_BEFORE = {".", ",", ";", ":", "?", "!", ")", "]", "}", "%"}
_PUNCT_NO_SPACE_AFTER = {"(", "[", "{", "¿", "¡"}
_FIGURE_TOKEN_SPECIALS = {"<eos>", "<pad>", "<s>", "</s>", "<unk>"}
_RAW_URL_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)
_MARKDOWN_LINK_ONLY_RE = re.compile(r"^\s*\[([^\]]+)\]\(([^)]+)\)\s*$")
_JUNK_LINK_RE = re.compile(
    r"\b(?:oceanofpdf|z-?library|1lib\.sk|bookzz|pdfdrive|libgen|vk\.com|t\.me)\b",
    re.IGNORECASE,
)
_EXPLICIT_PAGE_MARKER_RE = re.compile(
    r"^\s*(?:page|pagina|p[aá]gina|pag\.?|p\.)\s*"
    r"(?:\d{1,5}|[ivxlcdm]{1,10})"
    r"(?:\s*(?:of|de|/)\s*(?:\d{1,5}|[ivxlcdm]{1,10}))?\s*$",
    re.IGNORECASE,
)
_PAGE_COUNT_MARKER_RE = re.compile(
    r"^\s*(?:[-\u2013\u2014]\s*)?(?:\d{1,5}|[ivxlcdm]{1,10})"
    r"\s*(?:of|de|/)\s*(?:\d{1,5}|[ivxlcdm]{1,10})"
    r"(?:\s*[-\u2013\u2014])?\s*$",
    re.IGNORECASE,
)
_TOC_TITLE_RE = re.compile(r"^\s*(?:contents|contenido|indice|índice|table of contents)\s*$", re.IGNORECASE)
_TOC_ENTRY_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+){0,4}\s+)?[^\n]{2,120}?"
    r"(?:\s*\.{3,}\s*|\s{2,})(?:\d{1,5}|[ivxlcdm]{1,10})\s*$",
    re.IGNORECASE,
)
_GLOSSARY_HEADING_RE = re.compile(
    r"^\s*(?:glossar(?:y|ies)|glosario|lexicon|l[eé]xico|"
    r"pronunciation\s+key|clave\s+de\s+pronunciaci[oó]n)\s*$",
    re.IGNORECASE,
)
_PRONUNCIATION_MAPPING_RE = re.compile(
    r"\b[a-zà-öø-ÿ]{1,8}(?:[´'’\-][a-zà-öø-ÿ]{1,12}){0,5}\s+"
    r"(?:as\s+in|como\s+en)\s+[a-zà-öø-ÿ]{1,30}\b",
    re.IGNORECASE,
)
_GLOSSARY_PHONETIC_ENTRY_RE = re.compile(
    r"(?<!\w)[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ’'\-]{1,48}\s+"
    r"[a-zà-öø-ÿ]{1,12}(?:[´'’\-]+[a-zà-öø-ÿ]{1,16}){1,8}"
    r"\s*\)?\s*:",
)
_GLOSSARY_LOCATOR_RE = re.compile(r"\b(?:[1-9]|1\d|2[0-4])\.\d{1,4}\.")
_NOTE_RE = re.compile(r"^\s*(?:\[\d{1,4}\]|\d{1,4}\.|nota\s+\d{1,4}|note\s+\d{1,4})\s+", re.IGNORECASE)
_CRITICAL_APPARATUS_RE = re.compile(
    r"\b(?:ibid\.|op\. cit\.|doi:|isbn|issn|bibliograf[ií]a|references|works cited|"
    r"ed\.|vol\.|pp?\.\s*\d|cf\.)\b",
    re.IGNORECASE,
)
_CRITICAL_APPARATUS_HEADING_RE = re.compile(
    r"^\s*(?:notes|endnotes|footnotes|references|works cited|bibliograph(?:y|ies)|"
    r"notas|notas finales|notas al pie|referencias|obras citadas|bibliograf[ií]a)\b",
    re.IGNORECASE,
)
_CITATION_LINK_RE = re.compile(
    r"(?ix)"
    r"(?:https?://|www\.|doi(?:\.org|:))\S+"
    r"|(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,24}(?:/[^\s<>\[\]]*)?"
)
_CITATION_DATE_RE = re.compile(
    r"(?ix)"
    r"\((?:"
    r"(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|enero|febrero|marzo|abril|mayo|junio|"
    r"julio|agosto|septiembre|octubre|noviembre|diciembre)\s+"
    r")?\d{4}[a-z]?\)"
    r"|\b(?:"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|enero|febrero|marzo|abril|mayo|junio|"
    r"julio|agosto|septiembre|octubre|noviembre|diciembre"
    r")\s+\d{1,2},?\s+\d{4}\b"
)
_NOTE_BACKLINK_RE = re.compile(
    r"\b(?:go to note reference in text|return to note reference|"
    r"volver a la referencia de la nota|ir a la referencia de la nota)\b",
    re.IGNORECASE,
)
_PUBLISHER_YEAR_CITATION_RE = re.compile(
    r":\s*[^.;\n]{2,100},\s*(?:1[5-9]\d{2}|20\d{2})\b",
    re.IGNORECASE,
)
_BIBLIOGRAPHIC_TERMINAL_YEAR_RE = re.compile(
    r"(?:1[5-9]\d{2}|20\d{2})"
    r"(?:\s*[-\u2013\u2014]\s*(?:\d{2}|1[5-9]\d{2}|20\d{2}))?"
    r"[.)]?\s*$"
)
_BIBLIOGRAPHIC_LEAD_CONNECTORS = {
    "and",
    "d",
    "de",
    "del",
    "der",
    "des",
    "di",
    "du",
    "et",
    "of",
    "the",
    "van",
    "von",
    "y",
}
_INDEX_IDENTITY_QUALIFIERS = {
    "abbot", "admiral", "archbishop", "archduke", "baron", "baroness",
    "bishop", "captain", "colonel", "commander", "corporal", "count",
    "countess", "dauphin", "doctor", "dr", "duchess", "duke", "earl",
    "emperor", "empress", "father", "field", "general", "king", "lady",
    "lieutenant", "lord", "major", "marshal", "midshipman", "mother",
    "mr", "mrs", "ms", "president", "prince", "princess", "professor",
    "queen", "reverend", "saint", "sergeant", "sir", "staff", "surgeon",
    "tsar", "vice",
}
_INDEX_LOCATOR_TRAILER_RE = re.compile(
    r"(?P<locators>\d{1,4}(?:\s*,\s*\d{1,4})*)[.)]?\s*$"
)
_FORMULA_ONLY_RE = re.compile(
    r"^\s*(?:[$]{1,2})?[\w\s{}()[\].,+\-*/^_=<>≤≥∑∫√πα-ωΑ-Ω·×%]+(?:[$]{1,2})?\s*$"
)
_TITLE_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+")


def is_comma_delimited_bibliographic_record(value: str) -> bool:
    """Recognize legacy author/title/publisher/year citation records.

    Older EPUBs often encode bibliographies as comma-delimited paragraphs
    rather than modern ``Place: Publisher, Year`` records. Requiring a colon
    makes those lists look like numeric tables and leaves identity-only rows in
    an impossible translation loop. The lead field must still look like a name
    or published title so ordinary prose ending in a year is excluded.
    """
    line = " ".join(str(value or "").split())
    if (
        not line
        or len(line) > 600
        or not _BIBLIOGRAPHIC_TERMINAL_YEAR_RE.search(line)
    ):
        return False

    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 3 or any(not field for field in fields[:2]):
        return False

    lead = re.sub(r"[’']s\b", "", fields[0], flags=re.IGNORECASE)
    lead_words = _TITLE_WORD_RE.findall(lead)
    if not lead_words or len(lead_words) > 18:
        return False
    if any(
        word[:1].islower()
        and word.casefold() not in _BIBLIOGRAPHIC_LEAD_CONNECTORS
        for word in lead_words
    ):
        return False

    metadata_words = _TITLE_WORD_RE.findall(" ".join(fields[:-1]))
    return len(metadata_words) >= 3


def is_locator_index_entry(value: str) -> bool:
    """Return whether one entry contains only an identity and page locators."""
    line = " ".join(str(value or "").split())
    locator_match = _INDEX_LOCATOR_TRAILER_RE.search(line)
    if not locator_match:
        return False
    raw_prefix = line[:locator_match.start()].rstrip()
    prefix = raw_prefix.rstrip(" ,")
    words = _TITLE_WORD_RE.findall(prefix)
    if not words:
        return False
    if "," not in prefix and len(words) < 2:
        # OCR often writes regnal entries as ``Elizabeth 1,149,156``. The
        # first number is consumed by the locator expression, so retain the
        # entry only when there are multiple numeric fields and no comma after
        # the single identity word. ``Canterbury, 96`` remains too ambiguous.
        locators = locator_match.group("locators")
        if raw_prefix.endswith(",") or "," not in locators:
            return False
    title_words = sum(
        1
        for word in words
        if word[:1].isupper()
        or word.casefold() in _BIBLIOGRAPHIC_LEAD_CONNECTORS
        or word.casefold() in _INDEX_IDENTITY_QUALIFIERS
        or len(word) == 1
        or bool(re.fullmatch(r"[ivxlcdm]+", word, re.IGNORECASE))
    )
    return title_words >= max(1, math.ceil(len(words) * 0.65))


def is_locator_index_identity_fragment(value: str) -> bool:
    """Recognize split or flattened identity data in an analytical index."""
    line = " ".join(str(value or "").split())
    if not line or len(line) > 10000:
        return False
    if is_locator_index_entry(line):
        return True
    if "," not in line or re.search(r"[!?;:]", line):
        return False

    words = _TITLE_WORD_RE.findall(line)
    if not words:
        return False
    apostrophe_name_suffixes = {
        match.group(1).casefold()
        for match in re.finditer(
            r"[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+[’']"
            r"([a-záéíóúüñ]+)",
            line,
        )
    }
    missing_initial_names = {
        match.group(1).casefold()
        for match in re.finditer(
            r"(?:^|\s)\.\s+([a-záéíóúüñ]+)\s*,",
            line,
        )
    }
    missing_leading_name_tokens = {
        match.group(1).casefold()
        for pattern in (
            (
                r"(?:^|\d[\d,]*)\s+([a-záéíóúüñ]+)\s*,\s+"
                r"(?=(?:the\s+)?[A-ZÁÉÍÓÚÜÑ])"
            ),
            (
                r"(?:^|\d[\d,]*)\s+([a-záéíóúüñ]+)\s+"
                r"(?:[IVXLCDM]+|\d+)\s*,\s*\d"
            ),
        )
        for match in re.finditer(pattern, line)
    }
    ordinary_lowercase = [
        word
        for word in words
        if word[:1].islower()
        and word.casefold() not in _BIBLIOGRAPHIC_LEAD_CONNECTORS
        and word.casefold() not in _INDEX_IDENTITY_QUALIFIERS
        and word.casefold() not in apostrophe_name_suffixes
        and word.casefold() not in missing_initial_names
        and word.casefold() not in missing_leading_name_tokens
        and len(word) > 1
        and not re.fullmatch(r"[ivxlcdm]+", word, re.IGNORECASE)
    ]
    if ordinary_lowercase:
        return False
    return (
        bool(missing_initial_names or missing_leading_name_tokens)
        or any(word[:1].isupper() for word in words)
    )


def is_locator_index_identity_block(lines: Iterable[str]) -> bool:
    """Return whether a structured block is identity-only index material."""
    entries = [" ".join(str(line or "").split()) for line in lines]
    entries = [line for line in entries if line]
    if len(entries) < 3:
        return False
    locator_entries = sum(1 for line in entries if is_locator_index_entry(line))
    required = max(3, math.ceil(len(entries) * 0.60))
    return (
        locator_entries >= required
        and all(is_locator_index_identity_fragment(line) for line in entries)
    )


def is_locator_index_block(lines: Iterable[str]) -> bool:
    """Recognize flattened analytical-index entries by their page locators."""
    entries = [" ".join(str(line or "").split()) for line in lines]
    entries = [line for line in entries if line]
    if len(entries) < 3:
        return False

    name_shaped = sum(1 for line in entries if is_locator_index_entry(line))
    required = max(3, math.ceil(len(entries) * 0.60))
    return name_shaped >= required


@dataclass
class StructureBlock:
    block_id: str
    type: BlockType
    start_line: int
    end_line: int
    confidence: float
    title: str = ""
    strategy: str = ""
    rows: int = 0
    notes: list[str] = field(default_factory=list)
    policy: BlockPolicy = "translate"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentStructureReport:
    source_type: str = "text"
    version: str = DOCUMENT_STRUCTURE_VERSION
    tables_detected: int = 0
    figure_text_blocks_detected: int = 0
    formulas_detected: int = 0
    repairs_applied: int = 0
    excluded_blocks: int = 0
    cleaned_blocks: int = 0
    preserved_blocks: int = 0
    reconstructed_blocks: int = 0
    translatable_blocks: int = 0
    block_type_counts: dict[str, int] = field(default_factory=dict)
    policy_counts: dict[str, int] = field(default_factory=dict)
    blocks: list[StructureBlock] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["blocks"] = [block.to_dict() for block in self.blocks]
        return data

    @property
    def has_structured_blocks(self) -> bool:
        return bool(
            self.tables_detected
            or self.figure_text_blocks_detected
            or self.excluded_blocks
            or self.cleaned_blocks
            or self.preserved_blocks
            or self.reconstructed_blocks
        )

    def add_blocks(self, blocks: Iterable[StructureBlock]) -> None:
        for block in blocks:
            self.blocks.append(block)
            self.block_type_counts = _increment_count(self.block_type_counts, block.type)
            self.policy_counts = _increment_count(self.policy_counts, block.policy)
            if block.policy == "exclude":
                self.excluded_blocks += 1
            elif block.policy == "clean":
                self.cleaned_blocks += 1
            elif block.policy == "preserve":
                self.preserved_blocks += 1
            elif block.policy == "reconstruct":
                self.reconstructed_blocks += 1
            elif block.policy == "translate":
                self.translatable_blocks += 1


@dataclass
class DocumentIR:
    """Intermediate representation for structure-aware document workflows."""

    text: str
    source_type: str = "text"
    blocks: list[StructureBlock] = field(default_factory=list)
    report: DocumentStructureReport = field(default_factory=DocumentStructureReport)

    @classmethod
    def from_text(cls, text: str, *, source_type: str = "text") -> "DocumentIR":
        cleaned, report = normalize_document_structure(text, source_type=source_type)
        return cls(
            text=cleaned,
            source_type=source_type,
            blocks=list(report.blocks),
            report=report,
        )

    @property
    def llm_text(self) -> str:
        return self.text

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "text": self.text,
            "blocks": [block.to_dict() for block in self.blocks],
            "report": self.report.to_dict(),
        }


class DocumentBlockClassifier:
    """Classify sanitized document blocks before chunking/LLM calls."""

    def __init__(self, *, source_type: str = "text"):
        self.source_type = source_type

    def classify_text(self, text: str) -> list[StructureBlock]:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
        blocks: list[StructureBlock] = []
        block_index = 0
        for start, end, block_lines in _iter_line_blocks(lines):
            block_index += 1
            block_type, policy, confidence, strategy, notes = self.classify_block(block_lines)
            blocks.append(
                StructureBlock(
                    block_id=f"blk_{block_index:04d}",
                    type=block_type,
                    start_line=start,
                    end_line=end,
                    confidence=confidence,
                    title=_block_title(block_lines),
                    strategy=strategy,
                    policy=policy,
                    notes=notes,
                )
            )
        return blocks

    def apply_policies(self, text: str) -> tuple[str, list[StructureBlock]]:
        lines = (text or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()
        blocks = self.classify_text(text)
        output: list[str] = []
        block_by_start = {block.start_line: block for block in blocks}
        span_by_start = {
            start: (end, block_lines)
            for start, end, block_lines in _iter_line_blocks(lines)
        }
        line_number = 1
        while line_number <= len(lines):
            block = block_by_start.get(line_number)
            if block is None:
                output.append(lines[line_number - 1])
                line_number += 1
                continue
            end, block_lines = span_by_start[line_number]
            rendered = self.render_block(block, block_lines)
            if rendered:
                if output and output[-1].strip():
                    output.append("")
                output.extend(rendered)
            line_number = end + 1
        cleaned = "\n".join(output)
        cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()
        return cleaned, blocks

    def classify_block(
        self,
        block_lines: list[str],
    ) -> tuple[BlockType, BlockPolicy, float, str, list[str]]:
        text = "\n".join(line.strip() for line in block_lines if line.strip())
        lines = [line.strip() for line in block_lines if line.strip()]
        if not text:
            return "narrative", "translate", 0.0, "blank", []
        if _is_junk_link_block(text, lines):
            block_type = "watermark" if _JUNK_LINK_RE.search(text) else "junk_link"
            return block_type, "exclude", 0.98, "remove_source_link_or_watermark", []
        if _is_clean_link_block(lines):
            return "junk_link", "clean", 0.82, "strip_link_target_keep_label", []
        if len(lines) == 1 and _is_header_footer_block(lines[0]):
            return "header_footer", "exclude", 0.92, "remove_standalone_page_marker", []
        if _is_note_block(lines):
            return "note", "translate", 0.78, "translate_note_content_preserve_marker", []
        # Notes often contain several years, page ranges and durations on each
        # line. Those numbers resemble table cells, but citation semantics are
        # more specific and must win so language gates can distinguish
        # identity-bearing titles from untranslated narrative prose.
        if _is_critical_apparatus_block(text, lines):
            return "critical_apparatus", "preserve", 0.74, "preserve_bibliographic_apparatus", []
        if _is_glossary_block(text, lines):
            return (
                "glossary",
                "translate",
                0.90,
                "translate_definitions_preserve_lexical_structure",
                [],
            )
        if _is_table_block(lines):
            return "table", "reconstruct", 0.88, "normalize_or_preserve_table_grid", []
        if _is_formula_block(text, lines):
            return "formula", "preserve", 0.86, "preserve_formula", []
        if _is_toc_block(lines):
            return "toc", "reconstruct", 0.84, "reconstruct_index_or_toc", []
        if _is_title_block(lines):
            return "title", "translate", 0.78, "translate_heading", []
        return "narrative", "translate", 0.64, "translate_body", []

    def render_block(self, block: StructureBlock, block_lines: list[str]) -> list[str]:
        if block.policy == "exclude":
            return []
        if block.policy == "clean":
            return [_clean_block_line(line) for line in block_lines if _clean_block_line(line)]
        return block_lines


def _increment_count(counts: dict[str, int], key: str) -> dict[str, int]:
    updated = dict(counts)
    updated[str(key)] = updated.get(str(key), 0) + 1
    return updated


def _iter_line_blocks(lines: list[str]) -> Iterable[tuple[int, int, list[str]]]:
    start: int | None = None
    buffer: list[str] = []
    for index, line in enumerate(lines, start=1):
        if line.strip():
            if start is None:
                start = index
            buffer.append(line)
            continue
        if start is not None:
            yield start, index - 1, buffer
            start = None
            buffer = []
    if start is not None:
        yield start, len(lines), buffer


def _block_title(lines: list[str], *, limit: int = 100) -> str:
    text = " ".join(line.strip() for line in lines if line.strip())
    return text[:limit].strip()


def _is_junk_link_block(text: str, lines: list[str]) -> bool:
    if not lines:
        return False
    if _JUNK_LINK_RE.search(text) and _is_link_or_watermark_shaped(text, lines):
        return True
    if all(_RAW_URL_RE.match(line) for line in lines):
        return True
    return False


def _is_link_or_watermark_shaped(text: str, lines: list[str]) -> bool:
    if all(
        _RAW_URL_RE.match(line)
        or _MARKDOWN_LINK_ONLY_RE.match(line)
        or _JUNK_LINK_RE.search(line)
        for line in lines
    ):
        return True
    words = _TITLE_WORD_RE.findall(text)
    return len(lines) <= 2 and len(words) <= 10 and not re.search(r"[.!?]\s+\w", text)


def _is_clean_link_block(lines: list[str]) -> bool:
    if not lines:
        return False
    return all(_MARKDOWN_LINK_ONLY_RE.match(line) for line in lines)


def _is_header_footer_block(line: str) -> bool:
    value = (line or "").strip()
    return bool(_EXPLICIT_PAGE_MARKER_RE.match(value) or _PAGE_COUNT_MARKER_RE.match(value))


def _is_table_block(lines: list[str]) -> bool:
    if not lines:
        return False
    if any(_TABLE_CAPTION_RE.match(line) for line in lines):
        return True
    pipe_rows = sum(1 for line in lines if _PIPE_ROW_RE.match(line))
    if pipe_rows >= 2:
        return True
    numeric_rows = sum(1 for line in lines if len(_MATH_CELL_RE.findall(line)) >= 2)
    return len(lines) >= 3 and numeric_rows >= 2


def _is_formula_block(text: str, lines: list[str]) -> bool:
    if len(lines) > 4:
        return False
    if "$$" in text or re.search(r"\\(?:frac|sum|int|sqrt|begin|end)\b", text):
        return True
    math_chars = sum(1 for ch in text if ch in "=<>+-*/^_{}≤≥∑∫√π")
    alpha_words = _TITLE_WORD_RE.findall(text)
    if math_chars >= 2 and len(alpha_words) <= 8 and _FORMULA_ONLY_RE.match(text):
        return True
    return False


def _is_toc_block(lines: list[str]) -> bool:
    if not lines:
        return False
    if len(lines) == 1 and _TOC_TITLE_RE.match(lines[0]):
        return True
    toc_entries = sum(1 for line in lines if _TOC_ENTRY_RE.match(line))
    return toc_entries >= 2 and toc_entries / max(1, len(lines)) >= 0.5


def _is_note_block(lines: list[str]) -> bool:
    if not lines:
        return False
    if _NOTE_RE.match(lines[0]):
        return True
    return len(lines) <= 3 and bool(re.match(r"^\s*[*†‡]\s+", lines[0]))


def _is_glossary_block(text: str, lines: list[str]) -> bool:
    """Recognize lexical reference blocks without relying on a book profile.

    EPUBs frequently flatten an entire glossary page into one chunk. A title,
    a dense pronunciation key, or repeated headword/pronunciation/locator
    records is stronger evidence than any individual retained source word.
    """
    if not lines:
        return False
    if any(_GLOSSARY_HEADING_RE.match(line) for line in lines[:2]):
        return True

    pronunciation_mappings = len(_PRONUNCIATION_MAPPING_RE.findall(text))
    if pronunciation_mappings >= 3:
        return True

    phonetic_entries = len(_GLOSSARY_PHONETIC_ENTRY_RE.findall(text))
    locators = len(_GLOSSARY_LOCATOR_RE.findall(text))
    if phonetic_entries >= 3 and locators >= 2:
        return True

    lexical_lines = 0
    for line in lines:
        if ":" not in line:
            continue
        headword, definition = line.split(":", 1)
        words = _TITLE_WORD_RE.findall(headword)
        if (
            definition.strip()
            and 1 <= len(words) <= 5
            and len(headword) <= 60
            and not re.search(r"[,.!?;]", headword)
        ):
            lexical_lines += 1
    return lexical_lines >= 3 and lexical_lines / max(1, len(lines)) >= 0.55


def _is_critical_apparatus_block(text: str, lines: list[str]) -> bool:
    if not lines:
        return False
    if _CRITICAL_APPARATUS_RE.search(text):
        return True

    heading = bool(_CRITICAL_APPARATUS_HEADING_RE.match(text))
    link_count = len(_CITATION_LINK_RE.findall(text))
    date_count = len(_CITATION_DATE_RE.findall(text))
    backlink_count = len(_NOTE_BACKLINK_RE.findall(text))
    quoted_title_count = len(re.findall(r"[“\"]([^”\"]{8,220})[”\"]", text))
    publisher_year_count = len(_PUBLISHER_YEAR_CITATION_RE.findall(text))
    comma_record_count = sum(
        1 for line in lines if is_comma_delimited_bibliographic_record(line)
    )
    mixed_record_count = sum(
        1
        for line in lines
        if _PUBLISHER_YEAR_CITATION_RE.search(line)
        or is_comma_delimited_bibliographic_record(line)
    )

    # EPUB note sections are often flattened into one placeholder-delimited
    # line before the language gate sees them. Use combined citation signals
    # instead of depending on line layout or a literal ``doi:`` spelling.
    if heading and (link_count >= 1 or date_count >= 2 or backlink_count >= 1):
        return True
    if backlink_count >= 1 and link_count >= 1 and date_count >= 1:
        return True
    if backlink_count >= 2 and (link_count >= 1 or date_count >= 1):
        return True
    if link_count >= 2 and date_count >= 2 and quoted_title_count >= 1:
        return True
    # Reading lists frequently contain only author/title/place/publisher/year
    # records. They may have no URLs, DOI labels, quoted titles or parenthesized
    # dates, so recognize a repeated publication-record shape directly.
    if (
        publisher_year_count >= 2
        or comma_record_count >= 2
        or mixed_record_count >= 2
    ):
        return True

    citation_density = len(re.findall(r"\[[^\]]{1,40}\]|\(\d{4}[a-z]?\)", text))
    return citation_density >= 3 and len(lines) <= 8


def _is_title_block(lines: list[str]) -> bool:
    if len(lines) != 1:
        return False
    line = lines[0].strip()
    if not line or len(line) > 120:
        return False
    words = _TITLE_WORD_RE.findall(line)
    if not 1 <= len(words) <= 14:
        return False
    if _TOC_ENTRY_RE.match(line) or _NOTE_RE.match(line):
        return False
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    titlecase_words = sum(1 for word in words if word[:1].isupper())
    return uppercase_ratio >= 0.55 or titlecase_words >= max(1, len(words) - 1)


def _clean_block_line(line: str) -> str:
    value = _MARKDOWN_LINK_ONLY_RE.sub(lambda match: match.group(1), line or "")
    value = _RAW_URL_RE.sub("", value)
    return value.strip()


def normalize_document_structure(
    text: str,
    *,
    source_type: str = "text",
) -> tuple[str, DocumentStructureReport]:
    """Normalize structural artifacts before LLM chunking.

    The function is intentionally deterministic.  It does not guess missing
    content and it does not apply book-specific editorial rules.
    """
    report = DocumentStructureReport(source_type=source_type)
    value, repairs = repair_structural_artifacts(text or "")
    report.repairs_applied += repairs
    classifier = DocumentBlockClassifier(source_type=source_type)
    value, classified_blocks = classifier.apply_policies(value)
    report.add_blocks(classified_blocks)

    lines = value.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    lines, figure_blocks = _compact_visual_token_lines(lines)
    for block in figure_blocks:
        block.block_id = f"figtxt_{len(report.blocks) + 1:03d}"
        block.policy = "reconstruct"
        report.add_blocks([block])
    report.figure_text_blocks_detected += len(figure_blocks)

    lines, table_blocks = _normalize_table_blocks(lines)
    existing_table_titles = {block.title for block in report.blocks if block.type == "table"}
    for block in table_blocks:
        block.block_id = f"verbaloom_{len([b for b in report.blocks if b.type == 'table']) + 1:03d}"
        block.policy = "reconstruct"
        if not any(
            existing.startswith(block.title) or block.title.startswith(existing)
            for existing in existing_table_titles
        ):
            report.add_blocks([block])
    report.tables_detected = max(
        len(table_blocks),
        int(report.block_type_counts.get("table") or 0),
    )
    report.formulas_detected = max(
        int(report.block_type_counts.get("formula") or 0),
        len(re.findall(r"(?:\bO\([^)]*\)|[=<>+\-*/^_{}]|[α-ωΑ-Ω])", value)),
    )

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{4,}", "\n\n\n", normalized).strip()
    return normalized, report


def repair_structural_artifacts(text: str) -> tuple[str, int]:
    """Repair mechanical artifacts common in tables, formulas, and citations."""
    value = text or ""
    repairs = 0

    value, count = _SPACED_DECIMAL_RE.subn(".", value)
    repairs += count

    def sci_decimal_repl(match: re.Match) -> str:
        return f"{match.group(1)}.{match.group(2)}{match.group(3)}10^{match.group(4)}"

    value, count = _SCI_WITH_SPACED_DECIMAL_RE.subn(sci_decimal_repl, value)
    repairs += count

    def sci_repl(match: re.Match) -> str:
        return f"{match.group(1)}10^{match.group(2)}"

    value, count = _SPACED_SCI_RE.subn(sci_repl, value)
    repairs += count
    return value, repairs


def _normalize_table_blocks(lines: list[str]) -> tuple[list[str], list[StructureBlock]]:
    output: list[str] = []
    blocks: list[StructureBlock] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _TABLE_CAPTION_RE.match(line):
            output.append(line)
            index += 1
            continue

        next_nonblank = _next_nonblank(lines, index + 1)
        if next_nonblank is not None and _PIPE_ROW_RE.match(lines[next_nonblank]):
            output.append(line)
            index += 1
            continue

        start = index
        raw_block = [line]
        index += 1
        blank_seen = False
        while index < len(lines):
            candidate = lines[index]
            stripped = candidate.strip()
            if not stripped:
                if blank_seen:
                    break
                blank_seen = True
                raw_block.append(candidate)
                index += 1
                continue
            if _should_stop_table_block(stripped, raw_block):
                break
            blank_seen = False
            raw_block.append(candidate)
            index += 1

        rendered, strategy, row_count = _render_table_block(raw_block)
        output.extend(rendered)
        blocks.append(
            StructureBlock(
                block_id="",
                type="table",
                start_line=start + 1,
                end_line=index,
                confidence=0.82 if row_count else 0.58,
                title=raw_block[0].strip(),
                strategy=strategy,
                rows=row_count,
                notes=[] if row_count else ["Table caption detected but rows could not be split confidently."],
            )
        )
    return output, blocks


def _should_stop_table_block(line: str, raw_block: list[str]) -> bool:
    if _TABLE_CAPTION_RE.match(line):
        return True
    if _FIGURE_CAPTION_RE.match(line):
        return True
    if len(raw_block) >= 4 and _SECTION_HEADING_RE.match(line):
        return True
    if len(raw_block) >= 4 and _looks_like_prose_after_table(line):
        return True
    return False


def _looks_like_prose_after_table(line: str) -> bool:
    words = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", line)
    if len(words) < 10:
        return False
    math_cells = _MATH_CELL_RE.findall(line)
    if len(math_cells) >= 3:
        return False
    alpha_chars = sum(ch.isalpha() for ch in line)
    digit_chars = sum(ch.isdigit() for ch in line)
    has_sentence_punctuation = bool(re.search(r"[.!?]\s+[A-ZÁÉÍÓÚÑ]", line))
    return alpha_chars > 45 and digit_chars <= 8 and (
        has_sentence_punctuation or len(line) > 70
    )


def _render_table_block(raw_block: list[str]) -> tuple[list[str], str, int]:
    compact = [line.strip() for line in raw_block if line.strip()]
    if not compact:
        return raw_block, "unchanged", 0

    caption_lines: list[str] = []
    body_lines: list[str] = []
    seen_row = False
    parsed_rows: list[tuple[str, list[str]]] = []
    header_notes: list[str] = []

    for idx, line in enumerate(compact):
        if idx == 0:
            caption_lines.append(line)
            continue
        label, cells = _split_table_row(line)
        if cells:
            seen_row = True
            parsed_rows.append((label or " ", cells))
            body_lines.append(line)
        elif not seen_row:
            header_notes.append(line)
        else:
            body_lines.append(line)

    if parsed_rows:
        max_cells = max(len(cells) for _label, cells in parsed_rows)
        table_lines: list[str] = []
        table_lines.extend(caption_lines)
        if header_notes:
            table_lines.append("")
            table_lines.append("Original table header: " + " | ".join(header_notes))
        table_lines.append("")
        headers = ["Item"] + [f"Value {i}" for i in range(1, max_cells + 1)]
        table_lines.append("| " + " | ".join(headers) + " |")
        table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for label, cells in parsed_rows:
            padded = cells + [""] * (max_cells - len(cells))
            table_lines.append("| " + " | ".join(_escape_pipe_cell(cell) for cell in [label] + padded) + " |")
        table_lines.append("")
        return table_lines, "numeric_tail_markdown_table", len(parsed_rows)

    body_lines = [line for line in compact[1:] if line]
    table_lines = caption_lines + ["", "| Table content |", "| --- |"]
    for line in body_lines:
        table_lines.append(f"| {_escape_pipe_cell(line)} |")
    table_lines.append("")
    return table_lines, "single_column_markdown_table", len(body_lines)


def _split_table_row(line: str) -> tuple[str, list[str]]:
    matches = list(_MATH_CELL_RE.finditer(line))
    if not matches:
        return line.strip(), []
    word_count = len(re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+", line))
    if len(matches) == 1 and word_count > 6:
        return line.strip(), []
    if len(matches) == 1 and matches[0].start() < max(3, len(line) // 3):
        return line.strip(), []
    first = matches[0]
    label = line[: first.start()].strip()
    if not label and len(matches) < 2:
        return line.strip(), []

    cells: list[str] = []
    for match in matches:
        cell = _normalize_math_cell(match.group(0))
        if cell:
            cells.append(cell)
    return label, cells


def _normalize_math_cell(cell: str) -> str:
    value, _repairs = repair_structural_artifacts(re.sub(r"\s+", " ", cell.strip()))
    return value


def _escape_pipe_cell(value: str) -> str:
    return (value or "").replace("|", r"\|").strip()


def _compact_visual_token_lines(lines: list[str]) -> tuple[list[str], list[StructureBlock]]:
    output: list[str] = []
    blocks: list[StructureBlock] = []
    index = 0
    while index < len(lines):
        if not _starts_visual_token_run(lines, index):
            output.append(lines[index])
            index += 1
            continue

        start = index
        run: list[str] = []
        while index < len(lines):
            line = lines[index].strip()
            if not line:
                break
            if _FIGURE_CAPTION_RE.match(line) or _TABLE_CAPTION_RE.match(line):
                break
            if not _is_visual_token_line(line):
                break
            run.append(line)
            index += 1

        if len(run) < 12:
            output.extend(lines[start:index])
            continue

        compacted = _join_visual_tokens(run)
        output.append("Figure visual text:")
        output.append(compacted)
        output.append("")
        blocks.append(
            StructureBlock(
                block_id="",
                type="figure_text",
                start_line=start + 1,
                end_line=index,
                confidence=0.86,
                title="Figure visual text",
                strategy="compact_one_token_lines",
                rows=0,
            )
        )
    return output, blocks


def _starts_visual_token_run(lines: list[str], index: int) -> bool:
    current = lines[index].strip() if index < len(lines) else ""
    if not _is_visual_token_line(current):
        return False
    window = [line.strip() for line in lines[index:index + 80] if line.strip()]
    if len(window) < 12:
        return False
    tokenish = [line for line in window if _is_visual_token_line(line)]
    specials = {line.casefold() for line in window} & _FIGURE_TOKEN_SPECIALS
    return len(tokenish) / len(window) >= 0.82 and bool(specials)


def _is_visual_token_line(line: str) -> bool:
    if not line:
        return False
    folded = line.casefold()
    if folded in _FIGURE_TOKEN_SPECIALS:
        return True
    if line in _PUNCT_NO_SPACE_BEFORE or line in {"-", "–", "—"}:
        return True
    if " " in line:
        return False
    return len(line) <= 18


def _join_visual_tokens(tokens: Iterable[str]) -> str:
    parts: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        if not parts:
            parts.append(token)
            continue
        if token in _PUNCT_NO_SPACE_BEFORE:
            parts[-1] = parts[-1].rstrip() + token
        elif parts[-1] in _PUNCT_NO_SPACE_AFTER:
            parts[-1] += token
        elif token in {"-", "–", "—"} or parts[-1] in {"-", "–", "—"}:
            parts.append(token)
        else:
            parts.append(" " + token)
    return "".join(parts).strip()


def _next_nonblank(lines: list[str], start: int) -> int | None:
    for idx in range(start, len(lines)):
        if lines[idx].strip():
            return idx
    return None
