"""Editorial assembly agent for plain-text book exports.

This module owns the last production step before generated EPUB output:
clean obvious OCR artifacts, recover paragraphs, classify navigation headings,
and keep source text when a candidate heading is rejected. It is deliberately
token-free; optional LLM adjudication can operate on the compact candidates this
agent produces instead of on the full book.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata

from src.utils.text_encoding import clean_text_artifacts


@dataclass
class EditorialSection:
    title: str
    paragraphs: list[str]
    role: str = "body"
    source: str = "rules"


@dataclass
class EditorialAssemblyPlan:
    title: str
    sections: list[EditorialSection]
    removed_artifacts: list[str] = field(default_factory=list)
    demoted_headings: list[str] = field(default_factory=list)
    toc_entries_detected: int = 0


_UPPER_RE_CHARS = "A-Z\\u00C0-\\u00D6\\u00D8-\\u00DE"
_LETTER_RE_CHARS = "A-Za-z\\u00C0-\\u024F"
_WORD_CHARS = rf"{_LETTER_RE_CHARS}0-9"
_MONTH_RE = (
    "enero|febrero|marzo|abril|mayo|junio|julio|agosto|"
    "septiembre|setiembre|octubre|noviembre|diciembre|"
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)
_YEAR_RE = r"(?:[1-9]\d{2,3})(?:\s*[-\u2013\u2014]\s*\d{1,4})?"
_DATE_MARKER_RE = (
    rf"(?:c\.\s*)?{_YEAR_RE}|"
    rf"(?:\d{{1,2}}\s*[-\u2013\u2014]\s*)?\d{{1,2}}\s+de\s+(?:{_MONTH_RE})\s+de\s+\d{{3,4}}|"
    rf"(?:{_MONTH_RE})\s+\d{{1,2}},?\s+{_YEAR_RE}|"
    rf"(?:{_MONTH_RE})\s+de\s+{_YEAR_RE}"
)
_DATE_TITLE_DELIMITER_RE = rf"(?:,\s*|:\s*[^.!?\n]{{0,70}},\s*)"
_LEADING_DATE_TITLE_RE = re.compile(
    rf"^(?P<title>[{_UPPER_RE_CHARS}][^.!?\n]{{4,190}}?{_DATE_TITLE_DELIMITER_RE}(?:{_DATE_MARKER_RE})\b)"
    rf"[.!?:;,-]*\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
_EMBEDDED_DATE_TITLE_START_RE = re.compile(
    rf"([.!?\u2026][\"'\)\]\u00BB\u201D]?\s+)"
    rf"(?=[{_UPPER_RE_CHARS}][^.!?\n]{{6,190}}?{_DATE_TITLE_DELIMITER_RE}(?:{_DATE_MARKER_RE})\b)",
    re.IGNORECASE,
)
_AUTHOR_WORD_RE = rf"(?:[{_UPPER_RE_CHARS}][{_LETTER_RE_CHARS}'.\u2019-]*|[{_UPPER_RE_CHARS}]\.)"
_EMBEDDED_AUTHOR_DATE_TITLE_START_RE = re.compile(
    rf"(\b(?!(?:El|La|Los|Las|Un|Una|The|A|An)\s+)"
    rf"{_AUTHOR_WORD_RE}(?:\s+{_AUTHOR_WORD_RE}){{1,4}}\s+)"
    rf"(?=[{_UPPER_RE_CHARS}][^.!?\n]{{4,170}}?{_DATE_TITLE_DELIMITER_RE}(?:{_DATE_MARKER_RE})\b)",
)
_CONTENTS_HEADING_RE = re.compile(
    r"^(?:contents|contenido|tabla de contenidos?|indice|\u00EDndice)$",
    re.IGNORECASE,
)
_DECLARED_HEADING_RE = re.compile(
    r"^(?:chapter|capitulo|cap\u00EDtulo|part|parte|book|libro|section|secci\u00F3n)"
    r"(?:\s+[\wIVXLCDMivxlcdm.\-]+)?(?:\s*[:.\-]\s*.+|\s+.+)?$",
    re.IGNORECASE,
)
_FRONT_BACK_HEADING_RE = re.compile(
    r"^(?:prologo|pr\u00F3logo|prefacio|introducci\u00F3n|introduccion|"
    r"epilogo|ep\u00EDlogo|apendice|ap\u00E9ndice|bibliograf\u00EDa|"
    r"bibliografia|notas|agradecimientos|glosario)$",
    re.IGNORECASE,
)
_FRONTMATTER_CREDIT_RE = re.compile(
    r"\b(?:editado\s+por|edited\s+by|avon\s+books|copyright|isbn|"
    r"library\s+of\s+congress|biblioteca\s+del\s+congreso|all\s+rights\s+reserved|"
    r"todos\s+los\s+derechos\s+reservados|impreso\s+en|printed\s+in|"
    r"marca\s+registrada|published\s+by|publicado\s+por)\b",
    re.IGNORECASE,
)
_BODY_SENTENCE_VERB_RE = re.compile(
    r"\b(?:era|estaba|estaban|habia|hab\u00EDa|habian|hab\u00EDan|fue|fueron|"
    r"habria|habr\u00EDa|seria|ser\u00EDa|consistiria|consistir\u00EDa|dijo|dijeron|vino|llego|"
    r"lleg\u00F3|salio|sali\u00F3|vimos|hicimos|encontro|encontr\u00F3|"
    r"comenzo|comenz\u00F3|marcho|march\u00F3|murio|muri\u00F3|murieron|"
    r"avanzaron|retrocedieron|respondio|respondi\u00F3)\b",
    re.IGNORECASE,
)
_SENTENCE_BOUNDARY_RE = re.compile(
    rf"(?<=[.!?\u2026])\s+(?=[\(\"'\u00AB\u201C\u00BF\u00A1]?[{_UPPER_RE_CHARS}0-9])"
)
_CHRONICLE_MARKER_RE = re.compile(
    r"\s+(?=(?:"
    r"El\s+(?:d[i\u00ED]a|primero|segundo|tercero|cuarto|quinto|sexto|"
    r"s[e\u00E9]ptimo|octavo|noveno|d[e\u00E9]cimo|und[e\u00E9]cimo|"
    r"duod[e\u00E9]cimo|decimotercero|decimocuarto|decimoquinto|"
    r"diecis[e\u00E9]is|diecisiete|dieciocho|diecinueve|veinte|"
    r"veintiuno|veintid[o\u00F3]s|veintitr[e\u00E9]s|veinticuatro|"
    r"veinticinco|veintis[e\u00E9]is|veintisiete|veintiocho|"
    r"veintinueve|treinta|treinta y uno)\b|"
    r"Al\s+d[i\u00ED]a\s+siguiente\b|"
    r"A\s+la\s+ma\u00F1ana\s+siguiente\b"
    r"))",
    re.IGNORECASE,
)
_FORMULAISH_RE = re.compile(
    r"(?:[=<>+\-*/^_{}]|[\u0370-\u03ff]|10\^|\\(?:frac|sum|int|sqrt))"
)
_PAGE_TRAIL_RE = re.compile(r"\s+(?:\d{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)


class EditorialAssemblyAgent:
    """Build a conservative editorial plan for book-shaped plain text."""

    def assemble(self, text: str, document_title: str | None = None) -> EditorialAssemblyPlan:
        title = _clean_heading(document_title or "Libro traducido")
        normalized = normalize_editorial_text(text)
        blocks = _raw_blocks(normalized)

        removed: list[str] = []
        clean_blocks: list[str] = []
        for index, block in enumerate(blocks):
            block = _repair_ocr_date_tokens(_clean_block(block))
            if not block:
                continue
            if _is_artifact_block(block, front_region=index < 90):
                removed.append(_preview(block))
                continue
            clean_blocks.append(block)

        sections, demoted, toc_count, saw_heading = self._build_sections(clean_blocks, title)
        if not sections:
            sections = [EditorialSection(title=title, paragraphs=[])]

        if not saw_heading and len(sections) == 1 and len(sections[0].paragraphs) > 70:
            sections = _chunk_untitled_sections(sections[0].paragraphs)

        return EditorialAssemblyPlan(
            title=title,
            sections=sections,
            removed_artifacts=removed,
            demoted_headings=demoted,
            toc_entries_detected=toc_count,
        )

    def _build_sections(
        self,
        blocks: list[str],
        fallback_title: str,
    ) -> tuple[list[EditorialSection], list[str], int, bool]:
        sections: list[EditorialSection] = []
        current_title: str | None = None
        current_role = "body"
        current_paragraphs: list[str] = []
        demoted: list[str] = []
        toc_entries = 0
        in_toc = False
        saw_heading = False

        def flush(*, next_heading: bool = False) -> None:
            nonlocal current_title, current_role, current_paragraphs
            if current_title or current_paragraphs:
                if (
                    next_heading
                    and sections
                    and current_role == "body"
                    and _looks_like_sparse_list_section(current_title or "", current_paragraphs)
                ):
                    sections[-1].paragraphs.extend(
                        [p for p in [current_title, *current_paragraphs] if p]
                    )
                    if current_title:
                        demoted.append(_preview(current_title))
                    current_title = None
                    current_role = "body"
                    current_paragraphs = []
                    return
                title = current_title or ("Inicio" if not sections else f"Parte {len(sections) + 1}")
                sections.append(
                    EditorialSection(
                        title=_clean_heading(title),
                        paragraphs=[p for p in current_paragraphs if p],
                        role=current_role,
                    )
                )
            current_title = None
            current_role = "body"
            current_paragraphs = []

        def add_body(block: str) -> None:
            current_paragraphs.extend(_body_paragraphs(block))

        for index, block in enumerate(blocks):
            next_block = blocks[index + 1] if index + 1 < len(blocks) else ""

            if in_toc and _should_leave_toc(block, next_block, toc_entries):
                flush()
                in_toc = False

            if not in_toc and _is_toc_heading(block):
                flush()
                current_title = "Indice" if _is_ascii_only(block) else "\u00CDndice"
                current_role = "toc"
                in_toc = True
                saw_heading = True
                continue

            if in_toc:
                for line in _toc_lines(block):
                    if line:
                        current_paragraphs.append(line)
                        toc_entries += 1
                continue

            for segment in _split_embedded_section_starts(block):
                heading, rest = _extract_leading_heading(segment)
                if heading and _should_accept_heading(heading, rest):
                    flush(next_heading=True)
                    current_title = heading
                    current_role = "body"
                    saw_heading = True
                    if rest:
                        add_body(rest)
                    continue
                if heading:
                    demoted.append(_preview(heading))
                add_body(segment)

        flush()
        return sections, demoted, toc_entries, saw_heading


def normalize_editorial_text(text: str) -> str:
    """Normalize final text for editorial assembly without rewriting prose."""
    value = clean_text_artifacts(text or "")
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\ufeff", "").replace("\u00A0", " ")
    value = "".join(ch if ch in {"\n", "\t"} or ord(ch) >= 32 else " " for ch in value)
    value = _repair_ocr_date_tokens(value)
    value = re.sub(r"(\b\d{3})[.,]\s+(\d\b)", r"\1\2", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"[ \t]*\n[ \t]*", "\n", value)
    value = re.sub(r"\n{4,}", "\n\n\n", value)
    return value.strip()


def _repair_ocr_date_tokens(text: str) -> str:
    """Repair common OCR digit confusions inside date/page-like tokens."""
    if not text:
        return text
    value = text
    value = re.sub(r"\b([12]\d)[xX](\d)\b", r"\g<1>2\2", value)
    value = re.sub(r"\b(\d{3})[zZ]\b", r"\g<1>2", value)
    value = re.sub(r"\b[iIlL](\d)[rRlL][oO0]\b", r"1\g<1>10", value)
    value = re.sub(r"\b[iIlL][zZ]\b", "12", value)
    return value


def _raw_blocks(text: str) -> list[str]:
    normalized = normalize_editorial_text(text)
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", normalized) if b.strip()]
    if blocks:
        return blocks
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    return [" ".join(lines)] if lines else []


def _clean_block(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in (value or "").splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _clean_heading(value: str) -> str:
    heading = re.sub(r"\s+", " ", value or "").strip()
    heading = heading.strip(" \t\r\n-:;,.!?")
    return heading or "Seccion"


def _clean_paragraph(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _preview(value: str, limit: int = 120) -> str:
    text = _clean_paragraph(value)
    return text[:limit]


def _is_ascii_only(value: str) -> bool:
    try:
        value.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _is_toc_heading(block: str) -> bool:
    return bool(_CONTENTS_HEADING_RE.match(_clean_heading(block)))


def _toc_lines(block: str) -> list[str]:
    lines = [_clean_paragraph(line) for line in block.splitlines() if line.strip()]
    if len(lines) <= 1:
        return [_clean_paragraph(block)]
    return lines


def _should_leave_toc(block: str, next_block: str, entry_count: int) -> bool:
    cleaned = _clean_paragraph(block)
    if not cleaned:
        return False
    if entry_count >= 2 and _FRONT_BACK_HEADING_RE.match(_clean_heading(cleaned)):
        return True
    if entry_count < 4:
        return False
    heading, rest = _extract_leading_heading(cleaned)
    if _looks_like_toc_entry(cleaned):
        if (
            heading
            and next_block
            and len(_clean_paragraph(next_block)) > 220
            and not _looks_like_toc_entry(next_block)
            and not _toc_rest_looks_like_page_author(rest)
        ):
            return True
        return False
    if heading and rest and len(rest) > 80:
        return True
    if heading and next_block and len(_clean_paragraph(next_block)) > 220 and not _looks_like_toc_entry(next_block):
        return True
    if len(cleaned) > 420 and not _looks_like_toc_entry(cleaned):
        return True
    return False


def _looks_like_toc_entry(block: str) -> bool:
    text = _clean_paragraph(block)
    if not text:
        return False
    if len(text) > 220:
        return False
    if _PAGE_TRAIL_RE.search(text):
        return True
    if re.search(
        rf"\b\d{{1,4}}\s+[{_UPPER_RE_CHARS}][{_LETTER_RE_CHARS}'.\u2019-]*"
        rf"(?:\s+[{_LETTER_RE_CHARS}'.\u2019-]+){{0,5}}$",
        text,
    ):
        return True
    if re.search(r"\b(?:c\.\s*)?\d{3,4}(?:\s*[-\u2013\u2014]\s*\d{1,4})?\b", text):
        return True
    words = text.split()
    if len(words) <= 5 and _is_author_name(text):
        return True
    return False


def _toc_rest_looks_like_page_author(rest: str) -> bool:
    text = _clean_paragraph(rest)
    if not text:
        return False
    if re.match(r"^\d{1,4}\b", text):
        return True
    return bool(
        re.search(
            rf"\b\d{{1,4}}\s+[{_UPPER_RE_CHARS}][{_LETTER_RE_CHARS}'.\u2019-]*"
            rf"(?:\s+[{_LETTER_RE_CHARS}'.\u2019-]+){{0,5}}$",
            text,
        )
    )


def _is_artifact_block(block: str, *, front_region: bool) -> bool:
    text = _clean_paragraph(block)
    if not text:
        return True
    stripped = text.strip(" \t\r\n-_\u2013\u2014~*|.:;")
    if not stripped:
        return True
    if re.fullmatch(r"[A-Z]{2,5}\s+\d+(?:\s+\d+)+", stripped):
        return True
    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    if len(text) <= 80 and symbol_count / max(1, len(text)) > 0.35:
        return True
    if not front_region:
        return False
    if len(stripped) <= 3:
        if stripped.lower() in {"mm", "mu"}:
            return True
        if not re.fullmatch(r"(?:[IVXLCDM]+|\d+)", stripped):
            return True
    if re.fullmatch(rf"[{_LETTER_RE_CHARS}]{{5,18}}", stripped):
        vowels = len(re.findall(r"[AEIOUaeiou\u00C1\u00C9\u00CD\u00D3\u00DA\u00E1\u00E9\u00ED\u00F3\u00FA]", stripped))
        transitions = sum(
            1
            for prev, cur in zip(stripped, stripped[1:])
            if prev.islower() != cur.islower()
        )
        if vowels / max(1, len(stripped)) < 0.28 and transitions >= 3:
            return True
    return False


def _split_embedded_section_starts(block: str) -> list[str]:
    if len(block) < 80:
        return [block]

    def starts_with_uppercase(index: int) -> bool:
        return bool(re.match(rf"\s*[{_UPPER_RE_CHARS}]", block[index:]))

    split_points = sorted({
        match.end(1)
        for pattern in (_EMBEDDED_DATE_TITLE_START_RE, _EMBEDDED_AUTHOR_DATE_TITLE_START_RE)
        for match in pattern.finditer(block)
        if match.end(1) > 0 and starts_with_uppercase(match.end(1))
    })
    if not split_points:
        return [block]

    parts: list[str] = []
    last = 0
    for split_at in split_points:
        if split_at <= last:
            continue
        previous = block[last:split_at].strip()
        if previous:
            parts.append(previous)
        last = split_at
    tail = block[last:].strip()
    if tail:
        parts.append(tail)
    return parts or [block]


def _extract_leading_heading(block: str) -> tuple[str | None, str]:
    lines = [line.strip() for line in (block or "").splitlines() if line.strip()]
    if len(lines) >= 2 and _looks_like_standalone_heading(lines[0]):
        rest = "\n".join(lines[1:]).strip()
        return _clean_heading(lines[0]), rest

    cleaned = _clean_paragraph(block)
    if not cleaned:
        return None, ""

    match = _LEADING_DATE_TITLE_RE.match(cleaned)
    if match:
        return _clean_heading(match.group("title")), match.group("rest").strip()

    if _looks_like_standalone_heading(cleaned):
        return _clean_heading(cleaned), ""

    return None, cleaned


def _looks_like_standalone_heading(text: str) -> bool:
    text = _clean_heading(text)
    if not 2 <= len(text) <= 140:
        return False
    if "\n" in text:
        return False
    if _is_toc_heading(text) or _DECLARED_HEADING_RE.match(text) or _FRONT_BACK_HEADING_RE.match(text):
        return True
    if _LEADING_DATE_TITLE_RE.match(text):
        return True
    if re.match(r"^(?:[IVXLCDM]+|\d+)\.?$", text, re.IGNORECASE):
        return True
    if text.endswith((".", "!", "?", ";")):
        return False
    letters = re.findall(rf"[{_LETTER_RE_CHARS}]", text)
    uppercase = re.findall(rf"[{_UPPER_RE_CHARS}]", text)
    return bool(letters and len(uppercase) / len(letters) > 0.72)


def _should_accept_heading(heading: str, rest: str) -> bool:
    text = _clean_heading(heading)
    if not text or _is_artifact_block(text, front_region=True):
        return False
    if _FRONTMATTER_CREDIT_RE.search(text):
        return False
    if _is_author_name(text):
        return False
    if _DECLARED_HEADING_RE.match(text) or _FRONT_BACK_HEADING_RE.match(text) or _is_toc_heading(text):
        return True
    has_date = bool(re.search(rf"(?:{_DATE_MARKER_RE})\b", text, re.IGNORECASE))
    if has_date:
        if _is_sentence_like_heading(text) or _has_ambiguous_quantity_date(text):
            return False
        return True
    if re.match(r"^(?:[IVXLCDM]+|\d+)\.?$", text, re.IGNORECASE):
        return True
    letters = re.findall(rf"[{_LETTER_RE_CHARS}]", text)
    uppercase = re.findall(rf"[{_UPPER_RE_CHARS}]", text)
    if letters and len(uppercase) / len(letters) > 0.82 and len(text.split()) >= 2:
        return not _is_sentence_like_heading(text)
    return False


def _is_author_name(text: str) -> bool:
    words = text.strip().split()
    if not 2 <= len(words) <= 5:
        return False
    if any(re.search(r"\d|[,;:!?]", word) for word in words):
        return False
    if any(word.lower() in {"de", "del", "la", "le", "van", "von", "da", "dos"} for word in words):
        return False
    capitalized = [
        bool(re.match(rf"^[{_UPPER_RE_CHARS}][{_LETTER_RE_CHARS}'.\u2019-]*$", word))
        or bool(re.match(rf"^[{_UPPER_RE_CHARS}]\.$", word))
        for word in words
    ]
    return all(capitalized)


def _is_sentence_like_heading(text: str) -> bool:
    words = text.split()
    if len(words) >= 9 and _BODY_SENTENCE_VERB_RE.search(text):
        return True
    if re.match(
        r"^(?:Y|Pero|Luego|Despu[e\u00E9]s|Entonces|As[i\u00ED]|Era|Fue|"
        r"Hab[i\u00ED]a|Habr[i\u00ED]a|Estaba|La primera|El primero|"
        r"Al llegar|A la ma\u00F1ana siguiente)\b",
        text,
    ):
        return True
    if len(text) > 110 and _BODY_SENTENCE_VERB_RE.search(text):
        return True
    return False


def _has_ambiguous_quantity_date(text: str) -> bool:
    lower = text.lower()
    if "c." in lower or re.search(rf"\b(?:{_MONTH_RE})\b", lower, re.IGNORECASE):
        return False
    numbers = re.findall(r"(?<!\d)\d{2,4}(?!\d)", text)
    if len(numbers) <= 1:
        return False
    return len(text.split()) >= 8 or bool(_BODY_SENTENCE_VERB_RE.search(text))


def _looks_like_sparse_list_section(title: str, paragraphs: list[str]) -> bool:
    text = _clean_heading(title)
    if not text:
        return False
    if _DECLARED_HEADING_RE.match(text) or _FRONT_BACK_HEADING_RE.match(text):
        return False
    if re.match(r"^(?:[IVXLCDM]+|\d+)\.?$", text):
        return False
    if not paragraphs:
        return True
    if len(paragraphs) == 1:
        para = _clean_paragraph(paragraphs[0])
        if len(para) <= 95 and not re.search(r"[.!?]\s*$", para):
            return True
    return False


def _split_chronicle_markers(paragraph: str) -> list[str]:
    if len(paragraph) < 900:
        return [paragraph]
    parts = [p.strip() for p in _CHRONICLE_MARKER_RE.split(paragraph) if p and p.strip()]
    return parts or [paragraph]


def _split_long_paragraph(paragraph: str, target: int = 950) -> list[str]:
    text = _clean_paragraph(paragraph)
    if len(text) <= 1200:
        return [text] if text else []
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY_RE.split(text) if s.strip()]
    if len(sentences) <= 1:
        return _hard_wrap_paragraph(text, target)
    grouped: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len > target and current_len >= 450:
            grouped.append(" ".join(current).strip())
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len + 1
    if current:
        tail = " ".join(current).strip()
        if grouped and len(tail) < 220:
            grouped[-1] = f"{grouped[-1]} {tail}"
        else:
            grouped.append(tail)
    return grouped


def _hard_wrap_paragraph(text: str, target: int) -> list[str]:
    parts: list[str] = []
    remaining = text
    while len(remaining) > target:
        split_at = remaining.rfind(" ", 0, target)
        if split_at < 450:
            split_at = target
        parts.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    if remaining:
        parts.append(remaining)
    return parts


def _body_paragraphs(block: str) -> list[str]:
    paragraphs: list[str] = []
    cleaned = _clean_paragraph(block)
    if not cleaned:
        return []
    if _looks_like_formula_line(cleaned):
        return [cleaned]
    for candidate in _split_chronicle_markers(cleaned):
        paragraphs.extend(_split_long_paragraph(candidate))
    return [p for p in paragraphs if p]


def _looks_like_formula_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 240:
        return False
    if not _FORMULAISH_RE.search(line):
        return False
    digits_or_ops = sum(1 for ch in line if ch.isdigit() or ch in "=<>+-*/^_{}().,")
    return digits_or_ops >= 3


def _chunk_untitled_sections(paragraphs: list[str], chunk_size: int = 55) -> list[EditorialSection]:
    chunks: list[EditorialSection] = []
    for idx in range(0, len(paragraphs), chunk_size):
        chunks.append(
            EditorialSection(
                title=f"Parte {len(chunks) + 1}",
                paragraphs=paragraphs[idx:idx + chunk_size],
            )
        )
    return chunks
