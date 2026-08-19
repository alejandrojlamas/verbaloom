"""Source-proven repair of paragraphs split by pagination or OCR.

The repair deliberately keeps every XHTML element in place.  Continuation
text is moved into the preceding paragraph and the now-empty source-aligned
node is marked for the publication gate.  This preserves spine order, links,
images, IDs, and the source/output DOM correspondence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
import stat
import tempfile
import zipfile

from lxml import etree

from .dom_boundaries import _parse_xhtml, _serialize_xhtml, _write_member


REFLOW_ANCHOR_CLASS = "verbaloom-reflowed-paragraph"
REFLOW_CONTINUATION_CLASS = "verbaloom-merged-continuation"
_LEGACY_REFLOW_ANCHOR_CLASS = "tbl-reflowed-paragraph"
_LEGACY_REFLOW_CONTINUATION_CLASS = "tbl-merged-continuation"
_REFLOW_ANCHOR_CLASSES = {REFLOW_ANCHOR_CLASS, _LEGACY_REFLOW_ANCHOR_CLASS}
_REFLOW_CONTINUATION_CLASSES = {
    REFLOW_CONTINUATION_CLASS,
    _LEGACY_REFLOW_CONTINUATION_CLASS,
}
_REFLOW_CLASSES = _REFLOW_ANCHOR_CLASSES | _REFLOW_CONTINUATION_CLASSES
_TEXT_SUFFIXES = {".xhtml", ".html", ".htm"}
_TERMINAL_PUNCTUATION = ".!?…:;"
_CLOSING_PUNCTUATION = "\"'’”»)]}"
_OPENING_PUNCTUATION = "\"'‘“«([{"
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_TRAILING_HYPHEN_RE = re.compile(r"[-\u00ad]\s*[^\W\d_]?\s*$", re.UNICODE)
_SAFE_INLINE_ELEMENTS = {
    "b", "big", "cite", "code", "del", "em", "i", "ins", "kbd",
    "mark", "q", "s", "samp", "small", "span", "strong", "sub", "sup",
    "time", "u", "var",
}


@dataclass
class ParagraphReflowReport:
    scanned_files: int = 0
    scanned_boundaries: int = 0
    merged_continuations: int = 0
    dehyphenated_continuations: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.merged_continuations > 0

    def to_dict(self) -> dict:
        return asdict(self)


def normalized_text(element_or_text: etree._Element | str | None) -> str:
    if isinstance(element_or_text, etree._Element):
        value = " ".join(element_or_text.itertext())
    else:
        value = str(element_or_text or "")
    return _SPACE_RE.sub(" ", value).strip()


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname.lower()


def _classes(element: etree._Element) -> set[str]:
    return {item for item in str(element.get("class") or "").split() if item}


def _has_reflow_anchor_class(element: etree._Element) -> bool:
    return bool(_REFLOW_ANCHOR_CLASSES & _classes(element))


def _has_reflow_continuation_class(element: etree._Element) -> bool:
    return bool(_REFLOW_CONTINUATION_CLASSES & _classes(element))


def _add_class(element: etree._Element, class_name: str) -> None:
    classes = _classes(element)
    classes.add(class_name)
    element.set("class", " ".join(sorted(classes)))


def _base_attributes(element: etree._Element) -> dict[str, str]:
    attributes = dict(element.attrib)
    classes = {
        item for item in str(attributes.get("class") or "").split()
        if item not in _REFLOW_CLASSES
    }
    if classes:
        attributes["class"] = " ".join(sorted(classes))
    else:
        attributes.pop("class", None)
    return attributes


def _is_plain_paragraph(element: etree._Element) -> bool:
    return bool(
        _local_name(element) == "p"
        and len(element) == 0
        and not element.get("id")
        and not element.get("role")
        and not element.get("{http://www.idpf.org/2007/ops}type")
    )


def _is_safe_anchor_paragraph(element: etree._Element) -> bool:
    if _local_name(element) != "p" or element.get("id") or element.get("role"):
        return False
    if element.get("{http://www.idpf.org/2007/ops}type"):
        return False
    for descendant in element.iterdescendants():
        if (
            not isinstance(descendant.tag, str)
            or _local_name(descendant) not in _SAFE_INLINE_ELEMENTS
            or descendant.get("id")
            or descendant.get("role")
        ):
            return False
    return True


def _direct_previous_paragraph(element: etree._Element) -> etree._Element | None:
    previous = element.getprevious()
    if previous is None or not isinstance(previous.tag, str):
        return None
    if previous.getparent() is not element.getparent() or _local_name(previous) != "p":
        return None
    return previous


def _first_lexical_character(text: str) -> str:
    value = normalized_text(text).lstrip(_OPENING_PUNCTUATION)
    return value[:1]


def _ends_terminally(text: str) -> bool:
    value = normalized_text(text).rstrip(_CLOSING_PUNCTUATION)
    return bool(value and value[-1] in _TERMINAL_PUNCTUATION)


def is_continuation_boundary(previous_text: str, next_text: str) -> bool:
    """Return whether two adjacent text paragraphs are a likely page split."""
    previous = normalized_text(previous_text)
    following = normalized_text(next_text)
    if not previous or not following:
        return False
    first = _first_lexical_character(following)
    if not first or not first.isalpha() or not first.islower():
        return False
    if previous.endswith(("-", "\u00ad")) or _TRAILING_HYPHEN_RE.search(previous):
        return True
    if _ends_terminally(previous):
        return False
    # Long fragments that continue in lower case are characteristic of OCR or
    # page-boundary paragraph splits.  The minimums avoid joining intentional
    # short literary paragraphs, verse, captions, and dialogue beats.
    return len(previous) >= 40 and len(following) >= 20


def join_continuation_text(
    previous_text: str,
    next_text: str,
    *,
    source_proven_dehyphenation: bool = False,
) -> tuple[str, bool]:
    previous = str(previous_text or "").rstrip()
    following = str(next_text or "").lstrip()
    if previous.endswith(("-", "\u00ad")):
        return previous[:-1] + following, True
    if source_proven_dehyphenation:
        return previous + following, True
    return previous + " " + following, False


def _source_proves_translated_word_split(
    source_previous: str,
    source_current: str,
    output_previous: str,
    output_current: str,
    *,
    source_vocabulary: set[str],
    output_vocabulary: set[str],
) -> bool:
    """Recognize a translated word split after the model dropped the hyphen.

    Some scanned EPUBs divide a source word as ``remem-`` / ``bered``.  A
    model may return the translated halves as ``re`` / ``cordó`` without the
    source hyphen.  Joining those blocks with the normal paragraph separator
    creates a reader-visible OCR error.  We remove the separator only when the
    source boundary is explicitly hyphenated and the joined target token is
    independently attested elsewhere in the same output EPUB.
    """
    source_left = normalized_text(source_previous)
    source_right = normalized_text(source_current)
    left_matches = list(_WORD_RE.finditer(normalized_text(output_previous)))
    right_match = _WORD_RE.search(normalized_text(output_current))
    if not left_matches or right_match is None or right_match.start() != 0:
        return False
    left = left_matches[-1].group(0)
    right = right_match.group(0)
    if len(left) < 2 or len(right) < 2:
        return False
    if (left + right).casefold() not in output_vocabulary:
        return False
    if _TRAILING_HYPHEN_RE.search(source_left):
        return True

    source_left_matches = list(_WORD_RE.finditer(source_left))
    source_right_match = _WORD_RE.search(source_right)
    if not source_left_matches or source_right_match is None:
        return False
    source_left_word = source_left_matches[-1].group(0)
    source_right_word = source_right_match.group(0)
    if len(source_left_word) < 2 or len(source_right_word) < 2:
        return False
    return (source_left_word + source_right_word).casefold() in source_vocabulary


def _find_reflow_anchor(element: etree._Element) -> etree._Element | None:
    current = element
    while current is not None and _has_reflow_continuation_class(current):
        current = _direct_previous_paragraph(current)
    if current is None or not _is_safe_anchor_paragraph(current):
        return None
    return current


def _last_text_slot(element: etree._Element) -> tuple[etree._Element, str]:
    slots: list[tuple[etree._Element, str]] = [(element, "text")]

    def visit(node: etree._Element) -> None:
        for child in node:
            if isinstance(child.tag, str):
                slots.append((child, "text"))
                visit(child)
            slots.append((child, "tail"))

    visit(element)
    for slot in reversed(slots):
        if str(getattr(slot[0], slot[1]) or "").strip():
            return slot
    return element, "text"


def _append_continuation_text(
    anchor: etree._Element,
    continuation: str,
    *,
    source_proven_dehyphenation: bool = False,
) -> bool:
    node, attribute = _last_text_slot(anchor)
    merged, dehyphenated = join_continuation_text(
        str(getattr(node, attribute) or ""),
        continuation,
        source_proven_dehyphenation=source_proven_dehyphenation,
    )
    setattr(node, attribute, merged)
    return dehyphenated


def is_valid_marked_continuation(
    source_previous: etree._Element,
    source_current: etree._Element,
    output_current: etree._Element,
) -> bool:
    """Validate a reflow marker against the corresponding source boundary."""
    if not _has_reflow_continuation_class(output_current):
        return False
    if normalized_text(output_current):
        return False
    if (
        not _is_safe_anchor_paragraph(source_previous)
        or not _is_plain_paragraph(source_current)
        or not _is_plain_paragraph(output_current)
    ):
        return False
    if _direct_previous_paragraph(source_current) is not source_previous:
        return False
    if not is_continuation_boundary(
        normalized_text(source_previous),
        normalized_text(source_current),
    ):
        return False
    output_previous = _direct_previous_paragraph(output_current)
    if output_previous is None:
        return False
    anchor = _find_reflow_anchor(output_previous)
    return bool(
        anchor is not None
        and _has_reflow_anchor_class(anchor)
        and normalized_text(anchor)
    )


def is_valid_marked_anchor(
    source_blocks: list[etree._Element],
    output_blocks: list[etree._Element],
    block_index: int,
) -> bool:
    """Prove that an expanded output block owns marked source continuations."""
    if block_index < 0 or block_index >= len(output_blocks):
        return False
    output_anchor = output_blocks[block_index]
    if (
        not _has_reflow_anchor_class(output_anchor)
        or not _is_safe_anchor_paragraph(output_anchor)
        or not normalized_text(output_anchor)
    ):
        return False
    found = False
    index = block_index + 1
    while index < len(output_blocks):
        output_current = output_blocks[index]
        if not _has_reflow_continuation_class(output_current):
            break
        if not is_valid_marked_continuation(
            source_blocks[index - 1],
            source_blocks[index],
            output_current,
        ):
            return False
        found = True
        index += 1
    return found


def reflow_split_paragraphs(
    source_root: etree._Element,
    output_root: etree._Element,
    *,
    source_vocabulary: set[str] | None = None,
    output_vocabulary: set[str] | None = None,
) -> ParagraphReflowReport:
    """Merge only adjacent, source-proven, plain-text paragraph continuations."""
    report = ParagraphReflowReport(scanned_files=1)
    source_paragraphs = source_root.xpath("//*[local-name()='p']")
    output_paragraphs = output_root.xpath("//*[local-name()='p']")
    if len(source_paragraphs) != len(output_paragraphs):
        report.errors.append(
            f"paragraph count differs ({len(source_paragraphs)} != {len(output_paragraphs)})"
        )
        return report

    if source_vocabulary is None:
        source_vocabulary = {
            match.group(0).casefold()
            for match in _WORD_RE.finditer(normalized_text(source_root))
        }
    if output_vocabulary is None:
        output_vocabulary = {
            match.group(0).casefold()
            for match in _WORD_RE.finditer(normalized_text(output_root))
        }

    for index in range(1, len(source_paragraphs)):
        source_previous = source_paragraphs[index - 1]
        source_current = source_paragraphs[index]
        output_previous = output_paragraphs[index - 1]
        output_current = output_paragraphs[index]

        if (
            _direct_previous_paragraph(source_current) is not source_previous
            or _direct_previous_paragraph(output_current) is not output_previous
        ):
            continue
        report.scanned_boundaries += 1
        if _has_reflow_continuation_class(output_current):
            continue
        if (
            not _is_safe_anchor_paragraph(source_previous)
            or not _is_plain_paragraph(source_current)
            or not _is_safe_anchor_paragraph(output_previous)
            or not _is_plain_paragraph(output_current)
        ):
            continue
        if (
            _base_attributes(source_previous) != _base_attributes(source_current)
            or _base_attributes(output_previous) != _base_attributes(output_current)
        ):
            continue
        if not is_continuation_boundary(
            normalized_text(source_previous),
            normalized_text(source_current),
        ):
            continue

        anchor = _find_reflow_anchor(output_previous)
        if anchor is None:
            continue
        source_proven_dehyphenation = _source_proves_translated_word_split(
            normalized_text(source_previous),
            normalized_text(source_current),
            normalized_text(anchor),
            normalized_text(output_current),
            source_vocabulary=source_vocabulary,
            output_vocabulary=output_vocabulary,
        )
        if not source_proven_dehyphenation and not is_continuation_boundary(
            normalized_text(anchor),
            normalized_text(output_current),
        ):
            continue
        if _base_attributes(anchor) != _base_attributes(output_current):
            continue

        dehyphenated = _append_continuation_text(
            anchor,
            str(output_current.text or ""),
            source_proven_dehyphenation=source_proven_dehyphenation,
        )
        output_current.text = None
        _add_class(anchor, REFLOW_ANCHOR_CLASS)
        _add_class(output_current, REFLOW_CONTINUATION_CLASS)
        report.merged_continuations += 1
        report.dehyphenated_continuations += int(dehyphenated)
    return report


def repair_epub_split_paragraphs(
    source_epub: str | Path,
    output_epub: str | Path,
) -> ParagraphReflowReport:
    """Atomically reflow source-proven pagination splits in an output EPUB."""
    source_path = Path(source_epub)
    output_path = Path(output_epub)
    mode = stat.S_IMODE(output_path.stat().st_mode)
    report = ParagraphReflowReport()
    temporary = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=".epub",
        dir=output_path.parent,
        delete=False,
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        with (
            zipfile.ZipFile(source_path) as source,
            zipfile.ZipFile(output_path) as output,
            zipfile.ZipFile(temporary_path, "w") as rebuilt,
        ):
            source_names = set(source.namelist())
            source_vocabulary: set[str] = set()
            output_vocabulary: set[str] = set()
            for info in output.infolist():
                if (
                    Path(info.filename).suffix.lower() not in _TEXT_SUFFIXES
                    or info.filename not in source_names
                ):
                    continue
                try:
                    source_vocabulary.update(
                        match.group(0).casefold()
                        for match in _WORD_RE.finditer(
                            normalized_text(_parse_xhtml(source.read(info.filename)))
                        )
                    )
                    output_vocabulary.update(
                        match.group(0).casefold()
                        for match in _WORD_RE.finditer(
                            normalized_text(_parse_xhtml(output.read(info.filename)))
                        )
                    )
                except Exception:
                    # The normal member pass records parse failures with the
                    # exact filename; a malformed document must not prevent
                    # other package files from being repaired.
                    continue
            for info in output.infolist():
                payload = output.read(info.filename)
                if (
                    Path(info.filename).suffix.lower() in _TEXT_SUFFIXES
                    and info.filename in source_names
                ):
                    try:
                        output_root = _parse_xhtml(payload)
                        local = reflow_split_paragraphs(
                            _parse_xhtml(source.read(info.filename)),
                            output_root,
                            source_vocabulary=source_vocabulary,
                            output_vocabulary=output_vocabulary,
                        )
                        report.scanned_files += local.scanned_files
                        report.scanned_boundaries += local.scanned_boundaries
                        report.merged_continuations += local.merged_continuations
                        report.dehyphenated_continuations += local.dehyphenated_continuations
                        report.errors.extend(
                            f"{info.filename}: {error}" for error in local.errors
                        )
                        if local.changed:
                            payload = _serialize_xhtml(output_root, payload)
                    except Exception as exc:
                        report.errors.append(
                            f"{info.filename}: {type(exc).__name__}: {exc}"
                        )
                _write_member(rebuilt, info, payload)
        temporary_path.replace(output_path)
        output_path.chmod(mode)
    finally:
        temporary_path.unlink(missing_ok=True)
    return report
