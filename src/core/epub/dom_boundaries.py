"""Source-aware whitespace validation for XHTML inline-node boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.entities import name2codepoint
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping
import zipfile

from lxml import etree


_TEXT_SUFFIXES = {".xhtml", ".html", ".htm"}
_BLOCK_XPATH = (
    "//*[local-name()='p' or local-name()='h1' or local-name()='h2' or "
    "local-name()='h3' or local-name()='h4' or local-name()='h5' or "
    "local-name()='h6' or local-name()='li' or local-name()='blockquote' or "
    "local-name()='figcaption' or local-name()='td' or local-name()='th']"
)
_NAMED_ENTITY_RE = re.compile(rb"&([A-Za-z][A-Za-z0-9]+);")
_XML_BUILTINS = {b"amp", b"lt", b"gt", b"quot", b"apos"}
_SCENE_BREAK_RE = re.compile(r"^(?:[*.#~\-–—]\s*){3,}$")
_SANITIZED_ARTIFACT_CLASS = "verbaloom-sanitized-artifact"
_SANITIZED_ARTIFACT_CLASSES = {
    _SANITIZED_ARTIFACT_CLASS,
    "tbl-sanitized-artifact",
}


@dataclass(frozen=True)
class BoundaryFinding:
    file_href: str
    block_index: int
    left_slot: int
    right_slot: int
    reason: str


@dataclass
class BoundaryReport:
    scanned_files: int = 0
    scanned_boundaries: int = 0
    repaired_boundaries: int = 0
    findings: list[BoundaryFinding] = field(default_factory=list)
    structural_mismatches: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings and not self.structural_mismatches


@dataclass(frozen=True)
class MissingTextBlock:
    file_href: str
    block_index: int
    element: str
    source_text: str
    context_before: str = ""
    context_after: str = ""


def _slot_value(slot: tuple[etree._Element, str]) -> str:
    node, attribute = slot
    return str(getattr(node, attribute) or "")


def _set_slot_value(slot: tuple[etree._Element, str], value: str) -> None:
    node, attribute = slot
    setattr(node, attribute, value)


def _text_slots(root: etree._Element) -> list[tuple[etree._Element, str]]:
    slots: list[tuple[etree._Element, str]] = []

    def visit(node: etree._Element) -> None:
        slots.append((node, "text"))
        for child in node:
            if isinstance(child.tag, str):
                visit(child)
            slots.append((child, "tail"))

    visit(root)
    return slots


def _has_whitespace_between(values: list[str], left: int, right: int) -> bool:
    return bool(
        values[left][-1:].isspace()
        or values[right][:1].isspace()
        or any(value and any(char.isspace() for char in value) for value in values[left + 1:right])
    )


def _restore_boundary(
    source_values: list[str],
    output_slots: list[tuple[etree._Element, str]],
    output_values: list[str],
    left: int,
    right: int,
) -> bool:
    if output_values[left][-1:].isspace() or output_values[right][:1].isspace():
        return False
    if any(value and any(char.isspace() for char in value) for value in output_values[left + 1:right]):
        return False

    if source_values[left][-1:].isspace():
        _set_slot_value(output_slots[left], output_values[left] + " ")
        return True
    if source_values[right][:1].isspace():
        _set_slot_value(output_slots[right], " " + output_values[right])
        return True
    for index in range(left + 1, right):
        if source_values[index] and any(char.isspace() for char in source_values[index]):
            _set_slot_value(output_slots[index], " ")
            return True
    return False


def _visible_tokens_need_space(output_values: list[str], left: int, right: int) -> bool:
    left_text = output_values[left].rstrip()
    right_text = output_values[right].lstrip()
    if not left_text or not right_text:
        return False
    left_char = left_text[-1]
    right_char = right_text[0]
    # Target-language punctuation may legitimately move into the next slot.
    if right_char in ".,;:!?…)]}»”’%‰°":
        return False
    if left_char in "([{¿¡«“‘/\\":
        return False
    return bool(
        left_char.isalnum()
        or left_char in ".,;:!?…)]}»”’—–-"
    ) and bool(
        right_char.isalnum()
        or right_char in "([{¿¡«“‘"
    )


def _normalized_block_text(block: etree._Element) -> str:
    return re.sub(r"\s+", " ", " ".join(block.itertext())).strip()


def _spurious_table_cell_prefix(
    source_block: etree._Element,
    output_block: etree._Element,
) -> tuple[tuple[etree._Element, str] | None, str]:
    """Locate punctuation invented at the start of an isolated table cell."""
    source_text = _normalized_block_text(source_block)
    output_text = _normalized_block_text(output_block)
    if not source_text or not output_text:
        return None, ""
    if not source_text[0].isalnum() or output_text[0] not in ",;:":
        return None, ""
    for slot in _text_slots(output_block):
        value = _slot_value(slot)
        if not value.strip():
            continue
        match = re.match(r"^(\s*)[,;:]\s*", value)
        if not match:
            return None, ""
        return slot, value[: match.start()] + match.group(1) + value[match.end():]
    return None, ""


def _is_source_furniture(text: str) -> bool:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return True
    if re.fullmatch(r"[□■▪▫☐☑✓✔*•·.\-–—_~|/\\\s]+", value):
        return True
    if re.fullmatch(r"(?:p(?:age|ágina)?\s*)?\d{1,5}\.?", value, re.IGNORECASE):
        return True
    letters = sum(char.isalpha() for char in value)
    symbols = sum(not char.isalnum() and not char.isspace() for char in value)
    return bool(len(value) <= 12 and letters <= 3 and symbols >= max(2, letters))


def _block_content_mismatch(
    source_block: etree._Element,
    output_block: etree._Element,
    *,
    valid_reflowed_continuation: bool = False,
    valid_reflowed_anchor: bool = False,
) -> str:
    source_name = etree.QName(source_block).localname.lower()
    output_name = etree.QName(output_block).localname.lower()
    if source_name != output_name:
        return f"block element changed ({source_name} != {output_name})"

    source_text = _normalized_block_text(source_block)
    output_text = _normalized_block_text(output_block)
    source_is_meaningful = bool(
        re.search(r"[\w\d]", source_text, re.UNICODE)
        or _SCENE_BREAK_RE.fullmatch(source_text)
    )
    if (
        source_is_meaningful
        and not output_text
        and not _is_source_furniture(source_text)
        and not valid_reflowed_continuation
        and not _SANITIZED_ARTIFACT_CLASSES
        & set(str(output_block.get("class") or "").split())
    ):
        return "non-empty source block became empty"
    if not source_text or not output_text:
        return ""

    if source_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        if len(output_text) > max(180, len(source_text) * 6):
            return "heading absorbed body text"

    if len(source_text) >= 40:
        ratio = len(output_text) / len(source_text)
        if ratio < 0.28:
            return f"block content collapsed (length ratio {ratio:.2f})"
        if ratio > 4.5 and not valid_reflowed_anchor:
            return f"block absorbed neighboring content (length ratio {ratio:.2f})"
    return ""


def compare_xhtml_boundaries(
    source_root: etree._Element,
    output_root: etree._Element,
    *,
    file_href: str = "",
    repair: bool = False,
) -> BoundaryReport:
    """Compare corresponding DOM text slots and optionally restore lost spaces."""
    report = BoundaryReport(scanned_files=1)
    source_blocks = source_root.xpath(_BLOCK_XPATH)
    output_blocks = output_root.xpath(_BLOCK_XPATH)
    if len(source_blocks) != len(output_blocks):
        report.structural_mismatches.append(
            f"{file_href}: block count differs ({len(source_blocks)} != {len(output_blocks)})"
        )
        return report

    # Lazy import avoids an import cycle with the package-level EPUB rewriter.
    from .paragraph_reflow import (
        is_valid_marked_anchor,
        is_valid_marked_continuation,
    )

    for block_index, (source_block, output_block) in enumerate(zip(source_blocks, output_blocks)):
        valid_reflowed_continuation = False
        valid_reflowed_anchor = False
        if block_index > 0:
            valid_reflowed_continuation = is_valid_marked_continuation(
                source_blocks[block_index - 1],
                source_block,
                output_block,
            )
        valid_reflowed_anchor = is_valid_marked_anchor(
            source_blocks,
            output_blocks,
            block_index,
        )
        source_name = etree.QName(source_block).localname.lower()
        if source_name in {"td", "th"}:
            prefix_slot, repaired_value = _spurious_table_cell_prefix(
                source_block,
                output_block,
            )
            if prefix_slot is not None:
                finding = BoundaryFinding(
                    file_href=file_href,
                    block_index=block_index,
                    left_slot=0,
                    right_slot=0,
                    reason="invented_table_cell_prefix_punctuation",
                )
                if repair:
                    _set_slot_value(prefix_slot, repaired_value)
                    report.repaired_boundaries += 1
                else:
                    report.findings.append(finding)
        content_mismatch = _block_content_mismatch(
            source_block,
            output_block,
            valid_reflowed_continuation=valid_reflowed_continuation,
            valid_reflowed_anchor=valid_reflowed_anchor,
        )
        if content_mismatch:
            report.structural_mismatches.append(
                f"{file_href} block {block_index}: {content_mismatch}"
            )
        source_slots = _text_slots(source_block)
        output_slots = _text_slots(output_block)
        if len(source_slots) != len(output_slots):
            report.structural_mismatches.append(
                f"{file_href} block {block_index}: text-slot count differs "
                f"({len(source_slots)} != {len(output_slots)})"
            )
            continue
        source_values = [_slot_value(slot) for slot in source_slots]
        output_values = [_slot_value(slot) for slot in output_slots]
        content_slots = [index for index, value in enumerate(source_values) if value.strip()]
        for left, right in zip(content_slots, content_slots[1:]):
            report.scanned_boundaries += 1
            if not _has_whitespace_between(source_values, left, right):
                continue
            if not output_values[left].strip() or not output_values[right].strip():
                # Inline text may be reflowed into another slot in the same
                # visible block.  Slot emptiness alone is not structural loss;
                # the publication gate audits the aggregate block content.
                continue
            if _has_whitespace_between(output_values, left, right):
                continue
            if not _visible_tokens_need_space(output_values, left, right):
                continue
            finding = BoundaryFinding(
                file_href=file_href,
                block_index=block_index,
                left_slot=left,
                right_slot=right,
                reason="source_whitespace_lost",
            )
            if repair and _restore_boundary(source_values, output_slots, output_values, left, right):
                report.repaired_boundaries += 1
                output_values = [_slot_value(slot) for slot in output_slots]
            else:
                report.findings.append(finding)
    return report


def _parse_xhtml(payload: bytes) -> etree._Element:
    def replace(match: re.Match[bytes]) -> bytes:
        name = match.group(1)
        if name in _XML_BUILTINS:
            return match.group(0)
        codepoint = name2codepoint.get(name.decode("ascii", errors="ignore"))
        return f"&#{codepoint};".encode("ascii") if codepoint else match.group(0)

    normalized = _NAMED_ENTITY_RE.sub(replace, payload)
    return etree.fromstring(
        normalized,
        etree.XMLParser(
            recover=False,
            huge_tree=True,
            remove_blank_text=False,
            resolve_entities=False,
            load_dtd=False,
            no_network=True,
        ),
    )


def _serialize_xhtml(root: etree._Element, original: bytes) -> bytes:
    stripped = original.lstrip()
    doctype = root.getroottree().docinfo.doctype or None
    return etree.tostring(
        root.getroottree(),
        encoding="utf-8",
        xml_declaration=stripped.startswith(b"<?xml"),
        doctype=doctype,
        pretty_print=False,
    )


def _merge_report(target: BoundaryReport, source: BoundaryReport) -> None:
    target.scanned_files += source.scanned_files
    target.scanned_boundaries += source.scanned_boundaries
    target.repaired_boundaries += source.repaired_boundaries
    target.findings.extend(source.findings)
    target.structural_mismatches.extend(source.structural_mismatches)


def _write_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, payload: bytes) -> None:
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
    archive.writestr(clone, payload)


def audit_epub_dom_boundaries(source_epub: str | Path, output_epub: str | Path) -> BoundaryReport:
    """Audit an already-packaged EPUB against source DOM whitespace."""
    report = BoundaryReport()
    with zipfile.ZipFile(source_epub) as source, zipfile.ZipFile(output_epub) as output:
        source_names = set(source.namelist())
        for name in output.namelist():
            if Path(name).suffix.lower() not in _TEXT_SUFFIXES or name not in source_names:
                continue
            try:
                local = compare_xhtml_boundaries(
                    _parse_xhtml(source.read(name)),
                    _parse_xhtml(output.read(name)),
                    file_href=name,
                    repair=False,
                )
                _merge_report(report, local)
            except Exception as exc:
                report.structural_mismatches.append(f"{name}: {type(exc).__name__}: {exc}")
    return report


def find_epub_missing_text_blocks(
    source_epub: str | Path,
    output_epub: str | Path,
    *,
    limit: int = 32,
) -> list[MissingTextBlock]:
    """Return meaningful source blocks that became empty in the output DOM."""
    findings: list[MissingTextBlock] = []
    with zipfile.ZipFile(source_epub) as source, zipfile.ZipFile(output_epub) as output:
        source_names = set(source.namelist())
        for name in output.namelist():
            if (
                Path(name).suffix.lower() not in _TEXT_SUFFIXES
                or name not in source_names
            ):
                continue
            try:
                source_blocks = _parse_xhtml(source.read(name)).xpath(_BLOCK_XPATH)
                output_blocks = _parse_xhtml(output.read(name)).xpath(_BLOCK_XPATH)
            except Exception:
                continue
            if len(source_blocks) != len(output_blocks):
                continue

            from .paragraph_reflow import is_valid_marked_continuation

            for block_index, (source_block, output_block) in enumerate(
                zip(source_blocks, output_blocks)
            ):
                source_text = _normalized_block_text(source_block)
                output_text = _normalized_block_text(output_block)
                if not source_text or output_text or _is_source_furniture(source_text):
                    continue
                if _SANITIZED_ARTIFACT_CLASSES & set(
                    str(output_block.get("class") or "").split()
                ):
                    # The final sanitizer deliberately preserves the DOM slot
                    # while hiding reader-facing source furniture such as a
                    # standalone promotional URL. It is an explicit exclusion,
                    # not omitted translatable content.
                    continue
                source_is_meaningful = bool(
                    re.search(r"[\w\d]", source_text, re.UNICODE)
                    or _SCENE_BREAK_RE.fullmatch(source_text)
                )
                if not source_is_meaningful:
                    continue
                if block_index > 0 and is_valid_marked_continuation(
                    source_blocks[block_index - 1],
                    source_block,
                    output_block,
                ):
                    continue
                before = (
                    _normalized_block_text(source_blocks[block_index - 1])[-500:]
                    if block_index > 0
                    else ""
                )
                after = (
                    _normalized_block_text(source_blocks[block_index + 1])[:500]
                    if block_index + 1 < len(source_blocks)
                    else ""
                )
                findings.append(
                    MissingTextBlock(
                        file_href=name,
                        block_index=block_index,
                        element=etree.QName(source_block).localname.lower(),
                        source_text=source_text,
                        context_before=before,
                        context_after=after,
                    )
                )
                if len(findings) >= max(1, int(limit)):
                    return findings
    return findings


def apply_epub_missing_text_replacements(
    source_epub: str | Path,
    output_epub: str | Path,
    replacements: Mapping[tuple[str, int], str],
) -> int:
    """Atomically fill empty output blocks while preserving their XHTML structure."""
    normalized = {
        (str(file_href), int(block_index)): re.sub(r"\s+", " ", str(text or "")).strip()
        for (file_href, block_index), text in replacements.items()
        if str(text or "").strip()
    }
    if not normalized:
        return 0

    source_path = Path(source_epub)
    output_path = Path(output_epub)
    original_mode = stat.S_IMODE(output_path.stat().st_mode)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.", suffix=".epub", dir=output_path.parent, delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    changed = 0
    try:
        with (
            zipfile.ZipFile(source_path) as source,
            zipfile.ZipFile(output_path) as output,
            zipfile.ZipFile(tmp_path, "w") as rebuilt,
        ):
            source_names = set(source.namelist())
            by_file: dict[str, dict[int, str]] = {}
            for (file_href, block_index), text in normalized.items():
                by_file.setdefault(file_href, {})[block_index] = text

            for info in output.infolist():
                payload = output.read(info.filename)
                file_replacements = by_file.get(info.filename)
                if file_replacements and info.filename in source_names:
                    source_root = _parse_xhtml(source.read(info.filename))
                    output_root = _parse_xhtml(payload)
                    source_blocks = source_root.xpath(_BLOCK_XPATH)
                    output_blocks = output_root.xpath(_BLOCK_XPATH)
                    if len(source_blocks) == len(output_blocks):
                        for block_index, text in file_replacements.items():
                            if not 0 <= block_index < len(output_blocks):
                                continue
                            output_block = output_blocks[block_index]
                            if _normalized_block_text(output_block):
                                continue
                            source_slots = _text_slots(source_blocks[block_index])
                            output_slots = _text_slots(output_block)
                            target_slot = None
                            if len(source_slots) == len(output_slots):
                                for slot_index, source_slot in enumerate(source_slots):
                                    if _slot_value(source_slot).strip():
                                        target_slot = output_slots[slot_index]
                                        break
                            for slot in output_slots:
                                _set_slot_value(slot, "")
                            if target_slot is None:
                                output_block.text = text
                            else:
                                _set_slot_value(target_slot, text)
                            changed += 1
                        payload = _serialize_xhtml(output_root, payload)
                _write_member(rebuilt, info, payload)
        tmp_path.replace(output_path)
        output_path.chmod(original_mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return changed


def repair_epub_dom_boundaries(source_epub: str | Path, output_epub: str | Path) -> BoundaryReport:
    """Atomically restore only source-proven whitespace in an output EPUB."""
    source_path = Path(source_epub)
    output_path = Path(output_epub)
    mode = stat.S_IMODE(output_path.stat().st_mode)
    report = BoundaryReport()
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.", suffix=".epub", dir=output_path.parent, delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(output_path) as output, zipfile.ZipFile(tmp_path, "w") as rebuilt:
            source_names = set(source.namelist())
            for info in output.infolist():
                payload = output.read(info.filename)
                if Path(info.filename).suffix.lower() in _TEXT_SUFFIXES and info.filename in source_names:
                    try:
                        root = _parse_xhtml(payload)
                        local = compare_xhtml_boundaries(
                            _parse_xhtml(source.read(info.filename)),
                            root,
                            file_href=info.filename,
                            repair=True,
                        )
                        _merge_report(report, local)
                        if local.repaired_boundaries:
                            payload = _serialize_xhtml(root, payload)
                    except Exception as exc:
                        report.structural_mismatches.append(
                            f"{info.filename}: {type(exc).__name__}: {exc}"
                        )
                _write_member(rebuilt, info, payload)
        tmp_path.replace(output_path)
        output_path.chmod(mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return report
