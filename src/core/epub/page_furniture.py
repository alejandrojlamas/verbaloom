"""Source-aware cleanup for OCR page furniture in reconstructed EPUBs.

The cleaner never deletes DOM slots. It hides only blocks whose source/output
neighbours prove that the reader-facing content was already preserved, keeping
the source/output structure auditable by the publication gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import stat
import tempfile
import zipfile

from lxml import etree

from .dom_boundaries import _BLOCK_XPATH, _parse_xhtml, _serialize_xhtml, _write_member


_TEXT_SUFFIXES = {".xhtml", ".html", ".htm"}
_SANITIZED_CLASS = "verbaloom-sanitized-artifact"
_INLINE_HEADER_TAGS = {"b", "em", "i", "span", "strong"}
_OCR_PAGE_MARKER_RE = re.compile(r"^[\[(]?[0-9ILOZST$^°|]{1,8}[\]).]?$", re.IGNORECASE)
_TERMINAL_RE = re.compile(r"[.!?…:;»”’'\"]\s*$")


@dataclass
class PageFurnitureReport:
    scanned_files: int = 0
    sanitized_blocks: int = 0
    running_headers: int = 0
    page_markers: int = 0
    errors: list[str] = field(default_factory=list)


def _text(node: etree._Element) -> str:
    return re.sub(r"\s+", " ", " ".join(node.itertext())).strip()


def _is_substantial_prose(value: str) -> bool:
    return len(value) >= 80 and len(re.findall(r"\b\w+\b", value, re.UNICODE)) >= 12


def _looks_like_ocr_page_marker(value: str) -> bool:
    compact = re.sub(r"\s+", "", value or "")
    return bool(
        compact
        and any(char.isdigit() for char in compact)
        and _OCR_PAGE_MARKER_RE.fullmatch(compact)
    )


def _looks_like_running_header(value: str) -> bool:
    letters = [char for char in value if char.isalpha()]
    if not 3 <= len(letters) <= 64:
        return False
    uppercase_ratio = sum(char.isupper() for char in letters) / len(letters)
    return bool(
        uppercase_ratio >= 0.82
        and ("•" in value or "|" in value or re.search(r"\b[IVXLCDM0-9]+\b", value))
    )


def _leading_inline_header_and_continuation(
    block: etree._Element,
) -> tuple[str, str]:
    if (block.text or "").strip() or not len(block):
        return "", ""
    child = block[0]
    if not isinstance(child.tag, str):
        return "", ""
    if etree.QName(child).localname.lower() not in _INLINE_HEADER_TAGS:
        return "", ""
    header = _text(child)
    continuation = re.sub(r"\s+", " ", str(child.tail or "")).strip()
    return header, continuation


def _is_mixed_running_header_continuation(
    source_blocks: list[etree._Element],
    output_blocks: list[etree._Element],
    index: int,
) -> bool:
    if index <= 0:
        return False
    header, continuation = _leading_inline_header_and_continuation(source_blocks[index])
    if not header or len(continuation) < 30:
        return False
    if not _looks_like_running_header(header) or not continuation[:1].islower():
        return False

    source_previous = _text(source_blocks[index - 1])
    output_previous = _text(output_blocks[index - 1])
    output_current = _text(output_blocks[index])
    if not _is_substantial_prose(source_previous) or not _is_substantial_prose(output_previous):
        return False
    if _TERMINAL_RE.search(source_previous):
        return False
    return bool(
        len(output_current) <= 100
        and _looks_like_running_header(output_current)
        and _TERMINAL_RE.search(output_previous)
    )


def _is_standalone_page_marker(
    source_blocks: list[etree._Element],
    output_blocks: list[etree._Element],
    index: int,
) -> bool:
    if index <= 0 or index + 1 >= len(source_blocks):
        return False
    source_value = _text(source_blocks[index])
    output_value = _text(output_blocks[index])
    return bool(
        _looks_like_ocr_page_marker(source_value)
        and _looks_like_ocr_page_marker(output_value)
        and _is_substantial_prose(_text(source_blocks[index - 1]))
        and _is_substantial_prose(_text(source_blocks[index + 1]))
    )


def _hide_preserving_slot(node: etree._Element) -> None:
    node.text = ""
    for child in node.iterdescendants():
        child.text = ""
        child.tail = ""
    classes = str(node.get("class") or "").split()
    if _SANITIZED_CLASS not in classes:
        classes.append(_SANITIZED_CLASS)
    node.set("class", " ".join(classes))
    style = str(node.get("style") or "").strip().rstrip(";")
    declarations = [item.strip() for item in style.split(";") if item.strip()]
    if not any(item.casefold().startswith("display:") for item in declarations):
        declarations.append("display: none")
    node.set("style", "; ".join(declarations))


def sanitize_epub_page_furniture(
    source_epub: str | Path,
    output_epub: str | Path,
) -> PageFurnitureReport:
    """Hide source-proven OCR page furniture without changing DOM shape."""
    source_path = Path(source_epub)
    output_path = Path(output_epub)
    report = PageFurnitureReport()
    if not source_path.exists() or not output_path.exists():
        report.errors.append("Source or output EPUB does not exist.")
        return report

    mode = stat.S_IMODE(output_path.stat().st_mode)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=".epub",
        dir=output_path.parent,
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    changed = False
    try:
        with (
            zipfile.ZipFile(source_path) as source,
            zipfile.ZipFile(output_path) as output,
            zipfile.ZipFile(tmp_path, "w") as rebuilt,
        ):
            source_names = set(source.namelist())
            for info in output.infolist():
                payload = output.read(info.filename)
                if Path(info.filename).suffix.lower() in _TEXT_SUFFIXES and info.filename in source_names:
                    report.scanned_files += 1
                    try:
                        source_root = _parse_xhtml(source.read(info.filename))
                        output_root = _parse_xhtml(payload)
                        source_blocks = source_root.xpath(_BLOCK_XPATH)
                        output_blocks = output_root.xpath(_BLOCK_XPATH)
                        if len(source_blocks) != len(output_blocks):
                            raise ValueError(
                                f"block count differs ({len(source_blocks)} != {len(output_blocks)})"
                            )
                        local_changed = False
                        for index, output_block in enumerate(output_blocks):
                            if _SANITIZED_CLASS in str(output_block.get("class") or "").split():
                                continue
                            kind = ""
                            if _is_mixed_running_header_continuation(
                                source_blocks, output_blocks, index
                            ):
                                kind = "running_header"
                            elif _is_standalone_page_marker(
                                source_blocks, output_blocks, index
                            ):
                                kind = "page_marker"
                            if not kind:
                                continue
                            _hide_preserving_slot(output_block)
                            report.sanitized_blocks += 1
                            report.running_headers += int(kind == "running_header")
                            report.page_markers += int(kind == "page_marker")
                            local_changed = True
                        if local_changed:
                            payload = _serialize_xhtml(output_root, payload)
                            changed = True
                    except Exception as exc:
                        report.errors.append(f"{info.filename}: {type(exc).__name__}: {exc}")
                _write_member(rebuilt, info, payload)
        if changed:
            tmp_path.replace(output_path)
            output_path.chmod(mode)
        else:
            tmp_path.unlink(missing_ok=True)
    finally:
        tmp_path.unlink(missing_ok=True)
    return report
