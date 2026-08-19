"""Atomic, exact-match repairs for already packaged EPUB text units.

Repair rules are data, not book-specific code. A plan identifies an XHTML
element by archive path and DOM path, then replaces one exact text-slot value.
The operation refuses ambiguity or missing preconditions and never rewrites
binary resources, element structure, attributes, links or styles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import stat
import tempfile
from typing import Any, Iterable, Mapping
import zipfile

from lxml import etree

from .dom_boundaries import _parse_xhtml, _serialize_xhtml, _write_member


class EpubUnitRepairError(RuntimeError):
    """Raised before publication when a repair precondition is not exact."""


@dataclass(frozen=True)
class EpubUnitRepair:
    file_href: str
    dom_path: str
    old_text: str
    new_text: str
    expected_occurrences: int = 1
    unit_id: str = ""
    reason: str = ""
    operation: str = "replace_text"
    inline_markers: tuple[str, ...] = ()
    expected_visible_sha256: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EpubUnitRepair":
        markers = payload.get("inline_markers") or []
        if not isinstance(markers, list):
            raise EpubUnitRepairError("inline_markers must be a list")
        repair = cls(
            file_href=str(payload.get("file_href") or payload.get("source_document") or "").strip(),
            dom_path=str(payload.get("dom_path") or "").strip(),
            old_text=str(payload.get("old_text") or ""),
            new_text=str(payload.get("new_text") or ""),
            expected_occurrences=int(payload.get("expected_occurrences") or 1),
            unit_id=str(payload.get("unit_id") or "").strip(),
            reason=str(payload.get("reason") or "").strip(),
            operation=str(payload.get("operation") or "replace_text").strip().casefold(),
            inline_markers=tuple(str(item) for item in markers),
            expected_visible_sha256=str(payload.get("expected_visible_sha256") or "").strip().lower(),
        )
        if not repair.file_href or not repair.dom_path:
            raise EpubUnitRepairError("repair requires file_href and dom_path")
        if repair.operation not in {"replace_text", "realign_inline_markers"}:
            raise EpubUnitRepairError(f"unsupported repair operation: {repair.operation}")
        if repair.operation == "replace_text" and not repair.old_text:
            raise EpubUnitRepairError("repair old_text must be non-empty")
        if repair.operation == "realign_inline_markers":
            if not repair.inline_markers:
                raise EpubUnitRepairError("inline realignment requires inline_markers")
            if not repair.expected_visible_sha256:
                raise EpubUnitRepairError("inline realignment requires expected_visible_sha256")
        if repair.expected_occurrences < 1:
            raise EpubUnitRepairError("expected_occurrences must be at least one")
        return repair


@dataclass
class EpubUnitRepairReport:
    input_path: str
    output_path: str
    input_sha256: str
    output_sha256: str = ""
    requested_repairs: int = 0
    applied_repairs: int = 0
    changed_files: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "requested_repairs": self.requested_repairs,
            "applied_repairs": self.applied_repairs,
            "changed_files": self.changed_files,
            "entries": self.entries,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _apply_to_element(element: etree._Element, repair: EpubUnitRepair) -> int:
    if repair.operation == "realign_inline_markers":
        return _realign_inline_markers(element, repair)
    matches: list[tuple[etree._Element, str, int]] = []
    for node, attribute in _text_slots(element):
        value = str(getattr(node, attribute) or "")
        count = value.count(repair.old_text)
        if count:
            matches.append((node, attribute, count))
    total = sum(count for _node, _attribute, count in matches)
    if total != repair.expected_occurrences:
        label = repair.unit_id or f"{repair.file_href} {repair.dom_path}"
        raise EpubUnitRepairError(
            f"repair precondition failed for {label}: expected "
            f"{repair.expected_occurrences} exact occurrence(s), found {total}"
        )
    for node, attribute, _count in matches:
        value = str(getattr(node, attribute) or "")
        setattr(node, attribute, value.replace(repair.old_text, repair.new_text))
    return total


def _realign_inline_markers(element: etree._Element, repair: EpubUnitRepair) -> int:
    visible = "".join(element.itertext())
    visible_hash = hashlib.sha256(visible.encode("utf-8")).hexdigest()
    if visible_hash != repair.expected_visible_sha256:
        raise EpubUnitRepairError(
            f"visible-text hash mismatch for {repair.file_href} {repair.dom_path}: "
            f"expected {repair.expected_visible_sha256}, found {visible_hash}"
        )
    children = [child for child in element if isinstance(child.tag, str)]
    if len(children) != len(repair.inline_markers) or any(len(child) for child in children):
        raise EpubUnitRepairError(
            f"inline marker count/shape mismatch for {repair.file_href} {repair.dom_path}: "
            f"children={len(children)}, markers={len(repair.inline_markers)}"
        )

    positions: list[tuple[int, int]] = []
    cursor = 0
    for marker in repair.inline_markers:
        if not marker or visible.count(marker) != 1:
            raise EpubUnitRepairError(
                f"inline marker must occur exactly once in visible text: {marker!r}"
            )
        start = visible.find(marker, cursor)
        if start < cursor:
            raise EpubUnitRepairError(f"inline markers are out of order: {marker!r}")
        positions.append((start, start + len(marker)))
        cursor = start + len(marker)

    element.text = visible[:positions[0][0]]
    for index, child in enumerate(children):
        start, end = positions[index]
        child.text = visible[start:end]
        next_start = positions[index + 1][0] if index + 1 < len(positions) else len(visible)
        child.tail = visible[end:next_start]
    return 1


def load_epub_unit_repair_plan(path: str | Path) -> tuple[list[EpubUnitRepair], str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_repairs = payload
        expected_hash = ""
    elif isinstance(payload, Mapping):
        raw_repairs = payload.get("repairs") or []
        expected_hash = str(payload.get("expected_epub_sha256") or "").strip().lower()
    else:
        raise EpubUnitRepairError("repair plan must be a JSON object or list")
    if not isinstance(raw_repairs, list):
        raise EpubUnitRepairError("repair plan repairs must be a list")
    return [EpubUnitRepair.from_mapping(item) for item in raw_repairs], expected_hash


def apply_epub_unit_repairs(
    input_epub: str | Path,
    repairs: Iterable[EpubUnitRepair | Mapping[str, Any]],
    *,
    output_epub: str | Path | None = None,
    expected_epub_sha256: str = "",
) -> EpubUnitRepairReport:
    """Apply a validated plan atomically while preserving the EPUB package."""
    input_path = Path(input_epub)
    output_path = Path(output_epub) if output_epub else input_path
    input_hash = _sha256(input_path)
    expected = str(expected_epub_sha256 or "").strip().lower()
    if expected and input_hash != expected:
        raise EpubUnitRepairError(
            f"EPUB hash mismatch: expected {expected}, found {input_hash}"
        )
    normalized = [
        item if isinstance(item, EpubUnitRepair) else EpubUnitRepair.from_mapping(item)
        for item in repairs
    ]
    if not normalized:
        raise EpubUnitRepairError("repair plan is empty")

    by_file: dict[str, list[EpubUnitRepair]] = {}
    for repair in normalized:
        by_file.setdefault(repair.file_href, []).append(repair)

    report = EpubUnitRepairReport(
        input_path=str(input_path),
        output_path=str(output_path),
        input_sha256=input_hash,
        requested_repairs=len(normalized),
    )
    mode = stat.S_IMODE(input_path.stat().st_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.", suffix=".epub", dir=output_path.parent, delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(input_path) as source:
            names = set(source.namelist())
            missing = sorted(set(by_file) - names)
            if missing:
                raise EpubUnitRepairError(f"repair file(s) missing from EPUB: {missing}")
            rewritten: dict[str, bytes] = {}
            for file_href, file_repairs in by_file.items():
                original = source.read(file_href)
                root = _parse_xhtml(original)
                for repair in file_repairs:
                    matches = root.xpath(repair.dom_path)
                    if len(matches) != 1 or not isinstance(matches[0], etree._Element):
                        raise EpubUnitRepairError(
                            f"DOM path must resolve exactly once: {file_href} {repair.dom_path} "
                            f"(found {len(matches)})"
                        )
                    applied = _apply_to_element(matches[0], repair)
                    report.applied_repairs += applied
                    report.entries.append({
                        "unit_id": repair.unit_id,
                        "file_href": repair.file_href,
                        "dom_path": repair.dom_path,
                        "reason": repair.reason,
                        "operation": repair.operation,
                        "occurrences": applied,
                    })
                rewritten[file_href] = _serialize_xhtml(root, original)
                report.changed_files.append(file_href)

            with zipfile.ZipFile(tmp_path, "w") as destination:
                for info in source.infolist():
                    _write_member(
                        destination,
                        info,
                        rewritten.get(info.filename, source.read(info.filename)),
                    )
        tmp_path.replace(output_path)
        output_path.chmod(mode)
        report.output_sha256 = _sha256(output_path)
        return report
    finally:
        tmp_path.unlink(missing_ok=True)
