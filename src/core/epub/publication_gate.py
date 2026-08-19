"""Deterministic whole-EPUB publication gate and unit manifest builder."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import difflib
import hashlib
from io import BytesIO
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlsplit
import zipfile

from lxml import etree

from src.core.fidelity_supervisor import assess_fidelity, target_language_gate_issues
from src.core.language_evidence import (
    has_target_language_contextual_evidence,
    looks_like_preserved_title_or_citation_sequence,
    looks_like_structured_language_metadata,
    looks_like_translated_reference_context,
)
from src.utils.language_detector import LanguageDetector

from .lang_support import get_language_code
from .dom_boundaries import audit_epub_dom_boundaries
from .unit_contract import EPUB_PIPELINE_VERSION, EPUB_PROMPT_VERSION, stable_unit_id, text_sha256


_BLOCK_XPATH = (
    "//*[local-name()='p' or local-name()='h1' or local-name()='h2' or "
    "local-name()='h3' or local-name()='h4' or local-name()='h5' or "
    "local-name()='h6' or local-name()='li' or local-name()='blockquote' or "
    "local-name()='figcaption' or local-name()='td' or local-name()='th']"
)
_PLACEHOLDER_RE = re.compile(
    r"\[id\d+\]|\[\[\d+\]\]|<\s*/?\s*(?:translation(?:ation)*|source|target|input|output)\b",
    re.IGNORECASE,
)
_BAD_SPACING_RE = re.compile(
    r"(?<=[a-záéíóúüñ0-9])[.!?](?=[A-ZÁÉÍÓÚÜÑ])|"
    r"\b[IVXLCDM]{2,}\.(?=[a-záéíóúüñ])|"
    r"\s+[.,;:!?](?=[A-Za-zÁÉÍÓÚÜÑáéíóúüñ])"
)
_TEXT_SUFFIXES = {".xhtml", ".html", ".htm"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".avif"}
_FONT_SUFFIXES = {".otf", ".ttf", ".woff", ".woff2"}
_EXTERNAL_SCHEMES = {"http", "https", "mailto", "tel", "data"}
_KNOWN_EPUBCHECK_PATHS = (
    Path("/opt/homebrew/bin/epubcheck"),
    Path("/usr/local/bin/epubcheck"),
)


@dataclass(frozen=True)
class EpubUnit:
    file_href: str
    ordinal: int
    source_text: str
    text: str
    element: str
    spine_index: int = 0
    dom_path: str = ""
    source_order: int = 0

    @property
    def source_hash(self) -> str:
        return text_sha256(self.source_text)

    @property
    def unit_id(self) -> str:
        return stable_unit_id(
            self.file_href,
            self.ordinal,
            self.source_hash,
            spine_index=self.spine_index,
            dom_path=self.dom_path,
            source_order=self.source_order,
        )


@dataclass
class EpubSnapshot:
    path: str
    entry_names: list[str]
    opf_path: str
    spine: list[str]
    languages: list[str]
    xhtml_files: list[str]
    image_files: list[str]
    units: list[EpubUnit]
    element_counts: dict[str, Counter]
    image_refs: dict[str, list[str]]
    links: list[tuple[str, str]]
    ids: dict[str, set[str]]
    html_languages: dict[str, tuple[str, str]]
    mobile_viewports: dict[str, str]
    file_texts: dict[str, str]
    duplicate_id_count: int
    resource_hashes: dict[str, str]
    mimetype_first_stored: bool
    parse_errors: list[str] = field(default_factory=list)
    package_version: str = ""
    manifest_count: int = 0
    manifest_resources: dict[str, dict[str, str]] = field(default_factory=dict)
    css_files: list[str] = field(default_factory=list)
    font_files: list[str] = field(default_factory=list)
    nav_files: list[str] = field(default_factory=list)
    ncx_files: list[str] = field(default_factory=list)
    toc_entries: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, list[str]] = field(default_factory=dict)
    mimetype_exact: bool = False
    unsafe_paths: list[str] = field(default_factory=list)
    temporary_files: list[str] = field(default_factory=list)
    cover_image: str = ""
    cover_page: str = ""
    obvious_cover_image: str = ""
    sanitized_artifact_paths: dict[str, set[str]] = field(default_factory=dict)


@dataclass
class PublicationGateReport:
    source_path: str
    output_path: str
    target_language: str
    source_units: int = 0
    output_units: int = 0
    audited_units: int = 0
    source_language_units: int = 0
    mixed_language_units: int = 0
    duplicate_units: int = 0
    missing_units: int = 0
    reflowed_units: int = 0
    placeholder_findings: int = 0
    spacing_findings: int = 0
    dom_boundary_findings: int = 0
    dom_boundary_structural_mismatches: int = 0
    broken_links: int = 0
    inherited_broken_links: int = 0
    inherited_broken_toc_targets: int = 0
    preserved_images: int = 0
    preserved_resources: int = 0
    epubcheck_errors: int = 0
    inherited_epubcheck_errors: int = 0
    epubcheck_warnings: int = 0
    epubcheck_output: str = ""
    cover_image: str = ""
    cover_page: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    units: list[dict[str, Any]] = field(default_factory=list)

    @property
    def publishable(self) -> bool:
        return not self.errors and self.source_units == self.audited_units

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "pipeline_version": EPUB_PIPELINE_VERSION,
            "prompt_version": EPUB_PROMPT_VERSION,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "target_language": self.target_language,
            "publishable": self.publishable,
            "source_units": self.source_units,
            "output_units": self.output_units,
            "audited_units": self.audited_units,
            "source_language_units": self.source_language_units,
            "mixed_language_units": self.mixed_language_units,
            "duplicate_units": self.duplicate_units,
            "missing_units": self.missing_units,
            "reflowed_units": self.reflowed_units,
            "placeholder_findings": self.placeholder_findings,
            "spacing_findings": self.spacing_findings,
            "dom_boundary_findings": self.dom_boundary_findings,
            "dom_boundary_structural_mismatches": self.dom_boundary_structural_mismatches,
            "broken_links": self.broken_links,
            "inherited_broken_links": self.inherited_broken_links,
            "inherited_broken_toc_targets": self.inherited_broken_toc_targets,
            "preserved_images": self.preserved_images,
            "preserved_resources": self.preserved_resources,
            "epubcheck_errors": self.epubcheck_errors,
            "inherited_epubcheck_errors": self.inherited_epubcheck_errors,
            "epubcheck_warnings": self.epubcheck_warnings,
            "epubcheck_output": self.epubcheck_output,
            "cover_image": self.cover_image,
            "cover_page": self.cover_page,
            "errors": self.errors,
            "warnings": self.warnings,
            "units": self.units,
        }

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output


def _unsafe_archive_path(name: str) -> bool:
    normalized = str(name or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    return bool(
        normalized.startswith("/")
        or normalized.startswith("\\")
        or (parts and ":" in parts[0])
        or ".." in parts
    )


def _temporary_archive_member(name: str) -> bool:
    basename = posixpath.basename(str(name or ""))
    lowered = basename.casefold()
    return bool(
        lowered in {".ds_store", "thumbs.db"}
        or basename.startswith("._")
        or lowered.endswith((".tmp", ".temp", ".bak", ".swp", "~"))
    )


def _read_toc_entries(
    archive: zipfile.ZipFile,
    nav_files: list[str],
    ncx_files: list[str],
) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for toc_path in [*nav_files, *ncx_files]:
        if toc_path not in archive.namelist():
            continue
        try:
            root = etree.fromstring(
                archive.read(toc_path),
                etree.XMLParser(recover=False, huge_tree=True, remove_blank_text=False),
            )
        except Exception:
            continue
        toc_dir = posixpath.dirname(toc_path)
        if toc_path in ncx_files:
            nodes = root.xpath("//*[local-name()='navPoint']")
            for node in nodes:
                label = " ".join(node.xpath("./*[local-name()='navLabel']//*[local-name()='text']/text()"))
                href = str(node.xpath("string(./*[local-name()='content']/@src)") or "")
                if href:
                    entries.append((re.sub(r"\s+", " ", label).strip(), _resolve_toc_href(toc_dir, href)))
        else:
            nav_nodes = root.xpath(
                "//*[local-name()='nav' and "
                "(@*[local-name()='type']='toc' or contains(concat(' ', normalize-space(@*[local-name()='type']), ' '), ' toc '))]"
            )
            for anchor in [a for nav in nav_nodes for a in nav.xpath(".//*[local-name()='a'][@href]")]:
                label = re.sub(r"\s+", " ", " ".join(anchor.itertext())).strip()
                entries.append((label, _resolve_toc_href(toc_dir, str(anchor.get("href") or ""))))
    return entries


def _resolve_toc_href(base_dir: str, href: str) -> str:
    parsed = urlsplit(href)
    path = posixpath.normpath(posixpath.join(base_dir, unquote(parsed.path))) if parsed.path else ""
    return f"{path}#{parsed.fragment}" if parsed.fragment else path


def _broken_toc_targets(snapshot: EpubSnapshot) -> list[str]:
    broken: list[str] = []
    names = set(snapshot.entry_names)
    for _label, target in snapshot.toc_entries:
        path, _, fragment = target.partition("#")
        if not path or path not in names:
            broken.append(target)
        elif fragment and fragment not in snapshot.ids.get(path, set()):
            broken.append(target)
    return broken


def _probable_undeclared_cover(
    archive: zipfile.ZipFile,
    *,
    spine: list[str],
    image_refs: dict[str, list[str]],
) -> str:
    if not spine or not image_refs.get(spine[0]):
        return ""
    first_doc = spine[0]
    candidate = posixpath.normpath(
        posixpath.join(posixpath.dirname(first_doc), image_refs[first_doc][0])
    )
    if candidate not in archive.namelist() or Path(candidate).suffix.lower() not in _IMAGE_SUFFIXES:
        return ""
    try:
        from PIL import Image

        with Image.open(BytesIO(archive.read(candidate))) as image:
            width, height = image.size
        ratio = width / max(1, height)
        if width >= 250 and height >= 350 and 0.42 <= ratio <= 0.95:
            return candidate
    except Exception:
        return ""
    return ""


def snapshot_epub(path: str | Path, *, source_texts: Optional[list[str]] = None, recover: bool = False) -> EpubSnapshot:
    epub_path = Path(path)
    with zipfile.ZipFile(epub_path) as archive:
        infos = archive.infolist()
        # ZIP directory entries are packaging details, not EPUB resources.
        # Rebuilders commonly omit them while preserving every real file.
        file_infos = [item for item in infos if not item.is_dir()]
        names = [item.filename for item in file_infos]
        container = etree.fromstring(archive.read("META-INF/container.xml"))
        opf_path = container.xpath("string(//*[local-name()='rootfile']/@full-path)")
        package = etree.fromstring(archive.read(opf_path))
        opf_dir = posixpath.dirname(opf_path)
        manifest_nodes = package.xpath("//*[local-name()='manifest']/*[local-name()='item']")
        manifest = {item.get("id"): item.get("href") for item in manifest_nodes}
        manifest_resources = {
            posixpath.normpath(posixpath.join(opf_dir, str(item.get("href") or ""))): {
                "id": str(item.get("id") or ""),
                "media_type": str(item.get("media-type") or ""),
                "properties": str(item.get("properties") or ""),
            }
            for item in manifest_nodes
            if item.get("href")
        }
        spine = [
            posixpath.normpath(posixpath.join(opf_dir, manifest[item.get("idref")]))
            for item in package.xpath("//*[local-name()='spine']/*[local-name()='itemref']")
            if manifest.get(item.get("idref"))
        ]
        languages = [str(value).strip() for value in package.xpath("//*[local-name()='language']/text()")]
        xhtml_files = [name for name in names if Path(name).suffix.lower() in _TEXT_SUFFIXES]
        image_files = [name for name in names if Path(name).suffix.lower() in _IMAGE_SUFFIXES]
        css_files = [name for name in names if Path(name).suffix.lower() == ".css"]
        font_files = [name for name in names if Path(name).suffix.lower() in _FONT_SUFFIXES]
        nav_files = [
            path for path, item in manifest_resources.items()
            if "nav" in item.get("properties", "").split()
        ]
        ncx_files = [
            path for path, item in manifest_resources.items()
            if item.get("media_type") == "application/x-dtbncx+xml"
        ]
        toc_entries = _read_toc_entries(archive, nav_files, ncx_files)
        metadata = {
            key: [str(value).strip() for value in package.xpath(f"//*[local-name()='{key}']/text()") if str(value).strip()]
            for key in ("title", "creator", "identifier", "publisher", "date", "rights", "language")
        }
        manifest_by_id = {
            str(item.get("id") or ""): path
            for path, item in manifest_resources.items()
            if item.get("id")
        }
        cover_id = str(package.xpath(
            "string(//*[local-name()='metadata']/*[local-name()='meta' and "
            "translate(@name, 'COVER', 'cover')='cover']/@content)"
        ) or "")
        cover_image = manifest_by_id.get(cover_id, "")
        if not cover_image:
            cover_image = next(
                (
                    path for path, item in manifest_resources.items()
                    if "cover-image" in item.get("properties", "").split()
                ),
                "",
            )
        cover_page_href = str(package.xpath(
            "string(//*[local-name()='guide']/*[local-name()='reference' and "
            "translate(@type, 'COVER', 'cover')='cover']/@href)"
        ) or "")
        cover_page = (
            posixpath.normpath(posixpath.join(opf_dir, unquote(urlsplit(cover_page_href).path)))
            if cover_page_href
            else ""
        )
        units: list[EpubUnit] = []
        element_counts: dict[str, Counter] = {}
        image_refs: dict[str, list[str]] = {}
        links: list[tuple[str, str]] = []
        ids: dict[str, set[str]] = {}
        html_languages: dict[str, tuple[str, str]] = {}
        mobile_viewports: dict[str, str] = {}
        file_texts: dict[str, str] = {}
        duplicate_id_count = 0
        parse_errors: list[str] = []
        sanitized_artifact_paths: dict[str, set[str]] = {}
        source_index = 0

        for spine_index, file_href in enumerate(spine):
            if Path(file_href).suffix.lower() not in _TEXT_SUFFIXES:
                continue
            try:
                root = etree.fromstring(
                    archive.read(file_href),
                    etree.XMLParser(recover=recover, huge_tree=True, remove_blank_text=False),
                )
            except Exception as exc:
                parse_errors.append(f"{file_href}: {exc}")
                continue
            tags = [etree.QName(item).localname.lower() for item in root.iter() if isinstance(item.tag, str)]
            element_counts[file_href] = Counter(tags)
            raw_ids = [str(value) for value in root.xpath("//@id") if value]
            ids[file_href] = set(raw_ids)
            duplicate_id_count += len(raw_ids) - len(ids[file_href])
            html_languages[file_href] = (
                str(root.get("lang") or ""),
                str(root.get("{http://www.w3.org/XML/1998/namespace}lang") or ""),
            )
            mobile_viewports[file_href] = str(root.xpath(
                "string(//*[local-name()='head']/*[local-name()='meta' and "
                "translate(@name, 'VIEWPORT', 'viewport')='viewport']/@content)"
            ) or "")
            body_nodes = root.xpath("//*[local-name()='body']")
            visible_root = body_nodes[0] if body_nodes else root
            file_texts[file_href] = _normalized_text(" ".join(visible_root.itertext()))
            sanitized_artifact_paths[file_href] = {
                element.getroottree().getpath(element)
                for element in root.xpath(
                    "//*[contains(concat(' ', normalize-space(@class), ' '), "
                    "' verbaloom-sanitized-artifact ') or "
                    "contains(concat(' ', normalize-space(@class), ' '), "
                    "' tbl-sanitized-artifact ')]"
                )
            }
            image_refs[file_href] = [str(value) for value in root.xpath("//*[local-name()='img']/@src")]
            links.extend((file_href, str(value)) for value in root.xpath("//@href") if value)
            for ordinal, element in enumerate(root.xpath(_BLOCK_XPATH)):
                text = _normalized_text(" ".join(element.itertext()))
                if not text:
                    continue
                source_text = source_texts[source_index] if source_texts and source_index < len(source_texts) else text
                source_index += 1
                units.append(EpubUnit(
                    file_href=file_href,
                    ordinal=ordinal,
                    source_text=source_text,
                    text=text,
                    element=etree.QName(element).localname.lower(),
                    spine_index=spine_index,
                    dom_path=element.getroottree().getpath(element),
                    source_order=source_index - 1,
                ))

        resource_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names
            if Path(name).suffix.lower() not in _TEXT_SUFFIXES and name != opf_path
        }
        first_stored = bool(
            infos
            and infos[0].filename == "mimetype"
            and infos[0].compress_type == zipfile.ZIP_STORED
        )
        mimetype_exact = bool("mimetype" in names and archive.read("mimetype") == b"application/epub+zip")
        unsafe_paths = [name for name in names if _unsafe_archive_path(name)]
        temporary_files = [name for name in names if _temporary_archive_member(name)]
        obvious_cover_image = _probable_undeclared_cover(
            archive,
            spine=spine,
            image_refs=image_refs,
        )
    return EpubSnapshot(
        path=str(epub_path),
        entry_names=names,
        opf_path=opf_path,
        spine=spine,
        languages=languages,
        xhtml_files=xhtml_files,
        image_files=image_files,
        units=units,
        element_counts=element_counts,
        image_refs=image_refs,
        links=links,
        ids=ids,
        html_languages=html_languages,
        mobile_viewports=mobile_viewports,
        file_texts=file_texts,
        duplicate_id_count=duplicate_id_count,
        resource_hashes=resource_hashes,
        mimetype_first_stored=first_stored,
        parse_errors=parse_errors,
        package_version=str(package.get("version") or ""),
        manifest_count=len(manifest_nodes),
        manifest_resources=manifest_resources,
        css_files=css_files,
        font_files=font_files,
        nav_files=nav_files,
        ncx_files=ncx_files,
        toc_entries=toc_entries,
        metadata=metadata,
        mimetype_exact=mimetype_exact,
        unsafe_paths=unsafe_paths,
        temporary_files=temporary_files,
        cover_image=cover_image,
        cover_page=cover_page,
        obvious_cover_image=obvious_cover_image,
        sanitized_artifact_paths=sanitized_artifact_paths,
    )


def audit_epub_publication(
    source_path: str | Path,
    output_path: str | Path,
    *,
    source_language: str,
    target_language: str,
    broken_path: str | Path | None = None,
    epubcheck_command: Optional[list[str]] = None,
) -> PublicationGateReport:
    report = PublicationGateReport(str(source_path), str(output_path), target_language)
    source = snapshot_epub(source_path, recover=True)
    output = snapshot_epub(output_path, recover=False)
    broken = snapshot_epub(broken_path, recover=True) if broken_path else None
    boundary_report = audit_epub_dom_boundaries(source_path, output_path)
    report.dom_boundary_findings = len(boundary_report.findings)
    report.dom_boundary_structural_mismatches = len(boundary_report.structural_mismatches)
    report.spacing_findings += report.dom_boundary_findings
    if boundary_report.structural_mismatches:
        report.errors.append(
            f"{len(boundary_report.structural_mismatches)} DOM text-slot structure mismatch(es)"
        )

    report.source_units = len(source.units)
    report.output_units = len(output.units)
    report.cover_image = output.cover_image
    report.cover_page = output.cover_page
    if not output.mimetype_first_stored:
        report.errors.append("mimetype is not the first uncompressed ZIP entry")
    if not output.mimetype_exact:
        report.errors.append("mimetype content is not exactly application/epub+zip")
    if output.unsafe_paths:
        report.errors.append(f"unsafe ZIP paths found: {output.unsafe_paths[:5]}")
    if output.temporary_files:
        report.errors.append(f"temporary files found inside EPUB: {output.temporary_files[:5]}")
    if output.parse_errors:
        report.errors.extend(f"invalid XHTML: {item}" for item in output.parse_errors)
    expected_code = get_language_code(target_language)
    if not expected_code or expected_code not in output.languages:
        report.errors.append(f"dc:language must be {expected_code or target_language}; found {output.languages}")
    wrong_html_languages = [
        file_href
        for file_href, values in output.html_languages.items()
        if expected_code and values != (expected_code, expected_code)
    ]
    if wrong_html_languages:
        report.errors.append(
            f"XHTML lang/xml:lang must be {expected_code}; invalid files={wrong_html_languages[:5]}"
        )
    if output.duplicate_id_count:
        report.errors.append(f"{output.duplicate_id_count} duplicate XHTML id attribute(s)")
    if source.spine != output.spine:
        report.errors.append("spine order differs from the source EPUB")
    if set(source.entry_names) != set(output.entry_names):
        missing = sorted(set(source.entry_names) - set(output.entry_names))
        added = sorted(set(output.entry_names) - set(source.entry_names))
        allowed_added = [
            name for name in added
            if Path(name).name in {"verbaloom-professional.css", "tbl-professional.css"}
            and output.manifest_resources.get(name, {}).get("media_type") == "text/css"
        ]
        unexpected_added = sorted(set(added) - set(allowed_added))
        if allowed_added:
            report.warnings.append(f"professional reading stylesheet added: {allowed_added}")
        if not missing and not unexpected_added:
            added = []
        else:
            added = unexpected_added
    if set(source.entry_names) != set(output.entry_names) and (missing or added):
        report.errors.append(f"archive resource set changed; missing={missing[:5]}; added={added[:5]}")
    missing_manifest_resources = sorted(set(output.manifest_resources) - set(output.entry_names))
    if missing_manifest_resources:
        report.errors.append(f"manifest resources missing from ZIP: {missing_manifest_resources[:5]}")
    source_broken_toc = Counter(_broken_toc_targets(source))
    output_broken_toc = Counter(_broken_toc_targets(output))
    new_broken_toc = output_broken_toc - source_broken_toc
    inherited_broken_toc = output_broken_toc & source_broken_toc
    report.inherited_broken_toc_targets = sum(inherited_broken_toc.values())
    if new_broken_toc:
        report.errors.append(
            "table-of-contents targets are invalid: "
            f"{list(new_broken_toc.elements())[:5]}"
        )
    if report.inherited_broken_toc_targets:
        report.warnings.append(
            f"{report.inherited_broken_toc_targets} invalid table-of-contents target(s) "
            "were inherited unchanged from the source EPUB"
        )
    expected_cover = source.cover_image or source.obvious_cover_image
    if expected_cover and output.cover_image != expected_cover:
        report.errors.append(
            f"existing cover was not declared correctly: expected {expected_cover}; "
            f"found {output.cover_image or 'none'}"
        )
    if expected_cover and (
        not output.cover_page
        or output.cover_page not in set(output.entry_names)
    ):
        report.errors.append("cover guide reference is missing or invalid")
    if report.source_units != report.output_units:
        report.warnings.append(
            f"text-bearing DOM unit count differs before reflow alignment: "
            f"{report.source_units} != {report.output_units}"
        )

    changed_resources = [
        name for name, digest in source.resource_hashes.items()
        if output.resource_hashes.get(name) != digest
    ]
    report.preserved_resources = len(source.resource_hashes) - len(changed_resources)
    changed_images = [name for name in source.image_files if output.resource_hashes.get(name) != source.resource_hashes.get(name)]
    report.preserved_images = len(source.image_files) - len(changed_images)
    if changed_images:
        report.errors.append(f"image resources changed: {changed_images[:5]}")
    non_metadata_changes = [name for name in changed_resources if name not in {"META-INF/container.xml"}]
    if non_metadata_changes:
        report.warnings.append(f"non-XHTML resources changed: {non_metadata_changes[:8]}")

    for file_href in source.spine:
        source_counts = source.element_counts.get(file_href, Counter())
        output_counts = output.element_counts.get(file_href, Counter())
        if source_counts != output_counts:
            delta = output_counts - source_counts
            removed = source_counts - output_counts
            generated_css = any(
                Path(path).name in {"verbaloom-professional.css", "tbl-professional.css"}
                for path in output.css_files
            )
            viewport_added = bool(
                not source.mobile_viewports.get(file_href)
                and output.mobile_viewports.get(file_href)
                == "width=device-width, initial-scale=1.0"
            )
            allowed_professional_metadata = bool(
                not removed
                and set(delta) <= {"link", "meta"}
                and delta.get("link", 0) <= int(generated_css)
                and delta.get("meta", 0) <= int(viewport_added)
            )
            if not allowed_professional_metadata:
                report.errors.append(f"element structure changed in {file_href}")
        if source.image_refs.get(file_href) != output.image_refs.get(file_href):
            report.errors.append(f"image references changed in {file_href}")

    source_broken_links = _broken_internal_link_targets(source)
    output_broken_links = _broken_internal_link_targets(output)
    new_broken_links = output_broken_links - source_broken_links
    inherited_broken_links = output_broken_links & source_broken_links
    report.broken_links = sum(new_broken_links.values())
    report.inherited_broken_links = sum(inherited_broken_links.values())
    if report.broken_links:
        report.errors.append(f"{report.broken_links} internal link(s) are broken")
    if report.inherited_broken_links:
        report.warnings.append(
            f"{report.inherited_broken_links} broken internal link(s) were inherited "
            "unchanged from the source EPUB"
        )

    report.reflowed_units = abs(report.source_units - report.output_units)
    if report.reflowed_units:
        report.warnings.append(
            f"{report.reflowed_units} text-bearing DOM unit(s) changed visibility; "
            "coverage was audited at stable spine-file boundaries"
        )
    source_units_by_file: dict[str, list[EpubUnit]] = {}
    output_units_by_file: dict[str, list[EpubUnit]] = {}
    for unit in source.units:
        source_units_by_file.setdefault(unit.file_href, []).append(unit)
    for unit in output.units:
        output_units_by_file.setdefault(unit.file_href, []).append(unit)

    for group_index, file_href in enumerate(source.spine):
        source_units = source_units_by_file.get(file_href, [])
        if not source_units:
            continue
        output_file_units = output_units_by_file.get(file_href, [])
        source_text = source.file_texts.get(file_href, "")
        output_text = output.file_texts.get(file_href, "")
        issues: list[str] = []
        if not output_text.strip():
            issues.append("empty_output")
            report.missing_units += len(source_units)
        file_placeholder_findings = sum(
            1 for unit in output_file_units if _PLACEHOLDER_RE.search(unit.text)
        )
        if file_placeholder_findings:
            report.placeholder_findings += file_placeholder_findings
            issues.append("unresolved_placeholder")
        file_spacing_findings = sum(
            1 for unit in output_file_units if _BAD_SPACING_RE.search(unit.text)
        )
        if file_spacing_findings:
            report.spacing_findings += file_spacing_findings
            issues.append("invalid_punctuation_spacing")

        intentional_foreign_quote = _is_intentional_foreign_quote(
            source_text,
            output_text,
            source_language=source_language,
        )
        gate_issues = [] if intentional_foreign_quote else target_language_gate_issues(
            source_text,
            output_text,
            source_language=source_language,
            target_language=target_language,
            phase="epub_publication",
            prompt_options={"target_language_gate": True},
        )
        gate_codes = sorted({
            issue.code
            for issue in gate_issues
            if issue.code in {
                "target_script_mismatch",
                "target_language_missing",
                "untranslated_source",
            }
            and not _publication_gate_issue_is_exempt(
                issue.code,
                source_text=source_text,
                output_text=output_text,
                element="body",
                target_language=target_language,
            )
        })
        if gate_codes:
            issues.extend(gate_codes)

        detected, confidence = LanguageDetector.detect_language_from_text(
            output_text,
            confidence_threshold=0.90,
        )
        file_is_source_language = bool(
            _normalized_language(detected) == _normalized_language(source_language)
            and len(output_text) >= 80
            and not intentional_foreign_quote
            and not has_target_language_contextual_evidence(
                output_text,
                target_language=target_language,
            )
        )
        if file_is_source_language:
            report.source_language_units += len(source_units)
            if "source_language_output" not in issues:
                issues.append("source_language_output")

        previous_block = ""
        previous_output_unit: Optional[EpubUnit] = None
        source_by_path = {
            (unit.file_href, unit.dom_path): unit
            for unit in source_units
        }
        source_language_blocks = 0
        mixed_language_blocks = 0
        duplicate_blocks = 0
        for output_unit in output_file_units:
            normalized_block = _normalized_text(output_unit.text)
            translated_reference_context = looks_like_translated_reference_context(
                output_unit.text,
                target_language=target_language,
            ) or looks_like_structured_language_metadata(
                output_unit.text,
                document_hint=output_unit.file_href,
                target_language=target_language,
            )
            block_detected, block_confidence = LanguageDetector.detect_language_from_text(
                normalized_block,
                confidence_threshold=0.90,
            )
            block_is_source_language = bool(
                len(normalized_block) >= 80
                and _normalized_language(block_detected) == _normalized_language(source_language)
                and float(block_confidence or 0.0) >= 0.90
                and not _looks_like_reference_metadata(normalized_block, output_unit.element)
                and not looks_like_structured_language_metadata(
                    normalized_block,
                    document_hint=output_unit.file_href,
                    target_language=target_language,
                )
                and not has_target_language_contextual_evidence(
                    normalized_block,
                    target_language=target_language,
                )
            )
            if block_is_source_language:
                source_language_blocks += 1

            # Whole-block detection cannot see a copied source-language phrase
            # inside a much longer target-language paragraph. Compare each
            # stable DOM block with its source counterpart so mixed output is
            # blocked before publication as well as during generation.
            source_unit = source_by_path.get(
                (output_unit.file_href, output_unit.dom_path)
            )
            if source_unit is not None and not block_is_source_language:
                intentional_block_quote = _is_intentional_foreign_quote(
                    source_unit.text,
                    output_unit.text,
                    source_language=source_language,
                )
                audit_source_text = _normalize_ocr_spaced_words_for_language_audit(
                    source_unit.text
                )
                audit_output_text = _normalize_ocr_spaced_words_for_language_audit(
                    output_unit.text
                )
                block_gate_issues = [] if intentional_block_quote else target_language_gate_issues(
                    audit_source_text,
                    audit_output_text,
                    source_language=source_language,
                    target_language=target_language,
                    phase="epub_publication_block",
                    prompt_options={"target_language_gate": True},
                )
                has_actionable_gate_issue = any(
                    issue.code == "source_language_residual"
                    and issue.severity == "reject"
                    and (
                        str(issue.detail or "").startswith(
                            "tipo=short_source_pronoun;"
                        )
                        or _source_residual_is_actionable(
                            issue.detail,
                            source_text=source_unit.text,
                            source_language=source_language,
                            target_language=target_language,
                        )
                    )
                    and not _publication_gate_issue_is_exempt(
                        issue.code,
                        source_text=source_unit.text,
                        output_text=output_unit.text,
                        element=output_unit.element,
                        target_language=target_language,
                        file_href=output_unit.file_href,
                    )
                    for issue in block_gate_issues
                )
                has_exact_source_phrase_leak = _has_unquoted_source_phrase_leak(
                    audit_source_text,
                    audit_output_text,
                    source_language=source_language,
                    target_language=target_language,
                ) if not (
                    translated_reference_context
                    or _looks_like_reference_metadata(
                        output_unit.text,
                        output_unit.element,
                    )
                ) else False
                if (
                    not has_exact_source_phrase_leak
                    and audit_output_text != output_unit.text
                    and not translated_reference_context
                    and not _looks_like_reference_metadata(
                        output_unit.text,
                        output_unit.element,
                    )
                ):
                    # OCR fragmentation can split a short copied phrase around
                    # otherwise translated prose (for example ``arr i ve d
                    # at``). The stricter default exact-match threshold would
                    # miss it, so relax only when audit normalization actually
                    # repaired fragmented output tokens.
                    has_exact_source_phrase_leak = _has_unquoted_source_phrase_leak(
                        audit_source_text,
                        audit_output_text,
                        source_language=source_language,
                        target_language=target_language,
                        minimum_tokens=2,
                        minimum_source_markers=1,
                        original_output_text=output_unit.text,
                    )
                if has_actionable_gate_issue or has_exact_source_phrase_leak:
                    mixed_language_blocks += 1
            folded_block = normalized_block.casefold()
            if (
                previous_output_unit is not None
                and (
                    (len(folded_block) >= 160 and folded_block == previous_block)
                    or _has_substantial_adjacent_overlap(previous_block, folded_block)
                )
                and not _nested_dom_paths(
                    previous_output_unit.dom_path,
                    output_unit.dom_path,
                )
            ):
                previous_source = source_by_path.get(
                    (previous_output_unit.file_href, previous_output_unit.dom_path)
                )
                current_source = source_by_path.get(
                    (output_unit.file_href, output_unit.dom_path)
                )
                legitimate_source_repeat = bool(
                    previous_source
                    and current_source
                    and (
                        _normalized_text(previous_source.text).casefold()
                        == _normalized_text(current_source.text).casefold()
                        or _has_substantial_adjacent_overlap(
                            _normalized_text(previous_source.text).casefold(),
                            _normalized_text(current_source.text).casefold(),
                        )
                    )
                )
                if not legitimate_source_repeat:
                    duplicate_blocks += 1
            previous_block = folded_block
            previous_output_unit = output_unit
        if source_language_blocks and not file_is_source_language:
            report.source_language_units += source_language_blocks
            if "source_language_output" not in issues:
                issues.append("source_language_output")
        if mixed_language_blocks:
            report.mixed_language_units += mixed_language_blocks
            if "source_language_residual" not in issues:
                issues.append("source_language_residual")
        if duplicate_blocks:
            report.duplicate_units += duplicate_blocks
            issues.append("duplicate_output")

        fidelity = assess_fidelity(
            source_text,
            output_text,
            chunk_index=group_index,
            phase="epub_publication",
            source_language=source_language,
            target_language=target_language,
            prompt_options={"target_language_gate": True},
        )
        for issue in fidelity.issues:
            if issue.severity != "reject":
                continue
            if intentional_foreign_quote and issue.code in {
                "target_language_missing",
                "untranslated_source",
                "source_language_residual",
            }:
                continue
            if issue.code in {
                "empty_candidate",
                "severe_length_drop",
                "symbol_bearing_names_lost",
                "target_script_mismatch",
                "target_language_missing",
                "untranslated_source",
            } and not _publication_gate_issue_is_exempt(
                issue.code,
                source_text=source_text,
                output_text=output_text,
                element="body",
                target_language=target_language,
            ):
                if issue.code not in issues:
                    issues.append(issue.code)

        reused = bool(broken and broken.file_texts.get(file_href) == output_text)
        representative = output_file_units[0] if output_file_units else None
        status = "AUDITED" if not issues else "FAILED"
        if status == "AUDITED":
            report.audited_units += len(source_units)
        report.units.append(_unit_report_payload(
            source_units=source_units,
            output_unit=representative,
            status=status,
            issues=issues,
            detected=detected or "",
            confidence=float(confidence or 0.0),
            reused=reused,
            source_text_override=source_text,
            translation_override=output_text,
            dom_path_override=f"/{file_href}/body",
            intentional_exclusions=[
                unit.text
                for unit in source_units
                if unit.dom_path
                in output.sanitized_artifact_paths.get(file_href, set())
            ],
        ))

    if report.source_language_units:
        report.errors.append(f"{report.source_language_units} unit(s) remain in the source language")
    if report.mixed_language_units:
        report.errors.append(f"{report.mixed_language_units} unit(s) contain source-language residue")
    if report.duplicate_units:
        report.errors.append(f"{report.duplicate_units} unit(s) duplicate unrelated output")
    if report.placeholder_findings:
        report.errors.append(f"{report.placeholder_findings} unresolved placeholder/protocol finding(s)")
    if report.spacing_findings:
        report.errors.append(f"{report.spacing_findings} invalid punctuation-spacing finding(s)")
    failed_coverage = max(0, report.source_units - report.audited_units)
    if failed_coverage:
        report.errors.append(f"{failed_coverage} source unit(s) failed publication coverage")

    _run_epubcheck(
        report,
        Path(output_path),
        epubcheck_command,
        source_path=Path(source_path),
    )
    if report.epubcheck_errors:
        report.errors.append(f"EPUBCheck reported {report.epubcheck_errors} error(s)")
    return report


def normalize_epub_language_metadata(epub_path: str | Path, target_language: str) -> int:
    """Set OPF dc:language and XHTML lang/xml:lang without changing other resources."""
    path = Path(epub_path)
    original_mode = stat.S_IMODE(path.stat().st_mode)
    language_code = get_language_code(target_language)
    if not language_code:
        raise ValueError(f"Unknown target language: {target_language}")
    changed = 0
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}.", suffix=".epub", dir=str(path.parent), delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(tmp_path, "w") as output:
            container = etree.fromstring(source.read("META-INF/container.xml"))
            opf_path = container.xpath("string(//*[local-name()='rootfile']/@full-path)")
            for info in source.infolist():
                payload = source.read(info.filename)
                if info.filename == opf_path:
                    root = etree.fromstring(payload, etree.XMLParser(recover=False, remove_blank_text=False))
                    language_nodes = root.xpath("//*[local-name()='metadata']/*[local-name()='language']")
                    if language_nodes:
                        for node in language_nodes:
                            if node.text != language_code:
                                node.text = language_code
                                changed += 1
                    else:
                        metadata = root.xpath("//*[local-name()='metadata']")[0]
                        node = etree.SubElement(metadata, "{http://purl.org/dc/elements/1.1/}language")
                        node.text = language_code
                        changed += 1
                    payload = etree.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=False)
                elif Path(info.filename).suffix.lower() in _TEXT_SUFFIXES:
                    root = etree.fromstring(payload, etree.XMLParser(recover=False, remove_blank_text=False))
                    before = (
                        root.get("lang"),
                        root.get("{http://www.w3.org/XML/1998/namespace}lang"),
                    )
                    root.set("lang", language_code)
                    root.set("{http://www.w3.org/XML/1998/namespace}lang", language_code)
                    if before != (language_code, language_code):
                        changed += 1
                        payload = etree.tostring(
                            root, encoding="utf-8", xml_declaration=True, pretty_print=False
                        )
                _write_member(output, info, payload)
        tmp_path.replace(path)
        path.chmod(original_mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return changed


def _broken_internal_link_targets(snapshot: EpubSnapshot) -> Counter[tuple[str, str]]:
    broken: Counter[tuple[str, str]] = Counter()
    names = set(snapshot.entry_names)
    for origin, href in snapshot.links:
        parsed = urlsplit(href)
        if parsed.scheme.casefold() in _EXTERNAL_SCHEMES or href.startswith("//"):
            continue
        target_path = origin
        if parsed.path:
            target_path = posixpath.normpath(posixpath.join(posixpath.dirname(origin), unquote(parsed.path)))
        if target_path not in names:
            broken[(origin, f"{target_path}#{parsed.fragment}" if parsed.fragment else target_path)] += 1
            continue
        if parsed.fragment and parsed.fragment not in snapshot.ids.get(target_path, set()):
            broken[(origin, f"{target_path}#{parsed.fragment}")] += 1
    return broken


def _broken_internal_links(snapshot: EpubSnapshot) -> int:
    return sum(_broken_internal_link_targets(snapshot).values())


def _intentional_foreign_quote_indices(
    source_units: list[EpubUnit],
    output_units: list[EpubUnit],
    *,
    source_language: str,
) -> set[int]:
    """Find exact source quotations written in a third language, including short adjacent lines."""
    exact = {
        index
        for index, (source, output) in enumerate(zip(source_units, output_units))
        if source.file_href == output.file_href
        and _normalized_text(source.text).casefold() == _normalized_text(output.text).casefold()
    }
    seeds: set[int] = set()
    for index in exact:
        detected, _confidence = LanguageDetector.detect_language_from_text(
            source_units[index].text,
            confidence_threshold=0.50,
        )
        normalized = _normalized_language(detected)
        if normalized and normalized != _normalized_language(source_language):
            seeds.add(index)
    for run in _contiguous_runs(exact, source_units):
        combined = " ".join(source_units[index].text for index in run)
        detected, _confidence = LanguageDetector.detect_language_from_text(
            combined,
            confidence_threshold=0.50,
        )
        normalized = _normalized_language(detected)
        if normalized and normalized != _normalized_language(source_language):
            seeds.update(run)
    expanded = set(seeds)
    changed = True
    while changed:
        changed = False
        for index in exact - expanded:
            neighbors = (index - 1, index + 1)
            if any(
                neighbor in expanded
                and source_units[neighbor].file_href == source_units[index].file_href
                for neighbor in neighbors
            ):
                expanded.add(index)
                changed = True
    return expanded


def _contiguous_runs(indices: set[int], units: list[EpubUnit]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index in sorted(indices):
        if (
            current
            and index == current[-1] + 1
            and units[index].file_href == units[current[-1]].file_href
        ):
            current.append(index)
        else:
            if current:
                runs.append(current)
            current = [index]
    if current:
        runs.append(current)
    return runs


def _align_units_by_dom_path(
    source_units: list[EpubUnit],
    output_units: list[EpubUnit],
) -> tuple[
    list[tuple[list[EpubUnit], EpubUnit]],
    list[EpubUnit],
    list[EpubUnit],
    list[EpubUnit],
]:
    """Align text-bearing units without assuming identical non-empty slots.

    Translation may legitimately merge a short paragraph into the immediately
    adjacent block while preserving the XHTML element tree.  Index-based ZIP
    comparisons shift every following unit after the first merge and report a
    mostly correct book as missing.  DOM paths keep the alignment stable; only
    source blocks that became empty are attached to their nearest visible
    neighbour for a source-aware aggregate audit.
    """
    source_by_path = {(unit.file_href, unit.dom_path): unit for unit in source_units}
    groups: list[tuple[list[EpubUnit], EpubUnit]] = []
    group_by_output_path: dict[tuple[str, str], list[EpubUnit]] = {}
    added_output_units: list[EpubUnit] = []

    for output_unit in output_units:
        key = (output_unit.file_href, output_unit.dom_path)
        direct_source = source_by_path.get(key)
        if direct_source is None:
            added_output_units.append(output_unit)
            continue
        source_group = [direct_source]
        groups.append((source_group, output_unit))
        group_by_output_path[key] = source_group

    output_candidates_by_file: dict[str, list[EpubUnit]] = {}
    for _source_group, output_unit in groups:
        output_candidates_by_file.setdefault(output_unit.file_href, []).append(output_unit)

    excluded_source_units: list[EpubUnit] = []
    unaligned_source_units: list[EpubUnit] = []
    output_paths = {(unit.file_href, unit.dom_path) for unit in output_units}
    for source_unit in source_units:
        key = (source_unit.file_href, source_unit.dom_path)
        if key in output_paths:
            continue
        if _is_source_furniture(source_unit.text):
            excluded_source_units.append(source_unit)
            continue
        candidates = output_candidates_by_file.get(source_unit.file_href, [])
        if not candidates:
            unaligned_source_units.append(source_unit)
            continue
        nearest = min(
            candidates,
            key=lambda candidate: (
                abs(candidate.ordinal - source_unit.ordinal),
                0 if candidate.ordinal < source_unit.ordinal else 1,
            ),
        )
        group_by_output_path[(nearest.file_href, nearest.dom_path)].append(source_unit)

    for source_group, _output_unit in groups:
        source_group.sort(key=lambda unit: unit.source_order)
    groups.sort(key=lambda item: item[1].source_order)
    return groups, excluded_source_units, unaligned_source_units, added_output_units


def _is_source_furniture(text: str) -> bool:
    value = _normalized_text(text)
    if not value:
        return True
    if re.fullmatch(r"[□■▪▫☐☑✓✔*•·.\-–—_~|/\\\s]+", value):
        return True
    if re.fullmatch(r"(?:p(?:age|ágina)?\s*)?\d{1,5}\.?", value, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"(?:EAN|ISBN)(?:[- :]*[0-9Xx£$%?A-Za-z]+)+", value):
        return True
    letters = sum(char.isalpha() for char in value)
    symbols = sum(not char.isalnum() and not char.isspace() for char in value)
    if len(value) <= 12 and letters <= 3 and symbols >= max(2, letters):
        return True
    return False


def _nested_dom_paths(left: str, right: str) -> bool:
    left_path = str(left or "").rstrip("/")
    right_path = str(right or "").rstrip("/")
    return bool(
        left_path
        and right_path
        and (
            right_path.startswith(left_path + "/")
            or left_path.startswith(right_path + "/")
        )
    )


def _is_intentional_foreign_quote(
    source_text: str,
    output_text: str,
    *,
    source_language: str,
) -> bool:
    if _normalized_text(source_text).casefold() != _normalized_text(output_text).casefold():
        return False
    detected, confidence = LanguageDetector.detect_language_from_text(
        source_text,
        confidence_threshold=0.50,
    )
    return bool(
        detected
        and float(confidence or 0.0) >= 0.50
        and _normalized_language(detected) != _normalized_language(source_language)
    )


_REFERENCE_METADATA_RE = re.compile(
    r"(?i)(?:\bISBN\b|\bEAN\b|copyright|©|\ball rights reserved\b|"
    r"\bpublished by\b|\bpublishing\b|\bbooks?\s+(?:ltd|inc)\b|"
    r"\b(?:street|avenue|road|boulevard)\b|\bpress\b)"
)

_RESIDUAL_EXAMPLE_RE = re.compile(
    r"(?P<phrase>.+?)\s*\((?:0|1)\.\d{2}\)(?:;\s*|,\s*|$)"
)
_QUOTED_SOURCE_RE = re.compile(
    r"“[^”]{1,500}”|«[^»]{1,500}»|\"[^\"]{1,500}\"|'[^']{1,500}'",
    re.DOTALL,
)
_BOOLEAN_QUERY_RE = re.compile(r"\b(?:AND|OR|NOT|NEAR)\b")
_QUERY_CONTEXT_RE = re.compile(
    r"\b(?:search|query|define|look\s+for|find|buscar|b[uú]squeda|consulta)\b",
    re.IGNORECASE,
)
_LANGUAGE_MARKERS: dict[str, re.Pattern[str]] = {
    "english": re.compile(
        r"\b(?:the|and|of|to|in|that|is|was|for|with|as|on|by|from|this|it|"
        r"be|are|were|or|an|at|which|not|have|has|had|but|we|you|they|"
        r"our|their|them)\b",
        re.IGNORECASE,
    ),
    "spanish": re.compile(
        r"\b(?:el|la|los|las|un|una|de|del|que|en|para|con|por|se|no|su|"
        r"al|es|son|era|fue|est[aá]|hab[ií]a|pero|cuando|como|desde|hasta)\b",
        re.IGNORECASE,
    ),
    "french": re.compile(
        r"\b(?:le|la|les|des|du|de|et|que|qui|dans|pour|avec|sur|est|sont|"
        r"une|un|ce|cette|par|pas|plus|mais|comme)\b",
        re.IGNORECASE,
    ),
    "german": re.compile(
        r"\b(?:der|die|das|den|dem|des|ein|eine|einer|und|oder|aber|ist|sind|"
        r"war|waren|mit|von|zu|im|in|auf|f[uü]r|nicht|dass|als|auch)\b",
        re.IGNORECASE,
    ),
}
_OCR_SPACED_WORD_RE = re.compile(
    r"\b[^\W\d_]{3,4}(?:\s+[^\W\d_]{1,2}){2,3}\b",
    re.UNICODE,
)


def _normalize_ocr_spaced_words_for_language_audit(text: str) -> str:
    """Join obvious OCR letter fragments for detection without editing output."""
    def collapse(match: re.Match[str]) -> str:
        parts = re.findall(r"[^\W\d_]+", match.group(0), flags=re.UNICODE)
        # Ordinary phrases such as ``arrived at el`` contain short words but
        # no isolated OCR letter. Leave them untouched.
        if not any(len(part) == 1 for part in parts[1:]):
            return match.group(0)
        return "".join(parts)

    return _OCR_SPACED_WORD_RE.sub(
        collapse,
        text or "",
    )


def _source_residual_is_actionable(
    detail: str,
    *,
    source_text: str,
    source_language: str,
    target_language: str,
) -> bool:
    """Keep the final mixed-language gate strict without rejecting quotations.

    The generation gate intentionally errs on the side of review. Publication
    needs a narrower hard failure: an exact source phrase outside a quoted
    foreign-language utterance, with positive source-language evidence. This
    catches short prose leaks such as ``arrived at`` while preserving Portuguese
    dialogue, titles, names and literal search syntax inside an English novel.
    """
    value = str(detail or "")
    residuals = value.split("residuos=", 1)[-1] if "residuos=" in value else value
    phrases = [
        match.group("phrase").strip(" ;,")
        for match in _RESIDUAL_EXAMPLE_RE.finditer(residuals)
        if match.group("phrase").strip(" ;,")
    ]
    if not phrases:
        return False

    quoted_spans = [match.span() for match in _QUOTED_SOURCE_RE.finditer(source_text or "")]
    source_key = _normalized_language(source_language)
    target_key = _normalized_language(target_language)
    source_markers = _LANGUAGE_MARKERS.get(source_key)
    target_markers = _LANGUAGE_MARKERS.get(target_key)

    for phrase in phrases:
        phrase_words = re.findall(r"[^\W\d_]+", phrase, flags=re.UNICODE)
        if looks_like_preserved_title_or_citation_sequence(phrase_words):
            continue
        if (
            len(_BOOLEAN_QUERY_RE.findall(phrase)) >= 2
            and _QUERY_CONTEXT_RE.search(source_text or "")
        ):
            continue
        occurrence = re.search(re.escape(phrase), source_text or "", re.IGNORECASE)
        if occurrence and any(
            start <= occurrence.start() and occurrence.end() <= end
            for start, end in quoted_spans
        ):
            continue

        source_count = len(source_markers.findall(phrase)) if source_markers else 0
        target_count = len(target_markers.findall(phrase)) if target_markers else 0
        if (
            source_count > target_count
            or (source_markers is None and len(phrase_words) >= 4)
        ):
            return True
    return False


def _has_unquoted_source_phrase_leak(
    source_text: str,
    output_text: str,
    *,
    source_language: str,
    target_language: str,
    minimum_tokens: int = 5,
    minimum_source_markers: int = 2,
    original_output_text: str = "",
) -> bool:
    """Detect short exact source prose that whole-paragraph ID can miss."""
    token_re = re.compile(r"[^\W\d_]+", re.UNICODE)
    source_matches = list(token_re.finditer(source_text or ""))
    output_matches = list(token_re.finditer(output_text or ""))
    if len(source_matches) < 2 or len(output_matches) < 2:
        return False

    source_tokens = [match.group(0).casefold() for match in source_matches]
    output_tokens = [match.group(0).casefold() for match in output_matches]
    source_key = _normalized_language(source_language)
    target_key = _normalized_language(target_language)
    source_markers = _LANGUAGE_MARKERS.get(source_key)
    target_markers = _LANGUAGE_MARKERS.get(target_key)
    quoted_spans = [match.span() for match in _QUOTED_SOURCE_RE.finditer(source_text or "")]

    matcher = difflib.SequenceMatcher(None, source_tokens, output_tokens, autojunk=False)
    for block in matcher.get_matching_blocks():
        # Two or three shared words are common in proper names, coined terms
        # and foreign-language titles. They are not enough evidence of an
        # untranslated prose leak. Short OCR-split leaks are handled by the
        # source-aware language gate above after audit-only normalization.
        if block.size < max(2, int(minimum_tokens)):
            continue
        start_match = source_matches[block.a]
        end_match = source_matches[block.a + block.size - 1]
        if any(
            quote_start <= start_match.start() and end_match.end() <= quote_end
            for quote_start, quote_end in quoted_spans
        ):
            continue
        phrase = " ".join(
            match.group(0)
            for match in source_matches[block.a:block.a + block.size]
        )
        phrase_words = [
            match.group(0)
            for match in source_matches[block.a:block.a + block.size]
        ]
        if looks_like_preserved_title_or_citation_sequence(phrase_words):
            continue
        if (
            original_output_text
            and phrase.casefold() in _normalized_text(original_output_text).casefold()
        ):
            # The relaxed OCR pass should only evaluate phrases created by
            # audit-only fragment joining. Phrases already present in the real
            # output were evaluated by the normal threshold above.
            continue
        # Search expressions are intentionally language-neutral and often
        # appear without an explanatory word such as "search" in the same
        # DOM block.
        if len(_BOOLEAN_QUERY_RE.findall(phrase)) >= 2:
            continue
        source_count = len(source_markers.findall(phrase)) if source_markers else 0
        target_count = len(target_markers.findall(phrase)) if target_markers else 0
        lowercase_words = [
            word
            for word in re.findall(r"[^\W\d_]+", phrase, flags=re.UNICODE)
            if word[:1].islower()
        ]
        if (
            source_count >= max(1, int(minimum_source_markers))
            and source_count > target_count
            and len(lowercase_words) >= 2
        ):
            return True
        if source_markers is None and block.size >= 4:
            return True
    return False


def _looks_like_reference_metadata(text: str, element: str = "") -> bool:
    value = _normalized_text(text)
    if not value:
        return False
    if _REFERENCE_METADATA_RE.search(value):
        return True
    return bool(element in {"h1", "h2", "h3", "h4", "h5", "h6"} and len(value) <= 180)


def _publication_gate_issue_is_exempt(
    code: str,
    *,
    source_text: str,
    output_text: str,
    element: str,
    target_language: str = "",
    file_href: str = "",
) -> bool:
    if _is_source_furniture(source_text):
        return True
    if code in {
        "target_language_missing",
        "untranslated_source",
        "source_language_output",
        "source_language_residual",
    } and (
        _looks_like_reference_metadata(output_text, element)
        or looks_like_structured_language_metadata(
            output_text,
            document_hint=file_href,
            target_language=target_language,
        )
        or looks_like_translated_reference_context(
            output_text,
            target_language=target_language,
        )
    ):
        return True
    return False


def _unit_report_payload(
    *,
    source_units: list[EpubUnit],
    output_unit: Optional[EpubUnit],
    status: str,
    issues: list[str],
    detected: str,
    confidence: float,
    reused: bool,
    review_status: str = "",
    source_text_override: str = "",
    translation_override: Optional[str] = None,
    dom_path_override: str = "",
    intentional_exclusions: Iterable[str] = (),
) -> dict[str, Any]:
    first = source_units[0]
    source_text = source_text_override or _normalized_text(
        " ".join(unit.text for unit in source_units)
    )
    translation = (
        translation_override
        if translation_override is not None
        else (output_unit.text if output_unit is not None else "")
    )
    source_hash = text_sha256(source_text)
    unit_id = stable_unit_id(
        first.file_href,
        output_unit.ordinal if output_unit is not None else first.ordinal,
        source_hash,
        spine_index=first.spine_index,
        dom_path=(
            dom_path_override
            or (output_unit.dom_path if output_unit is not None else first.dom_path)
        ),
        source_order=first.source_order,
    )
    return {
        "unit_id": unit_id,
        "source_unit_ids": [unit.unit_id for unit in source_units],
        "source_unit_count": len(source_units),
        "reflowed": len(source_units) > 1,
        "file_href": first.file_href,
        "ordinal": output_unit.ordinal if output_unit is not None else first.ordinal,
        "element": output_unit.element if output_unit is not None else first.element,
        "source_document": first.file_href,
        "spine_index": first.spine_index,
        "dom_path": (
            dom_path_override
            or (output_unit.dom_path if output_unit is not None else first.dom_path)
        ),
        "source_order": first.source_order,
        "source_hash": source_hash,
        "source_text": source_text,
        "intentional_exclusions": [
            _normalized_text(value)
            for value in intentional_exclusions
            if _normalized_text(value)
        ],
        "translation": translation,
        "translation_hash": text_sha256(translation),
        "status": status,
        "attempts": None,
        "failure_reason": ", ".join(issues) if issues else None,
        "review": {
            "status": review_status or ("retained" if reused else "retranslated")
        },
        "audit": {
            "status": "pass" if status in {"AUDITED", "EXCLUDED"} else "fail",
            "method": "deterministic-source-aware-publication-gate",
            "language_detected": detected,
            "language_confidence": float(confidence or 0.0),
            "issues": issues,
        },
    }


def _write_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, payload: bytes) -> None:
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
    archive.writestr(clone, payload)


def _run_epubcheck(
    report: PublicationGateReport,
    output_path: Path,
    command: Optional[list[str]],
    *,
    source_path: Path | None = None,
) -> None:
    resolved = list(command or [])
    if not resolved:
        executable = shutil.which("epubcheck")
        if executable:
            resolved = [executable]
    if not resolved:
        executable = next(
            (
                str(candidate)
                for candidate in _KNOWN_EPUBCHECK_PATHS
                if candidate.is_file() and os.access(candidate, os.X_OK)
            ),
            "",
        )
        if executable:
            resolved = [executable]
    if not resolved:
        report.warnings.append("EPUBCheck is not installed; external validation was not run")
        return
    process = subprocess.run(
        [*resolved, str(output_path)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    combined = "\n".join(part for part in (process.stdout, process.stderr) if part).strip()
    report.epubcheck_output = combined[-12000:]
    output_errors = _epubcheck_error_signatures(combined)
    if process.returncode and not output_errors:
        output_errors[('process_returncode', str(process.returncode))] += 1

    source_errors: Counter[tuple[str, str]] = Counter()
    if source_path and source_path.exists() and source_path.resolve() != output_path.resolve():
        source_process = subprocess.run(
            [*resolved, str(source_path)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        source_combined = "\n".join(
            part for part in (source_process.stdout, source_process.stderr) if part
        ).strip()
        source_errors = _epubcheck_error_signatures(source_combined)
        if source_process.returncode and not source_errors:
            source_errors[('process_returncode', str(source_process.returncode))] += 1

    new_errors = output_errors - source_errors
    inherited_errors = output_errors & source_errors
    report.epubcheck_errors = sum(new_errors.values())
    report.inherited_epubcheck_errors = sum(inherited_errors.values())
    report.epubcheck_warnings = len(re.findall(r"(?im)^.*\bWARNING\b", combined))
    if report.inherited_epubcheck_errors:
        report.warnings.append(
            f"EPUBCheck found {report.inherited_epubcheck_errors} error(s) inherited "
            "unchanged from the source EPUB"
        )


def _epubcheck_error_signatures(output: str) -> Counter[tuple[str, str]]:
    """Normalize EPUBCheck errors so source/output paths and line numbers do not differ."""
    signatures: Counter[tuple[str, str]] = Counter()
    for line in str(output or "").splitlines():
        match = re.match(r"(?i)^.*?\bERROR(?:\(([^)]+)\))?:\s*(.*)$", line.strip())
        if not match:
            continue
        code = str(match.group(1) or "ERROR").upper()
        detail = str(match.group(2) or "")
        detail = re.sub(r"^.*?\.epub/", "", detail, count=1)
        detail = re.sub(r"\(\d+,\d+\)(?=:)", "", detail, count=1)
        signatures[(code, detail.strip())] += 1
    return signatures


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _has_substantial_adjacent_overlap(left: str, right: str) -> bool:
    """Detect a paragraph repeated as the prefix of its neighbor."""
    left_tokens = re.findall(r"\w+", str(left or "").casefold(), re.UNICODE)
    right_tokens = re.findall(r"\w+", str(right or "").casefold(), re.UNICODE)
    shorter, longer = (
        (left_tokens, right_tokens)
        if len(left_tokens) <= len(right_tokens)
        else (right_tokens, left_tokens)
    )
    if len(shorter) < 12 or len(" ".join(shorter)) < 70:
        return False
    matching_prefix = 0
    for short_token, long_token in zip(shorter, longer):
        if short_token != long_token:
            break
        matching_prefix += 1
    return matching_prefix / len(shorter) >= 0.90


def _normalized_language(value: Optional[str]) -> str:
    aliases = {
        "de": "german", "deutsch": "german", "alemán": "german", "aleman": "german",
        "es": "spanish", "español": "spanish", "espanol": "spanish",
        "en": "english", "fr": "french", "it": "italian", "pt": "portuguese",
    }
    key = str(value or "").strip().casefold()
    return aliases.get(key, key)
