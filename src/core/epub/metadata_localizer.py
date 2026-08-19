"""Localize EPUB metadata and navigation while preserving source identity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import posixpath
import stat
import tempfile
from typing import Mapping
from urllib.parse import unquote, urlsplit
import zipfile

from lxml import etree

from .lang_support import get_language_code


_TEXT_SUFFIXES = {".xhtml", ".html", ".htm"}
_PROTECTED_METADATA = {
    "creator", "contributor", "identifier", "publisher", "date", "rights",
    "source", "relation", "coverage", "subject", "description", "type", "format",
}
_DEFAULT_ES_LABELS = {
    "cover": "Portada",
    "titel": "Página de título",
    "title page": "Página de título",
    "impressum": "Créditos",
    "copyright": "Créditos",
    "der autor": "El autor",
    "the author": "El autor",
    "inhalt": "Índice",
    "inhaltsverzeichnis": "Índice",
    "contents": "Índice",
    "vorwort": "Prólogo",
    "preface": "Prólogo",
    "einleitung": "Introducción",
    "introduction": "Introducción",
    "anmerkungen": "Notas",
    "notes": "Notas",
    "bibliographie": "Bibliografía",
    "bibliography": "Bibliografía",
}


@dataclass
class MetadataLocalizationReport:
    opf_title_changes: int = 0
    language_changes: int = 0
    protected_metadata_restored: int = 0
    html_title_changes: int = 0
    navigation_label_changes: int = 0
    changed_files: int = 0


def _local_name(node: etree._Element) -> str:
    return etree.QName(node).localname.casefold()


def _container_opf(archive: zipfile.ZipFile) -> str:
    container = etree.fromstring(archive.read("META-INF/container.xml"))
    return str(container.xpath("string(//*[local-name()='rootfile']/@full-path)") or "")


def _parse(payload: bytes) -> etree._Element:
    return etree.fromstring(
        payload,
        etree.XMLParser(recover=False, huge_tree=True, remove_blank_text=False, no_network=True),
    )


def _serialize(root: etree._Element, original: bytes) -> bytes:
    return etree.tostring(
        root.getroottree(),
        encoding="utf-8",
        xml_declaration=original.lstrip().startswith(b"<?xml"),
        doctype=root.getroottree().docinfo.doctype or None,
        pretty_print=False,
    )


def _write_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, payload: bytes) -> None:
    clone = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    clone.comment = info.comment
    clone.extra = info.extra
    clone.internal_attr = info.internal_attr
    clone.external_attr = info.external_attr
    clone.create_system = info.create_system
    clone.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
    archive.writestr(clone, payload)


def _full_title(title: str, subtitle: str) -> str:
    title = str(title or "").strip()
    subtitle = str(subtitle or "").strip()
    return f"{title}: {subtitle}" if title and subtitle else title or subtitle


def _person_tokens(value: str) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def _spanish_display_case(value: str) -> str:
    words = str(value or "").casefold().split()
    if not words:
        return ""
    minor = {"a", "al", "de", "del", "el", "en", "la", "las", "los", "o", "para", "por", "un", "una", "y"}
    return " ".join(
        word.capitalize() if index == 0 or word not in minor else word
        for index, word in enumerate(words)
    )


def _metadata_title_parts(package: etree._Element) -> tuple[str, str]:
    titles = package.xpath("//*[local-name()='metadata']/*[local-name()='title']")
    if not titles:
        return "", ""
    title_types: dict[str, str] = {}
    for meta in package.xpath(
        "//*[local-name()='metadata']/*[local-name()='meta' and "
        "@property='title-type' and @refines]"
    ):
        target = str(meta.get("refines") or "").lstrip("#")
        if target:
            title_types[target] = " ".join(meta.itertext()).strip().casefold()
    main = ""
    subtitle = ""
    for index, node in enumerate(titles):
        value = " ".join(" ".join(node.itertext()).split())
        kind = title_types.get(str(node.get("id") or ""))
        if kind == "main":
            main = value
        elif kind == "subtitle":
            subtitle = value
        elif index == 0 and not main:
            main = value
        elif index == 1 and not subtitle:
            subtitle = value
    return main, subtitle


def infer_epub_title_page(output_epub: str | Path) -> tuple[str, str]:
    """Infer visible title/subtitle from a conventional title-page document."""
    with zipfile.ZipFile(output_epub) as archive:
        opf_path = _container_opf(archive)
        package = _parse(archive.read(opf_path))
        creators = package.xpath("//*[local-name()='creator']/text()")
        creator_tokens = set().union(*(_person_tokens(value) for value in creators)) if creators else set()
        opf_dir = posixpath.dirname(opf_path)
        ranked_candidates: list[tuple[int, str]] = []
        for item in package.xpath(
            "//*[local-name()='manifest']/*[local-name()='item']"
        ):
            href = str(item.get("href") or "")
            resolved = posixpath.normpath(posixpath.join(opf_dir, href))
            if Path(resolved).suffix.lower() not in _TEXT_SUFFIXES:
                continue
            item_id = str(item.get("id") or "").casefold()
            stem = Path(resolved).stem.casefold()
            if item_id == "title" or stem == "title":
                rank = 0
            elif any(token in item_id or token in stem for token in ("title", "titel", "portada")):
                rank = 1
            else:
                continue
            ranked_candidates.append((rank, resolved))
        candidates = [
            name
            for _rank, name in sorted(
                dict.fromkeys(ranked_candidates),
                key=lambda item: (item[0], item[1]),
            )
            if name in archive.namelist()
        ]
        for name in candidates:
            try:
                root = _parse(archive.read(name))
            except Exception:
                continue
            lines: list[tuple[str, bool, bool]] = []
            for node in root.xpath(
                "//*[local-name()='body']//*[local-name()='h1' or local-name()='h2' or local-name()='p']"
            ):
                text = " ".join(" ".join(node.itertext()).split())
                if not text:
                    continue
                tokens = _person_tokens(text)
                if creator_tokens and tokens and tokens <= creator_tokens:
                    continue
                if any(marker in text.casefold() for marker in (" verlag", "press", "editorial", "publishing")):
                    break
                if any(char.isdigit() for char in text) and len(text) < 80:
                    continue
                italic = bool(node.xpath(".//*[local-name()='i' or local-name()='em']"))
                heading = _local_name(node) in {"h1", "h2"}
                lines.append((text, italic, heading))
            emphasized = [text for text, italic, heading in lines if italic or heading]
            if emphasized:
                title = _spanish_display_case(" ".join(emphasized[:3]))
                remaining = [text for text, italic, heading in lines if not (italic or heading)]
                subtitle = _spanish_display_case(remaining[0]) if remaining else ""
                return title, subtitle
            if lines:
                return _spanish_display_case(lines[0][0]), ""
        # Image-only title pages are common. In that case the OPF is a safer
        # source of identity than a generic promotional file whose name happens
        # to contain "front".
        return _metadata_title_parts(package)


def _localized_label(value: str, labels: Mapping[str, str]) -> str:
    original = str(value or "")
    return labels.get(" ".join(original.casefold().split()), original)


def _manifest_info(package: etree._Element, opf_path: str) -> tuple[dict[str, str], list[str], list[str]]:
    opf_dir = posixpath.dirname(opf_path)
    href_by_id: dict[str, str] = {}
    nav_files: list[str] = []
    ncx_files: list[str] = []
    for node in package.xpath("//*[local-name()='manifest']/*[local-name()='item']"):
        href = str(node.get("href") or "")
        resolved = posixpath.normpath(posixpath.join(opf_dir, href))
        href_by_id[str(node.get("id") or "")] = resolved
        if "nav" in str(node.get("properties") or "").split():
            nav_files.append(resolved)
        if str(node.get("media-type") or "") == "application/x-dtbncx+xml":
            ncx_files.append(resolved)
    return href_by_id, nav_files, ncx_files


def _restore_protected_metadata(
    source_package: etree._Element,
    output_package: etree._Element,
) -> int:
    source_metadata = source_package.xpath("//*[local-name()='metadata']")[0]
    output_metadata = output_package.xpath("//*[local-name()='metadata']")[0]
    changed = 0
    for name in _PROTECTED_METADATA:
        source_nodes = [node for node in source_metadata if isinstance(node.tag, str) and _local_name(node) == name]
        output_nodes = [node for node in output_metadata if isinstance(node.tag, str) and _local_name(node) == name]
        source_xml = [etree.tostring(node) for node in source_nodes]
        output_xml = [etree.tostring(node) for node in output_nodes]
        if source_xml == output_xml:
            continue
        for node in output_nodes:
            output_metadata.remove(node)
        for node in source_nodes:
            output_metadata.append(deepcopy(node))
        changed += 1
    return changed


def _localize_package(
    source_payload: bytes,
    output_payload: bytes,
    *,
    title: str,
    subtitle: str,
    language_code: str,
    report: MetadataLocalizationReport,
) -> tuple[bytes, etree._Element]:
    source_root = _parse(source_payload)
    output_root = _parse(output_payload)
    report.protected_metadata_restored += _restore_protected_metadata(source_root, output_root)
    titles = output_root.xpath("//*[local-name()='metadata']/*[local-name()='title']")
    if title or subtitle:
        if not titles:
            metadata = output_root.xpath("//*[local-name()='metadata']")[0]
            source_titles = source_root.xpath("//*[local-name()='metadata']/*[local-name()='title']")
            node = deepcopy(source_titles[0]) if source_titles else etree.Element("{http://purl.org/dc/elements/1.1/}title")
            metadata.insert(0, node)
            titles = [node]
        source_main, source_subtitle = _metadata_title_parts(source_root)
        output_main, output_subtitle = _metadata_title_parts(output_root)
        has_refined_titles = bool(
            output_root.xpath(
                "//*[local-name()='metadata']/*[local-name()='meta' and "
                "@property='title-type' and @refines]"
            )
        )
        if has_refined_titles:
            title_types: dict[str, str] = {}
            for meta in output_root.xpath(
                "//*[local-name()='metadata']/*[local-name()='meta' and "
                "@property='title-type' and @refines]"
            ):
                title_types[str(meta.get("refines") or "").lstrip("#")] = (
                    " ".join(meta.itertext()).strip().casefold()
                )
            main_node = next(
                (
                    node for node in titles
                    if title_types.get(str(node.get("id") or "")) == "main"
                ),
                titles[0],
            )
            subtitle_node = next(
                (
                    node for node in titles
                    if title_types.get(str(node.get("id") or "")) == "subtitle"
                ),
                titles[1] if len(titles) > 1 else None,
            )
            desired_main = title or output_main or source_main
            desired_subtitle = subtitle or output_subtitle or source_subtitle
            if desired_main and main_node.text != desired_main:
                main_node.text = desired_main
                report.opf_title_changes += 1
            if (
                desired_subtitle
                and subtitle_node is not None
                and subtitle_node.text != desired_subtitle
            ):
                subtitle_node.text = desired_subtitle
                report.opf_title_changes += 1
        else:
            desired = _full_title(title, subtitle)
            if desired and titles[0].text != desired:
                titles[0].text = desired
                report.opf_title_changes += 1
    languages = output_root.xpath("//*[local-name()='metadata']/*[local-name()='language']")
    if not languages:
        metadata = output_root.xpath("//*[local-name()='metadata']")[0]
        languages = [etree.SubElement(metadata, "{http://purl.org/dc/elements/1.1/}language")]
    for node in languages:
        if node.text != language_code:
            node.text = language_code
            report.language_changes += 1
    return _serialize(output_root, output_payload), output_root


def _localize_ncx(
    payload: bytes,
    *,
    title: str,
    labels: Mapping[str, str],
    report: MetadataLocalizationReport,
) -> tuple[bytes, dict[str, str]]:
    root = _parse(payload)
    href_labels: dict[str, str] = {}
    for text_node in root.xpath("//*[local-name()='docTitle']/*[local-name()='text']"):
        if title and text_node.text != title:
            text_node.text = title
            report.navigation_label_changes += 1
    for point in root.xpath("//*[local-name()='navPoint']"):
        text_node = next(iter(point.xpath("./*[local-name()='navLabel']/*[local-name()='text']")), None)
        href = str(point.xpath("string(./*[local-name()='content']/@src)") or "")
        if text_node is None:
            continue
        localized = _localized_label(text_node.text or "", labels)
        if text_node.text != localized:
            text_node.text = localized
            report.navigation_label_changes += 1
        if href:
            href_labels[href] = localized
    return _serialize(root, payload), href_labels


def _localize_nav(
    payload: bytes,
    *,
    labels: Mapping[str, str],
    report: MetadataLocalizationReport,
) -> tuple[bytes, dict[str, str]]:
    root = _parse(payload)
    href_labels: dict[str, str] = {}
    navs = root.xpath(
        "//*[local-name()='nav' and "
        "(@*[local-name()='type']='toc' or contains(concat(' ', normalize-space(@*[local-name()='type']), ' '), ' toc '))]"
    )
    for anchor in [a for nav in navs for a in nav.xpath(".//*[local-name()='a'][@href]")]:
        current = " ".join(anchor.itertext()).strip()
        localized = _localized_label(current, labels)
        if current != localized:
            for child in list(anchor):
                anchor.remove(child)
            anchor.text = localized
            report.navigation_label_changes += 1
        href_labels[str(anchor.get("href") or "")] = localized
    return _serialize(root, payload), href_labels


def _resolved_label_map(toc_path: str, labels_by_href: Mapping[str, str]) -> dict[str, str]:
    base = posixpath.dirname(toc_path)
    resolved: dict[str, str] = {}
    for href, label in labels_by_href.items():
        parsed = urlsplit(href)
        path = posixpath.normpath(posixpath.join(base, unquote(parsed.path)))
        resolved[path] = label
    return resolved


def _localize_xhtml(
    payload: bytes,
    *,
    language_code: str,
    title: str,
    report: MetadataLocalizationReport,
) -> bytes:
    root = _parse(payload)
    changed = False
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"
    if root.get("lang") != language_code:
        root.set("lang", language_code)
        report.language_changes += 1
        changed = True
    if root.get(xml_lang) != language_code:
        root.set(xml_lang, language_code)
        report.language_changes += 1
        changed = True
    title_nodes = root.xpath("//*[local-name()='head']/*[local-name()='title']")
    if not title_nodes:
        heads = root.xpath("//*[local-name()='head']")
        if heads:
            namespace = etree.QName(heads[0]).namespace
            title_nodes = [etree.SubElement(heads[0], f"{{{namespace}}}title" if namespace else "title")]
    for node in title_nodes[:1]:
        if title and node.text != title:
            node.text = title
            report.html_title_changes += 1
            changed = True
    return _serialize(root, payload) if changed else payload


def localize_epub_metadata(
    source_epub: str | Path,
    output_epub: str | Path,
    *,
    target_language: str,
    title: str,
    subtitle: str = "",
    functional_labels: Mapping[str, str] | None = None,
) -> MetadataLocalizationReport:
    """Atomically localize metadata/navigation and restore protected source fields."""
    source_path = Path(source_epub)
    output_path = Path(output_epub)
    language_code = get_language_code(target_language)
    if not language_code:
        raise ValueError(f"Unknown target language: {target_language}")
    labels = {**_DEFAULT_ES_LABELS, **{
        " ".join(str(key).casefold().split()): str(value)
        for key, value in (functional_labels or {}).items()
    }}
    report = MetadataLocalizationReport()
    mode = stat.S_IMODE(output_path.stat().st_mode)
    tmp = tempfile.NamedTemporaryFile(prefix=f"{output_path.stem}.", suffix=".epub", dir=output_path.parent, delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(output_path) as output:
            source_opf = _container_opf(source)
            output_opf = _container_opf(output)
            if source_opf != output_opf:
                raise ValueError(f"OPF path differs from source: {source_opf} != {output_opf}")
            localized_opf, package = _localize_package(
                source.read(source_opf),
                output.read(output_opf),
                title=title,
                subtitle=subtitle,
                language_code=language_code,
                report=report,
            )
            _href_by_id, nav_files, ncx_files = _manifest_info(package, output_opf)
            replacements: dict[str, bytes] = {output_opf: localized_opf}
            document_labels: dict[str, str] = {}
            full_title = _full_title(title, subtitle)
            for toc_path in ncx_files:
                payload, href_labels = _localize_ncx(
                    output.read(toc_path), title=full_title, labels=labels, report=report
                )
                replacements[toc_path] = payload
                document_labels.update(_resolved_label_map(toc_path, href_labels))
            for nav_path in nav_files:
                payload, href_labels = _localize_nav(
                    output.read(nav_path), labels=labels, report=report
                )
                replacements[nav_path] = payload
                document_labels.update(_resolved_label_map(nav_path, href_labels))
            for name in output.namelist():
                if Path(name).suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                replacements[name] = _localize_xhtml(
                    replacements.get(name, output.read(name)),
                    language_code=language_code,
                    title=document_labels.get(name, full_title),
                    report=report,
                )
            with zipfile.ZipFile(tmp_path, "w") as rebuilt:
                for info in output.infolist():
                    payload = replacements.get(info.filename, output.read(info.filename))
                    if payload != output.read(info.filename):
                        report.changed_files += 1
                    _write_member(rebuilt, info, payload)
        tmp_path.replace(output_path)
        output_path.chmod(mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return report
