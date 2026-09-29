"""Deterministic cover, chapter, and navigation finishing for EPUB output."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import html
from io import BytesIO
import os
from pathlib import Path
import posixpath
import re
from typing import Dict, Optional
from urllib.parse import unquote, urlsplit

from lxml import etree
from PIL import Image, ImageDraw, ImageFont

from .lang_support import get_language_code


OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
XHTML_NS = "http://www.w3.org/1999/xhtml"
EPUB_NS = "http://www.idpf.org/2007/ops"
NCX_NS = "http://www.daisy.org/z3986/2005/ncx/"
XML_NS = "http://www.w3.org/XML/1998/namespace"

GENERATED_COVER_IMAGE = "images/verbaloom-cover.jpg"
GENERATED_COVER_PAGE = "verbaloom-cover.xhtml"
GENERATED_NAVIGATION = "verbaloom-nav.xhtml"
GENERATED_NCX = "verbaloom-toc.ncx"

_CHAPTER_CLASS = "verbaloom-chapter"
_CHAPTER_TITLE_CLASS = "verbaloom-chapter-title"
_EXCLUDED_HEADING_CLASSES = {
    "verbaloom-furniture",
    "verbaloom-cover-page",
    "verbaloom-merged-continuation",
}
_EXCLUDED_TOC_TYPES = {
    "cover",
    "copyright-page",
    "halftitlepage",
    "imprint",
    "titlepage",
}
_CHAPTER_HEADING_RE = re.compile(
    r"^(?:chapter|cap[ií]tulo|part|parte|book|libro|prologue|pr[oó]logo|"
    r"epilogue|ep[ií]logo|introduction|introducci[oó]n|preface|prefacio|"
    r"notes|notas|bibliography|bibliograf[ií]a|appendix|ap[eé]ndice)\b",
    re.IGNORECASE,
)

etree.register_namespace("epub", EPUB_NS)


@dataclass
class PublicationStructureReport:
    cover_image_href: str = ""
    cover_page_href: str = ""
    cover_generated: bool = False
    cover_page_created: bool = False
    navigation_href: str = ""
    navigation_created: bool = False
    ncx_href: str = ""
    ncx_created: bool = False
    navigation_entries: int = 0
    chapter_headings: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class _NavigationEntry:
    label: str
    target: str


def _qname(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def _package_namespace(root: etree._Element) -> str:
    return etree.QName(root).namespace or OPF_NS


def _local_name(node: etree._Element) -> str:
    return etree.QName(node).localname.casefold()


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _element_text(node: etree._Element) -> str:
    return _clean_text(" ".join(node.itertext()))


def _add_class(node: etree._Element, class_name: str) -> bool:
    classes = [value for value in str(node.get("class") or "").split() if value]
    if class_name in classes:
        return False
    node.set("class", " ".join([*classes, class_name]))
    return True


def _semantic_types(node: etree._Element) -> set[str]:
    values: set[str] = set()
    for candidate in node.xpath("ancestor-or-self::*"):
        for key, value in candidate.attrib.items():
            if str(key).rsplit("}", 1)[-1].rsplit(":", 1)[-1].casefold() == "type":
                values.update(str(value or "").casefold().split())
    return values


def _resolved_path(opf_dir: Path, href: str) -> Path:
    return (opf_dir / href).resolve()


def _safe_resource_path(
    package_root: Path,
    opf_dir: Path,
    href: str,
) -> Optional[Path]:
    if not href:
        return None
    path = _resolved_path(opf_dir, unquote(urlsplit(href).path))
    try:
        path.relative_to(package_root)
    except ValueError:
        return None
    return path


def _path_identity(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _manifest(root: etree._Element) -> etree._Element:
    nodes = root.xpath("//*[local-name()='manifest']")
    if nodes:
        return nodes[0]
    metadata = root.xpath("//*[local-name()='metadata']")
    node = etree.Element(_qname(_package_namespace(root), "manifest"))
    if metadata:
        metadata[0].addnext(node)
    else:
        root.insert(0, node)
    return node


def _spine(root: etree._Element) -> etree._Element:
    nodes = root.xpath("//*[local-name()='spine']")
    if nodes:
        return nodes[0]
    node = etree.Element(_qname(_package_namespace(root), "spine"))
    manifest = _manifest(root)
    manifest.addnext(node)
    return node


def _metadata(root: etree._Element) -> etree._Element:
    nodes = root.xpath("//*[local-name()='metadata']")
    if nodes:
        return nodes[0]
    node = etree.Element(_qname(_package_namespace(root), "metadata"))
    root.insert(0, node)
    return node


def _manifest_items(root: etree._Element) -> list[etree._Element]:
    return root.xpath("//*[local-name()='manifest']/*[local-name()='item']")


def _manifest_item_by_id(root: etree._Element, item_id: str) -> Optional[etree._Element]:
    return next(
        (node for node in _manifest_items(root) if str(node.get("id") or "") == item_id),
        None,
    )


def _manifest_item_by_href(root: etree._Element, href: str) -> Optional[etree._Element]:
    normalized = posixpath.normpath(href)
    return next(
        (
            node
            for node in _manifest_items(root)
            if posixpath.normpath(str(node.get("href") or "")) == normalized
        ),
        None,
    )


def _unique_manifest_id(root: etree._Element, preferred: str) -> str:
    existing = {str(node.get("id") or "") for node in _manifest_items(root)}
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}-{index}" in existing:
        index += 1
    return f"{preferred}-{index}"


def _ensure_manifest_item(
    root: etree._Element,
    *,
    preferred_id: str,
    href: str,
    media_type: str,
    properties: str = "",
) -> tuple[etree._Element, bool]:
    item = _manifest_item_by_href(root, href)
    created = item is None
    if item is None:
        item = etree.SubElement(_manifest(root), _qname(_package_namespace(root), "item"))
        item.set("id", _unique_manifest_id(root, preferred_id))
        item.set("href", href)
    elif not str(item.get("id") or "").strip():
        item.set("id", _unique_manifest_id(root, preferred_id))
    item.set("media-type", media_type)
    if properties:
        merged = set(str(item.get("properties") or "").split())
        merged.update(properties.split())
        item.set("properties", " ".join(sorted(merged)))
    return item, created


def _package_version(root: etree._Element) -> float:
    try:
        return float(str(root.get("version") or "2.0").split()[0])
    except ValueError:
        return 2.0


def _metadata_value(root: etree._Element, local_name: str) -> str:
    value = root.xpath(
        "string(//*[local-name()='metadata']/*[local-name()=$name][1])",
        name=local_name,
    )
    return _clean_text(str(value or ""))


def _cover_image_item(root: etree._Element) -> Optional[etree._Element]:
    cover_id = str(
        root.xpath(
            "string(//*[local-name()='metadata']/*[local-name()='meta' and "
            "translate(@name, 'COVER', 'cover')='cover']/@content)"
        )
        or ""
    )
    if cover_id:
        item = _manifest_item_by_id(root, cover_id)
        if item is not None:
            return item
    return next(
        (
            item
            for item in _manifest_items(root)
            if "cover-image" in str(item.get("properties") or "").split()
        ),
        None,
    )


def _cover_page_href(root: etree._Element) -> str:
    guide_href = str(
        root.xpath(
            "string(//*[local-name()='guide']/*[local-name()='reference' and "
            "translate(@type, 'COVER', 'cover')='cover']/@href)"
        )
        or ""
    )
    if guide_href:
        return unquote(urlsplit(guide_href).path) or guide_href
    for item in _manifest_items(root):
        item_id = str(item.get("id") or "").casefold()
        href = str(item.get("href") or "")
        if (
            str(item.get("media-type") or "") == "application/xhtml+xml"
            and "cover" in item_id
            and href
        ):
            return href
    return ""


def _font(size: int, *, bold: bool = False, italic: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
                "/System/Library/Fonts/NewYork.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
                "DejaVuSerif-Bold.ttf",
            ]
        )
    elif italic:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Georgia Italic.ttf",
                "/System/Library/Fonts/NewYorkItalic.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
                "DejaVuSerif-Italic.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/System/Library/Fonts/NewYork.ttf",
                "/System/Library/Fonts/Supplemental/Georgia.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
                "DejaVuSerif.ttf",
            ]
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    value: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = _clean_text(value).split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_title(
    draw: ImageDraw.ImageDraw,
    value: str,
    max_width: int,
    max_height: int,
) -> tuple[ImageFont.ImageFont, list[str], int]:
    for size in range(168, 75, -6):
        font = _font(size, bold=True)
        lines = _wrap_lines(draw, value, font, max_width)
        line_height = max(1, draw.textbbox((0, 0), "Ag", font=font)[3])
        leading = int(size * 0.24)
        total_height = len(lines) * line_height + max(0, len(lines) - 1) * leading
        if len(lines) <= 7 and total_height <= max_height:
            return font, lines, leading
    font = _font(74, bold=True)
    return font, _wrap_lines(draw, value, font, max_width), 16


def _cover_theme(title: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int], str]:
    normalized = title.casefold()
    if any(word in normalized for word in ("film", "cine", "cinema", "película")):
        return (20, 21, 24), (222, 65, 55), (244, 240, 229), "film"
    if any(word in normalized for word in ("mar", "sea", "road", "camino", "viaje", "odisea")):
        return (13, 43, 61), (68, 181, 189), (244, 236, 211), "journey"
    if any(word in normalized for word in ("historia", "history", "guerra", "war", "imperio")):
        return (40, 31, 32), (166, 54, 62), (238, 223, 193), "history"
    if any(word in normalized for word in ("mente", "mind", "psico", "filosof", "realidad")):
        return (20, 37, 41), (61, 163, 142), (238, 232, 211), "mind"
    palettes = [
        ((22, 43, 35), (212, 93, 72), (244, 236, 215)),
        ((35, 31, 52), (211, 145, 67), (242, 232, 214)),
        ((24, 42, 56), (77, 149, 171), (241, 231, 207)),
    ]
    index = int(hashlib.sha256(title.encode("utf-8")).hexdigest()[:2], 16) % len(palettes)
    background, accent, paper = palettes[index]
    return background, accent, paper, "literary"


def _draw_cover_motif(
    draw: ImageDraw.ImageDraw,
    motif: str,
    accent: tuple[int, int, int],
    paper: tuple[int, int, int],
) -> None:
    if motif == "film":
        for offset in range(4):
            top = 250 + offset * 115
            draw.rounded_rectangle((1120, top, 1470, top + 82), radius=12, outline=accent, width=12)
            draw.rectangle((1140, top + 18, 1180, top + 58), fill=paper)
            draw.rectangle((1410, top + 18, 1450, top + 58), fill=paper)
    elif motif == "journey":
        for offset in range(5):
            y = 340 + offset * 70
            draw.arc((960 - offset * 35, y, 1580 + offset * 35, y + 500), 185, 340, fill=accent, width=10)
    elif motif == "mind":
        for offset in range(6):
            inset = offset * 44
            draw.ellipse((1070 + inset, 220 + inset, 1580 - inset, 730 - inset), outline=accent, width=10)
    elif motif == "history":
        for offset, height in enumerate((400, 270, 480, 330, 420)):
            x = 1100 + offset * 78
            draw.rectangle((x, 220, x + 26, 220 + height), fill=accent)
    else:
        draw.rectangle((1090, 240, 1490, 270), fill=accent)
        draw.rectangle((1190, 320, 1490, 350), fill=paper)
        draw.rectangle((1290, 400, 1490, 430), fill=accent)


def render_typographic_cover_bytes(
    *,
    title: str,
    subtitle: str = "",
    author: str = "",
    language_code: str = "es",
) -> bytes:
    """Render a deterministic, reader-safe 1600x2560 JPEG cover."""
    title = _clean_text(title) or "Libro traducido"
    subtitle = _clean_text(subtitle)
    author = _clean_text(author)
    background, accent, paper, motif = _cover_theme(title)
    image = Image.new("RGB", (1600, 2560), background)
    draw = ImageDraw.Draw(image)
    _draw_cover_motif(draw, motif, accent, paper)

    left = 150
    max_width = 1220
    draw.rectangle((left, 215, left + 180, 234), fill=accent)
    edition_label = "EDICIÓN EN ESPAÑOL" if language_code == "es" else "TRANSLATED EDITION"
    label_font = _font(36, bold=True)
    draw.text((left, 270), edition_label, font=label_font, fill=accent)

    title_font, title_lines, leading = _fit_title(draw, title, max_width, 920)
    y = 720 if motif == "film" else 610
    for line in title_lines:
        draw.text((left, y), line, font=title_font, fill=paper)
        line_box = draw.textbbox((left, y), line, font=title_font)
        y = line_box[3] + leading

    if subtitle:
        y += 70
        subtitle_font = _font(60, italic=True)
        for line in _wrap_lines(draw, subtitle, subtitle_font, max_width):
            draw.text((left, y), line, font=subtitle_font, fill=paper)
            y = draw.textbbox((left, y), line, font=subtitle_font)[3] + 18

    draw.rectangle((left, 2160, 1450, 2165), fill=accent)
    if author:
        author_font = _font(58, bold=True)
        author_lines = _wrap_lines(draw, author, author_font, max_width)
        author_y = 2245
        for line in author_lines[:2]:
            draw.text((left, author_y), line, font=author_font, fill=paper)
            author_y = draw.textbbox((left, author_y), line, font=author_font)[3] + 14

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, optimize=True, progressive=True)
    return buffer.getvalue()


def _write_cover_page(
    path: Path,
    *,
    opf_dir: Path,
    title: str,
    cover_href: str,
    language_code: str,
    epub3: bool,
) -> None:
    cover_path = _resolved_path(opf_dir, unquote(urlsplit(cover_href).path))
    relative_cover = Path(os.path.relpath(cover_path, path.parent)).as_posix()
    epub_namespace = f' xmlns:epub="{EPUB_NS}"' if epub3 else ""
    cover_semantics = ' epub:type="cover"' if epub3 else ""
    payload = f'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="{XHTML_NS}"{epub_namespace} lang="{html.escape(language_code)}" xml:lang="{html.escape(language_code)}">
  <head>
    <title>{html.escape(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <style type="text/css">html,body{{margin:0;padding:0;min-height:100%;background:#111318;text-align:center}}body{{display:flex;align-items:center;justify-content:center;min-height:100vh}}img{{display:block;width:100%;height:auto;max-height:100vh;object-fit:contain;margin:0 auto}}</style>
  </head>
  <body{cover_semantics} class="verbaloom-cover-page">
    <div class="verbaloom-cover-image"><img src="{html.escape(relative_cover, quote=True)}" alt="{html.escape(title, quote=True)}"/></div>
  </body>
</html>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _ensure_cover_declaration(
    root: etree._Element,
    *,
    cover_item: etree._Element,
    cover_page_href: str,
    target_language: str,
) -> None:
    cover_id = str(cover_item.get("id") or "")
    metadata = _metadata(root)
    meta_nodes = metadata.xpath(
        "./*[local-name()='meta' and translate(@name, 'COVER', 'cover')='cover']"
    )
    if cover_id:
        if meta_nodes:
            meta_nodes[0].set("content", cover_id)
        else:
            meta = etree.SubElement(metadata, _qname(_package_namespace(root), "meta"))
            meta.set("name", "cover")
            meta.set("content", cover_id)
    if _package_version(root) >= 3.0:
        properties = set(str(cover_item.get("properties") or "").split())
        properties.add("cover-image")
        cover_item.set("properties", " ".join(sorted(properties)))

    guide_nodes = root.xpath("//*[local-name()='guide']")
    if guide_nodes:
        guide = guide_nodes[0]
    else:
        guide = etree.SubElement(root, _qname(_package_namespace(root), "guide"))
    references = guide.xpath(
        "./*[local-name()='reference' and translate(@type, 'COVER', 'cover')='cover']"
    )
    if references:
        reference = references[0]
    else:
        reference = etree.SubElement(guide, _qname(_package_namespace(root), "reference"))
        reference.set("type", "cover")
    reference.set("title", "Portada" if get_language_code(target_language) == "es" else "Cover")
    reference.set("href", cover_page_href)


def _ensure_first_spine_item(root: etree._Element, item_id: str) -> None:
    spine = _spine(root)
    matches = spine.xpath("./*[local-name()='itemref' and @idref=$item_id]", item_id=item_id)
    if matches:
        node = matches[0]
        if len(spine) and spine[0] is not node:
            spine.remove(node)
            spine.insert(0, node)
    else:
        node = etree.Element(_qname(_package_namespace(root), "itemref"))
        node.set("idref", item_id)
        node.set("linear", "yes")
        spine.insert(0, node)


def _heading_candidates(root: etree._Element) -> list[etree._Element]:
    candidates = root.xpath(
        "//*[local-name()='body']//*[local-name()='h1' or local-name()='h2' or local-name()='h3']"
    )
    accepted: list[etree._Element] = []
    has_h1 = any(_local_name(node) == "h1" for node in candidates)
    for node in candidates:
        classes = set(str(node.get("class") or "").split())
        text = _element_text(node)
        if (
            not text
            or len(text) > 180
            or classes & _EXCLUDED_HEADING_CLASSES
            or _semantic_types(node) & _EXCLUDED_TOC_TYPES
        ):
            continue
        if text.casefold() in {"índice", "indice", "contents", "table of contents"}:
            continue
        level = _local_name(node)
        if level == "h1" or _CHAPTER_HEADING_RE.search(text):
            accepted.append(node)
        elif level == "h2" and not has_h1 and not accepted:
            accepted.append(node)
    return accepted


def _ensure_unique_id(root: etree._Element, node: etree._Element, preferred: str) -> str:
    current = str(node.get("id") or "").strip()
    if current:
        return current
    existing = {str(value) for value in root.xpath("//@id") if value}
    candidate = preferred
    index = 2
    while candidate in existing:
        candidate = f"{preferred}-{index}"
        index += 1
    node.set("id", candidate)
    return candidate


def _mark_chapter_container(node: etree._Element, *, epub3: bool) -> None:
    containers = node.xpath("ancestor::*[local-name()='section' or local-name()='article'][1]")
    container = containers[0] if containers else next(
        iter(node.xpath("ancestor::*[local-name()='body'][1]")),
        None,
    )
    if container is None:
        return
    _add_class(container, _CHAPTER_CLASS)
    existing_types = [
        key
        for key in container.attrib
        if str(key).rsplit("}", 1)[-1].rsplit(":", 1)[-1].casefold() == "type"
    ]
    if epub3 and not existing_types:
        container.set(_qname(EPUB_NS, "type"), "chapter")


def _resolve_navigation_target(document_href: str, href: str) -> str:
    parsed = urlsplit(href)
    if parsed.scheme or not parsed.path:
        return ""
    base_dir = posixpath.dirname(document_href) or "."
    path = posixpath.normpath(posixpath.join(base_dir, unquote(parsed.path)))
    return f"{path}#{parsed.fragment}" if parsed.fragment else path


def _existing_navigation_entries(
    root: etree._Element,
    opf_dir: Path,
    package_root: Path,
) -> dict[str, list[_NavigationEntry]]:
    entries: dict[str, list[_NavigationEntry]] = {}
    for item in _manifest_items(root):
        href = str(item.get("href") or "")
        properties = set(str(item.get("properties") or "").split())
        media_type = str(item.get("media-type") or "")
        is_nav = "nav" in properties
        is_ncx = media_type == "application/x-dtbncx+xml"
        if not href or not (is_nav or is_ncx):
            continue
        path = _safe_resource_path(package_root, opf_dir, href)
        if path is None or not path.is_file():
            continue
        try:
            document = etree.parse(str(path), etree.XMLParser(recover=False, huge_tree=True)).getroot()
        except (OSError, etree.XMLSyntaxError):
            continue
        if is_ncx:
            raw_entries = [
                (
                    _clean_text(" ".join(node.xpath("./*[local-name()='navLabel']//*[local-name()='text']/text()"))),
                    str(node.xpath("string(./*[local-name()='content']/@src)") or ""),
                )
                for node in document.xpath("//*[local-name()='navPoint']")
            ]
        else:
            toc_nodes = [
                node
                for node in document.xpath("//*[local-name()='nav']")
                if "toc" in _nav_type(node).split()
            ]
            raw_entries = [
                (_element_text(anchor), str(anchor.get("href") or ""))
                for toc in toc_nodes
                for anchor in toc.xpath(".//*[local-name()='a'][@href]")
            ]
        for label, raw_target in raw_entries:
            target = _resolve_navigation_target(href, raw_target)
            target_path = target.partition("#")[0]
            if label and target_path:
                document_entries = entries.setdefault(target_path, [])
                if not any(entry.target == target for entry in document_entries):
                    document_entries.append(
                        _NavigationEntry(label=label, target=target)
                    )
    return entries


def _navigation_entries(
    *,
    opf_root: etree._Element,
    opf_dir: Path,
    content_files: list[str],
    parsed_xhtml_docs: Dict[str, etree._Element],
    package_root: Path,
    cover_page_href: str,
    target_language: str,
    epub3: bool,
) -> tuple[list[_NavigationEntry], int]:
    docs = {_path_identity(path): root for path, root in parsed_xhtml_docs.items()}
    existing_entries = _existing_navigation_entries(
        opf_root,
        opf_dir,
        package_root,
    )
    entries: list[_NavigationEntry] = []
    marked_headings = 0
    section_label = "Sección" if get_language_code(target_language) == "es" else "Section"
    for ordinal, href in enumerate(content_files, start=1):
        normalized_href = posixpath.normpath(str(href or ""))
        if not normalized_href or normalized_href == posixpath.normpath(cover_page_href):
            continue
        root = docs.get(_path_identity(opf_dir / normalized_href))
        if root is None:
            continue
        represented_ids: set[str] = set()
        document_headings = _heading_candidates(root)
        for existing in existing_entries.get(normalized_href, []):
            _target_path, _separator, fragment = existing.target.partition("#")
            target_node = next(
                iter(root.xpath("//*[@id=$fragment]", fragment=fragment)),
                None,
            ) if fragment else None
            if target_node is not None and _local_name(target_node) in {"h1", "h2", "h3"}:
                target_id = str(target_node.get("id") or fragment)
                label = _element_text(target_node) or existing.label
                _add_class(target_node, _CHAPTER_TITLE_CLASS)
                _mark_chapter_container(target_node, epub3=epub3)
                marked_headings += 1
            elif len(document_headings) == 1:
                target_node = document_headings[0]
                target_id = _ensure_unique_id(
                    root,
                    target_node,
                    f"verbaloom-chapter-{len(entries) + 1:03d}",
                )
                label = _element_text(target_node) or existing.label
                _add_class(target_node, _CHAPTER_TITLE_CLASS)
                _mark_chapter_container(target_node, epub3=epub3)
                marked_headings += 1
            else:
                body = next(iter(root.xpath("//*[local-name()='body']")), None)
                if body is None:
                    continue
                target_id = _ensure_unique_id(
                    root,
                    body,
                    f"verbaloom-section-{len(entries) + 1:03d}",
                )
                label = existing.label
            represented_ids.add(target_id)
            entries.append(
                _NavigationEntry(label=label, target=f"{normalized_href}#{target_id}")
            )

        for heading in document_headings:
            target_id = _ensure_unique_id(
                root,
                heading,
                f"verbaloom-chapter-{len(entries) + 1:03d}",
            )
            if target_id in represented_ids:
                continue
            _add_class(heading, _CHAPTER_TITLE_CLASS)
            _mark_chapter_container(heading, epub3=epub3)
            marked_headings += 1
            entries.append(
                _NavigationEntry(
                    label=_element_text(heading),
                    target=f"{normalized_href}#{target_id}",
                )
            )

    if not entries:
        for href in content_files:
            normalized_href = posixpath.normpath(str(href or ""))
            root = docs.get(_path_identity(opf_dir / normalized_href))
            if root is None or normalized_href == posixpath.normpath(cover_page_href):
                continue
            body = next(iter(root.xpath("//*[local-name()='body']")), None)
            if body is None:
                continue
            target_id = _ensure_unique_id(root, body, "verbaloom-section-001")
            _add_class(body, _CHAPTER_CLASS)
            titles = root.xpath("//*[local-name()='head']/*[local-name()='title'][1]")
            label = _element_text(titles[0]) if titles else f"{section_label} 1"
            entries.append(
                _NavigationEntry(label=label, target=f"{normalized_href}#{target_id}")
            )
            break
    return entries, marked_headings


def _relative_target(document_href: str, target: str) -> str:
    path, separator, fragment = target.partition("#")
    base_dir = posixpath.dirname(document_href) or "."
    relative = posixpath.relpath(path, base_dir)
    return f"{relative}{separator}{fragment}" if separator else relative


def _new_xhtml_document(language_code: str, title: str) -> etree._Element:
    root = etree.Element(
        _qname(XHTML_NS, "html"),
        nsmap={None: XHTML_NS, "epub": EPUB_NS},
    )
    root.set("lang", language_code)
    root.set(_qname(XML_NS, "lang"), language_code)
    head = etree.SubElement(root, _qname(XHTML_NS, "head"))
    title_node = etree.SubElement(head, _qname(XHTML_NS, "title"))
    title_node.text = title
    etree.SubElement(root, _qname(XHTML_NS, "body"))
    return root


def _nav_type(node: etree._Element) -> str:
    return str(node.get(_qname(EPUB_NS, "type")) or node.get("epub:type") or "")


def _write_navigation(
    path: Path,
    *,
    href: str,
    title: str,
    entries: list[_NavigationEntry],
    cover_page_href: str,
    language_code: str,
) -> None:
    if path.exists():
        try:
            root = etree.parse(str(path), etree.XMLParser(recover=False, huge_tree=True)).getroot()
        except (OSError, etree.XMLSyntaxError):
            root = _new_xhtml_document(language_code, title)
    else:
        root = _new_xhtml_document(language_code, title)
    root.set("lang", language_code)
    root.set(_qname(XML_NS, "lang"), language_code)
    body = next(iter(root.xpath("//*[local-name()='body']")), None)
    if body is None:
        body = etree.SubElement(root, _qname(XHTML_NS, "body"))
    for node in list(body.xpath(".//*[local-name()='nav']")):
        if "toc" in _nav_type(node).split():
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

    toc = etree.Element(_qname(XHTML_NS, "nav"))
    toc.set(_qname(EPUB_NS, "type"), "toc")
    toc.set("id", "toc")
    heading = etree.SubElement(toc, _qname(XHTML_NS, "h1"))
    heading.text = "Índice" if language_code == "es" else "Contents"
    ordered = etree.SubElement(toc, _qname(XHTML_NS, "ol"))
    for entry in entries:
        item = etree.SubElement(ordered, _qname(XHTML_NS, "li"))
        anchor = etree.SubElement(item, _qname(XHTML_NS, "a"))
        anchor.set("href", _relative_target(href, entry.target))
        anchor.text = entry.label
    body.insert(0, toc)

    landmarks = next(
        (
            node
            for node in body.xpath(".//*[local-name()='nav']")
            if "landmarks" in _nav_type(node).split()
        ),
        None,
    )
    if landmarks is None and cover_page_href:
        landmarks = etree.SubElement(body, _qname(XHTML_NS, "nav"))
        landmarks.set(_qname(EPUB_NS, "type"), "landmarks")
        landmarks.set("hidden", "hidden")
        landmarks_heading = etree.SubElement(landmarks, _qname(XHTML_NS, "h2"))
        landmarks_heading.text = "Guía" if language_code == "es" else "Guide"
        landmarks_list = etree.SubElement(landmarks, _qname(XHTML_NS, "ol"))
        landmark = etree.SubElement(landmarks_list, _qname(XHTML_NS, "li"))
        link = etree.SubElement(landmark, _qname(XHTML_NS, "a"))
        link.set(_qname(EPUB_NS, "type"), "cover")
        link.set("href", _relative_target(href, cover_page_href))
        link.text = "Portada" if language_code == "es" else "Cover"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        etree.tostring(
            root.getroottree(),
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True,
        )
    )


def _write_ncx(
    path: Path,
    *,
    href: str,
    title: str,
    identifier: str,
    entries: list[_NavigationEntry],
) -> None:
    if path.exists():
        try:
            root = etree.parse(str(path), etree.XMLParser(recover=False, huge_tree=True)).getroot()
        except (OSError, etree.XMLSyntaxError):
            root = etree.Element(
                _qname(NCX_NS, "ncx"),
                nsmap={None: NCX_NS},
                version="2005-1",
            )
    else:
        root = etree.Element(
            _qname(NCX_NS, "ncx"),
            nsmap={None: NCX_NS},
            version="2005-1",
        )
    namespace = etree.QName(root).namespace or NCX_NS
    head = next(iter(root.xpath("./*[local-name()='head']")), None)
    if head is None:
        head = etree.Element(_qname(namespace, "head"))
        root.insert(0, head)
    uid = next(
        iter(head.xpath("./*[local-name()='meta' and @name='dtb:uid']")),
        None,
    )
    if uid is None:
        uid = etree.SubElement(head, _qname(namespace, "meta"))
        uid.set("name", "dtb:uid")
    uid.set("content", identifier or "urn:uuid:verbaloom")
    doc_title = next(iter(root.xpath("./*[local-name()='docTitle']")), None)
    if doc_title is None:
        doc_title = etree.Element(_qname(namespace, "docTitle"))
        head.addnext(doc_title)
    title_text = next(iter(doc_title.xpath("./*[local-name()='text']")), None)
    if title_text is None:
        title_text = etree.SubElement(doc_title, _qname(namespace, "text"))
    title_text.text = title
    nav_map = next(iter(root.xpath("./*[local-name()='navMap']")), None)
    if nav_map is None:
        nav_map = etree.Element(_qname(namespace, "navMap"))
        doc_title.addnext(nav_map)
    else:
        for child in list(nav_map):
            nav_map.remove(child)
    for index, entry in enumerate(entries, start=1):
        point = etree.SubElement(nav_map, _qname(namespace, "navPoint"))
        point.set("id", f"navpoint-{index:03d}")
        point.set("playOrder", str(index))
        label = etree.SubElement(point, _qname(namespace, "navLabel"))
        etree.SubElement(label, _qname(namespace, "text")).text = entry.label
        content = etree.SubElement(point, _qname(namespace, "content"))
        content.set("src", _relative_target(href, entry.target))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        etree.tostring(
            root.getroottree(),
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=True,
        )
    )


def finish_epub_publication(
    *,
    opf_tree: etree._ElementTree,
    opf_dir: str | Path,
    content_files: list[str],
    parsed_xhtml_docs: Dict[str, etree._Element],
    target_language: str,
    title: str = "",
    subtitle: str = "",
    author: str = "",
    package_root: str | Path | None = None,
) -> PublicationStructureReport:
    """Guarantee a cover, chapter semantics, and valid package navigation."""
    report = PublicationStructureReport()
    root = opf_tree.getroot()
    opf_dir_path = Path(opf_dir)
    package_root_path = Path(package_root or opf_dir_path).resolve()
    language_code = get_language_code(target_language) or "en"
    title = _clean_text(title) or _metadata_value(root, "title") or "Libro traducido"
    subtitle = _clean_text(subtitle)
    author = _clean_text(author) or _metadata_value(root, "creator")
    epub3 = _package_version(root) >= 3.0

    cover_item = _cover_image_item(root)
    cover_href = str(cover_item.get("href") or "") if cover_item is not None else ""
    cover_path = _safe_resource_path(
        package_root_path,
        opf_dir_path,
        cover_href,
    )
    if cover_item is None or cover_path is None or not cover_path.is_file():
        stale_cover_item = cover_item
        if cover_item is not None:
            properties = set(str(cover_item.get("properties") or "").split())
            properties.discard("cover-image")
            if properties:
                cover_item.set("properties", " ".join(sorted(properties)))
            else:
                cover_item.attrib.pop("properties", None)
        cover_path = opf_dir_path / GENERATED_COVER_IMAGE
        cover_path.parent.mkdir(parents=True, exist_ok=True)
        cover_path.write_bytes(
            render_typographic_cover_bytes(
                title=title,
                subtitle=subtitle,
                author=author,
                language_code=language_code,
            )
        )
        report.cover_generated = True
        cover_item, _created = _ensure_manifest_item(
            root,
            preferred_id="verbaloom-cover-image",
            href=GENERATED_COVER_IMAGE,
            media_type="image/jpeg",
            properties="cover-image" if _package_version(root) >= 3.0 else "",
        )
        if stale_cover_item is not None and stale_cover_item is not cover_item:
            parent = stale_cover_item.getparent()
            if parent is not None:
                parent.remove(stale_cover_item)
    report.cover_image_href = str(cover_item.get("href") or "")

    cover_page_href = _cover_page_href(root)
    cover_page_item = _manifest_item_by_href(root, cover_page_href) if cover_page_href else None
    cover_page_path = _safe_resource_path(
        package_root_path,
        opf_dir_path,
        cover_page_href,
    )
    parsed_document_paths = {
        _path_identity(path) for path in parsed_xhtml_docs
    }
    cover_page_available = bool(
        cover_page_path is not None
        and (
            cover_page_path.is_file()
            or _path_identity(cover_page_path) in parsed_document_paths
        )
    )
    if (
        not cover_page_href
        or cover_page_item is None
        or cover_page_path is None
        or not cover_page_available
        or report.cover_generated
    ):
        stale_cover_page_item = cover_page_item
        stale_cover_page_id = str(cover_page_item.get("id") or "") if cover_page_item is not None else ""
        cover_page_href = GENERATED_COVER_PAGE
        cover_page_item, _created = _ensure_manifest_item(
            root,
            preferred_id="verbaloom-cover-page",
            href=cover_page_href,
            media_type="application/xhtml+xml",
        )
        cover_page_path = opf_dir_path / cover_page_href
        if stale_cover_page_id and stale_cover_page_id != str(cover_page_item.get("id") or ""):
            for itemref in _spine(root).xpath(
                "./*[local-name()='itemref' and @idref=$item_id]",
                item_id=stale_cover_page_id,
            ):
                _spine(root).remove(itemref)
        if stale_cover_page_item is not None and stale_cover_page_item is not cover_page_item:
            parent = stale_cover_page_item.getparent()
            if parent is not None:
                parent.remove(stale_cover_page_item)
    if not cover_page_path.is_file():
        _write_cover_page(
            cover_page_path,
            opf_dir=opf_dir_path,
            title=title,
            cover_href=report.cover_image_href,
            language_code=language_code,
            epub3=epub3,
        )
        report.cover_page_created = True
    report.cover_page_href = cover_page_href
    cover_page_id = str(cover_page_item.get("id") or "").strip()
    if not cover_page_id:
        cover_page_id = _unique_manifest_id(root, "verbaloom-cover-page")
        cover_page_item.set("id", cover_page_id)
    _ensure_cover_declaration(
        root,
        cover_item=cover_item,
        cover_page_href=cover_page_href,
        target_language=target_language,
    )
    _ensure_first_spine_item(root, cover_page_id)

    entries, report.chapter_headings = _navigation_entries(
        opf_root=root,
        opf_dir=opf_dir_path,
        content_files=content_files,
        parsed_xhtml_docs=parsed_xhtml_docs,
        package_root=package_root_path,
        cover_page_href=cover_page_href,
        target_language=target_language,
        epub3=epub3,
    )
    report.navigation_entries = len(entries)

    if epub3:
        nav_item = next(
            (
                item
                for item in _manifest_items(root)
                if "nav" in str(item.get("properties") or "").split()
            ),
            None,
        )
        nav_href = str(nav_item.get("href") or "") if nav_item is not None else ""
        nav_path = _safe_resource_path(package_root_path, opf_dir_path, nav_href)
        if nav_href and nav_path is None:
            stale_nav_item = nav_item
            nav_item = None
            nav_href = ""
            parent = stale_nav_item.getparent() if stale_nav_item is not None else None
            if parent is not None:
                parent.remove(stale_nav_item)
        if not nav_href:
            nav_href = GENERATED_NAVIGATION
            nav_item, report.navigation_created = _ensure_manifest_item(
                root,
                preferred_id="verbaloom-navigation",
                href=nav_href,
                media_type="application/xhtml+xml",
                properties="nav",
            )
            nav_path = opf_dir_path / nav_href
        else:
            properties = set(str(nav_item.get("properties") or "").split())
            properties.add("nav")
            nav_item.set("properties", " ".join(sorted(properties)))
            report.navigation_created = not nav_path.exists()
        _write_navigation(
            nav_path,
            href=nav_href,
            title=title,
            entries=entries,
            cover_page_href=cover_page_href,
            language_code=language_code,
        )
        report.navigation_href = nav_href
    else:
        ncx_item = next(
            (
                item
                for item in _manifest_items(root)
                if str(item.get("media-type") or "") == "application/x-dtbncx+xml"
            ),
            None,
        )
        ncx_href = str(ncx_item.get("href") or "") if ncx_item is not None else ""
        ncx_path = _safe_resource_path(package_root_path, opf_dir_path, ncx_href)
        if ncx_href and ncx_path is None:
            stale_ncx_item = ncx_item
            ncx_item = None
            ncx_href = ""
            parent = stale_ncx_item.getparent() if stale_ncx_item is not None else None
            if parent is not None:
                parent.remove(stale_ncx_item)
        if not ncx_href:
            ncx_href = GENERATED_NCX
            ncx_item, report.ncx_created = _ensure_manifest_item(
                root,
                preferred_id="verbaloom-ncx",
                href=ncx_href,
                media_type="application/x-dtbncx+xml",
            )
            ncx_path = opf_dir_path / ncx_href
        else:
            report.ncx_created = not ncx_path.exists()
        _spine(root).set("toc", str(ncx_item.get("id") or ""))
        _write_ncx(
            ncx_path,
            href=ncx_href,
            title=title,
            identifier=_metadata_value(root, "identifier"),
            entries=entries,
        )
        report.ncx_href = ncx_href

    return report
