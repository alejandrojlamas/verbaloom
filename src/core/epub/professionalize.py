"""Conservative publication-quality finishing for translated EPUBs.

The layer preserves source resources and rich publisher CSS. It only declares
an existing cover that the source failed to identify and supplies a namespaced
reading stylesheet when the source stylesheet is missing or effectively empty.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
from pathlib import Path
import re
from typing import Dict, Optional

from lxml import etree


OPF_NS = "http://www.idpf.org/2007/opf"
XHTML_NS = "http://www.w3.org/1999/xhtml"
XLINK_NS = "http://www.w3.org/1999/xlink"
VERBALOOM_CSS_NAME = "verbaloom-professional.css"
VERBALOOM_CSS_MARKER = "VerbaLoom professional reading layer"

# Deprecated import aliases for integrations that imported the old constant
# names.  Their values are canonical, so they cannot generate legacy output.
TBL_CSS_NAME = VERBALOOM_CSS_NAME
TBL_CSS_MARKER = VERBALOOM_CSS_MARKER

_BODY_CLASS = "verbaloom-book"
_COVER_CLASS = "verbaloom-cover-page"
_FURNITURE_CLASS = "verbaloom-furniture"
_SCENE_BREAK_CLASS = "verbaloom-scene-break"
_SECTION_MARKER_CLASS = "verbaloom-section-marker"
_MERGED_CONTINUATION_CLASS = "verbaloom-merged-continuation"
_BLOCKQUOTE_FLOW_REPAIR_CLASS = "verbaloom-blockquote-flow-repair"
_LEGACY_COVER_CLASS = "tbl-cover-page"

_BLOCK_LEVEL_ELEMENTS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "details",
    "div",
    "dl",
    "fieldset",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
}

_PROFESSIONAL_CSS = f"""
/* {VERBALOOM_CSS_MARKER} */
body.verbaloom-book {{
  margin: 5%;
  width: 90%;
  min-width: 0;
  max-width: 100%;
  box-sizing: border-box;
  overflow-x: hidden;
  overflow-wrap: anywhere;
  line-height: 1.55;
  text-align: left;
  widows: 2;
  orphans: 2;
}}
body.verbaloom-book p,
body.verbaloom-book li,
body.verbaloom-book blockquote,
body.verbaloom-book th,
body.verbaloom-book td {{
  overflow-wrap: anywhere;
}}
body.verbaloom-book pre,
body.verbaloom-book code {{
  white-space: pre-wrap;
}}
body.verbaloom-book .verbaloom-furniture {{
  display: none !important;
}}
body.verbaloom-book .verbaloom-merged-continuation {{
  display: none !important;
}}
body.verbaloom-book p {{
  margin: 0;
  text-indent: 1.25em;
}}
body.verbaloom-book h1,
body.verbaloom-book h2,
body.verbaloom-book h3,
body.verbaloom-book h4,
body.verbaloom-book h5,
body.verbaloom-book h6 {{
  margin: 2.5em 0 1.25em;
  text-align: center;
  text-indent: 0;
  page-break-after: avoid;
  break-after: avoid;
}}
body.verbaloom-book h1,
body.verbaloom-book h2 {{
  page-break-before: always;
  break-before: page;
}}
body.verbaloom-book h1 + p,
body.verbaloom-book h2 + p,
body.verbaloom-book h3 + p,
body.verbaloom-book .verbaloom-scene-break + p,
body.verbaloom-book .verbaloom-section-marker + p {{
  text-indent: 0;
}}
body.verbaloom-book img {{
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1em auto;
}}
body.verbaloom-book .verbaloom-cover-page {{
  margin: 0;
  padding: 0;
  width: 100%;
  max-width: 100%;
  overflow: hidden;
  text-align: center;
  text-indent: 0;
  page-break-after: always;
  break-after: page;
}}
body.verbaloom-book .verbaloom-cover-page img {{
  width: 100%;
  max-width: 100%;
  object-fit: contain;
  max-height: 95vh;
  margin: 0 auto;
}}
body.verbaloom-book .verbaloom-scene-break,
body.verbaloom-book .verbaloom-section-marker {{
  margin: 1.75em 0;
  text-align: center;
  text-indent: 0;
  page-break-after: avoid;
}}
body.verbaloom-book blockquote {{
  margin: 1em 8%;
}}
body.verbaloom-book table {{
  width: 100%;
  max-width: 100%;
  table-layout: auto;
  border-collapse: collapse;
  margin: 1em 0;
}}
body.verbaloom-book th,
body.verbaloom-book td {{
  padding: .35em .5em;
  vertical-align: top;
  text-align: left;
  word-spacing: normal;
  overflow-wrap: break-word;
  word-break: normal;
  hyphens: auto;
}}
body.verbaloom-book table p,
body.verbaloom-book table div {{
  margin: 0;
  text-align: left;
  text-indent: 0;
  word-spacing: normal;
  overflow-wrap: break-word;
  word-break: normal;
}}
@media (max-width: 42em) {{
  body.verbaloom-book table {{
    width: 100% !important;
    table-layout: fixed !important;
  }}
  body.verbaloom-book th,
  body.verbaloom-book td,
  body.verbaloom-book table p,
  body.verbaloom-book table div {{
    min-width: 0;
    white-space: normal !important;
    overflow-wrap: anywhere;
    word-break: break-word;
  }}
}}
""".strip() + "\n"

_SCENE_BREAK_RE = re.compile(r"^(?:[*.#~\-–—]\s*){3,}$")
_HEADING_SCENE_BREAK_RE = re.compile(r"^[*.#~\-–—](?:\s*[*.#~\-–—])*$")
_OCR_SCENE_BREAK_RE = re.compile(r"^[1Il|]\s*\.\s*\.$")
_SECTION_MARKER_RE = re.compile(
    r"^(?:\d{1,3}|[IVXLCDM]{1,12}|\d{4}|part(?:e)?\s+[\wIVXLCDM]+|"
    r"chapter\s+[\wIVXLCDM]+|cap[ií]tulo\s+[\wIVXLCDM]+)$",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(r"(?:[$€£]\s*\d+(?:[.,]\d{2})?|\d+(?:[.,]\d{2})?\s*[$€£])")
_ORDER_FURNITURE_RE = re.compile(
    r"\b(?:order|coupon|shipping|bookstore|credit card|visa|mastercard|"
    r"pedido|cup[oó]n|env[ií]o|librer[ií]a|tarjeta de cr[eé]dito)\b",
    re.IGNORECASE,
)
_PUBLISHER_FURNITURE_RE = re.compile(
    r"\b(?:isbn|ean|p\.\s*o\.\s*box|penguin\s+usa|dept\.?\s*#|"
    r"c[oó]digo\s+postal|postal\s+code)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGMENTS_RE = re.compile(
    r"^(?:acknowledg(?:e)?ments|agradecimientos|remerciements|danksagung|ringraziamenti)$",
    re.IGNORECASE,
)
_PROMOTIONAL_HEADING_RE = re.compile(
    r"\b(?:thrillers?|novels?|books?|bestsellers?|"
    r"novelas?|libros?|éxitos|apasionantes)\b",
    re.IGNORECASE,
)


@dataclass
class ProfessionalizationReport:
    cover_image_href: str = ""
    cover_declared: bool = False
    cover_page_declared: bool = False
    styled_documents: int = 0
    semantic_markers: int = 0
    furniture_markers: int = 0
    css_augmented: int = 0
    css_created: int = 0
    viewport_documents: int = 0
    obsolete_attributes_removed: int = 0
    flow_content_repairs: int = 0
    image_alt_repairs: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _qname(namespace: str, local: str) -> str:
    return f"{{{namespace}}}{local}"


def _local_name(element: etree._Element) -> str:
    return etree.QName(element).localname.lower()


def _add_class(element: etree._Element, class_name: str) -> bool:
    classes = [item for item in str(element.get("class") or "").split() if item]
    if class_name in classes:
        return False
    classes.append(class_name)
    element.set("class", " ".join(classes))
    return True


def _is_block_level_element(element: etree._Element) -> bool:
    return bool(
        isinstance(element.tag, str)
        and _local_name(element) in _BLOCK_LEVEL_ELEMENTS
    )


def count_invalid_blockquote_inline_runs(root: etree._Element) -> int:
    """Count direct inline-content runs that make EPUB blockquotes invalid.

    EPUB 2/3 validators require flow content inside ``blockquote``. Some source
    books place anchors or plain text directly below it. Counting contiguous
    runs lets the publication gate prove that a later wrapper is a bounded,
    source-derived repair instead of an arbitrary DOM change.
    """
    runs = 0
    for blockquote in root.xpath(
        "self::*[local-name()='blockquote'] | .//*[local-name()='blockquote']"
    ):
        in_inline_run = bool(str(blockquote.text or "").strip())
        runs += int(in_inline_run)
        for child in blockquote:
            if _is_block_level_element(child) or not isinstance(child.tag, str):
                in_inline_run = False
            elif not in_inline_run:
                runs += 1
                in_inline_run = True
            if str(child.tail or "").strip() and not in_inline_run:
                runs += 1
                in_inline_run = True
    return runs


def repair_invalid_blockquote_inline_runs(root: etree._Element) -> int:
    """Wrap invalid direct blockquote inline runs in neutral XHTML ``div``s.

    The transform is deterministic and idempotent. It preserves every inline
    element, attribute, text node and link target while adding only the flow
    container required by EPUBCheck.
    """
    repairs = 0

    def append_text(container: etree._Element, value: str) -> None:
        if not value:
            return
        if len(container):
            last = container[-1]
            last.tail = f"{last.tail or ''}{value}"
        else:
            container.text = f"{container.text or ''}{value}"

    for blockquote in root.xpath("//*[local-name()='blockquote']"):
        if not count_invalid_blockquote_inline_runs(blockquote):
            continue

        tokens: list[tuple[str, object]] = [("text", blockquote.text or "")]
        children = list(blockquote)
        for child in children:
            tokens.append(("element", child))
            tokens.append(("text", child.tail or ""))

        blockquote.text = None
        for child in children:
            child.tail = None
            blockquote.remove(child)

        wrapper: Optional[etree._Element] = None
        pending_whitespace = ""
        namespace = etree.QName(blockquote).namespace or XHTML_NS

        for kind, payload in tokens:
            if kind == "text":
                value = str(payload or "")
                if wrapper is not None:
                    append_text(wrapper, value)
                elif value.strip():
                    wrapper = etree.SubElement(
                        blockquote,
                        _qname(namespace, "div"),
                    )
                    wrapper.set("class", _BLOCKQUOTE_FLOW_REPAIR_CLASS)
                    repairs += 1
                    append_text(wrapper, pending_whitespace + value)
                    pending_whitespace = ""
                else:
                    pending_whitespace += value
                continue

            child = payload
            if not isinstance(child, etree._Element):
                continue
            if _is_block_level_element(child) or not isinstance(child.tag, str):
                if pending_whitespace:
                    if len(blockquote):
                        previous = blockquote[-1]
                        previous.tail = f"{previous.tail or ''}{pending_whitespace}"
                    else:
                        blockquote.text = f"{blockquote.text or ''}{pending_whitespace}"
                    pending_whitespace = ""
                blockquote.append(child)
                wrapper = None
                continue

            if wrapper is None:
                wrapper = etree.SubElement(
                    blockquote,
                    _qname(namespace, "div"),
                )
                wrapper.set("class", _BLOCKQUOTE_FLOW_REPAIR_CLASS)
                repairs += 1
                append_text(wrapper, pending_whitespace)
                pending_whitespace = ""
            wrapper.append(child)

        if pending_whitespace:
            if len(blockquote):
                previous = blockquote[-1]
                previous.tail = f"{previous.tail or ''}{pending_whitespace}"
            else:
                blockquote.text = f"{blockquote.text or ''}{pending_whitespace}"

    return repairs


def ensure_image_alt_attributes(root: etree._Element) -> int:
    """Supply required, source-aware ``alt`` attributes without inventing prose.

    A translated title or nearby figure caption is reused when available.
    Otherwise an empty alt marks the image as decorative, which is valid EPUB
    and more honest than synthesizing a description the source never supplied.
    """
    repairs = 0
    for image in root.xpath("//*[local-name()='img']"):
        if image.get("alt") is not None:
            continue
        alt_text = re.sub(r"\s+", " ", str(image.get("title") or "")).strip()
        if not alt_text:
            figures = image.xpath("ancestor::*[local-name()='figure'][1]")
            if figures:
                captions = figures[0].xpath(".//*[local-name()='figcaption'][1]")
                if captions:
                    alt_text = re.sub(
                        r"\s+",
                        " ",
                        " ".join(captions[0].itertext()),
                    ).strip()
        image.set("alt", alt_text)
        repairs += 1
    return repairs


def _normalized_identity(value: str) -> str:
    return re.sub(r"[^\w]+", " ", str(value or "").casefold(), flags=re.UNICODE).strip()


def _publication_identities(opf_root: etree._Element) -> set[str]:
    values = opf_root.xpath(
        "//*[local-name()='metadata']/*[local-name()='title' or local-name()='creator']/text()"
    )
    return {
        normalized
        for value in values
        if (normalized := _normalized_identity(str(value)))
    }


def _mark_running_furniture(body: etree._Element, identities: set[str]) -> int:
    """Hide source-proven page headers such as ``234 Author Name``."""
    if not identities:
        return 0
    marked = 0
    patterns = (
        re.compile(r"^\s*\d{1,5}\s+(.{3,100}?)\s*$"),
        re.compile(r"^\s*(.{3,100}?)\s+\d{1,5}\s*$"),
    )
    for element in body.xpath(".//*[local-name()='p' or local-name()='div']"):
        text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
        identity = ""
        for pattern in patterns:
            match = pattern.fullmatch(text)
            if match:
                identity = _normalized_identity(match.group(1))
                break
        if identity in identities:
            marked += int(_add_class(element, _FURNITURE_CLASS))
    return marked


def _normalize_structural_ocr_scene_break(
    element: etree._Element,
    text: str,
) -> bool:
    """Repair a decorative ellipsis misread as ``1. .`` in bulletless lists."""
    if _local_name(element) not in {"p", "div"}:
        return False
    if not _OCR_SCENE_BREAK_RE.fullmatch(text):
        return False
    if len(element) or not (element.text or "").strip():
        return False
    ancestors = list(element.iterancestors())
    list_depth = sum(_local_name(item) in {"ul", "ol", "li"} for item in ancestors)
    has_hidden_list_style = any(
        _local_name(item) in {"ul", "ol"}
        and re.search(r"list-style\s*:\s*none", str(item.get("style") or ""), re.I)
        for item in ancestors
    )
    if list_depth < 3 or not has_hidden_list_style:
        return False
    element.text = "* * *"
    return True


def _element_text(element: etree._Element) -> str:
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def _commercial_signal_score(text: str) -> int:
    prices = len(_PRICE_RE.findall(text))
    return (
        min(3, prices)
        + 2 * int(bool(_ORDER_FURNITURE_RE.search(text)))
        + int(bool(_PUBLISHER_FURNITURE_RE.search(text)))
    )


def _mark_commercial_furniture(body: etree._Element) -> int:
    """Hide source commercial inserts while preserving literary front/back matter."""
    children = list(body)
    marked = 0
    cover_index = next(
        (
            index
            for index, child in enumerate(children)
            if {_COVER_CLASS, _LEGACY_COVER_CLASS}
            & set(str(child.get("class") or "").split())
        ),
        None,
    )
    if cover_index is not None:
        title_index = next(
            (
                index
                for index in range(cover_index + 1, len(children))
                if _local_name(children[index]) in {"h1", "h2"}
            ),
            None,
        )
        if title_index is not None and title_index > cover_index + 1:
            between = " ".join(
                _element_text(child)
                for child in children[cover_index + 1:title_index]
            )
            if _commercial_signal_score(between) >= 4:
                for child in children[cover_index + 1:title_index]:
                    marked += int(_add_class(child, _FURNITURE_CLASS))

    acknowledgment_index = next(
        (
            index
            for index, child in enumerate(children)
            if _ACKNOWLEDGMENTS_RE.fullmatch(_element_text(child))
        ),
        None,
    )
    if acknowledgment_index is not None:
        tail_text = " ".join(
            _element_text(child) for child in children[acknowledgment_index + 1:]
        )
        if _commercial_signal_score(tail_text) >= 4:
            promo_index = next(
                (
                    index
                    for index in range(acknowledgment_index + 1, len(children))
                    if _PROMOTIONAL_HEADING_RE.search(_element_text(children[index]))
                    and len(_element_text(children[index])) <= 140
                ),
                None,
            )
            if promo_index is not None:
                for child in children[promo_index:]:
                    marked += int(_add_class(child, _FURNITURE_CLASS))
    return marked


def _ensure_mobile_viewport(root: etree._Element) -> bool:
    head = next(iter(root.xpath("//*[local-name()='head']")), None)
    if head is None:
        return False
    existing = head.xpath(
        "./*[local-name()='meta' and "
        "translate(@name, 'VIEWPORT', 'viewport')='viewport']"
    )
    if existing:
        return False
    meta = etree.SubElement(head, _qname(XHTML_NS, "meta"))
    meta.set("name", "viewport")
    meta.set("content", "width=device-width, initial-scale=1.0")
    return True


def _manifest_items(opf_root: etree._Element) -> list[etree._Element]:
    return opf_root.xpath("//*[local-name()='manifest']/*[local-name()='item']")


def _resolved_manifest_path(opf_dir: Path, href: str) -> Path:
    return (opf_dir / href).resolve()


def _path_identity(path: str | Path) -> str:
    """Return one stable key even when macOS aliases /var as /private/var."""
    return os.path.normcase(str(Path(path).resolve()))


def _existing_cover_id(opf_root: etree._Element) -> str:
    values = opf_root.xpath(
        "string(//*[local-name()='metadata']/*[local-name()='meta' and "
        "translate(@name, 'COVER', 'cover')='cover']/@content)"
    )
    if values:
        return str(values)
    for item in _manifest_items(opf_root):
        if "cover-image" in str(item.get("properties") or "").split():
            return str(item.get("id") or "")
    return ""


def _image_is_probable_cover(path: Path) -> bool:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        ratio = width / max(1, height)
        return width >= 250 and height >= 350 and 0.42 <= ratio <= 0.95
    except Exception:
        return False


def _image_reference(element: etree._Element) -> str:
    """Return a relative image reference from XHTML or inline SVG markup."""
    local_name = _local_name(element)
    if local_name == "img":
        return str(element.get("src") or "")
    if local_name == "image":
        return str(
            element.get("href")
            or element.get(_qname(XLINK_NS, "href"))
            or ""
        )
    return ""


def _remove_obsolete_epub_attributes(root: etree._Element) -> int:
    """Drop legacy no-op attributes that EPUB 3 rejects."""
    removed = 0
    for anchor in root.xpath("//*[local-name()='a' and @shape]"):
        del anchor.attrib["shape"]
        removed += 1
    for table in root.xpath(
        "//*[local-name()='table' and "
        "translate(normalize-space(@border), 'PX', 'px')='0']"
    ):
        del table.attrib["border"]
        removed += 1
    return removed


def _find_probable_cover(
    *,
    opf_root: etree._Element,
    opf_dir: Path,
    content_files: list[str],
    parsed_xhtml_docs: Dict[str, etree._Element],
) -> tuple[Optional[etree._Element], Optional[etree._Element], str]:
    if not content_files:
        return None, None, ""
    first_doc_path = _path_identity(opf_dir / content_files[0])
    docs = {
        _path_identity(path): root
        for path, root in parsed_xhtml_docs.items()
    }
    root = docs.get(first_doc_path)
    if root is None:
        return None, None, ""
    body = next(iter(root.xpath("//*[local-name()='body']")), None)
    if body is None:
        return None, None, ""

    manifest_by_path = {
        _resolved_manifest_path(opf_dir, str(item.get("href") or "")): item
        for item in _manifest_items(opf_root)
        if item.get("href")
    }
    doc_dir = (opf_dir / content_files[0]).parent
    visible_chars = 0
    for element in body.iter():
        image_reference = _image_reference(element)
        if image_reference:
            image_path = (doc_dir / image_reference).resolve()
            item = manifest_by_path.get(image_path)
            if item is not None and _image_is_probable_cover(image_path) and visible_chars <= 400:
                return item, element, content_files[0]
        visible_chars += len(str(element.text or "").strip())
        visible_chars += len(str(element.tail or "").strip())
        if visible_chars > 400:
            break
    return None, None, ""


def _find_cover_usage(
    *,
    cover_item: etree._Element,
    opf_dir: Path,
    content_files: list[str],
    parsed_xhtml_docs: Dict[str, etree._Element],
) -> tuple[Optional[etree._Element], str]:
    """Locate the XHTML element and page that render a manifest cover image."""
    cover_href = str(cover_item.get("href") or "")
    if not cover_href:
        return None, ""
    cover_path = _resolved_manifest_path(opf_dir, cover_href)
    docs = {
        _path_identity(path): root
        for path, root in parsed_xhtml_docs.items()
    }
    for content_href in content_files:
        doc_path = (opf_dir / content_href).resolve()
        root = docs.get(_path_identity(doc_path))
        if root is None:
            continue
        for image in root.xpath("//*[local-name()='img' or local-name()='image']"):
            image_reference = _image_reference(image)
            if not image_reference:
                continue
            image_path = (doc_path.parent / image_reference).resolve()
            if image_path == cover_path:
                return image, content_href
    return None, ""


def _declare_cover(
    *,
    opf_root: etree._Element,
    cover_item: etree._Element,
    cover_page_href: str,
    target_language: str,
) -> tuple[bool, bool]:
    metadata = next(iter(opf_root.xpath("//*[local-name()='metadata']")), None)
    cover_id = str(cover_item.get("id") or "")
    declared = False
    if metadata is not None and cover_id and not _existing_cover_id(opf_root):
        meta = etree.SubElement(metadata, _qname(OPF_NS, "meta"))
        meta.set("name", "cover")
        meta.set("content", cover_id)
        declared = True

    try:
        version = float(str(opf_root.get("version") or "2.0").split()[0])
    except ValueError:
        version = 2.0
    if version >= 3.0:
        properties = set(str(cover_item.get("properties") or "").split())
        if "cover-image" not in properties:
            properties.add("cover-image")
            cover_item.set("properties", " ".join(sorted(properties)))
            declared = True

    page_declared = False
    guide = next(iter(opf_root.xpath("//*[local-name()='guide']")), None)
    existing = (
        guide.xpath("./*[local-name()='reference' and @type='cover']")
        if guide is not None
        else []
    )
    if not existing and cover_page_href:
        if guide is None:
            guide = etree.SubElement(opf_root, _qname(OPF_NS, "guide"))
        ref = etree.SubElement(guide, _qname(OPF_NS, "reference"))
        ref.set("type", "cover")
        ref.set("title", "Portada" if str(target_language).casefold().startswith("span") else "Cover")
        ref.set("href", cover_page_href)
        page_declared = True
    return declared, page_declared


def _css_is_minimal(css: str) -> bool:
    compact = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL).strip()
    return len(compact) < 800 or compact.count("{") <= 6


def _ensure_stylesheet(
    *,
    opf_root: etree._Element,
    opf_dir: Path,
    parsed_xhtml_docs: Dict[str, etree._Element],
) -> tuple[int, int]:
    css_items = [
        item for item in _manifest_items(opf_root)
        if str(item.get("media-type") or "") == "text/css" and item.get("href")
    ]
    augmented = 0
    has_rich_css = False
    for item in css_items:
        path = opf_dir / str(item.get("href"))
        if not path.exists():
            has_rich_css = True
            continue
        css = path.read_text(encoding="utf-8", errors="replace")
        if VERBALOOM_CSS_MARKER in css or "TBL professional reading layer" in css:
            return 0, 0
        if _css_is_minimal(css):
            path.write_text(css.rstrip() + "\n\n" + _PROFESSIONAL_CSS, encoding="utf-8")
            augmented += 1
        else:
            has_rich_css = True
    if augmented and not has_rich_css:
        return augmented, 0

    # Rich publisher styles are publication content: never rewrite them. Add a
    # separate, namespaced reading layer after the source links so mobile table
    # layout and overflow rules can take effect without mutating the original.
    css_path = opf_dir / VERBALOOM_CSS_NAME
    css_path.write_text(_PROFESSIONAL_CSS, encoding="utf-8")
    manifest = next(iter(opf_root.xpath("//*[local-name()='manifest']")))
    existing_item = next(
        (
            item for item in _manifest_items(opf_root)
            if str(item.get("href") or "") == VERBALOOM_CSS_NAME
        ),
        None,
    )
    if existing_item is None:
        item = etree.SubElement(manifest, _qname(OPF_NS, "item"))
        item.set("id", "verbaloom-professional-css")
        item.set("href", VERBALOOM_CSS_NAME)
        item.set("media-type", "text/css")
    for doc_path, root in parsed_xhtml_docs.items():
        head = next(iter(root.xpath("//*[local-name()='head']")), None)
        if head is None:
            continue
        css_href = Path(os.path.relpath(css_path, Path(doc_path).parent)).as_posix()
        if head.xpath("./*[local-name()='link' and @href=$href]", href=css_href):
            continue
        link = etree.SubElement(head, _qname(XHTML_NS, "link"))
        link.set("rel", "stylesheet")
        link.set("type", "text/css")
        link.set("href", css_href)
    return augmented, int(existing_item is None)


def apply_professional_epub_layer(
    *,
    opf_tree: etree._ElementTree,
    opf_dir: str | Path,
    content_files: list[str],
    parsed_xhtml_docs: Dict[str, etree._Element],
    target_language: str,
) -> ProfessionalizationReport:
    """Apply conservative, idempotent publication semantics and styling."""
    report = ProfessionalizationReport()
    root = opf_tree.getroot()
    opf_dir_path = Path(opf_dir)
    publication_identities = _publication_identities(root)

    for doc in parsed_xhtml_docs.values():
        report.obsolete_attributes_removed += _remove_obsolete_epub_attributes(doc)
        report.flow_content_repairs += repair_invalid_blockquote_inline_runs(doc)
        report.image_alt_repairs += ensure_image_alt_attributes(doc)
        report.viewport_documents += int(_ensure_mobile_viewport(doc))
        body = next(iter(doc.xpath("//*[local-name()='body']")), None)
        if body is None:
            continue
        if _add_class(body, _BODY_CLASS):
            report.styled_documents += 1
        report.furniture_markers += _mark_running_furniture(
            body,
            publication_identities,
        )
        for element in body.xpath(
            ".//*[local-name()='p' or local-name()='div' or local-name()='h1' or "
            "local-name()='h2' or local-name()='h3']"
        ):
            text = re.sub(r"\s+", " ", " ".join(element.itertext())).strip()
            if _normalize_structural_ocr_scene_break(element, text):
                text = "* * *"
            elif (
                _local_name(element) in {"h3", "h4", "h5", "h6"}
                and text
                and _HEADING_SCENE_BREAK_RE.fullmatch(text)
            ):
                if not len(element):
                    element.text = "* * *"
                    text = "* * *"
            if text and _SCENE_BREAK_RE.fullmatch(text):
                report.semantic_markers += int(_add_class(element, _SCENE_BREAK_CLASS))
            elif text and len(text) <= 80 and _SECTION_MARKER_RE.fullmatch(text):
                report.semantic_markers += int(_add_class(element, _SECTION_MARKER_CLASS))

    cover_id = _existing_cover_id(root)
    cover_item = next(
        (item for item in _manifest_items(root) if str(item.get("id") or "") == cover_id),
        None,
    )
    cover_element = None
    cover_page_href = ""
    if cover_item is None:
        cover_item, cover_element, cover_page_href = _find_probable_cover(
            opf_root=root,
            opf_dir=opf_dir_path,
            content_files=content_files,
            parsed_xhtml_docs=parsed_xhtml_docs,
        )
    else:
        cover_element, cover_page_href = _find_cover_usage(
            cover_item=cover_item,
            opf_dir=opf_dir_path,
            content_files=content_files,
            parsed_xhtml_docs=parsed_xhtml_docs,
        )
    if cover_item is not None:
        report.cover_image_href = str(cover_item.get("href") or "")
        declared, page_declared = _declare_cover(
            opf_root=root,
            cover_item=cover_item,
            cover_page_href=cover_page_href,
            target_language=target_language,
        )
        report.cover_declared = declared or bool(_existing_cover_id(root))
        report.cover_page_declared = page_declared or bool(
            root.xpath("//*[local-name()='guide']/*[local-name()='reference' and @type='cover']")
        )
        if cover_element is not None:
            container = cover_element
            while container.getparent() is not None and _local_name(container.getparent()) != "body":
                container = container.getparent()
            _add_class(container, _COVER_CLASS)

    for doc in parsed_xhtml_docs.values():
        body = next(iter(doc.xpath("//*[local-name()='body']")), None)
        if body is not None:
            report.furniture_markers += _mark_commercial_furniture(body)

    report.css_augmented, report.css_created = _ensure_stylesheet(
        opf_root=root,
        opf_dir=opf_dir_path,
        parsed_xhtml_docs=parsed_xhtml_docs,
    )
    return report
