"""
Plain-text extraction and rebuild for Plain Text Mode (EPUB).

Walks an XHTML <body> in DOM order, collecting block-level paragraphs as
inline-markdown strings (bold/italic/links encoded as **/*/[](url) so the
LLM preserves them), anchoring any <img> to its parent block index, and
emitting one block per non-empty table cell with a geometry spec so tables
can be rebuilt with the same shape.

At rebuild time, the body is wiped and reconstructed as a flat sequence of
block elements (<p>, <h1..h6>, <li>, <blockquote>, <pre>), tables are
re-emitted as <table>/<tr>/<td> from their geometry, markdown decodes back
to <strong>/<em>/<a> inline tags, and after each block that originally
contained images an extra <p class="plain-text-images"> wrapper carries the
original <img> elements unchanged.

Historical note: earlier versions DROPPED tables and figures entirely (real
content loss, not just formatting) and flattened every inline tag. Both were
core causes of "format not preserved" reports for EPUB.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from lxml import etree

from src.common.inline_markdown import (
    InlineSegment,
    segments_to_markdown,
    parse_inline_markdown,
)


# Block-level tags we preserve at rebuild time (li flattens to p later — see replace_body_with_paragraphs).
BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre")
# Containers we descend into looking for blocks
CONTAINER_TAGS = (
    "div", "section", "article", "main", "header", "footer", "aside", "nav",
    "figure", "picture",
)
# Subtrees never sent to the LLM in Plain Text Mode (no recoverable text)
DROP_TAGS = ("svg", "video", "audio", "iframe", "form", "script", "style")
# List wrappers we descend into (the inner <li> items become individual blocks)
LIST_WRAPPER_TAGS = ("ul", "ol")
# Table wrappers (cells become blocks, geometry recorded separately)
TABLE_TAGS = ("table",)
# Inline tags encoded as markdown emphasis
_BOLD_TAGS = ("strong", "b")
_ITALIC_TAGS = ("em", "i")

XHTML_NS = "http://www.w3.org/1999/xhtml"


@dataclass
class TableSpec:
    """Geometry of one source table, in block-index space."""
    rows: int
    cols: int
    # grid[r][c] = block index for that cell, or -1 for an empty cell.
    grid: List[List[int]] = field(default_factory=list)
    cell_blocks: List[int] = field(default_factory=list)
    # Parallel to cell_blocks: True when the source cell was a <th>.
    cell_is_header: List[bool] = field(default_factory=list)


def _local_name(elem: etree._Element) -> str:
    """Return the lowercase local tag name, stripping XHTML namespace."""
    tag = elem.tag
    if isinstance(tag, str) and tag.startswith("{"):
        tag = tag.split("}", 1)[1]
    return tag.lower() if isinstance(tag, str) else ""


def _preserve_href_as_markdown(href: str) -> bool:
    """Only external links survive text-first EPUB reconstruction safely."""
    href = (href or "").strip()
    if not href or " " in href:
        return False
    return href.startswith(("http://", "https://", "mailto:", "tel:"))


def _extract_segments(
    elem: etree._Element,
    image_sink: List[etree._Element],
    *,
    bold: bool = False,
    italic: bool = False,
    href: Optional[str] = None,
) -> List[InlineSegment]:
    """
    Flatten an element's content into formatted inline segments.

    <strong>/<b> → bold, <em>/<i> → italic, <a href> → link; other inline
    tags contribute their text without formatting. Any <img> encountered is
    cloned into image_sink (DOM order preserved).
    """
    segments: List[InlineSegment] = []

    def emit(text: Optional[str], b: bool, i: bool, h: Optional[str]):
        if text:
            segments.append(InlineSegment(text=text, bold=b, italic=i, href=h))

    def walk(node: etree._Element, b: bool, i: bool, h: Optional[str], include_tail: bool):
        """Walk inline descendants iteratively so deep EPUB markup cannot recurse."""
        stack = [("node", node, b, i, h, include_tail)]
        while stack:
            event, current, cb, ci, ch, c_include_tail = stack.pop()
            if event == "tail":
                if c_include_tail and current.tail:
                    emit(current.tail, cb, ci, ch)
                continue

            name = _local_name(current)
            if name in DROP_TAGS or name in TABLE_TAGS:
                if c_include_tail and current.tail:
                    emit(current.tail, cb, ci, ch)
                continue
            if name == "img":
                image_sink.append(_clone_img(current))
                if c_include_tail and current.tail:
                    emit(current.tail, cb, ci, ch)
                continue
            if name == "br":
                emit(" ", cb, ci, ch)
                if c_include_tail and current.tail:
                    emit(current.tail, cb, ci, ch)
                continue

            nb = cb or name in _BOLD_TAGS
            ni = ci or name in _ITALIC_TAGS
            nh = ch
            if name == "a":
                link = current.get("href") or ""
                # Internal EPUB anchors/files are not stable after text-first
                # reconstruction; keep their visible note/title text plain.
                if _preserve_href_as_markdown(link):
                    nh = link

            emit(current.text, nb, ni, nh)
            if c_include_tail:
                stack.append(("tail", current, cb, ci, ch, c_include_tail))
            for child in reversed(list(current)):
                stack.append(("node", child, nb, ni, nh, True))

    emit(elem.text, bold, italic, href)
    for child in elem:
        walk(child, bold, italic, href, include_tail=True)

    return segments


def _segments_to_normalized_markdown(segments: List[InlineSegment]) -> str:
    markdown = segments_to_markdown(segments)
    return " ".join(markdown.split())


def _extract_text_keep_inline(
    elem: etree._Element, image_sink: List[etree._Element]
) -> str:
    """Markdown-encoded textual content of an element (whitespace-normalized)."""
    return _segments_to_normalized_markdown(_extract_segments(elem, image_sink))


def _clone_img(img: etree._Element) -> etree._Element:
    """Create a standalone copy of an <img> with its attributes, no namespace."""
    new = etree.Element("img")
    for k, v in img.attrib.items():
        if isinstance(k, str) and k.startswith("{"):
            k = k.split("}", 1)[1]
        new.set(k, v)
    return new


def _extract_table(
    table_elem: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    table_specs: Dict[int, TableSpec],
) -> None:
    """Emit one block per non-empty cell and record the table geometry."""
    rows: List[List[etree._Element]] = []

    stack = list(reversed(list(table_elem)))
    while stack:
        child = stack.pop()
        name = _local_name(child)
        if name == "tr":
            cells = [
                c for c in child
                if _local_name(c) in ("td", "th")
            ]
            if cells:
                rows.append(cells)
            continue
        if name in ("thead", "tbody", "tfoot"):
            for nested in reversed(list(child)):
                stack.append(nested)

    if not rows:
        return

    cols = max(len(r) for r in rows)
    spec = TableSpec(rows=len(rows), cols=cols)

    for row_cells in rows:
        grid_row: List[int] = []
        for c in range(cols):
            if c >= len(row_cells):
                grid_row.append(-1)
                continue
            cell = row_cells[c]
            images: List[etree._Element] = []
            text = _extract_text_keep_inline(cell, images)
            if not text.strip() and not images:
                grid_row.append(-1)
                continue

            block_index = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append("table_cell")
            if images:
                images_by_paragraph[block_index] = images
            spec.cell_blocks.append(block_index)
            spec.cell_is_header.append(_local_name(cell) == "th")
            grid_row.append(block_index)
        spec.grid.append(grid_row)

    if spec.cell_blocks:
        table_specs[spec.cell_blocks[0]] = spec


def _collect_blocks(
    root: etree._Element,
    paragraphs_text: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    table_specs: Dict[int, TableSpec],
) -> None:
    """
    DOM-walk a container, emitting one entry per block-level element found.

    For lists, we descend into <li> items individually (each is its own block).
    For containers (div, section, figure, ...), we descend iteratively.
    For tables, each non-empty cell becomes a block plus a geometry record.
    """
    stack = list(reversed(list(root)))
    while stack:
        child = stack.pop()
        name = _local_name(child)

        if name in DROP_TAGS:
            continue

        if name in TABLE_TAGS:
            _extract_table(
                child, paragraphs_text, paragraphs_tag,
                images_by_paragraph, table_specs,
            )
            continue

        if name in LIST_WRAPPER_TAGS:
            for nested in reversed(list(child)):
                stack.append(nested)
            continue

        if name in CONTAINER_TAGS:
            for nested in reversed(list(child)):
                stack.append(nested)
            continue

        if name in BLOCK_TAGS:
            images: List[etree._Element] = []
            if name == "pre":
                # Preserve code/pre verbatim — no markdown encoding inside.
                text = "".join(child.itertext())
            else:
                text = _extract_text_keep_inline(child, images)

            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append(name)
            if images:
                images_by_paragraph[idx] = images
            continue

        if name == "img":
            # Standalone <img> at body level — anchor to the previous block,
            # or create a synthetic anchor if it's first.
            img_copy = _clone_img(child)
            if paragraphs_text:
                anchor = len(paragraphs_text) - 1
                images_by_paragraph.setdefault(anchor, []).append(img_copy)
            else:
                paragraphs_text.append("")
                paragraphs_tag.append("p")
                images_by_paragraph[0] = [img_copy]
            continue

        # Anything else: try to extract textual content as a generic paragraph
        images: List[etree._Element] = []
        text = _extract_text_keep_inline(child, images)
        if text.strip() or images:
            idx = len(paragraphs_text)
            paragraphs_text.append(text)
            paragraphs_tag.append("p")
            if images:
                images_by_paragraph[idx] = images


def extract_plain_paragraphs(
    body_element: etree._Element,
) -> Tuple[List[str], List[str], Dict[int, List[etree._Element]], Dict[int, TableSpec]]:
    """
    Extract the body as a flat list of (text, tag) pairs plus image/table maps.

    Args:
        body_element: <body> element from a parsed XHTML doc.

    Returns:
        paragraphs_text:        list of inline-markdown strings, one per block
        paragraphs_tag:         parallel list of tag names ("p", "h1", "li",
                                "table_cell", ...)
        images_by_paragraph:    {paragraph_index: [<img> elements]}
        table_specs:            {first cell block index: TableSpec}
    """
    paragraphs_text: List[str] = []
    paragraphs_tag: List[str] = []
    images_by_paragraph: Dict[int, List[etree._Element]] = {}
    table_specs: Dict[int, TableSpec] = {}

    if body_element is None:
        return paragraphs_text, paragraphs_tag, images_by_paragraph, table_specs

    _collect_blocks(body_element, paragraphs_text, paragraphs_tag,
                    images_by_paragraph, table_specs)
    return paragraphs_text, paragraphs_tag, images_by_paragraph, table_specs


# --- Rebuild -------------------------------------------------------------------


def _write_inline_markdown(block: etree._Element, text: str) -> None:
    """Decode inline markdown into <strong>/<em>/<a> children of a block."""
    segments = parse_inline_markdown(text or "")
    last_child: Optional[etree._Element] = None

    def append_text(value: str):
        nonlocal last_child
        if not value:
            return
        if last_child is None:
            block.text = (block.text or "") + value
        else:
            last_child.tail = (last_child.tail or "") + value

    for seg in segments:
        if not seg.text:
            continue
        if not seg.has_formatting:
            append_text(seg.text)
            continue

        outer: Optional[etree._Element] = None
        inner: Optional[etree._Element] = None
        if seg.href:
            outer = etree.SubElement(block, "a")
            outer.set("href", seg.href)
            inner = outer
        if seg.bold:
            parent = inner if inner is not None else block
            node = etree.SubElement(parent, "strong")
            if outer is None:
                outer = node
            inner = node
        if seg.italic:
            parent = inner if inner is not None else block
            node = etree.SubElement(parent, "em")
            if outer is None:
                outer = node
            inner = node
        inner.text = seg.text
        last_child = outer


def _emit_table(
    parent: etree._Element,
    spec: TableSpec,
    translated_paragraphs: List[str],
) -> None:
    """Rebuild one <table> from its geometry + translated cell blocks."""
    table = etree.SubElement(parent, "table")
    header_blocks = {
        block for block, is_header in zip(spec.cell_blocks, spec.cell_is_header)
        if is_header
    }
    for r in range(spec.rows):
        tr = etree.SubElement(table, "tr")
        row = spec.grid[r] if r < len(spec.grid) else []
        for c in range(spec.cols):
            block_index = row[c] if c < len(row) else -1
            tag = "th" if block_index in header_blocks else "td"
            cell = etree.SubElement(tr, tag)
            if block_index < 0:
                continue
            text = (
                translated_paragraphs[block_index]
                if block_index < len(translated_paragraphs)
                else ""
            ) or ""
            _write_inline_markdown(cell, text.strip())


def replace_body_with_paragraphs(
    body_element: etree._Element,
    translated_paragraphs: List[str],
    paragraphs_tag: List[str],
    images_by_paragraph: Dict[int, List[etree._Element]],
    bilingual: bool = False,
    source_paragraphs: List[str] = None,
    table_specs: Optional[Dict[int, TableSpec]] = None,
) -> None:
    """
    Wipe body_element and refill it from the translated paragraphs.

    Args:
        body_element: target <body> to overwrite
        translated_paragraphs: same length as paragraphs_tag
        paragraphs_tag: tag name per paragraph ("p", "h1", "li", "table_cell", ...)
        images_by_paragraph: anchored images per paragraph index
        bilingual: when True, emit a <p class="src"> with the source text
                   right before each translated block.
        source_paragraphs: required when bilingual is True
        table_specs: {first cell block index: TableSpec} from extraction
    """
    table_specs = table_specs or {}
    table_cell_indices: set = set()
    for spec in table_specs.values():
        table_cell_indices.update(spec.cell_blocks)

    # Clear body
    body_element.text = None
    for child in list(body_element):
        body_element.remove(child)

    count = len(translated_paragraphs)
    i = 0
    while i < count:
        if i in table_specs:
            spec = table_specs[i]
            _emit_table(body_element, spec, translated_paragraphs)
            consumed = set(spec.cell_blocks)
            # Emit any images anchored inside the table's cells right after it.
            for cell_index in spec.cell_blocks:
                if cell_index in images_by_paragraph and images_by_paragraph[cell_index]:
                    img_wrapper = etree.SubElement(body_element, "p")
                    img_wrapper.set("class", "plain-text-images")
                    for img in images_by_paragraph[cell_index]:
                        img_wrapper.append(img)
            while i < count and i in consumed:
                i += 1
            continue

        if i in table_cell_indices:
            # Orphan safety: cell block whose table start was elsewhere.
            i += 1
            continue

        text = (translated_paragraphs[i] or "").strip()
        raw_tag = paragraphs_tag[i] if i < len(paragraphs_tag) else "p"
        # <li> outside <ul>/<ol> is not valid XHTML — flatten to <p> in Plain Text Mode.
        tag = "p" if raw_tag in ("li", "table_cell") else raw_tag

        # Bilingual: emit source first when we have it
        if bilingual and source_paragraphs and i < len(source_paragraphs):
            source_text = (source_paragraphs[i] or "").strip()
            if source_text:
                src_block = etree.SubElement(body_element, tag)
                src_block.set("class", "plain-text-source")
                _write_inline_markdown(src_block, source_text)

        # Emit translated block when there is text
        if text:
            block = etree.SubElement(body_element, tag)
            if bilingual:
                block.set("class", "plain-text-target")
            if tag == "pre":
                block.text = text
            else:
                _write_inline_markdown(block, text)

        # Emit anchored images right after
        if i in images_by_paragraph and images_by_paragraph[i]:
            img_wrapper = etree.SubElement(body_element, "p")
            img_wrapper.set("class", "plain-text-images")
            for img in images_by_paragraph[i]:
                img_wrapper.append(img)

        i += 1
