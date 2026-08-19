"""
Plain-text extraction and rebuild for Plain Text Mode (DOCX).

Reads the document body in true document order via python-docx, collecting:
- one block per paragraph, with inline formatting (bold/italic/hyperlinks)
  encoded as lightweight markdown the LLM preserves naturally
- one block per unique table cell, plus a geometry spec so the table can be
  rebuilt with the same shape (merged cells map to a single block)
- the structural style of each block ('heading{n}', 'list', 'normal',
  'quote', 'table_cell')
- inline images anchored to their parent block index (preserved as bytes
  with their original dimensions)
- page metadata (size + margins) for the rebuilt document

At rebuild time, a fresh Document() is created with the same page setup,
each translated block is added with the right style, markdown is decoded
back into formatted runs, tables are re-emitted with their geometry, and
anchored images follow their block.

Historical note: earlier versions iterated only ``doc.paragraphs`` (which
silently DROPS every table) and flattened all runs (destroying bold, italic
and links). Both were the main causes of "format not preserved" reports.
"""
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from docx import Document
from docx.shared import Inches
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.common.inline_markdown import (
    InlineSegment,
    segments_to_markdown,
    parse_inline_markdown,
)


# --- Data model ---------------------------------------------------------------


@dataclass
class _ImageRef:
    """One embedded image extracted from a paragraph."""
    blob: bytes
    width_emu: Optional[int] = None
    height_emu: Optional[int] = None


@dataclass
class _TableSpec:
    """Geometry of one source table, in block-index space."""
    rows: int
    cols: int
    # grid[r][c] = block index of that cell's content, or -1 for an empty cell.
    # Merged cells repeat the same block index across their span.
    grid: List[List[int]] = field(default_factory=list)
    # Unique cell block indices in emission order (for skip-ahead at rebuild).
    cell_blocks: List[int] = field(default_factory=list)


@dataclass
class DocxPlainContent:
    """Everything we need to rebuild a plain-text-mode DOCX."""
    paragraphs_text: List[str] = field(default_factory=list)
    # 'heading1'..'heading6', 'list', 'normal', 'quote', 'table_cell'
    paragraphs_style: List[str] = field(default_factory=list)
    images_by_paragraph: Dict[int, List[_ImageRef]] = field(default_factory=dict)
    # First block index of each table -> geometry spec
    tables: Dict[int, _TableSpec] = field(default_factory=dict)
    page_size: Optional[Dict[str, float]] = None
    margins: Optional[Dict[str, float]] = None

    def table_cell_indices(self) -> set:
        cells: set = set()
        for spec in self.tables.values():
            cells.update(spec.cell_blocks)
        return cells


# --- Extraction helpers --------------------------------------------------------


def _classify_paragraph_style(paragraph: Paragraph) -> str:
    """Map python-docx style name to one of our coarse buckets."""
    style_name = (paragraph.style.name if paragraph.style else "") or ""
    name = style_name.lower()
    if name.startswith("heading "):
        try:
            level = int(name.split(" ")[1])
            level = max(1, min(level, 6))
            return f"heading{level}"
        except (ValueError, IndexError):
            return "heading1"
    if name.startswith("title"):
        return "heading1"
    if "list" in name or "bullet" in name or "number" in name:
        return "list"
    if "quote" in name:
        return "quote"
    return "normal"


def _extract_inline_images(paragraph: Paragraph, doc: Document) -> List[_ImageRef]:
    """
    Collect inline images embedded in a paragraph, in document order.

    python-docx represents inline images as <w:drawing> elements containing a
    <a:blip r:embed="rId"> that references a relationship to the actual image
    part. We resolve those relationships to grab the raw bytes.
    """
    images: List[_ImageRef] = []
    rels = doc.part.rels

    blips = paragraph._element.findall(
        ".//" + qn("w:drawing") + "//" + qn("a:blip")
    )
    if not blips:
        blips = paragraph._element.findall(".//" + qn("a:blip"))

    for blip in blips:
        embed_id = blip.get(qn("r:embed"))
        if not embed_id or embed_id not in rels:
            continue
        rel = rels[embed_id]
        try:
            blob = rel.target_part.blob
        except AttributeError:
            continue
        width = height = None
        node = blip.getparent()
        while node is not None:
            extent = node.find(qn("wp:extent")) if hasattr(node, "find") else None
            if extent is not None:
                try:
                    width = int(extent.get("cx"))
                    height = int(extent.get("cy"))
                except (TypeError, ValueError):
                    width = height = None
                break
            node = node.getparent()
        images.append(_ImageRef(blob=blob, width_emu=width, height_emu=height))
    return images


def _run_segments(run_element, doc: Document, href: Optional[str]) -> List[InlineSegment]:
    """Convert one <w:r> element into inline segments (text only)."""
    texts: List[str] = []
    for child in run_element.iterchildren():
        tag = child.tag
        if tag == qn("w:t"):
            texts.append(child.text or "")
        elif tag in (qn("w:br"), qn("w:cr")):
            texts.append(" ")
        elif tag == qn("w:tab"):
            texts.append(" ")
    text = "".join(texts)
    if not text:
        return []

    rpr = run_element.find(qn("w:rPr"))
    bold = _bool_prop(rpr, "w:b")
    italic = _bool_prop(rpr, "w:i")
    return [InlineSegment(text=text, bold=bold, italic=italic, href=href)]


def _bool_prop(rpr, prop: str) -> bool:
    """Read a boolean run property (<w:b/>, <w:i/>), honoring w:val."""
    if rpr is None:
        return False
    node = rpr.find(qn(prop))
    if node is None:
        return False
    val = node.get(qn("w:val"))
    if val is None:
        return True
    return str(val).strip().lower() not in {"0", "false", "none", "off"}


def _resolve_hyperlink_url(hyperlink_element, doc: Document) -> Optional[str]:
    rel_id = hyperlink_element.get(qn("r:id"))
    if not rel_id:
        return None
    rels = doc.part.rels
    if rel_id not in rels:
        return None
    try:
        return rels[rel_id].target_ref
    except AttributeError:
        return None


def _paragraph_to_markdown(paragraph: Paragraph, doc: Document) -> str:
    """Encode a paragraph's runs (incl. hyperlinks) as inline markdown."""
    segments: List[InlineSegment] = []
    for child in paragraph._element.iterchildren():
        tag = child.tag
        if tag == qn("w:r"):
            segments.extend(_run_segments(child, doc, href=None))
        elif tag == qn("w:hyperlink"):
            url = _resolve_hyperlink_url(child, doc)
            for run in child.findall(qn("w:r")):
                segments.extend(_run_segments(run, doc, href=url))
    markdown = segments_to_markdown(segments)
    return " ".join(markdown.split())


def _iter_body_blocks(doc: Document):
    """Yield (kind, object) for each top-level body element in document order."""
    body = doc.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield "paragraph", Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield "table", Table(child, doc)


def _extract_table(table: Table, doc: Document, content: DocxPlainContent) -> None:
    """Emit one block per unique cell and record the table geometry."""
    rows = len(table.rows)
    cols = len(table.columns) if rows else 0
    if rows == 0 or cols == 0:
        return

    spec = _TableSpec(rows=rows, cols=cols)
    seen_tc: Dict[int, int] = {}  # id(tc element) -> block index (or -1)

    for r in range(rows):
        grid_row: List[int] = []
        row_cells = table.rows[r].cells
        for c in range(cols):
            try:
                cell = row_cells[c]
            except IndexError:
                grid_row.append(-1)
                continue
            tc_id = id(cell._tc)
            if tc_id in seen_tc:
                grid_row.append(seen_tc[tc_id])
                continue

            parts: List[str] = []
            for cell_par in cell.paragraphs:
                md = _paragraph_to_markdown(cell_par, doc)
                if md:
                    parts.append(md)
            cell_text = "\n".join(parts)

            if not cell_text.strip():
                seen_tc[tc_id] = -1
                grid_row.append(-1)
                continue

            block_index = len(content.paragraphs_text)
            content.paragraphs_text.append(cell_text)
            content.paragraphs_style.append("table_cell")
            spec.cell_blocks.append(block_index)
            seen_tc[tc_id] = block_index
            grid_row.append(block_index)
        spec.grid.append(grid_row)

    if spec.cell_blocks:
        content.tables[spec.cell_blocks[0]] = spec


def extract_plain_paragraphs(docx_path: str) -> DocxPlainContent:
    """
    Read a DOCX file as plain blocks (no HTML conversion), in document order.

    Paragraph blocks carry inline-markdown formatting; tables contribute one
    block per non-empty unique cell plus a geometry spec. Empty paragraphs
    are skipped, but any inline image they carry is anchored to the
    preceding block (or to a synthetic empty block if none exists yet).
    """
    doc = Document(docx_path)
    content = DocxPlainContent()

    if doc.sections:
        section = doc.sections[0]
        content.page_size = {
            "width": section.page_width.inches if section.page_width else None,
            "height": section.page_height.inches if section.page_height else None,
        }
        content.margins = {
            "top": section.top_margin.inches if section.top_margin else None,
            "bottom": section.bottom_margin.inches if section.bottom_margin else None,
            "left": section.left_margin.inches if section.left_margin else None,
            "right": section.right_margin.inches if section.right_margin else None,
        }

    for kind, obj in _iter_body_blocks(doc):
        if kind == "table":
            _extract_table(obj, doc, content)
            continue

        paragraph = obj
        markdown = _paragraph_to_markdown(paragraph, doc)
        images = _extract_inline_images(paragraph, doc)

        if not markdown and not images:
            continue

        if not markdown and images:
            if content.paragraphs_text:
                anchor = len(content.paragraphs_text) - 1
                content.images_by_paragraph.setdefault(anchor, []).extend(images)
            else:
                content.paragraphs_text.append("")
                content.paragraphs_style.append("normal")
                content.images_by_paragraph[0] = images
            continue

        idx = len(content.paragraphs_text)
        content.paragraphs_text.append(markdown)
        content.paragraphs_style.append(_classify_paragraph_style(paragraph))
        if images:
            content.images_by_paragraph[idx] = images

    return content


# --- Rebuild -------------------------------------------------------------------


def _apply_page_metadata(doc: Document, content: DocxPlainContent) -> None:
    if not doc.sections:
        return
    section = doc.sections[0]
    ps = content.page_size or {}
    if ps.get("width") is not None:
        section.page_width = Inches(ps["width"])
    if ps.get("height") is not None:
        section.page_height = Inches(ps["height"])
    m = content.margins or {}
    if m.get("top") is not None:
        section.top_margin = Inches(m["top"])
    if m.get("bottom") is not None:
        section.bottom_margin = Inches(m["bottom"])
    if m.get("left") is not None:
        section.left_margin = Inches(m["left"])
    if m.get("right") is not None:
        section.right_margin = Inches(m["right"])


def _write_markdown_runs(paragraph, text: str) -> None:
    """Decode inline markdown into formatted runs on an existing paragraph."""
    for seg in parse_inline_markdown(text):
        if not seg.text:
            continue
        run = paragraph.add_run(seg.text)
        if seg.bold:
            run.bold = True
        if seg.italic:
            run.italic = True
        if seg.href:
            # python-docx has no public hyperlink API; render the URL after
            # the link text so the information survives in a readable form.
            url_run = paragraph.add_run(f" ({seg.href})")
            url_run.italic = True


def _add_styled_paragraph(doc: Document, text: str, style: str):
    """Add a paragraph with the right python-docx style + markdown runs."""
    if style.startswith("heading"):
        try:
            level = int(style.replace("heading", ""))
            level = max(1, min(level, 6))
            paragraph = doc.add_heading("", level=level)
        except ValueError:
            paragraph = doc.add_paragraph()
    elif style == "list":
        try:
            paragraph = doc.add_paragraph(style="List Bullet")
        except KeyError:
            paragraph = doc.add_paragraph()
    elif style == "quote":
        try:
            paragraph = doc.add_paragraph(style="Intense Quote")
        except KeyError:
            paragraph = doc.add_paragraph()
    else:
        paragraph = doc.add_paragraph()
    _write_markdown_runs(paragraph, text)
    return paragraph


def _add_image(doc: Document, image: _ImageRef) -> None:
    """Insert an image as its own paragraph (no inline-positioning)."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    try:
        if image.width_emu and image.height_emu:
            width_in = image.width_emu / 914400.0 if image.width_emu else None
            height_in = image.height_emu / 914400.0 if image.height_emu else None
            run.add_picture(
                io.BytesIO(image.blob),
                width=Inches(width_in) if width_in else None,
                height=Inches(height_in) if height_in else None,
            )
        else:
            run.add_picture(io.BytesIO(image.blob))
    except Exception:
        # Unsupported format or corrupted blob - drop silently rather than
        # blocking the whole rebuild.
        return


def _emit_table(
    doc: Document,
    spec: _TableSpec,
    translated_paragraphs: List[str],
) -> None:
    """Rebuild one table from its geometry + translated cell blocks."""
    table = doc.add_table(rows=spec.rows, cols=spec.cols)
    try:
        table.style = "Table Grid"
    except KeyError:
        pass

    written: set = set()
    for r in range(spec.rows):
        for c in range(spec.cols):
            try:
                block_index = spec.grid[r][c]
            except IndexError:
                continue
            if block_index < 0 or block_index in written:
                continue
            written.add(block_index)
            text = (
                translated_paragraphs[block_index]
                if block_index < len(translated_paragraphs)
                else ""
            ) or ""
            cell = table.cell(r, c)
            lines = [ln for ln in text.split("\n")]
            first = True
            for line in lines:
                paragraph = cell.paragraphs[0] if first else cell.add_paragraph()
                first = False
                _write_markdown_runs(paragraph, line)


def build_minimal_docx(
    translated_paragraphs: List[str],
    content: DocxPlainContent,
    output_path: str,
    bilingual: bool = False,
) -> None:
    """
    Generate a fresh DOCX from translated blocks.

    Args:
        translated_paragraphs: parallel to content.paragraphs_text (same length)
        content: the result of extract_plain_paragraphs() on the source DOCX
        output_path: where to save
        bilingual: when True, emit source text before each translated paragraph
    """
    doc = Document()
    _apply_page_metadata(doc, content)

    table_cells = content.table_cell_indices()
    count = len(translated_paragraphs)
    i = 0
    while i < count:
        if i in content.tables:
            _emit_table(doc, content.tables[i], translated_paragraphs)
            spec = content.tables[i]
            consumed = set(spec.cell_blocks)
            # Skip past every block this table consumed (cells are contiguous).
            while i < count and i in consumed:
                i += 1
            continue

        if i in table_cells:
            # Orphan safety: a cell block whose table spec start was elsewhere.
            i += 1
            continue

        text = (translated_paragraphs[i] or "").strip()
        style = (
            content.paragraphs_style[i]
            if i < len(content.paragraphs_style)
            else "normal"
        )

        if bilingual and i < len(content.paragraphs_text):
            source_text = (content.paragraphs_text[i] or "").strip()
            if source_text:
                _add_styled_paragraph(doc, source_text, style)

        if text:
            _add_styled_paragraph(doc, text, style)

        for image in content.images_by_paragraph.get(i, []):
            _add_image(doc, image)

        i += 1

    try:
        from src.utils.text_encoding import derive_identifier_suffix
        doc.core_properties.last_modified_by = (
            f"VerbaLoom {derive_identifier_suffix()}"
        )
    except Exception:
        pass

    doc.save(output_path)
