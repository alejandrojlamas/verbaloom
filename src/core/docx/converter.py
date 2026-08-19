"""
DOCX ↔ HTML conversion for translation.

Uses:
- mammoth: Conversion DOCX → HTML (semantic, clean)
- python-docx: Metadata extraction + DOCX reconstruction
"""

import base64
import binascii
import io
import os
import re
import tempfile
import mammoth
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from typing import Tuple, Dict, Any, Optional
from lxml import etree


_W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
_M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
_XML_NS = 'http://www.w3.org/XML/1998/namespace'

# Marker injected in place of OMML equations before mammoth runs.
# Mammoth drops <m:oMath> entirely; we replace each with this text marker so
# it survives DOCX -> HTML -> DOCX, then we re-inject the original OMML.
_EQ_MARKER_PREFIX = '__TTBLLEQ'
_EQ_MARKER_SUFFIX = 'EQTTBLL__'
_EQ_MARKER_REGEX = re.compile(
    re.escape(_EQ_MARKER_PREFIX) + r'(\d+)' + re.escape(_EQ_MARKER_SUFFIX)
)

# In the HTML returned by to_html(), each equation appears as a self-closing
# <eq id="N"/> tag. Because TagPreserver groups anything matching <[^>]+>
# into [idN] placeholders that the LLM treats as opaque tokens, the equation
# never reaches the LLM as raw text. from_html() converts these tags back to
# text markers before the HTML->DOCX rebuild so _restore_equations can splice
# the original OMML back in.
_EQ_TAG_REGEX = re.compile(r'<eq id="(\d+)"\s*/>')

_OMML_XPATH = etree.XPath(
    './/m:oMathPara | .//m:oMath[not(ancestor::m:oMathPara)]',
    namespaces={'m': _M_NS},
)
_WT_XPATH = etree.XPath('.//w:t', namespaces={'w': _W_NS})


class DocxHtmlConverter:
    """Converts DOCX to/from HTML for translation."""

    def to_html(self, docx_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Convert DOCX → HTML + metadata.

        Process:
        1. Use mammoth for clean HTML conversion
        2. Extract metadata with python-docx (styles, fonts, etc.)
        3. Return HTML + metadata dict

        Args:
            docx_path: Path to input DOCX file

        Returns:
            (html_content, metadata)
            - html_content: Semantic HTML (<p>, <strong>, <em>, etc.)
            - metadata: Dict with styles, fonts, page settings
        """
        # 1a. Pre-extract OMML equations into markers (mammoth drops them).
        patched_path, equations = self._replace_equations_with_markers(docx_path)

        try:
            # 1b. Conversion via mammoth (clean semantic HTML)
            with open(patched_path, 'rb') as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_content = result.value

                # Log warnings if any
                if result.messages:
                    warnings = [msg.message for msg in result.messages]
                    # Store warnings in metadata for potential debugging

            # 1c. Promote raw equation markers to HTML tags so TagPreserver
            # protects them as opaque [idN] placeholders. Without this the
            # LLM sees the literal text "__TTBLLEQ0EQTTBLL__" and may
            # mangle, translate or drop it.
            if equations:
                html_content = _EQ_MARKER_REGEX.sub(
                    r'<eq id="\1"/>', html_content
                )

            # 2. Extract metadata via python-docx (use original to keep page settings)
            doc = Document(docx_path)
            metadata = self._extract_metadata(doc)
            metadata['equations'] = equations
            # Preserve original inline image dimensions in document order;
            # mammoth strips them from the HTML, so we restore at rebuild time.
            metadata['image_dimensions'] = [
                (shape.width, shape.height) for shape in doc.inline_shapes
            ]

            return html_content, metadata
        finally:
            if patched_path != docx_path and os.path.exists(patched_path):
                try:
                    os.remove(patched_path)
                except OSError:
                    pass

    def from_html(
        self,
        html_content: str,
        metadata: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Reconstruct DOCX from translated HTML + metadata.

        Process:
        1. Parse HTML with lxml
        2. Create empty Document() with python-docx
        3. For each HTML element, create DOCX paragraph/run
        4. Apply styles from metadata
        5. Save DOCX

        Args:
            html_content: Translated HTML content
            metadata: Original document metadata (styles, fonts, etc.)
            output_path: Path to output DOCX file
        """
        # Reverse the equation-tag substitution done in to_html so the rest
        # of the rebuild pipeline (which expects raw text markers) is unchanged.
        html_content = _EQ_TAG_REGEX.sub(
            lambda m: f'{_EQ_MARKER_PREFIX}{m.group(1)}{_EQ_MARKER_SUFFIX}',
            html_content,
        )

        # Parse HTML
        html_tree = etree.HTML(html_content)

        # Create DOCX document
        doc = Document()

        # Apply page metadata (page size, margins, etc.)
        self._apply_page_metadata(doc, metadata)

        # Queue of (width, height) pairs in document order, consumed by
        # _add_image_run as it embeds each <img>. mammoth drops sizing info
        # from the HTML, so we replay it from the original DOCX.
        self._image_dims_queue = list(metadata.get('image_dimensions', []))

        # Convert HTML → DOCX paragraphs
        if html_tree is not None:
            body = html_tree.find('.//body')
            if body is not None:
                for element in body:
                    self._convert_html_element_to_docx(doc, element, metadata)

        # Stamp core properties for cross-platform rendering diagnostics
        try:
            from src.utils.text_encoding import derive_identifier_suffix
            doc.core_properties.last_modified_by = (
                f"TranslateBookWithLLM {derive_identifier_suffix()}"
            )
        except Exception:
            pass

        # Save
        doc.save(output_path)

        # Restore OMML equations (replace text markers with original OMML XML)
        equations = metadata.get('equations') if metadata else None
        if equations:
            self._restore_equations(output_path, equations)

    def _replace_equations_with_markers(
        self, docx_path: str
    ) -> Tuple[str, Dict[int, str]]:
        """
        Pre-process a DOCX so OMML equations survive the mammoth round-trip.

        Mammoth drops <m:oMath> / <m:oMathPara> entirely. We extract each
        outermost equation, store its XML, and replace it with a <w:r><w:t>
        containing a unique text marker. Mammoth then carries the marker
        through into the HTML output, where it can later be matched by
        _restore_equations on the rebuilt DOCX.

        Returns:
            (path_to_use, equations_dict)
            - path_to_use: original path if no equations, else a temp DOCX
            - equations_dict: {idx: serialized_omml_xml}
        """
        doc = Document(docx_path)
        body = doc.element.body

        omml_elements = _OMML_XPATH(body)

        if not omml_elements:
            return docx_path, {}

        equations: Dict[int, str] = {}
        for idx, eq in enumerate(omml_elements):
            equations[idx] = etree.tostring(eq, encoding='unicode')

            new_r = etree.Element(f'{{{_W_NS}}}r')
            new_t = etree.SubElement(new_r, f'{{{_W_NS}}}t')
            new_t.text = f'{_EQ_MARKER_PREFIX}{idx}{_EQ_MARKER_SUFFIX}'
            new_t.set(f'{{{_XML_NS}}}space', 'preserve')

            parent = eq.getparent()
            eq_idx = list(parent).index(eq)
            parent.remove(eq)
            parent.insert(eq_idx, new_r)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.docx')
        os.close(tmp_fd)
        doc.save(tmp_path)
        return tmp_path, equations

    def _restore_equations(
        self, docx_path: str, equations: Dict[int, str]
    ) -> None:
        """
        Replace text markers in the saved DOCX with the original OMML XML.

        A marker may sit alone in a <w:t>, or share its run with surrounding
        text (mammoth often consolidates adjacent runs). In the second case
        we split the run into [text-prefix, OMML, text-suffix] within the
        same paragraph.
        """
        doc = Document(docx_path)
        body = doc.element.body

        for t_elem in list(_WT_XPATH(body)):
            text = t_elem.text
            if not text or _EQ_MARKER_PREFIX not in text:
                continue

            matches = list(_EQ_MARKER_REGEX.finditer(text))
            if not matches:
                continue

            r_elem = t_elem.getparent()
            if r_elem is None:
                continue
            p_elem = r_elem.getparent()
            if p_elem is None:
                continue

            r_idx = list(p_elem).index(r_elem)

            new_elements = []
            cursor = 0
            for m in matches:
                before = text[cursor:m.start()]
                if before:
                    rb = etree.Element(f'{{{_W_NS}}}r')
                    tb = etree.SubElement(rb, f'{{{_W_NS}}}t')
                    tb.text = before
                    tb.set(f'{{{_XML_NS}}}space', 'preserve')
                    new_elements.append(rb)

                eq_idx = int(m.group(1))
                omml_xml = equations.get(eq_idx)
                if omml_xml:
                    try:
                        new_elements.append(etree.fromstring(omml_xml))
                    except etree.XMLSyntaxError:
                        pass
                cursor = m.end()

            after = text[cursor:]
            if after:
                ra = etree.Element(f'{{{_W_NS}}}r')
                ta = etree.SubElement(ra, f'{{{_W_NS}}}t')
                ta.text = after
                ta.set(f'{{{_XML_NS}}}space', 'preserve')
                new_elements.append(ra)

            p_elem.remove(r_elem)
            for offset, elem in enumerate(new_elements):
                p_elem.insert(r_idx + offset, elem)

        doc.save(docx_path)

    def _extract_metadata(self, doc: Document) -> Dict[str, Any]:
        """
        Extract styles, fonts, page settings from DOCX.

        Args:
            doc: python-docx Document instance

        Returns:
            Dict with document metadata
        """
        metadata = {
            'styles': {},
            'default_font': None,
            'page_size': None,
            'margins': None,
        }

        # Extract page settings from first section
        if doc.sections:
            section = doc.sections[0]
            metadata['page_size'] = {
                'width': section.page_width.inches if section.page_width else None,
                'height': section.page_height.inches if section.page_height else None
            }
            metadata['margins'] = {
                'top': section.top_margin.inches if section.top_margin else None,
                'bottom': section.bottom_margin.inches if section.bottom_margin else None,
                'left': section.left_margin.inches if section.left_margin else None,
                'right': section.right_margin.inches if section.right_margin else None
            }

        # Extract default font if available
        # Note: python-docx doesn't provide easy access to default font,
        # so we'll use a common default
        metadata['default_font'] = {
            'name': 'Calibri',
            'size': 11
        }

        return metadata

    def _apply_page_metadata(self, doc: Document, metadata: Dict[str, Any]) -> None:
        """
        Apply page settings to document.

        Args:
            doc: python-docx Document instance
            metadata: Document metadata
        """
        if not doc.sections:
            return

        section = doc.sections[0]

        # Apply page size
        page_size = metadata.get('page_size', {})
        if page_size.get('width') is not None:
            section.page_width = Inches(page_size['width'])
        if page_size.get('height') is not None:
            section.page_height = Inches(page_size['height'])

        # Apply margins
        margins = metadata.get('margins', {})
        if margins.get('top') is not None:
            section.top_margin = Inches(margins['top'])
        if margins.get('bottom') is not None:
            section.bottom_margin = Inches(margins['bottom'])
        if margins.get('left') is not None:
            section.left_margin = Inches(margins['left'])
        if margins.get('right') is not None:
            section.right_margin = Inches(margins['right'])

    def _convert_html_element_to_docx(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """
        Convert an HTML element to appropriate DOCX element.

        Args:
            doc: python-docx Document instance
            element: lxml HTML element
            metadata: Document metadata for styling
        """
        tag = element.tag

        if tag == 'p':
            self._convert_paragraph(doc, element, metadata)
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self._convert_heading(doc, element, tag, metadata)
        elif tag in ['ul', 'ol']:
            self._convert_list(doc, element, metadata)
        elif tag == 'table':
            self._convert_table(doc, element, metadata)
        elif tag == 'br':
            doc.add_paragraph()  # Empty paragraph for line break
        elif tag == 'img':
            self._add_image_run(doc.add_paragraph(), element)
        # Skip other tags (div, span handled within paragraphs)

    def _convert_paragraph(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """Convert HTML <p> to DOCX paragraph."""
        p = doc.add_paragraph()
        self._add_runs_from_element(p, element, metadata)

    def _convert_heading(
        self,
        doc: Document,
        element: etree._Element,
        tag: str,
        metadata: Dict[str, Any]
    ):
        """Convert HTML heading to DOCX heading."""
        level = int(tag[1])  # h1 → 1, h2 → 2, etc.
        text = self._get_text_content(element)
        doc.add_heading(text, level=level)

    def _convert_list(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """Convert HTML list to DOCX list."""
        is_ordered = element.tag == 'ol'

        for li in element.findall('.//li'):
            text = self._get_text_content(li)
            p = doc.add_paragraph(text, style='List Number' if is_ordered else 'List Bullet')

    def _convert_table(
        self,
        doc: Document,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """Convert HTML table to DOCX table."""
        rows = element.findall('.//tr')
        if not rows:
            return

        # Count columns from first row
        first_row = rows[0]
        cols = len(first_row.findall('.//td')) + len(first_row.findall('.//th'))

        if cols == 0:
            return

        # Create table
        table = doc.add_table(rows=len(rows), cols=cols)
        table.style = 'Table Grid'

        # Fill cells
        for row_idx, tr in enumerate(rows):
            cells = tr.findall('.//td') + tr.findall('.//th')
            for col_idx, cell in enumerate(cells):
                if col_idx < cols:
                    text = self._get_text_content(cell)
                    table.rows[row_idx].cells[col_idx].text = text

    def _add_runs_from_element(
        self,
        paragraph,
        element: etree._Element,
        metadata: Dict[str, Any]
    ):
        """
        Add runs to paragraph from HTML element, preserving inline formatting.

        Handles <strong>, <em>, <b>, <i>, etc.
        """
        # Handle direct text
        if element.text:
            paragraph.add_run(element.text)

        # Handle child elements
        for child in element:
            if child.tag == 'strong' or child.tag == 'b':
                text = self._get_text_content(child)
                run = paragraph.add_run(text)
                run.bold = True
            elif child.tag == 'em' or child.tag == 'i':
                text = self._get_text_content(child)
                run = paragraph.add_run(text)
                run.italic = True
            elif child.tag == 'u':
                text = self._get_text_content(child)
                run = paragraph.add_run(text)
                run.underline = True
            elif child.tag == 'img':
                self._add_image_run(paragraph, child)
            else:
                # For other tags, just extract text
                text = self._get_text_content(child)
                paragraph.add_run(text)

            # Handle tail text (text after closing tag)
            if child.tail:
                paragraph.add_run(child.tail)

    def _add_image_run(self, paragraph, img_element: etree._Element) -> None:
        """
        Embed a base64 data-URI <img> into the paragraph as an inline picture.

        Mammoth emits images as `<img src="data:image/<fmt>;base64,...">`.
        Without this, images would be silently dropped on DOCX reconstruction.
        Non-data-URI sources (http://, file://) are skipped — they cannot be
        reliably resolved at translation time.
        """
        src = img_element.get('src', '')
        if not src.startswith('data:'):
            return
        try:
            header, payload = src.split(',', 1)
        except ValueError:
            return
        if ';base64' not in header:
            return
        try:
            image_bytes = base64.b64decode(payload)
        except (binascii.Error, ValueError):
            return

        width = height = None
        queue = getattr(self, '_image_dims_queue', None)
        if queue:
            width, height = queue.pop(0)

        try:
            paragraph.add_run().add_picture(
                io.BytesIO(image_bytes), width=width, height=height
            )
        except Exception:
            # python-docx raises UnrecognizedImageError for unsupported formats
            return

    def _get_text_content(self, element: etree._Element) -> str:
        """Extract all text content from an element and its children."""
        return ''.join(element.itertext())
