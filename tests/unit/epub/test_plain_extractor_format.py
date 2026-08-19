"""Round-trip tests for the EPUB plain extractor (Plain Text Mode).

Covers the historical failure modes: tables/figures dropped entirely and
inline tags (<strong>/<em>/<a>) flattened to plain text.
"""
import pytest

from lxml import etree

from src.core.epub.plain_extractor import (
    extract_plain_paragraphs,
    replace_body_with_paragraphs,
)

XHTML = "http://www.w3.org/1999/xhtml"


def _body(xhtml: str):
    root = etree.fromstring(xhtml.encode())
    return root.find(f".//{{{XHTML}}}body")


RICH = f"""<html xmlns="{XHTML}"><body>
<h1>Chapter <em>one</em></h1>
<p>Text with <strong>bold</strong>, <em>italic</em> and <a href="https://n.mx/ref">a note</a>.</p>
<div class="wrap">
  <p>Inside a div with <b>classic b</b>.</p>
  <table>
    <thead><tr><th>Col A</th><th>Col B</th></tr></thead>
    <tbody>
      <tr><td>cell <em>alpha</em></td><td>cell beta</td></tr>
      <tr><td></td><td><strong>gamma</strong></td></tr>
    </tbody>
  </table>
</div>
<figure><img src="img/photo.png" alt="x"/><figcaption>A caption</figcaption></figure>
<p>Closing.</p>
</body></html>"""


class TestExtraction:
    def test_inline_tags_become_markdown(self):
        texts, tags, _, _ = extract_plain_paragraphs(_body(RICH))
        assert texts[0] == "Chapter *one*"
        assert "**bold**" in texts[1]
        assert "*italic*" in texts[1]
        assert "[a note](https://n.mx/ref)" in texts[1]
        assert "**classic b**" in texts[2]

    def test_table_cells_extracted_with_geometry(self):
        texts, tags, _, tables = extract_plain_paragraphs(_body(RICH))
        assert len(tables) == 1
        spec = next(iter(tables.values()))
        assert spec.rows == 3 and spec.cols == 2
        # Empty cell encoded as -1 in the grid, no block emitted for it.
        assert spec.grid[2][0] == -1
        header_flags = dict(zip(spec.cell_blocks, spec.cell_is_header))
        headers = [texts[i] for i, is_h in header_flags.items() if is_h]
        assert sorted(headers) == ["Col A", "Col B"]
        assert tags.count("table_cell") == len(spec.cell_blocks)

    def test_figure_contents_preserved(self):
        texts, tags, images, _ = extract_plain_paragraphs(_body(RICH))
        assert any("A caption" in t for t in texts)
        all_srcs = [
            img.get("src") for imgs in images.values() for img in imgs
        ]
        assert "img/photo.png" in all_srcs

    def test_internal_anchor_links_stay_plain(self):
        body = _body(
            f'<html xmlns="{XHTML}"><body><p>see <a href="#fn1">note 1</a></p></body></html>'
        )
        texts, _, _, _ = extract_plain_paragraphs(body)
        assert texts[0] == "see note 1"

    def test_internal_epub_note_links_stay_plain(self):
        body = _body(
            f'<html xmlns="{XHTML}"><body><p>ver <a href="../Text/notas.xhtml#nt23">[23]</a></p></body></html>'
        )
        texts, _, _, _ = extract_plain_paragraphs(body)
        assert texts[0] == "ver [23]"

    def test_deep_container_markup_does_not_recurse(self):
        inner = "<p>Deep paragraph survives.</p>"
        for _ in range(1200):
            inner = f"<div>{inner}</div>"
        parser = etree.XMLParser(huge_tree=True, recover=True)
        root = etree.fromstring(f'<html xmlns="{XHTML}"><body>{inner}</body></html>'.encode(), parser)
        body = root.find(f".//{{{XHTML}}}body")

        texts, tags, _, _ = extract_plain_paragraphs(body)

        assert texts == ["Deep paragraph survives."]
        assert tags == ["p"]

    def test_deep_inline_markup_does_not_recurse(self):
        inner = "Deep inline survives."
        for _ in range(1200):
            inner = f"<span>{inner}</span>"
        parser = etree.XMLParser(huge_tree=True, recover=True)
        root = etree.fromstring(f'<html xmlns="{XHTML}"><body><p>{inner}</p></body></html>'.encode(), parser)
        body = root.find(f".//{{{XHTML}}}body")

        texts, _, _, _ = extract_plain_paragraphs(body)

        assert texts == ["Deep inline survives."]

    def test_deep_table_markup_does_not_recurse(self):
        inner = "<tr><td>Deep cell survives.</td></tr>"
        for _ in range(1200):
            inner = f"<tbody>{inner}</tbody>"
        parser = etree.XMLParser(huge_tree=True, recover=True)
        root = etree.fromstring(f'<html xmlns="{XHTML}"><body><table>{inner}</table></body></html>'.encode(), parser)
        body = root.find(f".//{{{XHTML}}}body")

        texts, tags, _, tables = extract_plain_paragraphs(body)

        assert texts == ["Deep cell survives."]
        assert tags == ["table_cell"]
        assert len(tables) == 1


class TestRebuild:
    def test_round_trip_rebuilds_table_and_inline_tags(self):
        body = _body(RICH)
        texts, tags, images, tables = extract_plain_paragraphs(body)
        translated = [t.replace("cell", "celda") for t in texts]
        replace_body_with_paragraphs(
            body, translated, tags, images, table_specs=tables
        )
        out = etree.tostring(body, encoding="unicode")

        assert "<table>" in out and out.count("<tr>") == 3
        assert "<th>Col A</th>" in out
        assert "celda <em>alpha</em>" in out
        assert "<strong>gamma</strong>" in out
        assert '<a href="https://n.mx/ref">a note</a>' in out
        assert '<img src="img/photo.png"' in out
        # Closing paragraph after the table.
        assert out.rindex("Closing.") > out.rindex("</table>")

    def test_table_cell_blocks_not_duplicated_as_paragraphs(self):
        body = _body(RICH)
        texts, tags, images, tables = extract_plain_paragraphs(body)
        replace_body_with_paragraphs(body, texts, tags, images, table_specs=tables)
        out = etree.tostring(body, encoding="unicode")
        # "Col A" must appear exactly once (inside the table, not echoed as <p>).
        assert out.count("Col A") == 1

    def test_rebuild_without_table_specs_is_backward_safe(self):
        body = _body(f'<html xmlns="{XHTML}"><body><p>solo</p></body></html>')
        texts, tags, images, _ = extract_plain_paragraphs(body)
        replace_body_with_paragraphs(body, texts, tags, images)
        out = etree.tostring(body, encoding="unicode")
        assert "<p>solo</p>" in out
