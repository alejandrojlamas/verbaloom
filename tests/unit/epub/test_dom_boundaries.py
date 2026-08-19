from __future__ import annotations

from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

import pytest
from lxml import etree

from src.core.epub.dom_boundaries import (
    apply_epub_missing_text_replacements,
    audit_epub_dom_boundaries,
    compare_xhtml_boundaries,
    find_epub_missing_text_blocks,
    repair_epub_dom_boundaries,
)
from src.core.epub.paragraph_reflow import (
    REFLOW_ANCHOR_CLASS,
    REFLOW_CONTINUATION_CLASS,
)


def _root(body: str):
    return etree.fromstring(
        f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{body}</body></html>'.encode()
    )


@pytest.mark.parametrize(
    ("source", "broken", "expected"),
    [
        ("<p>Terminó. <em>Comienza otra</em></p>", "<p>Terminó.<em>Comienza otra</em></p>", "Terminó. Comienza otra"),
        ("<p>una <em>palabra</em> importante</p>", "<p>una<em>palabra</em> importante</p>", "una palabra importante"),
        ("<p>Sr.<em> García</em></p>", "<p>Sr.<em>García</em></p>", "Sr. García"),
        ("<p><em>uno</em> <i>dos</i></p>", "<p><em>uno</em><i>dos</i></p>", "uno dos"),
        ("<p>Hola <span></span>mundo</p>", "<p>Hola<span></span>mundo</p>", "Hola mundo"),
        ("<p><span><em>voz</em></span> <strong>alta</strong></p>", "<p><span><em>voz</em></span><strong>alta</strong></p>", "voz alta"),
        ("<p>texto <a href='#n'>nota</a> final</p>", "<p>texto<a href='#n'>nota</a> final</p>", "texto nota final"),
        ("<p>H<sub>2</sub> O y x<sup>2</sup></p>", "<p>H<sub>2</sub>O y x<sup>2</sup></p>", "H2 O y x2"),
        ("<p>«Hola». <em>Después</em></p>", "<p>«Hola».<em>Después</em></p>", "«Hola». Después"),
        ("<p>Primero — <strong>después</strong></p>", "<p>Primero —<strong>después</strong></p>", "Primero — después"),
    ],
)
def test_restores_only_source_proven_inline_whitespace(source, broken, expected):
    source_root = _root(source)
    output_root = _root(broken)

    first = compare_xhtml_boundaries(source_root, output_root, repair=True)
    second = compare_xhtml_boundaries(source_root, output_root, repair=True)

    visible = "".join(output_root.xpath("//*[local-name()='p']")[0].itertext())
    assert visible == expected
    assert first.repaired_boundaries >= 1
    assert second.repaired_boundaries == 0
    assert second.findings == []


@pytest.mark.parametrize(
    "body",
    [
        "<p>micro<em>organismo</em></p>",
        "<p>3.<em>1416</em></p>",
        "<p>https://example.com/a.<em>b</em></p>",
        "<p>etc.<em>(incluido el humo)</em></p>",
        "<p>(<em>inciso</em>)</p>",
        "<p>pre<strong>existente</strong></p>",
    ],
)
def test_does_not_insert_space_when_source_boundary_has_none(body):
    source_root = _root(body)
    output_root = _root(body)
    before = etree.tostring(output_root)

    report = compare_xhtml_boundaries(source_root, output_root, repair=True)

    assert report.repaired_boundaries == 0
    assert etree.tostring(output_root) == before


def test_partial_inline_slot_reflow_is_not_a_structural_mismatch():
    source_root = _root("<p>Primero <em>segundo</em> tercero</p>")
    output_root = _root("<p>Primero segundo tercero<em></em></p>")

    report = compare_xhtml_boundaries(source_root, output_root, repair=False)

    assert report.structural_mismatches == []


def test_entire_visible_block_emptiness_is_a_structural_mismatch():
    source_root = _root("<p>Contenido que no puede desaparecer.</p>")
    output_root = _root("<p></p>")

    report = compare_xhtml_boundaries(source_root, output_root, repair=False)

    assert report.structural_mismatches
    assert "non-empty source block became empty" in report.structural_mismatches[0]


def test_marked_source_artifact_can_be_hidden_without_changing_dom_shape():
    source_root = _root(
        '<p><a href="https://example.com">www.example.com</a></p>'
    )
    output_root = _root(
        '<p class="verbaloom-sanitized-artifact" style="display: none"><a></a></p>'
    )

    report = compare_xhtml_boundaries(source_root, output_root, repair=False)

    assert report.structural_mismatches == []
    assert report.findings == []


def test_source_proven_marked_paragraph_reflow_is_not_a_structural_mismatch():
    source_root = _root(
        "<p>The account continued throughout the entire afternoon and described everyone who remained</p>"
        "<p>waiting by the harbor until the boats returned through the fog.</p>"
    )
    output_root = _root(
        f'<p class="{REFLOW_ANCHOR_CLASS}">El relato continuó durante toda la tarde y describió a quienes seguían esperando junto al puerto hasta que volvieron los barcos.</p>'
        f'<p class="{REFLOW_CONTINUATION_CLASS}"></p>'
    )

    report = compare_xhtml_boundaries(source_root, output_root, repair=False)

    assert report.structural_mismatches == []


def test_unmarked_or_falsely_marked_empty_paragraph_remains_a_mismatch():
    valid_source = _root(
        "<p>The account continued throughout the entire afternoon and described everyone who remained</p>"
        "<p>waiting by the harbor until the boats returned through the fog.</p>"
    )
    unmarked_output = _root("<p>El relato se conserva.</p><p></p>")
    terminal_source = _root(
        "<p>This complete source paragraph ends with a full stop.</p>"
        "<p>another paragraph begins independently.</p>"
    )
    forged_output = _root(
        f'<p class="{REFLOW_ANCHOR_CLASS}">Texto conservado.</p>'
        f'<p class="{REFLOW_CONTINUATION_CLASS}"></p>'
    )

    unmarked = compare_xhtml_boundaries(valid_source, unmarked_output, repair=False)
    forged = compare_xhtml_boundaries(terminal_source, forged_output, repair=False)

    assert unmarked.structural_mismatches
    assert forged.structural_mismatches


@pytest.mark.parametrize("source_text", ["4", "&J^", "* * *"])
def test_page_and_ocr_furniture_may_be_removed(source_text):
    source_root = _root(f"<p>{escape(source_text)}</p>")
    output_root = _root("<p></p>")

    report = compare_xhtml_boundaries(source_root, output_root, repair=False)

    assert report.structural_mismatches == []


def test_removes_source_unproven_leading_comma_from_table_cell():
    source_root = _root("<table><tr><td><p>Kapitän</p></td></tr></table>")
    output_root = _root("<table><tr><td><p>, capitán</p></td></tr></table>")

    first = compare_xhtml_boundaries(source_root, output_root, repair=True)
    second = compare_xhtml_boundaries(source_root, output_root, repair=True)

    cell = output_root.xpath("//*[local-name()='td']")[0]
    assert " ".join("".join(cell.itertext()).split()) == "capitán"
    assert first.repaired_boundaries == 1
    assert second.repaired_boundaries == 0


def test_preserves_leading_table_punctuation_when_source_has_it():
    source_root = _root("<table><tr><td>, continuation</td></tr></table>")
    output_root = _root("<table><tr><td>, continuación</td></tr></table>")
    before = etree.tostring(output_root)

    report = compare_xhtml_boundaries(source_root, output_root, repair=True)

    assert report.repaired_boundaries == 0
    assert etree.tostring(output_root) == before


def _write_epub(path: Path, paragraph: str):
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f'{paragraph}</body></html>'
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("Text/chapter.xhtml", chapter)
        archive.writestr("Images/preserved.bin", b"unchanged-resource")


def test_epub_repair_is_atomic_idempotent_and_preserves_binary_resources(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(source, "<p>Terminó. <em>Comienza</em></p>")
    _write_epub(output, "<p>Terminó.<em>Comienza</em></p>")

    first = repair_epub_dom_boundaries(source, output)
    second = repair_epub_dom_boundaries(source, output)
    audit = audit_epub_dom_boundaries(source, output)

    assert first.repaired_boundaries == 1
    assert second.repaired_boundaries == 0
    assert audit.clean
    with zipfile.ZipFile(output) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("Images/preserved.bin") == b"unchanged-resource"
        assert "Terminó. <em>Comienza" in archive.read("Text/chapter.xhtml").decode()


def test_finds_and_repairs_only_meaningful_empty_epub_blocks(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(source, "<p><em>I cold.</em></p><p>4</p>")
    _write_epub(output, "<p><em></em></p><p></p>")

    findings = find_epub_missing_text_blocks(source, output)

    assert [(item.file_href, item.block_index, item.source_text) for item in findings] == [
        ("Text/chapter.xhtml", 0, "I cold.")
    ]
    changed = apply_epub_missing_text_replacements(
        source,
        output,
        {(findings[0].file_href, findings[0].block_index): "Tenía frío."},
    )

    assert changed == 1
    assert audit_epub_dom_boundaries(source, output).structural_mismatches == []
    with zipfile.ZipFile(output) as archive:
        chapter = archive.read("Text/chapter.xhtml").decode()
        assert "<em>Tenía frío.</em>" in chapter
        assert archive.read("Images/preserved.bin") == b"unchanged-resource"
