from __future__ import annotations

from pathlib import Path

from lxml import etree
from PIL import Image

from src.core.epub.professionalize import (
    VERBALOOM_CSS_MARKER,
    apply_professional_epub_layer,
)


def _package(*, include_css: bool = True) -> etree._ElementTree:
    css_item = '<item id="css" href="book.css" media-type="text/css"/>' if include_css else ""
    return etree.ElementTree(etree.fromstring(f"""
    <package xmlns="http://www.idpf.org/2007/opf" version="2.0">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Book</dc:title></metadata>
      <manifest>
        <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
        <item id="image-one" href="images/first.jpg" media-type="image/jpeg"/>
        {css_item}
      </manifest>
      <spine><itemref idref="chapter"/></spine>
    </package>
    """.encode()))


def _document() -> etree._Element:
    return etree.fromstring(b"""
    <html xmlns="http://www.w3.org/1999/xhtml"><head></head><body>
      <div><img src="images/first.jpg" alt=""/></div>
      <h1>Book title</h1><p>Opening paragraph.</p><p>* * *</p><p>Next scene.</p>
    </body></html>
    """)


def test_declares_existing_probable_cover_and_augments_only_minimal_css(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    css_path = tmp_path / "book.css"
    css_path.write_text("img { max-width: 100%; }", encoding="utf-8")
    opf = _package()
    doc = _document()

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.cover_declared is True
    assert report.cover_page_declared is True
    assert opf.xpath("string(//*[local-name()='meta' and @name='cover']/@content)") == "image-one"
    assert opf.xpath("string(//*[local-name()='reference' and @type='cover']/@href)") == "chapter.xhtml"
    assert "verbaloom-book" in doc.xpath("string(//*[local-name()='body']/@class)")
    assert "verbaloom-cover-page" in doc.xpath("string(//*[local-name()='img']/../@class)")
    assert "verbaloom-scene-break" in doc.xpath("string(//*[local-name()='p'][2]/@class)")
    generated_css = css_path.read_text(encoding="utf-8")
    assert VERBALOOM_CSS_MARKER in generated_css
    assert "overflow-wrap: anywhere" in generated_css
    assert "overflow-x: hidden" in generated_css
    assert "width: 90%" in generated_css
    assert "object-fit: contain" in generated_css
    assert ".verbaloom-furniture" in generated_css
    assert report.viewport_documents == 1
    assert doc.xpath(
        "string(//*[local-name()='meta' and @name='viewport']/@content)"
    ) == "width=device-width, initial-scale=1.0"


def test_mobile_viewport_is_idempotent(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    (tmp_path / "book.css").write_text("img { max-width: 100%; }", encoding="utf-8")
    opf = _package()
    doc = _document()

    first = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )
    second = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert first.viewport_documents == 1
    assert second.viewport_documents == 0
    assert len(doc.xpath("//*[local-name()='meta' and @name='viewport']")) == 1


def test_preserves_rich_publisher_css_byte_for_byte(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    rich = "\n".join(f".rule-{index} {{ margin: {index}px; color: #111; }}" for index in range(30))
    css_path = tmp_path / "book.css"
    css_path.write_text(rich, encoding="utf-8")

    opf = _package()
    doc = _document()
    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.css_augmented == 0
    assert report.css_created == 1
    assert css_path.read_text(encoding="utf-8") == rich
    assert (tmp_path / "verbaloom-professional.css").exists()
    assert opf.xpath(
        "string(//*[local-name()='item' and @id='verbaloom-professional-css']/@href)"
    ) == "verbaloom-professional.css"
    assert doc.xpath(
        "string(//*[local-name()='head']/*[local-name()='link'][@href='verbaloom-professional.css']/@href)"
    ) == "verbaloom-professional.css"
    reading_css = (tmp_path / "verbaloom-professional.css").read_text(encoding="utf-8")
    assert "body.verbaloom-book table p" in reading_css
    assert "word-spacing: normal" in reading_css
    assert "@media (max-width: 42em)" in reading_css
    assert "table-layout: fixed" in reading_css


def test_creates_namespaced_stylesheet_when_source_has_none(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package(include_css=False)
    doc = _document()

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.css_created == 1
    assert (tmp_path / "verbaloom-professional.css").exists()
    assert opf.xpath("string(//*[local-name()='item' and @id='verbaloom-professional-css']/@href)") == "verbaloom-professional.css"
    assert doc.xpath("string(//*[local-name()='head']/*[local-name()='link']/@href)") == "verbaloom-professional.css"


def test_created_stylesheet_uses_relative_href_for_nested_chapter(tmp_path: Path):
    (tmp_path / "images").mkdir()
    (tmp_path / "Text").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package(include_css=False)
    chapter_item = opf.xpath("//*[local-name()='item' and @id='chapter']")[0]
    chapter_item.set("href", "Text/chapter.xhtml")
    doc = _document()
    doc.xpath("//*[local-name()='img']")[0].set("src", "../images/first.jpg")

    apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["Text/chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "Text" / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert doc.xpath("string(//*[local-name()='head']/*[local-name()='link']/@href)") == "../verbaloom-professional.css"


def test_existing_cover_declaration_gets_missing_guide_and_cover_class(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package()
    metadata = opf.xpath("//*[local-name()='metadata']")[0]
    meta = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
    meta.set("name", "cover")
    meta.set("content", "image-one")
    doc = _document()

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.cover_declared is True
    assert report.cover_page_declared is True
    assert opf.xpath("string(//*[local-name()='reference' and @type='cover']/@href)") == "chapter.xhtml"
    assert "verbaloom-cover-page" in doc.xpath("string(//*[local-name()='img']/../@class)")


def test_existing_svg_cover_gets_missing_guide_and_cover_class(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package()
    metadata = opf.xpath("//*[local-name()='metadata']")[0]
    meta = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
    meta.set("name", "cover")
    meta.set("content", "image-one")
    doc = etree.fromstring(b"""
    <html xmlns="http://www.w3.org/1999/xhtml"
          xmlns:svg="http://www.w3.org/2000/svg"
          xmlns:xlink="http://www.w3.org/1999/xlink">
      <head></head>
      <body>
        <div>
          <svg:svg viewBox="0 0 600 900">
            <svg:image width="600" height="900" xlink:href="images/first.jpg"/>
          </svg:svg>
        </div>
      </body>
    </html>
    """)

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.cover_page_declared is True
    assert opf.xpath(
        "string(//*[local-name()='reference' and @type='cover']/@href)"
    ) == "chapter.xhtml"
    assert "verbaloom-cover-page" in doc.xpath(
        "string(//*[local-name()='image']/ancestor::*[local-name()='div'][1]/@class)"
    )


def test_removes_obsolete_shape_from_anchors_without_changing_links(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package()
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    anchor = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}a")
    anchor.set("id", "note-1")
    anchor.set("href", "#reference-1")
    anchor.set("shape", "rect")
    anchor.text = "1"

    first = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )
    second = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert first.obsolete_attributes_removed == 1
    assert second.obsolete_attributes_removed == 0
    assert anchor.get("shape") is None
    assert anchor.get("id") == "note-1"
    assert anchor.get("href") == "#reference-1"
    assert anchor.text == "1"


def test_removes_obsolete_zero_table_border_without_changing_table(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package()
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    table = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}table")
    table.set("id", "results")
    table.set("class", "bodytable")
    table.set("border", "0")
    row = etree.SubElement(table, "{http://www.w3.org/1999/xhtml}tr")
    cell = etree.SubElement(row, "{http://www.w3.org/1999/xhtml}td")
    cell.text = "Resultado"

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.obsolete_attributes_removed == 1
    assert table.get("border") is None
    assert table.get("id") == "results"
    assert table.get("class") == "bodytable"
    assert "".join(table.itertext()) == "Resultado"


def test_cover_lookup_resolves_filesystem_aliases(tmp_path: Path):
    """Checkpoint paths may use a symlink alias while the OPF path is resolved."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    (real_root / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(real_root / "images" / "first.jpg")

    opf = _package()
    metadata = opf.xpath("//*[local-name()='metadata']")[0]
    meta = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
    meta.set("name", "cover")
    meta.set("content", "image-one")
    doc = _document()

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=real_root,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(alias_root / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.cover_page_declared is True
    assert opf.xpath("string(//*[local-name()='reference' and @type='cover']/@href)") == "chapter.xhtml"


def test_missing_cover_page_does_not_create_an_invalid_empty_guide(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    opf = _package()
    metadata = opf.xpath("//*[local-name()='metadata']")[0]
    meta = etree.SubElement(metadata, "{http://www.idpf.org/2007/opf}meta")
    meta.set("name", "cover")
    meta.set("content", "image-one")
    doc = _document()
    for image in doc.xpath("//*[local-name()='img']"):
        image.getparent().remove(image)

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.cover_page_declared is False
    assert not opf.xpath("//*[local-name()='guide']")


def test_marks_metadata_proven_running_page_headers_without_hiding_real_years(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    (tmp_path / "book.css").write_text("img { max-width: 100%; }", encoding="utf-8")
    opf = _package()
    metadata = opf.xpath("//*[local-name()='metadata']")[0]
    creator = etree.SubElement(metadata, "{http://purl.org/dc/elements/1.1/}creator")
    creator.text = "Ken Grimwood"
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    running = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    running.text = "234 Ken Grimwood"
    year = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    year.text = "1959"

    report = apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert report.furniture_markers == 1
    assert "verbaloom-furniture" in str(running.get("class") or "")
    assert "verbaloom-furniture" not in str(year.get("class") or "")


def test_normalizes_only_structurally_proven_ocr_scene_break(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    (tmp_path / "book.css").write_text("img { max-width: 100%; }", encoding="utf-8")
    opf = _package()
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    outer = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}ul")
    outer.set("style", "list-style:none;")
    outer_li = etree.SubElement(outer, "{http://www.w3.org/1999/xhtml}li")
    inner = etree.SubElement(outer_li, "{http://www.w3.org/1999/xhtml}ul")
    inner.set("style", "list-style: none")
    inner_li = etree.SubElement(inner, "{http://www.w3.org/1999/xhtml}li")
    artifact = etree.SubElement(inner_li, "{http://www.w3.org/1999/xhtml}p")
    artifact.text = "1. ."
    ordinary_list = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}ol")
    ordinary_item = etree.SubElement(ordinary_list, "{http://www.w3.org/1999/xhtml}li")
    ordinary = etree.SubElement(ordinary_item, "{http://www.w3.org/1999/xhtml}p")
    ordinary.text = "1. ."

    apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert artifact.text == "* * *"
    assert "verbaloom-scene-break" in str(artifact.get("class") or "")
    assert ordinary.text == "1. ."


def test_normalizes_punctuation_only_heading_as_scene_break(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    (tmp_path / "book.css").write_text("", encoding="utf-8")
    opf = _package()
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    damaged = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}h3")
    damaged.text = "."

    apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert damaged.text == "* * *"
    assert "verbaloom-scene-break" in str(damaged.get("class") or "")


def test_hides_only_commercial_inserts_around_reading_matter(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (600, 900), "navy").save(tmp_path / "images" / "first.jpg")
    (tmp_path / "book.css").write_text("", encoding="utf-8")
    opf = _package()
    doc = _document()
    body = doc.xpath("//*[local-name()='body']")[0]
    cover = body[0]
    heading = body[1]
    front_ad = etree.Element("{http://www.w3.org/1999/xhtml}p")
    front_ad.text = "Order at your bookstore: $5.99, $6.99. VISA or Mastercard."
    body.insert(1, front_ad)
    acknowledgments = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    acknowledgments.text = "Agradecimientos"
    thanks = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    thanks.text = "Gracias a quienes ayudaron con esta obra."
    back_ad = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    back_ad.text = "Apasionantes libros y thrillers"
    prices = etree.SubElement(body, "{http://www.w3.org/1999/xhtml}p")
    prices.text = "Libro uno $5.99. Libro dos $6.99. Libro tres $7.99. Haga su pedido."

    apply_professional_epub_layer(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc},
        target_language="Spanish",
    )

    assert "verbaloom-furniture" in str(front_ad.get("class") or "")
    assert "verbaloom-furniture" not in str(heading.get("class") or "")
    assert "verbaloom-furniture" not in str(acknowledgments.get("class") or "")
    assert "verbaloom-furniture" not in str(thanks.get("class") or "")
    assert "verbaloom-furniture" in str(back_ad.get("class") or "")
    assert "verbaloom-furniture" in str(prices.get("class") or "")
