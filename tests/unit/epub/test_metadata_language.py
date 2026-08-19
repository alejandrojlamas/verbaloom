from __future__ import annotations

from lxml import etree

from src.core.epub.translator import _update_epub_metadata


def test_spanish_metadata_uses_es_not_sp(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", False)
    opf_path = tmp_path / "content.opf"
    root = etree.fromstring(
        b'''<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title><dc:language>de-DE</dc:language></metadata>
             <manifest/><spine/>
             </package>'''
    )

    _update_epub_metadata(etree.ElementTree(root), str(opf_path), "Spanish")

    parsed = etree.parse(str(opf_path))
    languages = parsed.xpath(
        "//*[local-name()='language']/text()",
    )
    assert languages == ["es"]


def test_metadata_adds_missing_language_element(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", False)
    opf_path = tmp_path / "content.opf"
    root = etree.fromstring(
        b'''<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title></metadata>
             <manifest/><spine/>
             </package>'''
    )

    _update_epub_metadata(etree.ElementTree(root), str(opf_path), "Spanish")

    parsed = etree.parse(str(opf_path))
    assert parsed.xpath("string(//*[local-name()='language'])") == "es"


def test_metadata_update_is_idempotent_on_reprocessing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", True)
    opf_path = tmp_path / "content.opf"
    root = etree.fromstring(
        b'''<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title><dc:language>en</dc:language></metadata>
             <manifest/><spine/>
             </package>'''
    )
    tree = etree.ElementTree(root)

    _update_epub_metadata(tree, str(opf_path), "Spanish")
    _update_epub_metadata(tree, str(opf_path), "Spanish")

    parsed = etree.parse(str(opf_path))
    assert len(parsed.xpath("//*[local-name()='identifier' and @id='render-uid']")) == 1
    assert len(parsed.xpath("//*[local-name()='contributor']")) == 1
    description = parsed.xpath("string(//*[local-name()='description'])")
    assert description.count("Translated using") == 1
