from __future__ import annotations

import pytest
from lxml import etree

from src.core.epub.translator import _update_epub_metadata
from src.utils.branding import GENERATOR_NAME


def test_spanish_metadata_uses_es_not_sp(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", False)
    opf_path = tmp_path / "content.opf"
    root = etree.fromstring(
        b"""<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title><dc:language>de-DE</dc:language></metadata>
             <manifest/><spine/>
             </package>"""
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
        b"""<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title></metadata>
             <manifest/><spine/>
             </package>"""
    )

    _update_epub_metadata(etree.ElementTree(root), str(opf_path), "Spanish")

    parsed = etree.parse(str(opf_path))
    assert parsed.xpath("string(//*[local-name()='language'])") == "es"


def test_metadata_update_is_idempotent_on_reprocessing(tmp_path, monkeypatch):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", True)
    opf_path = tmp_path / "content.opf"
    root = etree.fromstring(
        b"""<package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/">
             <metadata><dc:title>Book</dc:title><dc:language>en</dc:language></metadata>
             <manifest/><spine/>
             </package>"""
    )
    tree = etree.ElementTree(root)

    _update_epub_metadata(tree, str(opf_path), "Spanish")
    _update_epub_metadata(tree, str(opf_path), "Spanish")

    parsed = etree.parse(str(opf_path))
    assert len(parsed.xpath("//*[local-name()='identifier' and @id='render-uid']")) == 1
    assert len(parsed.xpath("//*[local-name()='contributor']")) == 1
    description = parsed.xpath("string(//*[local-name()='description'])")
    assert description.count("Translated using") == 1


@pytest.mark.parametrize("version", ["2.0", "3.0", "3.3"])
@pytest.mark.parametrize("existing", [False, True])
def test_translator_role_matches_epub_version(tmp_path, monkeypatch, version, existing):
    monkeypatch.setattr("src.core.epub.translator.ATTRIBUTION_ENABLED", True)
    opf = "http://www.idpf.org/2007/opf"
    dc = "http://purl.org/dc/elements/1.1/"
    root = etree.Element(
        f"{{{opf}}}package", version=version, nsmap={None: opf, "dc": dc}
    )
    metadata = etree.SubElement(root, f"{{{opf}}}metadata")
    # IDs outside metadata must not collide with the application's contributor.
    etree.SubElement(root, f"{{{opf}}}manifest", id="verbaloom-translator")
    author = etree.SubElement(metadata, f"{{{dc}}}creator", id="author")
    author.text = "Original Author"
    original = etree.tostring(author)
    if existing:
        contributor = etree.SubElement(
            metadata, f"{{{dc}}}contributor", id="translator-old"
        )
        contributor.text = "TranslateBookWithLLM"
        contributor.set(f"{{{opf}}}role", "trl")
        if version.startswith("3"):
            etree.SubElement(
                metadata,
                f"{{{opf}}}meta",
                refines="#translator-old",
                property="role",
                scheme="marc:relators",
            ).text = "trl"
            etree.SubElement(
                metadata,
                f"{{{opf}}}meta",
                refines="#translator-old",
                property="alternate-script",
            ).text = "Existing attribution"
    tree = etree.ElementTree(root)
    path = str(tmp_path / "content.opf")
    _update_epub_metadata(tree, path, "Spanish")
    _update_epub_metadata(tree, path, "Spanish")
    contributors = metadata.findall(f"{{{dc}}}contributor")
    assert len(contributors) == 1
    contributor = contributors[0]
    assert contributor.text == GENERATOR_NAME
    assert etree.tostring(author) == original
    if version.startswith("2"):
        assert contributor.get(f"{{{opf}}}role") == "trl"
    else:
        assert contributor.get(f"{{{opf}}}role") is None
        assert contributor.get("id") == (
            "translator-old" if existing else "verbaloom-translator-2"
        )
        roles = metadata.xpath(
            "opf:meta[@property='role' and @refines=$ref]",
            namespaces={"opf": opf},
            ref="#" + contributor.get("id"),
        )
        assert [(node.text, node.get("scheme")) for node in roles] == [
            ("trl", "marc:relators")
        ]
        if existing:
            assert (
                metadata.xpath(
                    "string(opf:meta[@property='alternate-script'])",
                    namespaces={"opf": opf},
                )
                == "Existing attribution"
            )
