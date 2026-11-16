from urllib.parse import unquote, urlsplit

from lxml import etree
import pytest

from src.core.epub.publication_structure import _write_cover_page
from src.core.epub.professionalize import _ensure_stylesheet


@pytest.fixture
def package_paths(tmp_path):
    real = tmp_path / "actual"
    opf = real / "OEBPS"
    opf.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    return opf, alias / "OEBPS"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("image_name", ["cover.jpg", "front cover #1%.jpg"])
def test_cover_url_survives_symlinked_workspace(package_paths, nested, image_name):
    opf, alias = package_paths
    image = opf / "images" / image_name
    image.parent.mkdir()
    image.write_bytes(b"image content")
    page = alias / ("Text/cover.xhtml" if nested else "cover.xhtml")
    from urllib.parse import quote
    _write_cover_page(page, opf_dir=alias, title="Book",
                      cover_href="images/" + quote(image_name, safe=""),
                      language_code="es", epub3=True)
    href = etree.parse(str(page)).xpath("string(//*[local-name()='img']/@src)")
    parsed = urlsplit(href)
    assert not parsed.scheme and not parsed.netloc and not parsed.fragment
    assert (page.parent / unquote(parsed.path)).resolve() == image
    assert str(opf.parent) not in href
    assert href.count("../") == int(nested)


@pytest.mark.parametrize("alias_document", [False, True])
def test_stylesheet_uses_same_canonical_base_as_document(package_paths, alias_document):
    opf, alias = package_paths
    doc_path = (alias if alias_document else opf) / "Text" / "chapter.xhtml"
    root = etree.XML(b"<html><head/><body><p>Text</p></body></html>")
    package = etree.XML(b"<package><manifest/></package>")
    _ensure_stylesheet(opf_root=package, opf_dir=opf if alias_document else alias,
                       parsed_xhtml_docs={str(doc_path): root})
    href = root.xpath("string(//*[local-name()='link']/@href)")
    assert href == "../verbaloom-professional.css"
    assert (doc_path.parent / unquote(href)).resolve().is_file()
