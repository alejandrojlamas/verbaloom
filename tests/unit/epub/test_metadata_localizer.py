from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from lxml import etree

from src.core.epub.metadata_localizer import (
    infer_epub_title_page,
    localize_epub_metadata,
)


CONTAINER = b'''<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>'''


def _chapter(label: str = "Kapitel") -> bytes:
    return f'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="de" xml:lang="de">
<head><title></title></head><body><h1 id="start">{label}</h1><p>Texto traducido.</p></body></html>'''.encode()


def _write_epub2(path: Path, *, extra_identifier: bool = False):
    extra = '<dc:identifier id="invented">urn:invented</dc:identifier>' if extra_identifier else ''
    opf = f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0" unique-identifier="bookid">
<metadata><dc:title>Altes Buch</dc:title><dc:creator>Autor Original</dc:creator><dc:identifier id="bookid">urn:source</dc:identifier>{extra}<dc:language>de</dc:language><dc:date>1901</dc:date></metadata>
<manifest><item id="chapter" href="Text/chapter.xhtml" media-type="application/xhtml+xml"/><item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/><item id="image" href="Images/cover.jpg" media-type="image/jpeg"/></manifest>
<spine toc="ncx"><itemref idref="chapter"/></spine></package>'''.encode()
    ncx = b'''<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><docTitle><text>Altes Buch</text></docTitle><navMap>
<navPoint id="n1"><navLabel><text>Cover</text></navLabel><content src="Text/chapter.xhtml#start"/></navPoint>
</navMap></ncx>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/toc.ncx", ncx)
        archive.writestr("OEBPS/Text/chapter.xhtml", _chapter())
        archive.writestr("OEBPS/Images/cover.jpg", b"preserved-image")


def _write_epub3(path: Path):
    opf = b'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
<metadata><dc:title>Old Book</dc:title><dc:creator>Original Author</dc:creator><dc:identifier>urn:epub3</dc:identifier><dc:language>en</dc:language></metadata>
<manifest><item id="chapter" href="Text/chapter.xhtml" media-type="application/xhtml+xml"/><item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/></manifest>
<spine><itemref idref="chapter"/></spine></package>'''
    nav = b'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops"><head><title>Contents</title></head><body><nav epub:type="toc"><ol><li><a href="Text/chapter.xhtml#start">Introduction</a></li></ol></nav></body></html>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/nav.xhtml", nav)
        archive.writestr("OEBPS/Text/chapter.xhtml", _chapter("Introduction"))


def _write_epub3_with_refined_titles(path: Path):
    opf = b'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
<metadata>
  <dc:title id="t1">A Sudden Flicker of Light</dc:title>
  <meta property="title-type" refines="#t1">main</meta>
  <dc:title id="t2">A Revisionist History of Movies</dc:title>
  <meta property="title-type" refines="#t2">subtitle</meta>
  <dc:creator>David Thomson</dc:creator>
  <dc:identifier>urn:epub3-refined</dc:identifier>
  <dc:language>en</dc:language>
</metadata>
<manifest>
  <item id="signup_front" href="Text/signup_front.xhtml" media-type="application/xhtml+xml"/>
  <item id="title" href="Text/title.xhtml" media-type="application/xhtml+xml"/>
</manifest>
<spine><itemref idref="title"/><itemref idref="signup_front"/></spine>
</package>'''
    image_title_page = b'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head><title>Title Page</title></head>
<body><div><img alt="A Sudden Flicker of Light" src="../Images/title.jpg"/></div></body>
</html>'''
    signup = b'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head><title>Signup</title></head>
<body><p>Thanks for downloading this ebook from the publisher.</p></body>
</html>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            b"application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("META-INF/container.xml", CONTAINER)
        archive.writestr("OEBPS/content.opf", opf)
        # Deliberately put the promotional file first in ZIP order.
        archive.writestr("OEBPS/Text/signup_front.xhtml", signup)
        archive.writestr("OEBPS/Text/title.xhtml", image_title_page)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_localizes_epub2_without_inventing_identity_or_changing_links(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub2(source)
    _write_epub2(output, extra_identifier=True)
    source_hash = _sha(source)

    report = localize_epub_metadata(
        source,
        output,
        target_language="Spanish",
        title="Libro actual",
        subtitle="Una lectura",
    )

    assert _sha(source) == source_hash
    assert report.changed_files >= 3
    with zipfile.ZipFile(output) as archive:
        package = etree.fromstring(archive.read("OEBPS/content.opf"))
        assert package.xpath("//*[local-name()='title']/text()") == ["Libro actual: Una lectura"]
        assert package.xpath("//*[local-name()='language']/text()") == ["es"]
        assert package.xpath("//*[local-name()='identifier']/text()") == ["urn:source"]
        assert package.xpath("//*[local-name()='creator']/text()") == ["Autor Original"]
        assert package.xpath("//*[local-name()='date']/text()") == ["1901"]
        ncx = etree.fromstring(archive.read("OEBPS/toc.ncx"))
        assert ncx.xpath("//*[local-name()='navLabel']/*[local-name()='text']/text()") == ["Portada"]
        assert ncx.xpath("string(//*[local-name()='content']/@src)") == "Text/chapter.xhtml#start"
        chapter = etree.fromstring(archive.read("OEBPS/Text/chapter.xhtml"))
        assert chapter.get("lang") == "es"
        assert chapter.xpath("//*[local-name()='title']/text()") == ["Portada"]
        assert archive.read("OEBPS/Images/cover.jpg") == b"preserved-image"
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED


def test_localizes_epub3_nav_label_and_preserves_target(tmp_path):
    source = tmp_path / "source3.epub"
    output = tmp_path / "output3.epub"
    _write_epub3(source)
    _write_epub3(output)

    localize_epub_metadata(
        source,
        output,
        target_language="Spanish",
        title="Libro EPUB 3",
    )

    with zipfile.ZipFile(output) as archive:
        nav = etree.fromstring(archive.read("OEBPS/nav.xhtml"))
        anchor = nav.xpath("//*[local-name()='nav']//*[local-name()='a']")[0]
        assert "".join(anchor.itertext()) == "Introducción"
        assert anchor.get("href") == "Text/chapter.xhtml#start"


def test_image_only_title_page_uses_opf_identity_not_promotional_front_matter(
    tmp_path,
):
    epub = tmp_path / "refined.epub"
    _write_epub3_with_refined_titles(epub)

    assert infer_epub_title_page(epub) == (
        "A Sudden Flicker of Light",
        "A Revisionist History of Movies",
    )


def test_localizer_preserves_refined_title_targets_and_localizes_each_value(
    tmp_path,
):
    source = tmp_path / "source-refined.epub"
    output = tmp_path / "output-refined.epub"
    _write_epub3_with_refined_titles(source)
    _write_epub3_with_refined_titles(output)

    localize_epub_metadata(
        source,
        output,
        target_language="Spanish",
        title="Un repentino destello de luz",
        subtitle="Una historia revisionista del cine",
    )

    with zipfile.ZipFile(output) as archive:
        package = etree.fromstring(archive.read("OEBPS/content.opf"))
        titles = {
            node.get("id"): "".join(node.itertext())
            for node in package.xpath("//*[local-name()='title']")
        }
        refinements = {
            node.get("refines"): "".join(node.itertext())
            for node in package.xpath(
                "//*[local-name()='meta' and @property='title-type']"
            )
        }

    assert titles == {
        "t1": "Un repentino destello de luz",
        "t2": "Una historia revisionista del cine",
    }
    assert refinements == {"#t1": "main", "#t2": "subtitle"}
