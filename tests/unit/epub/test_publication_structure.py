from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import zipfile

from lxml import etree
from PIL import Image

from src.core.epub.publication_structure import finish_epub_publication


def _package(*, with_cover: bool = False) -> etree._ElementTree:
    cover_manifest = (
        '<item id="source-cover" href="images/source-cover.jpg" '
        'media-type="image/jpeg" properties="cover-image"/>'
        '<item id="source-cover-page" href="cover.xhtml" '
        'media-type="application/xhtml+xml"/>'
        if with_cover
        else ""
    )
    cover_spine = '<itemref idref="source-cover-page"/>' if with_cover else ""
    cover_meta = '<meta name="cover" content="source-cover"/>' if with_cover else ""
    cover_guide = (
        '<guide><reference type="cover" title="Portada" href="cover.xhtml"/></guide>'
        if with_cover
        else ""
    )
    return etree.ElementTree(etree.fromstring(f"""
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0"
             unique-identifier="bookid">
      <metadata>
        <dc:identifier id="bookid">urn:test:professional-epub</dc:identifier>
        <dc:title>La ciudad y la memoria</dc:title>
        <dc:creator>Ana Torres</dc:creator>
        <dc:language>es</dc:language>
        {cover_meta}
      </metadata>
      <manifest>
        <item id="chapter-one" href="chapter-1.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter-two" href="chapter-2.xhtml" media-type="application/xhtml+xml"/>
        {cover_manifest}
      </manifest>
      <spine>{cover_spine}<itemref idref="chapter-one"/><itemref idref="chapter-two"/></spine>
      {cover_guide}
    </package>
    """.encode()))


def _chapter(title: str, body: str) -> etree._Element:
    return etree.fromstring(f"""
    <html xmlns="http://www.w3.org/1999/xhtml">
      <head><title>{title}</title></head>
      <body><section><h1>{title}</h1><p>{body}</p></section></body>
    </html>
    """.encode())


def test_finisher_generates_cover_chapter_semantics_and_epub3_navigation(tmp_path: Path):
    opf = _package()
    chapter_one = _chapter("Capítulo uno", "La historia comienza aquí.")
    chapter_two = _chapter("Capítulo dos", "La historia continúa aquí.")
    docs = {
        str(tmp_path / "chapter-1.xhtml"): chapter_one,
        str(tmp_path / "chapter-2.xhtml"): chapter_two,
    }

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs=docs,
        target_language="Spanish",
        title="La ciudad y la memoria",
        subtitle="Una novela",
        author="Ana Torres",
    )

    assert report.cover_generated is True
    assert report.cover_page_created is True
    assert report.navigation_created is True
    assert report.navigation_entries == 2
    assert report.chapter_headings == 2
    assert (tmp_path / "images" / "verbaloom-cover.jpg").is_file()
    assert (tmp_path / "verbaloom-cover.xhtml").is_file()
    assert (tmp_path / "verbaloom-nav.xhtml").is_file()

    with Image.open(tmp_path / "images" / "verbaloom-cover.jpg") as cover:
        assert cover.size == (1600, 2560)

    package = opf.getroot()
    assert package.xpath(
        "string(//*[local-name()='item' and @id='verbaloom-cover-image']/@properties)"
    ) == "cover-image"
    assert package.xpath(
        "string(//*[local-name()='spine']/*[local-name()='itemref'][1]/@idref)"
    ) == "verbaloom-cover-page"
    assert package.xpath(
        "string(//*[local-name()='item' and contains(concat(' ', @properties, ' '), ' nav ')]/@href)"
    ) == "verbaloom-nav.xhtml"

    headings = [chapter_one.xpath("//*[local-name()='h1']")[0], chapter_two.xpath("//*[local-name()='h1']")[0]]
    assert [heading.get("id") for heading in headings] == [
        "verbaloom-chapter-001",
        "verbaloom-chapter-002",
    ]
    assert all("verbaloom-chapter-title" in (heading.get("class") or "") for heading in headings)
    nav = (tmp_path / "verbaloom-nav.xhtml").read_text(encoding="utf-8")
    assert 'chapter-1.xhtml#verbaloom-chapter-001' in nav
    assert 'chapter-2.xhtml#verbaloom-chapter-002' in nav
    assert 'epub:type="landmarks"' in nav
    assert "Capítulo uno" in nav and "Capítulo dos" in nav


def test_finisher_is_idempotent(tmp_path: Path):
    opf = _package()
    chapter_one = _chapter("Capítulo uno", "Texto.")
    chapter_two = _chapter("Capítulo dos", "Texto.")
    docs = {
        str(tmp_path / "chapter-1.xhtml"): chapter_one,
        str(tmp_path / "chapter-2.xhtml"): chapter_two,
    }
    kwargs = dict(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs=docs,
        target_language="Spanish",
    )

    finish_epub_publication(**kwargs)
    cover_digest = hashlib.sha256(
        (tmp_path / "images" / "verbaloom-cover.jpg").read_bytes()
    ).hexdigest()
    second = finish_epub_publication(**kwargs)

    assert second.cover_generated is False
    assert hashlib.sha256(
        (tmp_path / "images" / "verbaloom-cover.jpg").read_bytes()
    ).hexdigest() == cover_digest
    assert len(opf.xpath("//*[local-name()='item' and @id='verbaloom-cover-image']")) == 1
    assert len(opf.xpath("//*[local-name()='item' and @id='verbaloom-cover-page']")) == 1
    assert len(opf.xpath("//*[local-name()='item' and @id='verbaloom-navigation']")) == 1
    assert len(opf.xpath("//*[local-name()='spine']/*[local-name()='itemref' and @idref='verbaloom-cover-page']")) == 1


def test_finisher_preserves_existing_cover_bytes(tmp_path: Path):
    (tmp_path / "images").mkdir()
    source_cover = tmp_path / "images" / "source-cover.jpg"
    Image.new("RGB", (1200, 1800), "navy").save(source_cover, quality=91)
    source_digest = hashlib.sha256(source_cover.read_bytes()).hexdigest()
    (tmp_path / "cover.xhtml").write_text(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><img '
        'src="images/source-cover.jpg" alt="La ciudad y la memoria"/></body></html>',
        encoding="utf-8",
    )
    opf = _package(with_cover=True)
    chapter_one = _chapter("Capítulo uno", "Texto.")
    chapter_two = _chapter("Capítulo dos", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs={
            str(tmp_path / "chapter-1.xhtml"): chapter_one,
            str(tmp_path / "chapter-2.xhtml"): chapter_two,
        },
        target_language="Spanish",
    )

    assert report.cover_generated is False
    assert report.cover_image_href == "images/source-cover.jpg"
    assert hashlib.sha256(source_cover.read_bytes()).hexdigest() == source_digest
    assert not (tmp_path / "images" / "verbaloom-cover.jpg").exists()


def test_finisher_replaces_a_declared_but_missing_cover_resource(tmp_path: Path):
    opf = _package(with_cover=True)
    (tmp_path / "cover.xhtml").write_text(
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><img '
        'src="images/source-cover.jpg" alt="Missing cover"/></body></html>',
        encoding="utf-8",
    )
    chapter = _chapter("Capítulo uno", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter-1.xhtml"): chapter},
        target_language="Spanish",
    )

    assert report.cover_generated is True
    assert report.cover_page_created is True
    assert report.cover_image_href == "images/verbaloom-cover.jpg"
    assert (tmp_path / "images" / "verbaloom-cover.jpg").is_file()
    assert report.cover_page_href == "verbaloom-cover.xhtml"
    assert 'src="images/verbaloom-cover.jpg"' in (
        tmp_path / "verbaloom-cover.xhtml"
    ).read_text(encoding="utf-8")
    assert opf.xpath(
        "string(//*[local-name()='metadata']/*[local-name()='meta' and @name='cover']/@content)"
    ) == "verbaloom-cover-image"
    assert not opf.xpath(
        "//*[local-name()='item' and @id='source-cover' and "
        "contains(concat(' ', @properties, ' '), ' cover-image ')]"
    )
    assert not opf.xpath("//*[local-name()='item' and @id='source-cover']")


def test_finisher_replaces_a_missing_cover_page_at_a_bounded_generated_path(tmp_path: Path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (1200, 1800), "navy").save(
        tmp_path / "images" / "source-cover.jpg"
    )
    opf = _package(with_cover=True)
    cover_item = opf.xpath("//*[local-name()='item' and @id='source-cover-page']")[0]
    cover_item.set("href", "Text/cover.xhtml")
    cover_reference = opf.xpath(
        "//*[local-name()='guide']/*[local-name()='reference' and @type='cover']"
    )[0]
    cover_reference.set("href", "Text/cover.xhtml#cover")
    chapter = _chapter("Capítulo uno", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter-1.xhtml"): chapter},
        target_language="Spanish",
    )

    cover_page = tmp_path / "verbaloom-cover.xhtml"
    assert report.cover_page_created is True
    assert cover_page.is_file()
    assert not (tmp_path / "Text" / "cover.xhtml").exists()
    assert 'src="images/source-cover.jpg"' in cover_page.read_text(encoding="utf-8")
    assert not opf.xpath("//*[local-name()='item' and @id='source-cover-page']")


def test_finisher_indexes_multiple_chapters_in_one_xhtml_without_indexing_subheads(tmp_path: Path):
    opf = etree.ElementTree(etree.fromstring(b'''
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
      <metadata><dc:title>Libro</dc:title><dc:language>es</dc:language></metadata>
      <manifest><item id="body" href="body.xhtml" media-type="application/xhtml+xml"/></manifest>
      <spine><itemref idref="body"/></spine>
    </package>'''))
    document = etree.fromstring(b'''
    <html xmlns="http://www.w3.org/1999/xhtml"><head><title>Libro</title></head><body>
      <h1>Capitulo uno</h1><p>Texto uno.</p>
      <h2>Una observacion interna</h2><p>Texto interno.</p>
      <h1>Capitulo dos</h1><p>Texto dos.</p>
    </body></html>''')

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["body.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "body.xhtml"): document},
        target_language="Spanish",
    )

    nav = (tmp_path / "verbaloom-nav.xhtml").read_text(encoding="utf-8")
    assert report.navigation_entries == 2
    assert "Capitulo uno" in nav
    assert "Capitulo dos" in nav
    assert "Una observacion interna" not in nav


def test_finisher_reconciles_file_level_toc_entries_without_duplicate_chapters(tmp_path: Path):
    opf = _package()
    manifest = opf.xpath("//*[local-name()='manifest']")[0]
    nav_item = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
    nav_item.set("id", "source-nav")
    nav_item.set("href", "nav.xhtml")
    nav_item.set("media-type", "application/xhtml+xml")
    nav_item.set("properties", "nav")
    ncx_item = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
    ncx_item.set("id", "source-ncx")
    ncx_item.set("href", "toc.ncx")
    ncx_item.set("media-type", "application/x-dtbncx+xml")
    (tmp_path / "nav.xhtml").write_text(
        '''<html xmlns="http://www.w3.org/1999/xhtml"
        xmlns:epub="http://www.idpf.org/2007/ops"><body>
        <nav epub:type="toc"><ol>
        <li><a href="chapter-1.xhtml">Chapter one</a></li>
        <li><a href="chapter-2.xhtml">Chapter two</a></li>
        </ol></nav></body></html>''',
        encoding="utf-8",
    )
    (tmp_path / "toc.ncx").write_text(
        '''<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><navMap>
        <navPoint id="one"><navLabel><text>Chapter one</text></navLabel>
        <content src="chapter-1.xhtml"/></navPoint>
        <navPoint id="two"><navLabel><text>Chapter two</text></navLabel>
        <content src="chapter-2.xhtml"/></navPoint>
        </navMap></ncx>''',
        encoding="utf-8",
    )
    chapter_one = _chapter("Capítulo uno", "Texto.")
    chapter_two = _chapter("Capítulo dos", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs={
            str(tmp_path / "chapter-1.xhtml"): chapter_one,
            str(tmp_path / "chapter-2.xhtml"): chapter_two,
        },
        target_language="Spanish",
    )

    nav = etree.parse(str(tmp_path / "nav.xhtml"))
    toc = nav.xpath(
        "//*[local-name()='nav' and @*[local-name()='type']='toc']"
    )[0]
    labels = [
        "".join(anchor.itertext())
        for anchor in toc.xpath(".//*[local-name()='a']")
    ]
    targets = toc.xpath(".//*[local-name()='a']/@href")
    assert report.navigation_entries == 2
    assert labels == ["Capítulo uno", "Capítulo dos"]
    assert targets == [
        "chapter-1.xhtml#verbaloom-chapter-001",
        "chapter-2.xhtml#verbaloom-chapter-002",
    ]


def test_finisher_keeps_title_pages_out_of_the_chapter_toc(tmp_path: Path):
    opf = _package()
    manifest = opf.xpath("//*[local-name()='manifest']")[0]
    title_item = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
    title_item.set("id", "title-page")
    title_item.set("href", "title.xhtml")
    title_item.set("media-type", "application/xhtml+xml")
    spine = opf.xpath("//*[local-name()='spine']")[0]
    title_ref = etree.Element("{http://www.idpf.org/2007/opf}itemref")
    title_ref.set("idref", "title-page")
    spine.insert(0, title_ref)
    title_page = etree.fromstring(b'''
    <html xmlns="http://www.w3.org/1999/xhtml"
      xmlns:epub="http://www.idpf.org/2007/ops"><body>
      <section epub:type="titlepage"><h1>La ciudad y la memoria</h1></section>
      </body></html>''')
    chapter_one = _chapter("Capítulo uno", "Texto.")
    chapter_two = _chapter("Capítulo dos", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["title.xhtml", "chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs={
            str(tmp_path / "title.xhtml"): title_page,
            str(tmp_path / "chapter-1.xhtml"): chapter_one,
            str(tmp_path / "chapter-2.xhtml"): chapter_two,
        },
        target_language="Spanish",
    )

    nav = etree.parse(str(tmp_path / "verbaloom-nav.xhtml"))
    labels = [
        "".join(anchor.itertext())
        for anchor in nav.xpath(
            "//*[local-name()='nav' and @*[local-name()='type']='toc']"
            "//*[local-name()='a']"
        )
    ]
    assert report.navigation_entries == 2
    assert labels == ["Capítulo uno", "Capítulo dos"]


def test_finisher_confines_navigation_reads_and_writes_to_the_package(tmp_path: Path):
    package_root = tmp_path / "book"
    package_root.mkdir()
    outside_nav = tmp_path / "outside-nav.xhtml"
    outside_payload = b"<outside>do not touch</outside>"
    outside_nav.write_bytes(outside_payload)
    opf = _package()
    manifest = opf.xpath("//*[local-name()='manifest']")[0]
    nav_item = etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item")
    nav_item.set("id", "unsafe-nav")
    nav_item.set("href", "../outside-nav.xhtml")
    nav_item.set("media-type", "application/xhtml+xml")
    nav_item.set("properties", "nav")
    chapter_one = _chapter("Capítulo uno", "Texto.")
    chapter_two = _chapter("Capítulo dos", "Texto.")

    report = finish_epub_publication(
        opf_tree=opf,
        opf_dir=package_root,
        package_root=package_root,
        content_files=["chapter-1.xhtml", "chapter-2.xhtml"],
        parsed_xhtml_docs={
            str(package_root / "chapter-1.xhtml"): chapter_one,
            str(package_root / "chapter-2.xhtml"): chapter_two,
        },
        target_language="Spanish",
    )

    assert outside_nav.read_bytes() == outside_payload
    assert report.navigation_href == "verbaloom-nav.xhtml"
    assert (package_root / "verbaloom-nav.xhtml").is_file()
    assert not opf.xpath("//*[local-name()='item' and @id='unsafe-nav']")


def test_finisher_preserves_recovered_existing_epub_type(tmp_path: Path):
    opf = _package()
    document = etree.fromstring(
        b'<html><head><title>Portada</title></head><body><section '
        b'epub:type="titlepage"><h1>Libro</h1></section></body></html>',
        etree.XMLParser(recover=True),
    )

    finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter-1.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter-1.xhtml"): document},
        target_language="Spanish",
    )

    section = document.xpath("//*[local-name()='section']")[0]
    type_attributes = [
        key
        for key in section.attrib
        if str(key).rsplit("}", 1)[-1].rsplit(":", 1)[-1] == "type"
    ]
    assert type_attributes == ["epub:type"]
    assert section.get("epub:type") == "titlepage"


def test_native_epub_finisher_produces_epubcheck_valid_epub3(tmp_path: Path):
    epubcheck = shutil.which("epubcheck") or "/opt/homebrew/bin/epubcheck"
    if not Path(epubcheck).exists():
        return
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
        '</rootfiles></container>',
        encoding="utf-8",
    )
    opf = etree.ElementTree(etree.fromstring(b'''
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0"
             unique-identifier="bookid">
      <metadata>
        <dc:identifier id="bookid">urn:test:native-professional</dc:identifier>
        <dc:title>Libro de prueba</dc:title><dc:creator>Ana Torres</dc:creator>
        <dc:language>es</dc:language>
        <meta property="dcterms:modified">2026-09-29T12:00:00Z</meta>
      </metadata>
      <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
      <spine><itemref idref="chapter"/></spine>
    </package>'''))
    document = etree.fromstring('''
    <html xmlns="http://www.w3.org/1999/xhtml"><head><title>Capítulo uno</title></head>
      <body><div><h1>Capítulo uno</h1><p>La historia comienza aquí.</p></div></body>
    </html>'''.encode())
    finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): document},
        target_language="Spanish",
    )
    opf.write(tmp_path / "content.opf", encoding="utf-8", xml_declaration=True)
    (tmp_path / "chapter.xhtml").write_bytes(
        etree.tostring(document, encoding="utf-8", xml_declaration=True)
    )
    output = tmp_path / "native-professional.epub"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for path in sorted(tmp_path.rglob("*")):
            if not path.is_file() or path == output:
                continue
            relative = path.relative_to(tmp_path).as_posix()
            if relative == "mimetype":
                continue
            archive.write(path, relative, compress_type=zipfile.ZIP_DEFLATED)

    checked = subprocess.run(
        [epubcheck, str(output)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr


def test_native_epub_finisher_produces_epubcheck_valid_epub2(tmp_path: Path):
    epubcheck = shutil.which("epubcheck") or "/opt/homebrew/bin/epubcheck"
    if not Path(epubcheck).exists():
        return
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container version="1.0" '
        'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
        '<rootfile full-path="content.opf" media-type="application/oebps-package+xml"/>'
        '</rootfiles></container>',
        encoding="utf-8",
    )
    opf = etree.ElementTree(etree.fromstring(b'''
    <package xmlns="http://www.idpf.org/2007/opf"
             xmlns:dc="http://purl.org/dc/elements/1.1/" version="2.0"
             unique-identifier="bookid">
      <metadata>
        <dc:identifier id="bookid">urn:test:native-professional-epub2</dc:identifier>
        <dc:title>Libro de prueba</dc:title><dc:creator>Ana Torres</dc:creator>
        <dc:language>es</dc:language>
      </metadata>
      <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
      <spine><itemref idref="chapter"/></spine>
    </package>'''))
    document = etree.fromstring('''
    <html xmlns="http://www.w3.org/1999/xhtml"><head><title>Capítulo uno</title></head>
      <body><div><h1>Capítulo uno</h1><p>La historia comienza aquí.</p></div></body>
    </html>'''.encode())
    finish_epub_publication(
        opf_tree=opf,
        opf_dir=tmp_path,
        content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): document},
        target_language="Spanish",
    )
    opf.write(tmp_path / "content.opf", encoding="utf-8", xml_declaration=True)
    (tmp_path / "chapter.xhtml").write_bytes(
        etree.tostring(document, encoding="utf-8", xml_declaration=True)
    )
    output = tmp_path / "native-professional-epub2.epub"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        for path in sorted(tmp_path.rglob("*")):
            if not path.is_file() or path == output:
                continue
            relative = path.relative_to(tmp_path).as_posix()
            archive.write(path, relative, compress_type=zipfile.ZIP_DEFLATED)

    checked = subprocess.run(
        [epubcheck, str(output)],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
