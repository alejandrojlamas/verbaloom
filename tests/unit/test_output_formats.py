from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

from src.core.output_formats import (
    convert_output_file,
    ensure_output_extension,
    extract_readable_text,
    native_output_format,
    requested_format_for_job,
)
from src.utils.text_encoding import UTF8_BOM


def test_output_format_resolution_and_extension():
    assert native_output_format("pdf") == "txt"
    assert native_output_format("epub") == "epub"
    assert requested_format_for_job("pdf", "auto") == "txt"
    assert requested_format_for_job("txt", "docx") == "docx"
    assert ensure_output_extension("book.output.txt", "pdf") == "book.output.pdf"
    assert ensure_output_extension("book.output.txt", "epub") == "book.output.epub"
    assert ensure_output_extension("payload.cmd", "txt") == "payload.txt"


def test_convert_text_to_docx(tmp_path):
    docx = pytest.importorskip("docx")

    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.docx"
    source.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    convert_output_file(source, destination, "docx")

    document = docx.Document(str(destination))
    paragraphs = [p.text for p in document.paragraphs]
    assert paragraphs == ["First paragraph.", "Second paragraph."]


def test_convert_text_to_txt_writes_utf8_bom(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.txt"
    source.write_text("Texto con acentos: inglés, francés, atención.", encoding="utf-8")

    convert_output_file(source, destination, "txt")

    data = destination.read_bytes()
    assert data.startswith(UTF8_BOM)
    assert destination.read_text(encoding="utf-8-sig") == "Texto con acentos: inglés, francés, atención."


def test_convert_text_to_pdf_and_extract(tmp_path):
    pytest.importorskip("reportlab")

    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.pdf"
    source.write_text("PDF output paragraph.", encoding="utf-8")

    convert_output_file(source, destination, "pdf")

    assert destination.exists()
    assert "PDF output paragraph." in extract_readable_text(Path(destination))


def test_convert_text_to_epub_renders_markdown_tables_as_native_tables(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Table 2: Results\n\n"
        "| Model | BLEU | Cost |\n"
        "| --- | --- | --- |\n"
        "| ByteNet [18] | 23.75 | 1.0 · 10^20 |\n"
        "| Transformer | 28.4 | 2.3 · 10^19 |\n\n"
        "Next paragraph.",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination, "r") as zf:
        chapters = [name for name in zf.namelist() if name.startswith("OEBPS/chap-")]
        xhtml = "\n".join(zf.read(name).decode("utf-8") for name in chapters)

    assert "<table>" in xhtml
    assert "<th>Model</th>" in xhtml
    assert "<td>ByteNet [18]</td>" in xhtml
    assert "<td>23.75</td>" in xhtml
    assert "| Model | BLEU |" not in xhtml


def test_convert_text_to_epub_adds_editorial_chapter_semantics_and_breaks(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text("Capítulo 1\n\nPrimer párrafo.\n\nSegundo párrafo.", encoding="utf-8")

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination, "r") as zf:
        styles = zf.read("OEBPS/styles.css").decode("utf-8")
        chapter_name = next(name for name in zf.namelist() if name.startswith("OEBPS/chap-"))
        xhtml = zf.read(chapter_name).decode("utf-8")

    assert 'epub:type="chapter"' in xhtml
    assert 'class="chapter"' in xhtml
    assert "page-break-before: always" in styles
    assert "page-break-after: avoid" in styles
    assert "widows: 2" in styles


def test_convert_text_to_epub_builds_professional_package_with_cover_and_metadata(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Capítulo uno\n\nPrimer párrafo.\n\n"
        "Capítulo dos\n\nSegundo párrafo.",
        encoding="utf-8",
    )

    convert_output_file(
        source,
        destination,
        "epub",
        epub_metadata={
            "title": "La ciudad y la memoria",
            "subtitle": "Una novela",
            "creator": "Ana Torres",
            "language": "Spanish",
        },
    )

    with zipfile.ZipFile(destination) as zf:
        names = set(zf.namelist())
        package = zf.read("OEBPS/content.opf").decode("utf-8")
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        cover_page = zf.read("OEBPS/cover.xhtml").decode("utf-8")
        cover_bytes = zf.read("OEBPS/images/cover.jpg")

    assert "OEBPS/images/cover.jpg" in names
    assert "OEBPS/cover.xhtml" in names
    assert '<dc:title id="title-main">La ciudad y la memoria</dc:title>' in package
    assert '<dc:title id="title-subtitle">Una novela</dc:title>' in package
    assert '<dc:creator>Ana Torres</dc:creator>' in package
    assert '<dc:language>es</dc:language>' in package
    assert 'properties="cover-image"' in package
    assert '<itemref idref="cover-page" linear="yes"/>' in package
    assert 'epub:type="landmarks"' in nav
    assert 'href="cover.xhtml"' in nav
    assert 'lang="es" xml:lang="es"' in nav
    assert 'src="images/cover.jpg"' in cover_page
    assert cover_bytes.startswith(b"\xff\xd8")

    epubcheck = shutil.which("epubcheck") or "/opt/homebrew/bin/epubcheck"
    if Path(epubcheck).exists():
        checked = subprocess.run(
            [epubcheck, str(destination)],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def test_convert_text_to_epub_keeps_adjacent_markdown_tables_separate(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Table 2: Results\n"
        "| Model | BLEU |\n"
        "| --- | --- |\n"
        "| ByteNet | 23.75 |\n"
        "Residual dropout paragraph between tables.\n"
        "Table 3: Ablations\n"
        "| Variant | PPL |\n"
        "| --- | --- |\n"
        "| base | 4.92 |\n",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination, "r") as zf:
        chapters = [name for name in zf.namelist() if name.startswith("OEBPS/chap-")]
        xhtml = "\n".join(zf.read(name).decode("utf-8") for name in chapters)

    assert xhtml.count("<table>") == 2
    assert "<td>ByteNet</td>" in xhtml
    assert "<td>base</td>" in xhtml
    assert "Residual dropout paragraph between tables." in xhtml


def test_convert_text_to_epub_drops_malformed_markdown_table_header_fragments(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Table 2: Results with a malformed OCR header.\n"
        "| Model | BLEU Cost (FLOPs) | EN-DE EN-FR\n"
        "| Element | Value 1 | Value 2 |\n"
        "| --- | --- | --- |\n"
        "| ByteNet | 23.75 | 1.0 · 10^20 |\n",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination, "r") as zf:
        chapters = [name for name in zf.namelist() if name.startswith("OEBPS/chap-")]
        xhtml = "\n".join(zf.read(name).decode("utf-8") for name in chapters)

    assert "<table>" in xhtml
    assert "<td>ByteNet</td>" in xhtml
    assert "| Model | BLEU" not in xhtml


def test_extract_epub_text_uses_spine_order_not_zip_order(tmp_path):
    epub_path = tmp_path / "book.epub"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="front" href="front.xhtml" media-type="application/xhtml+xml"/>
    <item id="toc" href="toc.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="front"/>
    <itemref idref="toc"/>
    <itemref idref="chapter"/>
  </spine>
</package>
"""

    def xhtml(body: str) -> str:
        return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>{body}</body>
</html>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OPS/chapter.xhtml", xhtml("<h1>Chapter text</h1>"))
        zf.writestr("OPS/package.opf", package_opf)
        zf.writestr("OPS/front.xhtml", xhtml("<h1>Copyright page</h1><p>Edited by Someone.</p>"))
        zf.writestr("OPS/toc.xhtml", xhtml("<h1>Contents</h1><p>First entry.</p>"))

    text = extract_readable_text(epub_path)

    assert text.index("Copyright page") < text.index("Contents") < text.index("Chapter text")
    assert "Copyright page\n\nEdited by Someone." in text


def test_convert_epub_to_txt_uses_spine_order(tmp_path):
    epub_path = tmp_path / "translated.epub"
    txt_path = tmp_path / "translated.txt"

    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="front" href="front.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="front"/>
    <itemref idref="chapter"/>
  </spine>
</package>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        # Deliberately write the chapter before front matter to simulate the
        # failure mode from the user's exported book.
        zf.writestr(
            "OPS/chapter.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Chapter text</h1></body></html>',
        )
        zf.writestr("OPS/package.opf", package_opf)
        zf.writestr(
            "OPS/front.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Copyright page</h1><p>Contents</p></body></html>',
        )

    convert_output_file(epub_path, txt_path, "txt")

    assert txt_path.read_bytes().startswith(UTF8_BOM)
    text = txt_path.read_text(encoding="utf-8-sig")
    assert text.index("Copyright page") < text.index("Chapter text")
    assert text.startswith("Copyright page\n\nContents")


def test_extract_epub_text_does_not_append_unspined_nav_when_spine_exists(tmp_path):
    epub_path = tmp_path / "book.epub"
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OPS/package.opf", package_opf)
        zf.writestr(
            "OPS/chapter.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Real chapter</h1><p>Body text.</p></body></html>',
        )
        zf.writestr(
            "OPS/nav.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><nav><ol><li>Real chapter</li></ol></nav></body></html>',
        )

    text = extract_readable_text(epub_path)

    assert text == "Real chapter\n\nBody text."


def test_extract_epub_text_handles_uppercase_body_markup(tmp_path):
    epub_path = tmp_path / "uppercase-body.epub"
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OPS/package.opf", package_opf)
        zf.writestr(
            "OPS/chapter.xhtml",
            "<HTML><BODY><P>Under the Volcano readable text.</P></BODY></HTML>",
        )

    assert extract_readable_text(epub_path) == "Under the Volcano readable text."


def test_extract_epub_text_falls_back_to_content_root_without_body(tmp_path):
    epub_path = tmp_path / "root-only.epub"
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    package_opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" xmlns="http://www.idpf.org/2007/opf">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OPS/package.opf", package_opf)
        zf.writestr(
            "OPS/chapter.xhtml",
            "<section><p>Root-only readable passage.</p></section>",
        )

    assert extract_readable_text(epub_path) == "Root-only readable passage."


def test_convert_text_to_epub_builds_navigation_from_embedded_titles(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Atrapados en el hielo artico, 1596. La odisea de los marineros "
        "holandeses empieza aqui. El dia siguiente caminaron sobre el hielo. "
        "El dia veintiseis prepararon la casa.\n\n"
        "El texto anterior termina aqui. Sacrificio humano entre los aztecas, "
        "c. 1520 Jose de Acosta El autor fue misionero. En verdad, los "
        "mexicanos no sacrificaban sino cautivos.",
        encoding="utf-8",
    )

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination) as zf:
        names = set(zf.namelist())
        assert "mimetype" in names
        assert "OEBPS/nav.xhtml" in names
        assert "OEBPS/chap-001.xhtml" in names
        assert "OEBPS/chap-002.xhtml" in names
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        chapter = zf.read("OEBPS/chap-001.xhtml").decode("utf-8")

    assert "Atrapados en el hielo artico, 1596" in nav
    assert "Sacrificio humano entre los aztecas, c. 1520" in nav
    assert "<h1>Atrapados en el hielo artico, 1596</h1>" in chapter
    assert "<p class=\"first\">La odisea de los marineros" in chapter
    assert "<p class=\"first\">Atrapados en el hielo artico" not in chapter


def test_epub_structure_hints_reject_false_heading_without_dropping_text(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "translated.epub"
    source.write_text(
        "Atrapados en el hielo artico, 1596. Cuerpo inicial.\n\n"
        "Cierre de carta: Roma, 16 de enero de 1645. Esto debe quedar como texto.",
        encoding="utf-8",
    )

    convert_output_file(
        source,
        destination,
        "epub",
        structure_hints={"keep_indices": [1]},
    )

    with zipfile.ZipFile(destination) as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")
        chapter = zf.read("OEBPS/chap-001.xhtml").decode("utf-8")

    assert "Atrapados en el hielo artico, 1596" in nav
    assert "Cierre de carta" not in nav
    assert "Cierre de carta" in chapter


def test_history_epub_sorts_many_dated_sections_chronologically(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "Eyewitness to History repaired.epub"
    parts = ["Introduccion"]
    for idx in range(45):
        year = 1900 + idx
        parts.append(f"Entrada tardia {idx}, {year}. Texto tardio {idx}.")
    parts.append("La peste en Atenas, 430. Texto antiguo.")
    source.write_text("\n\n".join(parts), encoding="utf-8")

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination) as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")

    assert nav.index("Introduccion") < nav.index("La peste en Atenas, 430")
    assert nav.index("La peste en Atenas, 430") < nav.index("Entrada tardia 0, 1900")


def test_non_history_epub_keeps_source_section_order(tmp_path):
    source = tmp_path / "source.txt"
    destination = tmp_path / "Novel repaired.epub"
    parts = []
    for idx in range(45):
        year = 1900 + idx
        parts.append(f"Entrada tardia {idx}, {year}. Texto tardio {idx}.")
    parts.append("La peste en Atenas, 430. Texto antiguo.")
    source.write_text("\n\n".join(parts), encoding="utf-8")

    convert_output_file(source, destination, "epub")

    with zipfile.ZipFile(destination) as zf:
        nav = zf.read("OEBPS/nav.xhtml").decode("utf-8")

    assert nav.index("Entrada tardia 0, 1900") < nav.index("La peste en Atenas, 430")
