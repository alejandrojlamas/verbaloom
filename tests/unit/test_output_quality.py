import zipfile

from src.core.output_quality import analyze_output_file


def test_analyze_output_file_reports_mexican_editorial_suggestions(tmp_path):
    output = tmp_path / "under_the_volcano.txt"
    output.write_text(
        "Pidió zumo en una localización cutre.\n\n"
        "El ordenador del chaval estaba en el coche.",
        encoding="utf-8",
    )

    report = analyze_output_file(output)

    assert report["status"] == "fail"
    assert report["text"]["paragraphs"] == 2
    assert report["mexican_spanish"]["counts"]["zumo"] == 1
    assert report["mexican_spanish"]["counts"]["cutre"] == 1
    assert report["mexican_spanish"]["counts"]["localizaciones_locaciones"] == 1
    suggestions = {item["code"]: item for item in report["glossary_suggestions"]}
    assert "jugo" in suggestions["zumo"]["suggestion"]
    assert "locaciones" in suggestions["localizaciones_locaciones"]["suggestion"]


def test_output_quality_does_not_fail_on_historical_honorific(tmp_path):
    output = tmp_path / "literary_history.txt"
    output.write_text(
        "Firmó sus poemas como Scardanelli y recibió al visitante con el "
        "tratamiento de Vuestra Alteza y Majestad.",
        encoding="utf-8",
    )

    report = analyze_output_file(output)

    assert report["status"] == "pass"
    assert report["mexican_spanish"]["total"] == 0


def test_analyze_output_file_reports_reader_visible_protocol_leaks(tmp_path):
    output = tmp_path / "book.txt"
    output.write_text(
        "Texto antes.\n\n</TRANSLATIONATION>\n\nTexto despues.",
        encoding="utf-8",
    )

    report = analyze_output_file(output)

    assert report["status"] == "warn"
    assert report["final_readability"]["samples_checked"] == 1
    assert report["final_readability"]["protocol_issues"] >= 1
    assert any("LLM protocol" in warning for warning in report["warnings"])


def test_analyze_output_file_reports_reader_artifacts(tmp_path):
    output = tmp_path / "book.txt"
    output.write_text(
        "Texto real.[[23]](../Text/notas.xhtml#nt23)\n\n"
        "[OceanofPDF.com](https://oceanofpdf.com)\n\n"
        "12.\n\n13.\n\n14.\n\nSigue.",
        encoding="utf-8",
    )

    report = analyze_output_file(output)

    assert report["status"] == "warn"
    assert report["reader_artifacts"]["source_artifacts"] >= 1
    assert report["reader_artifacts"]["standalone_page_markers"] == 3
    assert any("marcadores de paginación" in warning for warning in report["warnings"])


def test_output_quality_does_not_treat_standalone_years_as_page_markers(tmp_path):
    output = tmp_path / "history.txt"
    output.write_text(
        "1948\n\nComenzó la historia.\n\n1959\n\nContinuó la historia.\n\n1969\n\nFinal.",
        encoding="utf-8",
    )

    report = analyze_output_file(output)

    assert report["reader_artifacts"]["standalone_page_markers"] == 0


def test_output_quality_does_not_treat_epub_chapter_numbers_as_pages(tmp_path):
    epub_path = tmp_path / "numbered-chapters.epub"
    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>""",
        )
        zf.writestr(
            "content.opf",
            """<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Libro</dc:title><dc:language>es</dc:language></metadata>
<manifest><item id="c" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
<spine><itemref idref="c"/></spine></package>""",
        )
        zf.writestr(
            "chapter.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h2 class="verbaloom-section-marker">12</h2><p>Texto del capítulo.</p>
<p>37</p><p>Texto posterior.</p><p>1959</p>
</body></html>""",
        )

    report = analyze_output_file(epub_path)

    assert report["reader_artifacts"]["standalone_page_markers"] == 1


def test_analyze_output_file_inspects_epub_structure(tmp_path):
    epub_path = tmp_path / "book.epub"
    container_xml = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    opf = """<?xml version="1.0" encoding="utf-8"?>
<package version="3.0" unique-identifier="bookid" xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <metadata>
    <dc:identifier id="bookid">urn:test</dc:identifier>
    <dc:title>Libro de prueba</dc:title>
    <dc:language>es</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="chap1" href="chap-001.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chap1"/>
  </spine>
</package>
"""
    chapter = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="es">
  <body><h1>Capítulo 1</h1><p>Pidió zumo en una localización cutre.</p></body>
</html>
"""
    nav = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <body><nav epub:type="toc"><ol><li><a href="chap-001.xhtml">Capítulo 1</a></li></ol></nav></body>
</html>
"""

    with zipfile.ZipFile(epub_path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", nav)
        zf.writestr("OEBPS/chap-001.xhtml", chapter)

    report = analyze_output_file(epub_path)

    assert report["status"] == "fail"
    assert report["epub"]["valid_zip"] is True
    assert report["epub"]["has_opf"] is True
    assert report["epub"]["has_nav"] is True
    assert report["epub"]["title"] == "Libro de prueba"
    assert report["epub"]["language"] == "es"
    assert report["epub"]["content_documents"] == 1
    assert report["epub"]["spine_items"] == 1
    assert report["mexican_spanish"]["counts"]["zumo"] == 1
