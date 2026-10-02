from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import zipfile

from PIL import Image

import src.core.epub.publication_gate as publication_gate_module
from src.core.epub.publication_gate import (
    audit_epub_publication,
    normalize_epub_language_metadata,
)


def _write_epub(
    path,
    *,
    language: str,
    html_language: str,
    paragraphs: list[str],
    chapter_name: str = "chapter.xhtml",
    broken_link=False,
    probable_cover=False,
    declare_cover=False,
    guide_cover=False,
    paragraph_attributes: list[str] | None = None,
    body_markup: str | None = None,
):
    href = "missing.xhtml#note" if broken_link else f"{chapter_name}#note"
    attributes = paragraph_attributes or [""] * len(paragraphs)
    body = body_markup if body_markup is not None else "".join(
        f"<p{attribute}>{text}</p>"
        for text, attribute in zip(paragraphs, attributes)
    )
    chapter = f'''<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml" lang="{html_language}" xml:lang="{html_language}">
      <body><img src="cover.jpg" alt="cover"/>{body}<p id="note"><a href="{href}">Nota</a></p></body>
    </html>'''.encode()
    container = b'''<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
      <rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>
    </container>'''
    cover_meta = '<meta name="cover" content="cover"/>' if declare_cover else ""
    cover_properties = ' properties="cover-image"' if declare_cover else ""
    cover_guide = (
        f'<guide><reference type="cover" title="Cover" href="{chapter_name}"/></guide>'
        if guide_cover else ""
    )
    opf = f'''<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
      <metadata><dc:title>Test</dc:title><dc:language>{language}</dc:language>{cover_meta}</metadata>
      <manifest>
        <item id="chapter" href="{chapter_name}" media-type="application/xhtml+xml"/>
        <item id="cover" href="cover.jpg" media-type="image/jpeg"{cover_properties}/>
      </manifest>
      <spine><itemref idref="chapter"/></spine>
      {cover_guide}
    </package>'''.encode()
    cover_bytes = b"same-image-bytes"
    if probable_cover:
        buffer = BytesIO()
        Image.new("RGB", (600, 900), "navy").save(buffer, format="JPEG")
        cover_bytes = buffer.getvalue()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr(chapter_name, chapter)
        archive.writestr("cover.jpg", cover_bytes)


def _write_coverless_source_and_finished_output(source: Path, output: Path) -> None:
    container = b'''<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
      <rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>
    </container>'''
    source_opf = b'''<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
      <metadata><dc:title>Testbuch</dc:title><dc:language>de</dc:language><meta name="cover" content="missing-cover"/></metadata>
      <manifest><item id="missing-cover" href="images/missing-cover.jpg" media-type="image/jpeg" properties="cover-image"/><item id="missing-cover-page" href="missing-cover.xhtml" media-type="application/xhtml+xml"/><item id="title" href="title.xhtml" media-type="application/xhtml+xml"/><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>
      <spine><itemref idref="missing-cover-page"/><itemref idref="title"/><itemref idref="chapter"/></spine>
      <guide><reference type="cover" title="Cover" href="missing-cover.xhtml"/></guide>
    </package>'''
    output_opf = b'''<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
      <metadata><dc:title>Libro de prueba</dc:title><dc:language>es</dc:language><meta name="cover" content="verbaloom-cover-image"/></metadata>
      <manifest>
        <item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
        <item id="verbaloom-cover-image" href="images/verbaloom-cover.jpg" media-type="image/jpeg" properties="cover-image"/>
        <item id="verbaloom-cover-page" href="verbaloom-cover.xhtml" media-type="application/xhtml+xml"/>
        <item id="verbaloom-navigation" href="verbaloom-nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
      </manifest>
      <spine><itemref idref="verbaloom-cover-page"/><itemref idref="title"/><itemref idref="chapter"/></spine>
      <guide><reference type="cover" title="Portada" href="verbaloom-cover.xhtml"/></guide>
    </package>'''
    source_chapter = b'''<html xmlns="http://www.w3.org/1999/xhtml" lang="de" xml:lang="de"><body><h1>Kapitel Eins</h1><p>Am Morgen begann die lange Reise durch die Landschaft und alle erinnerten sich an das alte Haus.</p></body></html>'''
    output_chapter = '''<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="es" xml:lang="es"><body class="verbaloom-chapter" epub:type="chapter"><h1 id="ch-1" class="verbaloom-chapter-title">Capítulo uno</h1><p>Por la mañana comenzó el largo viaje por el paisaje y todos recordaron la casa antigua.</p></body></html>'''.encode()
    source_title = b'''<html xmlns="http://www.w3.org/1999/xhtml" lang="de" xml:lang="de"><body><section><h1>Testbuch</h1></section></body></html>'''
    output_title = b'''<html xmlns="http://www.w3.org/1999/xhtml" lang="es" xml:lang="es"><body><section><h1>Libro de prueba</h1></section></body></html>'''
    cover_page = b'''<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="es" xml:lang="es"><head><meta name="viewport" content="width=device-width, initial-scale=1.0"/></head><body epub:type="cover"><img src="images/verbaloom-cover.jpg" alt="Libro de prueba"/></body></html>'''
    nav = '''<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="es" xml:lang="es"><body><nav epub:type="toc"><ol><li><a href="chapter.xhtml#ch-1">Capítulo uno</a></li></ol></nav></body></html>'''.encode()
    cover_buffer = BytesIO()
    Image.new("RGB", (1600, 2560), "navy").save(cover_buffer, format="JPEG")

    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", source_opf)
        archive.writestr("title.xhtml", source_title)
        archive.writestr("chapter.xhtml", source_chapter)
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", output_opf)
        archive.writestr("title.xhtml", output_title)
        archive.writestr("chapter.xhtml", output_chapter)
        archive.writestr("verbaloom-cover.xhtml", cover_page)
        archive.writestr("verbaloom-nav.xhtml", nav)
        archive.writestr("images/verbaloom-cover.jpg", cover_buffer.getvalue())


def test_publication_gate_accepts_complete_structurally_preserved_translation(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="de",
        html_language="de",
        paragraphs=[
            "Am folgenden Morgen gingen wir weiter durch die Landschaft und erinnerten uns an die lange Reise.",
            "Das Licht lag still über den Feldern, während der Zug langsam an der Küste entlangfuhr.",
            "Il faut surtout pardonner à ces âmes malheureuses qui ont choisi de faire le pèlerinage à pied.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "A la mañana siguiente continuamos por el paisaje y recordamos el largo viaje que habíamos emprendido.",
            "La luz descansaba sobre los campos mientras el tren avanzaba lentamente a lo largo de la costa.",
            "Il faut surtout pardonner à ces âmes malheureuses qui ont choisi de faire le pèlerinage à pied.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0
    assert report.source_units == report.audited_units
    assert report.preserved_images == 1
    assert report.broken_links == 0


def test_publication_gate_accepts_bounded_professional_cover_and_navigation_assets(tmp_path):
    source = tmp_path / "source-no-cover.epub"
    output = tmp_path / "output-professional.epub"
    _write_coverless_source_and_finished_output(source, output)

    report = audit_epub_publication(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.cover_image == "images/verbaloom-cover.jpg"
    assert report.cover_page == "verbaloom-cover.xhtml"
    assert any("professional publication assets added" in warning for warning in report.warnings)


def test_publication_gate_accepts_bounded_blockquote_flow_repair(tmp_path):
    source = tmp_path / "source-blockquote.epub"
    output = tmp_path / "output-blockquote.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[],
        body_markup=(
            '<blockquote><a href="chapter.xhtml#note">The World and Its '
            "People</a></blockquote>"
        ),
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[],
        body_markup=(
            '<blockquote><div class="verbaloom-blockquote-flow-repair">'
            '<a href="chapter.xhtml#note">El mundo y sus habitantes</a>'
            "</div></blockquote>"
        ),
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert not any("element structure changed" in item for item in report.errors)
    assert report.dom_boundary_structural_mismatches == 0


def test_publication_gate_rejects_unproven_added_div(tmp_path):
    source = tmp_path / "source-div.epub"
    output = tmp_path / "output-div.epub"
    source_text = (
        "They crossed the valley at dawn and continued toward the distant city."
    )
    output_text = (
        "Cruzaron el valle al amanecer y siguieron hacia la ciudad distante."
    )
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[],
        body_markup=f"<p>{source_text}</p>",
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[],
        body_markup=f"<div><p>{output_text}</p></div>",
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert any("element structure changed" in item for item in report.errors)


def test_publication_gate_finds_homebrew_epubcheck_without_service_path(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=["The complete source paragraph is ready for translation."],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["El párrafo completo está listo para su publicación."],
    )
    monkeypatch.setattr(publication_gate_module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        publication_gate_module,
        "_KNOWN_EPUBCHECK_PATHS",
        (Path("/usr/bin/true"),),
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
    )

    assert not any(
        "EPUBCheck is not installed" in warning for warning in report.warnings
    )


def test_publication_gate_records_explicitly_sanitized_source_artifacts(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "The author wrote many books and now lives in San Francisco.",
            "www.example.com/author",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "El autor escribió muchos libros y ahora vive en San Francisco.",
            "",
        ],
        paragraph_attributes=[
            "",
            ' class="verbaloom-sanitized-artifact" style="display: none"',
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.units[0]["intentional_exclusions"] == [
        "www.example.com/author"
    ]


def test_publication_gate_blocks_source_language_bad_metadata_and_broken_link(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    german = [
        "Am folgenden Morgen gingen wir weiter durch die Landschaft und erinnerten uns an die lange Reise.",
        "Das Licht lag still über den Feldern, während der Zug langsam an der Küste entlangfuhr.",
    ]
    _write_epub(source, language="de", html_language="de", paragraphs=german)
    _write_epub(
        output,
        language="sp",
        html_language="de",
        paragraphs=german,
        broken_link=True,
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.source_language_units >= 1
    assert report.broken_links == 1
    assert any("dc:language" in error for error in report.errors)
    assert any("lang/xml:lang" in error for error in report.errors)


def test_publication_gate_warns_for_broken_links_inherited_from_source(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="de",
        html_language="de",
        paragraphs=[
            "Am folgenden Morgen gingen wir weiter durch die Landschaft und erinnerten uns an die Reise."
        ],
        broken_link=True,
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "A la mañana siguiente continuamos por el paisaje y recordamos el largo viaje."
        ],
        broken_link=True,
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.broken_links == 0
    assert report.inherited_broken_links == 1
    assert any("inherited unchanged" in warning for warning in report.warnings)


def test_publication_gate_does_not_block_unchanged_epubcheck_source_errors(tmp_path, monkeypatch):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="de",
        html_language="de",
        paragraphs=["Eine vollständige Quellzeile mit genügend Text für die Prüfung."],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["Una línea traducida completa con suficiente texto para la revisión."],
    )

    def fake_run(command, **kwargs):
        path = command[-1]
        name = "source.epub" if str(path).endswith("source.epub") else "output.epub"
        return SimpleNamespace(
            stdout=(
                f"ERROR(RSC-005): /tmp/{name}/chapter.xhtml(12,7): "
                "inherited malformed attribute"
            ),
            stderr="",
            returncode=1,
        )

    monkeypatch.setattr("src.core.epub.publication_gate.subprocess.run", fake_run)
    report = audit_epub_publication(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["epubcheck"],
    )

    assert report.publishable is True, report.errors
    assert report.epubcheck_errors == 0
    assert report.inherited_epubcheck_errors == 1


def test_publication_gate_blocks_source_phrase_inside_target_language_paragraph(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "Sheila and Daniel arrived at the restaurant twenty minutes before the meeting, "
            "then asked for a quiet table where nobody could overhear them.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Sheila and Daniel arrived at el restaurante veinte minutos antes de la reunión, "
            "y luego pidieron una mesa tranquila donde nadie pudiera oírlos.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.mixed_language_units == 1
    assert any("source-language residue" in error for error in report.errors)


def test_publication_gate_preserves_embedded_third_language_dialogue(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "His father laughed and said, “O rapaz que fala com peixes,” before explaining "
            "that everyone called him the boy who spoke with fish.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Su padre se rio y dijo: «O rapaz que fala com peixes», antes de explicar "
            "que todos lo llamaban el chico que hablaba con los peces.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.mixed_language_units == 0


def test_publication_gate_preserves_unquoted_latin_liturgical_phrases(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "This done, she knelt and recited In Te Domine confido non "
            "confundar in aeternum before repeating In manus tuas Domine."
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Hecho esto, se arrodilló y recitó In Te Domine confido non "
            "confundar in aeternum antes de repetir In manus tuas Domine."
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.mixed_language_units == 0


def test_publication_gate_preserves_legacy_bibliography_and_name_index(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    records = [
        "Anón. («The Arrest of the Catholic Priest Edmund Campion and his "
        "Associates»), George Elliot, en Arber, English Gamer, 1877",
        "City of London Letter-books, de H. T. Riley (sel. y trad.), "
        "Memorials of London Life AD 1276-1419, 1868",
        "Oates, capitán Lawrence, 428 Oberstein, conde d’, 118 "
        "Orleans, duque de, 76 Pack, mayor general Sir Denis, 288",
    ]
    _write_epub(
        source,
        language="en",
        html_language="en",
        chapter_name="main-13.xhtml",
        paragraphs=records,
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        chapter_name="main-13.xhtml",
        paragraphs=records,
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0


def test_publication_gate_preserves_foreign_title_and_boolean_query(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "Toninho e as Toninhas sounded like the title of a children's song.",
            "DEFINE dolphin AND behavior OR action AND strange OR unusual",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Toninho e as Toninhas sonaba como el título de una canción infantil.",
            "DEFINE dolphin AND behavior OR action AND strange OR unusual",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.mixed_language_units == 0


def test_publication_gate_preserves_work_titles_inside_translated_prose(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "The Day of the Locust follows a young designer, while A Place in "
            "the Sun comes from Dreiser's novel. Out of the Past and You’re a "
            "Big Boy Now remain central to the argument.",
            "On Frank Capra, see Joseph McBride, Frank Capra: The Catastrophe "
            "of Success (1992); see also The Name Above the Title (1971).",
            "Jason Calacanis produces the podcast This Week in Startups every "
            "week and has built a considerable audience.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "The Day of the Locust sigue a un joven diseñador, mientras que "
            "A Place in the Sun proviene de la novela de Dreiser. Out of the Past "
            "y You’re a Big Boy Now siguen siendo centrales para el argumento.",
            "Sobre Frank Capra, véase Joseph McBride, Frank Capra: The Catastrophe "
            "of Success (1992); véase también The Name Above the Title (1971).",
            "Jason Calacanis produce el podcast This Week in Startups todas las "
            "semanas y ha construido una audiencia considerable.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0


def test_publication_gate_preserves_numbered_bibliographic_notes(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        chapter_name="Notes.xhtml",
        paragraphs=[
            "1 . Jeffrey Pfeffer, Power in Organizations, Marshfield, MA: "
            "Pitman, 1981.",
            "2 . Amy Cuddy, “Your Body Language May Shape Who You Are,” TED, "
            "June 2012.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        chapter_name="Notes.xhtml",
        paragraphs=[
            "1 . Jeffrey Pfeffer, Power in Organizations, Marshfield, MA: "
            "Pitman, 1981.",
            "2 . Amy Cuddy, “Your Body Language May Shape Who You Are,” TED, "
            "junio de 2012.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0


def test_publication_gate_accepts_translated_glossary_with_identity_data(tmp_path):
    source = tmp_path / "source-glossary.epub"
    output = tmp_path / "output-glossary.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "GLOSSARY PRONUNCIATION KEY: a as in cat; ah as in father.",
            "Acastus ( a-kas-tus ): king of Dulichium. 14.340.",
            "Achaean ( a-kee-an ): inhabitants of Achaea. 1.272.",
            "Aretias ( a-ree-tee-as ): grandfather of Amphinomus. 18.414.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "GLOSARIO Y CLAVE DE PRONUNCIACIÓN: a como en cat; ah como en father.",
            "Acasto ( a-kas-tus ): rey de Duliquio. 14.340.",
            "Aqueo ( a-kee-an ): habitantes de Acaya. 1.272.",
            "Aretias ( a-ree-tee-as ): abuelo de Anfínomo. 18.414.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0


def test_publication_gate_rejects_untranslated_prose_inside_notes_file(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    untranslated = (
        "The editor explains why this argument matters and how the evidence "
        "changes the interpretation of the complete chapter."
    )
    _write_epub(
        source,
        language="en",
        html_language="en",
        chapter_name="Notes.xhtml",
        paragraphs=[untranslated],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        chapter_name="Notes.xhtml",
        paragraphs=[untranslated],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.source_language_units >= 1


def test_publication_gate_preserves_titles_after_translated_note_locators(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "We kept working: Erin Woo, Anissa Gardizy, and Amir Efrati, "
            "OpenAI Optimistic It Can Bring Back Sam Altman, Greg Brockman, "
            "The Information, November 18, 2023.",
            "What did Altman think? Sarah Krouse, Deepa Seetharaman, and Joe "
            "Flint, Behind the Scenes of Scarlett Johansson’s Battle with "
            "OpenAI, Wall Street Journal, May 23, 2024.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "«Seguimos trabajando»: Erin Woo, Anissa Gardizy y Amir Efrati, "
            "«OpenAI ‘Optimistic’ It Can Bring Back Sam Altman, Greg "
            "Brockman», The Information, 18 de noviembre de 2023.",
            "¿Qué pensaba Altman? Sarah Krouse, Deepa Seetharaman y Joe Flint, "
            "«Behind the Scenes of Scarlett Johansson’s Battle with OpenAI», "
            "Wall Street Journal, 23 de mayo de 2024.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.source_language_units == 0
    assert report.mixed_language_units == 0


def test_publication_gate_rejects_untranslated_note_locator_and_citation(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    untranslated = (
        "On Frank Capra, see Joseph McBride, Frank Capra: The Catastrophe "
        "of Success (1992); see also The Name Above the Title (1971)."
    )
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[untranslated],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[untranslated],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.source_language_units >= 1


def test_publication_gate_blocks_short_source_pronoun_inside_translation(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            'On Capra, "Together we..." is discussed in the cited autobiography.',
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Sobre Capra, «Juntos we…» se analiza en la autobiografía citada.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.mixed_language_units == 1


def test_publication_gate_does_not_join_normal_spanish_function_words(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=["She spoke about him and about her while the others listened."],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["Habló de él y de ella mientras los demás escuchaban."],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.mixed_language_units == 0


def test_publication_gate_detects_ocr_spaced_source_language_leak(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "Sheila and Daniel arr i ve d at the restaurant before the scheduled meeting "
            "and requested a private table on the balcony.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "Sheila y Daniel arr i ve d at el restaurante antes de la reunión programada "
            "y pidieron una mesa privada en el balcón.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.mixed_language_units == 1


def test_publication_gate_requires_existing_probable_cover_to_be_declared(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    source_paragraph = "The opening paragraph contains enough text to require a faithful complete translation."
    output_paragraph = "El párrafo inicial contiene texto suficiente para exigir una traducción fiel y completa."
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[source_paragraph],
        probable_cover=True,
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[output_paragraph],
        probable_cover=True,
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert any("cover" in error for error in report.errors)


def test_publication_gate_accepts_declared_cover_with_valid_guide(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=["The opening paragraph contains enough text for complete translation."],
        probable_cover=True,
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["El párrafo inicial contiene texto suficiente para una traducción completa."],
        probable_cover=True,
        declare_cover=True,
        guide_cover=True,
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.cover_image == "cover.jpg"
    assert report.cover_page == "chapter.xhtml"


def test_language_normalizer_repairs_opf_and_xhtml_without_touching_images(tmp_path):
    epub = tmp_path / "book.epub"
    _write_epub(
        epub,
        language="sp",
        html_language="de",
        paragraphs=["Este texto ya está traducido y debe conservarse sin otros cambios."],
    )

    epub.chmod(0o640)
    changed = normalize_epub_language_metadata(epub, "Spanish")

    assert changed == 2
    assert os.stat(epub).st_mode & 0o777 == 0o640
    with zipfile.ZipFile(epub) as archive:
        assert b"<dc:language>es</dc:language>" in archive.read("content.opf")
        chapter = archive.read("chapter.xhtml")
        assert b'lang="es"' in chapter
        assert b'xml:lang="es"' in chapter
        assert archive.read("cover.jpg") == b"same-image-bytes"


def test_publication_gate_rejects_cross_paragraph_reflow_even_when_aggregate_text_survives(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "The first paragraph introduces the voyage and the people who undertook it.",
            "The second paragraph completes the same thought without changing the scene.",
        ],
    )
    with zipfile.ZipFile(source, "a") as archive:
        directory = zipfile.ZipInfo("META-INF/")
        directory.external_attr = 0o40775 << 16
        archive.writestr(directory, b"")
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "El primer párrafo presenta el viaje y a quienes lo emprendieron. "
            "El segundo completa la misma idea sin cambiar de escena.",
            "",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.dom_boundary_structural_mismatches >= 1
    assert any("DOM text-slot structure mismatch" in error for error in report.errors)
    assert not any("archive resource set changed" in error for error in report.errors)


def test_publication_gate_accepts_source_proven_marked_pagination_reflow(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "The narrator continued describing the long afternoon beside the harbor and everyone who was still",
            "waiting there when the fishing boats finally returned through the fog.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "El narrador siguió describiendo la larga tarde junto al puerto y a todos los que todavía esperaban allí cuando los barcos pesqueros regresaron por fin entre la niebla.",
            "",
        ],
        paragraph_attributes=[
            ' class="verbaloom-reflowed-paragraph"',
            ' class="verbaloom-merged-continuation"',
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is True, report.errors
    assert report.dom_boundary_structural_mismatches == 0
    assert report.audited_units == report.source_units


def test_publication_gate_blocks_prefix_repeated_in_adjacent_paragraph(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "Daniel stood at the rail and heard the sailors running behind him with the rope.",
            "He still had not seen Creighton and realized that the ship was turning toward the place where he fell.",
        ],
    )
    repeated_prefix = (
        "Daniel permaneció junto a la borda y oyó a los marineros que corrían "
        "detrás de él con la cuerda"
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            repeated_prefix,
            repeated_prefix
            + ", pero todavía no veía a Creighton y comprendió que el barco viraba hacia el lugar donde había caído.",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.duplicate_units == 1
    assert any("duplicate" in error for error in report.errors)


def test_publication_gate_rejects_heading_that_absorbs_neighboring_body(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "CHAPTER ONE",
            "The train crossed the valley while the passengers watched the storm approach.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=[
            "CAPÍTULO UNO El tren cruzó el valle mientras los pasajeros observaban cómo se acercaba la tormenta.",
            "",
        ],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert report.dom_boundary_structural_mismatches >= 1


def test_publication_gate_blocks_omission_hidden_by_empty_adjacent_block(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=[
            "The first paragraph contains a complete account of the expedition, its leaders, and the decision to cross the sea before winter.",
            "The second paragraph adds the storm, the loss of three boats, seventeen survivors, and their arrival on the island after nine days.",
        ],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["El primer párrafo relata la expedición.", ""],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.publishable is False
    assert any(
        "severe_length_drop" in (unit.get("failure_reason") or "")
        or "numbers_lost" in (unit.get("failure_reason") or "")
        for unit in report.units
    )


def test_publication_gate_does_not_treat_periods_in_acronyms_as_bad_spacing(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        language="en",
        html_language="en",
        paragraphs=["The U.S.A. office sent the P.O. box address to the U.T. archive."],
    )
    _write_epub(
        output,
        language="es",
        html_language="es",
        paragraphs=["La oficina de U.S.A. envió el apartado P.O. al archivo U.T."],
    )

    report = audit_epub_publication(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )

    assert report.spacing_findings == 0
