import zipfile

from src.core.final_source_sample_audit import (
    audit_final_output_against_source_samples,
    final_source_sample_audit_path,
)


def _write_epub(path, *, language: str, chapter_name: str, paragraphs: list[str]):
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    chapter = f"""<?xml version="1.0" encoding="utf-8"?>
    <html xmlns="http://www.w3.org/1999/xhtml" lang="{language}" xml:lang="{language}">
      <body>{body}</body>
    </html>""".encode()
    container = b"""<?xml version="1.0"?>
    <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
      <rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles>
    </container>"""
    opf = f"""<?xml version="1.0" encoding="utf-8"?>
    <package xmlns="http://www.idpf.org/2007/opf"
      xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">
      <metadata><dc:title>Test</dc:title><dc:language>{language}</dc:language></metadata>
      <manifest>
        <item id="chapter" href="{chapter_name}" media-type="application/xhtml+xml"/>
      </manifest>
      <spine><itemref idref="chapter"/></spine>
    </package>""".encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            "application/epub+zip",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("META-INF/container.xml", container)
        archive.writestr("content.opf", opf)
        archive.writestr(chapter_name, chapter)


def test_final_source_sample_audit_passes_clean_bounded_translation(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(
        (
            "Paris kept the archive in 1492 and the city recorded the same number in 1519. "
            "The witnesses described the river, the bridge, and the council with care. "
        )
        * 12,
        encoding="utf-8",
    )
    output.write_text(
        (
            "Paris conservó el archivo en 1492 y la ciudad registró el mismo número en 1519. "
            "Los testigos describieron el río, el puente y el consejo con cuidado. "
        )
        * 12,
        encoding="utf-8",
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
    )

    assert report.clean
    assert report.samples_checked >= 1
    assert not final_source_sample_audit_path(output).exists()


def test_final_source_sample_audit_defaults_to_start_middle_end_samples(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(" ".join(f"source-{idx}" for idx in range(1200)), encoding="utf-8")
    output.write_text(" ".join(f"salida-{idx}" for idx in range(1200)), encoding="utf-8")

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
        sample_chars=900,
    )

    assert report.samples_checked == 3


def test_final_source_sample_audit_reports_protocol_and_missing_numbers(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(
        (
            "The expedition counted 1492 witnesses, 1519 records, and 1521 letters. "
            "Cortes wrote again about Tenochtitlan and Cortes repeated the warning. "
        )
        * 12,
        encoding="utf-8",
    )
    output.write_text(
        (
            "La expedición contó testigos y registros. </TRANSLATIONATION> "
            "El informe repitió la advertencia sin conservar las cifras. "
        )
        * 12,
        encoding="utf-8",
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
    )

    codes = {issue.code for issue in report.issues}
    assert not report.clean
    assert "llm_protocol_leak_cleaned" in codes
    assert "numbers_missing_in_output_sample" in codes
    assert final_source_sample_audit_path(output).exists()


def test_final_source_sample_audit_flags_empty_output(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("A complete source paragraph with readable content.", encoding="utf-8")
    output.write_text("", encoding="utf-8")

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
    )

    assert not report.clean
    assert report.error_count == 1
    assert {issue.code for issue in report.issues} == {"output_text_empty"}
    assert final_source_sample_audit_path(output).exists()


def test_final_source_sample_audit_rejects_substantial_source_language_blocks(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    german = (
        "Die Geschichte dieser langen Wanderung führte durch Landschaften und Städte. "
        "Der Erzähler erinnerte sich an Menschen, Gebäude und vergangene Ereignisse. "
    ) * 18
    spanish = (
        "La historia de esta larga caminata atravesó paisajes y ciudades. "
        "El narrador recordó personas, edificios y acontecimientos pasados. "
    ) * 8
    source.write_text(german, encoding="utf-8")
    output.write_text(spanish + "\n\n" + german, encoding="utf-8")

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="German",
        target_language="Spanish",
        write_report=False,
    )

    issue = next(
        item for item in report.issues
        if item.code == "source_language_residual_coverage"
    )
    assert issue.severity == "error"
    assert report.error_count >= 1
    assert report.source_language_blocks >= 1
    assert report.source_language_characters >= 1200


def test_final_source_sample_audit_accepts_translated_bibliography_with_english_titles(
    tmp_path,
):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    english = (
        "On Frank Capra, see Joseph McBride, Frank Capra: The Catastrophe of "
        "Success (1992); see also Bob Thomas, King Cohn: The Life and Times "
        "of Harry Cohn (1967)."
    )
    spanish = (
        "Sobre Frank Capra, véase Joseph McBride, Frank Capra: The Catastrophe "
        "of Success (1992); véase también Bob Thomas, King Cohn: The Life and "
        "Times of Harry Cohn (1967)."
    )
    source.write_text("\n\n".join([english] * 4), encoding="utf-8")
    output.write_text("\n\n".join([spanish] * 4), encoding="utf-8")

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_accepts_spanish_note_locators_with_preserved_titles(
    tmp_path,
):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(
        "It is said: Joseph Weizenbaum, ELIZA—a Computer Program for the "
        "Study of Natural Language Communication Between Man and Machine, "
        "Communications of the ACM 9, no. 1 (January 1966): 36–45.",
        encoding="utf-8",
    )
    output.write_text(
        "\n\n".join(
            [
                "«Se dice»: Joseph Weizenbaum, «ELIZA—a Computer Program for "
                "the Study of Natural Language Communication Between Man and "
                "Machine», Communications of the ACM 9, n.º 1 (enero de 1966): "
                "36–45, doi.org/10.1145/365153.365168.",
                "Mirando hacia atrás varios años después: Julian Posada, «The "
                "Coloniality of Data Work: Power and Inequality in Outsourced "
                "Data Production for Machine Learning» (tesis doctoral, "
                "University of Toronto, 2022), 1–229.",
                "The Wall Street Journal, 33, 35, 41, 69, 102, 188, 193, 212, "
                "280, 367, 384, 390–91.",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_accepts_target_language_acknowledgements_with_names(
    tmp_path,
):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(
        "Then there are other friends and facilitators: Tom Thurman, "
        "Jean-Pierre Gorin, Ken Connor, Mark Danner, John Bleasdale, "
        "Mary Pickering, Holly Goldberg Sloan, and Gary Rosen.",
        encoding="utf-8",
    )
    output.write_text(
        "Luego hay otros amigos y facilitadores: Tom Thurman, Jean-Pierre "
        "Gorin, Ken Connor, Mark Danner, John Bleasdale, Mary Pickering, "
        "Holly Goldberg Sloan y Gary Rosen.",
        encoding="utf-8",
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_rejects_untranslated_english_bibliography(
    tmp_path,
):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    english = (
        "On Frank Capra, see Joseph McBride, Frank Capra: The Catastrophe of "
        "Success (1992); see also Bob Thomas, King Cohn: The Life and Times "
        "of Harry Cohn (1967)."
    )
    source.write_text("\n\n".join([english] * 4), encoding="utf-8")
    output.write_text("\n\n".join([english] * 4), encoding="utf-8")

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    issue = next(
        item for item in report.issues
        if item.code == "source_language_residual_coverage"
    )
    assert issue.severity == "error"
    assert report.source_language_blocks == 4


def test_final_source_sample_audit_exempts_numbered_epub_references(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    references = [
        "1 . Jeffrey Pfeffer, Power in Organizations, Marshfield, MA: "
        "Pitman, 1981.",
        "2 . Amy Cuddy, “Your Body Language May Shape Who You Are,” TED, "
        "junio de 2012.",
    ]
    _write_epub(
        source,
        language="en",
        chapter_name="Notes.xhtml",
        paragraphs=references,
    )
    _write_epub(
        output,
        language="es",
        chapter_name="Notes.xhtml",
        paragraphs=references,
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_exempts_epub_further_reading_titles(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    references = [
        "Weisbord, Marvin. Discovering Common Ground: How Future Search "
        "Conferences Bring People Together to Achieve Breakthrough Innovation. "
        "San Francisco: Berrett-Koehler, 1992.",
        "Whitmore, John. Coaching for Performance: Growing Human Potential and "
        "Purpose—the Principles and Practice of Coaching and Leadership. "
        "(4th rev. ed.) London: Nicholas Brealey, 2009.",
    ]
    _write_epub(
        source,
        language="en",
        chapter_name="FurtherReading.html",
        paragraphs=references,
    )
    _write_epub(
        output,
        language="es",
        chapter_name="FurtherReading.html",
        paragraphs=references,
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_exempts_localized_epub_identity_metadata(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    source_record = (
        "Flawless consulting: a guide to getting your expertise used / "
        "Peter Block; illustrated by Janis Nowlan. — 3rd ed."
    )
    output_record = (
        "Flawless consulting: a guide to getting your expertise used / "
        "Peter Block; ilustrado por Janis Nowlan. — 3.ª ed."
    )
    _write_epub(
        source,
        language="en",
        chapter_name="copyright.html",
        paragraphs=[source_record],
    )
    _write_epub(
        output,
        language="es",
        chapter_name="copyright.html",
        paragraphs=[output_record],
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_exempts_localized_epub_other_works_entry(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    source_entry = (
        "The Abundant Community: Awakening the Power of Families and "
        "Neighborhoods, coauthored with John McKnight"
    )
    output_entry = (
        "The Abundant Community: Awakening the Power of Families and "
        "Neighborhoods, en coautoría con John McKnight"
    )
    _write_epub(
        source,
        language="en",
        chapter_name="alsoby.html",
        paragraphs=[source_entry],
    )
    _write_epub(
        output,
        language="es",
        chapter_name="alsoby.html",
        paragraphs=[output_entry],
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage"
        for issue in report.issues
    )


def test_final_source_sample_audit_rejects_untranslated_epub_notes_prose(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    untranslated = (
        "The editor explains why this argument matters and how the evidence "
        "changes the interpretation of the complete chapter."
    )
    paragraphs = [untranslated] * 12
    _write_epub(
        source,
        language="en",
        chapter_name="Notes.xhtml",
        paragraphs=paragraphs,
    )
    _write_epub(
        output,
        language="es",
        chapter_name="Notes.xhtml",
        paragraphs=paragraphs,
    )

    report = audit_final_output_against_source_samples(
        source,
        output,
        source_language="English",
        target_language="Spanish",
        write_report=False,
    )

    assert report.source_language_blocks >= 1
    assert any(
        issue.code == "source_language_residual_coverage"
        and issue.severity == "error"
        for issue in report.issues
    )
