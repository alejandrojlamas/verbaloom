"""Incomplete citation shapes must not exempt untranslated narrative."""

import sys
import zipfile

import pytest

from src.core.document_structure import is_comma_delimited_bibliographic_record
from src.core.epub.publication_gate import audit_epub_publication
from src.core.final_source_sample_audit import audit_final_output_against_source_samples
from src.core.language_evidence import looks_like_structured_language_metadata

PROSE = [
    pytest.param(
        "John, who had never told the children why he left the house that night, "
        "was afraid of the press,",
        id="dangling-press",
    ),
    pytest.param(
        "John, who had never told the children why he left the house that night, "
        "was looking for his books,",
        id="dangling-books",
    ),
    pytest.param(
        "John, who had never told the children why he left the house that night, "
        "was interested in publishing,",
        id="dangling-publishing",
    ),
    pytest.param(
        "John, who had never told the children why he left the house that night, "
        "sent the letter to Smith, Elder Co.,",
        id="real-publisher-after-prose",
    ),
    pytest.param(
        "John, who remembered the cities where he had lived, compared 1991 Paris, "
        "1992 London, and the home that he had lost.",
        id="year-capitalized-place",
    ),
    pytest.param(
        "Bamford, Samuel, Passages in the Life of a Radical, H. Dunckley "
        "(ed.), 1893 Bayerlein, who told the children that he was afraid, "
        "never returned to the house.",
        id="valid-record-before-narrative-tail",
    ),
]

RECORDS = [
    pytest.param(
        "Anón. («The Arrest of the Catholic Priest Edmund Campion and his "
        "Associates»), George Elliot, en Arber, English Gamer, 1877",
        id="quoted-title-lead",
    ),
    pytest.param(
        "City of London Letter-books, de H. T. Riley (sel. y trad.), "
        "Memorials of London Life AD 1276-1419, 1868",
        id="hyphenated-title",
    ),
    pytest.param(
        "Froissart, Sir John, Chronicles of England, France and Spain, "
        "trad. Lord Berners, 1523-5",
        id="short-year-range",
    ),
    pytest.param(
        "Bamford, Samuel, Passages in the Life of a Radical, H. Dunckley "
        "(ed.), 1893 Bayerlein, general, («American Break-Out in Normandy",
        id="merged-truncated-records",
    ),
    pytest.param(
        "Victoria, reina, Journal of our Life in the Highlands, Smith, Elder Co.,",
        id="dangling-publisher",
    ),
]

SOURCE_CONTEXT = [
    (
        "The family returned to the village after a long journey. They opened the "
        "windows of the house and prepared the evening meal while the children "
        "played in the garden. Everyone was happy to be home again."
    ),
    (
        "The next morning the neighbors arrived with fresh bread and fruit. They "
        "sat around the kitchen table and talked about their plans for the coming "
        "winter. Outside the rain fell softly on the empty street."
    ),
]
TARGET_CONTEXT = [
    (
        "La familia regresó al pueblo después de un largo viaje. Abrieron las "
        "ventanas de la casa y prepararon la cena mientras los niños jugaban en el "
        "jardín. Todos estaban contentos de encontrarse de nuevo en su hogar."
    ),
    (
        "A la mañana siguiente los vecinos llegaron con pan fresco y fruta. Se "
        "sentaron alrededor de la mesa de la cocina y hablaron de sus planes para "
        "el próximo invierno. Afuera la lluvia caía suavemente sobre la calle vacía."
    ),
]


def _write_epub(path, language, paragraphs):
    body = "".join(f"<p>{text}</p>" for text in paragraphs)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
            'version="1.0"><rootfiles><rootfile full-path="content.opf" '
            'media-type="application/oebps-package+xml"/></rootfiles></container>',
        )
        archive.writestr(
            "content.opf",
            '<package xmlns="http://www.idpf.org/2007/opf" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0">'
            f'<metadata><dc:title>Test</dc:title><dc:language>{language}'
            '</dc:language></metadata><manifest><item id="chapter" '
            'href="chapter.xhtml" media-type="application/xhtml+xml"/></manifest>'
            '<spine><itemref idref="chapter"/></spine></package>',
        )
        archive.writestr(
            "chapter.xhtml",
            '<html xmlns="http://www.w3.org/1999/xhtml" '
            f'lang="{language}" xml:lang="{language}"><body>{body}</body></html>',
        )


def _audit_echo_in_translated_chapter(tmp_path, text):
    source, output = tmp_path / "source.epub", tmp_path / "output.epub"
    _write_epub(source, "en", [SOURCE_CONTEXT[0], text, SOURCE_CONTEXT[1]])
    _write_epub(output, "es", [TARGET_CONTEXT[0], text, TARGET_CONTEXT[1]])
    languages = {"source_language": "English", "target_language": "Spanish"}
    gate = audit_epub_publication(
        source,
        output,
        **languages,
        # Exercise the real language/coverage gates, without requiring Java
        # or pretending that this unit test validates EPUBCheck conformance.
        epubcheck_command=[sys.executable, "-c", "pass"],
    )
    audit = audit_final_output_against_source_samples(
        source, output, **languages, write_report=False,
    )
    return gate, audit


@pytest.mark.parametrize("text", PROSE)
def test_incomplete_citation_boundaries_do_not_classify_prose(text):
    assert not is_comma_delimited_bibliographic_record(text)


@pytest.mark.parametrize("text", PROSE)
def test_incomplete_citation_boundaries_do_not_exempt_language_evidence(text):
    assert not looks_like_structured_language_metadata(
        text, document_hint="chapter.xhtml", target_language="Spanish",
    )


@pytest.mark.parametrize("text", PROSE)
def test_publication_blocks_prose_echo_with_incomplete_citation_boundary(tmp_path, text):
    gate, audit = _audit_echo_in_translated_chapter(tmp_path, text)
    assert gate.publishable is False
    assert gate.source_language_units + gate.mixed_language_units >= 1
    assert audit.source_language_blocks >= 1
    assert any(issue.code == "source_language_residual_coverage" for issue in audit.issues)


@pytest.mark.parametrize("text", RECORDS)
def test_legacy_records_retain_classification_and_publication_exemption(tmp_path, text):
    assert is_comma_delimited_bibliographic_record(text)
    assert looks_like_structured_language_metadata(
        text, document_hint="chapter.xhtml", target_language="Spanish",
    )
    gate, audit = _audit_echo_in_translated_chapter(tmp_path, text)
    assert gate.publishable is True, gate.errors
    assert gate.source_language_units == 0
    assert gate.mixed_language_units == 0
    assert audit.source_language_blocks == 0
    assert not any(
        issue.code == "source_language_residual_coverage" for issue in audit.issues
    )


@pytest.mark.parametrize("publisher", [
    "Oxford University Press", "Viking Press", "Bloomsbury Publishing",
])
def test_publication_preserves_publisher_identity(tmp_path, publisher):
    gate, _audit = _audit_echo_in_translated_chapter(tmp_path, publisher)
    assert gate.publishable is True, gate.errors
    assert gate.source_language_units == 0
    assert gate.mixed_language_units == 0
