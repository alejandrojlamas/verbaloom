from __future__ import annotations

import zipfile

from lxml import etree

from src.core.epub.paragraph_reflow import (
    REFLOW_ANCHOR_CLASS,
    REFLOW_CONTINUATION_CLASS,
    is_valid_marked_continuation,
    repair_epub_split_paragraphs,
    reflow_split_paragraphs,
)


def _root(body: str) -> etree._Element:
    return etree.fromstring(
        f'<html xmlns="http://www.w3.org/1999/xhtml"><body>{body}</body></html>'.encode()
    )


def _classes(element: etree._Element) -> set[str]:
    return set(str(element.get("class") or "").split())


def test_reflows_source_proven_plain_text_continuation_and_is_idempotent():
    source = _root(
        "<p>The narrator continued describing the long afternoon beside the harbor and the people who were still</p>"
        "<p>waiting there when the fishing boats finally returned through the fog.</p>"
    )
    output = _root(
        "<p>El narrador siguió describiendo la larga tarde junto al puerto y a las personas que todavía estaban</p>"
        "<p>esperando allí cuando los barcos pesqueros regresaron por fin entre la niebla.</p>"
    )

    first = reflow_split_paragraphs(source, output)
    second = reflow_split_paragraphs(source, output)
    paragraphs = output.xpath("//*[local-name()='p']")

    assert first.merged_continuations == 1
    assert second.merged_continuations == 0
    assert len(paragraphs) == 2
    assert "todavía estaban esperando allí" in paragraphs[0].text
    assert paragraphs[1].text is None
    assert REFLOW_ANCHOR_CLASS in _classes(paragraphs[0])
    assert REFLOW_CONTINUATION_CLASS in _classes(paragraphs[1])
    assert not any(value.startswith("tbl-") for value in _classes(paragraphs[0]))
    assert not any(value.startswith("tbl-") for value in _classes(paragraphs[1]))


def test_legacy_reflow_markers_remain_valid_for_resumed_artifacts():
    source = _root(
        "<p>The narrator continued describing the long afternoon beside the harbor and the people who were still</p>"
        "<p>waiting there when the fishing boats finally returned through the fog.</p>"
    )
    output = _root(
        '<p class="tbl-reflowed-paragraph">El narrador siguió describiendo la larga tarde junto al puerto y a las personas que todavía estaban esperando allí cuando regresaron los barcos.</p>'
        '<p class="tbl-merged-continuation"></p>'
    )
    source_paragraphs = source.xpath("//*[local-name()='p']")
    output_paragraphs = output.xpath("//*[local-name()='p']")

    assert is_valid_marked_continuation(
        source_paragraphs[0],
        source_paragraphs[1],
        output_paragraphs[1],
    )


def test_reflows_and_dehyphenates_page_split_word():
    source = _root("<p>remem-</p><p>bering the journey changed them forever.</p>")
    output = _root("<p>recor-</p><p>dar el viaje los cambió para siempre.</p>")

    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    assert output.xpath("string(//*[local-name()='p'][1])") == (
        "recordar el viaje los cambió para siempre."
    )


def test_dehyphenates_from_source_evidence_when_model_dropped_the_hyphen():
    source = _root(
        "<p>He finally re-</p>"
        "<p>membered the harbor after many years.</p>"
        "<p>Later, another witness remembered it too.</p>"
    )
    output = _root(
        "<p>Por fin lo re</p>"
        "<p>cordó después de muchos años.</p>"
        "<p>Más tarde, otro testigo también lo recordó.</p>"
    )

    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    assert output.xpath("string(//*[local-name()='p'][1])") == (
        "Por fin lo recordó después de muchos años."
    )


def test_dehyphenates_when_source_hyphen_has_one_character_ocr_noise():
    source = _root(
        "<p>The device was esti- j</p>"
        "<p>mated to arrive soon.</p>"
        "<p>Another source estimates that it will arrive.</p>"
    )
    output = _root(
        "<p>Se es</p>"
        "<p>tima que el dispositivo llegará pronto.</p>"
        "<p>Otra fuente también estima que llegará.</p>"
    )

    report = reflow_split_paragraphs(
        source,
        output,
        output_vocabulary={"estima"},
    )

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    assert output.xpath("string(//*[local-name()='p'][1])") == (
        "Se estima que el dispositivo llegará pronto."
    )


def test_dehyphenates_unmarked_split_name_when_both_books_attest_joined_form():
    source = _root(
        "<p>The acknowledgments thanked Brigitte Bor</p>"
        "<p>guss and Lloyd Borguss for their assistance.</p>"
    )
    output = _root(
        "<p>Los agradecimientos mencionaron a Brigitte Bor</p>"
        "<p>guss y a Lloyd Borguss por su ayuda.</p>"
    )

    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    assert "Brigitte Borguss" in output.xpath("string(//*[local-name()='p'][1])")


def test_epub_reflow_uses_package_wide_target_vocabulary(tmp_path):
    source_path = tmp_path / "source.epub"
    output_path = tmp_path / "output.epub"
    source_members = {
        "chapter.xhtml": (
            "<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            "<p>The device was esti- j</p><p>mated to arrive soon.</p>"
            "</body></html>"
        ),
        "attestation.xhtml": (
            "<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            "<p>Another source estimates its arrival.</p>"
            "</body></html>"
        ),
    }
    output_members = {
        "chapter.xhtml": (
            "<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            "<p>Se es</p><p>tima que el dispositivo llegará pronto.</p>"
            "</body></html>"
        ),
        "attestation.xhtml": (
            "<html xmlns='http://www.w3.org/1999/xhtml'><body>"
            "<p>Otra fuente también estima cuándo llegará.</p>"
            "</body></html>"
        ),
    }
    for path, members in (
        (source_path, source_members),
        (output_path, output_members),
    ):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            for name, payload in members.items():
                archive.writestr(name, payload)

    report = repair_epub_split_paragraphs(source_path, output_path)

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    with zipfile.ZipFile(output_path) as archive:
        chapter = etree.fromstring(archive.read("chapter.xhtml"))
    assert chapter.xpath("string(//*[local-name()='p'][1])") == (
        "Se estima que el dispositivo llegará pronto."
    )


def test_does_not_guess_target_dehyphenation_without_independent_word_evidence():
    source = _root(
        "<p>He finally re-</p>"
        "<p>membered the harbor after many years.</p>"
        "<p>A separate paragraph ends normally.</p>"
    )
    output = _root(
        "<p>Por fin decidió</p>"
        "<p>recordar el puerto después de muchos años.</p>"
        "<p>Otro párrafo termina normalmente.</p>"
    )

    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 0
    assert report.dehyphenated_continuations == 0


def test_does_not_reflow_terminal_or_uppercase_paragraph_boundaries():
    source = _root(
        "<p>This paragraph is intentionally complete and contains enough text to exceed the minimum threshold.</p>"
        "<p>another intentional paragraph begins here with a lower-case word.</p>"
        "<p>This unfinished-looking paragraph contains enough text but is followed by a proper new sentence</p>"
        "<p>Another paragraph begins with an uppercase letter.</p>"
    )
    output = _root(
        "<p>Este párrafo está completo deliberadamente y contiene texto suficiente para superar el mínimo.</p>"
        "<p>otro párrafo intencional comienza aquí con una palabra en minúscula.</p>"
        "<p>Este párrafo parece inconcluso y contiene bastante texto, pero sigue una oración nueva</p>"
        "<p>Otro párrafo comienza con una letra mayúscula.</p>"
    )

    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 0
    assert all(node.text for node in output.xpath("//*[local-name()='p']"))


def test_does_not_reflow_semantically_addressable_paragraphs():
    source = _root(
        '<p id="first">This long source paragraph has an identifier and therefore must remain independently addressable</p>'
        "<p>even though its wording otherwise resembles a continuation across a page.</p>"
    )
    output = _root(
        '<p id="first">Este párrafo largo tiene un identificador y debe seguir siendo accesible de forma independiente</p>'
        "<p>aunque su redacción parezca una continuación entre páginas.</p>"
    )

    before = etree.tostring(output)
    report = reflow_split_paragraphs(source, output)

    assert report.merged_continuations == 0
    assert etree.tostring(output) == before


def test_reflows_safe_inline_anchor_without_flattening_markup():
    source = _root(
        "<p>This long paragraph contains <sup>1</sup> a footnote marker and ends with a word that was split as remem-</p>"
        "<p>bering the journey changed them forever.</p>"
    )
    output = _root(
        "<p>Este párrafo contiene <sup>1</sup> una llamada de nota y termina con una palabra partida: recor-</p>"
        "<p>dar el viaje los cambió para siempre.</p>"
    )

    report = reflow_split_paragraphs(source, output)
    first = output.xpath("//*[local-name()='p'][1]")[0]

    assert report.merged_continuations == 1
    assert report.dehyphenated_continuations == 1
    assert len(first.xpath("./*[local-name()='sup']")) == 1
    assert "recordar el viaje" in " ".join(first.itertext())
