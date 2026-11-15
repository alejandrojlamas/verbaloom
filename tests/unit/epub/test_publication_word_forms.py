"""Final publication must distinguish cited word forms from leaked prose."""

import pytest

from src.core.epub import publication_gate as gate
from tests.unit.epub.test_publication_gate import _write_epub
from tests.unit.test_word_form_language_gate import SOURCE, TARGET


RESIDUE = "The workers must return to the factory before dawn."
DETAIL = (
    "fuente=English; destino=Spanish; residuos="
    "centre to-day up-stairs (1.00); center today upstairs (1.00)"
)


def test_literal_residual_is_excluded_only_with_translated_source_proof():
    source, output, masked = gate._mask_source_proven_word_forms(SOURCE, TARGET, target_language="Spanish")
    assert masked and source != SOURCE and output != TARGET
    assert "centre" not in source and "centre" not in output
    for candidate in ("", SOURCE):
        assert gate._mask_source_proven_word_forms(SOURCE, candidate, target_language="Spanish") == (SOURCE, candidate, False)


def test_literal_exclusion_does_not_hide_another_residual_in_the_same_block():
    source, output, _ = gate._mask_source_proven_word_forms(
        SOURCE + " " + RESIDUE, TARGET + " " + RESIDUE, target_language="Spanish",
    )
    assert RESIDUE in source and RESIDUE in output
    assert gate._source_residual_is_actionable(
        "residuos=The workers must return to the factory before dawn (1.00)",
        source_text=source,
        source_language="English", target_language="Spanish",
    )


def test_exact_phrase_scan_preserves_punctuation_when_proving_word_forms():
    source = "Such word forms as in-between, to-and-fro, for-the-record appeared in the manuscript."
    target = "Las formas in-between, to-and-fro, for-the-record aparecían en el manuscrito de la autora."
    options = dict(source_language="English", target_language="Spanish")
    masked_source, masked_output, masked = gate._mask_source_proven_word_forms(source, target, target_language="Spanish")
    assert masked and not gate._has_unquoted_source_phrase_leak(masked_source, masked_output, **options)
    assert gate._has_unquoted_source_phrase_leak(masked_source + " " + RESIDUE, masked_output + " " + RESIDUE, **options)
    assert gate._has_unquoted_source_phrase_leak(source, source, **options)


@pytest.mark.parametrize("case", ["translated", "same_block_leak", "next_block_leak", "later_short_leak", "untranslated"])
def test_full_epub_publication_checks_every_nonliteral_span(tmp_path, case):
    source = tmp_path / "source.epub"
    output = tmp_path / "translated.epub"
    source_paragraphs, target_paragraphs = [SOURCE], [TARGET]
    if case == "same_block_leak":
        source_paragraphs[0] += " " + RESIDUE
        target_paragraphs[0] += " " + RESIDUE
    elif case == "next_block_leak":
        source_paragraphs.append(RESIDUE)
        target_paragraphs.append(RESIDUE)
    elif case == "untranslated":
        target_paragraphs = [SOURCE]
    elif case == "later_short_leak":
        lists = ["centre, to-day, up-stairs", "center, today, upstairs", "theater, organize, recognize"]
        source_paragraphs = [
            " ".join("Word forms: " + words + "." for words in lists)
            + " The pilot looked outside. The engines stopped."
        ]
        target_paragraphs = [
            " ".join("Las formas que se citan son las siguientes: " + words + "." for words in lists)
            + " El piloto miró hacia afuera. The engines stopped."
        ]
    _write_epub(source, language="en", html_language="en", paragraphs=source_paragraphs)
    _write_epub(output, language="es", html_language="es", paragraphs=target_paragraphs)
    report = gate.audit_epub_publication(
        source, output, source_language="English", target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )
    assert report.publishable == (case == "translated"), report.errors
    assert report.missing_units == 0
    if case == "translated":
        assert report.mixed_language_units == report.source_language_units == 0
        assert report.audited_units == report.source_units
    else:
        assert report.mixed_language_units or report.source_language_units


def test_ordinary_list_is_not_exempted_by_incidental_word_form_context():
    source = "The spelling lesson describes in-between, to-and-fro, for-the-record on a map."
    target = "La lección de ortografía describe in-between, to-and-fro, for-the-record en el mapa de la ciudad."
    assert gate._has_unquoted_source_phrase_leak(
        source, target, source_language="English", target_language="Spanish",
    )


def test_literal_list_cannot_mask_an_unrelated_repeated_occurrence():
    source = "Word forms: centre, to-day, up-stairs. The visitor chose centre, to-day, up-stairs."
    target = (
        "Las formas que se citan son las siguientes: centre, to-day, up-stairs. "
        "La visitante eligió las opciones centre, to-day, up-stairs en la sala."
    )
    assert gate._mask_source_proven_word_forms(source, target, target_language="Spanish") == (source, target, False)
