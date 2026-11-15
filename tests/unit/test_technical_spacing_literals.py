"""Cleanup and final validators share literal boundaries without waiving prose."""

import re

import pytest

from src.core.technical_identifiers import matches_outside_technical_literals
from src.core.epub.publication_gate import audit_epub_publication
from src.core.quality_assurance.config import QualityAssuranceConfig
from src.core.quality_assurance.validators import validate_unit_entities, validate_unit_typography
from tests.unit.epub.test_publication_gate import _write_epub
from tests.unit.test_universal_quality_assurance import _unit


@pytest.mark.parametrize("literal", [
    "www.Example.com/Authors/First-Last",
    "HTTPS://example.com/author.Name?First=One.Next",
    "editor.Name@example.COM",
    "doi:10.12345/abc.Def",
    "../Project.One/ReadMe",
])
def test_typography_keeps_literal_data_but_rejects_adjacent_prose_join(literal):
    config = QualityAssuranceConfig()
    unit = _unit(f"Consult {literal} for additional information.", f"Consulte {literal} para obtener más información.")
    assert validate_unit_typography(unit, config).passed
    unit.final_text += " Volvió.Después se fue."
    result = validate_unit_typography(unit, config)
    assert not result.passed
    assert any(issue.code == "spacing_period_join" for issue in result.issues)


def test_literal_boundaries_and_crossing_matches_remain_outside():
    value = "www.Example.org"
    assert [m.start() for m in matches_outside_technical_literals(re.compile(r"(?=w)|(?=E)|$"), value)] == [0, len(value)]
    value = "A HTTPS://Example.com"
    assert [m.group() for m in matches_outside_technical_literals(re.compile(r" HTTPS"), value)] == [" HTTPS"]


def test_url_entity_mismatch_is_not_waived_by_typography_protection():
    unit = _unit(
        "Visit www.Example.com/Authors/First-Last for details.",
        "Visite www. Example.com/Authors/First-Last para obtener detalles.",
    )
    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())
    assert not result.passed
    assert any(issue.code == "url_mismatch" and issue.blocking for issue in result.issues)


@pytest.mark.parametrize("joined_prose", [False, True])
def test_epub_publication_checks_spacing_outside_mixed_case_url(tmp_path, joined_prose):
    source, output = tmp_path / "source.epub", tmp_path / "output.epub"
    literal = "www.Example.com/Authors/First-Last"
    source_text = f"Visit {literal} for additional information about the author's published works. Then return home."
    output_text = f"Visite {literal} para obtener más información sobre las obras publicadas por el autor. Después vuelva a casa."
    if joined_prose:
        output_text = output_text.replace("autor. Después", "autor.Después")
    _write_epub(source, language="en", html_language="en", paragraphs=[source_text])
    _write_epub(output, language="es", html_language="es", paragraphs=[output_text])
    report = audit_epub_publication(
        source, output, source_language="English", target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )
    assert report.publishable is not joined_prose, report.errors
    assert bool(report.spacing_findings) == joined_prose
