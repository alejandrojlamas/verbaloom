"""Numeric notation equivalence must not waive changed or missing values."""

import pytest

from src.core.quality_assurance.config import QualityAssuranceConfig
from src.core.quality_assurance.models import TranslationUnit
from src.core.quality_assurance.validators import validate_unit_entities


def compare(source, target, source_language="English"):
    unit = TranslationUnit.create(
        document_id="numeric-test", order_index=0, source_text=source,
        source_language=source_language, target_language="Spanish",
        translated_text=target, final_text=target,
    )
    result, details = validate_unit_entities(unit, QualityAssuranceConfig())
    return result, details


@pytest.mark.parametrize("source,target", [
    ("On the 24th of October.", "El 24 de octubre."),
    ("The 92nd regiment.", "El 92.º regimiento."),
    ("In the 4th year.", "En el cuarto año."),
    ("The fourth regiment.", "El 4.º regimiento."),
    ("He served in the years 1864-5.", "Sirvió en los años 1864-65."),
    ("There were 5000 people.", "Había 5,000 personas."),
    ("There were 5,000 people.", "Había 5000 personas."),
    ("It happened on zi October 1805.", "Ocurrió el 21 de octubre de 1805."),
    ("It happened on iz December 1940.", "Ocurrió el 12 de diciembre de 1940."),
    ("It happened on z June 1915.", "Ocurrió en z de junio de 1915."),
    ("It happened on i z December 1901.", "Ocurrió en i z de diciembre de 1901."),
    ("After the speech, 1 went home.", "Después del discurso, fui a casa."),
    ("In 1586,1 went home.", "En 1586,1 fui a casa."),
    ("It cost £2.1 Os.", "Costó £2.10s."),
    ("It was 2 o'clock.", "Eran las dos."),
    ("The twelve divisions.", "Las 12 divisiones."),
    ("One doctor and two nurses.", "Un médico y 2 enfermeras."),
])
def test_equivalent_numeric_surfaces_are_not_changed_values(source, target):
    result, detail = compare(source, target)
    assert result.passed
    assert detail["number"]["source"] == detail["number"]["target"]
    assert detail["currency"]["source"] == detail["currency"]["target"]


@pytest.mark.parametrize("source,target", [
    ("On the 24th of October.", "El 25 de octubre."),
    ("On the 24th of October.", "En octubre."),
    ("The 4th regiment.", "El quinto regimiento."),
    ("The 24th arrived; the 24th departed.", "El 24 llegó y se marchó."),
    ("He served in 1864-5.", "Sirvió en 1864-66."),
    ("Calculate 1864-5.", "Calcula 1864-65."),
    ("Calculate in 1864-5.", "Calcula en 1864-65."),
    ("There were 5000 people.", "Había 5,001 personas."),
    ("It happened on zi October 1805.", "Ocurrió el 22 de octubre de 1805."),
    ("It happened on 0 October 1805.", "Ocurrió el 30 de octubre de 1805."),
    ("He walked 1 mile.", "Caminó millas."),
    ("1 person was waiting.", "Había personas esperando."),
    ("There were 4 samples.", "Llegó al cuarto piso."),
    ("There was 1 patient.", "Llegó el primero."),
    ("It cost £2.1 Os.", "Costó £2.11s."),
    ("It cost £2.10.", "Costó £2.10s."),
    ("In 1913 a specimen was found.", "En 1931 se encontró un ejemplar."),
    ("There were 12 samples.", "Había trece muestras."),
    ("There were twelve samples.", "Había 13 muestras."),
    ("There were 12 samples.", "Había muestras."),
    ("Two nurses arrived.", "Dos enfermeras y 2 médicos llegaron."),
    ("The twenty fourth regiment.", "El 4.º regimiento."),
    ("The twenty-fourth regiment.", "El 4.º regimiento."),
    ("The twenty - fourth regiment.", "El 4.º regimiento."),
    ("The 20th regiment.", "El vigésimo segundo regimiento."),
    ("The 1st floor.", "El vigésimo primero piso."),
    ("The first guard arrived.", "Llegó el primer guardia y el 1.º oficial."),
    ("The first guard and the 1st officer arrived.", "Llegó el primer guardia."),
])
def test_real_numeric_changes_and_omissions_remain_blocking(source, target):
    result, detail = compare(source, target)
    assert not result.passed
    assert any(issue.blocking and issue.code in {"number_mismatch", "currency_mismatch"} for issue in result.issues)


def test_numeric_normalization_does_not_turn_un_into_a_cardinal():
    result, _ = compare("There was 1 patient.", "Había un edificio.")
    assert not result.passed


def test_ocr_pronoun_contract_requires_english():
    result, _ = compare("After the speech, 1 went home.", "Después del discurso, fui a casa.", "French")
    assert not result.passed


def test_historical_sterling_retains_pounds_and_shillings_separately():
    result, detail = compare("It cost £2.1 Os.", "Costó £2.10s.")
    assert result.passed
    assert detail["number"]["source"] == {"2": 1, "10": 1}
    assert detail["currency"]["source"] == {"GBP:2L:10S": 1}


def test_numeric_normalization_never_changes_source_or_exported_text():
    source = "On the 24th I paid £2.1 Os. for twelve items."
    target = "El 24 pagué £2.10s. por 12 artículos."
    unit = TranslationUnit.create(
        document_id="immutable-numeric-test", order_index=0, source_text=source,
        source_language="English", target_language="Spanish",
        translated_text=target, final_text=target,
    )
    before = unit.to_dict()
    result, _ = validate_unit_entities(unit, QualityAssuranceConfig())
    assert result.passed
    assert unit.to_dict() == before
    assert unit.source_text is source
    assert unit.translated_text is target
    assert unit.final_text is target


@pytest.mark.parametrize("source,language", [
    ("two hundred", "English"), ("two thousand", "English"),
    ("two million", "English"), ("two dozen", "English"),
    ("two score", "English"), ("one hundred and two", "English"),
    ("twenty-two", "English"), ("forty two", "English"),
    ("ciento dos", "Spanish"), ("doscientos dos", "Spanish"),
    ("dos mil", "Spanish"), ("dos millones", "Spanish"),
    ("treinta y dos", "Spanish"), ("veinte y dos", "Spanish"),
    ("twenty - two", "English"), ("two and a half", "English"),
    ("two point five", "English"), ("two thirds", "English"),
    ("two trillion", "English"), ("dos y medio", "Spanish"),
    ("dos coma cinco", "Spanish"), ("dos terceras partes", "Spanish"),
])
def test_compound_cardinals_never_match_only_one_component(source, language):
    result, _ = compare(source + " samples", "2 muestras", source_language=language)
    assert not result.passed
