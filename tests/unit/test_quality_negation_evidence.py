"""Polarity evidence must distinguish actual negation from surface spelling."""

import pytest

from src.core.quality_assurance.config import QualityAssuranceConfig
from src.core.quality_assurance.models import TranslationUnit
from src.core.quality_assurance.validators import _negation_count, validate_unit_semantics


def _semantics(source, target):
    unit = TranslationUnit.create(
        document_id="doc:negation-regression", order_index=0,
        source_text=source, source_language="English", target_language="Spanish",
        translated_text=target, final_text=target,
    )
    return validate_unit_semantics(unit, QualityAssuranceConfig())


@pytest.mark.parametrize("source,target", [
    ("Never can I forget the mournful appearance.", "Jamás podré olvidar aquel aspecto lúgubre."),
    ("No one can ever listen to that word.", "Nadie podrá jamás escuchar esa palabra."),
    ("He did not see anything.", "Nada vio."),
    ("She did not agree either.", "Ella tampoco estuvo de acuerdo."),
    ("There was not a single witness.", "Ningún testigo estaba presente."),
])
def test_spanish_negation_is_recognized_without_the_word_no(source, target):
    assert _negation_count(source) > 0
    assert _negation_count(target) > 0
    assert _semantics(source, target).passed is True


def test_archaic_concessive_never_so_much_does_not_reverse_polarity():
    source = "The Asse lyeth starke still, though hee command and beate him never so much."
    target = "El asno permanece inmóvil, por más que le dé órdenes y lo golpee."
    assert _negation_count(source) == 0
    assert _semantics(source, target).passed is True


@pytest.mark.parametrize("source", [
    "Though he was warned, he never returned.",
    "Though he was warned, he never so much as raised his voice.",
    "Though he asked never so much, he did not agree.",
    "He never so much as raised his voice.",
    "Although he was warned. He never returned.",
])
def test_genuine_lost_negation_remains_blocking(source):
    assert _negation_count(source) >= 1
    result = _semantics(source, "El hombre hizo lo que esperaba su familia y todos quedaron contentos.")
    assert result.passed is False
    assert any(issue.code == "possible_negation_loss" and issue.blocking for issue in result.issues)


def test_comparison_paraphrase_is_not_waived_by_concessive_rule():
    result = _semantics("It is not distinguishable from snow.", "Parece nieve.")
    assert any(issue.code == "possible_negation_loss" for issue in result.issues)
