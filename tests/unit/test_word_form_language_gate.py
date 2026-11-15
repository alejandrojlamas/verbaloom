"""Word-form examples must not be confused with untranslated prose."""

from dataclasses import replace

import pytest

from src.core.fidelity_supervisor import (
    _is_translated_word_form_enumeration,
    assess_fidelity,
    recheck_saved_language_rejection,
    target_language_gate_issues,
)

SOURCE = (
    "The publisher removed the old conventions. Such forms as centre, to-day, "
    "up-stairs, and criticising were replaced with the American word forms: "
    "center, today, upstairs, and criticizing. The explanation describes the editorial policy."
)
TARGET = (
    "La editorial eliminó las convenciones antiguas. Formas como centre, to-day, "
    "up-stairs y criticising se sustituyeron por las formas estadounidenses: "
    "center, today, upstairs y criticizing. La explicación describe la política editorial."
)
OPTIONS = {
    "audit_entire_book": True, "permit_unaudited_segments": False,
    "fidelity_supervisor_mode": "strict_full",
}


def issues(source=SOURCE, candidate=TARGET, options=None):
    return target_language_gate_issues(
        source, candidate, source_language="English", target_language="Spanish",
        prompt_options=OPTIONS if options is None else options,
    )


def test_cited_forms_are_warnings_for_required_contextual_audit():
    result = issues()
    assert not [issue for issue in result if issue.severity == "reject"]
    assert any(issue.code == "source_language_residual" for issue in result)


def test_saved_exact_candidate_can_reuse_favorable_semantic_audit():
    options = dict(source_language="English", target_language="Spanish", prompt_options=OPTIONS)
    current = assess_fidelity(SOURCE, TARGET, chunk_index=1, phase="final_epub_unit_audit", **options)
    assert current.accepted
    prior = replace(
        current, accepted=False, local_accepted=False, judge_decision="pass", judge_confidence=0.99,
        issues=[replace(i, severity="reject") if i.code == "source_language_residual" else i for i in current.issues],
    )
    assert recheck_saved_language_rejection(prior, SOURCE, TARGET, **options).accepted
    assert recheck_saved_language_rejection(prior, SOURCE, TARGET + " Texto añadido.", **options) is None
    assert recheck_saved_language_rejection(replace(prior, judge_decision="fail"), SOURCE, TARGET, **options) is None
    assert recheck_saved_language_rejection(
        replace(prior, judge_changed_facts=["Changed the date"]), SOURCE, TARGET, **options,
    ) is None


def test_examples_do_not_excuse_untranslated_neighboring_prose():
    source = SOURCE + " The workers must return to the factory before dawn."
    candidate = TARGET + " The workers must return to the factory before dawn."
    assert any(issue.severity == "reject" for issue in issues(source, candidate))


@pytest.mark.parametrize("source,candidate", [
    (
        "They visited the schools and the gardens before returning home.",
        "Visitaron the schools and the gardens antes de volver a casa.",
    ),
    (
        'The chapter on spelling quotes: "The workers must return to the factory before dawn."',
        'El capítulo de ortografía cita: "The workers must return to the factory before dawn."',
    ),
    (
        "NOTE ON THE TEXT. The workers must return to the factory before dawn.",
        "NOTA SOBRE EL TEXTO. The workers must return to the factory before dawn.",
    ),
])
def test_plain_prose_lists_quotations_and_headings_keep_language_gate(source, candidate):
    assert any(issue.severity == "reject" for issue in issues(source, candidate))


def test_word_forms_need_contextual_audit_and_translated_framing():
    assert any(issue.severity == "reject" for issue in issues(options={}))
    assert any(issue.severity == "reject" for issue in issues(candidate=SOURCE))


@pytest.mark.parametrize("mode", ["off", "local"])
def test_word_forms_cannot_bypass_disabled_semantic_review(mode):
    options = {**OPTIONS, "fidelity_supervisor_mode": mode}
    assert any(issue.severity == "reject" for issue in issues(options=options))


@pytest.mark.parametrize("source", [
    "The landscape forms houses, streets, gardens, and markets in the valley.",
    "Such forms as houses, streets, gardens, and markets appear in the architect's design.",
    "The spelling lesson describes houses, streets, gardens, and markets in the valley.",
    "The spelling lesson's illustrations depict such forms as houses, streets, gardens, and markets.",
])
def test_untranslated_list_needs_a_word_form_introduction(source):
    candidate = "El paisaje contiene houses, streets, gardens y mercados en el valle junto a la ciudad."
    assert not _is_translated_word_form_enumeration(
        "houses streets gardens", source=source, candidate=candidate, target_language="Spanish",
    )


def test_explicit_glossary_translation_is_not_overridden():
    options = {**OPTIONS, "glossary_terms": {"centre": "centro", "centre to-day up-stairs": "formas traducidas"}}
    assert any(issue.severity == "reject" for issue in issues(options=options))


@pytest.mark.parametrize("term,translation", [
    ("unseen phrase", "frase no presente"),
    ("editorial policy", "política editorial"),
])
def test_unrelated_or_already_translated_glossary_phrase_keeps_word_form_review(term, translation):
    options = {**OPTIONS, "glossary_terms": {term: translation}}
    result = issues(options=options)
    assert not [issue for issue in result if issue.severity == "reject"]
    assert any(issue.code == "source_language_residual" for issue in result)


def test_actual_untranslated_glossary_phrase_still_blocks_word_forms():
    options = {**OPTIONS, "glossary_terms": {"centre, to-day": "centro, hoy"}}
    assert any(issue.severity == "reject" for issue in issues(options=options))
