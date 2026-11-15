from dataclasses import replace

import pytest

from src.core.fidelity_supervisor import (
    assess_fidelity,
    recheck_saved_language_rejection,
    target_language_gate_issues,
)


CONTACT = "Northwind Books 12400 Ventura Blvd, Suite 1111 Studio City, CA 91604 tel. 818.555.0101 fax 818.555.0102"
PROSE = "The actors must verify the terms before they sign the contract."
TRANSLATED = "Los actores deben verificar las condiciones antes de firmar el contrato."


def _issues(source, candidate):
    return target_language_gate_issues(source, candidate, source_language="English", target_language="Spanish")


def test_unchanged_postal_contact_does_not_block_translated_credits():
    issues = _issues(
        f"{CONTACT} Cover Design: Art Studio. {PROSE}",
        f"{CONTACT} Diseño de portada: Estudio de Arte. {TRANSLATED}",
    )
    assert not [issue for issue in issues if issue.severity == "reject"]
    assert any(issue.code == "source_language_residual" and issue.severity == "warning" for issue in issues)


def test_saved_favorable_semantic_judgment_can_be_reused_for_exact_contact():
    source = f"{CONTACT} Cover Design: Art Studio. {PROSE}"
    candidate = f"{CONTACT} Diseño de portada: Estudio de Arte. {TRANSLATED}"
    options = {"source_language": "English", "target_language": "Spanish", "prompt_options": {}}
    current = assess_fidelity(source, candidate, chunk_index=1, phase="final_epub_unit_audit", **options)
    assert current.accepted
    saved = replace(
        current, accepted=False, local_accepted=False, judge_decision="pass", judge_confidence=0.99,
        issues=[
            replace(issue, severity="reject") if issue.code == "source_language_residual" else issue
            for issue in current.issues
        ],
    )
    rechecked = recheck_saved_language_rejection(saved, source, candidate, **options)
    assert rechecked is not None and rechecked.accepted


@pytest.mark.parametrize("separator", [" ", " [id0] "])
def test_contact_does_not_exempt_untranslated_neighboring_prose(separator):
    introduction = (
        "The rehearsal starts after the director arrives. "
        "Bring a pencil and make sure the room is ready."
    )
    translated = (
        "El ensayo empieza después de que llega la directora. "
        "Trae un lápiz y comprueba que la sala esté lista."
    )
    issues = _issues(
        introduction + " " + CONTACT + separator + PROSE,
        translated + " " + CONTACT + separator + PROSE,
    )
    assert any(issue.code == "source_language_residual" and issue.severity == "reject" for issue in issues)


@pytest.mark.parametrize("modified", [
    CONTACT.replace("91604", "91607"),
    CONTACT.replace("818.555.0102", "818.555.0103"),
])
def test_changed_contact_does_not_receive_postal_exception(modified):
    issues = _issues(f"{CONTACT} {PROSE}", f"{modified} {TRANSLATED}")
    assert any(issue.code == "source_language_residual" and issue.severity == "reject" for issue in issues)
