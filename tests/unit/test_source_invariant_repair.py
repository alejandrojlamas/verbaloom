import pytest

from src.core import translator
from src.core.source_invariant_repair import repair_source_invariants


def test_restores_repeated_q_star_omissions_without_touching_existing_exact_form():
    source = (
        "An algorithm called Q*. Q* mattered. Q* improved reasoning. "
        "The Q*-related documents were restricted."
    )
    candidate = (
        "Un algoritmo llamado Q. Q importaba. Q mejoró el razonamiento. "
        "Los documentos relacionados con Q* fueron restringidos."
    )

    repaired, repairs = repair_source_invariants(source, candidate)

    assert repaired.count("Q*") == 4
    assert "llamado Q." not in repaired
    assert len(repairs) == 1
    assert repairs[0].source == "Q*"
    assert repairs[0].replacement_count == 3


def test_does_not_guess_when_source_uses_symbolic_and_bare_forms():
    source = "Compare Q* with Q before choosing the algorithm."
    candidate = "Compara Q con Q antes de elegir el algoritmo."

    repaired, repairs = repair_source_invariants(source, candidate)

    assert repaired == candidate
    assert repairs == ()


def test_does_not_rewrite_ambiguous_single_letter_words():
    source = "The grade was A*."
    candidate = "La calificación fue A."

    repaired, repairs = repair_source_invariants(source, candidate)

    assert repaired == candidate
    assert repairs == ()


def test_preserves_structural_placeholders_while_repairing():
    source = "[id0]Q* was renamed.[id1]"
    candidate = "[id0]Q fue renombrado.[id1]"

    repaired, repairs = repair_source_invariants(source, candidate)

    assert repaired == "[id0]Q* fue renombrado.[id1]"
    assert repairs[0].replacement_count == 1


@pytest.mark.asyncio
async def test_translation_request_repairs_identifier_before_caller_audits(
    monkeypatch,
):
    async def fake_request(**_kwargs):
        return (
            "Un algoritmo llamado Q. Q era importante.",
            "An algorithm called Q*. Q* was important.",
            None,
        )

    monkeypatch.setattr(
        translator,
        "_make_llm_request_with_adaptive_context",
        fake_request,
    )
    events = []

    result = await translator.generate_translation_request(
        "An algorithm called Q*. Q* was important.",
        "",
        "",
        "",
        source_language="English",
        target_language="Spanish",
        log_callback=lambda event, message, **_kwargs: events.append(
            (event, message)
        ),
    )

    assert result == "Un algoritmo llamado Q*. Q* era importante."
    assert any(event == "source_invariant_repaired" for event, _ in events)
