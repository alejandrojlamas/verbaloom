import pytest
from types import SimpleNamespace

from src.core.fidelity_supervisor import (
    FidelityReport,
    _is_bibliographic_registry_overlap,
    _looks_like_preservable_name_index_echo,
    _normalize_structured_index_audit_policy,
    _profile_fidelity_audit_context,
    apply_fidelity_audit_assessment,
    assess_fidelity,
    build_fidelity_audit_prompt,
    build_fidelity_retry_prompt_options,
    fidelity_supervisor_enabled,
    fidelity_report_path,
    parse_fidelity_audit_response,
    supervise_fidelity,
)
from src.core.llm.base import LLMResponse


def test_supervisor_requires_explicit_runtime_opt_in():
    assert fidelity_supervisor_enabled({}) is False
    assert fidelity_supervisor_enabled({"fidelity_supervisor_mode": "alerted"}) is True
    assert fidelity_supervisor_enabled({"fidelity_supervisor": True}) is True
    assert fidelity_supervisor_enabled({
        "fidelity_supervisor": True,
        "fidelity_supervisor_mode": "off",
    }) is False


@pytest.mark.asyncio
async def test_strict_fidelity_audit_retries_one_invalid_json_response():
    class SequencedClient:
        def __init__(self):
            self.calls = 0

        async def make_request(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(content="The passage looks faithful, but this is not JSON.")
            return LLMResponse(content=(
                '<FIDELITY_AUDIT_JSON>{"verdict":"pass","confidence":0.98,'
                '"reason":"Complete and faithful.","issues":[],'
                '"missing_from_source":[],"added_not_in_source":[],'
                '"changed_facts":[],"censored_or_softened":[],'
                '"structure_issues":[],"evidence_source":[],'
                '"evidence_candidate":[]}</FIDELITY_AUDIT_JSON>'
            ))

    client = SequencedClient()
    decision, response = await supervise_fidelity(
        "The complete sentence remains intact.",
        "La oración completa permanece intacta.",
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
        client=client,
        prompt_options={
            "fidelity_supervisor": True,
            "fidelity_supervisor_mode": "strict_full",
        },
    )

    assert client.calls == 2
    assert response is not None
    assert decision.accepted is True
    assert decision.judge_decision == "pass"


@pytest.mark.asyncio
async def test_strict_fidelity_audit_uses_third_zero_temperature_attempt():
    class SequencedClient:
        def __init__(self):
            self.calls = []

        async def make_request(self, *_args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) < 3:
                return LLMResponse(content="not valid JSON")
            return LLMResponse(content=(
                '<FIDELITY_AUDIT_JSON>{"verdict":"pass","confidence":0.98,'
                '"reason":"Complete and faithful.","issues":[],'
                '"missing_from_source":[],"added_not_in_source":[],'
                '"changed_facts":[],"censored_or_softened":[],'
                '"structure_issues":[],"evidence_source":[],'
                '"evidence_candidate":[]}</FIDELITY_AUDIT_JSON>'
            ))

    client = SequencedClient()
    decision, response = await supervise_fidelity(
        "The complete sentence remains intact.",
        "La oración completa permanece intacta.",
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
        client=client,
        prompt_options={
            "fidelity_supervisor": True,
            "fidelity_supervisor_mode": "strict_full",
        },
    )

    assert len(client.calls) == 3
    assert all(call["temperature"] == 0.0 for call in client.calls)
    assert response is not None
    assert decision.accepted is True
    assert decision.judge_decision == "pass"


def test_rejects_severe_content_drop():
    source = (
        "Barents died a week later, but most of the others survived. "
        "In 1871, the house where Barents had spent the winter was discovered, "
        "with many relics still intact. The sailors had been trapped in the ice "
        "for months and debated whether to abandon the ship, carry supplies over "
        "the floes, or wait for a change in the weather before moving south. "
        "Their testimony describes the fear, hunger, arguments, and practical "
        "decisions that shaped the expedition day by day."
    )
    decision = assess_fidelity(
        source,
        "Barents murio una semana despues.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "severe_length_drop" in {issue.code for issue in decision.rejections}


def test_rejects_truncated_candidate_for_medium_sized_source():
    source = (
        "The first paragraph contains the expedition, its leaders, and the decision to cross "
        "the sea. The second adds the storm, the loss of three boats, and seventeen survivors."
    )
    decision = assess_fidelity(
        source,
        "El primer párrafo relata la expedición.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "severe_length_drop" in {issue.code for issue in decision.rejections}


def test_rejects_changed_or_lost_numbers():
    decision = assess_fidelity(
        "The expedition began in 1596 and returned with 17 survivors.",
        "La expedicion comenzo en 1596 y regreso con sobrevivientes.",
        chunk_index=2,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "numbers_lost" in {issue.code for issue in decision.rejections}


def test_localized_clock_separator_preserves_numeric_time():
    decision = assess_fidelity(
        "The time was about 9.30 p.m.",
        "La hora era aproximadamente las 9:30 p. m.",
        chunk_index=3,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" not in {issue.code for issue in decision.rejections}
    assert "numbers_lost" not in {issue.code for issue in decision.warnings}


def test_colon_does_not_replace_decimal_without_time_context():
    decision = assess_fidelity(
        "The measurement was 9.30 millimeters.",
        "La medida era 9:30 milímetros.",
        chunk_index=3,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" in {issue.code for issue in decision.rejections}


def test_english_ocr_one_used_as_pronoun_is_not_a_lost_quantity():
    decision = assess_fidelity(
        (
            "[id0]&gt; 1 Would Learn of Events Far Distant&lt;[id1] she said. "
            "[id2]&gt; 1 Seek to Accomplish a Melding&lt;[id3] she continued. "
            "[id4]&gt; 1 Have Great Admiration for Dj\\Tal&lt;[id5] he replied."
        ),
        (
            "[id0]&gt;Desearía conocer sucesos muy distantes&lt;[id1] dijo ella. "
            "[id2]&gt;Busco realizar una fusión&lt;[id3] continuó ella. "
            "[id4]&gt;Siento gran admiración por Dj\\Tal&lt;[id5] respondió él."
        ),
        chunk_index=6,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" not in {issue.code for issue in decision.rejections}
    assert "numbers_lost" not in {issue.code for issue in decision.warnings}


def test_english_ocr_one_after_comma_is_not_a_lost_quantity():
    source = (
        "Being led through the house, 1 found an old man with a child about "
        "7 years old beside the 5 books of Moses; the rest 1 suppose were witnesses."
    )
    candidate = (
        "Al entrar en la casa, encontré a un anciano con un niño de unos 7 años "
        "junto a los 5 libros de Moisés; supongo que los demás eran testigos."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=6,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" not in {issue.code for issue in decision.rejections}
    assert "numbers_lost" not in {issue.code for issue in decision.warnings}


def test_english_ocr_one_handles_placeholders_clause_cues_and_past_verbs():
    source = (
        "1 almost wished it were night. [id0]1 went inside, where the fellows "
        "1 knew waited. When 1 warm my hands, till 1 have enough heat, then 1 "
        "notice the fire; and 1 shook off the snow. ‘It is warmer,’ 1 said. "
        "At night - 1 [id1]think - we will leave. February 1913."
    )
    candidate = (
        "Casi deseé que fuera de noche. [id0]Entré, donde esperaban los compañeros "
        "que conocía. Cuando me caliento las manos, hasta tener suficiente calor, "
        "entonces noto el fuego; y me sacudí la nieve. «Hace más calor», dije. "
        "Por la noche, creo [id1], nos iremos. Febrero de 1913."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=6,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" not in {issue.code for issue in decision.rejections}
    assert "numbers_lost" not in {issue.code for issue in decision.warnings}


def test_real_single_quantity_remains_protected_from_ocr_filter():
    decision = assess_fidelity(
        "There was 1 survivor after the storm.",
        "No hubo sobrevivientes después de la tormenta.",
        chunk_index=7,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" in {issue.code for issue in decision.rejections}


@pytest.mark.parametrize(
    "source,candidate",
    [
        (
            "Chapter 1 said nothing about the storm.",
            "El capítulo no decía nada sobre la tormenta.",
        ),
        (
            "I had 1 thought before the meeting.",
            "Tuve una idea antes de la reunión.",
        ),
        (
            "There was 1 can of food in the shelter.",
            "Había comida en el refugio.",
        ),
        (
            "1 can of food remained in the shelter.",
            "Quedaba comida en el refugio.",
        ),
    ],
)
def test_english_ocr_filter_keeps_quantities_before_verb_like_words(
    source,
    candidate,
):
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=8,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "numbers_lost" in {issue.code for issue in decision.rejections}


def test_fidelity_audit_prompt_requires_measurements_but_not_source_language_thoughts():
    prompt = build_fidelity_audit_prompt(
        "He thought, Goddamn it, as the 450-pound dolphin jumped ten feet.",
        "Pensó: Maldita sea, mientras el delfín de 450 libras saltaba diez pies.",
        source_language="English",
        target_language="Spanish",
        phase="translation",
    )

    assert "internal monologue are supposed to be translated" in prompt.system
    assert "original numeric value and unit to remain present" in prompt.system
    assert "rounded conversion is a fidelity defect" in prompt.system


def test_numeric_citations_are_not_structural_placeholders():
    decision = assess_fidelity(
        "Previous work used attention mechanisms [8] and neural GPUs [14].",
        "Trabajos previos usaron mecanismos de atención [8] y GPUs neuronales [14].",
        chunk_index=26,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "placeholder_mismatch" not in {issue.code for issue in decision.rejections}


def test_id_placeholders_still_reject_when_lost():
    decision = assess_fidelity(
        "[id0]Previous work used attention mechanisms.[id1]",
        "[id0]Trabajos previos usaron mecanismos de atención.",
        chunk_index=27,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "placeholder_mismatch" in {issue.code for issue in decision.rejections}


def test_decades_translated_with_s_suffix_preserve_number():
    decision = assess_fidelity(
        "En la década de 1990, el cine cambio.",
        "In the 1990s, cinema changed.",
        chunk_index=20,
        phase="translation",
        source_language="Spanish",
        target_language="English",
    )

    assert "numbers_lost" not in {issue.code for issue in decision.rejections}


def test_isolated_pdf_page_numbers_are_warning_not_hard_number_loss():
    decision = assess_fidelity(
        "El argumento continua en la siguiente pagina.\n76",
        "The argument continues on the next page.",
        chunk_index=21,
        phase="translation",
        source_language="Spanish",
        target_language="English",
    )

    assert decision.accepted is True
    assert "source_pdf_extraction_noise" in {issue.code for issue in decision.warnings}
    assert "numbers_lost" not in {issue.code for issue in decision.rejections}


def test_boundary_truncation_judge_failure_becomes_warning_for_pdf_split():
    decision = assess_fidelity(
        "Así, Halliwell concluye que la catarsis no es solo el resultado final de ver una\n76",
        "Thus, Halliwell concludes that catharsis is not only the final result of seeing a",
        chunk_index=22,
        phase="translation",
        source_language="Spanish",
        target_language="English",
    )
    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.98,
            "reason": "The candidate truncates the final sentence at a chunk boundary.",
            "issues": ["truncation", "incomplete_sentence"],
            "missing_from_source": ["sentence incomplete in source and continues in the next chunk"],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
            "structure_issues": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_ocr_boundary_warning" in {issue.code for issue in decision.warnings}
    assert "fidelity_judge_reject" not in {issue.code for issue in decision.rejections}


def test_judge_warn_cannot_downgrade_structured_omission():
    decision = assess_fidelity(
        "The model achieves 28.4 BLEU and 41.8 BLEU on two tasks.",
        "El modelo alcanza BLEU alto en dos tareas.",
        chunk_index=24,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    assert "numbers_lost" in {issue.code for issue in decision.rejections}

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.92,
            "reason": "The candidate is semantically related but should preserve numeric scores.",
            "issues": ["numbers should be checked"],
            "missing_from_source": ["28.4 BLEU", "41.8 BLEU"],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
            "structure_issues": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "numbers_lost" in {issue.code for issue in decision.rejections}
    assert "fidelity_judge_reject" in {issue.code for issue in decision.rejections}


def test_judge_fail_downgrades_numeric_claim_disproved_by_its_own_evidence():
    source = (
        'Kevin Roose, “OpenAI Insiders Warn,” New York Times, June 4, 2023, '
        "nytimes.com/2024/06/04/example."
    )
    candidate = (
        "Kevin Roose, «OpenAI Insiders Warn», New York Times, 4 de junio de 2023, "
        "nytimes.com/2024/06/04/example."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=24,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.98,
            "reason": "The citation date changed from June 4, 2024 to June 4, 2023.",
            "issues": ["changed date"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": ["source says June 4, 2024; candidate says June 4, 2023"],
            "censored_or_softened": [],
            "structure_issues": [],
            "evidence_source": [source],
            "evidence_candidate": [candidate],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert decision.judge_decision == "warn"
    assert decision.judge_changed_facts == []
    assert "fidelity_judge_numeric_self_contradiction" in {
        issue.code for issue in decision.warnings
    }
    assert "fidelity_judge_reject" not in {
        issue.code for issue in decision.rejections
    }


def test_judge_fail_keeps_real_numeric_change_when_evidence_differs():
    source = (
        'Kevin Roose, “OpenAI Insiders Warn,” New York Times, June 4, 2023, '
        "nytimes.com/2024/06/04/example."
    )
    candidate = (
        "Kevin Roose, «OpenAI Insiders Warn», New York Times, 4 de junio de 2024, "
        "nytimes.com/2024/06/04/example."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=24,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.98,
            "reason": "The citation date changed from 2023 to 2024.",
            "issues": ["changed date"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": ["source says 2023; candidate says 2024"],
            "censored_or_softened": [],
            "structure_issues": [],
            "evidence_source": [source],
            "evidence_candidate": [candidate],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "fidelity_judge_reject" in {
        issue.code for issue in decision.rejections
    }


def test_judge_warn_rejects_structured_changed_fact_even_without_local_rejection():
    decision = assess_fidelity(
        "Er wirkte wie ein fahrender Geselle aus einem vergangenen Jahrhundert.",
        "Parecía un oficial ambulante de un siglo pasado.",
        chunk_index=24,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.92,
            "reason": "Mostly faithful, but one clear factual error remains.",
            "issues": ["mistranslated occupation"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [
                "fahrender Geselle was rendered as officer rather than wandering journeyman"
            ],
            "censored_or_softened": [],
            "structure_issues": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "fidelity_judge_reject" in {issue.code for issue in decision.rejections}


def test_judge_warn_does_not_reject_explicitly_acceptable_localization_notes():
    decision = assess_fidelity(
        "Daniel touched the sleeve of his T-shirt but could not hold on.",
        "Daniel tocó la manga de su playera, pero no pudo sujetarlo.",
        chunk_index=25,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.88,
            "reason": "The translation is faithful; only minor localization notes remain.",
            "issues": ["regional vocabulary"],
            "missing_from_source": [
                "T-shirt is rendered as playera, a regional term that is not inaccurate"
            ],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [
                "playera is not softening but a localization choice"
            ],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_reject" not in {issue.code for issue in decision.rejections}


def test_judge_warn_does_not_block_when_minor_softening_preserves_intensity():
    decision = assess_fidelity(
        "Goddamnit, why did you destroy the work?",
        "¡Maldita sea! ¿Por qué destruiste el trabajo?",
        chunk_index=26,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.92,
            "reason": (
                "The candidate is faithful with one slightly softer exclamation; "
                "no material losses, additions, or changed facts were found."
            ),
            "issues": ["softened_exclamation"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [
                "The curse is slightly less religiously charged, though it preserves "
                "the emotional intensity."
            ],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_reject" not in {issue.code for issue in decision.rejections}


def test_judge_warn_does_not_block_equally_colloquial_idiom_misfiled_as_softening():
    decision = assess_fidelity(
        '"Worried" my ass, I am scared shitless.',
        '—¿«Preocupado»? ¡Y una mierda! Estoy cagado de miedo.',
        chunk_index=27,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.92,
            "reason": (
                "The candidate is a faithful translation with only minor stylistic choices. "
                "No material semantic losses, additions, changed facts, or censorship were found."
            ),
            "issues": ["minor_softening_of_expletive"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [
                "The source idiom uses different wording in Spanish, but it is an equally "
                "colloquial idiom and preserves the vulgarity."
            ],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_reject" not in {issue.code for issue in decision.rejections}


def test_judge_warn_does_not_block_direct_vulgar_equivalent_misfiled_as_softening():
    decision = assess_fidelity(
        "The phrase uses hot snatch and slut as deliberately vulgar insults.",
        "La frase usa coño ardiente y puta como insultos deliberadamente vulgares.",
        chunk_index=28,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.92,
            "reason": (
                "The candidate is faithful with one minor register note that does not "
                "materially alter the source's meaning or tone."
            ),
            "issues": ["minor_register_note"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [
                "Source 'hot snatch' is translated as 'coño ardiente', which is a direct "
                "equivalent but slightly less vulgar in some dialects; the pragmatic force "
                "is preserved."
            ],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_reject" not in {issue.code for issue in decision.rejections}


def test_symbol_bearing_name_is_a_force_rejection_when_punctuation_is_lost():
    decision = assess_fidelity(
        "Ch*Tril greeted Dj\\Tal before the dive.",
        "ChTril saludó a DjTal antes de la inmersión.",
        chunk_index=26,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "symbol_bearing_names_lost" in {issue.code for issue in decision.rejections}

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "pass",
            "confidence": 0.99,
            "reason": "The names are recognizable.",
            "issues": [],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "symbol_bearing_names_lost" in {issue.code for issue in decision.rejections}


def test_symbol_bearing_names_pass_when_exact_spelling_is_preserved():
    decision = assess_fidelity(
        "Ch*Tril greeted Dj\\Tal before the dive.",
        "Ch*Tril saludó a Dj\\Tal antes de la inmersión.",
        chunk_index=27,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is True
    assert "symbol_bearing_names_lost" not in {issue.code for issue in decision.issues}


def test_symbol_bearing_name_accepts_terminal_i_l_ocr_variant_when_count_is_preserved():
    decision = assess_fidelity(
        "Ch*Tril followed Ch*TriI while Dj\\Tal watched.",
        "Ch*Tril siguio a Ch*Tril mientras Dj\\Tal observaba.",
        chunk_index=28,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is True
    assert "symbol_bearing_names_lost" not in {issue.code for issue in decision.issues}


def test_symbol_bearing_name_ocr_reconciliation_does_not_hide_a_real_omission():
    decision = assess_fidelity(
        "Ch*Tril followed Ch*TriI while Dj\\Tal watched.",
        "Ch*Tril siguio adelante mientras Dj\\Tal observaba.",
        chunk_index=29,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "symbol_bearing_names_lost" in {issue.code for issue in decision.rejections}


def test_judge_warn_does_not_override_force_rejections():
    decision = assess_fidelity(
        "The sailors were trapped in the Arctic ice.",
        "The sailors were trapped in the Arctic ice.",
        chunk_index=25,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    assert "untranslated_source" in {issue.code for issue in decision.rejections}

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "warn",
            "confidence": 0.95,
            "reason": "The text remains in the source language.",
            "issues": ["untranslated"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
            "structure_issues": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "untranslated_source" in {issue.code for issue in decision.rejections}


def test_target_language_gate_rejects_auto_greek_returned_for_spanish():
    source = (
        "Ἄνδρα μοι ἔννεπε, Μοῦσα, πολύτροπον, ὃς μάλα πολλὰ "
        "πλάγχθη, ἐπεὶ Τροίης ἱερὸν πτολίεθρον ἔπερσεν."
    )
    decision = assess_fidelity(
        source,
        source,
        chunk_index=28,
        phase="translation",
        source_language="Auto",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "target_script_mismatch" in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_spanish_translation_of_greek():
    source = (
        "Ἄνδρα μοι ἔννεπε, Μοῦσα, πολύτροπον, ὃς μάλα πολλὰ "
        "πλάγχθη, ἐπεὶ Τροίης ἱερὸν πτολίεθρον ἔπερσεν."
    )
    candidate = (
        "Háblame, Musa, de aquel hombre de muchos recursos, que anduvo errante "
        "largo tiempo después de destruir la ciudad sagrada de Troya."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=29,
        phase="translation",
        source_language="Auto",
        target_language="Spanish",
    )

    assert "target_script_mismatch" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {issue.code for issue in decision.rejections}


def test_target_language_gate_rejects_auto_english_returned_for_spanish():
    source = (
        "The model learns a representation of the sentence and then uses the "
        "representation to generate another sentence. The authors explain that "
        "attention allows the system to connect distant words, preserve context, "
        "and reduce the sequential operations that recurrent models require."
    )
    decision = assess_fidelity(
        source,
        source,
        chunk_index=30,
        phase="translation",
        source_language="Auto",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "target_language_missing" in {issue.code for issue in decision.rejections}


def test_target_language_gate_normalizes_display_language_labels():
    source = (
        "The model learns a representation of the sentence and then uses the "
        "representation to generate another sentence. The authors explain that "
        "attention allows the system to connect distant words, preserve context, "
        "and reduce the sequential operations that recurrent models require."
    )
    decision = assess_fidelity(
        source,
        source,
        chunk_index=32,
        phase="translation",
        source_language="Auto",
        target_language="Spanish (Mexico)",
    )

    assert decision.accepted is False
    assert "target_language_missing" in {issue.code for issue in decision.rejections}


def test_target_language_gate_rejects_short_auto_english_echo_for_spanish():
    source = "The beginning of the journey"
    decision = assess_fidelity(
        source,
        source,
        chunk_index=33,
        phase="translation",
        source_language="Auto",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "target_language_missing" in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_same_language_modernization():
    source = (
        "Agora que el caballero habia salido de su casa, todos comenzaron a "
        "hablar de la extrana determinacion que habia tomado."
    )
    decision = assess_fidelity(
        source,
        source,
        chunk_index=31,
        phase="modernize",
        source_language="Spanish",
        target_language="Spanish",
    )

    assert decision.accepted is True
    assert "target_script_mismatch" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {issue.code for issue in decision.rejections}


def test_target_language_gate_rejects_isolated_german_source_residuals():
    source = (
        "Die Wirklichkeit dieser Geschichte blieb in seiner Erinnerung, und die "
        "Zerstörung der Gesellschaft erschien ihm noch Jahre später unbegreiflich."
    )
    candidate = (
        "La Wirklichkeit de esta historia permaneció en su memoria, y la "
        "Zerstörung de la sociedad todavía le parecía incomprensible años después."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert decision.accepted is False
    issue = next(item for item in decision.rejections if item.code == "source_language_residual")
    assert "Wirklichkeit" in issue.detail
    assert "Zerstörung" in issue.detail


def test_target_language_gate_rejects_copied_source_language_phrase():
    source = (
        "Später beschloss der Rat, dem heiligen himelsfursten Sand Sebolten "
        "einen Sarg aus Messing machen zu lassen."
    )
    candidate = (
        "Más tarde, el consejo decidió encargar un sarcófago de latón para el "
        "heiligen himelsfursten Sand Sebolten."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    issue = next(item for item in decision.rejections if item.code == "source_language_residual")
    assert "heiligen himelsfursten" in issue.detail


def test_target_language_gate_rejects_short_source_dialogue_and_idiom_residuals():
    source = (
        "“Oh, my God, Sheila, my God; you did it!” "
        "She wanted her friends nearby, for Christ’s sake."
    )
    candidate = (
        "—¡Oh, my God, Sheila, my God; lo lograste! "
        "Quería tener cerca a sus amigos, for Christ’s sake."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    issue = next(item for item in decision.rejections if item.code == "source_language_residual")
    assert "Oh my God Sheila my God" in issue.detail


def test_target_language_gate_warns_for_repeated_stylized_sound_effect():
    source = (
        "The train went crash rattle-ti-boom crash through the night while "
        "Dean kept talking about the road ahead."
    )
    candidate = (
        "El tren avanzó con un crash rattle-ti-boom crash en plena noche, "
        "mientras Dean seguía hablando del camino que tenían por delante."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" in {item.code for item in decision.warnings}


def test_target_language_gate_still_rejects_stylized_english_lyrics():
    source = (
        "She sang Ma-a-a-ake it dream-y for dancing, and everyone listened "
        "until the band stopped."
    )
    candidate = (
        "Cantó Ma-a-a-ake it dream-y for dancing, y todos escucharon hasta que "
        "la banda dejó de tocar."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_keeps_two_token_person_name_in_dialogue():
    decision = assess_fidelity(
        "“John Smith,” she said, pointing to the signature.",
        "—John Smith —dijo, señalando la firma.",
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_three_token_person_name_in_dialogue():
    decision = assess_fidelity(
        '"John Cameron Swayze," she said, pointing at the television.',
        '—John Cameron Swayze —dijo, señalando el televisor.',
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_surname_particle_after_closing_quote():
    source = (
        "“Peter did it again! His work still inspires leaders everywhere.”\n"
        "Louise van Rhyn, change activist and nation builder, South Africa"
    )
    candidate = (
        "«¡Peter lo hizo de nuevo! Su obra aún inspira a líderes de todas partes».\n"
        "Louise van Rhyn, activista del cambio y constructora de nación, Sudáfrica"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_shared_repeated_function_word():
    decision = assess_fidelity(
        "“No, no, no; we already have enough experts,” Sam said.",
        "—No, no, no; ya tenemos suficientes expertos —dijo Sam.",
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_honors_canonical_name_with_shared_dialogue_particle():
    source = (
        "But I said, ‘Circe, no! What decent man could bear to taste his food "
        "before he saw his men with his own eyes?’"
    )
    candidate = (
        "Pero dije: «¡Circe, no! ¿Qué hombre decente soportaría probar la comida "
        "antes de ver a sus hombres con sus propios ojos?»"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"glossary_terms": {"Circe": "Circe"}},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections + decision.warnings
    }


def test_target_language_gate_warns_instead_of_rejecting_ambiguous_vocative():
    source = (
        "But I said, ‘Circe, no! What decent man could bear to taste his food "
        "before he saw his men with his own eyes?’"
    )
    candidate = (
        "Pero dije: «¡Circe, no! ¿Qué hombre decente soportaría probar la comida "
        "antes de ver a sus hombres con sus propios ojos?»"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" in {item.code for item in decision.warnings}


def test_target_language_gate_allows_multiword_german_proper_names():
    source = (
        "Das Buch erschien bei Vito von Eichhorn GmbH und nennt Die Andere Bibliothek."
    )
    candidate = (
        "El libro apareció en Vito von Eichhorn GmbH y menciona Die Andere Bibliothek."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=34,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_allows_translated_rank_before_person_name():
    decision = assess_fidelity(
        "Der Bericht erwähnt General Robert Swinburne in London.",
        "El informe menciona al general Robert Swinburne en Londres.",
        chunk_index=34,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_allows_fully_translated_german_chunk():
    source = (
        "Die Wirklichkeit dieser Geschichte blieb in seiner Erinnerung, und die "
        "Zerstörung der Gesellschaft erschien ihm noch Jahre später unbegreiflich."
    )
    candidate = (
        "La realidad de esta historia permaneció en su memoria, y la destrucción "
        "de la sociedad todavía le parecía incomprensible años después."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=35,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_does_not_guess_short_person_names_as_residuals():
    source = "Sebald schrieb über Norwich, und Janine erinnerte sich an Rembrandt."
    candidate = "Sebald escribió sobre Norwich, y Janine recordó a Rembrandt."
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=36,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_allows_publisher_name_in_german_title_page():
    source = "W. G. SEBALD DIE RINGE DES SATURN Eichhorn Verlag Frankfurt am Main"
    candidate = "W. G. SEBALD LOS ANILLOS DE SATURNO Editorial Eichhorn, Fráncfort del Meno"
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=37,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_respects_epub_boundaries_around_titles_and_credits():
    source = (
        "A scorching read! —John Grisham[id13](179803—$5.99)[id14]"
        "WINTER FIRE by William R. Trotter[id15]"
        "New York Times Book Review[id18]Published by the Penguin Group[id38]"
        "Penguin Books USA Inc., Hudson Street, New York"
    )
    candidate = (
        "¡Una lectura abrasadora! —John Grisham[id13](179803—$5.99)[id14]"
        "WINTER FIRE, de William R. Trotter[id15]"
        "New York Times Book Review[id18]Publicado por Penguin Group[id38]"
        "Penguin Books USA Inc., Hudson Street, New York"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_allows_translated_bibliography_with_preserved_titles():
    source = (
        "Chapter 3[id0]On Marcus Loew, see Robert Sobel, "
        "“Marcus Loew: An Artist in Spite of Himself,” in "
        "[id1]The Entrepreneurs: Explorations Within the American Business Tradition[id2] "
        "(1974); see also Scott Eyman, [id3]Lion of Hollywood: The Life and Legend "
        "of Louis B. Mayer[id4] (2005). On Thomas Dixon, see Anthony Slide, "
        "[id5]American Racist: The Life and Films of Thomas Dixon[id6] (2004)."
    )
    candidate = (
        "[[[TBLBLOCK000]]]\nCapítulo 3\n[[[/TBLBLOCK000]]]\n\n"
        "[[[TBLBLOCK001]]]\nSobre Marcus Loew, véase Robert Sobel, "
        "«Marcus Loew: An Artist in Spite of Himself», en "
        "The Entrepreneurs: Explorations Within the American Business Tradition "
        "(1974); véase también Scott Eyman, Lion of Hollywood: The Life and Legend "
        "of Louis B. Mayer (2005). Sobre Thomas Dixon, véase Anthony Slide, "
        "American Racist: The Life and Films of Thomas Dixon (2004).\n"
        "[[[/TBLBLOCK001]]]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "target_language_missing" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_translated_epub_notes_with_preserved_urls():
    source = (
        "[id0]Notes[id1]Epigraphs[id2]“It is said”:[id3] Joseph Weizenbaum, "
        "“ELIZA—a Computer Program for the Study of Natural Language "
        "Communication Between Man and Machine,” [id4]Communications of the ACM"
        "[id5] 9, no. 1 (January 1966): 36–45, "
        "[id6]doi.org/10.1145/365153.365168[id7]"
        "GO TO NOTE REFERENCE IN TEXT[id8]"
        "“Successful people create companies”:[id9] Sam Altman, "
        "“Successful People,” [id10]Sam Altman[id11] (blog), March 7, 2013, "
        "[id12]blog.samaltman.com/successful-people[id13]"
        "GO TO NOTE REFERENCE IN TEXT[id14]"
        "Prologue: A Run for the Throne[id15]“How can I help”:[id16] "
        "Tripp Mickle, Cade Metz, Mike Isaac, and Karen Weise, "
        "“Inside OpenAI’s Crisis over the Future of Artificial Intelligence,” "
        "[id17]New York Times[id18], December 9, 2023, "
        "[id19]nytimes.com/2023/12/09/technology/"
        "openai-altman-inside-crisis.html[id20]"
    )
    candidate = (
        "[id0]Notas[id1]Epígrafes[id2]«Se dice»:[id3] Joseph Weizenbaum, "
        "“ELIZA—a Computer Program for the Study of Natural Language "
        "Communication Between Man and Machine,” [id4]Communications of the ACM"
        "[id5] 9, núm. 1 (enero de 1966): 36–45, "
        "[id6]doi.org/10.1145/365153.365168[id7]"
        "VOLVER A LA REFERENCIA DE LA NOTA EN EL TEXTO[id8]"
        "«Las personas exitosas crean empresas»:[id9] Sam Altman, "
        "“Successful People,” [id10]Sam Altman[id11] (blog), "
        "7 de marzo de 2013, [id12]blog.samaltman.com/successful-people[id13]"
        "VOLVER A LA REFERENCIA DE LA NOTA EN EL TEXTO[id14]"
        "Prólogo: Una carrera por el trono[id15]«¿Cómo puedo ayudar?»:[id16] "
        "Tripp Mickle, Cade Metz, Mike Isaac y Karen Weise, "
        "“Inside OpenAI’s Crisis over the Future of Artificial Intelligence,” "
        "[id17]New York Times[id18], 9 de diciembre de 2023, "
        "[id19]nytimes.com/2023/12/09/technology/"
        "openai-altman-inside-crisis.html[id20]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {
                "Artificial Intelligence": "Inteligencia Artificial",
            },
        },
    )

    rejection_codes = {item.code for item in decision.rejections}
    assert "target_language_missing" not in rejection_codes
    assert "source_language_residual" not in rejection_codes


def test_target_language_gate_defers_short_article_title_with_mixed_case_brand():
    source = (
        "Musk was expanding: Dara Kerr, "
        "“How Memphis Became a Battleground over Elon Musk’s xAI Supercomputer,” "
        "NPR, September 11, 2024, "
        "npr.org/2024/09/11/example."
    )
    candidate = (
        "Musk se estaba expandiendo: Dara Kerr, "
        "«How Memphis Became a Battleground over Elon Musk’s xAI Supercomputer», "
        "NPR, 11 de septiembre de 2024, "
        "npr.org/2024/09/11/example."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_defers_sentence_case_cited_talk_title():
    source = (
        "Among the over seven thousand: "
        "“Kevin Scannell on ‘Language from Below: Grassroots Efforts to "
        "Develop Language Technology for Minoritized Languages’ 24.S96 "
        "Special Seminar: Linguistics & social justice,” posted on "
        "November 17, 2021, by MIT-Haiti Initiative, Facebook, 2 hr., "
        "56 min., 46 sec., facebook.com/mithaiti/videos/1060463734714819."
    )
    candidate = (
        "Entre las más de siete mil: "
        "«Kevin Scannell on ‘Language from Below: Grassroots Efforts to "
        "Develop Language Technology for Minoritized Languages’ 24.S96 "
        "Special Seminar: Linguistics & social justice», publicado el "
        "17 de noviembre de 2021 por MIT-Haiti Initiative, Facebook, "
        "2 h, 56 min, 46 s, "
        "facebook.com/mithaiti/videos/1060463734714819."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_still_rejects_untranslated_social_post_body():
    source = (
        "Barret Zoph (@barret_zoph), “I posted this note to OpenAI.,” "
        "September 25, 2024, "
        "x.com/barret_zoph/status/1839095143397515452."
    )
    candidate = (
        "Barret Zoph (@barret_zoph), «I posted this note to OpenAI», "
        "25 de septiembre de 2024, "
        "x.com/barret_zoph/status/1839095143397515452."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_defers_short_scientific_registry_metadata():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "“the problem of accidents”:[id1] Dario Amodei, Chris Olah, "
        "Jacob Steinhardt, Paul Christiano, John Schulman, and Dan Mané, "
        "“Concrete Problems in AI Safety,” preprint, arXiv, July 25, 2016, "
        "1–29, [id2]doi.org/10.48550/arXiv.1606.06565[id3]"
        "GO TO NOTE REFERENCE IN TEXT"
    )
    candidate = (
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO[id0]"
        "«el problema de los accidentes»:[id1] Dario Amodei, Chris Olah, "
        "Jacob Steinhardt, Paul Christiano, John Schulman y Dan Mané, "
        "«Concrete Problems in AI Safety», preprint, arXiv, "
        "25 de julio de 2016, 1–29, "
        "[id2]doi.org/10.48550/arXiv.1606.06565[id3]"
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert any(
        item.code == "source_language_residual"
        and "preprint arXiv" in item.detail
        for item in decision.warnings
    )


def test_target_language_gate_defers_compact_media_citation_metadata():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "“Sequence Learning: A Decade,” posted December 14, 2024, "
        "by seremot, YouTube, 24 min., 36 sec., "
        "[id1]youtu.be/1yvBqasHLZs[id2]"
        "GO TO NOTE REFERENCE IN TEXT"
    )
    candidate = (
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO[id0]"
        "«Sequence Learning: A Decade», publicado el 14 de diciembre de 2024 "
        "por seremot, YouTube, 24 min., 36 s, "
        "[id1]youtu.be/1yvBqasHLZs[id2]"
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert any(
        item.code == "source_language_residual"
        and "seremot YouTube min" in item.detail
        for item in decision.warnings
    )


def test_target_language_gate_defers_preprint_publisher_metadata():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "“Language Models,” preprint, OpenAI, February 14, 2019, 1–24, "
        "[id1]cdn.openai.com/paper.pdf[id2]"
        "GO TO NOTE REFERENCE IN TEXT"
    )
    candidate = (
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO[id0]"
        "«Language Models», preprint, OpenAI, 14 de febrero de 2019, 1–24, "
        "[id1]cdn.openai.com/paper.pdf[id2]"
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert any(
        item.code == "source_language_residual"
        and "preprint OpenAI" in item.detail
        for item in decision.warnings
    )


def test_target_language_gate_defers_complete_bibliographic_identity_runs():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "They set their sights:[id1] Alec Radford, Jeffrey Wu, Rewon Child, "
        "David Luan, Dario Amodei, and Ilya Sutskever, "
        "“Language Models Are Unsupervised Multitask Learners,” preprint, "
        "OpenAI, February 14, 2019, 1–24, "
        "[id2]cdn.openai.com/paper.pdf[id3]"
        "His team called them collectively:[id4] Jared Kaplan, Sam McCandlish, "
        "Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child et al., "
        "“Scaling Laws for Neural Language Models,” preprint, arXiv, "
        "January 23, 2020, 1–30, "
        "[id5]doi.org/10.48550/arXiv.2001.08361[id6]"
    )
    candidate = (
        "IR A LA REFERENCIA DE LA NOTA EN EL TEXTO[id0]"
        "Se fijaron un objetivo:[id1] Alec Radford, Jeffrey Wu, Rewon Child, "
        "David Luan, Dario Amodei e Ilya Sutskever, "
        "«Language Models Are Unsupervised Multitask Learners», preprint, "
        "OpenAI, 14 de febrero de 2019, 1–24, "
        "[id2]cdn.openai.com/paper.pdf[id3]"
        "Su equipo los llamó colectivamente:[id4] Jared Kaplan, Sam McCandlish, "
        "Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child et al., "
        "«Scaling Laws for Neural Language Models», preprint, arXiv, "
        "23 de enero de 2020, 1–30, "
        "[id5]doi.org/10.48550/arXiv.2001.08361[id6]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {"language models": "modelos de lenguaje"},
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" in {
        item.code for item in decision.warnings
    }


def test_target_language_gate_defers_cognate_joined_to_bibliographic_identity_run():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "That position became:[id1] Julia Black, "
        "“Elon Musk Had Twins Last Year with One of His Top Executives,” "
        "[id2]Business Insider[id3], July 6, 2022, "
        "[id4]businessinsider.com/article[id5]"
        "GO TO NOTE REFERENCE IN TEXT[id6]"
        "In her most popular:[id7] Helen Toner, "
        "“Leaning into EA Disillusionment,” Effective Altruism Forum, "
        "July 22, 2022, [id8]forum.effectivealtruism.org/post[id9]"
    )
    candidate = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "Esa posicion se convirtio en:[id1] Julia Black, "
        "«Elon Musk Had Twins Last Year with One of His Top Executives», "
        "[id2]Business Insider[id3], 6 de julio de 2022, "
        "[id4]businessinsider.com/article[id5]"
        "GO TO NOTE REFERENCE IN TEXT[id6]"
        "En su publicacion mas popular:[id7] Helen Toner, "
        "«Leaning into EA Disillusionment», Effective Altruism Forum, "
        "22 de julio de 2022, [id8]forum.effectivealtruism.org/post[id9]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert _is_bibliographic_registry_overlap(
        "popular Helen Toner Leaning into EA Disillusionment Effective Altruism Forum",
        source_text=source,
    )


def test_target_language_gate_defers_surname_particle_and_backlink_after_work_title():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "The victim's brain:[id1] Author interviews with a therapist, "
        "June and August 2024, who also referenced the bestselling book: "
        "Bessel van der Kolk, M.D., [id2]The Body Keeps the Score: Brain, "
        "Mind, and Body in the Healing of Trauma[id3] (Penguin Books, 2015), "
        "1-464.[id4]GO TO NOTE REFERENCE IN TEXT[id5]"
        "In the three months:[id6] Screenshot of Annie's OnlyFans income history."
    )
    candidate = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "El cerebro de la victima:[id1] Entrevistas de la autora con una terapeuta, "
        "junio y agosto de 2024, quien tambien hizo referencia al libro superventas: "
        "Bessel van der Kolk, M.D., [id2]The Body Keeps the Score: Brain, "
        "Mind, and Body in the Healing of Trauma[id3] (Penguin Books, 2015), "
        "1-464.[id4]GO TO NOTE REFERENCE IN TEXT[id5]"
        "En los tres meses:[id6] Captura del historial de ingresos de Annie en OnlyFans."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert _is_bibliographic_registry_overlap(
        (
            "Bessel van der Kolk M.D The Body Keeps the Score Brain Mind and "
            "Body in the Healing of Trauma Penguin Books GO TO NOTE REFERENCE IN TEXT"
        ),
        source_text=source,
    )


def test_target_language_gate_decodes_html_entities_inside_citation_titles():
    source = (
        "[id0]Chapter 15: The Gambit[id1]"
        "The shift happened:[id2] Christopher Jarvis, "
        "“The Rise and Fall of Albania's Pyramid Schemes,” "
        "[id3]Finance &amp; Development[id4], International Monetary Fund, "
        "March 2000, [id5]imf.org/article[id6]"
    )
    candidate = (
        "[id0]Capitulo 15: La estratagema[id1]"
        "El cambio ocurrio:[id2] Christopher Jarvis, "
        "«The Rise and Fall of Albania's Pyramid Schemes», "
        "[id3]Finance &amp; Development[id4], Fondo Monetario Internacional, "
        "marzo de 2000, [id5]imf.org/article[id6]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_ignores_social_handle_after_post_text_is_translated():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "“Accountability is important”:[id1] Helen Toner released the statement "
        "in a screenshot on X: Helen Toner (@hlntnr), "
        "“A statement from Helen Toner and Tasha McCauley:,” "
        "Twitter (now X), March 8, 2024, "
        "[id2]x.com/hlntnr/status/1766269137628590185[id3]"
    )
    candidate = (
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO[id0]"
        "«La rendicion de cuentas es importante»:[id1] Helen Toner publico "
        "la declaracion en una captura de pantalla en X: Helen Toner (@hlntnr), "
        "«Una declaracion de Helen Toner y Tasha McCauley:», "
        "Twitter (ahora X), 8 de marzo de 2024, "
        "[id2]x.com/hlntnr/status/1766269137628590185[id3]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_semantic_endnotes_with_exact_titles():
    source = """Chapter 5: Scale of Ambition
“How about now?”: Cade Metz, Genius Makers: The Mavericks Who Brought AI to Google, Facebook, and the World (Dutton, 2021), 93; “Geoffrey Hinton | On Working with Ilya, Choosing Problems, and the Power of Intuition,” posted May 20, 2024, by Sana, YouTube, 45 min., 45 sec., youtu.be/n4IQOBka8bc
GO TO NOTE REFERENCE IN TEXT
He stunned Hinton: Author interview with Geoffrey Hinton, November 2023.
GO TO NOTE REFERENCE IN TEXT
At times he grew: Metz, Genius Makers, 94.
GO TO NOTE REFERENCE IN TEXT
“One doesn’t bet”: Will Douglas Heaven, “Rogue Superintelligence and Merging with Machines: Inside the Mind of OpenAI’s Chief Scientist,” MIT Technology Review, October 26, 2023, technologyreview.com/article"""
    candidate = """Capítulo 5: Escala de ambición
«¿Qué tal ahora?»: Cade Metz, Genius Makers: The Mavericks Who Brought AI to Google, Facebook, and the World (Dutton, 2021), 93; «Geoffrey Hinton | On Working with Ilya, Choosing Problems, and the Power of Intuition», publicado el 20 de mayo de 2024, por Sana, YouTube, 45 min., 45 seg., youtu.be/n4IQOBka8bc
IR A LA NOTA DE REFERENCIA EN EL TEXTO
Dejó atónito a Hinton: Entrevista de la autora con Geoffrey Hinton, noviembre de 2023.
IR A LA NOTA DE REFERENCIA EN EL TEXTO
A veces se volvía: Metz, Genius Makers, 94.
IR A LA NOTA DE REFERENCIA EN EL TEXTO
«Uno no apuesta»: Will Douglas Heaven, «Rogue Superintelligence and Merging with Machines: Inside the Mind of OpenAI’s Chief Scientist», MIT Technology Review, 26 de octubre de 2023, technologyreview.com/article"""

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    rejection_codes = {item.code for item in decision.rejections}
    assert "target_language_missing" not in rejection_codes
    assert "source_language_residual" not in rejection_codes
    assert "source_language_residual" in {
        item.code for item in decision.warnings
    }


def test_target_language_gate_preserves_quoted_report_title_with_but_connector():
    source = """GO TO NOTE REFERENCE IN TEXT
In 2021, the World Bank: “Continued Rebound, but Storms Cloud the Horizon: Policies to Accelerate the Productive Economy for Inclusive Growth,” Kenya Economic Update, no. 26 (World Bank, 2022), 1–54, hdl.handle.net/10986/38386"""
    candidate = """IR A LA NOTA DE REFERENCIA EN EL TEXTO
En 2021, el Banco Mundial: «Continued Rebound, but Storms Cloud the Horizon: Policies to Accelerate the Productive Economy for Inclusive Growth», Kenya Economic Update, núm. 26 (Banco Mundial, 2022), 1–54, hdl.handle.net/10986/38386"""

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_defers_glossary_term_inside_unquoted_citation_title():
    source = """Chapter 7: Science in Captivity
GO TO NOTE REFERENCE IN TEXT
In China, GPT-3 similarly: Jeffrey Ding and Jenny W. Xiao, Recent Trends in China’s Large Language Model Landscape, Centre for the Governance of AI, April 28, 2023, 1–14, cdn.governance.ai/Trends_in_Chinas_LLMs.pdf"""
    candidate = """Capítulo 7: Ciencia en cautiverio
IR A LA NOTA DE REFERENCIA EN EL TEXTO
En China, GPT-3 de manera similar: Jeffrey Ding y Jenny W. Xiao, Recent Trends in China’s Large Language Model Landscape, Centre for the Governance of AI, 28 de abril de 2023, 1–14, cdn.governance.ai/Trends_in_Chinas_LLMs.pdf"""

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {"language model": "modelo de lenguaje"},
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }
    assert decision.accepted is True


def test_target_language_gate_still_requires_glossary_term_in_narrative_prose():
    decision = assess_fidelity(
        "The language model predicted the next word correctly.",
        "El language model predijo correctamente la siguiente palabra.",
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {"language model": "modelo de lenguaje"},
        },
    )

    issue = next(
        item
        for item in decision.rejections
        if item.code == "source_language_residual"
    )
    assert "language model" in issue.detail


def test_target_language_gate_rejects_narrative_prose_in_semantic_endnotes():
    source = """Notes
The research team posted the complete interview on December 14, 2024, with a short introduction, YouTube, 24 min., youtu.be/example.
GO TO NOTE REFERENCE IN TEXT
The interview changed the direction of the research program."""
    candidate = """Notas
The research team posted the complete interview el 14 de diciembre de 2024, con una breve introducción, YouTube, 24 min., youtu.be/example.
IR A LA NOTA DE REFERENCIA EN EL TEXTO
The interview changed the direction of the research program."""

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_still_rejects_prose_inside_media_citation():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "The research team posted the complete interview on December 14, 2024, "
        "with a short introduction, YouTube, 24 min., "
        "[id1]youtu.be/example[id2]"
        "GO TO NOTE REFERENCE IN TEXT"
    )
    candidate = (
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO[id0]"
        "The research team posted the complete interview el 14 de diciembre de "
        "2024, con una breve introducción, YouTube, 24 min., "
        "[id1]youtu.be/example[id2]"
        "IR A LA NOTA DE REFERENCIA EN EL TEXTO"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_long_work_title_inside_translated_notes():
    source = (
        "Chapter 5[id0]On Stroheim, see Richard Koszarski, "
        "[id1]The Man You Loved to Hate: Erich von Stroheim and Hollywood[id2] "
        "(1983). His life and career are still full of holes. In view of his "
        "genius as a director, we deserve a new work to match the man."
    )
    candidate = (
        "Capítulo 5[id0]Sobre Stroheim, véase Richard Koszarski, "
        "[id1]The Man You Loved to Hate: Erich von Stroheim and Hollywood[id2] "
        "(1983). Su vida y su carrera aún están llenas de lagunas. En vista de "
        "su genio como director, merecemos una nueva obra a la altura del personaje."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_enforce_glossary_inside_longer_work_title(
    tmp_path,
    monkeypatch,
):
    import yaml

    from src.core.book_profiles import create_profile

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("nested_title_gate", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Everything",
                    "target": "Todo",
                    "type": "term",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
                {
                    "source": "Biography",
                    "target": "Biografía",
                    "type": "term",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    source = (
        "See [id0]Everything Is Cinema: The Working Life of Jean-Luc Godard[id1] "
        "and [id2]Luis Buñuel: A Critical Biography[id3]."
    )
    candidate = (
        "Véase [id0]Everything Is Cinema: The Working Life of Jean-Luc Godard[id1] "
        "y [id2]Luis Buñuel: A Critical Biography[id3]."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "editorial_mode": "book_profile",
            "profile_id": "nested_title_gate",
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_enforce_glossary_inside_semantic_index_title(
    tmp_path,
    monkeypatch,
):
    import yaml

    from src.core.book_profiles import create_profile

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("semantic_index_title", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Machine Learning",
                "target": "aprendizaje automático",
                "type": "term",
                "status": "approved",
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    source = (
        "MacAskill, William, , machine learning, , , Machine Learning for "
        "Health, Mądry, Aleksander, , , , , , , Maduro, Nicolás, Mahelona, "
        "Keoni, Makanju, Anna, , , , ,"
    )
    candidate = (
        "MacAskill, William, , aprendizaje automático, , , Machine Learning "
        "for Health, Mądry, Aleksander, , , , , , , Maduro, Nicolás, Mahelona, "
        "Keoni, Makanju, Anna, , , , ,"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "editorial_mode": "book_profile",
            "profile_id": "semantic_index_title",
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_still_enforces_glossary_in_ordinary_prose(
    tmp_path,
    monkeypatch,
):
    import yaml

    from src.core.book_profiles import create_profile

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("ordinary_machine_learning", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Machine Learning",
                "target": "aprendizaje automático",
                "type": "term",
                "status": "approved",
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    decision = assess_fidelity(
        "Machine Learning improves the model when the training data is reliable.",
        "Machine Learning mejora el modelo cuando los datos de entrenamiento son fiables.",
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "editorial_mode": "book_profile",
            "profile_id": "ordinary_machine_learning",
        },
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_still_rejects_untranslated_bibliography():
    source = (
        "Chapter 3. On Marcus Loew, see Robert Sobel, Marcus Loew: An Artist in "
        "Spite of Himself, in The Entrepreneurs: Explorations Within the American "
        "Business Tradition (1974); see also Scott Eyman, Lion of Hollywood: The "
        "Life and Legend of Louis B. Mayer (2005). On Thomas Dixon, see Anthony "
        "Slide, American Racist: The Life and Films of Thomas Dixon (2004)."
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "target_language_missing" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_block_one_ambiguous_loanword():
    decision = assess_fidelity(
        "This gripping thriller follows a journalist across Europe.",
        "Este thriller apasionante sigue a una periodista por Europa.",
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_does_not_apply_multiword_glossary_by_token():
    decision = assess_fidelity(
        (
            "Shakespeare remains central to American Cinema, while the critic "
            "compares that tradition with American Tragedy."
        ),
        (
            "Shakespeare sigue siendo central para American Cinema, mientras el "
            "crítico compara esa tradición con Una tragedia americana."
        ),
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {
                "American Tragedy": "Una tragedia americana",
            },
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_rejects_full_required_glossary_phrase():
    decision = assess_fidelity(
        "The critic reread American Tragedy before writing the essay.",
        "El crítico volvió a leer American Tragedy antes de escribir el ensayo.",
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {
                "American Tragedy": "Una tragedia americana",
            },
        },
    )

    issue = next(
        item
        for item in decision.rejections
        if item.code == "source_language_residual"
    )
    assert "American Tragedy" in issue.detail


def test_target_language_gate_warns_but_does_not_reject_scattered_ambiguous_tokens():
    source = (
        "I lifted a highball to my lips while we discussed what I had written "
        "from Paterson. It was horrible because the station was already closing."
    )
    candidate = (
        "Levanté el highball hasta los labios mientras hablábamos de lo que yo había "
        "escrito desde Paterson. Fue horrible porque la estación ya estaba cerrando."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is True
    assert "source_language_residual" not in {item.code for item in decision.rejections}
    assert "source_language_residual" in {item.code for item in decision.warnings}


def test_target_language_gate_defers_short_coordinated_loan_phrase_to_judge():
    decision = assess_fidelity(
        (
            "The soundtrack moves from jazz into rock and roll before the final "
            "scene returns to silence."
        ),
        (
            "La banda sonora pasa del jazz al rock and roll antes de que la escena "
            "final vuelva al silencio."
        ),
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert decision.accepted is True
    assert "source_language_residual" not in {item.code for item in decision.rejections}
    assert "source_language_residual" in {item.code for item in decision.warnings}


def test_target_language_gate_still_rejects_required_coordinated_phrase():
    decision = assess_fidelity(
        "The critic dismissed life and death as an easy theme.",
        "El crítico descartó life and death como un tema fácil.",
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "target_language_gate": True,
            "glossary_terms": {"life and death": "vida y muerte"},
        },
    )

    assert "source_language_residual" in {item.code for item in decision.rejections}


def test_target_language_gate_allows_shared_cognate_before_split_character_name():
    source = (
        "The current carried them forward, and the inexorable Ch*Tril continued "
        "swimming beside Daniel while the others watched from the dark water."
    )
    candidate = (
        "La corriente los arrastró hacia delante, y la inexorable Ch*Tril siguió "
        "nadando junto a Daniel mientras los demás observaban desde el agua oscura."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_preserves_boolean_search_query_and_word_form_examples():
    source = (
        "The terminal displayed the query:[id0]"
        "DEFINE: dolphin AND behav OR acti AND odd OR unusual OR strange[id1]"
        "It searches for strings of letters: Behavior, behaving, behaved, "
        "actions, activity, acting."
    )
    candidate = (
        "La terminal mostró la consulta:[id0]"
        "DEFINE: dolphin AND behav OR acti AND odd OR unusual OR strange[id1]"
        "Busca esas cadenas de letras: Behavior, behaving, behaved, actions, "
        "activity, acting."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is True
    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_allows_epithet_name_before_epub_placeholder():
    source = "Der arme Algernon[id19]Siebter Teil"
    candidate = "El pobre Algernon[id19]Séptima parte"
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=38,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_does_not_reject_single_capitalized_place_name():
    source = (
        "Nach einer Stunde erreichte er das Gefängnis von Blundeston und ging "
        "anschließend weiter zur Küste."
    )
    candidate = (
        "Después de una hora llegó a la prisión de Blundeston y luego continuó "
        "hacia la costa."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=39,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_does_not_reject_historical_entity_after_article():
    source = (
        "Die Wehrmacht deportierte Gefangene nach Theresienstadt, wie der Bericht erklärt."
    )
    candidate = (
        "La Wehrmacht deportó prisioneros a Theresienstadt, según explica el informe."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=40,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_does_not_reject_list_of_capitalized_places():
    source = (
        "Die Fahrt führte über die Lietzenburgerstraße und die Kantstraße, durch "
        "den Tiergarten, den Grunewald und schließlich zum Westkreuz."
    )
    candidate = (
        "El trayecto pasó por Lietzenburgerstraße y Kantstraße, atravesó Tiergarten "
        "y Grunewald y finalmente llegó a Westkreuz."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=41,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_repeated_honorific_names_with_initials():
    source = (
        "The mayor arrived with sir T. A. Bridges, sir John Browne, and others. "
        "They entered the church together."
    )
    candidate = (
        "El alcalde llegó con sir T. A. Bridges, sir John Browne y otros. "
        "Entraron juntos en la iglesia."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_hide_clause_between_honorific_names():
    source = "Sir John told Sir Thomas to wait outside before sunset."
    candidate = "Sir John told Sir Thomas to wait outside antes del anochecer."

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_reject_capitalized_name_with_diacritic():
    source = (
        "Hölderlin schrieb über die Landschaft, während der Erzähler seine Reise fortsetzte."
    )
    candidate = (
        "El poema de Hölderlin habla del paisaje, mientras el narrador continúa su viaje."
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=42,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_place_name_with_foreign_connector():
    source = (
        "Pongo das Mortes, früher Vormittag. Urwald, steile Berge und "
        "dampfender Nebel liegen über dem reißenden Fluss."
    )
    candidate = (
        "Pongo das Mortes, a media mañana. La selva, las montañas escarpadas "
        "y la niebla humeante cubren el río embravecido."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="German",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_connector_acronym_organization_in_index():
    source = (
        "Bezos, Jeff, Biden, Joe, Bing, biological viruses, "
        "biological weapons, Birhane, Abeba, “black box,” Black in AI, "
        "blacklists, Black Lives Matter,"
    )
    candidate = (
        "Bezos, Jeff, Biden, Joe, Bing, virus biológicos, "
        "armas biológicas, Birhane, Abeba, «caja negra», Black in AI, "
        "listas negras, Black Lives Matter,"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_does_not_exempt_acronym_ended_source_clause():
    source = "“Everything in AI,” she said, “must change rapidly.”"
    candidate = "«Everything in AI», dijo ella, «debe cambiar rápidamente»."

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_keeps_partially_shared_scene_heading():
    source = (
        "Iquitos, Bar, Abend[id0]Vor der Bar warten noch viele Menschen, "
        "während die Auswahl im Inneren weitergeht."
    )
    candidate = (
        "Iquitos, bar, noche[id0]Frente al bar todavía espera mucha gente, "
        "mientras la selección continúa en el interior."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="German",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_still_rejects_unquoted_source_exclamation():
    source = "Oh my God! Danach lief er ohne ein weiteres Wort davon."
    candidate = "Oh my God! Después se marchó sin decir una palabra más."

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_dialogue_already_in_target_language():
    source = (
        "Er wachte auf und fluchte: puta su madre! Danach starrten ihn die "
        "anderen Spieler schweigend an."
    )
    candidate = (
        "Despertó y soltó una maldición: ¡puta su madre! Después los demás "
        "jugadores lo miraron en silencio."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="German",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {item.code for item in decision.rejections}


def test_target_language_gate_keeps_repeated_target_markers_when_detector_is_wrong():
    source = (
        "He pointed toward the flames and whispered, 'Aviones... bombas... "
        "mucho, mucho.' Then the houses collapsed."
    )
    candidate = (
        "Señaló hacia las llamas y susurró: «Aviones... bombas... mucho, mucho». "
        "Luego las casas se derrumbaron."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_rejects_ambiguous_short_source_dialogue():
    source = "No problem! He stood up and walked toward the door."
    candidate = "No problem! Se levantó y caminó hacia la puerta."

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" in {item.code for item in decision.rejections}


def test_target_language_gate_still_rejects_german_clause_with_capitalized_ends():
    source = (
        "Wir gingen nach Hause, weil der Regen immer stärker wurde und niemand "
        "mehr auf der Straße bleiben wollte."
    )
    candidate = (
        "Nos retiramos temprano. Wir gingen nach Hause, porque la lluvia se "
        "volvía cada vez más intensa."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=0,
        phase="translation",
        source_language="German",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" in {item.code for item in decision.rejections}


def test_target_language_gate_rejects_long_german_candidate_for_spanish():
    source = (
        "Die Geschichte dieser langen Reise wurde in einem alten Buch beschrieben. "
        "Der Erzähler erinnerte sich an die Landschaft, die Menschen und die Städte. "
    ) * 8
    decision = assess_fidelity(
        source,
        source,
        chunk_index=42,
        phase="translation",
        source_language="German",
        target_language="Spanish",
    )

    assert "target_language_missing" in {item.code for item in decision.rejections}


def test_glued_pdf_toc_judge_failure_becomes_warning():
    decision = assess_fidelity(
        (
            "4. Violenciaimplícita: Nochedefuego(TatianaHuezo2021)........269"
            "4.1. Introducciónycontexto........269"
            "4.2. Tiposdeviolencia........272"
        ),
        (
            "4. Implicit Violence: Prayers for the Stolen (Tatiana Huezo 2021)........269\n"
            "4.1. Introduction and Context........269\n"
            "4.2. Types of Violence........272"
        ),
        chunk_index=23,
        phase="translation",
        source_language="Spanish",
        target_language="English",
    )
    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.98,
            "reason": "Proper noun and page number concerns are caused by OCR concatenated forms.",
            "issues": ["proper_noun_concatenated", "page number mismatch"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": ["page number and proper-name concatenation from OCR"],
            "censored_or_softened": [],
            "structure_issues": [],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_ocr_boundary_warning" in {issue.code for issue in decision.warnings}


def test_damaged_pdf_markdown_table_failure_becomes_warning():
    source = (
        "| Item | Value 1 | Value 2 | Value 3 |\n"
        "| --- | --- | --- | --- |\n"
        "| GPT- | 2 |  |  |\n"
        "| Fine-Tune | 354M | 27.7 | 64.2 |\n"
        "| LoRA | 0.35M | 46.7"
    )
    candidate = (
        "| Elemento | Valor 1 | Valor 2 | Valor 3 |\n"
        "| --- | --- | --- | --- |\n"
        "| GPT- | 2 |  |  |\n"
        "| Fine-Tune | 354M | 27.7 | 64.2 |\n"
        "| LoRA | 0.35M | 46.7"
    )
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=26,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    assert "source_pdf_extraction_noise" in {issue.code for issue in decision.warnings}

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.99,
            "reason": "The source table is visibly damaged and the last row is incomplete.",
            "issues": ["incomplete_table_row", "truncated_data"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
            "structure_issues": ["broken table row caused by source extraction"],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_ocr_boundary_warning" in {issue.code for issue in decision.warnings}


def test_rejects_untranslated_source_only_for_real_text():
    decision = assess_fidelity(
        "The sailors were trapped in the Arctic ice.",
        "The sailors were trapped in the Arctic ice.",
        chunk_index=3,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    punctuation = assess_fidelity(
        "...",
        "...",
        chunk_index=4,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "untranslated_source" in {issue.code for issue in decision.rejections}
    assert punctuation.accepted is True


def test_target_language_gate_allows_unchanged_name_only_analytical_index():
    source = (
        "Cameron, Julia Margaret, [id0]Campion, Jane, [id1], [id2]"
        "Capra, Frank, [id3], [id4], [id5]Casablanca[id6], [id7]"
    )
    candidate = (
        "[[[TBLBLOCK000]]]\nCameron, Julia Margaret,\n[[[/TBLBLOCK000]]]\n"
        "[[[TBLBLOCK001]]]\nCampion, Jane,\n[[[/TBLBLOCK001]]]\n"
        "[[[TBLBLOCK002]]]\nCapra, Frank,\n[[[/TBLBLOCK002]]]\n"
        "[[[TBLBLOCK003]]]\nCasablanca\n[[[/TBLBLOCK003]]]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=4,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_name_only_table_recovery_batch():
    source = (
        "[[[TBLBLOCK000]]]\nMary Kay Ash\n[[[/TBLBLOCK000]]]\n"
        "[[[TBLBLOCK001]]]\nMary Kay Cosmetics\n[[[/TBLBLOCK001]]]\n"
        "[[[TBLBLOCK002]]]\nRay Kroc\n[[[/TBLBLOCK002]]]\n"
        "[[[TBLBLOCK003]]]\nMcDonald’s\n[[[/TBLBLOCK003]]]"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=5,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "table"},
    )

    assert "untranslated_source" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {
        issue.code for issue in decision.rejections
    }


def test_target_language_gate_rejects_untranslated_table_description():
    source = (
        "[[[TBLBLOCK000]]]\nSir Richard Branson\n[[[/TBLBLOCK000]]]\n"
        "[[[TBLBLOCK001]]]\nVirgin Records, Airlines, and others\n"
        "[[[/TBLBLOCK001]]]"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=6,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "table"},
    )

    assert "untranslated_source" in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_translated_table_description_with_names():
    source = (
        "[[[TBLBLOCK000]]]\nSir Richard Branson\n[[[/TBLBLOCK000]]]\n"
        "[[[TBLBLOCK001]]]\nVirgin Records, Airlines, and others\n"
        "[[[/TBLBLOCK001]]]"
    )
    candidate = (
        "[[[TBLBLOCK000]]]\nSir Richard Branson\n[[[/TBLBLOCK000]]]\n"
        "[[[TBLBLOCK001]]]\nVirgin Records, aerolíneas y otras empresas\n"
        "[[[/TBLBLOCK001]]]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=7,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "table"},
    )

    assert "untranslated_source" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {
        issue.code for issue in decision.rejections
    }


def test_target_language_gate_allows_flattened_name_table_with_one_translation():
    source = (
        "Nike Mary Kay Ash Mary Kay Cosmetics Ray Kroc McDonald’s "
        "David Packard Hewlett-Packard Howard Schultz Starbucks Sam Walton "
        "Walmart Reid Hoffman LinkedIn Kim Scott Dropbox John Mackey "
        "Whole Foods Tony Hsieh Zappos Sir Richard Branson "
        "Virgin Records, Airlines, and others Satya Nadella Microsoft "
        "Carly Fiorina David Novak YUM! Brands Andrew Yang "
        "Manhattan Prep (test preparation) and political candidate "
        "Ken Langone Venture capitalist, instrumental in financing Home Depot"
    )
    candidate = (
        "Nike Mary Kay Ash Mary Kay Cosmetics Ray Kroc McDonald’s "
        "David Packard Hewlett-Packard Howard Schultz Starbucks Sam Walton "
        "Walmart Reid Hoffman LinkedIn Kim Scott Dropbox John Mackey "
        "Whole Foods Tony Hsieh Zappos Sir Richard Branson "
        "Virgin Records, Virgin Airlines y otros Satya Nadella Microsoft "
        "Carly Fiorina David Novak YUM! Brands Andrew Yang "
        "Manhattan Prep (preparación de exámenes) y candidato político "
        "Ken Langone Capitalista de riesgo, fundamental en la financiación "
        "de Home Depot"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=8,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "table"},
    )

    assert "target_language_missing" not in {
        issue.code for issue in decision.rejections
    }
    assert "source_language_residual" not in {
        issue.code for issue in decision.rejections
    }


def test_target_language_gate_ignores_inline_structural_recovery_markers():
    source = (
        "[[[TBLBLOCK000]]]Kelsey Piper, a senior reporter, wrote the article."
        "[[[/TBLBLOCK000]]]"
    )
    candidate = (
        "[[[TBLBLOCK000]]]Kelsey Piper, reportera sénior, escribió el artículo."
        "[[[/TBLBLOCK000]]]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=5,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        issue.code for issue in decision.rejections
    }
    assert "target_language_missing" not in {
        issue.code for issue in decision.rejections
    }


def test_target_language_gate_rejects_unchanged_translatable_index_subjects():
    source = (
        "catastrophes, profiting on, [id0]celebrity, [id1]censorship, [id2]"
        "children, protection of, [id3]children's movies, [id4]"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=5,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_mixed_index_when_subject_entries_are_translated():
    source = (
        "Kubrick, Stanley, [id0]Ku Klux Klan, [id1]Kuleshov, Lev, [id2]"
        "Kurosawa, Akira, [id3]La Baie des Anges[id4]labor unions, [id5]"
        "Laemmle, Carl, [id6]Laemmle family, [id7]La Grande Illusion[id8]"
        "Lang, Fritz, [id9]language, [id10]Scott, A. O., [id11]"
    )
    candidate = (
        "Kubrick, Stanley, [id0]Ku Klux Klan, [id1]Kuleshov, Lev, [id2]"
        "Kurosawa, Akira, [id3]La Baie des Anges[id4]sindicatos, [id5]"
        "Laemmle, Carl, [id6]familia Laemmle, [id7]La gran ilusión[id8]"
        "Lang, Fritz, [id9]idioma, [id10]Scott, A. O., [id11]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=6,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {issue.code for issue in decision.rejections}


def test_target_language_gate_allows_name_dominated_index_with_translated_descriptor():
    source = (
        "Goffee, Rob[id0]Goizueta Business School[id1]Goldman Sachs[id2]"
        "Good to Great[id3] (Collins)[id4]Goodwin, Doris Kearns[id5]"
        "Google[id6]Google Scholar[id7]Graham, Lindsey[id8]"
        "Granovetter, Mark[id9]Grant, Adam[id10]Great Society[id11]"
        "Green Bay Packers[id12]Groupon[id13]Grove, Andrew[id14]"
        "Gruenfeld, Deborah[id15]Guardian[id16]H[id17]"
        "Haas Business School[id18]Halstead, Richard[id19]"
        "Handy Dan (home improvement company)[id20]HarperCollins[id21]"
        "Harrah’s Entertainment[id22]Harvard Business Review[id23]"
    )
    candidate = (
        "Goffee, Rob[id0]Goizueta Business School[id1]Goldman Sachs[id2]"
        "Good to Great[id3] (Collins)[id4]Goodwin, Doris Kearns[id5]"
        "Google[id6]Google Scholar[id7]Graham, Lindsey[id8]"
        "Granovetter, Mark[id9]Grant, Adam[id10]Great Society[id11]"
        "Green Bay Packers[id12]Groupon[id13]Grove, Andrew[id14]"
        "Gruenfeld, Deborah[id15]Guardian[id16]H[id17]"
        "Haas Business School[id18]Halstead, Richard[id19]"
        "Handy Dan (empresa de mejoras para el hogar)[id20]"
        "HarperCollins[id21]Harrah’s Entertainment[id22]"
        "Harvard Business Review[id23]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=12,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )

    assert "target_language_missing" not in {
        issue.code for issue in decision.rejections
    }
    assert "untranslated_source" not in {
        issue.code for issue in decision.rejections
    }


def test_fidelity_audit_prompt_explains_mixed_language_index_policy():
    prompt = build_fidelity_audit_prompt(
        "Walker, Ross; weak ties",
        "Walker, Ross; lazos débiles",
        source_language="English",
        target_language="Spanish",
        document_context="index",
    )

    compact = " ".join(prompt.user.split())
    assert "analytical index or catalog" in compact
    assert "Mixed-language surface text is expected and correct" in compact
    assert "preserved names remain in the source language" in compact


def test_index_policy_false_rejection_is_downgraded_without_masking_names():
    source = (
        "Walker, Ross Walker and Company Brands warfare, asymmetric "
        "warmth, competence vs. Warner Media weak ties WebMD"
    )
    candidate = (
        "Walker, Ross Walker and Company Brands guerra asimétrica "
        "calidez frente a competencia Warner Media lazos débiles WebMD"
    )
    assessment = {
        "verdict": "fail",
        "confidence": 0.98,
        "reason": (
            "Two index entries are translated while the rest remain in "
            "English, creating an inconsistent translation policy."
        ),
        "issues": ["Inconsistent translation policy for index entries"],
        "missing_from_source": [
            "warfare, asymmetric",
            "warmth, competence vs.",
            "weak ties",
        ],
        "added_not_in_source": [
            "guerra asimétrica",
            "calidez frente a competencia",
            "lazos débiles",
        ],
        "changed_facts": [],
        "censored_or_softened": [],
        "structure_issues": [],
    }

    normalized = _normalize_structured_index_audit_policy(
        source,
        candidate,
        assessment,
        document_context="index",
    )

    assert normalized["verdict"] == "warn"
    assert normalized["missing_from_source"] == []
    assert normalized["added_not_in_source"] == []

    name_change = dict(assessment)
    name_change["changed_facts"] = ["Walker, Ross -> Walter, Ross"]
    unchanged = _normalize_structured_index_audit_policy(
        source,
        candidate,
        name_change,
        document_context="index",
    )
    assert unchanged["verdict"] == "fail"


def test_target_language_gate_rejects_unchanged_mixed_index_with_subject_entries():
    source = (
        "Kubrick, Stanley, [id0]Ku Klux Klan, [id1]Kuleshov, Lev, [id2]"
        "Kurosawa, Akira, [id3]labor unions, [id4]Laemmle, Carl, [id5]"
        "Laemmle family, [id6]Lang, Fritz, [id7]language, [id8]"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=7,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" in {issue.code for issue in decision.rejections}


def test_target_language_gate_ignores_epub_placeholders_for_short_name_entry():
    decision = assess_fidelity(
        "Ozark, [id0], [id1]",
        "Ozark, [id0], [id1]",
        chunk_index=8,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" not in {issue.code for issue in decision.rejections}
    assert "target_language_missing" not in {issue.code for issue in decision.rejections}


def test_target_language_gate_still_rejects_short_prose_with_epub_placeholders():
    source = "The sailors safely escaped.[id0][id1]"
    decision = assess_fidelity(
        source,
        source,
        chunk_index=9,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "untranslated_source" in {issue.code for issue in decision.rejections}


def test_parse_fidelity_audit_response_from_wrapped_json():
    parsed = parse_fidelity_audit_response(
        """
        <FIDELITY_AUDIT_JSON>
        {
          "verdict": "fail",
          "confidence": 0.93,
          "reason": "missing survival clause",
          "issues": ["omission"],
          "missing_from_source": ["most survived"],
          "added_not_in_source": [],
          "changed_facts": [],
          "censored_or_softened": [],
          "structure_issues": [],
          "evidence_source": ["most of the others survived"],
          "evidence_candidate": []
        }
        </FIDELITY_AUDIT_JSON>
        """
    )

    assert parsed["verdict"] == "fail"
    assert parsed["confidence"] == 0.93
    assert parsed["missing_from_source"] == ["most survived"]


def test_parse_fidelity_audit_response_ignores_incomplete_large_json():
    parsed = parse_fidelity_audit_response(
        "<FIDELITY_AUDIT_JSON>"
        '{"verdict":"pass","confidence":0.9,'
        + ("x" * 100_000)
    )

    assert parsed is None


def test_judge_fail_rejects_candidate_for_censorship():
    decision = assess_fidelity(
        "The prisoner was tortured and executed.",
        "El prisionero fue castigado.",
        chunk_index=5,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "fail",
            "confidence": 0.91,
            "reason": "softens torture and execution",
            "issues": ["censorship"],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": ["tortured and executed -> castigado"],
        },
        model="deepseek-v4-pro",
        provider="deepseek",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is False
    assert "fidelity_judge_reject" in {issue.code for issue in decision.rejections}
    assert decision.independence == "weak"


def test_high_confidence_pass_can_downgrade_non_force_local_reject():
    decision = assess_fidelity(
        "There were 17 men.",
        "Habia diecisiete hombres.",
        chunk_index=6,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    assert decision.accepted is False

    apply_fidelity_audit_assessment(
        decision,
        {
            "verdict": "pass",
            "confidence": 0.94,
            "reason": "17 is translated as diecisiete",
            "issues": [],
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
        },
        model="audit-model",
        provider="other",
        primary_model="deepseek-v4-pro",
        primary_provider="deepseek",
    )

    assert decision.accepted is True
    assert "fidelity_judge_override_pass" in {issue.code for issue in decision.warnings}
    assert decision.independence == "strong"


def test_literary_hyphens_do_not_count_as_lost_formulas():
    decision = assess_fidelity(
        "A hereditary estate—co-heirs disputed it—and a three-month rental.",
        "Una propiedad hereditaria, disputada por coherederos, y una renta de tres meses.",
        chunk_index=7,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "formula_fragments_lost" not in {issue.code for issue in decision.issues}


def test_prompt_and_retry_options_are_auditor_specific():
    decision = assess_fidelity(
        "The prisoners were massacred after surrendering.",
        "Los prisioneros murieron despues.",
        chunk_index=7,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )
    prompt = build_fidelity_audit_prompt(
        "Source text",
        "Texto candidato",
        source_language="English",
        target_language="Spanish",
        phase="translation",
        local_decision=decision,
    )
    retry_options = build_fidelity_retry_prompt_options(
        {"custom_instructions": "Usa espanol mexicano."},
        decision,
    )

    assert "independent bilingual fidelity auditor" in prompt.system
    assert "Do not improve the translation" in prompt.system
    assert "<FIDELITY_AUDIT_JSON>" in prompt.system
    assert "FIDELITY RETRY" in retry_options["custom_instructions"]
    assert "Usa espanol mexicano." in retry_options["custom_instructions"]


def test_adjudication_prompt_allows_standard_localization_and_obvious_source_typos():
    prompt = build_fidelity_audit_prompt(
        "Apollo 11 launched. Liftoff on Apollo 111.",
        "El Apolo 11 despegó. Despegue del Apolo 11.",
        source_language="English",
        target_language="Spanish",
        phase="final_epub_unit_adjudication",
        prior_assessment={
            "judge_decision": "fail",
            "judge_reason": "Apollo was localized and an obvious typo was corrected.",
        },
    )

    assert "localized spellings" in prompt.system
    assert "isolated obvious source" in prompt.system
    assert "PRIOR_AUDIT as an untrusted opinion" in prompt.system
    assert "PRIOR AUDIT TO ADJUDICATE" in prompt.user


def test_profile_fidelity_context_separates_approved_rules_from_pending_hints(monkeypatch):
    approved = SimpleNamespace(
        source="Terry Southern",
        target="Terry Southern",
        approved=True,
        pending=False,
        confidence=0.99,
        translation_policy="preserve_exact",
        injection_policy="preserve",
        entry_type="proper_noun",
    )
    pending = SimpleNamespace(
        source="Eyes Wide Shut",
        target="Ojos bien cerrados",
        approved=False,
        pending=True,
        confidence=1.0,
        translation_policy="",
        injection_policy="",
        entry_type="title",
    )
    unused_pending = SimpleNamespace(
        source="The Shining",
        target="El resplandor",
        approved=False,
        pending=True,
        confidence=0.99,
        translation_policy="",
        injection_policy="",
        entry_type="title",
    )
    profile = SimpleNamespace(
        profile_id="book_a",
        glossary_entries=(approved, pending, unused_pending),
        approved_entries=(approved,),
    )
    monkeypatch.setattr(
        "src.core.book_profiles.loader.load_book_profile",
        lambda profile_id: profile if profile_id == "book_a" else None,
    )
    monkeypatch.setattr(
        "src.core.book_profiles.rendering.load_book_profile",
        lambda profile_id: profile if profile_id == "book_a" else None,
    )
    monkeypatch.setattr(
        "src.core.book_profiles.rendering.profile_terms_dict",
        lambda _options, source_text="": {"Terry Southern": "Terry Southern"},
    )

    context = _profile_fidelity_audit_context(
        "Terry Southern wrote about Eyes Wide Shut and The Shining.",
        "Terry sur escribió sobre Ojos bien cerrados y otra película.",
        {"editorial_mode": "book_profile", "profile_id": "book_a"},
    )
    prompt = build_fidelity_audit_prompt(
        "source",
        "candidate",
        profile_context=context,
    )

    assert 'APPROVED [preserve_exact]: "Terry Southern" -> "Terry Southern"' in context
    assert 'PENDING HINT confidence=1.00 [title]: "Eyes Wide Shut" -> "Ojos bien cerrados"' in context
    assert "The Shining" not in context
    assert context in prompt.user


def test_report_writes_markdown(tmp_path):
    report = FidelityReport(
        document_name="book.txt",
        source_language="English",
        target_language="Spanish",
        translator_model="deepseek-v4-pro",
        auditor_model="deepseek-v4-pro",
        translator_provider="deepseek",
        auditor_provider="deepseek",
    )
    report.add(assess_fidelity(
        "The expedition began in 1596 and returned with 17 survivors.",
        "La expedicion comenzo en 1596.",
        chunk_index=8,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    ))

    path = report.write(fidelity_report_path(tmp_path / "book.txt"))
    markdown = path.read_text(encoding="utf-8")

    assert path.name == "book - reporte fidelidad.md"
    assert "Reporte de fidelidad al original" in markdown
    assert "numbers_lost" in markdown


def test_fidelity_gate_rejects_lowercase_source_pronoun_inside_translation():
    decision = assess_fidelity(
        'On Capra, "Together we..." is discussed in the cited autobiography.',
        'Sobre Capra, «Juntos we…» se analiza en la autobiografía citada.',
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    issue = next(
        item
        for item in decision.rejections
        if item.code == "source_language_residual"
    )
    assert "we (1.00)" in issue.detail


def test_fidelity_gate_ignores_source_pronoun_inside_preserved_url():
    source = (
        '"I hope for us to": OpenAI, "Elon Musk Wanted an OpenAI For-Profit," '
        "December 13, 2024, openai.com/index/elon-musk-wanted-an-openai-for-"
        "profit/#summer-2017-we-and-elon-agreed-that-a-for-profit-was-the-"
        "next-step-for-openai-to-advance-the-mission"
    )
    candidate = (
        "«Espero que nosotros»: OpenAI, «Elon Musk Wanted an OpenAI "
        "For-Profit», 13 de diciembre de 2024, "
        "openai.com/index/elon-musk-wanted-an-openai-for-profit/"
        "#summer-2017-we-and-elon-agreed-that-a-for-profit-was-the-next-"
        "step-for-openai-to-advance-the-mission"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert not any(
        item.code == "source_language_residual"
        and "tipo=short_source_pronoun" in item.detail
        for item in decision.rejections
    )


def test_fidelity_gate_preserves_title_with_capitalized_source_pronoun():
    decision = assess_fidelity(
        "She recommended We Need to Talk About Kevin to the class.",
        "Recomendó We Need to Talk About Kevin al grupo.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_defers_translated_critical_apparatus_registry():
    source = (
        "Kenneth J. Harris, K. Michel Kacmer, Suzanne Zivnuska, and Jason D. "
        'Shaw (2007), “The Impact of Political Skill on Impression Management '
        'Effectiveness,” Journal of Applied Psychology, 92 (1), 278–285.[id0]'
        "Darren C. Treadway, Gerald R. Ferris, Allison B. Duke, Garry L. Adams, "
        'and Jason B. Thatcher (2007), “The Moderating Role of Subordinate '
        'Political Skill on Supervisors’ Impressions of Subordinate '
        'Ingratiation,” Journal of Applied Psychology, 92 (3), 848–855; '
        "quote is from p. 850.[id1]"
    )
    candidate = (
        "Kenneth J. Harris, K. Michel Kacmer, Suzanne Zivnuska y Jason D. "
        'Shaw (2007), «The Impact of Political Skill on Impression Management '
        'Effectiveness», Journal of Applied Psychology, 92 (1), 278–285.[id0]'
        "Darren C. Treadway, Gerald R. Ferris, Allison B. Duke, Garry L. Adams "
        'y Jason B. Thatcher (2007), «The Moderating Role of Subordinate '
        'Political Skill on Supervisors’ Impressions of Subordinate '
        'Ingratiation», Journal of Applied Psychology, 92 (3), 848–855; '
        "la cita es de la p. 850.[id1]"
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert "untranslated_source" not in {
        item.code for item in decision.rejections
    }
    assert "target_language_missing" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_defers_identity_data_in_translated_glossary():
    source = (
        "Each entry ends with a book and line reference. "
        "Acastus a-kas´-tus ): king of Dulichium.14.340. "
        "Achaean a-kee´-an ): inhabitants of Achaea.1.272. "
        "Heracles.2.120. Alector al-ek´-tor ): father of Leonteus.4.10."
    )
    candidate = (
        "Cada entrada termina con una referencia al libro y la línea. "
        "Acasto a-kas´-tus ): rey de Duliquio.14.340. "
        "Aqueo a-kee´-an ): habitantes de Acaya.1.272. "
        "Heracles.2.120. Alector al-ek´-tor ): padre de Leonteo.4.10."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "glossary"},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_infers_translated_glossary_in_final_aggregate():
    source = (
        "GLOSSARY AND INDEX. Each entry ends with a book and line reference. "
        "Acastus a-kas-tus: king of Dulichium. 14.340. "
        "Achaean a-kee-an: inhabitants of Achaea. 1.272. "
        "Alector al-ek-tor: father of Leonteus. 4.10."
    )
    candidate = (
        "GLOSARIO E ÍNDICE. Cada entrada termina con una referencia al libro y "
        "la línea. Acasto a-kas-tus: rey de Duliquio. 14.340. "
        "Aqueo a-kee-an: habitantes de Acaya. 1.272. "
        "Alector al-ek-tor: padre de Leonteo. 4.10."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="epub_publication",
        source_language="English",
        target_language="Spanish",
        prompt_options={"target_language_gate": True},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_keeps_pronunciation_examples_in_glossary():
    source = (
        "PRONUNCIATION KEY a as in cat ah as in father ai as in light "
        "u as in us you as in you zh as in vision; the mark identifies stress."
    )
    candidate = (
        "CLAVE DE PRONUNCIACIÓN: a como en cat, ah como en father, "
        "ai como en light, u como en us, you como en you y zh como en vision; "
        "la marca identifica la sílaba acentuada."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "glossary"},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_accepts_short_translated_glossary_definition():
    decision = assess_fidelity(
        "Aretias ( a-ree-tee-as ): grandfather of Amphinomus. 18.414.",
        "Aretias ( a-ree-tee-as ): abuelo de Anfínomo. 18.414.",
        chunk_index=1,
        phase="epub_publication_block",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "glossary"},
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_rejects_untranslated_short_glossary_definition():
    source = "Aretias ( a-ree-tee-as ): grandfather of Amphinomus. 18.414."
    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="epub_publication_block",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "glossary"},
    )

    assert decision.rejections
    assert {item.code for item in decision.rejections} & {
        "untranslated_source",
        "target_language_missing",
        "source_language_residual",
    }


def test_target_language_gate_still_rejects_source_pronoun_in_narrative():
    decision = assess_fidelity(
        "They told you that you should wait for the train.",
        "Ellos dijeron que you debía esperar el tren porque ya era tarde.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_rejects_exact_critical_apparatus_echo():
    source = (
        "Chapter 3. On Marcus Loew, see Robert Sobel, Marcus Loew: An Artist in "
        "Spite of Himself, in The Entrepreneurs: Explorations Within the "
        "American Business Tradition (1974)."
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert "target_language_missing" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_exact_citation_only_bibliography():
    source = (
        "Interpersonal Peacemaking: Confrontations and Third-Party Consultation. "
        "Reading, Mass.: Addison-Wesley, 1969. "
        "Weisbord, Marvin. Discovering Common Ground: How Future Search "
        "Conferences Bring People Together to Achieve Breakthrough Innovation, "
        "Empowerment, Shared Vision, and Collaborative Action. "
        "San Francisco: Berrett-Koehler, 1992."
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert decision.accepted is True
    assert "untranslated_source" not in {
        item.code for item in decision.rejections
    }
    assert "target_language_missing" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_allows_legacy_identity_only_citation_record():
    source = "S. A. Handford, Penguin, 1951"

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert decision.accepted is True
    assert not decision.rejections


def test_target_language_gate_rejects_year_terminated_narrative_echo():
    source = "The firm moved to Boston, expanded rapidly, 1951."

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert decision.accepted is False
    assert "untranslated_source" in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_accepts_translated_qualifier_in_name_index():
    source = (
        "Coleridge, Hartley, 259\n"
        "Collingwood, Vice Admiral, 261,265\n"
        "Cranmer, Thomas, Archbishop of\n"
        "Canterbury, 96\n"
        "Cromwell, Oliver, 177"
    )
    candidate = source.replace("Archbishop of", "arzobispo de")

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )

    assert decision.accepted is True
    assert not {
        "untranslated_source",
        "target_language_missing",
    } & {item.code for item in decision.rejections}


def test_target_language_gate_does_not_hide_changed_name_in_index():
    source = (
        "Coleridge, Hartley, 259\n"
        "Collingwood, Vice Admiral, 261,265\n"
        "Cranmer, Thomas, Archbishop of\n"
        "Canterbury, 96\n"
        "Cromwell, Oliver, 177"
    )
    candidate = source.replace("Archbishop of", "arzobispo de").replace(
        "Coleridge",
        "Coleridgez",
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )

    assert decision.accepted is False
    assert {item.code for item in decision.rejections} & {
        "untranslated_source",
        "target_language_missing",
    }


def test_target_language_gate_accepts_exact_identity_index_with_roles():
    source = (
        "Elizabeth 1,149,156\n"
        "Elliott, Grace, 247\n"
        "Ellis, Lieutenant, 262\n"
        "Emily, Princess, 219\n"
        "Erpingham, Sir Thomas, 72\n"
        "Ferdinand, Archduke Franz, 441\n"
        "Flanders, Earl of, 45\n"
        "Firmont, Henry Essex Edgeworth de,"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )

    assert decision.accepted is True
    assert not decision.rejections


def test_target_language_gate_rejects_exact_subject_index_with_ordinary_words():
    source = (
        "battles, decisive, 45\n"
        "children, protection of, 83\n"
        "war, causes of, 107"
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )

    assert decision.accepted is False
    assert {item.code for item in decision.rejections} & {
        "untranslated_source",
        "target_language_missing",
    }


def test_target_language_gate_accepts_translated_all_caps_quote_after_name_index():
    source = (
        "Wavrin, Jehan de, 68[id0]Wellesley, Sir Arthur, 267[id1]"
        "Wellington, Duke of, 281,305[id2]Wendel, Else, 584[id3]"
        "Werth, Alexander, 576,598,600[id4]Wesley, John, 223[id5]"
        "Whitman, Walt, 371[id6]Wordsworth, Dorothy, 256[id7]"
        "Wordsworth, William, 256[id8]Yurovsky, Commandant, 485[id9]"
        "Zaharoff, Sir Basil, 444[id10]Zeiser, Benno, 575[id11]"
        "“A FOUND TREASURE...MAKES READING HISTORY A WONDROUS HOBBY!” "
        "[id12]Chicago Tribune[id13]"
    )
    candidate = source.replace(
        "A FOUND TREASURE...MAKES READING HISTORY A WONDROUS HOBBY!",
        "UN TESORO ENCONTRADO... LEER HISTORIA, UN PASATIEMPO MARAVILLOSO",
    )
    options = {"_document_block_context": "index"}

    assert _looks_like_preservable_name_index_echo(
        source,
        candidate,
        prompt_options=options,
    ) is True
    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options=options,
    )

    assert decision.accepted is True
    assert not decision.rejections


def test_name_index_exemption_requires_all_caps_quote_to_be_translated():
    source = (
        "Wavrin, Jehan de, 68[id0]Wellesley, Sir Arthur, 267[id1]"
        "Wellington, Duke of, 281,305[id2]Wendel, Else, 584[id3]"
        "Werth, Alexander, 576,598,600[id4]Wesley, John, 223[id5]"
        "Whitman, Walt, 371[id6]Wordsworth, Dorothy, 256[id7]"
        "Wordsworth, William, 256[id8]Yurovsky, Commandant, 485[id9]"
        "Zaharoff, Sir Basil, 444[id10]Zeiser, Benno, 575[id11]"
        "“A FOUND TREASURE...MAKES READING HISTORY A WONDROUS HOBBY!” "
        "[id12]Chicago Tribune[id13]"
    )
    partially_translated = source.replace("TREASURE", "TESORO")

    assert _looks_like_preservable_name_index_echo(
        source,
        partially_translated,
        prompt_options={"_document_block_context": "index"},
    ) is False


def test_target_language_gate_accepts_exact_index_with_missing_ocr_initials():
    source = (
        "Greenhalgh, John, 186[id0]Grenville, George, 231[id1]"
        "Hamilton, Lady Emma, 263[id2]Hemingway, Ernest, 497[id3]"
        ". ardine, Douglas, 505[id4]. enkins, David, 122[id5]"
        ". evtic, Borijove, 441[id6]Hillary, Sir Edmund, 660[id7]"
        "Hitler, Adolf, 507,626,641[id8]Hugo, Victor, 328[id9]"
        "ahangir, the Great Mogul, 168,171[id10]ames 1,172[id11]"
    )
    options = {"_document_block_context": "index"}

    assert _looks_like_preservable_name_index_echo(
        source,
        source,
        prompt_options=options,
    ) is True
    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options=options,
    )

    assert decision.accepted is True
    assert not decision.rejections

    semantic_source = source
    for index in range(12):
        semantic_source = semantic_source.replace(f"[id{index}]", " ")
    semantic_source += (
        " Morrison, lan. 559 N^xrleon, Bonaparte, 254,278,285 "
        "Miller, Webb, xxx, 495,501"
    )
    assert _looks_like_preservable_name_index_echo(
        semantic_source,
        semantic_source,
        prompt_options=options,
    ) is True

    multiline_source = source
    for index in range(12):
        multiline_source = multiline_source.replace(f"[id{index}]", "\n")
    assert _looks_like_preservable_name_index_echo(
        multiline_source,
        multiline_source,
        prompt_options=options,
    ) is True

    split_ocr_source = "\n".join(
        [
            "Schmeling, Max, 523",
            "Schnirdel, Hu Ider ike, 92",
            "Scot, Edmund, 159",
            "Scott, Captain Robert, 431",
        ]
    )
    assert _looks_like_preservable_name_index_echo(
        split_ocr_source,
        split_ocr_source.replace("\n", " "),
        prompt_options=options,
    ) is True


def test_index_ocr_initial_exemption_does_not_hide_subject_prose():
    source = (
        "Greenhalgh, John, 186[id0]Grenville, George, 231[id1]"
        ". children, protection of, 83[id2]Hugo, Victor, 328[id3]"
        "children, protection of, 83[id4]"
    )

    assert _looks_like_preservable_name_index_echo(
        source,
        source,
        prompt_options={"_document_block_context": "index"},
    ) is False


def test_target_language_gate_accepts_long_identity_only_index_chunk():
    source = "".join(
        f"Greenhalgh, John, {100 + index}[id{index}]"
        for index in range(70)
    )
    options = {"_document_block_context": "index"}

    assert _looks_like_preservable_name_index_echo(
        source,
        source,
        prompt_options=options,
    ) is True
    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options=options,
    )

    assert decision.accepted is True
    assert not decision.rejections


def test_target_language_gate_allows_metadata_localization_in_citation_only_bibliography():
    source = (
        "Weisbord, Marvin, and Janoff, Sandra. Future Search: Getting the Whole "
        "System in the Room for Vision, Commitment, and Action. (Rev. ed.) "
        "San Francisco: Berrett-Koehler, 2010. "
        "Whitmore, John. Coaching for Performance: Growing Human Potential and "
        "Purpose—the Principles and Practice of Coaching and Leadership. "
        "(4th rev. ed.) London: Nicholas Brealey, 2009."
    )
    candidate = (
        "Weisbord, Marvin, y Janoff, Sandra. Future Search: Getting the Whole "
        "System in the Room for Vision, Commitment, and Action. (Ed. rev.) "
        "San Francisco: Berrett-Koehler, 2010. "
        "Whitmore, John. Coaching for Performance: Growing Human Potential and "
        "Purpose—the Principles and Practice of Coaching and Leadership. "
        "(4.ª ed. rev.) Londres: Nicholas Brealey, 2009."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert decision.accepted is True
    assert "untranslated_source" not in {
        item.code for item in decision.rejections
    }
    assert "target_language_missing" not in {
        item.code for item in decision.rejections
    }
    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_target_language_gate_rejects_exact_bibliographic_intro_prose():
    source = (
        "Visit www.example.com for more suggested reading and updates. "
        "Weisbord, Marvin. Discovering Common Ground. "
        "San Francisco: Berrett-Koehler, 1992."
    )

    decision = assess_fidelity(
        source,
        source,
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "critical_apparatus"},
    )

    assert decision.accepted is False
    assert "untranslated_source" in {
        item.code for item in decision.rejections
    }


def test_glossary_translation_does_not_rewrite_citation_institution_identity():
    source = (
        "Jeffrey Pfeffer, “Strategy in Action,” Case #OB95, Stanford, CA: "
        "Graduate School of Business, Stanford University, November 30, 2018; "
        "quote is from p. 13."
    )
    candidate = (
        "Jeffrey Pfeffer, «Strategy in Action», caso #OB95, Stanford, CA: "
        "Graduate School of Business, Stanford University, 30 de noviembre de "
        "2018; la cita es de la p. 13."
    )

    decision = assess_fidelity(
        source,
        candidate,
        chunk_index=1,
        phase="final_epub_unit_audit",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "_document_block_context": "critical_apparatus",
            "glossary_terms": {
                "Graduate School of Business": "Escuela de Posgrado en Negocios",
            },
        },
    )

    assert "source_language_residual" not in {
        item.code for item in decision.rejections
    }


def test_glossary_translation_remains_binding_in_ordinary_prose():
    decision = assess_fidelity(
        "She joined the Graduate School of Business as a professor.",
        "Se incorporó a Graduate School of Business como profesora.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options={
            "glossary_terms": {
                "Graduate School of Business": "Escuela de Posgrado en Negocios",
            },
        },
    )

    assert "source_language_residual" in {
        item.code for item in decision.rejections
    }
