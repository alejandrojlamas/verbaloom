from __future__ import annotations

import pytest

from src.core.llm.base import LLMResponse
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.xhtml_translator import translate_chunk_with_fallback
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.translator import (
    _normalize_all_caps_prose_for_translation,
    generate_translation_request,
)


class SequenceClient:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def generate(self, prompt: str, system_prompt: str = None, temperature=None):
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt or "")
        return LLMResponse(content=self.outputs.pop(0))

    def extract_translation(self, response: str):
        start = response.find("<TRANSLATION>")
        end = response.find("</TRANSLATION>")
        if start < 0 or end < 0:
            return None
        return response[start + len("<TRANSLATION>"):end]


@pytest.mark.asyncio
async def test_translation_repairs_duplicate_elision_before_quality_gates():
    source = (
        "Then came a feature film, [id0]L’Atalante[id1], which opened as Vigo "
        "was dying. It remains one of the most poetic love stories ever filmed."
    )
    client = SequenceClient([
        (
            "<TRANSLATION>Luego vino un largometraje, [id0]L'L'Atalante[id1], "
            "que se estrenó mientras Vigo moría. Sigue siendo una de las historias "
            "de amor más poéticas jamás filmadas.</TRANSLATION>"
        ),
    ])

    translated = await generate_translation_request(
        main_content=source,
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        model="deepseek-v4-pro",
        llm_client=client,
        has_placeholders=True,
        placeholder_format=("[id", "]"),
        prompt_options={"quality_alert_model": "same"},
    )

    assert translated is not None
    assert "[id0]L'Atalante[id1]" in translated
    assert "L'L'Atalante" not in translated
    assert len(client.prompts) == 1


@pytest.mark.asyncio
async def test_translation_repairs_source_language_residue_before_outer_retry():
    source = (
        "Die Wirklichkeit dieser Geschichte blieb in seiner Erinnerung, und die "
        "Zerstörung der Gesellschaft erschien ihm unbegreiflich."
    )
    client = SequenceClient([
        (
            "<TRANSLATION>La Wirklichkeit de esta historia permaneció en su memoria, "
            "y la Zerstörung de la sociedad le parecía incomprensible.</TRANSLATION>"
        ),
        (
            "<TRANSLATION>La realidad de esta historia permaneció en su memoria, "
            "y la destrucción de la sociedad le parecía incomprensible.</TRANSLATION>"
        ),
    ])

    translated = await generate_translation_request(
        main_content=source,
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="German",
        target_language="Spanish",
        model="deepseek-v4-pro",
        llm_client=client,
        prompt_options={"quality_alert_model": "same"},
    )

    assert translated == (
        "La realidad de esta historia permaneció en su memoria, y la destrucción "
        "de la sociedad le parecía incomprensible."
    )
    assert len(client.prompts) == 2
    assert "TARGET-LANGUAGE GATE REPAIR" in client.system_prompts[1]


@pytest.mark.asyncio
async def test_translation_repairs_embedded_third_language_quote_as_bounded_span():
    source = (
        "The rabbi spoke in Italian to the visitor: ‘Ecco Signior mio, Un "
        "Miracolo di dio’; because the 7-year-old child immediately stopped crying."
    )
    candidate = (
        "El rabino habló en italiano al visitante: «Ecco Signior mio, Un "
        "Miracolo di dio»; porque el niño de 7 años dejó de llorar de inmediato."
    )
    client = SequenceClient([
        f"<TRANSLATION>{candidate}</TRANSLATION>",
        "<TRANSLATION>He aquí, señor mío, un milagro de Dios</TRANSLATION>",
    ])

    translated = await generate_translation_request(
        main_content=source,
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        model="deepseek-flash",
        llm_client=client,
        prompt_options={"quality_alert_model": "same"},
    )

    assert translated == (
        "El rabino habló en italiano al visitante: «He aquí, señor mío, un "
        "milagro de Dios»; porque el niño de 7 años dejó de llorar de inmediato."
    )
    assert "Ecco Signior" not in translated
    assert "7 años" in translated
    assert len(client.prompts) == 2
    assert "third language" in client.system_prompts[1]


def test_all_caps_retry_normalizes_prose_but_keeps_short_titles_and_names():
    source = (
        "[id0]INTO THE DEEP[id1]“AN ADVENTURE STORY, A ROMANCE, AND AN ECOLOGICAL "
        "WARNING...EXPLORES COMPASSION FOR HUMANS AND DOLPHINS.”[id2]"
        "—San Francisco Chronicle[id3]"
    )

    normalized = _normalize_all_caps_prose_for_translation(source)

    assert "INTO THE DEEP" in normalized
    assert "An adventure story, a romance" in normalized
    assert "San Francisco Chronicle" in normalized
    assert "AN ADVENTURE STORY" not in normalized


@pytest.mark.asyncio
async def test_translation_retries_all_caps_running_prose_in_sentence_case():
    source = (
        "[id0]INTO THE DEEP[id1]“AN ADVENTURE STORY, A ROMANCE, AND AN ECOLOGICAL "
        "WARNING...EXPLORES THE MAGIC OF COMPASSION FOR HUMANS AND DOLPHINS.”[id2]"
        "—San Francisco Chronicle[id3]"
    )
    translated = (
        "[id0]INTO THE DEEP[id1]«Una historia de aventuras, un romance y una "
        "advertencia ecológica... explora la magia de la compasión por los seres "
        "humanos y los delfines».[id2]—San Francisco Chronicle[id3]"
    )
    client = SequenceClient([
        f"<TRANSLATION>{source}</TRANSLATION>",
        f"<TRANSLATION>{translated}</TRANSLATION>",
    ])

    result = await generate_translation_request(
        main_content=source,
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        model="deepseek-v4-pro",
        llm_client=client,
        has_placeholders=True,
        placeholder_format=("[id", "]"),
        prompt_options={"quality_alert_model": "same"},
    )

    assert result == translated
    assert len(client.prompts) == 2
    assert "An adventure story, a romance" in client.prompts[1]
    assert "AN ADVENTURE STORY" not in client.prompts[1]


@pytest.mark.asyncio
async def test_epub_blocks_untranslated_fallback_after_gate_repair_fails(monkeypatch):
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)
    client = SequenceClient([
        "<TRANSLATION>[id0]La Wirklichkeit quedó intacta.</TRANSLATION>",
        "<TRANSLATION>[id0]La Wirklichkeit quedó intacta.</TRANSLATION>",
    ])
    stats = TranslationMetrics()

    with pytest.raises(RuntimeError, match="untranslated fallback was blocked"):
        await translate_chunk_with_fallback(
            chunk_text="[id0]Die Wirklichkeit blieb erhalten.",
            local_tag_map={"[id0]": "<p>"},
            global_indices=[0],
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=client,
            stats=stats,
            max_retries=1,
            placeholder_format=("[id", "]"),
            prompt_options={
                "_source_language": "German",
                "abort_on_profile_fail": True,
                "glossary_terms": {"Wirklichkeit": "realidad"},
                "quality_alert_model": "same",
            },
        )

    assert stats.failed_chunks == 1
    assert stats.fallback_used == 0


@pytest.mark.asyncio
async def test_epub_never_returns_source_text_when_every_attempt_fails(monkeypatch):
    """A provider/validation failure must fail the unit, never publish German."""
    monkeypatch.setattr("src.config.EPUB_TOKEN_ALIGNMENT_ENABLED", False)

    async def no_candidate(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.generate_translation_request",
        no_candidate,
    )
    stats = TranslationMetrics()

    with pytest.raises(RuntimeError, match="untranslated fallback was blocked"):
        await translate_chunk_with_fallback(
            chunk_text="[id0]Wir gingen am folgenden Morgen weiter.[id1]",
            local_tag_map={"[id0]": "<p>", "[id1]": "</p>"},
            global_indices=[0, 1],
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=object(),
            stats=stats,
            max_retries=2,
            placeholder_format=("[id", "]"),
            prompt_options={},
        )

    assert stats.failed_chunks == 1
    assert stats.fallback_used == 0


@pytest.mark.asyncio
async def test_epub_retries_accumulate_prior_fidelity_defects(monkeypatch):
    generated_prompt_options = []
    audit_calls = 0

    async def fake_generate(*_args, **kwargs):
        generated_prompt_options.append(dict(kwargs.get("prompt_options") or {}))
        return "[id0]Traduccion candidata completa."

    async def fake_supervise(*_args, **_kwargs):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 1:
            return FidelityDecision(
                chunk_index=8,
                phase="translation",
                section="chapter.xhtml",
                accepted=False,
                issues=[FidelityIssue("fidelity_judge_reject", "reject", "changed gender")],
                judge_changed_facts=[
                    "the landwalker Receptive is male, not female"
                ],
            ), None
        if audit_calls == 2:
            return FidelityDecision(
                chunk_index=8,
                phase="translation",
                section="chapter.xhtml",
                accepted=False,
                issues=[FidelityIssue("fidelity_judge_reject", "reject", "missing action")],
                judge_missing_from_source=[
                    "without thinking, without hoping"
                ],
            ), None
        return FidelityDecision(
            chunk_index=8,
            phase="translation",
            section="chapter.xhtml",
            accepted=True,
        ), None

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.generate_translation_request",
        fake_generate,
    )
    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.supervise_fidelity",
        fake_supervise,
    )

    result = await translate_chunk_with_fallback(
        chunk_text="[id0]Source passage.",
        local_tag_map={"[id0]": "<p>"},
        global_indices=[0],
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=object(),
        stats=TranslationMetrics(),
        max_retries=3,
        placeholder_format=("[id", "]"),
        prompt_options={"fidelity_supervisor": True},
    )

    assert result == "[id0]Traduccion candidata completa."
    third_instructions = generated_prompt_options[2]["custom_instructions"]
    assert "landwalker Receptive is male, not female" in third_instructions
    assert "without thinking, without hoping" in third_instructions
