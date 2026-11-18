import pytest

from src.prompts.prompts import (
    generate_post_processing_prompt,
    generate_refinement_prompt,
    generate_subtitle_block_prompt,
    generate_subtitle_refinement_block_prompt,
    generate_translation_prompt,
)


def test_translation_prompt_defaults_to_mexican_spanish_for_spanish_target():
    prompt = generate_translation_prompt(
        main_content="Hello.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
    )

    assert "Mexican Spanish" in prompt.system
    assert "professional Mexican Spanish translator" in prompt.system
    assert "neutral Latin American Spanish" in prompt.system
    assert "avispaos" in prompt.system
    assert "vosotras" in prompt.system
    assert "precipitaos" in prompt.system
    assert "elevated Mexican Spanish" in prompt.system
    assert "ustedes" in prompt.system
    assert "original numeric value and unit" in prompt.system
    assert "must not replace or round away" in prompt.system
    assert "Translate source-language chapter titles" in prompt.system
    assert "preserve the complete authentic title" in prompt.system
    assert "translated or mixed-language bibliographic title" in prompt.system
    assert "social-media posts" in prompt.system


def test_refinement_prompt_defaults_to_mexican_spanish_for_spanish_target():
    prompt = generate_refinement_prompt(
        draft_translation="Chitón, que viene hacia aquí.",
        target_language="Spanish",
        has_placeholders=False,
    )

    assert "Mexican Spanish" in prompt.system
    assert "elite Mexican Spanish literary editor" in prompt.system
    assert "cállense/guarden silencio" in prompt.system


def test_subtitle_prompts_default_to_mexican_spanish_for_spanish_target():
    translation = generate_subtitle_block_prompt(
        [(0, "Look sharp!")],
        previous_translation_block="",
        source_language="English",
        target_language="Spanish",
    )
    refinement = generate_subtitle_refinement_block_prompt(
        [(0, "¡Avispaos!")],
        target_language="Spanish",
    )

    assert "Mexican Spanish" in translation.system
    assert "Mexican Spanish" in refinement.system


def test_non_spanish_prompt_does_not_get_spanish_locale_rules():
    prompt = generate_translation_prompt(
        main_content="Hello.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="French",
        has_placeholders=False,
    )

    assert "Mexican Spanish" not in prompt.system


def test_auto_source_language_prompt_uses_passage_language_label():
    prompt = generate_translation_prompt(
        main_content="Μῆνιν ἄειδε θεὰ.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="Auto",
        target_language="Spanish",
        has_placeholders=False,
    )

    assert "Translate Auto to Spanish" not in prompt.system
    assert "Translate the source language of each passage to Spanish" in prompt.system
    assert "Do NOT write in the source language of each passage" in prompt.system


def test_spanish_locale_rules_can_be_disabled():
    prompt = generate_translation_prompt(
        main_content="Hello.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={"spanish_variant": "generic"},
    )

    assert "Mexican Spanish" not in prompt.system
    assert "professional Spanish translator" in prompt.system


def test_literary_polish_preserves_intentional_period_voice_in_mexican_spanish():
    prompt = generate_refinement_prompt(
        draft_translation="Hízome entonces el escribano una seña.",
        target_language="Spanish",
        prompt_options={"text_transform_mode": "literary_polish", "spanish_variant": "mexican"},
    )

    assert "Preserve an intentional period voice" in prompt.system
    assert "remove every Peninsular/archaic Spanish form" not in prompt.system


@pytest.mark.parametrize("target", ["Spanish", "es-MX"])
def test_mexican_adaptation_uses_one_editorial_policy_in_writing_and_refinement(target):
    options = {"text_transform_mode": "mexican_spanish", "spanish_variant": "neutral"}
    source = "[id0]—Ya ni la amuelas —dijo Nora. El patio seguía a oscuras.[id1]"
    writing = generate_translation_prompt(
        source, "", "", "", source_language="Spanish", target_language=target,
        prompt_options=options, has_placeholders=True, placeholder_format=("[id", "]"),
    )
    refinement = generate_refinement_prompt(
        source, target_language=target, prompt_options=options, has_placeholders=False,
    )
    epub = generate_post_processing_prompt(
        source, target_language=target, prompt_options=options, has_placeholders=True,
        placeholder_format=("[id", "]"),
    )

    for request in (writing, refinement, epub):
        assert request.system.count("contemporary editorial Mexican Spanish (es-MX)") == 1
        assert "neutral Latin American Spanish" not in request.system
        assert "rebuild sentences or passages substantially" in request.system
        assert "Change clause order, sentence rhythm" in request.system
        assert "Recast the line to do that in the character's own voice" in request.system
        assert "Adapt the dialogue turn as a unit" in request.system
        assert "kinship terms, forms of address" in request.system
        assert "fresh version as the character would say it" in request.system
        assert "Carry each metaphor through in Spanish" in request.system
        assert "Preserve verbatim quotations from documents and real cited sources" in request.system
        assert "Preserve direct quotations and dialogue treatment" not in request.system
        assert "same actions, who performs them" in request.system
        assert "Mexican colloquialisms, slang or profanity" in request.system
        assert "speaker, relationship, social register, era and scene" in request.system
        assert "Keep effective passages verbatim" in request.system
        assert "irony, ambiguity and factual details" in request.system
        assert source in request.user
        assert "Review every spoken turn before returning the text." in request.user
    assert "placeholder markers exactly" in writing.system
    assert "placeholder" in epub.system.lower()


@pytest.mark.parametrize("old_description", [
    "Adapt into natural Mexican or Latin American Spanish.",
    "Adapta a español mexicano o latinoamericano natural.",
])
def test_mexican_adaptation_ignores_only_legacy_generated_descriptions(old_description):
    options = {"text_transform_mode": "mexican_spanish"}
    prompt = generate_refinement_prompt(
        "El patio seguía a oscuras.", target_language="Spanish",
        prompt_options=options, additional_instructions=old_description,
    )
    custom = generate_refinement_prompt(
        "El patio seguía a oscuras.", target_language="Spanish",
        prompt_options=options, additional_instructions="Keep the narrator's long pauses.",
    )

    assert old_description not in prompt.system
    assert "Keep the narrator's long pauses." in custom.system
