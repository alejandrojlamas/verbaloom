from src.prompts.prompts import (
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
