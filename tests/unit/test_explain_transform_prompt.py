from src.config import INPUT_TAG_IN
from src.prompts.prompts import (
    build_text_transform_instructions,
    generate_refinement_prompt,
    generate_translation_prompt,
)


def test_simplify_mode_is_university_level_explain_mode():
    instructions = build_text_transform_instructions(
        {"text_transform_mode": "simplify"},
        "Spanish",
    )

    assert "TEXT TRANSFORMATION MODE: Explicar" in instructions
    assert "university-level prose" in instructions
    assert "common language" in instructions
    assert "complex ideas, tables, formulas, and arguments" in instructions
    assert "integrate supported explanations into the prose" in instructions
    assert "do not invent facts, numbers, citations, examples, or conclusions" in instructions
    assert "Do not turn the output into a study guide" in instructions


def test_explain_translation_prompt_allows_integrated_explanation_not_notes():
    prompt = generate_translation_prompt(
        main_content="The ablation table shows the model loses 4.2 F1 when attention is removed.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="Spanish",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={"text_transform_mode": "simplify"},
    )

    assert "mode: Explicar" in prompt.system
    assert "integrated explanatory rewriting is allowed" in prompt.system
    assert "Do not add explanations, notes, summaries" not in prompt.system
    assert "Do NOT add meta-explanations" in prompt.system


def test_explain_refinement_prompt_keeps_explanations_inside_prose():
    prompt = generate_refinement_prompt(
        draft_translation="La fórmula x = y + z combina dos señales latentes.",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={"text_transform_mode": "simplify"},
    )

    assert "same-language transformation: Explicar" in prompt.system
    assert "supported explanations must be integrated into the prose" in prompt.system
    assert "Free of explanations, notes, summaries" not in prompt.system
    assert "Formula handling" in prompt.system


def test_translation_prompt_adds_context_for_table_continuations_only_outside_input():
    prompt = generate_translation_prompt(
        main_content="49±. 0\nGPT-2 L (AdapterL) 23. 00M 68. 9±. 3",
        context_before="Model & Method # Trainable E2E NLG Challenge\nParameters BLEU NIST MET ROUGE-L CIDEr",
        context_after="Table 3: GPT-2 medium and large with different adaptation methods",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={},
    )

    assert "# SURROUNDING SOURCE CONTEXT" in prompt.user
    assert "Do not translate, transform, copy, summarize, or output this context" in prompt.user
    assert prompt.user.index("# SURROUNDING SOURCE CONTEXT") < prompt.user.index("# TEXT TO TRANSLATE")
    assert prompt.user.index("Model & Method") < prompt.user.index(INPUT_TAG_IN)
    assert "GPT-2 L (AdapterL) 23. 00M" in prompt.user


def test_translation_prompt_omits_context_for_plain_prose():
    prompt = generate_translation_prompt(
        main_content="This paragraph explains the central argument without tables or formulas.",
        context_before="The previous paragraph is ordinary prose.",
        context_after="The next paragraph is ordinary prose too.",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={},
    )

    assert "# SURROUNDING SOURCE CONTEXT" not in prompt.user


def test_explain_refinement_prompt_adds_table_context_outside_input():
    prompt = generate_refinement_prompt(
        draft_translation="0.55\n0.60\n0.65\n0.70\n0.75Exactitud de validación",
        context_before="Figura 2: rendimiento de métodos de adaptación",
        context_after="WikiSQL\nMétodo\nFine-Tune\nPrefixEmbed\nPrefixLayer",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={"text_transform_mode": "simplify"},
    )

    assert "# SURROUNDING DRAFT CONTEXT" in prompt.user
    assert prompt.user.index("# SURROUNDING DRAFT CONTEXT") < prompt.user.index("# TEXT TO TRANSFORM")
    assert prompt.user.index("Figura 2") < prompt.user.index(INPUT_TAG_IN)
