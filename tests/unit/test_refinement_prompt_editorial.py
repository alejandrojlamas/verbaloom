from src.prompts.prompts import generate_refinement_prompt


def test_refinement_prompt_is_editorial_and_conservative():
    prompt = generate_refinement_prompt(
        draft_translation="This sentence is already clear.",
        target_language="English",
        has_placeholders=False,
    )

    assert "FINAL EDITORIAL REVIEW" in prompt.system
    assert "Correct grammar, punctuation, style, and mechanical defects only where needed" in prompt.user
    assert "leave it close to unchanged when it is already good" in prompt.system
    assert "Editorial restraint is part of the task" in prompt.system
    assert "OCR artifacts" in prompt.system
