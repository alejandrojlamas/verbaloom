from pathlib import Path

from src.core.literary_continuity import (
    LiteraryContinuityState,
    build_literary_continuity_block,
    detect_text_profile,
    extract_literary_names,
    export_literary_continuity_state,
    import_literary_continuity_state,
    literary_continuity_report_path,
    observe_literary_continuity,
    should_enable_literary_continuity,
    write_literary_continuity_report,
)
from src.prompts.prompts import (
    generate_refinement_prompt,
    generate_translation_prompt,
)


LITERARY_SAMPLE = """
Chapter 1

Ada Lovelace stood at the window while Charles Babbage waited by the door.
"You promised the engine would sing," Ada said.
Charles looked away. "It will, if the city lets us survive the night."

The rain pressed against the glass and the house seemed to hold its breath.
"""


TECHNICAL_SAMPLE = """
The Transformer architecture uses scaled dot-product attention. The model
computes softmax(QK^T / sqrt(d_k))V and is trained on WMT 2014 with BLEU
metrics, GPUs, batch normalization, and optimizer schedules.
"""


def test_detect_text_profile_identifies_literature():
    profile = detect_text_profile(LITERARY_SAMPLE)

    assert profile.kind == "literature"
    assert profile.confidence >= 0.45
    assert profile.signals["dialogue_marks"] > 0
    assert "Ada Lovelace" in extract_literary_names(LITERARY_SAMPLE)


def test_auto_mode_does_not_enable_for_technical_text():
    prompt_options = {"literary_continuity": True, "text_type": "auto"}

    assert should_enable_literary_continuity(prompt_options, TECHNICAL_SAMPLE) is False


def test_literature_mode_can_be_forced_even_when_sample_is_short():
    prompt_options = {"literary_continuity": True, "text_type": "literature"}

    assert should_enable_literary_continuity(prompt_options, "Ada waited.") is True


def test_continuity_block_is_compact_and_uses_runtime_state():
    runtime_state = {}
    prompt_options = {
        "literary_continuity": True,
        "text_type": "literature",
        "continuity_max_prompt_chars": 700,
    }
    block = build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text=LITERARY_SAMPLE,
        source_language="English",
        target_language="Spanish",
    )

    assert "LITERARY CONTINUITY MEMORY" in block
    assert len(block) <= 705

    observe_literary_continuity(
        runtime_state=runtime_state,
        source_text=LITERARY_SAMPLE,
        translated_text="Ada Lovelace miró a Charles Babbage y recordó la promesa.",
        section="Chapter 1",
    )

    state = runtime_state["literary_continuity_state"]
    assert isinstance(state, LiteraryContinuityState)
    assert state.chunk_index == 1
    assert "Ada Lovelace" in state.characters

    next_block = build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text="Ada Lovelace crossed the street.",
        source_language="English",
        target_language="Spanish",
    )
    assert "Recent continuity" in next_block
    assert "Ada Lovelace" in next_block


def test_prompt_injects_continuity_in_user_prompt_only():
    continuity_block = "# LITERARY CONTINUITY MEMORY\n- Ada Lovelace: preserve consistently"
    pair = generate_translation_prompt(
        main_content="Ada waited.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
        continuity_block=continuity_block,
    )

    assert continuity_block in pair.user
    assert continuity_block not in pair.system


def test_refinement_prompt_injects_continuity():
    continuity_block = "# LITERARY CONTINUITY MEMORY\n- Keep the narrator restrained."
    pair = generate_refinement_prompt(
        draft_translation="Ada esperó.",
        target_language="Spanish",
        has_placeholders=False,
        continuity_block=continuity_block,
    )

    assert continuity_block in pair.user
    assert continuity_block not in pair.system


def test_spanish_prompts_include_stable_literary_style_contract():
    translation = generate_translation_prompt(
        main_content='"I cannot go," Anna said.',
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
    )
    refinement = generate_refinement_prompt(
        draft_translation='"No puedo ir", dijo Anna.',
        target_language="Spanish",
        has_placeholders=False,
    )

    for pair in (translation, refinement):
        assert "LITERARY STYLE CONTRACT" in pair.system
        assert "Use the em dash/raya" in pair.system
        assert "Descripción de imagen" in pair.system
        assert "LITERARY STYLE CONTRACT" not in pair.user


def test_literary_style_contract_can_be_disabled_for_non_literary_spanish_jobs():
    pair = generate_translation_prompt(
        main_content='"I cannot go," Anna said.',
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="English",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={"literary_style_contract": False},
    )

    assert "LITERARY STYLE CONTRACT" not in pair.system


def test_writes_literary_continuity_report(tmp_path: Path):
    runtime_state = {}
    prompt_options = {"literary_continuity": True, "text_type": "literature"}
    build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text=LITERARY_SAMPLE,
        source_language="English",
        target_language="Spanish",
    )
    observe_literary_continuity(
        runtime_state=runtime_state,
        source_text=LITERARY_SAMPLE,
        translated_text="Ada Lovelace habló con Charles Babbage.",
        section="Chapter 1",
    )

    output = tmp_path / "novel.txt"
    output.write_text("translated", encoding="utf-8")
    report = write_literary_continuity_report(output, runtime_state)

    assert report == literary_continuity_report_path(output)
    assert report.exists()
    content = report.read_text(encoding="utf-8")
    assert "Reporte de continuidad literaria" in content
    assert "Ada Lovelace" in content


def test_aliases_merge_titles_and_possessives():
    state = LiteraryContinuityState(max_prompt_tokens=220)
    state.observe("Captain Ahab watched the sea. Ahab's hand shook.", "Captain Ahab observó el mar.", section="CHAPTER 1")

    assert len(state.characters) == 1
    entity = next(iter(state.characters.values()))
    assert entity.canonical_source == "Captain Ahab"
    assert "Ahab" in entity.aliases


def test_false_positive_sentence_starters_are_filtered():
    text = "There was rain. Though the night was cold, Besides nothing moved. Where was the lamp?"

    names = extract_literary_names(text, max_names=20)

    assert "There" not in names
    assert "Though" not in names
    assert "Besides" not in names
    assert "Where" not in names


def test_token_budget_omits_lines_without_hard_truncation():
    runtime_state = {}
    prompt_options = {
        "literary_continuity": True,
        "text_type": "literature",
        "continuity_max_prompt_tokens": 120,
    }
    build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text=LITERARY_SAMPLE,
        source_language="English",
        target_language="Spanish",
    )

    for idx in range(40):
        name = f"Person {chr(65 + (idx % 26))}{chr(97 + (idx // 26))}a"
        observe_literary_continuity(
            runtime_state=runtime_state,
            source_text=f"{name} met Captain Ahab in Chapter {idx}.",
            translated_text=f"{name} se reunió con Captain Ahab.",
            section=f"CHAPTER {idx}",
        )

    block = build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text="Captain Ahab returned.",
        source_language="English",
        target_language="Spanish",
    )
    state = runtime_state["literary_continuity_state"]

    assert not block.rstrip().endswith("...")
    assert state.last_render_stats["tokens"] <= 120
    assert state.last_render_stats["omitted_lines"] > 0


def test_serialization_roundtrip_preserves_memory():
    runtime_state = {}
    prompt_options = {"literary_continuity": True, "text_type": "literature"}
    build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text=LITERARY_SAMPLE,
        source_language="English",
        target_language="Spanish",
    )
    observe_literary_continuity(
        runtime_state=runtime_state,
        source_text="Captain Ahab spoke to Queequeg.",
        translated_text="Captain Ahab habló con Queequeg.",
        section="CHAPTER 1",
    )

    serialized = export_literary_continuity_state(runtime_state)
    restored_runtime = {}
    import_literary_continuity_state(restored_runtime, serialized)
    restored = restored_runtime["literary_continuity_state"]

    assert restored.chunk_index == 1
    assert "Captain Ahab" in restored.characters
    assert "Queequeg" in restored.characters


def test_glossary_boosts_confirmed_entities():
    runtime_state = {}
    prompt_options = {
        "literary_continuity": True,
        "text_type": "literature",
        "glossary_terms": {"The White Whale|White Whale": "la Ballena Blanca"},
        "glossary_term_metadata": {"The White Whale|White Whale": {"category": "item"}},
    }

    block = build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text="Ahab dreamed of the White Whale.",
        source_language="English",
        target_language="Spanish",
    )

    state = runtime_state["literary_continuity_state"]
    entity = next(e for e in state.characters.values() if e.target == "la Ballena Blanca")
    assert entity.glossary_confirmed is True
    assert entity.category == "item"
    assert "la Ballena Blanca" in block
