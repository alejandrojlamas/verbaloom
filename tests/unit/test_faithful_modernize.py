import pytest

from src.core.editorial_quality import QualityDecision, QualityIssue, build_source_aware_guard_prompt
from src.core.fidelity_supervisor import FidelityDecision, build_fidelity_audit_prompt
from src.core.text_transform import (
    modernize_guard_findings,
    protect_meaningful_blocks,
    restore_protected_blocks,
)
from src.core.translator import refine_chunks
from src.prompts.prompts import build_text_transform_instructions, generate_refinement_prompt


def test_modernize_prompt_requires_current_book_readability_and_preserves_blocks():
    base_instructions = build_text_transform_instructions(
        {"text_transform_mode": "modernize"},
        "Spanish",
    )
    high_strength_instructions = build_text_transform_instructions(
        {"text_transform_mode": "modernize", "modernization_strength": "high"},
        "Spanish",
    )

    assert "read like a current published book" in base_instructions
    assert "Content fidelity is non-negotiable" in base_instructions
    assert "Preserve paragraph breaks" in base_instructions
    assert "obsolete verb morphology must change" in high_strength_instructions
    assert "[id900000]" in base_instructions


def test_modernize_spanish_locale_demands_current_editorial_dialogue():
    prompt = generate_refinement_prompt(
        draft_translation="Quoted historical dialogue.",
        target_language="Spanish",
        prompt_options={
            "spanish_variant": "mexican",
            "text_transform_mode": "modernize",
        },
    )

    assert "Forbidden output forms" not in prompt.system
    assert "Rewrite historically marked dialogue" in prompt.system
    assert "current publishable editorial Spanish" in prompt.system
    assert "not a light copyedit pass" in prompt.system


def test_modernize_translation_prompt_is_transformation_not_translation():
    from src.prompts.prompts import generate_translation_prompt

    prompt = generate_translation_prompt(
        main_content="No ha mucho tiempo que vivía un hidalgo.",
        context_before="",
        context_after="",
        previous_translation_context="",
        source_language="Spanish",
        target_language="Spanish",
        has_placeholders=False,
        prompt_options={
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
        },
    )

    assert "same-language transformation editor" in prompt.system
    assert "not a translation-between-languages prompt" in prompt.system
    assert "# TEXT TO TRANSFORM" in prompt.user


def test_modernize_local_guard_is_structure_only_not_lexical():
    findings = modernize_guard_findings(
        "A speaker addresses another person in a formal quotation.",
        "A speaker addresses another person in an informal quotation.",
        prompt_options={"text_transform_mode": "modernize"},
    )

    assert findings == []


def test_block_markers_restore_exact_block_order():
    source = "Capítulo primero.\n\nEn un lugar de la Mancha.\n\nFinal."

    protection = protect_meaningful_blocks(source)
    ok, restored, reason = restore_protected_blocks(
        "[id900000]\nCapítulo primero.\n\n"
        "[id900001]\nEn un lugar de la Mancha.\n\n"
        "[id900002]\nFinal.",
        protection,
    )

    assert ok is True
    assert reason == ""
    assert restored == source


def test_block_markers_reject_collapsed_or_missing_blocks():
    source = "Capítulo primero.\n\nEn un lugar de la Mancha."
    protection = protect_meaningful_blocks(source)

    ok, _restored, reason = restore_protected_blocks(
        "Capítulo primero. En un lugar de la Mancha.",
        protection,
    )

    assert ok is False
    assert "block marker sequence changed" in reason


def test_fidelity_auditor_prompt_handles_same_language_semantics_generically():
    prompt = build_fidelity_audit_prompt(
        "Original passage.",
        "Modernized passage.",
        source_language="Spanish",
        target_language="Spanish",
        phase="refinement",
    )

    assert "same-language transformations" in prompt.system
    assert "speaker/addressee relationships" in prompt.system
    assert "grammatical person" in prompt.system
    assert "quoted meaning" in prompt.system


def test_source_aware_guard_prompt_handles_same_language_semantics_generically():
    prompt = build_source_aware_guard_prompt(
        "Original passage.",
        "Initial draft.",
        "Modernized candidate.",
        source_language="Spanish",
        target_language="Spanish",
    )

    assert "same-language transformations" in prompt.system
    assert "speaker/addressee relations" in prompt.system
    assert "intentional ambiguity" in prompt.system


@pytest.mark.asyncio
async def test_modernize_refine_keeps_source_when_llm_drops_block_markers(monkeypatch):
    source = "Capítulo primero.\n\nEn un lugar de la Mancha."

    class FakeClient:
        async def close(self):
            pass

    async def fake_make_refinement_request(**_kwargs):
        return "Capítulo primero. En un lugar de la Mancha.", None

    async def fake_supervise_fidelity(*_args, **kwargs):
        return FidelityDecision(
            chunk_index=kwargs["chunk_index"],
            phase=kwargs["phase"],
            section=kwargs["section"],
            accepted=True,
        ), None

    monkeypatch.setattr("src.core.translator.create_llm_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr("src.core.translator._make_refinement_request", fake_make_refinement_request)
    monkeypatch.setattr("src.core.translator.supervise_fidelity", fake_supervise_fidelity)

    result = await refine_chunks(
        translated_chunks=[source],
        original_chunks=[{"context_before": "", "main_content": source, "context_after": ""}],
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.com/chat/completions",
        llm_provider="deepseek",
        prompt_options={
            "text_transform_mode": "modernize",
            "transform_repair_attempts": 0,
            "editorial_quality_report": False,
            "source_aware_editorial_guard": False,
        },
    )

    assert result == [source]


@pytest.mark.asyncio
async def test_modernize_refine_strips_valid_block_markers(monkeypatch):
    source = "Capítulo primero.\n\nEn un lugar de la Mancha."

    class FakeClient:
        async def close(self):
            pass

    async def fake_make_refinement_request(**_kwargs):
        return "[id900000]\nCapítulo primero.\n\n[id900001]\nEn un sitio de la Mancha.", None

    async def fake_supervise_fidelity(*_args, **kwargs):
        return FidelityDecision(
            chunk_index=kwargs["chunk_index"],
            phase=kwargs["phase"],
            section=kwargs["section"],
            accepted=True,
        ), None

    monkeypatch.setattr("src.core.translator.create_llm_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr("src.core.translator._make_refinement_request", fake_make_refinement_request)
    monkeypatch.setattr("src.core.translator.supervise_fidelity", fake_supervise_fidelity)

    result = await refine_chunks(
        translated_chunks=[source],
        original_chunks=[{"context_before": "", "main_content": source, "context_after": ""}],
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.com/chat/completions",
        llm_provider="deepseek",
        prompt_options={
            "text_transform_mode": "modernize",
            "editorial_quality_guard": True,
            "editorial_quality_report": False,
            "source_aware_editorial_guard": False,
        },
    )

    assert result == ["Capítulo primero.\n\nEn un sitio de la Mancha."]


@pytest.mark.asyncio
async def test_modernize_refine_keeps_candidate_on_repairable_style_reject(monkeypatch):
    source = "No ha mucho tiempo que vivía un hidalgo."
    candidate = "Hace no tanto vivía un hidalgo."

    class FakeClient:
        async def close(self):
            pass

    async def fake_make_refinement_request(**_kwargs):
        return candidate, None

    async def fake_guard(**kwargs):
        return QualityDecision(
            chunk_index=kwargs["chunk_index"],
            section=kwargs["section"],
            accepted=False,
            issues=[
                QualityIssue(
                    "length_regression",
                    "reject",
                    "Cambio de longitud reparable",
                )
            ],
        ), None

    async def fake_supervise_fidelity(*_args, **kwargs):
        return FidelityDecision(
            chunk_index=kwargs["chunk_index"],
            phase=kwargs["phase"],
            section=kwargs["section"],
            accepted=True,
        ), None

    monkeypatch.setattr("src.core.translator.create_llm_client", lambda *a, **k: FakeClient())
    monkeypatch.setattr("src.core.translator._make_refinement_request", fake_make_refinement_request)
    monkeypatch.setattr("src.core.translator._assess_refinement_with_editorial_guard", fake_guard)
    monkeypatch.setattr("src.core.translator.supervise_fidelity", fake_supervise_fidelity)

    result = await refine_chunks(
        translated_chunks=[source],
        original_chunks=[{"context_before": "", "main_content": source, "context_after": ""}],
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        api_endpoint="https://api.deepseek.com/chat/completions",
        llm_provider="deepseek",
        prompt_options={
            "text_transform_mode": "modernize",
            "editorial_quality_guard": True,
            "editorial_quality_report": False,
            "source_aware_editorial_guard": False,
        },
    )

    assert result == [candidate]
