import asyncio

import pytest

from src.core.adapters import GenericTranslator, TxtAdapter
from src.core.adapters.translate_file import translate_file
from src.core.editorial_quality import QualityDecision
from src.core.output_formats import _write_epub
from src.persistence.checkpoint_manager import CheckpointManager
from src.api.handlers import (
    _ensure_layout_sanitizer_options,
    _resume_requires_layout_sanitizer_restart,
    _uses_text_first_pipeline,
)


class FakeLLMClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass


def test_epub_text_first_is_opt_in_docx_is_sanitized_by_default():
    assert _uses_text_first_pipeline({
        "file_type": "epub",
        "prompt_options": {
            "text_type": "literature",
            "literary_continuity": True,
        },
    }) is False
    assert _uses_text_first_pipeline({
        "file_type": "docx",
        "prompt_options": {},
    }) is True
    assert _uses_text_first_pipeline({
        "file_type": "docx",
        "prompt_options": {"preserve_source_formatting": True},
    }) is False


def test_pdf_docx_checkpoints_carry_layout_sanitizer_version():
    config = {
        "file_type": "pdf",
        "prompt_options": {},
    }

    assert _resume_requires_layout_sanitizer_restart(config) is True
    _ensure_layout_sanitizer_options(config)

    assert config["prompt_options"]["layout_sanitizer_active"] is True
    assert config["prompt_options"]["layout_sanitizer_version"]
    assert _resume_requires_layout_sanitizer_restart(config) is False


@pytest.mark.asyncio
async def test_txt_inline_refinement_saves_refined_chunks_in_order(tmp_path, monkeypatch):
    source = tmp_path / "book.txt"
    output = tmp_path / "book_es.txt"
    source.write_text(
        "First paragraph has enough words to become one translation unit.\n\n"
        "Second paragraph also has enough words to become another unit.",
        encoding="utf-8",
    )

    events = []

    async def fake_translate(**kwargs):
        idx = kwargs["main_content"].split()[0].lower()
        events.append(("translate_start", idx))
        await asyncio.sleep(0.02)
        events.append(("translate_end", idx))
        return f"traducido {kwargs['main_content']}"

    async def fake_refine(**kwargs):
        idx = kwargs["draft_translation"].split()[1].lower()
        events.append(("refine_start", idx))
        await asyncio.sleep(0.05)
        events.append(("refine_end", idx))
        return f"editado {kwargs['draft_translation']}", None

    async def fake_guard(**kwargs):
        return QualityDecision(
            chunk_index=kwargs["chunk_index"],
            section=kwargs["section"],
            accepted=True,
        ), None

    monkeypatch.setattr("src.core.llm_client.LLMClient", FakeLLMClient)
    monkeypatch.setattr("src.core.translator.generate_translation_request", fake_translate)
    monkeypatch.setattr("src.core.translator._make_refinement_request", fake_refine)
    monkeypatch.setattr("src.core.translator._assess_refinement_with_editorial_guard", fake_guard)

    checkpoint_manager = CheckpointManager(db_path=str(tmp_path / "checkpoints.db"))
    adapter = TxtAdapter(
        str(source),
        str(output),
        {"max_tokens_per_chunk": 12, "prompt_options": {"inline_refinement": True}},
    )
    translator = GenericTranslator(adapter, checkpoint_manager, "inline-test")

    ok = await translator.translate(
        source_language="English",
        target_language="Spanish",
        model_name="fake-model",
        llm_provider="fake",
        prompt_options={
            "inline_refinement": True,
            "translation_memory_enabled": False,
            "fidelity_supervisor_mode": "off",
            "editorial_quality_report": False,
        },
    )

    assert ok is True
    text = output.read_text(encoding="utf-8")
    assert "editado traducido First paragraph" in text
    assert "editado traducido Second paragraph" in text

    chunks = checkpoint_manager.load_checkpoint("inline-test")["chunks"]
    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1]
    assert all((chunk["chunk_data"] or {}).get("inline_refinement") for chunk in chunks)

    # The second translation starts before the first refinement finishes:
    # that proves the pipeline overlaps translation and editing while still
    # saving completed chunks in order.
    assert events.index(("translate_start", "second")) < events.index(("refine_end", "first"))


@pytest.mark.asyncio
async def test_epub_text_first_uses_inline_refinement_pipeline(tmp_path, monkeypatch):
    source = tmp_path / "book.epub"
    output = tmp_path / "book_es.txt"
    _write_epub(
        "Chapter One\n\nA quiet room waited at the end of the corridor.",
        source,
        title="Book",
    )

    async def fake_translate(**kwargs):
        return f"traducido {kwargs['main_content']}"

    async def fake_refine(**kwargs):
        return f"editado {kwargs['draft_translation']}", None

    async def fake_guard(**kwargs):
        return QualityDecision(
            chunk_index=kwargs["chunk_index"],
            section=kwargs["section"],
            accepted=True,
        ), None

    monkeypatch.setattr("src.core.llm_client.LLMClient", FakeLLMClient)
    monkeypatch.setattr("src.core.translator.generate_translation_request", fake_translate)
    monkeypatch.setattr("src.core.translator._make_refinement_request", fake_refine)
    monkeypatch.setattr("src.core.translator._assess_refinement_with_editorial_guard", fake_guard)

    checkpoint_manager = CheckpointManager(db_path=str(tmp_path / "checkpoints.db"))
    ok = await translate_file(
        input_filepath=str(source),
        output_filepath=str(output),
        source_language="English",
        target_language="Spanish",
        model_name="fake-model",
        llm_provider="fake",
        checkpoint_manager=checkpoint_manager,
        translation_id="epub-text-first",
        max_tokens_per_chunk=200,
        prompt_options={
            "refine": True,
            "plain_text_mode": True,
            "literary_continuity": True,
            "text_type": "literature",
            "fidelity_supervisor_mode": "off",
            "editorial_quality_report": False,
        },
    )

    assert ok is True
    assert "editado traducido" in output.read_text(encoding="utf-8")
    checkpoint = checkpoint_manager.load_checkpoint("epub-text-first")
    assert checkpoint["resume_from_index"] == 1
    config = checkpoint["job"]["config"]
    options = config["prompt_options"]
    assert options["text_first_pipeline_active"] is True
    assert options["inline_refinement"] is True
