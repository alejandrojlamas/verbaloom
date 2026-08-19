import asyncio

import pytest

from src.core.adapters import translate_file
from src.core.text_processor import split_text_into_chunks
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


def test_text_resume_retries_only_failed_holes_after_first_error(tmp_path, monkeypatch):
    source = "\n\n".join(
        f"Paragraph {idx}. " + ("This sentence has enough words for a stable test. " * 2)
        for idx in range(8)
    )
    input_path = tmp_path / "book.txt"
    output_path = tmp_path / "book.es.txt"
    input_path.write_text(source, encoding="utf-8")

    chunks = split_text_into_chunks(source, max_tokens_per_chunk=24)
    assert len(chunks) >= 4
    failed_index = 2

    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    translation_id = "sparse_resume"
    manager.start_job(translation_id, "txt", {"output_filename": output_path.name}, str(input_path))
    for idx, chunk in enumerate(chunks):
        manager.save_checkpoint(
            translation_id=translation_id,
            chunk_index=idx,
            original_text=chunk["main_content"],
            translated_text=None if idx == failed_index else f"completed-{idx}",
            chunk_data={"chunk_index": idx, "total_chunks": len(chunks)},
            total_chunks=len(chunks),
            completed_chunks=idx + 1,
            failed_chunks=1 if idx >= failed_index else 0,
        )

    calls = []

    async def fake_generate_translation_request(**kwargs):
        calls.append(kwargs["main_content"])
        return "repaired-hole"

    monkeypatch.setattr(
        "src.core.translator.generate_translation_request",
        fake_generate_translation_request,
    )

    ok = asyncio.run(
        translate_file(
            input_filepath=str(input_path),
            output_filepath=str(output_path),
            source_language="English",
            target_language="Spanish",
            model_name="fake-echo",
            llm_provider="poe",
            checkpoint_manager=manager,
            translation_id=translation_id,
            resume_from_index=manager.load_checkpoint(translation_id)["resume_from_index"],
            poe_api_key="UNUSED_FAKE_KEY",
            max_tokens_per_chunk=24,
            context_window=4096,
            auto_adjust_context=False,
            prompt_options={"translation_memory_enabled": False},
        )
    )

    assert ok is True
    assert calls == [chunks[failed_index]["main_content"]]
    assert "completed-3" in output_path.read_text(encoding="utf-8")
    assert "repaired-hole" in output_path.read_text(encoding="utf-8")
