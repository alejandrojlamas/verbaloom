"""Tests for per-chunk paragraph realignment in the plain-text pipeline.

The pipeline previously reassembled ALL translated chunks and re-split the
result globally: a single paragraph merge by the LLM in an early chunk
shifted every later block, mis-anchoring styles, images, and table cells.
Reconciliation now happens per chunk, confining the damage.
"""
import pytest

import src.core.common.plain_text_pipeline as pipeline_module
from src.core.common.plain_text_pipeline import translate_paragraphs_plain
from src.core.epub.exceptions import ChunkTranslationFailedError


@pytest.mark.asyncio
async def test_paragraph_merge_in_one_chunk_does_not_shift_later_blocks(monkeypatch):
    # 12 distinctive paragraphs, small chunks force several LLM calls.
    source = [f"Paragraph number {i} with unique token tok{i}." for i in range(12)]

    chunk_contents = []

    async def fake_translation(main_content, **kwargs):
        chunk_contents.append(main_content)
        translated = main_content.upper()
        if len(chunk_contents) == 1:
            # Simulate the LLM merging the first two paragraphs of the chunk.
            translated = translated.replace("\n\n", " ", 1)
        return translated

    monkeypatch.setattr(
        pipeline_module, "generate_translation_request", fake_translation
    )

    translated, stats, interrupted = await translate_paragraphs_plain(
        paragraphs=source,
        source_language="English",
        target_language="English",
        model_name="fake",
        llm_client=None,
        max_tokens_per_chunk=60,  # tiny chunks -> multiple calls
    )

    assert not interrupted
    assert len(chunk_contents) >= 2, "test requires multiple chunks to be meaningful"
    assert len(translated) == len(source)

    # The contract of per-chunk reconciliation: damage from a merge in chunk 0
    # is CONFINED to chunk 0's block range. Every block belonging to later
    # chunks must still contain its own unique token. (Without the fix, the
    # merge shifted every later block by one: block i held token i+1.)
    first_chunk_size = len(
        pipeline_module._split_translated_back_to_paragraphs(chunk_contents[0])
    )
    assert first_chunk_size < len(source)
    for i in range(first_chunk_size, len(source)):
        assert f"TOK{i}" in translated[i], (
            f"block {i} drifted: {translated[i][:60]!r}"
        )
    # Inside the damaged chunk, content is preserved (merged, padded) — the
    # merged pair lives in slot 0 and the pad lands inside the chunk range.
    assert "TOK0" in translated[0] and "TOK1" in translated[0]


@pytest.mark.asyncio
async def test_inline_markdown_detection_sets_prompt_option(monkeypatch):
    source = ["Plain paragraph.", "One with **bold** inside."]
    seen_options = {}

    async def fake_translation(main_content, **kwargs):
        seen_options.update(kwargs.get("prompt_options") or {})
        return main_content

    monkeypatch.setattr(
        pipeline_module, "generate_translation_request", fake_translation
    )

    await translate_paragraphs_plain(
        paragraphs=source,
        source_language="English",
        target_language="Spanish",
        model_name="fake",
        llm_client=None,
        max_tokens_per_chunk=500,
    )
    assert seen_options.get("inline_markdown") is True


@pytest.mark.asyncio
async def test_no_markdown_no_flag(monkeypatch):
    source = ["Plain paragraph.", "Another plain one with 2 * 3 math."]
    seen_options = {}

    async def fake_translation(main_content, **kwargs):
        seen_options.update(kwargs.get("prompt_options") or {})
        return main_content

    monkeypatch.setattr(
        pipeline_module, "generate_translation_request", fake_translation
    )

    await translate_paragraphs_plain(
        paragraphs=source,
        source_language="English",
        target_language="Spanish",
        model_name="fake",
        llm_client=None,
        max_tokens_per_chunk=500,
    )
    assert not seen_options.get("inline_markdown")


@pytest.mark.asyncio
async def test_failed_plain_epub_chunk_never_falls_back_to_source(monkeypatch):
    async def failed_translation(main_content, **kwargs):
        return None

    monkeypatch.setattr(
        pipeline_module, "generate_translation_request", failed_translation
    )

    with pytest.raises(ChunkTranslationFailedError) as error:
        await translate_paragraphs_plain(
            paragraphs=["Dieser Absatz darf nicht als Übersetzung erscheinen."],
            source_language="German",
            target_language="Spanish",
            model_name="fake",
            llm_client=None,
            max_tokens_per_chunk=500,
        )

    assert error.value.reason == "no_valid_candidate"
