"""A candidate with damaged placeholders is reused instead of re-requested."""

from __future__ import annotations

import pytest

from src import config
from src.core.epub import xhtml_translator
from src.core.epub.translation_metrics import TranslationMetrics

TAG_MAP = {
    "[id0]": "<p>",
    "[id1]": "</p><p>",
    "[id2]": "<i>",
    "[id3]": "</i>",
    "[id4]": "</p><p>",
    "[id5]": "</p>",
}
SOURCE = (
    "[id0]First paragraph.[id1]He was [id2]very[id3] tired.[id4]Third paragraph.[id5]"
)
EXPECTED = (
    "[id0]Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo.[id5]"
)


async def _translate(monkeypatch, responses, *, mode="first_failure", max_retries=3):
    """Run one unit against scripted provider responses; return the trace."""
    monkeypatch.setattr(config, "EPUB_PLACEHOLDER_SALVAGE", mode)
    requests = []

    async def fake_request(text, **kwargs):
        requests.append((text, kwargs.get("has_placeholders")))
        return responses[len(requests) - 1]

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    stats = TranslationMetrics(total_chunks=1)
    unit = {}
    events = []
    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=SOURCE,
        local_tag_map=TAG_MAP,
        global_indices=list(range(len(TAG_MAP))),
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=stats,
        log_callback=lambda event, message: events.append(event),
        max_retries=max_retries,
        prompt_options={"fidelity_supervisor_mode": "off"},
        runtime_state={},
        unit_record=unit,
    )
    return result, requests, stats, unit, events


@pytest.mark.asyncio
async def test_dropped_block_boundaries_cost_no_second_request(monkeypatch):
    candidate = (
        "Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo."
    )

    result, requests, stats, unit, events = await _translate(monkeypatch, [candidate])

    assert result == EXPECTED
    assert [text for text, _flag in requests] == [SOURCE]
    assert unit["translation_method"] == "placeholder_salvage"
    assert "placeholder_salvage_success" in events
    assert (stats.successful_first_try, stats.successful_after_retry) == (0, 1)
    assert (stats.correction_attempts, stats.correction_success) == (1, 1)
    assert stats.placeholder_errors == 1
    assert stats.token_alignment_used == 0


@pytest.mark.asyncio
async def test_only_the_damaged_block_is_requested_again(monkeypatch):
    # The italic span of the second paragraph lost its opening tag.
    candidate = (
        "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado.[id4]Tercer párrafo.[id5]"
    )
    regional = "Estaba [id0]muy[id1] agotado."

    result, requests, stats, unit, _events = await _translate(
        monkeypatch, [candidate, regional],
    )

    assert requests == [
        (SOURCE, True),
        # One paragraph, renumbered from zero so strict validation stays local.
        ("He was [id0]very[id1] tired.", True),
    ]
    assert result == (
        "[id0]Primer párrafo.[id1]Estaba [id2]muy[id3] agotado.[id4]Tercer párrafo.[id5]"
    )
    assert unit["translation_method"] == "placeholder_salvage"
    assert stats.correction_success == 1


@pytest.mark.asyncio
async def test_failed_regional_request_falls_back_to_surviving_anchors(monkeypatch):
    candidate = (
        "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado.[id4]Tercer párrafo.[id5]"
    )

    result, requests, _stats, unit, _events = await _translate(
        monkeypatch, [candidate, "Estaba muy agotado."],
    )

    # The first candidate still has the closing tag exactly where the model
    # put it, so its prose is kept and only the opening tag is re-inserted.
    assert result == EXPECTED
    assert len(requests) == 2
    assert unit["translation_method"] == "placeholder_salvage"


@pytest.mark.asyncio
async def test_merged_paragraphs_are_requested_as_one_region(monkeypatch):
    candidate = (
        "[id0]Primer párrafo. Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo.[id5]"
    )
    # Two of three readable blocks exceed the default share: legacy retry.
    result, requests, stats, unit, _events = await _translate(
        monkeypatch, [candidate, EXPECTED],
    )

    assert result == EXPECTED
    assert [text for text, _flag in requests] == [SOURCE, SOURCE]
    assert "translation_method" not in unit
    assert (stats.correction_attempts, stats.correction_success) == (1, 0)
    assert stats.placeholder_errors == 1

    monkeypatch.setattr(config, "EPUB_PLACEHOLDER_SALVAGE_MAX_DAMAGED_SHARE", 1.0)
    regional = "Primer párrafo.[id0]Estaba [id1]muy[id2] cansado."
    result, requests, _stats, unit, _events = await _translate(
        monkeypatch, [candidate, regional],
    )

    assert result == EXPECTED
    assert requests[1] == ("First paragraph.[id0]He was [id1]very[id2] tired.", True)
    assert unit["translation_method"] == "placeholder_salvage"


@pytest.mark.asyncio
async def test_regional_request_without_answer_still_reuses_the_candidate(monkeypatch):
    candidate = (
        "[id0]Primer párrafo.[id1]Estaba muy cansado.[id4]Tercer párrafo.[id5]"
    )
    # The regional request returns nothing. Both lost tags are formatting
    # spans, so they are re-inserted in the first candidate's own prose.
    result, requests, _stats, unit, _events = await _translate(
        monkeypatch, [candidate, None],
    )

    assert len(requests) == 2
    assert unit["translation_method"] == "placeholder_salvage"
    assert result.count("[id2]") == 1 and result.count("[id3]") == 1
    assert result.startswith("[id0]Primer párrafo.[id1]")
    assert result.endswith("[id4]Tercer párrafo.[id5]")


@pytest.mark.asyncio
async def test_salvage_off_keeps_legacy_behavior(monkeypatch):
    candidate = (
        "Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo."
    )

    result, requests, stats, unit, _events = await _translate(
        monkeypatch, [candidate, EXPECTED], mode="off",
    )

    assert result == EXPECTED
    assert [text for text, _flag in requests] == [SOURCE, SOURCE]
    assert "translation_method" not in unit
    assert stats.correction_attempts == 0


@pytest.mark.asyncio
async def test_after_retries_salvages_only_the_last_candidate(monkeypatch):
    damaged = (
        "Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo."
    )

    result, requests, stats, unit, _events = await _translate(
        monkeypatch, [damaged, damaged], mode="after_retries", max_retries=2,
    )

    assert result == EXPECTED
    assert [text for text, _flag in requests] == [SOURCE, SOURCE]
    assert unit["translation_method"] == "placeholder_salvage"
    assert (stats.correction_attempts, stats.correction_success) == (1, 1)


@pytest.mark.asyncio
async def test_lost_line_break_is_never_guessed(monkeypatch):
    """A position-critical tag has no last resort: the legacy retry runs."""
    tag_map = dict(TAG_MAP)
    tag_map["[id2]"] = "<br/>"
    tag_map["[id3]"] = "</i>"
    monkeypatch.setattr(config, "EPUB_PLACEHOLDER_SALVAGE", "first_failure")
    requests = []
    responses = [
        # Line break lost in the second paragraph.
        "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado.[id4]Tercer párrafo.[id5]",
        # Regional request loses it again.
        "Estaba muy[id1] cansado.",
        # Second whole-unit attempt is valid.
        EXPECTED,
    ]

    async def fake_request(text, **kwargs):
        requests.append(text)
        return responses[len(requests) - 1]

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    unit = {}
    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=SOURCE,
        local_tag_map=tag_map,
        global_indices=list(range(len(tag_map))),
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=TranslationMetrics(total_chunks=1),
        max_retries=3,
        prompt_options={"fidelity_supervisor_mode": "off"},
        runtime_state={},
        unit_record=unit,
    )

    assert result == EXPECTED
    assert requests == [SOURCE, "He was [id0]very[id1] tired.", SOURCE]
    assert "translation_method" not in unit


@pytest.mark.asyncio
async def test_defect_in_salvage_never_breaks_the_legacy_retry(monkeypatch):
    def broken_plan(*_args, **_kwargs):
        raise KeyError("unexpected")

    monkeypatch.setattr(
        "src.core.epub.placeholder_salvage.plan_placeholder_salvage", broken_plan,
    )
    damaged = "Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]Tercer párrafo."

    result, requests, _stats, unit, events = await _translate(
        monkeypatch, [damaged, EXPECTED],
    )

    assert result == EXPECTED
    assert [text for text, _flag in requests] == [SOURCE, SOURCE]
    assert "placeholder_salvage_error" in events
    assert "translation_method" not in unit


@pytest.mark.asyncio
async def test_provider_stop_during_regional_request_reaches_the_job(monkeypatch):
    from src.core.llm.exceptions import RateLimitError

    monkeypatch.setattr(config, "EPUB_PLACEHOLDER_SALVAGE", "first_failure")
    requests = []

    async def fake_request(text, **kwargs):
        requests.append(text)
        if len(requests) == 1:
            return (
                "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado."
                "[id4]Tercer párrafo.[id5]"
            )
        raise RateLimitError("provider wait")

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)

    with pytest.raises(RateLimitError):
        await xhtml_translator.translate_chunk_with_fallback(
            chunk_text=SOURCE,
            local_tag_map=TAG_MAP,
            global_indices=list(range(len(TAG_MAP))),
            source_language="English",
            target_language="Spanish",
            model_name="test",
            llm_client=object(),
            stats=TranslationMetrics(total_chunks=1),
            max_retries=3,
            prompt_options={"fidelity_supervisor_mode": "off"},
            runtime_state={},
            unit_record={},
        )

    assert len(requests) == 2
