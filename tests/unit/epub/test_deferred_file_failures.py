"""One unresolved XHTML file must not stop the rest of the book."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest
from lxml import etree

from src.core.epub import translator as subject
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.translator import (
    _completed_epub_file_indices,
    _process_all_content_files,
    _translate_single_xhtml_file,
    translate_epub_file,
)
from src.core.llm.exceptions import (
    ProviderActionRequiredError,
    ProviderUnavailableError,
    RateLimitError,
    is_unit_scoped_provider_stop,
)

FILES = ["one.xhtml", "two.xhtml", "three.xhtml", "four.xhtml"]


class _Checkpoint:
    """Records what the EPUB loop persists; serves rows back on resume."""

    def __init__(self, rows=None, progress=None):
        self.rows = list(rows or [])
        self.progress = dict(progress or {})
        self.saved_files = []
        self.progress_updates = []
        self.uploads_dir = None

    def get_job(self, _translation_id):
        return {"progress": dict(self.progress)}

    def load_checkpoint(self, _translation_id):
        return {"chunks": list(self.rows)}

    def load_xhtml_partial_state(self, _translation_id, _href):
        return None

    def update_progress(self, _translation_id, **fields):
        self.progress_updates.append(fields)
        self.progress.update(fields)
        return True

    def save_epub_file(self, translation_id, file_href, file_content):
        self.saved_files.append(file_href)
        return True

    def delete_xhtml_partial_state(self, _translation_id, _href):
        return True

    def save_checkpoint(self, translation_id, chunk_index, original_text,
                        translated_text, chunk_data=None, **fields):
        self.rows.append({
            "chunk_index": chunk_index,
            "status": "completed",
            "translated_text": translated_text,
            "chunk_data": chunk_data,
        })
        self.progress.update({
            key: value for key, value in fields.items()
            if key in {"total_chunks", "completed_chunks", "failed_chunks",
                       "epub_accumulated_stats"}
        })
        return True


def _row(index, href, *, status="completed"):
    return {
        "chunk_index": index,
        "status": status,
        "translated_text": href if status == "completed" else None,
        "chunk_data": {"last_file": href, "file_type": "epub_xhtml"},
    }


def _fake_translate(calls, failing):
    async def fake(**kwargs):
        href = kwargs["content_href"]
        calls.append(href)
        if href in failing:
            return None, False, None
        return etree.Element("html"), True, TranslationMetrics()

    return fake


async def _run(tmp_path, monkeypatch, *, failing, checkpoint=None,
               resume_from_index=0, restored=(), completed=None,
               restore_attempted=False, files=FILES, translate=None, stats_callback=None):
    monkeypatch.setattr(
        subject, "_precount_chunks",
        AsyncMock(return_value=(len(files) * 2, [2] * len(files))),
    )
    calls = []
    monkeypatch.setattr(
        subject, "_translate_single_xhtml_file", translate or _fake_translate(calls, set(failing)),
    )
    restored_docs = {
        os.path.normpath(os.path.join(str(tmp_path), href)): etree.Element("html")
        for href in restored
    }
    result = await _process_all_content_files(
        content_files=list(files),
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        max_attempts=1,
        context_manager=None,
        translation_id="job" if checkpoint else None,
        resume_from_index=resume_from_index,
        checkpoint_manager=checkpoint,
        prompt_options={"abort_on_profile_fail": True},
        restored_docs=restored_docs or None,
        completed_file_indices=completed,
        restore_attempted=restore_attempted,
        stats_callback=stats_callback,
    )
    return result, calls


@pytest.mark.asyncio
async def test_nested_progress_survives_deferred_file_and_clears_between_files(tmp_path, monkeypatch):
    ticks = []

    async def translate(**kwargs):
        if kwargs["content_href"] == FILES[0]:
            kwargs["stats_callback"]({"completed_chunks": 1})
            return None, False, None
        for done in (0, 1, 2):
            kwargs["stats_callback"]({
                "completed_chunks": 2,
                "stage_progress": {"stage": "audit", "done": done, "total": 2},
            })
        return etree.Element("html"), True, TranslationMetrics()

    await _run(tmp_path, monkeypatch, failing=(), files=FILES[:2],
               translate=translate, stats_callback=ticks.append)
    assert [tick["completed_chunks"] for tick in ticks] == [1, 1, 3, 3, 3, 3]
    assert [tick["stage_progress"] for tick in ticks] == [
        None, None,
        *[{"stage": "audit", "done": done, "total": 2, "section": 2, "sections": 2}
          for done in (0, 1, 2)],
        None,
    ]


@pytest.mark.asyncio
async def test_later_files_are_translated_after_a_failed_file(tmp_path, monkeypatch):
    checkpoint = _Checkpoint()

    result, calls = await _run(
        tmp_path, monkeypatch, failing={"two.xhtml"}, checkpoint=checkpoint,
    )

    assert calls == FILES
    assert result["failed_files"] == 1
    assert result["failed_chunks"] == 1
    assert result["completed_chunks"] == 6
    assert result["unresolved_files"] == ["two.xhtml"]
    # No completed row for the failed file: the resume index stays on it.
    assert sorted(row["chunk_index"] for row in checkpoint.rows) == [0, 2, 3]
    # The checkpoint itself proves that a unit is pending.
    assert checkpoint.progress["failed_chunks"] == 1


@pytest.mark.asyncio
async def test_failures_in_a_row_still_stop_the_book(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.EPUB_MAX_CONSECUTIVE_FAILED_FILES", 2)

    result, calls = await _run(
        tmp_path, monkeypatch, failing={"two.xhtml", "three.xhtml", "four.xhtml"},
    )

    assert calls == ["one.xhtml", "two.xhtml", "three.xhtml"]
    assert result["failed_files"] == 2


@pytest.mark.asyncio
async def test_isolated_failures_do_not_trip_the_consecutive_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.EPUB_MAX_CONSECUTIVE_FAILED_FILES", 2)

    result, calls = await _run(
        tmp_path, monkeypatch, failing={"one.xhtml", "three.xhtml"},
    )

    assert calls == FILES
    assert result["failed_files"] == 2
    assert result["unresolved_files"] == ["one.xhtml", "three.xhtml"]


@pytest.mark.asyncio
async def test_resume_retries_only_the_pending_file(tmp_path, monkeypatch):
    # Previous pass: two.xhtml failed, one/three/four were completed.
    checkpoint = _Checkpoint(
        rows=[_row(0, "one.xhtml"), _row(2, "three.xhtml"), _row(3, "four.xhtml")],
        progress={
            "total_chunks": 8,
            "completed_chunks": 6,
            "failed_chunks": 1,
            "epub_accumulated_stats": {"failed_chunks": 1, "processed_chunks": 6},
        },
    )
    completed = _completed_epub_file_indices(checkpoint, "job", FILES, 1)
    assert completed == {0, 2, 3}

    result, calls = await _run(
        tmp_path, monkeypatch, failing=set(), checkpoint=checkpoint,
        resume_from_index=1, restored=["one.xhtml", "three.xhtml", "four.xhtml"],
        completed=completed, restore_attempted=True,
    )

    assert calls == ["two.xhtml"]
    assert result["failed_files"] == 0
    # The failure counted in the previous pass must not block publication.
    assert result["failed_chunks"] == 0
    assert result["completed_chunks"] == 8
    assert result["unresolved_files"] == []
    assert len(result["parsed_docs"]) == 4
    assert checkpoint.progress["failed_chunks"] == 0


@pytest.mark.asyncio
async def test_failed_first_file_does_not_retranslate_the_book(tmp_path, monkeypatch):
    checkpoint = _Checkpoint(
        rows=[_row(1, "two.xhtml"), _row(2, "three.xhtml"), _row(3, "four.xhtml")],
        progress={"total_chunks": 8, "completed_chunks": 6, "failed_chunks": 1},
    )
    completed = _completed_epub_file_indices(checkpoint, "job", FILES, 0)

    result, calls = await _run(
        tmp_path, monkeypatch, failing={"one.xhtml"}, checkpoint=checkpoint,
        resume_from_index=0, restored=["two.xhtml", "three.xhtml", "four.xhtml"],
        completed=completed, restore_attempted=True,
    )

    assert calls == ["one.xhtml"]
    assert result["failed_files"] == 1
    assert result["completed_chunks"] == 6
    assert result["unresolved_files"] == ["one.xhtml"]


@pytest.mark.asyncio
async def test_completed_row_without_restored_document_is_translated_again(tmp_path, monkeypatch):
    checkpoint = _Checkpoint(
        rows=[_row(0, "one.xhtml"), _row(1, "two.xhtml"), _row(3, "four.xhtml")],
        progress={"total_chunks": 8, "completed_chunks": 6},
    )
    completed = _completed_epub_file_indices(checkpoint, "job", FILES, 2)

    # two.xhtml (prefix) and four.xhtml (sparse) could not be restored.
    result, calls = await _run(
        tmp_path, monkeypatch, failing=set(), checkpoint=checkpoint,
        resume_from_index=2, restored=["one.xhtml"],
        completed=completed, restore_attempted=True,
    )

    assert calls == ["two.xhtml", "three.xhtml", "four.xhtml"]
    assert result["unresolved_files"] == []


def test_completed_files_require_a_matching_completed_row():
    checkpoint = _Checkpoint(rows=[
        _row(0, "one.xhtml"),
        _row(2, "renamed.xhtml"),                    # different document
        _row(3, "four.xhtml", status="failed"),      # not completed
        {"chunk_index": 9, "status": "completed", "translated_text": "x",
         "chunk_data": {"last_file": "x", "file_type": "epub_xhtml"}},
        {"chunk_index": "bad"},
    ])

    assert _completed_epub_file_indices(checkpoint, "job", FILES, 1) == {0}
    assert _completed_epub_file_indices(None, "job", FILES, 2) == {0, 1}
    assert _completed_epub_file_indices(object(), "job", FILES, 9) == {0, 1, 2, 3}


def test_unit_scoped_provider_stops():
    def stop(reason):
        return ProviderActionRequiredError("stop", reason=reason, provider="deepseek")

    assert is_unit_scoped_provider_stop(stop("provider_refusal"))
    assert is_unit_scoped_provider_stop(stop("audit_invalid_response"))
    assert not is_unit_scoped_provider_stop(stop("authentication_error"))
    assert not is_unit_scoped_provider_stop(stop("invalid_request"))
    assert not is_unit_scoped_provider_stop(stop("rate_limit_recovery_exhausted"))
    assert not is_unit_scoped_provider_stop(RateLimitError("429"))
    assert not is_unit_scoped_provider_stop(ProviderUnavailableError("down"))


async def _translate_file_raising(tmp_path, monkeypatch, error):
    source = tmp_path / "chapter.xhtml"
    source.write_text(
        "<html xmlns='http://www.w3.org/1999/xhtml'><body><p>Text.</p></body></html>",
        encoding="utf-8",
    )

    async def raising_translate(self, **_kwargs):
        raise error

    monkeypatch.setattr(
        subject.GenericTranslationOrchestrator, "translate", raising_translate,
    )
    events = []
    result = await _translate_single_xhtml_file(
        file_path=str(source),
        content_href="chapter.xhtml",
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        max_attempts=1,
        context_manager=None,
        log_callback=lambda event, message: events.append(event),
        prompt_options={},
    )
    return result, events


@pytest.mark.asyncio
async def test_refused_passage_fails_its_file_not_the_job(tmp_path, monkeypatch):
    refusal = ProviderActionRequiredError(
        "refused", reason="provider_refusal", provider="deepseek",
    )

    (doc, success, _stats), events = await _translate_file_raising(
        tmp_path, monkeypatch, refusal,
    )

    assert doc is None and success is False
    assert "epub_unit_requires_review" in events


@pytest.mark.asyncio
async def test_job_scoped_provider_stops_still_reach_the_job_boundary(tmp_path, monkeypatch):
    for error in (
        ProviderActionRequiredError("bad key", reason="authentication_error", provider="deepseek"),
        ProviderUnavailableError("down", reason="timeout", provider="deepseek"),
        RateLimitError("429"),
    ):
        with pytest.raises(type(error)):
            await _translate_file_raising(tmp_path, monkeypatch, error)


@pytest.mark.asyncio
async def test_refusal_stops_the_job_when_deferral_is_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr("src.config.EPUB_DEFER_FAILED_FILES", False)
    refusal = ProviderActionRequiredError(
        "refused", reason="provider_refusal", provider="deepseek",
    )

    with pytest.raises(ProviderActionRequiredError):
        await _translate_file_raising(tmp_path, monkeypatch, refusal)


@pytest.mark.asyncio
async def test_missing_document_blocks_publication_even_when_counters_agree(monkeypatch, tmp_path):
    input_path = tmp_path / "source.epub"
    output_path = tmp_path / "translated.epub"
    input_path.write_bytes(b"epub")

    monkeypatch.setattr(subject, "_create_llm_client", lambda **_kwargs: object())
    monkeypatch.setattr(subject, "_create_context_manager", lambda **_kwargs: None)
    monkeypatch.setattr(subject, "_extract_epub", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        subject, "_parse_epub_manifest",
        lambda temp_dir, _log: {
            "content_files": ["one.xhtml", "two.xhtml"],
            "opf_dir": temp_dir,
            "opf_tree": etree.ElementTree(etree.Element("package")),
            "opf_path": str(tmp_path / "content.opf"),
        },
    )
    monkeypatch.setattr(
        subject, "_process_all_content_files",
        AsyncMock(return_value={
            "parsed_docs": {},
            "completed_files": 2,
            "failed_files": 0,
            "unresolved_files": ["two.xhtml"],
            "total_chunks": 2,
            "completed_chunks": 2,
            "failed_chunks": 0,
            "translation_stats": TranslationMetrics(),
            "was_interrupted": False,
        }),
    )
    monkeypatch.setattr(subject, "_save_translated_files", AsyncMock())
    repackage = AsyncMock()
    monkeypatch.setattr(subject, "_repackage_epub", repackage)
    monkeypatch.setattr(
        subject, "write_fidelity_report_from_options", lambda *_args, **_kwargs: None,
    )

    success = await translate_epub_file(
        str(input_path),
        str(output_path),
        source_language="English",
        target_language="Spanish",
    )

    assert success is False
    repackage.assert_not_awaited()
