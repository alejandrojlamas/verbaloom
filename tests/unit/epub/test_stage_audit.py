from __future__ import annotations

import pytest

from src.core.epub import xhtml_translator
from src.core.epub.exceptions import ChunkTranslationFailedError
from src.core.epub.unit_contract import ensure_unit_records, mark_audited, mark_reviewed, text_sha256
from src.core.epub.xhtml_translator import (
    _audit_epub_chunks,
    _refine_epub_chunks,
    _repair_audited_candidate,
)
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ContentRiskError


class AuditClient:
    provider_type = "deepseek"

    def __init__(self, content: str):
        self.content = content

    async def generate(self, *_args, **_kwargs):
        return LLMResponse(content=self.content)


class SequenceAuditClient:
    provider_type = "deepseek"

    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        if not self.responses:
            raise AssertionError("unexpected LLM request")
        return LLMResponse(content=self.responses.pop(0))


class ContentRiskAuditClient:
    provider_type = "deepseek"

    def __init__(self):
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        raise ContentRiskError("Content Exists Risk", provider="deepseek")


def _chunks():
    chunks = [{
        "text": "Der Wanderer betrachtete lange das ruhige Meer und erinnerte sich an seine Kindheit.",
        "local_tag_map": {},
        "global_indices": [],
    }]
    records = ensure_unit_records(
        chunks,
        "Text/chapter.xhtml",
        source_language="German",
        target_language="Spanish",
    )
    candidate = "El viajero contempló largo rato el mar tranquilo y recordó su infancia."
    mark_reviewed(records[0], candidate, review={"status": "pass"})
    return chunks, candidate


def _assessment(verdict: str = "pass", confidence: float = 0.96) -> str:
    return (
        "<FIDELITY_AUDIT_JSON>"
        f'{{"verdict":"{verdict}","confidence":{confidence},'
        '"reason":"comparison complete","issues":[],"missing_from_source":[],'
        '"added_not_in_source":[],"changed_facts":[],"censored_or_softened":[],'
        '"structure_issues":[],"evidence_source":[],"evidence_candidate":[]}'
        "</FIDELITY_AUDIT_JSON>"
    )


@pytest.mark.asyncio
async def test_full_stage_audit_records_a_real_structured_judgement():
    chunks, candidate = _chunks()

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=AuditClient(_assessment()),
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True},
    )

    record = chunks[0]["unit"]
    assert result == [candidate]
    assert record["audit_status"] == "COMPLETED"
    assert record["status"] == "AUDITED"
    assert record["audit"]["judge_decision"] == "pass"
    assert record["audit_attempts"] == 1


@pytest.mark.asyncio
async def test_full_stage_audit_preserves_table_context(monkeypatch):
    chunks = [{
        "text": (
            "[id0]Nike[id1]Virgin Records, Airlines, and others[id2]"
        ),
        "local_tag_map": {
            "[id0]": "<table><tr><td>",
            "[id1]": "</td></tr><tr><td>",
            "[id2]": "</td></tr></table>",
        },
        "global_indices": [0, 1, 2],
    }]
    records = ensure_unit_records(
        chunks,
        "Text/business-table.xhtml",
        source_language="English",
        target_language="Spanish",
    )
    candidate = (
        "[id0]Nike[id1]Virgin Records, aerolíneas y otros[id2]"
    )
    mark_reviewed(records[0], candidate, review={"status": "pass"})
    seen_options = []
    real_supervise = xhtml_translator.supervise_fidelity

    async def capture_supervise(*args, **kwargs):
        seen_options.append(dict(kwargs.get("prompt_options") or {}))
        return await real_supervise(*args, **kwargs)

    monkeypatch.setattr(
        xhtml_translator,
        "supervise_fidelity",
        capture_supervise,
    )

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=AuditClient(_assessment()),
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True},
    )

    assert result == [candidate]
    assert seen_options
    assert seen_options[0]["_document_block_context"] == "table"


@pytest.mark.asyncio
async def test_full_stage_audit_blocks_empty_or_malformed_judge_response():
    chunks, candidate = _chunks()

    with pytest.raises(ChunkTranslationFailedError, match="structured verdict"):
        await _audit_epub_chunks(
            [candidate],
            chunks,
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=AuditClient("truncated {"),
            placeholder_format=("[id", "]"),
            prompt_options={"audit_entire_book": True},
        )

    assert chunks[0]["unit"]["audit_status"] == "FAILED"


@pytest.mark.asyncio
async def test_full_stage_audit_accepts_placeholder_only_cover_without_llm_request():
    chunks = [{
        "text": "[id0]",
        "local_tag_map": {"[id0]": '<img src="../Images/cover.jpg" alt="Cover"/>'},
        "global_indices": [0],
    }]
    records = ensure_unit_records(
        chunks,
        "Text/cover.xhtml",
        source_language="English",
        target_language="Spanish",
    )
    candidate = "[id0]"
    mark_reviewed(records[0], candidate, review={"status": "pass"})
    client = SequenceAuditClient()

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=client,
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True},
    )

    assert result == [candidate]
    assert client.calls == 0
    assert records[0]["audit_status"] == "COMPLETED"
    assert records[0]["status"] == "AUDITED"


@pytest.mark.asyncio
async def test_full_stage_audit_blocks_semantic_rejection():
    chunks, candidate = _chunks()

    with pytest.raises(ChunkTranslationFailedError, match="rejected"):
        await _audit_epub_chunks(
            [candidate],
            chunks,
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=AuditClient(_assessment("fail", 0.97)),
            placeholder_format=("[id", "]"),
            prompt_options={"audit_entire_book": True},
        )

    assert chunks[0]["unit"]["audit_status"] == "FAILED"
    assert chunks[0]["unit"]["failure_reason"] == "audit_rejected"


@pytest.mark.asyncio
async def test_full_stage_audit_uses_local_checks_when_provider_filters_clean_passage():
    chunks, candidate = _chunks()
    client = ContentRiskAuditClient()

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=client,
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True},
    )

    record = chunks[0]["unit"]
    assert result == [candidate]
    assert client.calls == 1
    assert record["audit_status"] == "COMPLETED"
    assert record["status"] == "AUDITED"
    assert any(
        issue["code"] == "fidelity_auditor_content_filter"
        for issue in record["audit"]["issues"]
    )


@pytest.mark.asyncio
async def test_full_stage_audit_does_not_hide_local_fidelity_failure_behind_filter():
    chunks, _candidate = _chunks()
    untranslated = chunks[0]["text"]
    mark_reviewed(chunks[0]["unit"], untranslated, review={"status": "pass"})
    client = ContentRiskAuditClient()

    with pytest.raises(ChunkTranslationFailedError, match="publication is blocked"):
        await _audit_epub_chunks(
            [untranslated],
            chunks,
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=client,
            placeholder_format=("[id", "]"),
            prompt_options={
                "audit_entire_book": True,
                "repair_until_pass": False,
            },
        )

    assert client.calls == 1
    assert chunks[0]["unit"]["audit_status"] == "FAILED"


@pytest.mark.asyncio
async def test_full_stage_audit_repairs_only_rejected_unit_then_reaudits(monkeypatch):
    chunks, candidate = _chunks()
    repaired = "El caminante contempló largo rato el mar sereno y recordó su infancia."
    client = SequenceAuditClient(
        _assessment("fail", 0.97),
        _assessment("pass", 0.98),
    )
    repair_calls = []
    checkpoints = []

    async def fake_local_repair(**kwargs):
        repair_calls.append(kwargs)
        return repaired

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator._repair_audited_candidate",
        fake_local_repair,
    )

    async def unexpected_full_retranslation(**_kwargs):
        raise AssertionError("a valid issue-local repair must avoid full retranslation")

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.translate_chunk_with_fallback",
        unexpected_full_retranslation,
    )

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=client,
        placeholder_format=("[id", "]"),
        prompt_options={
            "audit_entire_book": True,
            "repair_until_pass": True,
            "max_repair_rounds": 1,
            "fidelity_adjudicate_rejections": False,
        },
        checkpoint_callback=lambda index, text, meta: checkpoints.append((index, text, meta)),
    )

    assert result == [repaired]
    assert len(repair_calls) == 1
    assert client.calls == 2
    assert chunks[0]["unit"]["audit_status"] == "COMPLETED"
    assert chunks[0]["unit"]["audit_attempts"] == 2
    assert [item[2]["status"] for item in checkpoints] == ["repaired", "audited"]


@pytest.mark.asyncio
async def test_full_stage_audit_adjudicates_false_positive_before_rewriting(monkeypatch):
    chunks, candidate = _chunks()
    client = SequenceAuditClient(
        _assessment("fail", 0.97),
        _assessment("pass", 0.98),
    )

    async def unexpected_repair(**_kwargs):
        raise AssertionError("an accepted adjudication must not trigger a rewrite")

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.translate_chunk_with_fallback",
        unexpected_repair,
    )

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=client,
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True, "repair_until_pass": True, "max_repair_rounds": 1},
    )

    assert result == [candidate]
    assert client.calls == 2
    assert chunks[0]["unit"]["audit_status"] == "COMPLETED"
    assert chunks[0]["unit"]["audit_attempts"] == 2


@pytest.mark.asyncio
async def test_rejected_audit_repair_persists_exact_resumable_candidate(monkeypatch):
    chunks, candidate = _chunks()
    repaired = "El caminante contempló el mar y recordó su infancia."
    client = SequenceAuditClient(
        _assessment("fail", 0.97),
        _assessment("fail", 0.98),
        _assessment("fail", 0.98),
    )
    checkpoints = []

    async def fake_repair(**_kwargs):
        return repaired

    async def unavailable_local_repair(**_kwargs):
        return None

    monkeypatch.setattr(
        "src.core.epub.xhtml_translator._repair_audited_candidate",
        unavailable_local_repair,
    )
    monkeypatch.setattr(
        "src.core.epub.xhtml_translator.translate_chunk_with_fallback",
        fake_repair,
    )

    with pytest.raises(ChunkTranslationFailedError, match="rejected"):
        await _audit_epub_chunks(
            [candidate],
            chunks,
            source_language="German",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=client,
            placeholder_format=("[id", "]"),
            prompt_options={
                "audit_entire_book": True,
                "repair_until_pass": True,
                "max_repair_rounds": 1,
                "fidelity_adjudicate_rejections": False,
            },
            checkpoint_callback=lambda index, text, meta: checkpoints.append((index, text, meta)),
        )

    record = chunks[0]["unit"]
    assert record["translation"] == repaired
    assert record["translation_hash"] == text_sha256(repaired)
    assert record["translation_status"] == "COMPLETED"
    assert record["review_status"] == "PENDING"
    assert record["audit_status"] == "FAILED"
    assert checkpoints[-1][1] == repaired


@pytest.mark.asyncio
async def test_issue_local_audit_repair_preserves_placeholders_and_returns_global_ids():
    class RepairDecision:
        def to_dict(self):
            return {
                "accepted": False,
                "judge_changed_facts": ["Terry sur -> Terry Southern"],
            }

    source = (
        "Terry Southern schrieb den Text.[id0]"
        "Der vollständige Absatz bleibt in derselben Reihenfolge."
    )
    candidate = (
        "Terry sur escribió el texto.[id0]"
        "El párrafo completo permanece en el mismo orden."
    )
    repaired = (
        "<TRANSLATION>Terry Southern escribió el texto.[id0]"
        "El párrafo completo permanece en el mismo orden.</TRANSLATION>"
    )
    chunk = {
        "local_tag_map": {"[id0]": "</p><p>"},
        "global_indices": [17],
    }

    result = await _repair_audited_candidate(
        source_text=source,
        candidate_text=candidate,
        decision=RepairDecision(),
        chunk=chunk,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=AuditClient(repaired),
        prompt_options={},
        placeholder_format=("[id", "]"),
    )

    assert result == (
        "Terry Southern escribió el texto.[id17]"
        "El párrafo completo permanece en el mismo orden."
    )


@pytest.mark.asyncio
async def test_full_stage_audit_reuses_completed_unit_without_new_request():
    chunks, candidate = _chunks()
    mark_audited(
        chunks[0]["unit"],
        candidate,
        review={"status": "pass"},
        audit={"judge_decision": "pass"},
    )
    client = SequenceAuditClient()
    checkpoints = []

    result = await _audit_epub_chunks(
        [candidate],
        chunks,
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=client,
        placeholder_format=("[id", "]"),
        prompt_options={"audit_entire_book": True},
        checkpoint_callback=lambda index, text, meta: checkpoints.append((index, text, meta)),
    )

    assert result == [candidate]
    assert client.calls == 0
    assert checkpoints[0][2]["status"] == "cached"


@pytest.mark.asyncio
async def test_refinement_reuses_completed_review_without_new_request():
    chunks, candidate = _chunks()
    checkpoints = []

    result = await _refine_epub_chunks(
        translated_chunks=[candidate],
        chunks=chunks,
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=SequenceAuditClient(),
        context_manager=None,
        placeholder_format=("[id", "]"),
        log_callback=None,
        prompt_options={"strict_stage_contract": True},
        checkpoint_callback=lambda *args: checkpoints.append(args),
        source_language="German",
    )

    assert result == [candidate]
    assert checkpoints[0][3]["status"] == "cached"
