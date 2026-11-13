"""Real EPUB repair orchestration with deterministic model boundary responses."""

import json
from types import SimpleNamespace

import pytest

from src.core.epub import xhtml_translator as subject
from src.core.epub.exceptions import ChunkTranslationFailedError
from src.core.epub.unit_contract import (
    ensure_unit_records,
    mark_reviewed,
    text_sha256,
    validate_publishable_units,
)
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.llm.base import LLMResponse
from src.core.pipeline_config import pipeline_preset, validate_pipeline

SOURCE = "The traveler watched the quiet sea and remembered his childhood."
ORIGINAL = "El viajero contempla el mar tranquilo y recuerda su infancia."
REPAIRED = "El viajero contemplaba el mar tranquilo y recordaba su infancia."
OPTIONS = {
    "audit_entire_book": True,
    "fidelity_adjudicate_rejections": False,
    "max_repair_rounds": 1,
    "fidelity_repair_policy": "localized_v1",
}


def setup():
    chunks = [{"text": SOURCE, "local_tag_map": {}, "global_indices": []}]
    for record in ensure_unit_records(chunks, "Text/chapter.xhtml"):
        mark_reviewed(record, ORIGINAL, review={"status": "pass"})
    return chunks, [ORIGINAL]


def verdict(source, candidate, accepted):
    return FidelityDecision(
        chunk_index=1,
        phase="final_epub_unit_audit",
        section="chapter",
        accepted=accepted,
        source_hash=text_sha256(source),
        candidate_hash=text_sha256(candidate),
        judge_decision="pass" if accepted else "fail",
        judge_changed_facts=[] if accepted else ["tense"],
        issues=[]
        if accepted
        else [FidelityIssue("fidelity_judge_reject", "reject", "Wrong tense")],
    ), LLMResponse(content="synthetic assessment")


def patch_reply(contract):
    data = {
        key: contract[key]
        for key in (
            "schema",
            "unit_id",
            "source_sha256",
            "candidate_sha256",
            "contract_sha256",
        )
    }
    data["patches"] = [
        {
            "finding_id": contract["findings"][0]["id"],
            "block_id": contract["candidate_blocks"][0]["id"],
            "operation": "replace",
            "anchor": before,
            "replacement": after,
            "source_block_id": contract["source_blocks"][0]["id"],
            "source_quote": evidence,
        }
        for before, after, evidence in (
            ("contempla", "contemplaba", "watched"),
            ("recuerda", "recordaba", "remembered"),
        )
    ]
    return data


def models(monkeypatch, *, response_change=None, approved=True):
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(("audit", candidate))
        return verdict(source, candidate, candidate == REPAIRED and approved)

    async def generate(_client, prompt, system_prompt, **kwargs):
        calls.append(("repair", prompt))
        assert "Do not return a complete rewritten candidate" in system_prompt
        assert "Return only the complete repaired candidate" not in system_prompt
        assert kwargs["phase"] == "repair"
        contract = json.loads(prompt.split("\n", 1)[1])
        response = LLMResponse(
            content=json.dumps(patch_reply(contract)), request_id="synthetic-request"
        )
        return response_change(response, contract) if response_change else response

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_generate_alert_repair", generate)
    return calls


async def run(chunks, candidates, checkpoint=None, options=None):
    return await subject._audit_epub_chunks(
        candidates,
        chunks,
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=SimpleNamespace(provider_type="deepseek"),
        placeholder_format=("[id", "]"),
        prompt_options=OPTIONS if options is None else options,
        checkpoint_callback=checkpoint,
    )


@pytest.mark.asyncio
async def test_patch_is_saved_before_full_candidate_audit_and_then_selected(
    monkeypatch,
):
    chunks, candidates = setup()
    calls = models(monkeypatch)
    snapshots = []
    assert await run(
        chunks,
        candidates,
        lambda *args: snapshots.append(json.loads(json.dumps(chunks))),
    ) == [REPAIRED]
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    envelope = chunks[0]["unit"]["audit_repair"]["localized_patches"][0]
    assert envelope["valid"] and envelope["processed"]
    assert envelope["response_state"]["request_id"] == "synthetic-request"
    assert len(envelope["edits"]) == 2
    received = next(
        s for s in snapshots if s[0]["unit"]["audit_repair"].get("localized_patches")
    )
    assert received[0]["unit"]["translation"] == ORIGINAL
    assert not received[0]["unit"]["audit_repair"]["localized_patches"][0]["processed"]
    assert validate_publishable_units(chunks)[0]
    assert await run(json.loads(json.dumps(chunks)), list(candidates)) == [REPAIRED]
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_policy", ["compatible", "localized_v1"])
async def test_crash_after_response_resumes_without_another_model_repair(
    monkeypatch, resume_policy
):
    class WorkerLost(BaseException):
        pass

    chunks, candidates = setup()
    calls = models(monkeypatch)
    snapshots = []

    def save(_index, _text, metadata):
        snapshots.append(json.loads(json.dumps(chunks)))
        if metadata["status"] == "repair_patch_received":
            raise WorkerLost()

    with pytest.raises(WorkerLost):
        await run(chunks, candidates, save)
    restored = snapshots[-1]
    assert await run(
        restored,
        [ORIGINAL],
        options={**OPTIONS, "fidelity_repair_policy": resume_policy},
    ) == [REPAIRED]
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    assert restored[0]["unit"]["audit_repair_attempts"] == 1


@pytest.mark.asyncio
async def test_received_patch_survives_real_checkpoint_reload(monkeypatch, tmp_path):
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState
    from src.persistence.checkpoint_manager import CheckpointManager

    class WorkerLost(BaseException):
        pass

    monkeypatch.chdir(tmp_path)
    chunks, candidates = setup()
    calls = models(monkeypatch)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    state = XHTMLTranslationState(
        file_path="Text/chapter.xhtml",
        translation_id="test",
        file_href="Text/chapter.xhtml",
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=450,
        max_retries=1,
        chunks=chunks,
        global_tag_map={},
        placeholder_format=("[id", "]"),
        translated_chunks=candidates,
        current_chunk_index=1,
        original_body_html=SOURCE,
        doc_metadata={},
        stats={},
        created_at="test",
        updated_at="test",
        strict_contract=True,
        config_fingerprint="configuration",
        source_document_hash=text_sha256(SOURCE),
    )

    def checkpoint(_index, _text, metadata):
        assert manager.save_xhtml_partial_state("test", state.file_href, state)
        if metadata["status"] == "repair_patch_received":
            raise WorkerLost()

    try:
        with pytest.raises(WorkerLost):
            await run(chunks, candidates, checkpoint)
        loaded = manager.load_xhtml_partial_state("test", state.file_href)
        assert loaded is not None and loaded.validate()
        assert loaded.translated_chunks == [ORIGINAL]
        assert await run(loaded.chunks, loaded.translated_chunks) == [REPAIRED]
        assert [call[0] for call in calls] == ["audit", "repair", "audit"]
        assert loaded.chunks[0]["unit"]["audit_repair_attempts"] == 1
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_crash_with_invalid_saved_patch_does_not_buy_a_second_attempt(
    monkeypatch,
):
    class WorkerLost(BaseException):
        pass

    chunks, candidates = setup()

    def malformed(response, _contract):
        response.content = "not JSON"
        return response

    calls = models(monkeypatch, response_change=malformed)

    def checkpoint(_index, _text, metadata):
        if metadata["status"] == "repair_patch_received":
            raise WorkerLost()

    with pytest.raises(WorkerLost):
        await run(chunks, candidates, checkpoint)
    restored = json.loads(json.dumps(chunks))
    with pytest.raises(ChunkTranslationFailedError, match="repair_attempts_exhausted"):
        await run(restored, [ORIGINAL])
    assert len(calls) == 2
    journal = restored[0]["unit"]["audit_repair"]
    assert journal["repair_failures"][0]["attempt"] == 1
    assert journal["localized_patches"][0]["processed"]


@pytest.mark.asyncio
async def test_corrupt_saved_patch_stops_before_a_second_request(monkeypatch):
    class WorkerLost(BaseException):
        pass

    chunks, candidates = setup()
    calls = models(monkeypatch)

    def save(_index, _text, metadata):
        if metadata["status"] == "repair_patch_received":
            raise WorkerLost()

    with pytest.raises(WorkerLost):
        await run(chunks, candidates, save)
    restored = json.loads(json.dumps(chunks))
    restored[0]["unit"]["audit_repair"]["localized_patches"][0]["raw_response"] = (
        "tampered"
    )
    with pytest.raises(ChunkTranslationFailedError, match="repair_checkpoint_invalid"):
        await run(restored, [ORIGINAL])
    assert len(calls) == 2 and not validate_publishable_units(restored)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_response", ["rewrite", "truncated", "wrong_version", "unrelated_id", "overlap"]
)
async def test_invalid_patch_never_falls_back_to_full_retranslation(
    monkeypatch, bad_response
):
    chunks, candidates = setup()

    def change(response, contract):
        data = json.loads(response.content)
        if bad_response == "rewrite":
            response.content = f"<TRANSLATION>{REPAIRED}</TRANSLATION>"
            return response
        if bad_response == "truncated":
            response.was_truncated = True
        if bad_response == "wrong_version":
            data["candidate_sha256"] = "wrong"
        if bad_response == "unrelated_id":
            data["patches"][0]["finding_id"] = "invented"
        if bad_response == "overlap":
            data["patches"].append(dict(data["patches"][0]))
        response.content = json.dumps(data)
        return response

    calls = models(monkeypatch, response_change=change)
    for _ in range(2):
        with pytest.raises(ChunkTranslationFailedError):
            await run(chunks, candidates)
        chunks = json.loads(json.dumps(chunks))
    assert candidates == [ORIGINAL]
    assert [call[0] for call in calls] == ["audit", "repair"]
    assert not validate_publishable_units(chunks)[0]


@pytest.mark.asyncio
async def test_valid_patch_is_not_semantic_approval(monkeypatch):
    chunks, candidates = setup()
    calls = models(monkeypatch, approved=False)
    with pytest.raises(ChunkTranslationFailedError, match="repair_no_improvement"):
        await run(chunks, candidates)
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    assert candidates == [ORIGINAL]
    assert not validate_publishable_units(chunks)[0]


def test_policy_is_explicit_and_serializable_without_changing_compatible_default():
    baseline = pipeline_preset("compatible", {})
    assert baseline["epub_repair_policy"] == "compatible"
    selected = validate_pipeline({**baseline, "epub_repair_policy": "localized_v1"})
    assert validate_pipeline(json.loads(json.dumps(selected))) == selected
    assert baseline["epub_repair_policy"] == "compatible"
    for invalid in ([], True, "unknown"):
        with pytest.raises(ValueError):
            validate_pipeline({**baseline, "epub_repair_policy": invalid})


def test_patch_policy_does_not_invalidate_existing_semantic_approvals():
    from src.core.epub.unit_contract import stage_fingerprints

    def fingerprints(options):
        return stage_fingerprints(
            source_language="English",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            max_tokens_per_chunk=450,
            max_retries=2,
            prompt_options=options,
        )

    original, updated = (
        fingerprints({}),
        fingerprints({"fidelity_repair_policy": "localized_v1"}),
    )
    assert original["translation"] == updated["translation"]
    assert original["review"] == updated["review"]
    assert original["audit"] == updated["audit"]


@pytest.mark.asyncio
async def test_changing_repair_format_does_not_repeat_completed_audit(monkeypatch):
    chunks, candidates = setup()
    calls = models(monkeypatch)
    assert await run(chunks, candidates) == [REPAIRED]
    assert await run(
        json.loads(json.dumps(chunks)),
        candidates,
        options={**OPTIONS, "fidelity_repair_policy": "compatible"},
    ) == [REPAIRED]
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["compatible", "localized_v1"])
async def test_effective_policy_reaches_real_xhtml_entrypoint(monkeypatch, policy):
    from lxml import etree

    from src.core.epub.unit_contract import mark_audited
    from src.core.pipeline_runtime import active_pipeline

    config = validate_pipeline(
        {**pipeline_preset("compatible", {}), "epub_repair_policy": policy}
    )
    received = []

    async def translate(**kwargs):
        candidates = []
        for chunk in kwargs["chunks"]:
            candidate = chunk["text"].replace(SOURCE, ORIGINAL)
            mark_reviewed(chunk["unit"], candidate, review={"status": "pass"})
            candidates.append(candidate)
        return candidates, kwargs["stats"], False

    async def audit(candidates, chunks, **kwargs):
        received.append(
            kwargs["prompt_options"].get("fidelity_repair_policy", "compatible")
        )
        for candidate, chunk in zip(candidates, chunks):
            mark_audited(chunk["unit"], candidate, audit={"accepted": True})
        return candidates

    monkeypatch.setattr(subject, "_translate_all_chunks_with_checkpoint", translate)
    monkeypatch.setattr(subject, "_audit_epub_chunks", audit)
    token = active_pipeline.set(SimpleNamespace(config=config))
    try:
        root = etree.fromstring(f"<html><body><p>{SOURCE}</p></body></html>")
        success, _ = await subject.translate_xhtml_simplified(
            root,
            "English",
            "Spanish",
            "deepseek-v4-pro",
            SimpleNamespace(provider_type="deepseek"),
            prompt_options={**OPTIONS, "refine": True},
            translation_id="test",
            file_href="Text/chapter.xhtml",
        )
        assert (
            success
            and root.xpath("string(//*[local-name()='body']/*[local-name()='p'])")
            == ORIGINAL
        )
        assert received == [policy]
    finally:
        active_pipeline.reset(token)
