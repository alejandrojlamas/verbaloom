"""Compatible repair responses are durable evidence, never audit approvals."""

import copy
import json
from types import SimpleNamespace

import pytest

from src.core.epub import audit_repair, xhtml_translator as subject
from src.core.epub.exceptions import ChunkTranslationFailedError
from src.core.epub.unit_contract import ensure_unit_records, mark_reviewed, text_sha256
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ProviderActionRequiredError

SOURCE = "The traveler watched the quiet sea and remembered his childhood."
ORIGINAL = "El viajero contempla el mar tranquilo y recuerda su infancia."
REPAIRED = "El viajero contemplaba el mar tranquilo y recordaba su infancia."
OPTIONS = {
    "audit_entire_book": True,
    "fidelity_adjudicate_rejections": False,
    "epub_audit_repair_rounds": 1,
    "fidelity_repair_policy": "compatible",
}


class WorkerLost(BaseException):
    pass


def fixture(monkeypatch, *, response_state=None):
    chunks = [{"text": SOURCE, "local_tag_map": {}, "global_indices": []}]
    record, = ensure_unit_records(chunks, "Text/chapter.xhtml")
    mark_reviewed(record, ORIGINAL, review={"status": "pass"})
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(("audit", candidate))
        accepted = candidate == REPAIRED
        return FidelityDecision(
            chunk_index=1, phase=kwargs["phase"], section="chapter",
            accepted=accepted, source_hash=text_sha256(source),
            candidate_hash=text_sha256(candidate),
            judge_decision="pass" if accepted else "fail", judge_confidence=0.99,
            judge_changed_facts=[] if accepted else ["Wrong tense"],
            issues=[] if accepted else [FidelityIssue(
                "fidelity_judge_reject", "reject", "Wrong tense",
            )],
        ), LLMResponse(content="synthetic assessment")

    async def generate(*args, **kwargs):
        calls.append(("repair", ORIGINAL))
        return LLMResponse(
            content=f"{subject.TRANSLATE_TAG_IN}{REPAIRED}{subject.TRANSLATE_TAG_OUT}",
            request_id="synthetic-compatible-request", **(response_state or {}),
        )

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_generate_alert_repair", generate)
    return chunks, [ORIGINAL], calls


async def run(chunks, candidates, checkpoint=None, options=None):
    return await subject._audit_epub_chunks(
        candidates, chunks, source_language="English", target_language="Spanish",
        model_name="test-model", llm_client=SimpleNamespace(provider_type="openai"),
        placeholder_format=("[id", "]"), prompt_options=options or OPTIONS,
        checkpoint_callback=checkpoint,
    )


def crash_after_response(_index, _text, metadata):
    if metadata["status"] == "repair_patch_received":
        raise WorkerLost()


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_mode", ["compatible", "localized_v1"])
async def test_compatible_response_survives_crash_and_keeps_original_mode(
    monkeypatch, resume_mode,
):
    chunks, candidates, calls = fixture(monkeypatch)
    with pytest.raises(WorkerLost):
        await run(chunks, candidates, crash_after_response)
    saved = json.loads(json.dumps(chunks))
    envelope = saved[0]["unit"]["audit_repair"]["localized_patches"][0]
    assert envelope["mode"] == "compatible" and envelope["valid"] is None
    assert envelope["response_state"]["request_id"] == "synthetic-compatible-request"
    assert REPAIRED in envelope["raw_response"] and not envelope["processed"]
    assert await run(saved, candidates, options={
        **OPTIONS, "fidelity_repair_policy": resume_mode,
    }) == [REPAIRED]
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    assert saved[0]["unit"]["audit_repair_attempts"] == 1
    assert envelope["processed"]
    completed = copy.deepcopy(saved[0]["unit"]["audit_repair"])
    monkeypatch.setattr(audit_repair, "COMPATIBLE_REPAIR_VALIDATION_VERSION", "future-revision")
    assert await run(saved, candidates) == [REPAIRED]
    assert saved[0]["unit"]["audit_repair"] == completed and len(calls) == 3


@pytest.mark.asyncio
async def test_changed_local_validation_reuses_invalid_response_without_new_repair(
    monkeypatch,
):
    chunks, candidates, calls = fixture(monkeypatch)
    reject = FidelityIssue("target_language_missing", "reject", "Former local defect")
    monkeypatch.setattr(subject, "_target_language_gate_rejections", lambda *a, **k: [reject])
    with pytest.raises(ChunkTranslationFailedError, match="repair_attempts_exhausted"):
        await run(chunks, candidates)
    saved = json.loads(json.dumps(chunks))
    unit = saved[0]["unit"]
    journal = unit["audit_repair"]
    old_audit = copy.deepcopy(journal["candidates"][0]["assessments"])
    assert len(journal["candidates"]) == 1
    assert REPAIRED in journal["localized_patches"][0]["raw_response"]
    assert unit["audit_repair_attempts"] == 1 and unit["audit_attempts"] == 1
    # Same revision does not re-run paid work or replenish exhausted requests.
    with pytest.raises(ChunkTranslationFailedError):
        await run(saved, candidates)
    assert len(calls) == 2
    monkeypatch.setattr(subject, "_target_language_gate_rejections", lambda *a, **k: [])
    monkeypatch.setattr(audit_repair, "COMPATIBLE_REPAIR_VALIDATION_VERSION", "test-fixed-v2")
    assert await run(saved, candidates) == [REPAIRED]
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    assert unit["audit_repair_attempts"] == 1 and unit["audit_attempts"] == 2
    assert journal["candidates"][0]["assessments"] == old_audit
    assert len(journal["repair_failures"]) == 1
    assert await run(saved, candidates) == [REPAIRED]
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_replayed_candidate_still_obeys_the_existing_audit_budget(monkeypatch):
    chunks, candidates, calls = fixture(monkeypatch)
    reject = FidelityIssue("target_language_missing", "reject", "Former local defect")
    monkeypatch.setattr(subject, "_target_language_gate_rejections", lambda *a, **k: [reject])
    options = {**OPTIONS, "epub_audit_max_attempts": 1}
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates, options=options)
    monkeypatch.setattr(subject, "_target_language_gate_rejections", lambda *a, **k: [])
    monkeypatch.setattr(audit_repair, "COMPATIBLE_REPAIR_VALIDATION_VERSION", "test-fixed-v2")
    with pytest.raises(ChunkTranslationFailedError, match="audit_attempts_exhausted"):
        await run(chunks, candidates, options=options)
    unit = chunks[0]["unit"]
    assert len(calls) == 2
    assert unit["audit_attempts"] == unit["audit_repair_attempts"] == 1
    assert unit["audit_status"] != "COMPLETED"
    assert len(unit["audit_repair"]["candidates"]) == 2
    assert candidates == [ORIGINAL]
    assert await run(chunks, candidates, options={
        **options, "epub_audit_max_attempts": 2,
    }) == [REPAIRED]
    assert [call[0] for call in calls] == ["audit", "repair", "audit"]
    assert unit["audit_attempts"] == 2 and unit["audit_repair_attempts"] == 1


@pytest.mark.asyncio
async def test_tampered_saved_compatible_response_blocks_before_any_request(monkeypatch):
    chunks, candidates, calls = fixture(monkeypatch)
    with pytest.raises(WorkerLost):
        await run(chunks, candidates, crash_after_response)
    saved = json.loads(json.dumps(chunks))
    saved[0]["unit"]["audit_repair"]["localized_patches"][0]["raw_response"] += "tampered"
    with pytest.raises(ChunkTranslationFailedError, match="repair_checkpoint_invalid"):
        await run(saved, candidates)
    assert [call[0] for call in calls] == ["audit", "repair"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "candidate", "unit", "model", "decision"])
async def test_saved_compatible_response_rejects_stale_request_identity(monkeypatch, changed):
    chunks, candidates, calls = fixture(monkeypatch)
    with pytest.raises(WorkerLost):
        await run(chunks, candidates, crash_after_response)
    chunk = copy.deepcopy(chunks[0])
    journal = chunk["unit"]["audit_repair"]
    response = journal["localized_patches"][0]
    decision = FidelityDecision.from_dict(journal["candidates"][0]["assessments"]["audit"]["decision"])
    source, candidate, model = SOURCE, ORIGINAL, "test-model"
    if changed == "source":
        source += " Another sentence."
    elif changed == "candidate":
        candidate += " Otra oración."
    elif changed == "unit":
        chunk["unit"]["unit_id"] = "another-unit"
    elif changed == "model":
        model = "another-model"
    else:
        decision.judge_changed_facts = ["Different defect"]
    with pytest.raises(ProviderActionRequiredError, match="current contract"):
        await subject._repair_audited_candidate(
            source_text=source, candidate_text=candidate, decision=decision, chunk=chunk,
            source_language="English", target_language="Spanish", model_name=model,
            llm_client=SimpleNamespace(provider_type="openai"), prompt_options=OPTIONS,
            placeholder_format=("[id", "]"), saved_patch=response,
        )
    assert [call[0] for call in calls] == ["audit", "repair"]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [
    {"refusal": "Provider refused", "status": "refused"},
    {"was_truncated": True, "finish_reason": "length"},
])
async def test_saved_refusal_or_truncation_cannot_become_an_approved_candidate(
    monkeypatch, state,
):
    chunks, candidates, calls = fixture(monkeypatch, response_state=state)
    with pytest.raises(WorkerLost):
        await run(chunks, candidates, crash_after_response)
    saved = json.loads(json.dumps(chunks))
    with pytest.raises((ChunkTranslationFailedError, ProviderActionRequiredError)):
        await run(saved, candidates)
    monkeypatch.setattr(audit_repair, "COMPATIBLE_REPAIR_VALIDATION_VERSION", "test-fixed-v2")
    with pytest.raises((ChunkTranslationFailedError, ProviderActionRequiredError)):
        await run(saved, candidates)
    assert [call[0] for call in calls] == ["audit", "repair"]
    assert saved[0]["unit"]["audit_repair_attempts"] == 1
    assert saved[0]["unit"]["audit_status"] != "COMPLETED"
    assert candidates == [ORIGINAL]
