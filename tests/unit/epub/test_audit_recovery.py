"""Fault injection for persisted final-audit candidates, not literary scores."""

from __future__ import annotations

import json
from dataclasses import asdict
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
from src.core.fidelity_supervisor import (
    FidelityDecision, FidelityIssue, apply_fidelity_audit_assessment, assess_fidelity,
)
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ProviderActionRequiredError, RateLimitError

SOURCE = "The traveler watched the quiet sea and remembered his childhood."
ORIGINAL = "El viajero contempla el mar tranquilo y recuerda su infancia."
REPAIRED = "El viajero contemplaba el mar tranquilo y recordaba su infancia."
OPTIONS = {
    "audit_entire_book": True,
    "fidelity_adjudicate_rejections": False,
    "max_repair_rounds": 2,
}


def fixture(count=1):
    chunks = [
        {"text": SOURCE, "local_tag_map": {}, "global_indices": []}
        for _ in range(count)
    ]
    for record in ensure_unit_records(chunks, "Text/chapter.xhtml"):
        mark_reviewed(record, ORIGINAL, review={"status": "pass"})
    return chunks, [ORIGINAL] * count


def verdict(source, candidate, accepted, defects=()):
    return FidelityDecision(
        chunk_index=1,
        phase="final_epub_unit_audit",
        section="chapter",
        accepted=accepted,
        source_hash=text_sha256(source),
        candidate_hash=text_sha256(candidate),
        judge_decision="pass" if accepted else "fail",
        judge_confidence=0.99,
        judge_changed_facts=list(defects),
        issues=[]
        if accepted
        else [FidelityIssue("fidelity_judge_reject", "reject", "Defect")],
    ), LLMResponse(content="structured assessment")


async def run(chunks, candidates, *, options=None, checkpoint=None, interrupted=None):
    return await subject._audit_epub_chunks(
        candidates,
        chunks,
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        llm_client=SimpleNamespace(provider_type="deepseek"),
        placeholder_format=("[id", "]"),
        prompt_options=options or OPTIONS,
        checkpoint_callback=checkpoint,
        check_interruption_callback=interrupted,
    )


@pytest.mark.asyncio
async def test_worsening_repair_keeps_best_candidate_and_stops_after_reload(
    monkeypatch,
):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(("audit", candidate))
        defects = ["tense"] if candidate == ORIGINAL else ["tense", "invented event"]
        return verdict(source, candidate, False, defects)

    async def repair(**kwargs):
        calls.append(("repair", kwargs["candidate_text"]))
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates)
    assert candidates == [ORIGINAL]
    assert chunks[0]["unit"]["translation"] == ORIGINAL
    assert chunks[0]["unit"]["review_status"] == "COMPLETED"
    saved = json.loads(json.dumps(chunks))
    assert REPAIRED in json.dumps(saved)
    before = list(calls)
    with pytest.raises(ChunkTranslationFailedError):
        await run(saved, list(candidates))
    assert calls == before
    assert len([c for c in calls if c[0] == "repair"]) == 2
    assert not validate_publishable_units(saved)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_stop", [False, True])
async def test_remaining_round_uses_failed_repair_evidence_without_rejudging(
    monkeypatch, legacy_stop
):
    chunks, candidates = fixture()
    corrected = "El viajero miraba el mar en calma y evocaba su infancia."
    audits, repairs = [], []

    async def audit(source, candidate, **kwargs):
        audits.append(candidate)
        if candidate == ORIGINAL:
            return verdict(source, candidate, False, ["Wrong tense"])
        if candidate == REPAIRED:
            return verdict(source, candidate, False, ["Different reported defect"])
        return verdict(source, candidate, candidate == corrected)

    async def repair(**kwargs):
        assert kwargs["candidate_text"] == ORIGINAL
        repairs.append(kwargs)
        return REPAIRED if len(repairs) == 1 else corrected

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    if legacy_stop:
        # Recreate the old durable stop with one round left under current limits.
        with pytest.raises(ChunkTranslationFailedError):
            await run(chunks, candidates, options={**OPTIONS, "max_repair_rounds": 1})
        journal = chunks[0]["unit"]["audit_repair"]
        journal["stop_reason"] = "repair_no_improvement"
        chunks = json.loads(json.dumps(chunks))
        assert candidates == [ORIGINAL]
        assert len(audits) == 2

    assert await run(chunks, candidates) == [corrected]
    assert audits == [ORIGINAL, REPAIRED, corrected]
    assert len(repairs) == 2
    assert not repairs[0].get("repair_feedback")
    assert repairs[1]["repair_feedback"]["judge_changed_facts"] == [
        "Different reported defect"
    ]
    assert chunks[0]["unit"]["audit_repair_attempts"] == 2
    assert validate_publishable_units(chunks)[0]


@pytest.mark.asyncio
async def test_generated_candidate_survives_audit_rate_limit_without_regeneration(
    monkeypatch,
):
    chunks, candidates = fixture()
    repairs = []
    audits = []
    snapshots = []
    fail_once = True

    async def audit(source, candidate, **kwargs):
        nonlocal fail_once
        audits.append(candidate)
        if candidate == ORIGINAL:
            return verdict(source, candidate, False, ["tense"])
        assert REPAIRED in json.dumps(snapshots[-1]), "save generation before audit"
        if fail_once:
            fail_once = False
            raise RateLimitError("wait", provider="deepseek", retry_after=1)
        return verdict(source, candidate, True)

    async def repair(**kwargs):
        repairs.append(kwargs)
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    checkpoint = lambda *_: snapshots.append(json.loads(json.dumps(chunks)))
    with pytest.raises(RateLimitError):
        await run(chunks, candidates, checkpoint=checkpoint)
    restored = snapshots[-1]
    result = await run(restored, [restored[0]["unit"]["translation"]])
    assert result == [REPAIRED]
    assert len(repairs) == 1
    assert audits == [ORIGINAL, REPAIRED, REPAIRED]
    assert validate_publishable_units(restored)[0]


@pytest.mark.asyncio
async def test_explicit_zero_repairs_does_not_fall_back_to_two(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        calls.append(kwargs)
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates, options={**OPTIONS, "max_repair_rounds": 0})
    assert calls == []


@pytest.mark.asyncio
async def test_rejected_unit_does_not_prevent_independent_audits(monkeypatch):
    chunks, candidates = fixture(3)
    visited = []

    async def audit(source, candidate, **kwargs):
        visited.append(kwargs["chunk_index"])
        return verdict(source, candidate, kwargs["chunk_index"] == 2, ["tense"])

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    with pytest.raises(ChunkTranslationFailedError) as caught:
        await run(chunks, candidates, options={**OPTIONS, "repair_until_pass": False})
    assert visited == [1, 2, 3]
    assert caught.value.failed_unit_indices == [0, 2]
    assert chunks[1]["unit"]["status"] == "AUDITED"
    assert not validate_publishable_units(chunks)[0]
    visited.clear()
    with pytest.raises(ChunkTranslationFailedError):
        await run(
            json.loads(json.dumps(chunks)),
            candidates,
            options={**OPTIONS, "repair_until_pass": False},
        )
    assert visited == []


@pytest.mark.asyncio
async def test_candidate_changed_during_repair_is_not_overwritten(monkeypatch):
    chunks, candidates = fixture()
    newer = "A newer independently reviewed candidate."

    async def audit(source, candidate, **kwargs):
        return verdict(source, candidate, candidate != ORIGINAL, ["tense"])

    async def repair(**kwargs):
        mark_reviewed(chunks[0]["unit"], newer, review={"status": "pass"})
        candidates[0] = newer
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError) as caught:
        await run(chunks, candidates)
    assert caught.value.reason == "audit_version_conflict"
    assert candidates == [newer]
    assert chunks[0]["unit"]["translation"] == newer
    assert chunks[0]["unit"]["audit_status"] != "COMPLETED"


@pytest.mark.asyncio
async def test_no_checkpoint_means_no_new_paid_request(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, True)

    def rejected_save(*_):
        raise OSError("disk full")

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    with pytest.raises(OSError, match="disk full"):
        await run(chunks, candidates, checkpoint=rejected_save)
    assert calls == []


@pytest.mark.asyncio
async def test_interrupted_adjudication_reuses_first_judgment(monkeypatch):
    chunks, candidates = fixture()
    phases = []

    async def audit(source, candidate, **kwargs):
        phase = kwargs["phase"]
        phases.append(phase)
        if len(phases) == 2:
            raise RateLimitError("temporary")
        return verdict(source, candidate, "adjudication" in phase, ["tense"])

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    options = {**OPTIONS, "fidelity_adjudicate_rejections": True}
    with pytest.raises(RateLimitError):
        await run(chunks, candidates, options=options)
    restored = json.loads(json.dumps(chunks))
    assert await run(restored, candidates, options=options) == [ORIGINAL]
    assert phases == [
        "final_epub_unit_audit",
        "final_epub_unit_adjudication",
        "final_epub_unit_adjudication",
    ]
    assert restored[0]["unit"].get("audit_repair_attempts", 0) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [ORIGINAL, None])
async def test_same_or_invalid_repairs_are_finite_across_resume(
    monkeypatch, replacement
):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append("audit")
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        calls.append("repair")
        return replacement

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    for _ in range(4):
        with pytest.raises(ChunkTranslationFailedError):
            await run(chunks, candidates)
        chunks = json.loads(json.dumps(chunks))
    assert calls.count("audit") == 1
    assert calls.count("repair") == (1 if replacement else 2)
    assert candidates == [ORIGINAL]


@pytest.mark.asyncio
async def test_never_approving_auditor_repeated_findings_stop_without_style_ranking(
    monkeypatch,
):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    for _ in range(3):
        with pytest.raises(ChunkTranslationFailedError):
            await run(chunks, candidates)
        chunks = json.loads(json.dumps(chunks))
    assert calls == [ORIGINAL, REPAIRED]
    assert candidates == [ORIGINAL]
    assert chunks[0]["unit"]["audit_repair"]["stop_reason"] == "repair_repeated_candidate"


@pytest.mark.asyncio
async def test_transient_audit_failures_have_durable_per_candidate_limit(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(*args, **kwargs):
        calls.append(kwargs)
        raise RateLimitError("temporary")

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    for _ in range(3):
        with pytest.raises(RateLimitError):
            await run(chunks, candidates)
        chunks = json.loads(json.dumps(chunks))
    for _ in range(2):
        with pytest.raises(ChunkTranslationFailedError):
            await run(chunks, candidates)
    assert len(calls) == 3
    assert chunks[0]["unit"]["audit_attempts"] == 3


@pytest.mark.asyncio
async def test_later_refusal_is_saved_and_never_rewritten_on_resume(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        if candidate == REPAIRED:
            raise ProviderActionRequiredError(
                "refused", reason="provider_refusal", provider="deepseek"
            )
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    for _ in range(2):
        with pytest.raises(ProviderActionRequiredError) as caught:
            await run(chunks, candidates)
        assert caught.value.reason == "provider_refusal"
        chunks = json.loads(json.dumps(chunks))
    assert calls == [ORIGINAL, REPAIRED]
    assert candidates == [ORIGINAL]
    assert REPAIRED in json.dumps(chunks)


@pytest.mark.asyncio
async def test_runtime_checkpoint_rejection_stops_before_final_audit(monkeypatch):
    from lxml import etree

    calls = []

    async def translate(**kwargs):
        results = []
        for chunk in kwargs["chunks"]:
            candidate = chunk["text"].replace(SOURCE, ORIGINAL)
            mark_reviewed(chunk["unit"], candidate, review={"status": "pass"})
            results.append(candidate)
        return results, kwargs["stats"], False

    async def audit(*args, **kwargs):
        calls.append(kwargs)
        pytest.fail("A failed durable save must prevent another request")

    monkeypatch.setattr(subject, "_translate_all_chunks_with_checkpoint", translate)
    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    manager = SimpleNamespace(save_xhtml_partial_state=lambda *_: False)
    root = etree.fromstring(f"<html><body><p>{SOURCE}</p></body></html>")
    with pytest.raises(ProviderActionRequiredError) as caught:
        await subject.translate_xhtml_simplified(
            root,
            "English",
            "Spanish",
            "deepseek-v4-pro",
            SimpleNamespace(provider_type="deepseek"),
            prompt_options={**OPTIONS, "refine": True},
            checkpoint_manager=manager,
            translation_id="test",
            file_href="Text/chapter.xhtml",
        )
    assert caught.value.reason == "checkpoint_write_failed"
    assert calls == []


@pytest.mark.asyncio
async def test_pending_candidate_survives_real_checkpoint_reload(monkeypatch, tmp_path):
    from src.core.epub.xhtml_translation_state import XHTMLTranslationState
    from src.persistence.checkpoint_manager import CheckpointManager

    monkeypatch.chdir(tmp_path)
    chunks, candidates = fixture()
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
    repairs = []
    fail_once = True

    async def audit(source, candidate, **kwargs):
        nonlocal fail_once
        if candidate == ORIGINAL:
            return verdict(source, candidate, False, ["tense"])
        if fail_once:
            fail_once = False
            raise RateLimitError("temporary")
        return verdict(source, candidate, True)

    async def repair(**kwargs):
        repairs.append(kwargs)
        return REPAIRED

    def checkpoint(*_):
        assert manager.save_xhtml_partial_state("test", state.file_href, state)

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    try:
        with pytest.raises(RateLimitError):
            await run(chunks, candidates, checkpoint=checkpoint)
        loaded = manager.load_xhtml_partial_state("test", state.file_href)
        assert loaded is not None and loaded.validate()
        assert loaded.translated_chunks == [ORIGINAL]
        assert await run(loaded.chunks, loaded.translated_chunks) == [REPAIRED]
        assert len(repairs) == 1
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_pause_after_generation_resumes_with_audit_not_another_repair(
    monkeypatch,
):
    chunks, candidates = fixture()
    paused = False
    repairs = []

    async def audit(source, candidate, **kwargs):
        return verdict(source, candidate, candidate == REPAIRED, ["tense"])

    async def repair(**kwargs):
        repairs.append(kwargs)
        return REPAIRED

    def checkpoint(_idx, _text, metadata):
        nonlocal paused
        if metadata["status"] == "candidate_saved":
            paused = True

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    assert (
        await run(chunks, candidates, checkpoint=checkpoint, interrupted=lambda: paused)
        == []
    )
    assert chunks[0]["unit"]["audit_status"] != "COMPLETED"
    assert await run(json.loads(json.dumps(chunks)), candidates) == [REPAIRED]
    assert len(repairs) == 1


@pytest.mark.asyncio
async def test_explicit_larger_repair_allowance_does_not_repeat_saved_audit(
    monkeypatch,
):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, candidate == REPAIRED, ["tense"])

    async def repair(**kwargs):
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates, options={**OPTIONS, "max_repair_rounds": 0})
    assert await run(
        chunks, candidates, options={**OPTIONS, "max_repair_rounds": 1}
    ) == [REPAIRED]
    assert calls == [ORIGINAL, REPAIRED]


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", ["repeated_dialogue", "untranslated_clause", "material_defect", "checksum"])
async def test_terminal_checkpoint_rechecks_only_superseded_language_alert(monkeypatch, blocker):
    source = "Mire, mire! called the driver from the other side of the bridge."
    translation = "\u2014\u00a1Mire, mire! \u2014grit\u00f3 el conductor desde el otro lado del puente."
    if blocker == "untranslated_clause":
        source += " [id0]The bridge collapsed into the river."
        translation += " [id0]The bridge collapsed into the river."
    tags = {"[id0]": "<br/>"} if blocker == "untranslated_clause" else {}
    chunks = [{"text": source, "local_tag_map": tags, "global_indices": [0] if tags else []}]
    record, = ensure_unit_records(chunks, "Text/chapter.xhtml")
    mark_reviewed(record, translation, review={"status": "pass"})
    calls = []

    async def legacy_audit(source, candidate, **kwargs):
        calls.append("audit")
        decision = assess_fidelity(
            source, candidate, chunk_index=1, phase=kwargs["phase"],
            source_language=kwargs["source_language"], target_language=kwargs["target_language"],
        )
        # Simulate the former hard rejection for a copied, repeated utterance.
        if blocker != "untranslated_clause":
            decision.issues = [
                FidelityIssue(**{**asdict(issue), "severity": "reject"})
                if issue.code == "source_language_residual" else issue
                for issue in decision.issues
            ]
        decision.local_accepted = decision.accepted = False
        return apply_fidelity_audit_assessment(
            decision,
            {
                "verdict": "warn", "confidence": 0.91,
                "reason": "The repeated expression is already in the target language.",
                "changed_facts": ["Changed the speaker's identity"] if blocker == "material_defect" else [],
            },
            model="deepseek-v4-pro",
        ), LLMResponse(content="structured assessment")

    async def legacy_repair(**kwargs):
        calls.append("repair")
        return None

    monkeypatch.setattr(subject, "supervise_fidelity", legacy_audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", legacy_repair)
    with pytest.raises(ChunkTranslationFailedError, match="repair_attempts_exhausted"):
        await run(chunks, [translation])
    assert calls == ["audit", "repair", "repair"]
    saved = json.loads(json.dumps(chunks))
    record = saved[0]["unit"]
    if blocker == "checksum":
        record["audit_repair"]["candidates"][0]["assessments"]["audit"]["checksum"] = "invalid"

    if blocker == "repeated_dialogue":
        assert await run(saved, [translation]) == [translation]
        assert validate_publishable_units(saved)[0]
        assert await run(json.loads(json.dumps(saved)), [translation]) == [translation]
    else:
        with pytest.raises(ChunkTranslationFailedError):
            await run(saved, [translation])
        assert not validate_publishable_units(saved)[0]
    assert record["audit_attempts"] == 1
    assert record["audit_repair_attempts"] == 2
    assert calls == ["audit", "repair", "repair"]


@pytest.mark.asyncio
async def test_profile_content_change_invalidates_saved_judgment(monkeypatch, tmp_path):
    from dataclasses import replace

    from src.core.book_profiles import loader
    from src.core.book_profiles.models import BookProfile

    chunks, candidates = fixture()
    profiles = [
        BookProfile("test", "test", tmp_path, policy_text="Preserve formal address.")
    ]
    monkeypatch.setattr(loader, "load_book_profile", lambda _: profiles[0])
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, True)

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    options = {**OPTIONS, "profile_id": "test"}
    assert await run(chunks, candidates, options=options) == [ORIGINAL]
    assert await run(chunks, candidates, options=options) == [ORIGINAL]
    assert len(calls) == 1
    profiles[0] = replace(
        profiles[0], policy_text="Keep the narrator's regional vocabulary."
    )
    assert await run(chunks, candidates, options=options) == [ORIGINAL]
    assert len(calls) == 2
    assert len(chunks[0]["unit"]["audit_repair_history"]) == 1


@pytest.mark.asyncio
async def test_changed_decision_checksum_cannot_approve_on_resume(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, False, ["tense"])

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates, options={**OPTIONS, "max_repair_rounds": 0})
    saved = chunks[0]["unit"]["audit_repair"]["candidates"][0]["assessments"]["audit"]
    saved["decision"]["accepted"] = True
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates, options={**OPTIONS, "max_repair_rounds": 1})
    assert len(calls) == 1
    assert not validate_publishable_units(chunks)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("was_truncated", True),
        ("status", "incomplete"),
        ("finish_reason", "length"),
        ("refusal", "refused"),
    ],
)
async def test_repair_never_accepts_incomplete_or_refused_response(field, value):
    chunks, _ = fixture()
    response = LLMResponse(content=f"<TRANSLATION>{REPAIRED}</TRANSLATION>")
    setattr(response, field, value)

    async def generate(*args, **kwargs):
        return response

    async def attempt():
        return await subject._repair_audited_candidate(
            source_text=SOURCE,
            candidate_text=ORIGINAL,
            decision=verdict(SOURCE, ORIGINAL, False, ["tense"])[0],
            chunk=chunks[0],
            source_language="English",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_client=SimpleNamespace(provider_type="deepseek", generate=generate),
            prompt_options={},
            placeholder_format=("[id", "]"),
        )

    if field == "refusal":
        with pytest.raises(ProviderActionRequiredError):
            await attempt()
    else:
        assert await attempt() is None


def test_best_candidate_requires_objective_improvement_not_confidence():
    from src.core.epub.audit_repair import strictly_better

    previous = verdict(SOURCE, ORIGINAL, False, ["tense", "omission"])[0]
    improved = verdict(SOURCE, REPAIRED, False, ["tense"])[0]
    assert strictly_better(previous, improved)
    assert not strictly_better(improved, previous)
    cosmetic = verdict(SOURCE, REPAIRED, False, ["tense", "omission"])[0]
    cosmetic.judge_confidence = 1.0
    assert not strictly_better(previous, cosmetic)
    improved.judge_issues = ["A new invented event"]
    assert not strictly_better(previous, improved)
    passed = verdict(SOURCE, ORIGINAL, True)[0]
    assert not strictly_better(passed, cosmetic)


@pytest.mark.asyncio
async def test_observability_updates_do_not_invalidate_the_candidate_contract(
    monkeypatch,
):
    chunks, candidates = fixture()
    records = []

    async def audit(source, candidate, **kwargs):
        kwargs["prompt_options"]["_candidate_results"].append({"candidate": candidate})
        return verdict(source, candidate, True)

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    options = {**OPTIONS, "_candidate_results": records}
    assert await run(chunks, candidates, options=options) == [ORIGINAL]
    assert await run(chunks, candidates, options=options) == [ORIGINAL]
    assert len(records) == 1


@pytest.mark.asyncio
async def test_completed_unit_does_not_bypass_persisted_decision_integrity(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, True)

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    assert await run(chunks, candidates) == [ORIGINAL]
    saved = json.loads(json.dumps(chunks))
    assessment = saved[0]["unit"]["audit_repair"]["candidates"][0]["assessments"]["audit"]
    assessment["decision"]["accepted"] = False
    with pytest.raises(ChunkTranslationFailedError, match="audit_checkpoint_invalid"):
        await run(saved, candidates)
    assert calls == [ORIGINAL]
    assert not validate_publishable_units(saved)[0]


@pytest.mark.asyncio
async def test_explicit_audit_allowance_increase_preserves_counters(monkeypatch):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(candidate)
        return verdict(source, candidate, True)

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    options = {**OPTIONS, "epub_audit_max_attempts": 0}
    with pytest.raises(ChunkTranslationFailedError, match="audit_attempts_exhausted"):
        await run(chunks, candidates, options=options)
    saved = json.loads(json.dumps(chunks))
    with pytest.raises(ChunkTranslationFailedError, match="audit_attempts_exhausted"):
        await run(saved, candidates, options=options)
    assert calls == []
    assert await run(saved, candidates, options={**options, "epub_audit_max_attempts": 1}) == [ORIGINAL]
    assert calls == [ORIGINAL] and saved[0]["unit"]["audit_attempts"] == 1


@pytest.mark.asyncio
async def test_repair_does_not_add_another_adjudication_layer(monkeypatch):
    chunks, candidates = fixture()
    phases = []

    async def audit(source, candidate, **kwargs):
        phases.append((candidate, kwargs["phase"]))
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError):
        await run(
            chunks,
            candidates,
            options={**OPTIONS, "fidelity_adjudicate_rejections": True},
        )
    assert phases == [
        (ORIGINAL, "final_epub_unit_audit"),
        (ORIGINAL, "final_epub_unit_adjudication"),
        (REPAIRED, "final_epub_unit_audit"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["pass", "fail", "warn", "structure", "unverified"])
@pytest.mark.parametrize("legacy", [False, True])
async def test_conflicting_repairs_adjudicated_once_and_reused(monkeypatch, outcome, legacy):
    chunks, candidates = fixture()
    calls = []

    async def audit(source, candidate, **kwargs):
        phase = kwargs["phase"].removeprefix("final_epub_unit_")
        calls.append(phase)
        if phase == "repair_adjudication":
            context = kwargs["prompt_options"]["_fidelity_adjudication_context"]["repair_conflict"]
            assert context["shared_source_evidence"] == ["remembered his childhood"]
            assert context["previous_rendering_evidence"] == ["recuerda su infancia"]
            assert context["repaired_rendering_evidence"] == ["recordaba su infancia"]
            decision, response = verdict(source, candidate, outcome != "fail")
            if outcome == "warn":
                decision.judge_decision = "warn"
            if outcome == "structure":
                decision.judge_structure_issues = ["Lost a heading"]
            return decision, response
        decision, response = verdict(source, candidate, False, ["Disputed reading"])
        decision.judge_evidence_source = [
            "invented source quote" if outcome == "unverified" else "remembered his childhood"
        ]
        decision.judge_evidence_candidate = [
            "recuerda su infancia" if candidate == ORIGINAL else "recordaba su infancia"
        ]
        return decision, response

    async def repair(**kwargs):
        calls.append("repair")
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    options = {**OPTIONS, "max_repair_rounds": 1, "fidelity_adjudicate_rejections": True}
    if legacy:
        with pytest.raises(ChunkTranslationFailedError, match="repair_no_improvement"):
            await run(chunks, candidates, options={**options, "fidelity_adjudicate_rejections": False})
        # Old workers used the same contract with adjudication enabled for
        # the original but not for repairs. Reproduce that terminal journal.
        from src.core.epub.audit_repair import contract_hash

        chunk, = chunks
        chunk["unit"]["audit_repair"]["contract"] = contract_hash(
            chunk["unit"], chunk,
            {**options, "fidelity_supervisor": True, "fidelity_supervisor_mode": "strict_full",
             "fidelity_supervisor_model": "deepseek-v4-pro"},
            source_language="English", target_language="Spanish",
            model_name="deepseek-v4-pro", provider="deepseek", placeholder_format=("[id", "]"),
        )
        chunks = json.loads(json.dumps(chunks))
    for _ in range(2):
        if outcome == "pass":
            assert await run(chunks, candidates, options=options) == [REPAIRED]
            assert validate_publishable_units(chunks)[0]
        else:
            with pytest.raises(ChunkTranslationFailedError, match="repair_no_improvement"):
                await run(chunks, candidates, options=options)
            assert candidates == [ORIGINAL]
            assert not validate_publishable_units(chunks)[0]
        chunks = json.loads(json.dumps(chunks))
    assert calls.count("repair") == 1
    assert calls.count("repair_adjudication") == int(outcome != "unverified")
    assert chunks[0]["unit"]["audit_repair_attempts"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["audit_attempts", "audit_repair_attempts"])
@pytest.mark.parametrize("value", [None, -1, "2", True, 1.5])
async def test_corrupt_lifetime_counters_cannot_replenish_requests(
    monkeypatch, field, value
):
    chunks, candidates = fixture()
    chunks[0]["unit"][field] = value
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append("audit")
        return verdict(source, candidate, False, ["tense"])

    async def repair(**kwargs):
        calls.append("repair")
        return REPAIRED

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    with pytest.raises(ChunkTranslationFailedError):
        await run(chunks, candidates)
    assert calls == ([] if field == "audit_attempts" else ["audit"])


@pytest.mark.asyncio
async def test_runtime_counts_every_unresolved_audit_unit(monkeypatch):
    from lxml import etree

    chunks, _ = fixture(3)
    saved = []
    monkeypatch.setattr(subject, "_create_chunks", lambda *args: chunks)

    async def translate(**kwargs):
        for chunk in kwargs["chunks"]:
            mark_reviewed(chunk["unit"], ORIGINAL, review={"status": "pass"})
        return [ORIGINAL] * 3, kwargs["stats"], False

    async def audit(source, candidate, **kwargs):
        return verdict(source, candidate, kwargs["chunk_index"] == 2, ["tense"])

    def save(_job, _file, state):
        saved.append(json.loads(json.dumps(state.to_dict())))
        return True

    monkeypatch.setattr(subject, "_translate_all_chunks_with_checkpoint", translate)
    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    root = etree.fromstring(f"<html><body><p>{SOURCE}</p></body></html>")
    with pytest.raises(ChunkTranslationFailedError) as caught:
        await subject.translate_xhtml_simplified(
            root,
            "English",
            "Spanish",
            "deepseek-v4-pro",
            SimpleNamespace(provider_type="deepseek"),
            prompt_options={**OPTIONS, "refine": True, "repair_until_pass": False},
            checkpoint_manager=SimpleNamespace(save_xhtml_partial_state=save),
            translation_id="test",
            file_href="Text/chapter.xhtml",
            global_total_chunks=3,
            global_completed_chunks=0,
        )
    assert caught.value.failed_unit_indices == [0, 2]
    assert caught.value.stats.failed_chunks == 2
    assert saved[-1]["global_stats"]["failed_chunks"] == 2
    assert root.find("body/p").text == SOURCE
