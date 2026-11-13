"""Real TXT orchestration and SQLite checkpoints; provider boundaries are synthetic."""

import asyncio
import subprocess
import sys
from collections import Counter

import pytest

from src.core.adapters import GenericTranslator, TxtAdapter
from src.core.adapters.stage_journal import JOURNAL_KEY
from src.core.editorial_quality import QualityDecision
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.llm.exceptions import ProviderActionRequiredError
from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.execution_lock import ExecutionLockedError, exclusive_execution

SOURCE = "The door was closed, but she did not turn away from the quiet house."
DRAFT = "La puerta estaba cerrada, pero ella no se alejo de la casa silenciosa."
REFINED = "La puerta estaba cerrada, pero ella no se aparto de la casa silenciosa."
OPTIONS = {
    "inline_refinement": True,
    "translation_memory_enabled": False,
    "fidelity_supervisor_mode": "always",
    "editorial_quality_report": False,
    "_source_language": "English",
}


class WorkerLost(BaseException):
    pass


class Client:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass


@pytest.fixture
def engine(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text(SOURCE, encoding="utf-8")
    output = tmp_path / "translated.txt"
    manager = CheckpointManager(
        str(tmp_path / "jobs.db"),
        uploads_dir=tmp_path / "uploads",
    )
    calls = Counter()
    fail = {"stage": None}

    def visit(stage):
        calls[stage] += 1
        if fail["stage"] == stage:
            fail["stage"] = None
            raise WorkerLost()

    async def translate(**kwargs):
        visit("translation")
        return DRAFT

    async def refine(**kwargs):
        visit("refinement")
        return REFINED, None

    async def quality(**kwargs):
        visit("editorial_audit")
        return QualityDecision(1, "chunk_0", True), None

    async def fidelity(source, candidate, **kwargs):
        visit(kwargs["phase"] + "_audit")
        return FidelityDecision(1, kwargs["phase"], "chunk_0", True), None

    monkeypatch.setattr("src.core.llm_client.LLMClient", Client)
    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    monkeypatch.setattr("src.core.translator._make_refinement_request", refine)
    monkeypatch.setattr(
        "src.core.translator._assess_refinement_with_editorial_guard", quality
    )
    monkeypatch.setattr(
        "src.core.adapters.generic_translator.supervise_fidelity", fidelity
    )

    async def run(
        max_tokens_per_chunk=200,
        interruption=None,
        stats=None,
        job_id="stage-recovery",
        **changes,
    ):
        options = {**OPTIONS, **changes}
        adapter = TxtAdapter(
            source,
            output,
            {
                "max_tokens_per_chunk": max_tokens_per_chunk,
                "prompt_options": options,
            },
        )
        translator = GenericTranslator(adapter, manager, job_id)
        return await translator.translate(
            source_language="English",
            target_language="Spanish",
            llm_provider="fake",
            model_name="fake-model",
            prompt_options=options,
            check_interruption_callback=interruption,
            stats_callback=stats,
        )

    return run, calls, fail, manager, output


@pytest.mark.asyncio
async def test_baseline_stages_and_approved_resume(engine):
    run, calls, _, manager, output = engine
    assert await run()
    assert output.read_text(encoding="utf-8-sig").strip() == REFINED
    assert calls == Counter(
        {
            name: 1
            for name in (
                "translation",
                "translation_audit",
                "refinement",
                "editorial_audit",
                "refinement_audit",
            )
        }
    )
    before = calls.copy()
    assert await run()
    assert calls == before
    assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "interrupted",
    [
        "translation_audit",
        "refinement",
        "editorial_audit",
        "refinement_audit",
    ],
)
async def test_worker_loss_reuses_each_received_stage(engine, interrupted):
    run, calls, fail, manager, output = engine
    fail["stage"] = interrupted
    with pytest.raises(WorkerLost):
        await run()
    saved = manager.load_checkpoint("stage-recovery")
    assert saved["actual_completed_chunks"] == 0
    assert saved["checkpoint_complete"] is False
    assert not output.exists()

    assert await run()
    assert output.read_text(encoding="utf-8-sig").strip() == REFINED
    assert calls["translation"] == 1, (
        "Received translation must survive a later-stage worker loss"
    )
    for stage in (
        "translation_audit",
        "refinement",
        "editorial_audit",
        "refinement_audit",
    ):
        assert calls[stage] == (2 if stage == interrupted else 1), stage


@pytest.mark.asyncio
async def test_stage_attempt_allowance_survives_repeated_worker_loss(engine):
    run, calls, fail, manager, output = engine
    for _ in range(3):
        fail["stage"] = "translation_audit"
        with pytest.raises(WorkerLost):
            await run()
    with pytest.raises(ProviderActionRequiredError) as error:
        await run()
    assert error.value.reason == "stage_attempts_exhausted"
    assert calls == Counter(translation=1, translation_audit=3)
    saved = manager.load_checkpoint("stage-recovery")
    entries = saved["chunks"][0]["chunk_data"][JOURNAL_KEY]["stages"]
    assert [e["attempt"] for e in entries["translation_audit"]] == [1, 2, 3]
    assert entries["translation"][0]["result"] == DRAFT
    assert not output.exists()


@pytest.mark.asyncio
async def test_changed_policy_cannot_reuse_completed_checkpoint(engine):
    run, calls, _, _, output = engine
    assert await run()
    before, file_before = calls.copy(), output.read_bytes()
    with pytest.raises(ProviderActionRequiredError) as error:
        await run(custom_instructions="Use a different editorial policy")
    assert error.value.reason == "checkpoint_dependencies_changed"
    assert calls == before
    assert output.read_bytes() == file_before


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["result", "attempt", "state"])
async def test_corrupt_stage_receipt_never_spends_or_approves(engine, damage):
    run, calls, fail, manager, output = engine
    fail["stage"] = "translation_audit"
    with pytest.raises(WorkerLost):
        await run()
    chunk = manager.db.get_chunks("stage-recovery")[0]
    entry = chunk["chunk_data"][JOURNAL_KEY]["stages"]["translation"][0]
    entry[damage] = {"result": "damaged", "attempt": True, "state": "unknown"}[damage]
    manager.db.save_chunk(
        "stage-recovery", 0, SOURCE, None, chunk["chunk_data"], "pending"
    )
    before = calls.copy()
    with pytest.raises(ProviderActionRequiredError) as error:
        await run()
    assert error.value.reason == "checkpoint_corrupt"
    assert calls == before
    assert not output.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("after_response", [False, True])
async def test_checkpoint_write_failure_stops_before_next_stage(
    engine, monkeypatch, after_response
):
    run, calls, _, manager, output = engine
    save = manager.db.save_chunk

    def failing_save(*args, **kwargs):
        if after_response:
            stages = args[4][JOURNAL_KEY]["stages"]
            if stages["translation"][0]["state"] != "completed":
                return save(*args, **kwargs)
        return False

    monkeypatch.setattr(manager.db, "save_chunk", failing_save)
    with pytest.raises(ProviderActionRequiredError) as error:
        await run()
    assert error.value.reason == "checkpoint_write_failed"
    assert calls == (Counter(translation=1) if after_response else Counter())
    assert not output.exists()


@pytest.mark.asyncio
async def test_two_engines_cannot_charge_same_unit(engine, monkeypatch):
    run, calls, _, manager, _ = engine
    started, release = asyncio.Event(), asyncio.Event()

    async def held_translation(**kwargs):
        calls["translation"] += 1
        started.set()
        await release.wait()
        return DRAFT

    monkeypatch.setattr(
        "src.core.translator.generate_translation_request", held_translation
    )
    first = asyncio.create_task(run())
    try:
        await asyncio.wait_for(started.wait(), 2)
        assert await run() is False
        assert calls == Counter(translation=1)
        assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 0
    finally:
        release.set()
        assert await first


def test_process_death_releases_real_execution_lock(tmp_path):
    lock = tmp_path / "worker.lock"
    code = """
import sys, time
from pathlib import Path
from src.persistence.execution_lock import exclusive_execution
with exclusive_execution(Path(sys.argv[1])):
    print('owned', flush=True)
    time.sleep(60)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "owned"
        assert process.poll() is None
        with pytest.raises(ExecutionLockedError), exclusive_execution(lock):
            pass
    finally:
        process.kill()
        process.communicate(timeout=5)
    with exclusive_execution(lock):
        assert lock.exists()


@pytest.mark.asyncio
async def test_rejected_retry_preserves_both_candidates_and_best_decision(
    engine, monkeypatch
):
    run, calls, _, manager, output = engine

    async def rejected(source, candidate, **kwargs):
        calls[kwargs["phase"] + "_audit"] += 1
        missing = (
            ["one sentence"]
            if kwargs["phase"] == "translation"
            else ["one sentence", "another sentence"]
        )
        return FidelityDecision(
            1,
            kwargs["phase"],
            "chunk_0",
            False,
            issues=[
                FidelityIssue(
                    "fidelity_judge_reject", "reject", "Missing source material"
                )
            ],
            judge_missing_from_source=missing,
        ), None

    monkeypatch.setattr(
        "src.core.adapters.generic_translator.supervise_fidelity", rejected
    )
    assert await run() is False
    chunk = manager.db.get_chunks("stage-recovery")[0]
    assert chunk["translated_text"] is None
    assert chunk["chunk_data"]["fidelity_decision"]["phase"] == "translation"
    stages = chunk["chunk_data"][JOURNAL_KEY]["stages"]
    assert stages["translation"][0]["result"] == DRAFT
    assert stages["translation_retry"][0]["result"] == DRAFT
    before = calls.copy()
    assert await run() is False
    assert calls == before, (
        "A never-approving auditor must not start a new repair loop on resume"
    )
    assert not output.exists() or not output.read_text(encoding="utf-8-sig").strip()


@pytest.mark.asyncio
async def test_incompatible_legacy_segmentation_stops_before_paid_work(engine):
    run, calls, _, manager, output = engine
    manager.start_job("stage-recovery", "txt", {})
    manager.save_checkpoint(
        "stage-recovery", 0, SOURCE + " Another sentence.", DRAFT, total_chunks=1
    )
    with pytest.raises(ProviderActionRequiredError) as error:
        await run()
    assert error.value.reason == "checkpoint_source_changed"
    assert calls == Counter()
    assert not output.exists()


@pytest.mark.asyncio
async def test_manual_pause_does_not_replace_prior_delivery(engine):
    run, calls, _, manager, output = engine
    output.write_bytes(b"previous valid delivery")
    assert await run(interruption=lambda: True) is False
    assert calls == Counter()
    assert output.read_bytes() == b"previous valid delivery"
    assert manager.get_job("stage-recovery")["status"] == "paused"


@pytest.mark.asyncio
async def test_atomic_publication_failure_keeps_previous_file_and_reuses_units(
    engine, monkeypatch
):
    from pathlib import Path

    run, calls, _, _, output = engine
    output.write_bytes(b"previous valid delivery")
    replace = Path.replace

    def fail_replace(path, target):
        if Path(target) == output:
            raise OSError("injected replace failure")
        return replace(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        assert await run() is False
    assert output.read_bytes() == b"previous valid delivery"
    assert not list(output.parent.glob(".*.tmp"))
    before = calls.copy()
    assert await run()
    assert calls == before
    assert output.read_text(encoding="utf-8-sig").strip() == REFINED


@pytest.mark.asyncio
async def test_txt_adapter_never_fills_missing_units_with_source(tmp_path):
    source, output = tmp_path / "source.txt", tmp_path / "output.txt"
    source.write_text(SOURCE, encoding="utf-8")
    adapter = TxtAdapter(source, output, {"max_tokens_per_chunk": 200})
    assert await adapter.prepare_for_translation()
    with pytest.raises(ValueError, match="untranslated"):
        await adapter.reconstruct_output()


@pytest.mark.asyncio
async def test_rejected_and_provisional_candidates_do_not_update_shared_memory(
    engine, monkeypatch
):
    from src.core.literary_continuity import export_literary_continuity_state

    run, calls, _, manager, output = engine
    source = output.parent / "source.txt"
    source.write_text(
        "First paragraph has enough words to become one translation unit.\n\n"
        "Second paragraph also has enough words to become another unit.",
        encoding="utf-8",
    )
    contexts = []
    stats = []

    async def provisional(**kwargs):
        calls["translation"] += 1
        state = kwargs["runtime_state"]["literary_continuity_state"]
        contexts.append(export_literary_continuity_state(kwargs["runtime_state"]))
        state.recent_events.append((1, "provisional", "REJECTED_CANDIDATE"))
        return DRAFT

    async def audit(source, candidate, **kwargs):
        return FidelityDecision(
            kwargs["chunk_index"],
            kwargs["phase"],
            kwargs["section"],
            not source.startswith("First"),
        ), None

    monkeypatch.setattr("src.core.translator.generate_translation_request", provisional)
    monkeypatch.setattr(
        "src.core.adapters.generic_translator.supervise_fidelity", audit
    )
    assert (
        await run(
            max_tokens_per_chunk=12,
            stats=stats.append,
            inline_refinement=False,
            fidelity_supervisor_retry=False,
            literary_continuity=True,
            text_type="literature",
        )
        is False
    )
    assert len(contexts) == 2
    assert all(not c["recent_events"] for c in contexts)
    saved = manager.load_checkpoint("stage-recovery")
    assert saved["actual_completed_chunks"] == 1
    assert stats[-1] == {"total_chunks": 2, "completed_chunks": 1, "failed_chunks": 1}
    assert "REJECTED_CANDIDATE" not in str(saved["translation_context"])
    assert not output.exists()


@pytest.mark.asyncio
async def test_other_job_cannot_reuse_unapproved_receipts(engine):
    run, calls, fail, manager, _ = engine
    fail["stage"] = "translation_audit"
    with pytest.raises(WorkerLost):
        await run()
    assert await run(job_id="another-book")
    assert calls["translation"] == 2
    assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 0


def test_contract_binds_profile_glossary_roles_context_and_generation_parameters(
    tmp_path, monkeypatch
):
    from dataclasses import replace

    from src.core.adapters.stage_journal import stage_contract
    from src.core.adapters.translation_unit import TranslationUnit
    from src.core.book_profiles import loader
    from src.core.book_profiles.models import BookProfile, ProfileGlossaryEntry

    profile = BookProfile(
        profile_id="book",
        name="Book",
        root=tmp_path,
        policy_text="Preserve rhythm.",
        raw_config={},
    )
    monkeypatch.setattr(loader, "load_book_profile", lambda *_: profile)
    unit = TranslationUnit(
        "paragraph-1", SOURCE, "preceding source", "following source"
    )
    base = {
        "units": [unit],
        "source_language": "English",
        "target_language": "Spanish",
        "model": "fake-model",
        "provider": "fake",
        "options": {"profile_id": "book"},
        "parameters": {"max_tokens": 4000},
    }
    before = stage_contract(**base)
    for changed in (
        {"units": [replace(unit, unit_id="paragraph-2")]},
        {"units": [replace(unit, context_after="different neighbor")]},
        {"model": "other-model"},
        {"source_language": "Czech"},
        {"target_language": "French"},
        {"parameters": {"max_tokens": 8000}},
        {
            "options": {
                "profile_id": "book",
                "_pipeline_roles": {"audit": {"model": "another-judge"}},
            }
        },
    ):
        assert stage_contract(**{**base, **changed}) != before
    assert (
        stage_contract(
            **{
                **base,
                "parameters": {"max_tokens": 4000, "api_key": "YOUR_API_KEY_HERE"},
            }
        )
        == before
    )
    profile = replace(profile, policy_text="Preserve narrator distinctions.")
    assert stage_contract(**base) != before
    before = stage_contract(**base)
    profile = replace(
        profile,
        glossary_entries=(
            ProfileGlossaryEntry("Captain", "capitan", status="approved"),
        ),
    )
    assert stage_contract(**base) != before


@pytest.mark.asyncio
async def test_task_cancellation_retains_draft_and_releases_worker(engine, monkeypatch):
    run, calls, _, manager, _ = engine
    entered = asyncio.Event()

    async def wait_once(source, candidate, **kwargs):
        calls[kwargs["phase"] + "_audit"] += 1
        if not entered.is_set():
            entered.set()
            await asyncio.Event().wait()
        return FidelityDecision(1, kwargs["phase"], "chunk_0", True), None

    monkeypatch.setattr(
        "src.core.adapters.generic_translator.supervise_fidelity", wait_once
    )
    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await run()
    assert calls["translation"] == 1
    assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("literary", [False, True])
async def test_adjacent_stage_resume_uses_same_accepted_context(
    engine, monkeypatch, literary
):
    run, calls, _, manager, output = engine
    (output.parent / "source.txt").write_text(
        "First paragraph has enough words to become one translation unit.\n\n"
        "Second paragraph also has enough words to become another unit.",
        encoding="utf-8",
    )
    interrupted = False
    contexts = []

    async def translate(**kwargs):
        calls["translation"] += 1
        contexts.append(kwargs["previous_translation_context"])
        return DRAFT

    async def fidelity(source, candidate, **kwargs):
        nonlocal interrupted
        if (
            kwargs["chunk_index"] == 2
            and kwargs["phase"] == "refinement"
            and not interrupted
        ):
            interrupted = True
            raise WorkerLost()
        return FidelityDecision(
            kwargs["chunk_index"], kwargs["phase"], kwargs["section"], True
        ), None

    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    monkeypatch.setattr(
        "src.core.adapters.generic_translator.supervise_fidelity", fidelity
    )
    options = {
        "max_tokens_per_chunk": 12,
        "literary_continuity": literary,
        "text_type": "literature",
    }
    with pytest.raises(WorkerLost):
        await run(**options)
    assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 1
    assert await run(**options)
    assert calls["translation"] == 2, (
        "A retained adjacent draft must not be regenerated after resume"
    )
    assert contexts == ["", REFINED], (
        "Adjacent generation depends on the approved final neighbor"
    )


@pytest.mark.asyncio
async def test_auto_literary_memory_starts_after_nonliterary_frontmatter(
    engine, monkeypatch
):
    from src.core.literary_continuity import (
        TextProfile,
        export_literary_continuity_state,
    )

    run, _, _, manager, output = engine
    (output.parent / "source.txt").write_text(
        "Copyright notice has enough words to become one translation unit.\n\n"
        "First scene also has enough words to become another unit.\n\n"
        "Second scene also has enough words to become another unit.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "src.core.literary_continuity.detect_text_profile",
        lambda text: TextProfile(
            "general" if text.startswith("Copyright") else "literature", 1.0, {}
        ),
    )
    contexts = []

    async def translate(**kwargs):
        contexts.append(export_literary_continuity_state(kwargs["runtime_state"]))
        return DRAFT

    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    assert await run(
        max_tokens_per_chunk=12,
        inline_refinement=False,
        literary_continuity=True,
        text_type="auto",
    )
    assert len(contexts) == 3
    assert contexts[0] is None
    assert contexts[1]["chunk_index"] == 0
    assert contexts[2]["chunk_index"] == 1
    saved = manager.load_checkpoint("stage-recovery")
    assert saved["translation_context"]["literary_continuity_state"]["chunk_index"] == 2


@pytest.mark.asyncio
async def test_sparse_approved_units_join_context_in_source_order(engine, monkeypatch):
    from src.core.literary_continuity import export_literary_continuity_state

    run, calls, _, manager, output = engine
    source = output.parent / "source.txt"
    source.write_text(
        "First paragraph has enough words to become one translation unit.\n\n"
        "Second paragraph also has enough words to become another unit.\n\n"
        "Third paragraph also has enough words to become another unit.",
        encoding="utf-8",
    )
    adapter = TxtAdapter(source, output, {"max_tokens_per_chunk": 12})
    assert await adapter.prepare_for_translation()
    units = adapter.get_translation_units()
    assert len(units) == 3
    assert manager.start_job("stage-recovery", "txt", {}, str(source))
    accepted_middle = "El segundo parrafo ya esta aprobado y debe conservarse."
    assert manager.save_checkpoint(
        translation_id="stage-recovery",
        chunk_index=1,
        original_text=units[1].content,
        translated_text=accepted_middle,
        chunk_data={"inline_refinement": True},
        total_chunks=3,
        completed_chunks=1,
        translation_context={"previous_translation_context": "STALE_FUTURE_CONTEXT"},
    )
    contexts = []

    async def translate(**kwargs):
        calls["translation"] += 1
        contexts.append(
            (
                kwargs["previous_translation_context"],
                export_literary_continuity_state(kwargs["runtime_state"]),
            )
        )
        return DRAFT

    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    assert await run(
        max_tokens_per_chunk=12, literary_continuity=True, text_type="literature"
    )
    assert calls["translation"] == 2
    assert contexts[0][0] == ""
    assert contexts[0][1]["chunk_index"] == 0
    assert contexts[1][0] == accepted_middle
    assert contexts[1][1]["chunk_index"] == 2
    assert contexts[1][1]["recent_events"][-2][2] == REFINED
    assert contexts[1][1]["recent_events"][-1][2] == accepted_middle
    assert "STALE_FUTURE_CONTEXT" not in str(contexts)
    assert output.read_text(encoding="utf-8-sig").count(accepted_middle) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [None, "", "   "])
async def test_empty_stage_result_can_recover_without_approving_empty_text(
    engine, monkeypatch, empty
):
    run, calls, _, manager, output = engine

    async def translate(**kwargs):
        calls["translation"] += 1
        return empty if calls["translation"] == 1 else DRAFT

    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    assert not await run(inline_refinement=False)
    assert not output.exists()
    assert manager.load_checkpoint("stage-recovery")["actual_completed_chunks"] == 0
    assert await run(inline_refinement=False)
    assert calls["translation"] == 2
    assert output.read_text(encoding="utf-8-sig").strip() == DRAFT


@pytest.mark.asyncio
async def test_empty_stage_results_exhaust_persisted_allowance(engine, monkeypatch):
    run, calls, _, _, output = engine

    async def translate(**kwargs):
        calls["translation"] += 1

    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    for _ in range(3):
        assert not await run(inline_refinement=False)
    with pytest.raises(ProviderActionRequiredError) as error:
        await run(inline_refinement=False)
    assert error.value.reason == "stage_attempts_exhausted"
    assert calls["translation"] == 3
    assert not output.exists()


@pytest.mark.asyncio
async def test_repeated_cancellation_during_cleanup_releases_execution_lock(
    engine, monkeypatch
):
    run, _, _, manager, _ = engine
    entered, closing = asyncio.Event(), asyncio.Event()

    class ClosingClient(Client):
        async def close(self):
            closing.set()
            await asyncio.Event().wait()

    async def wait_for_cancellation(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("src.core.llm_client.LLMClient", ClosingClient)
    monkeypatch.setattr(
        "src.core.translator.generate_translation_request", wait_for_cancellation
    )
    task = asyncio.create_task(run(inline_refinement=False))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    await asyncio.wait_for(closing.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with manager.execution_lock("stage-recovery"):
        pass


@pytest.mark.asyncio
async def test_disagreeing_checkpoint_indices_stop_before_any_generation(engine):
    run, calls, _, manager, _ = engine
    assert await run()
    before = calls.copy()
    chunk = manager.db.get_chunks("stage-recovery")[0]
    chunk["chunk_data"]["chunk_index"] = 1
    assert manager.db.save_chunk(
        "stage-recovery",
        0,
        chunk["original_text"],
        chunk["translated_text"],
        chunk["chunk_data"],
        status="completed",
    )
    with pytest.raises(ProviderActionRequiredError) as error:
        await run()
    assert error.value.reason == "checkpoint_source_changed"
    assert calls == before
