"""Production TXT orchestration and real ledgers with synthetic HTTP only."""

import asyncio
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import editorial_benchmark_run as runner
from src.core.book_profiles.loader import frozen_profile_scope, load_book_profile
from src.core.book_profiles.models import (
    BookProfile,
    ProfileDetector,
    ProfileGlossaryEntry,
    ProfileGlossaryExample,
)
from src.core.editorial_benchmark import corpus_digest, prepare_corpus
from src.core.pipeline_runtime import active_pipeline, request_role
from src.core.usage.budget import SpendBudget
from src.core.usage.context import get_usage_context
from tests.unit.test_editorial_benchmark import clean_assessment
from tests.unit.test_source_audited_strategy import ASSESSMENT

CREDENTIALS = {"openai": "YOUR_API_KEY_HERE", "deepseek": "YOUR_API_KEY_HERE"}


def corpus(*, book_id="synthetic"):
    profile = asdict(
        BookProfile(
            "frozen-test",
            "Frozen test",
            Path("unused"),
            target_locale="es-ES",
            policy_text="Preserve voice and meaning.",
        )
    )
    profile.pop("root")
    return prepare_corpus(
        [
            {
                "locator": f"p:{i}",
                "position": i,
                "section": "chapter",
                "text": f"The traveler watched the quiet sea and remembered his childhood on day {i}.",
            }
            for i in range(4)
        ],
        book_id=book_id,
        source_sha256="a" * 64,
        source_language="English",
        target_language="Spanish",
        target_locale="es-ES",
        profile=profile,
        sequences=[
            {"start": 0, "count": 2, "split": "tuning"},
            {"start": 2, "count": 2, "split": "heldout"},
        ],
    )


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = corpus()
    baseline = runner.baseline_example(value)
    plan = runner.make_plan(value, baseline, split="tuning", seed="qa")
    directory = runner.initialize(tmp_path / "private", value, plan)
    calls = []
    fault = {"judge": None}

    class HTTP:
        async def post(self, url, **kwargs):
            runtime = active_pipeline.get()
            assert runtime is not None
            scope = next(
                s for s in runtime.budget_scope_ids if s.startswith("experiment:")
            )
            assert SpendBudget(runtime.usage_store).snapshot(scope)["reserved_usd"] > 0
            payload = kwargs["json"]
            system = payload["messages"][0]["content"]
            user = payload["messages"][-1]["content"]
            role = request_role.get()
            is_judge = "independent bilingual editorial evaluator" in system
            calls.append(
                {
                    "url": url,
                    "role": role,
                    "model": payload["model"],
                    "user": user,
                    "judge": is_judge,
                    "context": get_usage_context(),
                    "scope_ids": runtime.budget_scope_ids,
                }
            )
            if fault.get("dispatch"):
                await fault["dispatch"](calls[-1])
            if is_judge:
                answer = fault["judge"] or json.dumps(clean_assessment())
            elif role == "audit":
                answer = ASSESSMENT
            elif fault.get("echo_source") and role in {"refine", "repair"}:
                unit = next(
                    u
                    for u in value["units"]
                    if u["unit_id"] == get_usage_context()["unit_id"]
                )
                answer = "<TRANSLATION>" + unit["source_text"] + "</TRANSLATION>"
            else:
                source = re.findall(
                    r"<SOURCE_TEXT>\s*(.*?)\s*</SOURCE_TEXT>", user, re.DOTALL
                )[-1]
                number = re.search(r"(?:day|d[ií]a) (\d+)", source).group(1)
                answer = (
                    "<TRANSLATION>El viajero contemplaba el mar tranquilo y recordaba "
                    f"su infancia el día {number}.</TRANSLATION>"
                )
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": answer}}
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 30},
                    "model": payload["model"],
                    "id": "synthetic-response",
                },
            )

    async def get_client(_):
        return HTTP()

    monkeypatch.setattr("src.core.llm.base.LLMProvider._get_client", get_client)
    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.get_deepseek_pricing_status",
        lambda: SimpleNamespace(disabled=False),
    )
    return directory, calls, fault


def test_plan_keeps_roles_language_profile_and_code_fixed():
    value = corpus()
    plan = runner.make_plan(
        value, runner.baseline_example(value), split="heldout", seed="qa"
    )
    assert (
        plan["variants"]["A"]["pipeline"]["roles"]
        == plan["variants"]["C"]["pipeline"]["roles"]
    )
    assert (
        plan["variants"]["B"]["pipeline"]["roles"]
        == plan["variants"]["D"]["pipeline"]["roles"]
    )
    assert all(
        c["prompt_options"]["target_locale"] == "es-ES"
        for c in plan["variants"].values()
    )
    assert all(t["sequence"] == "sequence-2" for t in plan["order"])
    assert plan["code_sha256"] == runner.code_digest()


@pytest.mark.asyncio
async def test_abcd_runs_real_engine_captures_candidates_and_resumes_without_paid_repeats(
    experiment,
):
    directory, calls, _ = experiment
    result = await runner.execute(
        directory, credentials=CREDENTIALS, evidence="simulated"
    )
    assert result["status"] == "completed"
    assert len(result["evaluations"]) == 8
    assert result["budget"]["calls"] == len(calls)
    assert {c["url"] for c in calls} == {
        "https://api.deepseek.com/chat/completions",
        "https://api.openai.com/v1/chat/completions",
    }
    assert sum(c["role"] == "translate" for c in calls) == 8
    assert sum(c["role"] == "refine" for c in calls) == 4
    assert not any(c["role"] == "repair" for c in calls), (
        "An unchanged good refinement is not untranslated source"
    )
    assert sum(c["judge"] for c in calls) == 8
    assert not any("day 2." in c["user"] or "day 3." in c["user"] for c in calls)
    assert all(c["context"]["process_type"] == "editorial_benchmark" for c in calls)
    assert all(c["context"].get("unit_id") for c in calls)
    expected_units = {u["unit_id"] for u in corpus()["units"] if u["split"] == "tuning"}
    assert {c["context"]["unit_id"] for c in calls} == expected_units
    assert "Simulated execution" in (directory / "blind.html").read_text()
    assert list(directory.glob("*/candidate-*.json"))
    assert not (directory.parent / "data").exists()
    report = json.loads((directory / "report.json").read_text())
    assert report["variants"]["A"]["generation"]["calls_by_stage"]["refine"] == 2
    assert report["variants"]["C"]["generation"]["calls_by_stage"].get("refine", 0) == 0
    assert report["variants"]["D"]["evaluation"]["calls"] == 2
    assert report["variants"]["B"]["independently_approved_source_words"] > 0
    assert report["api_costs_are_simulated"]
    assert report["variants"]["C"]["audited_unit_fraction"] == 1
    assert report["variants"]["D"]["repaired_unit_fraction"] == 0
    assert set(report["variants"]["A"]["generation"]["units"]) == expected_units
    for variant in report["variants"].values():
        for category in ("generation", "evaluation"):
            group = variant[category]
            assert sum(s["calls"] for s in group["stages"].values()) == group["calls"]
            assert sum(s["calls"] for s in group["units"].values()) == group["calls"]
            assert (
                group["prompt_tokens"]
                == group["input_uncached_tokens"]
                + group["input_cache_read_tokens"]
                + group["input_cache_write_tokens"]
            )
    before = len(calls)
    await runner.execute(
        directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
    )
    assert len(calls) == before
    assert active_pipeline.get() is None and get_usage_context() == {}


@pytest.mark.asyncio
async def test_real_source_echo_still_rejects_refinement_and_retains_approved_translation(
    experiment,
):
    directory, calls, fault = experiment
    fault["echo_source"] = True
    await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    assert any(c["role"] == "repair" for c in calls)
    records = [
        json.loads(p.read_text())["candidate"]
        for p in directory.glob("*/candidate-*.json")
    ]
    assert any(
        issue["code"] == "untranslated_source" for r in records for issue in r["issues"]
    )
    candidates = json.loads((directory / "candidates.json").read_text())
    assert all(c["text"].startswith("El viajero") for c in candidates)


@pytest.mark.asyncio
async def test_malformed_judge_is_saved_once_and_never_approved_or_silently_retried(
    experiment,
):
    directory, calls, fault = experiment
    fault["judge"] = '{"verdict":"pass"}'
    with pytest.raises(ValueError):
        await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    assert list(directory.glob("*/judge-*.json"))
    assert not (directory / "blind.html").exists()
    with pytest.raises(ValueError):
        await runner.execute(
            directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["candidates.json", "blind.html", "answer-key.json"])
async def test_resume_refuses_changed_published_evidence(experiment, name):
    directory, calls, _ = experiment
    await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    (directory / name).write_text("{}" if name.endswith("json") else "Changed sample")
    with pytest.raises(ValueError, match="artifact"):
        await runner.execute(
            directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


def test_profile_snapshot_preserves_glossary_examples_detectors_and_editorial_artifacts(
    tmp_path,
):
    profile = BookProfile(
        "frozen-test",
        "Snapshot test",
        Path("do-not-load"),
        target_locale="es-ES",
        prompt_texts={"translation": "Retain the changing narrative register."},
        glossary_entries=(
            ProfileGlossaryEntry(
                "captain",
                target="capit\u00e1n",
                status="approved",
                target_options=("capitana",),
                examples=(ProfileGlossaryExample("the captain", "el capit\u00e1n"),),
                restrictions=("naval rank",),
                applies_to={"characters": ["Ada"]},
                do_not_apply_if=("sports",),
            ),
            ProfileGlossaryEntry("pending term", status="pending"),
        ),
        detectors=(ProfileDetector("marker", r"\[broken\]", "critical"),),
        editorial_artifacts={"voice_map": {"Ada": "abrupt"}},
        editorial_signal_index={"locations": [4, 9]},
        raw_config={"loaded_glossaries": ["common", "book"]},
    )
    value = corpus()
    value["book"]["profile"] = asdict(profile)
    value["book"]["profile"].pop("root")
    value["content_sha256"] = corpus_digest(value)
    snapshot = runner.profile_snapshot(json.loads(json.dumps(value)), tmp_path)
    assert asdict(snapshot) | {"root": profile.root} == asdict(profile)
    assert snapshot.approved_entries[0].examples[0].target_excerpt == "el capit\u00e1n"


@pytest.mark.asyncio
async def test_unknown_inflight_spend_is_not_freed_or_retried_on_resume(experiment):
    directory, calls, fault = experiment
    from src.core.editorial_benchmark import _digest
    from src.core.usage.budget import BudgetLimitError

    plan = json.loads((directory / "plan.json").read_text())
    plan["budget"]["max_calls"] = 1
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    runner._replace(directory / "plan.json", plan)

    async def cancel_after_dispatch(_):
        raise asyncio.CancelledError()

    fault["dispatch"] = cancel_after_dispatch
    with pytest.raises(asyncio.CancelledError):
        await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    first = json.loads((directory / "state.json").read_text())["budget"]
    assert first["reserved_usd"] > 0 and first["calls"] == 1
    with pytest.raises(BudgetLimitError):
        await runner.execute(
            directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == 1
    assert json.loads((directory / "state.json").read_text())["budget"] == first


@pytest.mark.asyncio
async def test_cancellation_drains_pending_inline_refinement(experiment, monkeypatch):
    directory, calls, _ = experiment
    from src.core.editorial_benchmark import _digest

    plan = json.loads((directory / "plan.json").read_text())
    plan["order"].sort(key=lambda task: task["variant"])
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    runner._replace(directory / "plan.json", plan)
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def unfinished_refinement(self, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    monkeypatch.setattr(
        runner.GenericTranslator, "_refine_unit_inline", unfinished_refinement
    )
    task = asyncio.create_task(
        runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    )
    await asyncio.wait_for(started.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=3)
    assert cancelled.is_set()
    assert sum(c["role"] == "translate" for c in calls) == 1
    assert active_pipeline.get() is None and get_usage_context() == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changed", ["assessment", "receipt", "missing_receipt", "missing_state"]
)
async def test_resume_checks_raw_evidence_even_after_completion(experiment, changed):
    directory, calls, _ = experiment
    await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    if changed == "assessment":
        path = directory / "state.json"
        state = json.loads(path.read_text())
        next(iter(state["evaluations"].values()))["accepted"] = False
        runner._replace(path, state)
    elif changed == "receipt":
        path = next(directory.glob("*/judge-*.json"))
        receipt = json.loads(path.read_text())
        receipt["input_sha256"] = "changed"
        runner._replace(path, receipt)
    elif changed == "missing_receipt":
        next(directory.glob("*/judge-*.json")).unlink()
    else:
        (directory / "state.json").unlink()
    with pytest.raises(ValueError):
        await runner.execute(
            directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


@pytest.mark.asyncio
async def test_money_cap_refuses_dispatch_and_survives_resume(experiment):
    directory, calls, _ = experiment
    from src.core.editorial_benchmark import _digest
    from src.core.usage.budget import BudgetLimitError

    plan = json.loads((directory / "plan.json").read_text())
    plan["budget"]["limit_usd"] = "0.01"
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    runner._replace(directory / "plan.json", plan)
    for resume in (False, True):
        with pytest.raises(BudgetLimitError):
            await runner.execute(
                directory, credentials=CREDENTIALS, resume=resume, evidence="simulated"
            )
        assert not calls
        state = json.loads((directory / "state.json").read_text())
        assert state["budget"]["calls"] == 0
        assert state["status"] == "needs_review"


@pytest.mark.asyncio
async def test_missing_key_stops_entire_matrix_before_any_dispatch(experiment):
    directory, calls, _ = experiment
    with pytest.raises(ValueError, match="Both providers"):
        await runner.execute(directory, credentials={"deepseek": "YOUR_API_KEY_HERE"})
    assert not calls and not (directory / "usage.db").exists()


@pytest.mark.asyncio
async def test_second_executor_cannot_steal_a_live_experiment(experiment):
    directory, calls, _ = experiment
    with (
        runner.execution_lock(directory),
        pytest.raises(ValueError, match="already owns"),
    ):
        await runner.execute(directory, credentials=CREDENTIALS)
    assert not calls


def test_profile_scope_never_reads_live_or_cross_book_profiles(tmp_path):
    snapshot = runner.profile_snapshot(corpus(), tmp_path)
    with frozen_profile_scope(snapshot):
        assert load_book_profile(snapshot.profile_id) == snapshot
        with pytest.raises(RuntimeError, match="outside"):
            load_book_profile("another-book", allow_missing=True)
    assert (
        load_book_profile(
            snapshot.profile_id, profiles_root=tmp_path, allow_missing=True
        )
        is None
    )


@pytest.mark.parametrize(
    "field", ["profile_id", "target_locale", "text_transform_mode"]
)
def test_plan_refuses_uncontrolled_editorial_changes(field):
    value = corpus()
    baseline = runner.baseline_example(value)
    baseline["prompt_options"][field] = "changed"
    with pytest.raises(ValueError):
        runner.make_plan(value, baseline, split="tuning", seed="qa")


@pytest.mark.asyncio
async def test_modified_code_requires_a_new_experiment(experiment, monkeypatch):
    directory, calls, _ = experiment
    monkeypatch.setattr(runner, "code_digest", lambda: "changed")
    with pytest.raises(ValueError, match="Frozen code"):
        await runner.execute(directory, credentials=CREDENTIALS)
    assert not calls


class WorkerLost(BaseException):
    pass


@pytest.mark.asyncio
async def test_worker_loss_after_judge_receipt_reuses_translation_and_raw_judgment(
    experiment, monkeypatch
):
    directory, calls, _ = experiment
    original = runner.write_private
    lost = []

    def save_then_crash(path, payload):
        original(path, payload)
        if path.name.startswith("judge-") and not lost:
            lost.append(path)
            raise WorkerLost()

    monkeypatch.setattr(runner, "write_private", save_then_crash)
    with pytest.raises(WorkerLost):
        await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    prior_calls = list(calls)
    result = await runner.execute(
        directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
    )
    assert result["status"] == "completed"
    assert sum(c["judge"] for c in calls) == 8
    assert sum(c["role"] == "translate" for c in calls) == 8
    assert calls[: len(prior_calls)] == prior_calls
    assert result["budget"]["calls"] == len(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", ["usage.db", "jobs.db"])
async def test_missing_resume_state_never_resets_spend_or_repeats_approved_work(
    experiment, deleted
):
    directory, calls, _ = experiment
    await runner.execute(directory, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    (directory / deleted).unlink()
    with pytest.raises(ValueError, match="missing"):
        await runner.execute(
            directory, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


@pytest.mark.asyncio
async def test_call_cap_is_shared_and_does_not_reset_on_resume(experiment):
    directory, calls, _ = experiment
    from src.core.editorial_benchmark import _digest
    from src.core.usage.budget import BudgetLimitError

    plan = json.loads((directory / "plan.json").read_text())
    plan["budget"]["max_calls"] = 1
    plan["plan_sha256"] = _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    runner._replace(directory / "plan.json", plan)
    for resume in (False, True):
        with pytest.raises(BudgetLimitError):
            await runner.execute(
                directory, credentials=CREDENTIALS, resume=resume, evidence="simulated"
            )
        assert len(calls) == 1
    assert not (directory / "blind.html").exists()
