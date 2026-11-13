"""One gross allowance across books, splits, judges and resumed workers."""

import asyncio
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from scripts import editorial_benchmark_run as runner
from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.store import TokenUsageStore
from tests.unit import test_benchmark_execution as engine_tests

CREDENTIALS = engine_tests.CREDENTIALS
experiment = engine_tests.experiment


def pilot_module():
    from scripts import editorial_benchmark_pilot

    return editorial_benchmark_pilot


def make_child(
    root,
    parent,
    name,
    *,
    split="tuning",
    book="synthetic",
    max_calls=300,
    limit_usd="5",
):
    value = engine_tests.corpus(book_id=book)
    plan = runner.make_plan(
        value,
        runner.baseline_example(value),
        split=split,
        seed=name,
        pilot=parent,
        max_calls=max_calls,
        limit_usd=limit_usd,
    )
    return runner.initialize(root / name, value, plan)


@pytest.mark.asyncio
async def test_two_books_and_splits_share_parent_but_keep_reports_separate(experiment):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    one = make_child(original.parent, parent, "one")
    two = make_child(original.parent, parent, "two", split="heldout", book="other")
    first = await runner.execute(one, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    second = await runner.execute(two, credentials=CREDENTIALS, evidence="simulated")
    snapshot = pilot.pilot_status(parent)
    assert first["status"] == second["status"] == "completed"
    assert snapshot["budget"]["calls"] == len(calls) == 2 * count
    assert snapshot["budget"]["limit_usd"] == 5
    assert first["budget"]["calls"] == second["budget"]["calls"] == count
    assert second["pilot_budget"]["calls"] == len(calls)
    assert len(snapshot["experiments"]) == 2
    assert any(c["judge"] for c in calls)
    assert all(len(c["scope_ids"]) == 3 for c in calls)
    assert all(snapshot["budget"]["scope_id"] in c["scope_ids"] for c in calls)
    assert not (one / "usage.db").exists() and not (two / "usage.db").exists()
    for child in (one, two):
        report = json.loads((child / "report.json").read_text())
        assert (
            sum(
                v[k]["calls"]
                for v in report["variants"].values()
                for k in ("generation", "evaluation")
            )
            == count
        )
        await runner.execute(
            child, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == 2 * count


@pytest.mark.asyncio
async def test_parent_attempt_cap_survives_new_child_and_resume(experiment):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(
        original.parent / "pilot", max_calls=1, evidence="simulated"
    )
    for name in ("one", "two"):
        child = make_child(original.parent, parent, name)
        for resume in (False, True):
            with pytest.raises(BudgetLimitError):
                await runner.execute(
                    child, credentials=CREDENTIALS, resume=resume, evidence="simulated"
                )
    assert len(calls) == pilot.pilot_status(parent)["budget"]["calls"] == 1


@pytest.mark.asyncio
async def test_parent_money_cap_cannot_be_bypassed_by_child_allowance(experiment):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(
        original.parent / "pilot", limit_usd="0.01", evidence="simulated"
    )
    for name in ("one", "two"):
        child = make_child(original.parent, parent, name)
        with pytest.raises(BudgetLimitError):
            await runner.execute(child, credentials=CREDENTIALS, evidence="simulated")
    assert not calls
    assert pilot.pilot_status(parent)["budget"]["calls"] == 0


@pytest.mark.asyncio
async def test_uncertain_dispatch_remains_reserved_for_other_books(experiment):
    original, calls, fault = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(
        original.parent / "pilot", max_calls=1, evidence="simulated"
    )
    one = make_child(original.parent, parent, "one")
    two = make_child(original.parent, parent, "two")

    async def cancelled(_):
        raise asyncio.CancelledError()

    fault["dispatch"] = cancelled
    with pytest.raises(asyncio.CancelledError):
        await runner.execute(one, credentials=CREDENTIALS, evidence="simulated")
    before = pilot.pilot_status(parent)["budget"]
    assert before["uncertain_calls"] == 1 and before["reserved_usd"] > 0
    fault["dispatch"] = None
    with pytest.raises(BudgetLimitError):
        await runner.execute(two, credentials=CREDENTIALS, evidence="simulated")
    assert pilot.pilot_status(parent)["budget"] == before
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    ["missing_ledger", "replaced_ledger", "manifest", "missing_state", "copied_child"],
)
async def test_parent_and_child_identity_fail_closed_before_redispatch(
    experiment, damage
):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    child = make_child(original.parent, parent, "one")
    await runner.execute(child, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    if damage.endswith("ledger"):
        (parent / "usage.db").unlink()
        if damage == "replaced_ledger":
            TokenUsageStore(parent / "usage.db")
    elif damage == "manifest":
        manifest = json.loads((parent / "pilot.json").read_text())
        manifest["budget"]["limit_usd"] = "100"
        runner._replace(parent / "pilot.json", manifest)
    elif damage == "missing_state":
        (child / "state.json").unlink()
    else:
        copy = original.parent / "copy"
        shutil.copytree(child, copy)
        child = copy
    with pytest.raises(ValueError):
        await runner.execute(
            child, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


@pytest.mark.asyncio
async def test_paid_execution_requires_parent_and_never_uses_simulated_allowance(
    experiment,
):
    original, calls, _ = experiment
    with pytest.raises(ValueError, match="pilot"):
        await runner.execute(original, credentials=CREDENTIALS)
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    child = make_child(original.parent, parent, "one")
    with pytest.raises(ValueError, match="evidence"):
        await runner.execute(child, credentials=CREDENTIALS)
    assert not calls


def test_separate_connections_reserve_all_three_scopes_atomically(tmp_path):
    pilot = pilot_module()
    parent = pilot.create_pilot(tmp_path / "pilot", limit_usd="1", evidence="simulated")
    children = [make_child(tmp_path, parent, name) for name in ("one", "two")]
    parent_scope = pilot.pilot_status(parent)["budget"]["scope_id"]
    experiment_scopes = [
        "experiment:" + json.loads((c / "plan.json").read_text())["plan_sha256"][:24]
        for c in children
    ]

    def reserve(i):
        budget = SpendBudget(TokenUsageStore(parent / "usage.db"))
        job = "job:concurrent:" + str(i % 2)
        budget.configure(job, limit_usd="5", max_calls=300)
        try:
            budget.reserve(
                (job, experiment_scopes[i % 2], parent_scope), upper_usd="0.11"
            )
            return True
        except BudgetLimitError:
            return False

    with ThreadPoolExecutor(max_workers=8) as workers:
        assert sum(workers.map(reserve, range(30))) == 9
    status = pilot.pilot_status(parent)
    assert status["budget"]["reserved_usd"] == pytest.approx(0.99)
    assert sum(child["budget"]["calls"] for child in status["experiments"]) == 9


def test_parent_creation_is_exclusive_and_cli_reports_shared_totals(tmp_path, capsys):
    from scripts.editorial_benchmark import main

    directory = tmp_path / "pilot"
    main(["pilot-init", "--output", str(directory)])
    assert json.loads(capsys.readouterr().out)["budget"]["limit_usd"] == 5
    main(["pilot-status", str(directory)])
    assert json.loads(capsys.readouterr().out)["budget"]["calls"] == 0
    with pytest.raises(ValueError):
        pilot_module().create_pilot(directory)
    assert directory.stat().st_mode & 0o777 == 0o700
    assert (directory / "usage.db").stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_same_units_with_different_child_caps_have_independent_judge_jobs(
    experiment,
):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    for name, maximum in (("one", 100), ("two", 101)):
        child = make_child(original.parent, parent, name, max_calls=maximum)
        result = await runner.execute(
            child, credentials=CREDENTIALS, evidence="simulated"
        )
        assert result["status"] == "completed"
    judge_scopes = [
        next(s for s in c["scope_ids"] if s.startswith("job:"))
        for c in calls
        if c["judge"]
    ]
    assert len(judge_scopes) == len(set(judge_scopes)) == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("allowance, expected_calls", [("5", 1), ("20", 25)])
async def test_http_retry_is_an_additional_parent_reservation(
    experiment, monkeypatch, allowance, expected_calls
):
    original, calls, fault = experiment
    pilot = pilot_module()
    # A synthetic larger allowance isolates retry accounting. Paid default stays USD 5.
    parent = pilot.create_pilot(
        original.parent / "pilot", evidence="simulated", limit_usd=allowance
    )
    child = make_child(original.parent, parent, "one", limit_usd=allowance)
    monkeypatch.setattr(
        "src.core.adapters.retry_manager.RetryManager.delay_for_attempt",
        lambda *args: 0,
    )

    async def timeout_once(call):
        if len(calls) == 1:
            raise httpx.ReadTimeout("Synthetic ambiguous read timeout")

    fault["dispatch"] = timeout_once
    if allowance == "5":
        with pytest.raises(BudgetLimitError):
            await runner.execute(child, credentials=CREDENTIALS, evidence="simulated")
        result = json.loads((child / "state.json").read_text())
        assert result["status"] == "needs_review"
        assert result["stop_reason"] == "budget_exhausted"
    else:
        result = await runner.execute(
            child, credentials=CREDENTIALS, evidence="simulated"
        )
        assert result["status"] == "completed"
    status = pilot.pilot_status(parent)
    assert status["budget"]["calls"] == len(calls) == expected_calls
    assert status["budget"]["uncertain_calls"] == 1
    assert status["budget"]["reserved_usd"] > 0
    count = len(calls)
    if allowance == "5":
        with pytest.raises(BudgetLimitError):
            await runner.execute(
                child, credentials=CREDENTIALS, resume=True, evidence="simulated"
            )
    else:
        await runner.execute(
            child, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count
    assert pilot.pilot_status(parent)["budget"] == status["budget"]


@pytest.mark.asyncio
async def test_two_live_children_cannot_dispatch_beyond_the_shared_cap(experiment):
    original, calls, fault = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(
        original.parent / "pilot", max_calls=1, evidence="simulated"
    )
    one = make_child(original.parent, parent, "one")
    two = make_child(original.parent, parent, "two")
    started, release = asyncio.Event(), asyncio.Event()

    async def hold_dispatch(_):
        started.set()
        await release.wait()

    fault["dispatch"] = hold_dispatch
    first = asyncio.create_task(
        runner.execute(one, credentials=CREDENTIALS, evidence="simulated")
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        with pytest.raises(BudgetLimitError):
            await runner.execute(two, credentials=CREDENTIALS, evidence="simulated")
        assert len(calls) == 1
    finally:
        release.set()
        with pytest.raises(BudgetLimitError):
            await asyncio.wait_for(first, timeout=3)
    assert pilot.pilot_status(parent)["budget"]["calls"] == 1


@pytest.mark.asyncio
async def test_lost_initial_state_cannot_reset_started_child_in_shared_ledger(
    experiment,
):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    child = make_child(original.parent, parent, "one")
    plan = json.loads((child / "plan.json").read_text())
    pilot.start_experiment(child, plan, None, evidence="simulated")
    with pytest.raises(ValueError, match="state is missing"):
        await runner.execute(
            child, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert not calls


@pytest.mark.asyncio
async def test_old_ledger_backup_cannot_reset_retained_spend(experiment):
    original, calls, _ = experiment
    pilot = pilot_module()
    parent = pilot.create_pilot(original.parent / "pilot", evidence="simulated")
    child = make_child(original.parent, parent, "one")
    backup = original.parent / "before.db"
    shutil.copy2(parent / "usage.db", backup)
    await runner.execute(child, credentials=CREDENTIALS, evidence="simulated")
    count = len(calls)
    shutil.copy2(backup, parent / "usage.db")
    with pytest.raises(ValueError, match="predates"):
        await runner.execute(
            child, credentials=CREDENTIALS, resume=True, evidence="simulated"
        )
    assert len(calls) == count


def test_process_filter_is_applied_before_limit_and_combines_with_job_filter(tmp_path):
    store = TokenUsageStore(tmp_path / "usage.db")
    store.record_call(
        provider="openai",
        model="gpt-6-luna",
        prompt="fixture",
        context={"process_id": "first", "translation_id": "target"},
    )
    store.record_call(
        provider="openai",
        model="gpt-6-luna",
        prompt="fixture",
        context={"process_id": "second", "translation_id": "target"},
    )
    assert len(store.events(limit=1, process_id="first")) == 1
    assert store.events(limit=1, process_id="first")[0]["process_id"] == "first"
    assert len(store.events(process_id="second", translation_id="target")) == 1
    assert store.events(process_id="first", translation_id="missing") == []


def test_status_totals_share_one_read_snapshot_during_a_concurrent_reservation(
    tmp_path, monkeypatch
):
    pilot = pilot_module()
    parent = pilot.create_pilot(tmp_path / "pilot", evidence="simulated")
    for name in ("one", "two"):
        make_child(tmp_path, parent, name)
    initial = pilot.pilot_status(parent)
    scopes = [initial["experiments"][0]["scope_id"], initial["budget"]["scope_id"]]
    writer = SpendBudget(TokenUsageStore(parent / "usage.db"))
    # WAL permits a genuine independent commit while the reader holds a snapshot.
    with writer.store._connect() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    original_usage = SpendBudget._usage
    injected = False

    def read_then_commit(conn, scope):
        nonlocal injected
        result = original_usage(conn, scope)
        if scope == scopes[0] and not injected:
            injected = True
            writer.reserve(scopes, upper_usd="0.11")
        return result

    monkeypatch.setattr(SpendBudget, "_usage", staticmethod(read_then_commit))
    snapshot = pilot.pilot_status(parent)
    assert injected
    assert snapshot["budget"]["calls"] == 0
    assert sum(c["budget"]["calls"] for c in snapshot["experiments"]) == 0
    after = pilot.pilot_status(parent)
    assert after["budget"]["calls"] == 1
    assert sum(c["budget"]["calls"] for c in after["experiments"]) == 1
