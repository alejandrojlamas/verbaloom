"""Durable shared budgets; no network or production database access."""

import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from src.core.usage.budget import BudgetLimitError, SpendBudget
from src.core.usage.store import TokenUsageStore


@pytest.mark.parametrize(
    "module",
    [
        "src.core.usage.store",
        "src.core.usage.budget",
        "src.core.usage.tracking_provider",
        "src.core.usage.request_budget",
    ],
)
def test_cold_import_does_not_require_pytest_preloading(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def budget(tmp_path):
    return SpendBudget(TokenUsageStore(tmp_path / "usage.db"))


def test_job_and_experiment_are_reserved_atomically(budget):
    budget.configure("job", limit_usd="1", max_calls=10)
    budget.configure("experiment", limit_usd="0.5", max_calls=10)
    budget.reserve(["job", "experiment"], upper_usd="0.4")
    with pytest.raises(BudgetLimitError):
        budget.reserve(["job", "experiment"], upper_usd="0.2")
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(0.4)
    assert budget.snapshot("job")["calls"] == 1


def test_concurrent_connections_cannot_overspend(budget):
    budget.configure("shared", limit_usd="1", max_calls=100)

    def reserve(_):
        other = SpendBudget(TokenUsageStore(budget.store.db_path))
        try:
            other.reserve(["shared"], upper_usd="0.11")
            return True
        except BudgetLimitError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(30))) == 9
    assert budget.snapshot("shared")["reserved_usd"] == pytest.approx(0.99)


def test_pending_and_unknown_cost_survive_restart(budget):
    budget.configure("job", limit_usd="1", max_calls=10)
    first = budget.reserve(["job"], upper_usd="0.4")
    budget.reserve(["job"], upper_usd="0.5")  # worker dies before recording
    budget.settle(first, actual_usd=None)
    restored = SpendBudget(TokenUsageStore(budget.store.db_path))
    restored.configure("job", limit_usd="1", max_calls=10)
    assert restored.snapshot("job")["reserved_usd"] == pytest.approx(0.9)
    assert restored.snapshot("job")["uncertain_calls"] == 1
    with pytest.raises(BudgetLimitError):
        restored.reserve(["job"], upper_usd="0.2")


def test_settlement_releases_only_observed_amount_and_is_idempotent(budget):
    budget.configure("job", limit_usd="1", max_calls=2)
    reservation = budget.reserve(["job"], upper_usd="0.9")
    budget.settle(reservation, actual_usd="0.1")
    budget.settle(reservation, actual_usd="0.1")
    with pytest.raises(ValueError):
        budget.settle(reservation, actual_usd="0.2")
    budget.reserve(["job"], upper_usd="0.9")
    with pytest.raises(BudgetLimitError):
        budget.reserve(["job"], upper_usd="0.00001")
    snapshot = budget.snapshot("job")
    assert snapshot["settled_usd"] == pytest.approx(0.1)
    assert snapshot["calls"] == 2


def test_configuration_cannot_reset_spend_or_change_on_resume(budget):
    budget.configure("job", limit_usd="1", max_calls=2)
    budget.reserve(["job"], upper_usd="0.5")
    with pytest.raises(ValueError):
        budget.configure("job", limit_usd="5", max_calls=20)
    assert budget.snapshot("job")["calls"] == 1


def test_overrun_blocks_future_calls_without_erasing_actual_cost(budget):
    budget.configure("job", limit_usd="10", max_calls=20)
    reservation = budget.reserve(["job"], upper_usd="1")
    budget.settle(reservation, actual_usd="2")
    assert budget.snapshot("job")["settled_usd"] == 2
    with pytest.raises(BudgetLimitError, match="overrun"):
        budget.reserve(["job"], upper_usd="1")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", True])
def test_invalid_money_is_rejected(budget, value):
    with pytest.raises(ValueError):
        budget.configure("job", limit_usd=value, max_calls=10)


def test_rounding_is_conservative_and_unknown_scopes_fail_closed(budget):
    budget.configure("job", limit_usd="0.000000001", max_calls=10)
    budget.reserve(["job", "job"], upper_usd=Decimal("0.0000000001"))
    with pytest.raises(BudgetLimitError):
        budget.reserve(["job"], upper_usd="0.0000000001")
    with pytest.raises(BudgetLimitError):
        budget.reserve(["missing"], upper_usd="1")


def test_usage_event_and_settlement_commit_together(budget):
    budget.configure("job", limit_usd="1", max_calls=10)
    reservation = budget.reserve(["job"], upper_usd="0.5")
    with budget.store._connect() as conn:
        conn.execute("""CREATE TRIGGER fail_usage BEFORE INSERT ON token_usage_events
                        BEGIN SELECT RAISE(ABORT, 'simulated disk failure'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        budget.store.record_call(
            provider="openai",
            model="gpt-6-luna",
            prompt="fixture",
            prompt_tokens=1000,
            completion_tokens=100,
            metadata={
                "budget_reservation_id": reservation,
                "usage_observed": True,
                "parameters": {"service_tier": "default"},
            },
        )
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(0.5)
    assert budget.store.events() == []


def test_unknown_timeout_is_not_settled_as_zero(budget):
    budget.configure("job", limit_usd="1", max_calls=10)
    reservation = budget.reserve(["job"], upper_usd="0.5")
    budget.store.record_call(
        provider="openai",
        model="gpt-6-luna",
        prompt="fixture",
        status="error",
        metadata={
            "budget_reservation_id": reservation,
            "usage_observed": False,
            "billing_uncertain": True,
            "error": "ambiguous_timeout",
        },
    )
    event = budget.store.events()[0]
    assert event["pricing_source"] == "usage_unknown"
    assert budget.snapshot("job")["reserved_usd"] == pytest.approx(0.5)
    assert budget.store.summary()["totals"]["unknown_cost_events"] == 1


def test_duplicate_uncertain_event_rolls_back_and_reconciliation_keeps_identity(budget):
    budget.configure("job", limit_usd="1", max_calls=10)
    reservation = budget.reserve(["job"], upper_usd="0.5")
    args = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "prompt": "fixture",
        "status": "error",
        "metadata": {"budget_reservation_id": reservation, "billing_uncertain": True},
    }
    event_id = budget.store.record_call(**args)
    with pytest.raises(ValueError, match="already has"):
        budget.store.record_call(**args)
    assert len(budget.store.events()) == 1
    budget.settle(reservation, actual_usd="0.01")
    with budget.store._connect() as conn:
        row = conn.execute(
            "SELECT event_id FROM usage_budget_reservations WHERE reservation_id=?",
            (reservation,),
        ).fetchone()
        assert row["event_id"] == event_id
