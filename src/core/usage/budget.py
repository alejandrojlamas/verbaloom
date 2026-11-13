"""Atomic gross-API spend reservations in the existing usage database.

Uncertain/in-flight requests keep their full allowance across process restarts.
Reservations are not provider idempotency keys and never expire automatically.
"""

import json
import time
import uuid
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation

from src.core.llm.exceptions import ProviderActionRequiredError

NANODOLLARS = Decimal(1_000_000_000)


class BudgetLimitError(ProviderActionRequiredError):
    def __init__(self, message, *, reason="budget_exhausted"):
        super().__init__(message, reason=reason, provider="budget")


def _money(value, *, ceiling=True):
    try:
        number = Decimal(str(value))
        if isinstance(value, bool) or not number.is_finite() or number < 0:
            raise ValueError("Budget amounts must be finite, non-negative USD")
        result = int(
            (number * NANODOLLARS).to_integral_value(
                rounding=ROUND_CEILING if ceiling else ROUND_FLOOR
            )
        )
        if result > 2**62:
            raise ValueError("Budget amount exceeds storage range")
        return result
    except InvalidOperation as exc:
        raise ValueError("Invalid budget amount") from exc


class SpendBudget:
    def __init__(self, store):
        self.store = store

    @staticmethod
    def ensure_schema(conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS usage_budget_scopes (
            scope_id TEXT PRIMARY KEY, limit_nusd INTEGER NOT NULL,
            max_calls INTEGER NOT NULL, created_at REAL NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS usage_budget_reservations (
            reservation_id TEXT PRIMARY KEY, upper_nusd INTEGER NOT NULL,
            actual_nusd INTEGER, state TEXT NOT NULL, created_at REAL NOT NULL,
            settled_at REAL, event_id INTEGER, metadata TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS usage_budget_memberships (
            scope_id TEXT NOT NULL, reservation_id TEXT NOT NULL,
            PRIMARY KEY (scope_id, reservation_id))""")

    def configure(self, scope_id, *, limit_usd, max_calls):
        if not isinstance(scope_id, str) or not scope_id.strip():
            raise ValueError("Budget scope needs a stable identifier")
        limit = _money(limit_usd, ceiling=False)
        if type(max_calls) is not int or not 1 <= max_calls <= 1_000_000:
            raise ValueError("A finite positive call limit is required")
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM usage_budget_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            if existing:
                if (existing["limit_nusd"], existing["max_calls"]) != (
                    limit,
                    max_calls,
                ):
                    raise ValueError(
                        "Saved budget differs; resume cannot reset its limits"
                    )
                return
            conn.execute(
                "INSERT INTO usage_budget_scopes VALUES (?, ?, ?, ?)",
                (scope_id, limit, max_calls, time.time()),
            )

    @staticmethod
    def _usage(conn, scope_id):
        return conn.execute(
            """SELECT COUNT(*) AS calls,
            COALESCE(SUM(CASE WHEN state IN ('settled', 'overrun')
                THEN actual_nusd ELSE upper_nusd END), 0) AS committed,
            COALESCE(SUM(CASE WHEN state IN ('settled', 'overrun')
                THEN actual_nusd ELSE 0 END), 0) AS settled,
            COALESCE(SUM(CASE WHEN state IN ('reserved', 'uncertain')
                THEN upper_nusd ELSE 0 END), 0) AS reserved,
            COALESCE(SUM(state='uncertain'), 0) AS uncertain_calls,
            COALESCE(SUM(state='overrun'), 0) AS overruns
            FROM usage_budget_reservations r JOIN usage_budget_memberships m
              ON r.reservation_id=m.reservation_id WHERE m.scope_id=?""",
            (scope_id,),
        ).fetchone()

    def reserve(self, scope_ids, *, upper_usd, metadata=None):
        if isinstance(scope_ids, str) or not scope_ids:
            raise ValueError("At least one configured budget scope is required")
        scopes = sorted(set(scope_ids))
        upper = _money(upper_usd)
        if upper <= 0:
            raise ValueError("A positive request bound is required")
        reservation_id = uuid.uuid4().hex
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for scope in scopes:
                config = conn.execute(
                    "SELECT * FROM usage_budget_scopes WHERE scope_id=?", (scope,)
                ).fetchone()
                if config is None:
                    raise BudgetLimitError(
                        "Budget scope is not configured", reason="budget_configuration"
                    )
                usage = self._usage(conn, scope)
                if usage["overruns"]:
                    raise BudgetLimitError(
                        "Prior reservation overrun requires reconciliation",
                        reason="budget_overrun",
                    )
                if usage["calls"] >= config["max_calls"]:
                    raise BudgetLimitError(
                        "HTTP attempt budget exhausted", reason="call_budget_exhausted"
                    )
                if usage["committed"] + upper > config["limit_nusd"]:
                    raise BudgetLimitError(
                        "Insufficient gross API budget for another request"
                    )
            conn.execute(
                """INSERT INTO usage_budget_reservations
                (reservation_id, upper_nusd, state, created_at, metadata)
                VALUES (?, ?, 'reserved', ?, ?)""",
                (
                    reservation_id,
                    upper,
                    time.time(),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
            conn.executemany(
                "INSERT INTO usage_budget_memberships VALUES (?, ?)",
                [(scope, reservation_id) for scope in scopes],
            )
        return reservation_id

    @staticmethod
    def settle_in_transaction(conn, reservation_id, *, actual_usd, event_id=None):
        row = conn.execute(
            "SELECT * FROM usage_budget_reservations WHERE reservation_id=?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown budget reservation")
        if (
            event_id is not None
            and row["event_id"] is not None
            and event_id != row["event_id"]
        ):
            raise ValueError("Reservation already has a usage event")
        actual = None if actual_usd is None else _money(actual_usd)
        if row["state"] in {"settled", "overrun"}:
            if actual != row["actual_nusd"] or (
                event_id is not None and event_id != row["event_id"]
            ):
                raise ValueError("Conflicting budget settlement")
            return
        state = (
            "uncertain"
            if actual is None
            else ("overrun" if actual > row["upper_nusd"] else "settled")
        )
        conn.execute(
            """UPDATE usage_budget_reservations
            SET actual_nusd=?, state=?, settled_at=?, event_id=COALESCE(?, event_id) WHERE reservation_id=?""",
            (actual, state, time.time(), event_id, reservation_id),
        )

    def settle(self, reservation_id, *, actual_usd):
        """Explicit reconciliation only; normal settlement shares the event transaction."""
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.settle_in_transaction(conn, reservation_id, actual_usd=actual_usd)

    def snapshot(self, scope_id):
        with self.store._lock, self.store._connect() as conn:
            conn.execute("BEGIN")
            return self.snapshot_in_transaction(conn, scope_id)

    @staticmethod
    def snapshot_in_transaction(conn, scope_id):
        """Read within the caller's transaction so related scopes stay consistent."""
        config = conn.execute(
            "SELECT * FROM usage_budget_scopes WHERE scope_id=?", (scope_id,)
        ).fetchone()
        if config is None:
            raise ValueError("Unknown budget scope")
        usage = SpendBudget._usage(conn, scope_id)
        return {
            "scope_id": scope_id,
            "limit_usd": config["limit_nusd"] / 1e9,
            "max_calls": config["max_calls"],
            "calls": usage["calls"],
            "settled_usd": usage["settled"] / 1e9,
            "reserved_usd": usage["reserved"] / 1e9,
            "available_usd": max(0, config["limit_nusd"] - usage["committed"]) / 1e9,
            "uncertain_calls": usage["uncertain_calls"],
            "overruns": usage["overruns"],
        }
