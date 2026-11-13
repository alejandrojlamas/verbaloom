"""Frozen pilot identity and child registration; accounting stays in SpendBudget."""

import json
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path

from scripts.editorial_benchmark import private_directory, write_private
from src.core.editorial_benchmark import _digest
from src.core.usage.budget import SpendBudget, _money
from src.core.usage.store import TokenUsageStore

SCHEMA = "verbaloom.editorial-pilot/v1"


def create_pilot(
    directory, *, limit_usd="5", max_calls=1000, evidence="provider_response"
):
    if evidence not in {"provider_response", "simulated"}:
        raise ValueError("Unknown pilot evidence type")
    if not 0 < _money(limit_usd, ceiling=False) <= 1_000_000 * 10**9:
        raise ValueError("Invalid pilot gross USD budget")
    if type(max_calls) is not int or not 1 <= max_calls <= 1000:
        raise ValueError("A pilot requires 1 to 1000 maximum HTTP attempts")
    directory = private_directory(directory)
    if any(directory.iterdir()):
        raise ValueError("Use a new empty pilot directory; never reset an allowance")
    manifest = {
        "schema": SCHEMA,
        "scope_id": "pilot:" + uuid.uuid4().hex,
        "budget": {"limit_usd": str(limit_usd), "max_calls": max_calls},
        "evidence": evidence,
        "created_at": time.time(),
    }
    manifest["pilot_sha256"] = _digest(manifest)
    # Exclusive publication wins ownership before initializing any accounting.
    write_private(directory / "pilot.json", manifest)
    store = TokenUsageStore(directory / "usage.db")
    store.db_path.chmod(0o600)
    SpendBudget(store).configure(manifest["scope_id"], **manifest["budget"])
    with store._lock, store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE editorial_pilot_identity (manifest_sha256 TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO editorial_pilot_identity VALUES (?)",
            (manifest["pilot_sha256"],),
        )
        conn.execute("""CREATE TABLE editorial_pilot_experiments (
            plan_sha256 TEXT PRIMARY KEY, directory TEXT NOT NULL UNIQUE,
            scope_id TEXT NOT NULL UNIQUE, started INTEGER NOT NULL DEFAULT 0)""")
    return directory


def _open(directory, *, evidence=None):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError("Pilot directory is missing")
    private_directory(directory)
    for name in ("pilot.json", "usage.db"):
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "Pilot manifest or ledger is missing; never recreate spent allowances"
            )
    manifest = json.loads((directory / "pilot.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("pilot_sha256") != _digest(
        {k: v for k, v in manifest.items() if k != "pilot_sha256"}
    ):
        raise ValueError("Frozen pilot manifest changed")
    if evidence is not None and evidence != manifest["evidence"]:
        raise ValueError("Simulated and observed evidence cannot share a pilot")
    # Check identity in an existing database before TokenUsageStore can initialize it.
    try:
        with closing(
            sqlite3.connect((directory / "usage.db").as_uri() + "?mode=rw", uri=True)
        ) as conn:
            identity = conn.execute(
                "SELECT manifest_sha256 FROM editorial_pilot_identity"
            ).fetchall()
            scope = conn.execute(
                "SELECT scope_id FROM usage_budget_scopes WHERE scope_id=?",
                (manifest["scope_id"],),
            ).fetchone()
        if identity != [(manifest["pilot_sha256"],)] or scope is None:
            raise ValueError("Pilot ledger identity or saved allowance differs")
    except sqlite3.Error as exc:
        raise ValueError("Pilot ledger is missing or invalid; do not reset it") from exc
    store = TokenUsageStore(directory / "usage.db")
    SpendBudget(store).configure(manifest["scope_id"], **manifest["budget"])
    return directory, manifest, store


def pilot_binding(directory):
    directory, manifest, _ = _open(directory)
    return {
        "directory": str(directory),
        "pilot_sha256": manifest["pilot_sha256"],
        "scope_id": manifest["scope_id"],
    }


def _bound(plan, *, evidence=None):
    binding = plan["pilot"]
    directory, manifest, store = _open(binding["directory"], evidence=evidence)
    expected = {
        "directory": str(directory),
        "pilot_sha256": manifest["pilot_sha256"],
        "scope_id": manifest["scope_id"],
    }
    if binding != expected:
        raise ValueError("Experiment pilot binding changed")
    return manifest, store


def register_experiment(directory, plan):
    _, store = _bound(plan)
    scope = "experiment:" + plan["plan_sha256"][:24]
    SpendBudget(store).configure(scope, **plan["budget"])
    try:
        with store._lock, store._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO editorial_pilot_experiments VALUES (?, ?, ?, 0)",
                (plan["plan_sha256"], str(directory), scope),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError(
            "Experiment already registered; resume its original directory"
        ) from exc


def start_experiment(directory, plan, state, *, evidence):
    manifest, store = _bound(plan, evidence=evidence)
    scope = "experiment:" + plan["plan_sha256"][:24]
    budget = SpendBudget(store)
    snapshot = budget.snapshot(scope)  # Missing scopes are never recreated here.
    budget.configure(scope, **plan["budget"])
    if state and snapshot["calls"] < state.get("budget", {}).get("calls", 0):
        raise ValueError("Pilot ledger predates retained experiment spend")
    with store._lock, store._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        saved = conn.execute(
            "SELECT * FROM editorial_pilot_experiments WHERE plan_sha256=?",
            (plan["plan_sha256"],),
        ).fetchone()
        if (
            saved is None
            or saved["directory"] != str(directory)
            or saved["scope_id"] != scope
        ):
            raise ValueError("Experiment is not registered at this pilot directory")
        if saved["started"] and state is None:
            raise ValueError("Started experiment state is missing; do not reset it")
        if not saved["started"] and state is not None:
            raise ValueError("Pilot ledger does not match retained experiment state")
        conn.execute(
            "UPDATE editorial_pilot_experiments SET started=1 WHERE plan_sha256=?",
            (plan["plan_sha256"],),
        )
    return store, manifest["scope_id"]


def pilot_status(directory):
    directory, manifest, store = _open(directory)
    budget = SpendBudget(store)
    with store._lock, store._connect() as conn:
        conn.execute("BEGIN")
        experiments = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM editorial_pilot_experiments ORDER BY directory"
            )
        ]
        for child in experiments:
            child["budget"] = budget.snapshot_in_transaction(conn, child["scope_id"])
        parent_budget = budget.snapshot_in_transaction(conn, manifest["scope_id"])
    return {
        "directory": str(directory),
        "evidence": manifest["evidence"],
        "budget": parent_budget,
        "experiments": experiments,
    }
