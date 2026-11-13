"""Finite recovery accounting over the existing durable job configuration.

These are worker handoffs, not HTTP attempts or editorial repair rounds. Those
retain their own stricter limits and monetary reservations. Pricing schedules
do not consume this budget: waiting for an authorized tariff is not a failure.
"""

from collections.abc import Mapping
from typing import Any

RECOVERY_COUNTERS = {
    "chunks": "_failure_recovery_cycle",
    "finalization": "_finalization_recovery_cycle",
    "worker": "_worker_exception_recovery_cycle",
    "rate_limit": "_rate_limit_recovery_cycle",
    "startup": "_startup_recovery_cycle",
}


def recovery_limit(value: Any, default: int) -> int:
    """Missing limits use policy defaults; invalid explicit limits fail closed."""
    if value is None:
        return default
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
        if result < 0 or str(result) != str(value).strip():
            return 0
        return result
    except (ValueError, TypeError, OverflowError):
        return 0


def recovery_count(config: Mapping[str, Any], scope: str) -> int:
    field = RECOVERY_COUNTERS[scope]
    # Older 429 waits only retained the current point's count. It is a known
    # lower bound, not proof that previous waits cost no attempts.
    value = config.get(field)
    if value is None and scope == "rate_limit":
        value = config.get("_auto_resume_stuck_count")
    return recovery_counter(value)


def recovery_counter(value: Any) -> int:
    """Read known attempts without treating corrupt state as unused budget."""
    if value is None:
        return 0
    try:
        result = int(value)
        if isinstance(value, bool) or result < 0 or str(result) != str(value).strip():
            raise ValueError("Invalid recovery counter")
        return result
    except (ValueError, TypeError, OverflowError):
        # A corrupt counter must not silently replenish the retry budget.
        return 2**63 - 1


def recovery_budget(config: Mapping[str, Any], total_units: int = 0) -> dict[str, int]:
    # Allow two handoffs per source unit across all causes, with enough room for
    # the five-step availability backoff on a short input. Freeze at reservation.
    default = max(10, 2 * recovery_limit(total_units, 0))
    return {
        "used": sum(recovery_count(config, scope) for scope in RECOVERY_COUNTERS),
        "limit": recovery_limit(config.get("max_automatic_recoveries"), default),
    }


def recovery_stop_reason(config: Mapping[str, Any], total_units: int = 0) -> str:
    if config.get("_manual_pause_requested"):
        return "manual"
    budget = recovery_budget(config, total_units)
    if budget["used"] >= budget["limit"]:
        return "automatic_recovery_budget_exhausted"
    return ""


def reserve_recovery(
    config: Mapping[str, Any],
    scope: str,
    total_units: int = 0,
) -> dict[str, Any] | None:
    """Return the next config to persist BEFORE waiting or dispatching work.

    The caller owns the existing job lifecycle lock and checkpoint transaction;
    this pure policy neither starts a worker nor creates another state machine.
    """
    if recovery_stop_reason(config, total_units):
        return None
    updated = dict(config)
    updated["max_automatic_recoveries"] = recovery_budget(config, total_units)["limit"]
    updated[RECOVERY_COUNTERS[scope]] = recovery_count(config, scope) + 1
    return updated
