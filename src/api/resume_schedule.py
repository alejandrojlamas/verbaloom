"""Durable wall-clock schedules for self-healing translation workers.

Provider pauses and recovery backoffs are lifecycle state, not an in-memory
sleep.  The canonical deadline lives in the persisted job config so a laptop
sleep, process restart, or browser refresh cannot lose or extend it.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

_SCHEDULE_KEYS = (
    "_scheduled_resume_at_epoch",
    "_scheduled_resume_at_utc",
    "_scheduled_resume_at_local",
    "_scheduled_resume_timezone",
    "_scheduled_resume_reason",
    "_scheduled_resume_status",
    # Legacy DeepSeek-only keys are read and cleared during migration.
    "_pricing_pause_until_utc",
    "_pricing_pause_timezone",
)


@dataclass(frozen=True)
class ResumeSchedule:
    resume_at_epoch: float
    reason: str
    status: str
    resume_at_utc: str
    resume_at_local: str = ""
    display_timezone: str = ""


def _parse_utc_epoch(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def build_resume_schedule(
    *,
    delay_seconds: float,
    reason: str,
    status: str,
    resume_at_utc: str = "",
    resume_at_local: str = "",
    display_timezone: str = "",
    now: float | None = None,
) -> ResumeSchedule:
    """Create an absolute deadline from provider or recovery information."""
    current = float(time.time() if now is None else now)
    try:
        delay = max(0.0, float(delay_seconds))
    except (TypeError, ValueError):
        delay = 0.0
    deadline = current + delay
    parsed_deadline = _parse_utc_epoch(resume_at_utc)
    if parsed_deadline is not None:
        # Provider retry delays can include a small boundary-safety margin. Keep
        # that margin while still treating the provider timestamp as durable.
        deadline = max(deadline, parsed_deadline)
    canonical_utc = datetime.fromtimestamp(
        deadline,
        tz=timezone.utc,
    ).isoformat()
    return ResumeSchedule(
        resume_at_epoch=deadline,
        reason=str(reason or "provider_wait"),
        status=str(status or "provider_wait"),
        resume_at_utc=canonical_utc,
        resume_at_local=str(resume_at_local or ""),
        display_timezone=str(display_timezone or ""),
    )


def resume_schedule_from_config(config: Mapping[str, Any] | None) -> ResumeSchedule | None:
    """Load the canonical schedule, migrating a legacy pricing deadline."""
    values = config or {}
    try:
        deadline = float(values.get("_scheduled_resume_at_epoch"))
    except (TypeError, ValueError):
        deadline = None
    canonical_utc = str(values.get("_scheduled_resume_at_utc") or "").strip()
    if deadline is None:
        deadline = _parse_utc_epoch(canonical_utc)

    legacy_utc = str(values.get("_pricing_pause_until_utc") or "").strip()
    if deadline is None:
        deadline = _parse_utc_epoch(legacy_utc)
    if deadline is None:
        return None

    pricing_legacy = bool(legacy_utc) and not values.get("_scheduled_resume_reason")
    return ResumeSchedule(
        resume_at_epoch=deadline,
        reason=str(
            values.get("_scheduled_resume_reason")
            or ("deepseek_peak_pricing" if pricing_legacy else "provider_wait")
        ),
        status=str(
            values.get("_scheduled_resume_status")
            or ("pricing_wait" if pricing_legacy else "provider_wait")
        ),
        resume_at_utc=canonical_utc or legacy_utc or datetime.fromtimestamp(
            deadline,
            tz=timezone.utc,
        ).isoformat(),
        resume_at_local=str(values.get("_scheduled_resume_at_local") or ""),
        display_timezone=str(
            values.get("_scheduled_resume_timezone")
            or values.get("_pricing_pause_timezone")
            or ""
        ),
    )


def with_resume_schedule(
    config: Mapping[str, Any] | None,
    schedule: ResumeSchedule,
) -> dict[str, Any]:
    """Return a config carrying one canonical durable resume schedule."""
    updated = clear_resume_schedule(config)
    updated.update({
        "_scheduled_resume_at_epoch": schedule.resume_at_epoch,
        "_scheduled_resume_at_utc": schedule.resume_at_utc,
        "_scheduled_resume_at_local": schedule.resume_at_local,
        "_scheduled_resume_timezone": schedule.display_timezone,
        "_scheduled_resume_reason": schedule.reason,
        "_scheduled_resume_status": schedule.status,
    })
    return updated


def clear_resume_schedule(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Remove canonical and legacy schedule metadata from a copied config."""
    updated = dict(config or {})
    for key in _SCHEDULE_KEYS:
        updated.pop(key, None)
    return updated


async def wait_for_resume_schedule(
    state_manager: Any,
    translation_id: str,
    schedule: ResumeSchedule,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[Any]] | None = None,
    poll_seconds: float = 1.0,
) -> bool:
    """Wait against wall time while honoring deletion or a manual interrupt."""
    now = clock or time.time
    sleeper = sleep or asyncio.sleep
    poll = max(0.05, float(poll_seconds))
    while True:
        if not state_manager.exists(translation_id):
            return False
        if state_manager.get_translation_field(translation_id, "interrupted"):
            return False
        remaining = schedule.resume_at_epoch - float(now())
        if remaining <= 0:
            return True
        await sleeper(min(poll, remaining))


def wait_for_resume_schedule_sync(
    state_manager: Any,
    translation_id: str,
    schedule: ResumeSchedule,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Any] | None = None,
    poll_seconds: float = 1.0,
) -> bool:
    """Synchronous counterpart used by daemon recovery handoff threads."""
    now = clock or time.time
    sleeper = sleep or time.sleep
    poll = max(0.05, float(poll_seconds))
    while True:
        if not state_manager.exists(translation_id):
            return False
        if state_manager.get_translation_field(translation_id, "interrupted"):
            return False
        remaining = schedule.resume_at_epoch - float(now())
        if remaining <= 0:
            return True
        sleeper(min(poll, remaining))
