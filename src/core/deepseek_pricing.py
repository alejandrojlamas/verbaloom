"""DeepSeek off-peak usage policy.

DeepSeek publishes pricing windows in UTC.  Keep the policy in UTC and only
convert timestamps for presentation so local day boundaries cannot change the
meaning of the provider schedule.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


UTC = timezone.utc
DISPLAY_TIMEZONE = "America/Mexico_City"
OFFICIAL_PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing/"
SCHEDULE_VERIFIED_AT = "2026-09-29"
SCHEDULE_ID = "deepseek-off-peak-2026-09-10"

# Official peak periods: Monday-Friday, 01:00-04:00 and 06:00-10:00 UTC.
PEAK_WEEKDAYS_UTC = frozenset(range(5))
PEAK_WINDOWS_UTC = ((time(1, 0), time(4, 0)), (time(6, 0), time(10, 0)))


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def pricing_guard_enabled() -> bool:
    """Return whether paid DeepSeek generation is restricted to off-peak."""
    return _env_flag("DEEPSEEK_OFF_PEAK_ONLY", True)


def is_official_deepseek_endpoint(endpoint: str | None) -> bool:
    """Restrict only DeepSeek's paid API, not tests or compatible gateways."""
    try:
        return (urlparse(str(endpoint or "")).hostname or "").lower() == "api.deepseek.com"
    except ValueError:
        return False


def _as_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo(DISPLAY_TIMEZONE)


def _window_on(day: date, start: time, end: time) -> tuple[datetime, datetime]:
    return (
        datetime.combine(day, start, tzinfo=UTC),
        datetime.combine(day, end, tzinfo=UTC),
    )


def _active_peak_window(now_utc: datetime) -> tuple[datetime, datetime] | None:
    if now_utc.weekday() not in PEAK_WEEKDAYS_UTC:
        return None
    for start, end in PEAK_WINDOWS_UTC:
        window = _window_on(now_utc.date(), start, end)
        if window[0] <= now_utc < window[1]:
            return window
    return None


def _next_peak_start(now_utc: datetime) -> datetime:
    for day_offset in range(8):
        candidate_day = now_utc.date() + timedelta(days=day_offset)
        if candidate_day.weekday() not in PEAK_WEEKDAYS_UTC:
            continue
        for start, end in PEAK_WINDOWS_UTC:
            window_start, _ = _window_on(candidate_day, start, end)
            if window_start > now_utc:
                return window_start
    raise RuntimeError("DeepSeek pricing schedule has no future peak window")


@dataclass(frozen=True)
class DeepSeekPricingStatus:
    provider: str
    enabled: bool
    available: bool
    disabled: bool
    pricing_tier: str
    display_timezone: str
    now_utc: str
    now_local: str
    peak_started_at_utc: str | None
    peak_started_at_local: str | None
    next_available_at_utc: str | None
    next_available_at_local: str | None
    seconds_until_available: int
    next_peak_at_utc: str
    next_peak_at_local: str
    schedule_id: str
    schedule_verified_at: str
    source_url: str

    def to_dict(self) -> dict:
        return asdict(self)


def get_deepseek_pricing_status(
    now: datetime | None = None,
    *,
    display_timezone: str = DISPLAY_TIMEZONE,
) -> DeepSeekPricingStatus:
    """Evaluate the official schedule and return API/UI-ready timestamps."""
    now_utc = _as_utc(now)
    local_zone = _zone(display_timezone)
    enabled = pricing_guard_enabled()
    active_window = _active_peak_window(now_utc)
    in_peak = active_window is not None
    disabled = enabled and in_peak
    next_available = active_window[1] if disabled and active_window else None
    next_peak = _next_peak_start(now_utc)
    seconds_until_available = (
        max(1, math.ceil((next_available - now_utc).total_seconds()))
        if next_available
        else 0
    )

    def iso(value: datetime | None, zone=UTC) -> str | None:
        return value.astimezone(zone).isoformat() if value else None

    return DeepSeekPricingStatus(
        provider="deepseek",
        enabled=enabled,
        available=not disabled,
        disabled=disabled,
        pricing_tier="peak" if in_peak else "off_peak",
        display_timezone=display_timezone,
        now_utc=iso(now_utc) or "",
        now_local=iso(now_utc, local_zone) or "",
        peak_started_at_utc=iso(active_window[0]) if active_window else None,
        peak_started_at_local=iso(active_window[0], local_zone) if active_window else None,
        next_available_at_utc=iso(next_available),
        next_available_at_local=iso(next_available, local_zone),
        seconds_until_available=seconds_until_available,
        next_peak_at_utc=iso(next_peak) or "",
        next_peak_at_local=iso(next_peak, local_zone) or "",
        schedule_id=SCHEDULE_ID,
        schedule_verified_at=SCHEDULE_VERIFIED_AT,
        source_url=OFFICIAL_PRICING_URL,
    )


def effective_estimate_tier(status: DeepSeekPricingStatus) -> str:
    """Return the tier a newly queued job is expected to pay.

    With the default guard enabled, a job submitted during peak pricing waits
    and starts in the next off-peak window. If the guard is disabled, estimates
    must follow the current provider tier.
    """

    return "off_peak" if status.enabled else status.pricing_tier


def cdmx_peak_schedule_description() -> tuple[str, ...]:
    """Human-readable reference for docs; runtime decisions never use it."""
    return (
        "Sunday 19:00-22:00",
        "Monday-Thursday 00:00-04:00 and 19:00-22:00",
        "Friday 00:00-04:00",
    )
