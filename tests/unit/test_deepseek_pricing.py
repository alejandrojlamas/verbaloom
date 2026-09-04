from datetime import datetime, timezone

import pytest

from src.core.deepseek_pricing import (
    DISPLAY_TIMEZONE,
    get_deepseek_pricing_status,
    is_official_deepseek_endpoint,
)


UTC = timezone.utc


@pytest.mark.parametrize(
    ("now", "disabled", "next_available_hour"),
    [
        (datetime(2026, 9, 3, 0, 59, tzinfo=UTC), False, None),
        (datetime(2026, 9, 3, 1, 0, tzinfo=UTC), True, 4),
        (datetime(2026, 9, 3, 3, 59, 59, tzinfo=UTC), True, 4),
        (datetime(2026, 9, 3, 4, 0, tzinfo=UTC), False, None),
        (datetime(2026, 9, 3, 6, 0, tzinfo=UTC), True, 10),
        (datetime(2026, 9, 3, 9, 59, 59, tzinfo=UTC), True, 10),
        (datetime(2026, 9, 3, 10, 0, tzinfo=UTC), False, None),
    ],
)
def test_official_weekday_boundaries(now, disabled, next_available_hour, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_OFF_PEAK_ONLY", raising=False)

    status = get_deepseek_pricing_status(now)

    assert status.disabled is disabled
    assert status.available is not disabled
    assert status.display_timezone == DISPLAY_TIMEZONE
    if next_available_hour is None:
        assert status.next_available_at_utc is None
        assert status.seconds_until_available == 0
    else:
        resumed = datetime.fromisoformat(status.next_available_at_utc)
        assert resumed.hour == next_available_hour
        assert resumed.tzinfo is not None
        assert status.seconds_until_available > 0


def test_cdmx_conversion_is_explicit_and_crosses_local_day(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_OFF_PEAK_ONLY", raising=False)

    evening_peak = get_deepseek_pricing_status(
        datetime(2026, 9, 3, 1, 30, tzinfo=UTC)
    )
    midnight_peak = get_deepseek_pricing_status(
        datetime(2026, 9, 4, 6, 30, tzinfo=UTC)
    )

    # Thursday 01:30 UTC is Wednesday 19:30 in Mexico City; resumes at 22:00.
    assert evening_peak.now_local.startswith("2026-09-02T19:30:00-06:00")
    assert evening_peak.next_available_at_local.startswith("2026-09-02T22:00:00-06:00")
    # Friday 06:30 UTC is Friday 00:30 in Mexico City; resumes at 04:00.
    assert midnight_peak.now_local.startswith("2026-09-04T00:30:00-06:00")
    assert midnight_peak.next_available_at_local.startswith("2026-09-04T04:00:00-06:00")


def test_weekend_has_no_peak_and_next_peak_is_sunday_evening_cdmx(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_OFF_PEAK_ONLY", raising=False)

    status = get_deepseek_pricing_status(
        datetime(2026, 9, 5, 18, 0, tzinfo=UTC)
    )

    assert status.pricing_tier == "off_peak"
    assert status.disabled is False
    assert status.next_peak_at_utc.startswith("2026-09-07T01:00:00+00:00")
    assert status.next_peak_at_local.startswith("2026-09-06T19:00:00-06:00")


def test_guard_can_be_explicitly_disabled_without_hiding_peak_tier(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_OFF_PEAK_ONLY", "false")

    status = get_deepseek_pricing_status(
        datetime(2026, 9, 3, 1, 30, tzinfo=UTC)
    )

    assert status.enabled is False
    assert status.pricing_tier == "peak"
    assert status.disabled is False
    assert status.available is True


@pytest.mark.parametrize(
    ("endpoint", "expected"),
    [
        ("https://api.deepseek.com/chat/completions", True),
        ("https://api.deepseek.com/v1/chat/completions", True),
        ("https://API.DEEPSEEK.COM/chat/completions", True),
        ("https://api.deepseek.test/chat/completions", False),
        ("https://gateway.example.com/deepseek", False),
        ("http://127.0.0.1:8080/v1/chat/completions", False),
    ],
)
def test_only_official_endpoint_is_restricted(endpoint, expected):
    assert is_official_deepseek_endpoint(endpoint) is expected
