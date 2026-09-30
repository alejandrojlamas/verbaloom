import asyncio

from src.api.resume_schedule import (
    ResumeSchedule,
    build_resume_schedule,
    clear_resume_schedule,
    resume_schedule_from_config,
    wait_for_resume_schedule,
    with_resume_schedule,
)


class _ActiveState:
    def exists(self, _translation_id):
        return True

    def get_translation_field(self, _translation_id, field):
        assert field == "interrupted"
        return False


def test_schedule_round_trip_and_clear_migrates_all_wait_metadata():
    schedule = build_resume_schedule(
        delay_seconds=120,
        reason="deepseek_peak_pricing",
        status="pricing_wait",
        resume_at_local="22:00 CDMX",
        display_timezone="America/Mexico_City",
        now=1000,
    )

    config = with_resume_schedule({"model": "deepseek-flash"}, schedule)
    restored = resume_schedule_from_config(config)

    assert restored == schedule
    assert clear_resume_schedule(config) == {"model": "deepseek-flash"}


def test_legacy_pricing_deadline_is_restored_as_a_durable_schedule():
    restored = resume_schedule_from_config({
        "_pricing_pause_until_utc": "2026-09-30T04:00:00+00:00",
        "_pricing_pause_timezone": "America/Mexico_City",
    })

    assert restored is not None
    assert restored.status == "pricing_wait"
    assert restored.reason == "deepseek_peak_pricing"
    assert restored.display_timezone == "America/Mexico_City"


def test_wall_clock_jump_finishes_wait_after_computer_sleep():
    clock_values = iter((100.0, 500.0))
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    schedule = ResumeSchedule(
        resume_at_epoch=200.0,
        reason="deepseek_peak_pricing",
        status="pricing_wait",
        resume_at_utc="1970-01-01T00:03:20+00:00",
    )
    completed = asyncio.run(
        wait_for_resume_schedule(
            _ActiveState(),
            "book",
            schedule,
            clock=lambda: next(clock_values),
            sleep=fake_sleep,
        )
    )

    assert completed is True
    assert sleeps == [1.0]
