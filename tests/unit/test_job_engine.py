from __future__ import annotations

import asyncio

import pytest

from src.core.job_engine import JobEngine, JobPhase, JobPhaseStatus, phase_names


def test_job_engine_declares_expected_pipeline():
    assert phase_names(JobEngine.expected_pipeline()) == [
        "ingest",
        "prepare",
        "translate",
        "refine",
        "audit",
        "repair",
        "assemble",
        "publish",
    ]


def test_job_engine_runs_sync_phase_and_records_events():
    events = []
    engine = JobEngine(job_id="job-1", on_phase_event=events.append)

    engine.mark_phase(JobPhase.INGEST, "ingest")
    result = asyncio.run(engine.run_phase(
        JobPhase.PREPARE,
        lambda value: value + 1,
        41,
        message="prepare",
    ))

    assert result == 42
    assert [event.status for event in events] == [
        JobPhaseStatus.STARTED,
        JobPhaseStatus.STARTED,
        JobPhaseStatus.COMPLETED,
    ]
    assert events[0].phase == JobPhase.INGEST
    assert events[1].phase == JobPhase.PREPARE
    assert engine.loaded_phases() == (JobPhase.INGEST, JobPhase.PREPARE)
    assert engine.active_phase is None
    assert events[-1].elapsed_seconds >= 0


def test_job_engine_runs_async_phase():
    async def work():
        await asyncio.sleep(0)
        return "done"

    engine = JobEngine(job_id="job-2")

    assert asyncio.run(engine.run_phase(JobPhase.TRANSLATE, work)) == "done"
    assert engine.history[-1].status == JobPhaseStatus.COMPLETED


def test_job_engine_records_failed_phase_and_reraises():
    events = []
    engine = JobEngine(job_id="job-3", on_phase_event=events.append)

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(engine.run_phase(JobPhase.AUDIT, fail))

    assert [event.status for event in events] == [
        JobPhaseStatus.STARTED,
        JobPhaseStatus.FAILED,
    ]
    assert events[-1].phase == JobPhase.AUDIT
    assert events[-1].message == "boom"


def test_job_engine_callback_errors_do_not_break_phase_execution():
    def broken_callback(event):
        raise RuntimeError(f"callback failed for {event.phase.value}")

    engine = JobEngine(job_id="job-4", on_phase_event=broken_callback)

    result = asyncio.run(engine.run_phase(JobPhase.TRANSLATE, lambda: "ok"))

    assert result == "ok"
    assert len(engine.history) == 2
    assert engine.history[-1].status == JobPhaseStatus.COMPLETED
    assert engine.callback_errors
