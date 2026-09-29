from __future__ import annotations

import threading

from src.api.job_supervisor import JobSupervisor


def test_duplicate_start_never_runs_two_workers_for_one_job():
    supervisor = JobSupervisor()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def worker(label):
        calls.append(label)
        started.set()
        release.wait(timeout=2)

    assert supervisor.start("book", worker, args=("first",)) == "started"
    assert started.wait(timeout=1)
    assert supervisor.start("book", worker, args=("duplicate",)) == "already_running"

    release.set()
    for _ in range(100):
        if not supervisor.is_running("book"):
            break
        threading.Event().wait(0.01)

    assert calls == ["first"]


def test_recovery_handoff_starts_only_after_current_worker_exits():
    supervisor = JobSupervisor()
    first_started = threading.Event()
    first_release = threading.Event()
    second_done = threading.Event()
    active = 0
    max_active = 0
    lock = threading.Lock()
    calls = []

    def worker(label):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        calls.append(label)
        if label == "first":
            first_started.set()
            first_release.wait(timeout=2)
        else:
            second_done.set()
        with lock:
            active -= 1

    assert supervisor.start("book", worker, args=("first",)) == "started"
    assert first_started.wait(timeout=1)
    assert supervisor.start(
        "book",
        worker,
        args=("recovery",),
        allow_handoff=True,
    ) == "handoff_queued"
    assert not second_done.is_set()

    first_release.set()
    assert second_done.wait(timeout=1)
    assert calls == ["first", "recovery"]
    assert max_active == 1


def test_only_one_recovery_handoff_can_be_queued():
    supervisor = JobSupervisor()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def worker(label):
        calls.append(label)
        if label == "first":
            started.set()
            release.wait(timeout=2)

    supervisor.start("book", worker, args=("first",))
    assert started.wait(timeout=1)
    assert supervisor.start(
        "book", worker, args=("recovery",), allow_handoff=True
    ) == "handoff_queued"
    assert supervisor.start(
        "book", worker, args=("duplicate-recovery",), allow_handoff=True
    ) == "already_queued"

    release.set()
    for _ in range(100):
        if calls == ["first", "recovery"]:
            break
        threading.Event().wait(0.01)
    assert calls == ["first", "recovery"]


def test_manual_resume_atomically_replaces_pending_automatic_handoff():
    supervisor = JobSupervisor()
    started = threading.Event()
    release = threading.Event()
    manual_done = threading.Event()
    calls = []

    def worker(label):
        calls.append(label)
        if label == "first":
            started.set()
            release.wait(timeout=2)
        if label == "manual":
            manual_done.set()

    supervisor.start("book", worker, args=("first",))
    assert started.wait(timeout=1)
    assert supervisor.start(
        "book", worker, args=("automatic",), allow_handoff=True
    ) == "handoff_queued"
    assert supervisor.start(
        "book",
        worker,
        args=("manual",),
        allow_handoff=True,
        replace_handoff=True,
    ) == "handoff_replaced"

    release.set()
    assert manual_done.wait(timeout=1)
    assert calls == ["first", "manual"]


def test_explicit_pause_cancels_pending_handoff():
    supervisor = JobSupervisor()
    started = threading.Event()
    release = threading.Event()
    calls = []

    def worker(label):
        calls.append(label)
        if label == "first":
            started.set()
            release.wait(timeout=2)

    supervisor.start("book", worker, args=("first",))
    assert started.wait(timeout=1)
    supervisor.start("book", worker, args=("recovery",), allow_handoff=True)

    assert supervisor.cancel_pending_handoff("book") is True
    assert supervisor.has_pending_handoff("book") is False
    release.set()
    for _ in range(100):
        if not supervisor.is_running("book"):
            break
        threading.Event().wait(0.01)
    assert calls == ["first"]

    assert supervisor.start(
        "book", worker, args=("stale-timer",), allow_handoff=True
    ) == "cancelled"
    assert supervisor.start(
        "book",
        worker,
        args=("manual",),
        allow_handoff=True,
        replace_handoff=True,
    ) == "started"
    for _ in range(100):
        if not supervisor.is_running("book"):
            break
        threading.Event().wait(0.01)
    assert calls == ["first", "manual"]
