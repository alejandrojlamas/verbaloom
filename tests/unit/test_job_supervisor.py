from __future__ import annotations

import threading

from src.api.handlers import _claim_translation_worker
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


def test_resume_fence_stays_raised_until_replacement_worker_claims_checkpoint():
    class Checkpoints:
        def __init__(self):
            self.status = "interrupted"

        def load_checkpoint(self, _translation_id):
            return {"resume_from_index": 12}

        def update_job_config(self, _translation_id, _config):
            return True

        def mark_running(self, _translation_id):
            self.status = "running"
            return True

    class State:
        def __init__(self):
            self.data = {
                "config": {},
                "interrupted": False,
                "status": "running",
                "pause_reason": None,
            }
            self.checkpoints = Checkpoints()

        def exists(self, _translation_id):
            return True

        def get_translation_field(self, _translation_id, field):
            return self.data.get(field)

        def set_translation_field(self, _translation_id, field, value):
            self.data[field] = value

        def get_checkpoint_manager(self):
            return self.checkpoints

    supervisor = JobSupervisor()
    state = State()
    old_started = threading.Event()
    old_saw_interrupt = threading.Event()
    replacement_done = threading.Event()
    observations = []

    def old_worker():
        old_started.set()
        assert old_saw_interrupt.wait(timeout=1)
        observations.append(("old_exit", state.data["interrupted"]))

    def replacement_worker():
        config = _claim_translation_worker(
            "book",
            {
                "resume_from_index": 5,
                "_explicit_resume_requested": True,
            },
            state,
        )
        observations.append(
            (
                "replacement_claim",
                config["resume_from_index"],
                state.data["interrupted"],
                state.checkpoints.status,
            )
        )
        replacement_done.set()

    assert supervisor.start("book", old_worker) == "started"
    assert old_started.wait(timeout=1)

    # Simulate the route raising the fence before queuing a manual handoff.
    state.data.update({"status": "queued", "interrupted": True})
    assert supervisor.start(
        "book",
        replacement_worker,
        allow_handoff=True,
        replace_handoff=True,
    ) == "handoff_queued"
    assert state.data["interrupted"] is True

    old_saw_interrupt.set()
    assert replacement_done.wait(timeout=1)
    assert observations == [
        ("old_exit", True),
        ("replacement_claim", 12, False, "running"),
    ]
