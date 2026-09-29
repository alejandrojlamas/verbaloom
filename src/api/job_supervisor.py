"""Single-owner execution for long-running background jobs.

The translation lifecycle can request a fresh worker from several places
(manual resume, provider recovery, failed-unit recovery, and startup recovery).
Starting raw threads in each caller made those paths race: the old worker could
still be unwinding while a new worker began writing the same checkpoint.

``JobSupervisor`` gives each job id exactly one active worker.  A recovery path
may request one handoff; that replacement starts only after the current worker
has exited.  Repeated UI clicks are ignored rather than queued.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Dict, Optional, Tuple


WorkerTarget = Callable[..., Any]


@dataclass(frozen=True)
class _WorkerRequest:
    target: WorkerTarget
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    name: str


class JobSupervisor:
    """Own at most one worker thread and one pending handoff per job id."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._workers: Dict[str, threading.Thread] = {}
        self._pending: Dict[str, _WorkerRequest] = {}
        self._cancelled: set[str] = set()

    def start(
        self,
        job_id: str,
        target: WorkerTarget,
        *,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        allow_handoff: bool = False,
        replace_handoff: bool = False,
        thread_name: str = "",
    ) -> str:
        """Start a worker, reject a duplicate, or queue one recovery handoff.

        Returns ``started``, ``handoff_queued``, ``handoff_replaced``,
        ``already_running``, or ``already_queued``.  Recovery paths may queue a
        handoff; an explicit manual resume may replace that pending request so
        the user's current model and settings win atomically.
        """
        normalized_id = str(job_id or "").strip()
        if not normalized_id:
            raise ValueError("job_id is required")
        request = _WorkerRequest(
            target=target,
            args=tuple(args),
            kwargs=dict(kwargs or {}),
            name=thread_name or f"job-{normalized_id[-12:]}",
        )

        with self._lock:
            if normalized_id in self._cancelled:
                if not replace_handoff:
                    return "cancelled"
                # Only an explicit manual resume uses replace_handoff.  It is
                # the operation that revokes a previous user pause.
                self._cancelled.discard(normalized_id)
            worker = self._workers.get(normalized_id)
            if worker is not None and worker.is_alive():
                if not allow_handoff:
                    return "already_running"
                if normalized_id in self._pending:
                    if replace_handoff:
                        self._pending[normalized_id] = request
                        return "handoff_replaced"
                    return "already_queued"
                self._pending[normalized_id] = request
                return "handoff_queued"

            # A dead thread can remain visible briefly between its target
            # returning and its finalizer acquiring this lock.
            self._workers.pop(normalized_id, None)
            if normalized_id in self._pending:
                if replace_handoff:
                    self._pending[normalized_id] = request
                    return "handoff_replaced"
                if not allow_handoff:
                    return "already_queued"
                return "already_queued"
            self._launch_locked(normalized_id, request)
            return "started"

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            worker = self._workers.get(str(job_id or ""))
            return bool(worker and worker.is_alive())

    def has_pending_handoff(self, job_id: str) -> bool:
        with self._lock:
            return str(job_id or "") in self._pending

    def cancel_pending_handoff(self, job_id: str) -> bool:
        with self._lock:
            normalized_id = str(job_id or "")
            self._cancelled.add(normalized_id)
            return self._pending.pop(normalized_id, None) is not None

    def active_job_ids(self) -> set[str]:
        with self._lock:
            return {
                job_id
                for job_id, worker in self._workers.items()
                if worker.is_alive()
            }

    def _launch_locked(self, job_id: str, request: _WorkerRequest) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(job_id, request),
            name=request.name,
            daemon=True,
        )
        self._workers[job_id] = thread
        thread.start()

    def _run(self, job_id: str, request: _WorkerRequest) -> None:
        try:
            request.target(*request.args, **request.kwargs)
        finally:
            with self._lock:
                current = self._workers.get(job_id)
                if current is threading.current_thread():
                    self._workers.pop(job_id, None)
                pending = self._pending.pop(job_id, None)
                if pending is not None:
                    self._launch_locked(job_id, pending)
