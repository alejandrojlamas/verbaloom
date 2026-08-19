"""Generic job orchestration primitives.

This module is intentionally framework-agnostic. Web handlers can subscribe to
phase events, but the engine itself knows nothing about Flask, Socket.IO, or the
translation state manager.
"""

from __future__ import annotations

import inspect
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable


class JobPhase(str, Enum):
    INGEST = "ingest"
    PREPARE = "prepare"
    TRANSLATE = "translate"
    REFINE = "refine"
    AUDIT = "audit"
    REPAIR = "repair"
    ASSEMBLE = "assemble"
    PUBLISH = "publish"


class JobPhaseStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class JobPhaseEvent:
    job_id: str
    phase: JobPhase
    status: JobPhaseStatus
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "phase": self.phase.value,
            "status": self.status.value,
            "message": self.message,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
            "elapsed_seconds": self.elapsed_seconds,
        }


PhaseCallback = Callable[[JobPhaseEvent], None]


class JobEngine:
    """Small phase runner for long document jobs.

    The engine provides a stable contract for orchestration without owning any
    domain-specific translation logic. Callers can mark phases explicitly or run
    synchronous/asynchronous functions inside a phase.
    """

    def __init__(
        self,
        *,
        job_id: str,
        on_phase_event: PhaseCallback | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.job_id = job_id
        self._on_phase_event = on_phase_event
        self._clock = clock
        self._history: list[JobPhaseEvent] = []
        self._active_phase: JobPhase | None = None
        self._active_started_at: float | None = None
        self._phase_depth = 0
        self._callback_errors: list[str] = []

    @property
    def active_phase(self) -> JobPhase | None:
        return self._active_phase

    @property
    def history(self) -> tuple[JobPhaseEvent, ...]:
        return tuple(self._history)

    @property
    def callback_errors(self) -> tuple[str, ...]:
        return tuple(self._callback_errors)

    def mark_phase(
        self,
        phase: JobPhase | str,
        message: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> JobPhaseEvent:
        normalized = self._normalize_phase(phase)
        self._active_phase = normalized
        self._active_started_at = self._clock()
        return self._emit(
            normalized,
            JobPhaseStatus.STARTED,
            message,
            metadata=metadata,
            elapsed_seconds=0.0,
        )

    @asynccontextmanager
    async def phase(
        self,
        phase: JobPhase | str,
        message: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ):
        normalized = self._normalize_phase(phase)
        started_at = self._clock()
        parent_depth = self._phase_depth
        previous_phase = self._active_phase
        previous_started_at = self._active_started_at
        self._phase_depth += 1
        self._active_phase = normalized
        self._active_started_at = started_at
        self._emit(normalized, JobPhaseStatus.STARTED, message, metadata=metadata)
        try:
            yield
        except Exception as exc:
            self._emit(
                normalized,
                JobPhaseStatus.FAILED,
                str(exc),
                metadata=metadata,
                elapsed_seconds=max(0.0, self._clock() - started_at),
            )
            raise
        else:
            self._emit(
                normalized,
                JobPhaseStatus.COMPLETED,
                message,
                metadata=metadata,
                elapsed_seconds=max(0.0, self._clock() - started_at),
            )
        finally:
            self._phase_depth = max(0, self._phase_depth - 1)
            if parent_depth > 0:
                self._active_phase = previous_phase
                self._active_started_at = previous_started_at
            else:
                self._active_phase = None
                self._active_started_at = None

    async def run_phase(
        self,
        phase: JobPhase | str,
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *args: Any,
        message: str = "",
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        async with self.phase(phase, message, metadata=metadata):
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result

    def loaded_phases(self) -> tuple[JobPhase, ...]:
        seen: list[JobPhase] = []
        for event in self._history:
            if event.phase not in seen:
                seen.append(event.phase)
        return tuple(seen)

    @staticmethod
    def expected_pipeline() -> tuple[JobPhase, ...]:
        return (
            JobPhase.INGEST,
            JobPhase.PREPARE,
            JobPhase.TRANSLATE,
            JobPhase.REFINE,
            JobPhase.AUDIT,
            JobPhase.REPAIR,
            JobPhase.ASSEMBLE,
            JobPhase.PUBLISH,
        )

    def _emit(
        self,
        phase: JobPhase,
        status: JobPhaseStatus,
        message: str = "",
        *,
        metadata: dict[str, Any] | None = None,
        elapsed_seconds: float = 0.0,
    ) -> JobPhaseEvent:
        event = JobPhaseEvent(
            job_id=self.job_id,
            phase=phase,
            status=status,
            message=message,
            metadata=dict(metadata or {}),
            elapsed_seconds=elapsed_seconds,
        )
        self._history.append(event)
        if self._on_phase_event is not None:
            try:
                self._on_phase_event(event)
            except Exception as exc:
                self._callback_errors.append(str(exc))
        return event

    @staticmethod
    def _normalize_phase(phase: JobPhase | str) -> JobPhase:
        if isinstance(phase, JobPhase):
            return phase
        return JobPhase(str(phase))


def phase_names(phases: Iterable[JobPhase]) -> list[str]:
    return [phase.value for phase in phases]
