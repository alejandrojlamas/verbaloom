"""Request-scoped attempt observation, independent of persistence and Flask."""

from contextvars import ContextVar

attempt_observer = ContextVar("llm_attempt_observer", default=None)
attempt_reserver = ContextVar("llm_attempt_reserver", default=None)


def reserve_attempt(**request):
    reserver = attempt_reserver.get()
    return reserver(**request) if reserver is not None else None


def observe_attempt(**event):
    observer = attempt_observer.get()
    if observer is not None:
        observer(**event)
