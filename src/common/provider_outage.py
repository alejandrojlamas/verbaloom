"""Tell a provider outage apart from a rejected candidate.

A provider's ``generate`` returns ``None`` both when the model answered badly
and when the provider could not be reached at all (timeouts, connection
resets, HTTP 5xx, an open circuit breaker). The translation pipeline treats
``None`` as "no usable candidate": it burns the unit's attempts within
seconds, marks the unit FAILED and stops the book, although nothing is wrong
with the unit.

This module carries the missing bit of information without changing any
provider's return contract:

* the provider retry loop calls :func:`note_transport_outage` whenever the
  connection itself failed;
* ``await_llm_call`` collects those notes for one operation and, only inside
  :func:`provider_outage_recovery`, turns "``None`` because of an outage" into
  a retryable ``ProviderUnavailableError``. The web job boundary answers it
  with a durable, bounded wait and resumes from the checkpoint.

Outside that scope (CLI, previews, tests, optional helpers) nothing changes.
The module imports nothing from the project so every layer can use it.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Dict, Iterator, Optional

_recovery_enabled: ContextVar[bool] = ContextVar(
    "verbaloom_provider_outage_recovery",
    default=False,
)
_outage_sink: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "verbaloom_provider_outage_sink",
    default=None,
)


@contextmanager
def provider_outage_recovery() -> Iterator[None]:
    """Let LLM operations in this context report outages as retryable."""
    token = _recovery_enabled.set(True)
    try:
        yield
    finally:
        _recovery_enabled.reset(token)


def outage_recovery_enabled() -> bool:
    return bool(_recovery_enabled.get())


@contextmanager
def collect_transport_outage() -> Iterator[Dict[str, str]]:
    """Collect outage notes raised while one LLM operation runs.

    The yielded mapping is shared with tasks started inside the block, because
    a copied context still references the same object.
    """
    sink: Dict[str, str] = {}
    token = _outage_sink.set(sink)
    try:
        yield sink
    finally:
        _outage_sink.reset(token)


def note_transport_outage(reason: str) -> None:
    """Record that the connection to the provider failed, not the model."""
    sink = _outage_sink.get()
    if sink is not None:
        sink["reason"] = str(reason or "transport_failure")
