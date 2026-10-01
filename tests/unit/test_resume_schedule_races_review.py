from __future__ import annotations

import asyncio
import copy
import threading
from typing import ClassVar

from flask import Flask

from src.api.blueprints.translation_routes import (
    _blocks_resume,
    create_translation_blueprint,
)
from src.api.handlers import (
    _claim_translation_worker,
    _honor_persisted_resume_schedule,
    _schedule_failed_chunk_recovery,
)
from src.api.translation_state import translation_lifecycle_guard


def _scheduled_config(model: str = "old-model") -> dict:
    return {
        "model": model,
        "_scheduled_resume_at_epoch": 4_000_000_000.0,
        "_scheduled_resume_at_utc": "2096-10-02T07:06:40+00:00",
        "_scheduled_resume_reason": "provider_wait",
        "_scheduled_resume_status": "provider_wait",
    }


class _Checkpoints:
    def __init__(self, config: dict | None = None):
        self.config = copy.deepcopy(config or {})
        self.status = "running"
        self.events: list[str] = []
        self.checkpoint = {
            "job": {
                "status": "running",
                "config": self.config,
                "progress": {"total_chunks": 2, "completed_chunks": 1},
            },
            "resume_from_index": 1,
        }

    def update_job_config(self, _translation_id, config):
        self.config = copy.deepcopy(config)
        self.checkpoint["job"]["config"] = self.config
        return True

    def mark_running(self, _translation_id):
        self.status = "running"
        self.events.append("running")
        self.checkpoint["job"]["status"] = "running"
        return True

    def mark_paused(self, _translation_id):
        self.status = "paused"
        self.events.append("paused")
        self.checkpoint["job"]["status"] = "paused"
        return True

    def mark_interrupted(self, _translation_id):
        self.status = "interrupted"
        self.events.append("interrupted")
        self.checkpoint["job"]["status"] = "interrupted"
        return True

    def update_progress(self, _translation_id, **_updates):
        return True

    def load_checkpoint(self, _translation_id):
        return copy.deepcopy(self.checkpoint)

    def get_preserved_input_path(self, _translation_id):
        return self.config.get("preserved_input_path")


class _State:
    def __init__(self, config: dict | None = None, *, status: str = "running"):
        self._lock = threading.RLock()
        self.checkpoint_manager = _Checkpoints(config)
        self.data = {
            "book": {
                "config": copy.deepcopy(config or {}),
                "status": status,
                "interrupted": False,
                "pause_reason": None,
                "stats": {},
                "recovery_scheduled": False,
            }
        }

    def exists(self, translation_id):
        with self._lock:
            return translation_id in self.data

    def get_translation(self, translation_id):
        with self._lock:
            return copy.deepcopy(self.data.get(translation_id))

    def get_all_translations(self):
        with self._lock:
            return copy.deepcopy(self.data)

    def get_translation_field(self, translation_id, field, default=None):
        with self._lock:
            return self.data[translation_id].get(field, default)

    def set_translation_field(self, translation_id, field, value):
        with self._lock:
            self.data[translation_id][field] = value
        return True

    def set_interrupted(self, translation_id, interrupted=True):
        return self.set_translation_field(translation_id, "interrupted", interrupted)

    def get_checkpoint_manager(self):
        return self.checkpoint_manager

    def restore_job_from_checkpoint(self, translation_id, *, pending_resume=False):
        checkpoint = self.checkpoint_manager.load_checkpoint(translation_id)
        self.data[translation_id] = {
            "config": copy.deepcopy(checkpoint["job"]["config"]),
            "status": "queued" if pending_resume else "paused",
            "interrupted": bool(pending_resume),
            "pause_reason": None,
            "stats": copy.deepcopy(checkpoint["job"]["progress"]),
        }
        return True


def _persist_manual_pause(state: _State) -> None:
    with translation_lifecycle_guard(state):
        config = dict(state.get_translation_field("book", "config") or {})
        config["_manual_pause_requested"] = True
        state.set_translation_field("book", "config", config)
        state.set_translation_field("book", "interrupted", True)
        state.set_translation_field("book", "status", "interrupted")
        state.checkpoint_manager.update_job_config("book", config)
        state.checkpoint_manager.mark_interrupted("book")


def test_pause_between_worker_claim_and_restored_wait_is_not_erased(monkeypatch):
    config = _scheduled_config()
    state = _State(config)
    claimed = _claim_translation_worker("book", config, state)
    assert claimed is not None
    _persist_manual_pause(state)

    async def should_not_wait(*_args, **_kwargs):
        raise AssertionError("manual pause must be observed before waiting")

    monkeypatch.setattr(
        "src.api.handlers._interruptible_provider_wait", should_not_wait
    )
    result, ready = asyncio.run(
        _honor_persisted_resume_schedule("book", claimed, state, None)
    )

    assert ready is False
    assert result["_manual_pause_requested"] is True
    assert state.get_translation_field("book", "status") == "interrupted"
    assert state.checkpoint_manager.status == "interrupted"


def test_pause_after_wait_expires_keeps_durable_manual_intent(monkeypatch):
    config = _scheduled_config()
    state = _State(config)

    async def pause_then_finish(*_args, **_kwargs):
        _persist_manual_pause(state)
        return True

    monkeypatch.setattr(
        "src.api.handlers._interruptible_provider_wait", pause_then_finish
    )
    monkeypatch.setattr("src.api.handlers.emit_update", lambda *_a, **_k: None)
    result, ready = asyncio.run(
        _honor_persisted_resume_schedule("book", config, state, None)
    )

    assert ready is False
    assert result["_manual_pause_requested"] is True
    assert state.get_translation_field("book", "status") == "interrupted"
    assert state.checkpoint_manager.config["_manual_pause_requested"] is True


def test_old_waiter_cannot_demote_pending_explicit_resume_checkpoint(monkeypatch):
    config = _scheduled_config()
    state = _State(config)

    async def queue_explicit_resume(*_args, **_kwargs):
        replacement = {
            "model": "new-model",
            "is_resume": True,
            "_explicit_resume_requested": True,
        }
        with translation_lifecycle_guard(state):
            state.set_translation_field("book", "config", replacement)
            state.set_translation_field("book", "status", "queued")
            state.set_translation_field("book", "interrupted", True)
            state.checkpoint_manager.update_job_config("book", replacement)
            state.checkpoint_manager.mark_running("book")
        return False

    monkeypatch.setattr(
        "src.api.handlers._interruptible_provider_wait", queue_explicit_resume
    )
    monkeypatch.setattr("src.api.handlers.emit_update", lambda *_a, **_k: None)
    result, ready = asyncio.run(
        _honor_persisted_resume_schedule("book", config, state, None)
    )

    assert ready is False
    assert result["model"] == "new-model"
    assert state.checkpoint_manager.status == "running"
    assert state.checkpoint_manager.events[-1] == "running"


class _CapturedThread:
    callbacks: ClassVar[list] = []

    def __init__(self, *, target, daemon, name):
        del daemon, name
        self.target = target

    def start(self):
        self.callbacks.append(self.target)


def _schedule_plan(model: str, delay: int = 2) -> dict:
    return {
        "config": {"model": model, "is_resume": True},
        "scope": "chunks",
        "delay_seconds": delay,
    }


def _replace_recovery_generation(state: _State) -> None:
    with translation_lifecycle_guard(state):
        state.set_translation_field("book", "recovery_scheduled", False)
        state.set_translation_field("book", "_recovery_token", None)
        state.set_translation_field("book", "interrupted", False)
        state.set_translation_field("book", "status", "running")


def test_cancelled_old_daemon_cannot_clear_new_recovery(monkeypatch, tmp_path):
    _CapturedThread.callbacks = []
    monkeypatch.setattr("src.api.handlers.threading.Thread", _CapturedThread)
    monkeypatch.setattr(
        "src.api.handlers.wait_for_resume_schedule_sync",
        lambda *_a, **_k: False,
    )
    state = _State({"model": "base"})

    assert _schedule_failed_chunk_recovery(
        "book", _schedule_plan("old"), state, str(tmp_path), None
    )
    old_callback = _CapturedThread.callbacks[0]
    _replace_recovery_generation(state)
    assert _schedule_failed_chunk_recovery(
        "book", _schedule_plan("new", 10), state, str(tmp_path), None
    )
    new_token = state.get_translation_field("book", "_recovery_token")

    old_callback()

    assert state.get_translation_field("book", "recovery_scheduled") is True
    assert state.get_translation_field("book", "_recovery_token") == new_token
    assert state.get_translation_field("book", "config")["model"] == "new"


def test_old_deadline_cannot_consume_new_schedule_or_restore_old_model(
    monkeypatch, tmp_path
):
    _CapturedThread.callbacks = []
    starts = []
    monkeypatch.setattr("src.api.handlers.threading.Thread", _CapturedThread)
    monkeypatch.setattr(
        "src.api.handlers.wait_for_resume_schedule_sync",
        lambda *_a, **_k: True,
    )
    monkeypatch.setattr(
        "src.api.handlers.start_translation_job",
        lambda *args, **kwargs: starts.append((args, kwargs)),
    )
    state = _State({"model": "base"})

    _schedule_failed_chunk_recovery(
        "book", _schedule_plan("old"), state, str(tmp_path), None
    )
    old_callback = _CapturedThread.callbacks[0]
    _replace_recovery_generation(state)
    _schedule_failed_chunk_recovery(
        "book", _schedule_plan("new", 10), state, str(tmp_path), None
    )

    old_callback()

    assert starts == []
    assert state.get_translation_field("book", "config")["model"] == "new"
    assert state.get_translation_field("book", "recovery_scheduled") is True


def test_pause_during_workerless_deterministic_backoff_allows_resume(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "book.txt"
    source.write_text("source", encoding="utf-8")
    config = {
        "file_type": "txt",
        "preserved_input_path": str(source),
        "llm_provider": "ollama",
        "llm_api_endpoint": "http://127.0.0.1:11434/api/generate",
        "model": "local-model",
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": "book-es.txt",
    }
    state = _State(config)
    state.set_translation_field("book", "recovery_scheduled", True)
    state.set_translation_field("book", "_recovery_token", "old-generation")
    starts = []
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda *args, **kwargs: starts.append((args, kwargs)) or "started",
            output_dir=tmp_path,
            cancel_translation_handoff=lambda _translation_id: True,
        )
    )
    client = app.test_client()

    paused = client.post("/api/translation/book/interrupt")
    resumed = client.post("/api/resume/book", json={})

    assert paused.status_code == 200
    assert resumed.status_code == 200
    assert state.get_translation_field("book", "status") == "queued"
    assert state.get_translation_field("book", "_recovery_token") is None
    assert starts and starts[0][1]["replace_handoff"] is True


def test_manual_resume_can_reclaim_its_own_provider_wait(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    source = uploads / "book.txt"
    source.write_text("source", encoding="utf-8")
    config = {
        **_scheduled_config("waiting-model"),
        "file_type": "txt",
        "preserved_input_path": str(source),
        "llm_provider": "ollama",
        "llm_api_endpoint": "http://127.0.0.1:11434/api/generate",
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": "book-es.txt",
    }
    config["_scheduled_resume_at_epoch"] = 1.0
    state = _State(config, status="provider_wait")
    starts = []
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda *args, **kwargs: starts.append((args, kwargs)) or "started",
            output_dir=tmp_path,
            cancel_translation_handoff=lambda _translation_id: True,
        )
    )

    response = app.test_client().post("/api/resume/book", json={})

    assert response.status_code == 200
    assert state.get_translation_field("book", "status") == "queued"
    assert starts and starts[0][1]["replace_handoff"] is True


def test_manual_resume_does_not_bypass_live_provider_backoff():
    assert _blocks_resume(
        "book",
        "book",
        {"status": "provider_wait", "config": _scheduled_config()},
    ) is True
