"""A resume requested during the DeepSeek high-price window is queued."""

from __future__ import annotations

import time

from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.resume_schedule import resume_schedule_from_config


def _app(tmp_path, monkeypatch, *, peak, provider="deepseek"):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    preserved = uploads / "book.txt"
    preserved.write_text("source", encoding="utf-8")
    endpoint = (
        "https://api.deepseek.com/chat/completions"
        if provider == "deepseek"
        else "http://127.0.0.1:11434/api/generate"
    )
    config = {
        "file_type": "txt",
        "preserved_input_path": str(preserved),
        "llm_provider": provider,
        "llm_api_endpoint": endpoint,
        "deepseek_api_key": "",
        "model": "deepseek-v4-pro",
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": "book-es.txt",
    }
    checkpoint = {
        "job": {
            "status": "interrupted",
            "config": config,
            "progress": {"total_chunks": 4, "completed_chunks": 2},
        },
        "resume_from_index": 2,
    }

    class Checkpoints:
        uploads_dir = tmp_path / "checkpoint-uploads"

        def load_checkpoint(self, _translation_id):
            return checkpoint

        def update_job_config(self, _translation_id, updated):
            checkpoint["job"]["config"] = updated

        def mark_running(self, _translation_id):
            checkpoint["job"]["status"] = "running"

    class State:
        def __init__(self):
            self.checkpoint_manager = Checkpoints()
            self.data = {}

        def get_all_translations(self):
            return self.data

        def restore_job_from_checkpoint(self, translation_id, *, pending_resume=False):
            self.data[translation_id] = {
                "status": "queued", "config": dict(config), "interrupted": True,
            }
            return True

        def set_translation_field(self, translation_id, field, value):
            self.data[translation_id][field] = value

    pricing = type("Pricing", (), {
        "disabled": peak,
        "to_dict": lambda self: {
            "disabled": peak,
            "display_timezone": "America/Mexico_City",
            "seconds_until_available": 5400,
            "next_available_at_utc": "",
            "next_available_at_local": "2026-11-13T04:00:00-06:00",
        },
    })()
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes.get_deepseek_pricing_status",
        lambda: pricing,
    )
    starts = []
    state = State()
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, resumed, **kwargs: starts.append(resumed) or "started",
            output_dir=tmp_path,
        )
    )
    return app, starts


def test_resume_during_peak_is_scheduled_instead_of_refused(tmp_path, monkeypatch):
    app, starts = _app(tmp_path, monkeypatch, peak=True)
    before = time.time()

    response = app.test_client().post("/api/resume/job-1", json={})

    assert response.status_code == 200
    body = response.get_json()
    assert body["scheduled_resume"]["status"] == "pricing_wait"
    assert body["scheduled_resume"]["reason"] == "deepseek_peak_pricing"
    assert body["scheduled_resume"]["resume_at_local"] == "2026-11-13T04:00:00-06:00"

    # The worker is handed a durable schedule that it honors before any
    # provider request, so no token is sent during the window.
    assert len(starts) == 1
    schedule = resume_schedule_from_config(starts[0])
    assert schedule is not None
    assert schedule.status == "pricing_wait"
    assert schedule.reason == "deepseek_peak_pricing"
    assert before + 5400 <= schedule.resume_at_epoch <= time.time() + 5410
    assert starts[0]["_explicit_resume_requested"] is True


def test_resume_outside_peak_starts_immediately(tmp_path, monkeypatch):
    app, starts = _app(tmp_path, monkeypatch, peak=False)

    response = app.test_client().post("/api/resume/job-1", json={})

    assert response.status_code == 200
    assert response.get_json()["scheduled_resume"] is None
    assert resume_schedule_from_config(starts[0]) is None


def test_other_providers_are_never_scheduled(tmp_path, monkeypatch):
    app, starts = _app(tmp_path, monkeypatch, peak=True, provider="ollama")

    response = app.test_client().post("/api/resume/job-1", json={})

    assert response.status_code == 200
    assert response.get_json()["scheduled_resume"] is None
    assert resume_schedule_from_config(starts[0]) is None
