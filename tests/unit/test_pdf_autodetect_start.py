from pathlib import Path

from flask import Flask
import yaml

from src.api.blueprints.security_routes import _detect_uploaded_file_language
from src.api.blueprints.translation_routes import create_translation_blueprint
from src.core.book_profiles import create_profile


def test_upload_language_detection_prefers_saved_filename_extension(monkeypatch):
    calls = []

    def fake_detect(file_data, filename):
        calls.append(filename)
        if filename.endswith(".pdf"):
            return "English", 0.99
        return None, 0.0

    monkeypatch.setattr(
        "src.api.blueprints.security_routes.LanguageDetector.detect_language_from_file",
        fake_detect,
    )

    language, confidence = _detect_uploaded_file_language(
        b"%PDF",
        "How to Build Your Career in_ (z-library.sk, 1lib.sk, z-lib.sk)",
        Path("secure_name.pdf"),
    )

    assert language == "English"
    assert confidence == 0.99
    assert calls == ["secure_name.pdf"]


class FakeStateManager:
    def __init__(self):
        self.created = {}

    def create_translation(self, translation_id, config):
        self.created[translation_id] = config


def test_translation_start_is_blocked_before_state_or_network_during_peak(
    tmp_path,
    monkeypatch,
):
    state = FakeStateManager()
    started = []
    pricing = type("Pricing", (), {
        "disabled": True,
        "to_dict": lambda self: {
            "disabled": True,
            "display_timezone": "America/Mexico_City",
            "next_available_at_local": "2026-09-02T22:00:00-06:00",
        },
    })()
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes.get_deepseek_pricing_status",
        lambda: pricing,
    )
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda *args: started.append(args),
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post(
        "/api/translate",
        json={
            "text": "Hello world",
            "file_type": "txt",
            "source_language": "English",
            "target_language": "Spanish",
            "model": "deepseek-v4-pro",
            "llm_provider": "deepseek",
            "llm_api_endpoint": "https://api.deepseek.com/chat/completions",
            "output_filename": "book.txt",
        },
    )

    assert response.status_code == 423
    assert response.get_json()["code"] == "deepseek_peak_pricing"
    assert response.get_json()["availability"]["display_timezone"] == "America/Mexico_City"
    assert state.created == {}
    assert started == []


def test_interrupting_pricing_wait_is_immediate_and_keeps_checkpoint(tmp_path):
    class Checkpoints:
        def __init__(self):
            self.interrupted = []

        def update_job_config(self, *_args):
            return True

        def mark_paused(self, *_args):
            return True

        def mark_interrupted(self, translation_id):
            self.interrupted.append(translation_id)

    class WaitingState:
        def __init__(self):
            self.checkpoint_manager = Checkpoints()
            self.data = {
                "job-1": {
                    "status": "pricing_wait",
                    "config": {},
                    "interrupted": False,
                    "resume_at_utc": "future",
                    "resume_at_local": "future-local",
                }
            }

        def exists(self, translation_id):
            return translation_id in self.data

        def get_translation(self, translation_id):
            return dict(self.data[translation_id])

        def get_translation_field(self, translation_id, field):
            return self.data[translation_id].get(field)

        def set_translation_field(self, translation_id, field, value):
            self.data[translation_id][field] = value

        def set_interrupted(self, translation_id, interrupted=True):
            self.data[translation_id]["interrupted"] = interrupted

    state = WaitingState()
    cancelled_handoffs = []
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda *_args: None,
            output_dir=tmp_path,
            cancel_translation_handoff=cancelled_handoffs.append,
        )
    )

    response = app.test_client().post("/api/translation/job-1/interrupt")

    assert response.status_code == 200
    assert state.data["job-1"]["status"] == "interrupted"
    assert state.data["job-1"]["interrupted"] is True
    assert state.data["job-1"]["resume_at_utc"] is None
    assert state.checkpoint_manager.interrupted == ["job-1"]
    assert cancelled_handoffs == ["job-1"]


def test_manual_resume_replaces_any_pending_automatic_handoff(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    preserved = uploads / "book.txt"
    preserved.write_text("source", encoding="utf-8")

    config = {
        "file_type": "txt",
        "preserved_input_path": str(preserved),
        "llm_provider": "ollama",
        "llm_api_endpoint": "http://127.0.0.1:11434/api/generate",
        "model": "local-model",
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": "book-es.txt",
        "_manual_pause_requested": True,
    }
    checkpoint = {
        "job": {
            "status": "interrupted",
            "config": config,
            "progress": {"total_chunks": 2, "completed_chunks": 1},
        },
        "resume_from_index": 1,
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

        def restore_job_from_checkpoint(self, translation_id):
            self.data[translation_id] = {
                "status": "paused",
                "config": dict(config),
                "interrupted": False,
            }
            return True

        def set_translation_field(self, translation_id, field, value):
            self.data[translation_id][field] = value

    starts = []

    def start_job(translation_id, resumed_config, **kwargs):
        starts.append((translation_id, resumed_config, kwargs))
        return "handoff_replaced"

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            State(),
            start_job,
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post("/api/resume/job-1", json={})

    assert response.status_code == 200
    assert response.get_json()["worker_state"] == "handoff_replaced"
    assert starts[0][0] == "job-1"
    assert starts[0][2] == {
        "allow_handoff": True,
        "replace_handoff": True,
    }
    assert "_manual_pause_requested" not in starts[0][1]


def test_file_translate_rejects_unmanaged_input_and_output_traversal(tmp_path):
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF sample bytes")
    state = FakeStateManager()
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(state, lambda *_args: None, output_dir=tmp_path / "managed")
    )
    payload = {
        "file_path": str(outside),
        "file_type": "pdf",
        "source_language": "English",
        "target_language": "Spanish",
        "model": "local-model",
        "llm_provider": "ollama",
        "llm_api_endpoint": "http://127.0.0.1:11434/api/generate",
        "output_filename": "book.epub",
    }

    response = app.test_client().post("/api/translate", json=payload)
    assert response.status_code == 403

    payload["output_filename"] = "../book.epub"
    response = app.test_client().post("/api/translate", json=payload)
    assert response.status_code == 400
    assert state.created == {}


def test_text_translate_does_not_route_saved_openai_key_to_custom_endpoint(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_API_KEY", "saved-environment-secret")
    state = FakeStateManager()
    started = {}
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, config: started.update(config),
            output_dir=tmp_path,
        )
    )
    payload = {
        "text": "Hello world",
        "file_type": "txt",
        "source_language": "English",
        "target_language": "Spanish",
        "model": "local-model",
        "llm_provider": "openai",
        "llm_api_endpoint": "http://127.0.0.1:1234/v1/chat/completions",
        "output_filename": "book.txt",
    }

    response = app.test_client().post("/api/translate", json=payload)
    assert response.status_code == 200
    assert started["openai_api_key"] == ""

    payload["openai_api_key"] = "__USE_ENV__"
    response = app.test_client().post("/api/translate", json=payload)
    assert response.status_code == 400


def test_file_translate_autodetects_empty_source_language(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pdf_path = uploads / "uploaded.pdf"
    pdf_path.write_bytes(b"%PDF sample bytes")
    state = FakeStateManager()
    started = {}

    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._detect_source_language_from_file",
        lambda file_path: ("English", 0.98),
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._probe_readable_content",
        lambda file_path, file_type: (True, 12, ""),
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, config: started.update(
                {"translation_id": translation_id, "config": config}
            ),
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post(
        "/api/translate",
        json={
            "file_path": str(pdf_path),
            "file_type": "pdf",
            "source_language": "",
            "target_language": "Spanish",
            "model": "deepseek-v4-pro",
            "llm_provider": "ollama",
            "llm_api_endpoint": "http://localhost:11434/api/generate",
            "output_filename": "Book (Spanish).epub",
            "prompt_options": {},
        },
    )

    assert response.status_code == 200
    assert started["config"]["source_language"] == "English"
    assert started["config"]["prompt_options"]["_input_readable_characters"] == 12
    assert started["config"]["prompt_options"]["_source_language_autodetected"] is True
    assert started["config"]["prompt_options"]["_source_language_confidence"] == 0.98
    assert next(iter(state.created.values()))["source_language"] == "English"


def test_file_translate_falls_back_to_auto_when_autodetect_fails(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    pdf_path = uploads / "uploaded.pdf"
    pdf_path.write_bytes(b"%PDF sample bytes")
    state = FakeStateManager()
    started = {}

    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._detect_source_language_from_file",
        lambda file_path: (None, 0.0),
    )
    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._probe_readable_content",
        lambda file_path, file_type: (True, 12, ""),
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, config: started.update(
                {"translation_id": translation_id, "config": config}
            ),
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post(
        "/api/translate",
        json={
            "file_path": str(pdf_path),
            "file_type": "pdf",
            "source_language": "",
            "target_language": "Spanish",
            "model": "deepseek-v4-pro",
            "llm_provider": "ollama",
            "llm_api_endpoint": "http://localhost:11434/api/generate",
            "output_filename": "Book (Spanish).epub",
            "prompt_options": {},
        },
    )

    assert response.status_code == 200
    assert started["config"]["source_language"] == "Auto"
    assert started["config"]["prompt_options"]["_input_readable_characters"] == 12
    assert started["config"]["prompt_options"]["_source_language_autodetected"] is False
    assert started["config"]["prompt_options"]["_source_language_confidence"] == 0.0
    assert started["config"]["prompt_options"]["_source_language_autodetect_failed"] is True
    assert next(iter(state.created.values()))["source_language"] == "Auto"


def test_file_translate_preserves_mobile_input_filename_metadata(tmp_path, monkeypatch):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    epub_path = uploads / "secure-upload-name.epub"
    epub_path.write_bytes(b"PK sample bytes")
    state = FakeStateManager()
    started = {}

    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._probe_readable_content",
        lambda file_path, file_type: (True, 123, ""),
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, config: started.update(
                {"translation_id": translation_id, "config": config}
            ),
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post(
        "/api/translate",
        json={
            "file_path": str(epub_path),
            "file_type": "epub",
            "source_language": "English",
            "target_language": "Spanish",
            "model": "deepseek-v4-pro",
            "llm_provider": "ollama",
            "llm_api_endpoint": "http://localhost:11434/api/generate",
            "input_filename": "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
            "output_filename": "Under the Volcano (Spanish).epub",
            "prompt_options": {},
        },
    )

    assert response.status_code == 200
    assert started["config"]["input_filename"] == "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub"
    assert started["config"]["original_filename"] == "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub"
    assert next(iter(state.created.values()))["input_filename"] == "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub"


def test_file_translate_queues_mobile_under_with_inferred_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path / "profiles"))
    profile_dir = create_profile(
        "auto_under_the_volcano_malcolm_lowry",
        profiles_root=tmp_path / "profiles",
    )
    profile_path = profile_dir / "profile.yml"
    profile_config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    profile_config.update({
        "source_name": "Under the Volcano - Malcolm Lowry.epub",
        "target_locale": "es-MX",
        "generated_profile": True,
        "auto_detect": {"enabled": False},
    })
    profile_path.write_text(
        yaml.safe_dump(profile_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    epub_path = uploads / "secure-upload-name.epub"
    epub_path.write_bytes(b"PK sample bytes")
    state = FakeStateManager()
    started = {}

    monkeypatch.setattr(
        "src.api.blueprints.translation_routes._probe_readable_content",
        lambda file_path, file_type: (True, 123, ""),
    )

    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state,
            lambda translation_id, config: started.update(
                {"translation_id": translation_id, "config": config}
            ),
            output_dir=tmp_path,
        )
    )

    response = app.test_client().post(
        "/api/translate",
        json={
            "file_path": str(epub_path),
            "file_type": "epub",
            "source_language": "English",
            "target_language": "Spanish",
            "model": "deepseek-v4-pro",
            "llm_provider": "ollama",
            "llm_api_endpoint": "http://localhost:11434/api/generate",
            "input_filename": "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
            "output_filename": "Under the Volcano (Spanish).epub",
            "prompt_options": {},
        },
    )

    assert response.status_code == 200
    created_config = next(iter(state.created.values()))
    assert created_config["prompt_options"]["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert created_config["prompt_options"]["editorial_mode"] == "book_profile"
    assert created_config["prompt_options"]["translation_profile_mode"] is True
    assert created_config["prompt_options"]["use_profile_glossary"] is True
    assert started["config"]["prompt_options"]["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert response.get_json()["config_received"]["prompt_options"]["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
