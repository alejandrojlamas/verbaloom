from copy import deepcopy

import pytest
from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.translation_state import TranslationStateManager
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.mark.parametrize("case", [
    "match", "numbered_publication", "different_input", "different_mode",
    "different_instructions", "missing_final", "distinct_filename",
])
def test_resume_only_redirects_when_later_publication_matches_delivery(tmp_path, case):
    root = tmp_path / "outputs"
    uploads = root / "uploads"
    uploads.mkdir(parents=True)
    first = uploads / "input_old.epub"
    second = uploads / "input_new.epub"
    first.write_bytes(b"identical book")
    second.write_bytes(b"different book" if case == "different_input" else first.read_bytes())
    intended = "Book - Mexican Spanish.epub"
    published = "Book - Mexican Spanish (1).epub" if case in {"numbered_publication", "distinct_filename"} else intended

    config = {
        "file_type": "epub", "output_format": "epub", "source_language": "Spanish",
        "target_language": "Spanish", "operation": "transform", "refine_only": True,
        "llm_provider": "ollama", "model": "fixture",
        "text_transform_mode": "mexican_spanish",
        "prompt_options": {"text_transform_mode": "mexican_spanish", "target_locale": "es-MX"},
        "output_filename": f"[partial] {intended}", "_job_output_filename": intended,
        "preserved_input_path": str(first),
    }
    latest = deepcopy(config)
    latest.update(output_filename=published, _job_output_filename=published, preserved_input_path=str(second))
    latest["prompt_options"]["custom_instruction_file"] = ""
    if case == "different_mode":
        latest.update(text_transform_mode="literary_polish")
    if case == "different_instructions":
        latest["prompt_options"]["refinement_instructions"] = "Use formal diction."
    if case == "numbered_publication":
        latest["_requested_output_filename"] = intended
    if case == "distinct_filename":
        latest["_requested_output_filename"] = published
    if case != "missing_final":
        (root / published).write_bytes(b"final book")

    manager = CheckpointManager(str(tmp_path / "jobs.db"))
    try:
        assert manager.db.create_job("old", "epub", config, status="interrupted")
        assert manager.db.create_job("new", "epub", latest)
        assert manager.mark_completed("new")
        state = TranslationStateManager(manager)
        calls = []
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_translation_blueprint(
            state, lambda run_id, _config, **_kwargs: calls.append(run_id), output_dir=str(root),
        ))
        with app.test_client() as client:
            resumable = client.get("/api/resumable").json["resumable_jobs"]
            assert len(resumable) == 1
            response = client.post("/api/resume/old", json={})

        new_output = resumable[0].get("superseded_by")
        if case in {"match", "numbered_publication"}:
            assert new_output == {
                "translation_id": "new", "output_filename": published,
                "completed_at": manager.get_job("new").get("completed_at"),
            }
            assert response.status_code == 409 and response.json["code"] == "superseded_job"
            assert response.json["superseded_by"] == new_output
            assert calls == []
            assert manager.get_job("old")["status"] == "interrupted"
        else:
            assert new_output is None
            assert response.status_code == 200, response.json
            assert calls == ["old"]
    finally:
        manager.close()
