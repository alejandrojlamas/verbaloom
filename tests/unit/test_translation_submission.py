"""Submission ownership uses real durable jobs, never a second work queue."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest
from flask import Flask

from src.api.blueprints.translation_routes import create_translation_blueprint
from src.api.translation_state import TranslationStateManager
from src.persistence.checkpoint_manager import CheckpointManager


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("src.api.blueprints.translation_routes._deepseek_pricing_block", lambda *_: None)
    manager = CheckpointManager(str(tmp_path / "jobs.db"))
    state = TranslationStateManager(manager)
    calls = []

    def make_app(current_state=state, worker=None):
        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(create_translation_blueprint(
            current_state,
            worker or (lambda run_id, config, **kwargs: calls.append((run_id, deepcopy(config)))),
            output_dir=str(tmp_path / "outputs"),
        ))
        return app

    yield make_app, state, manager, calls
    manager.close()


def payload():
    return {"submission_id": str(uuid4()), "text": "A complete fixture paragraph.",
            "source_language": "English", "target_language": "Spanish",
            "model": "fixture-model", "llm_provider": "ollama",
            "output_filename": "fixture.txt", "file_type": "txt"}


def test_repeated_submission_launches_once_and_is_durable_before_worker(setup):
    make_app, state, manager, calls = setup
    data = payload()
    with make_app().test_client() as client:
        first = client.post("/api/translate", json=data)
        second = client.post("/api/translate", json=data)
    assert first.status_code == second.status_code == 200
    run_id = first.json["translation_id"]
    assert run_id == "trans_" + data["submission_id"].replace("-", "")
    assert second.json["translation_id"] == run_id
    assert second.json["reused"] is True
    assert len(calls) == 1
    assert manager.get_job(run_id)["status"] == "queued"
    assert state.get_translation(run_id)["config"]["_submission_fingerprint"]


def test_different_payload_cannot_reuse_submission_id(setup):
    make_app, _, _, calls = setup
    data = payload()
    with make_app().test_client() as client:
        assert client.post("/api/translate", json=data).status_code == 200
        conflict = client.post("/api/translate", json={**data, "text": "Another book."})
    assert conflict.status_code == 409
    assert len(calls) == 1


def test_retry_after_restart_does_not_reset_pause_or_launch_worker(setup):
    make_app, _state, manager, calls = setup
    data = payload()
    with make_app().test_client() as client:
        run_id = client.post("/api/translate", json=data).json["translation_id"]
        assert client.post(f"/api/translation/{run_id}/interrupt").status_code == 200
    restored = TranslationStateManager(manager)
    with make_app(restored).test_client() as client:
        replay = client.post("/api/translate", json=data)
        status = client.get(f"/api/translation/{run_id}")
    assert replay.status_code == 200
    assert replay.json["translation_id"] == run_id
    assert len(calls) == 1
    assert status.status_code == 200
    assert status.json["persisted_only"] is True
    assert status.json["pause_reason"] == "manual"
    assert manager.get_job(run_id)["config"]["_manual_pause_requested"] is True


def test_concurrent_replay_across_state_managers_launches_once(setup):
    make_app, _state, manager, calls = setup
    apps = [make_app(), make_app(TranslationStateManager(manager))]
    data = payload()

    def send(index):
        with apps[index % 2].test_client() as client:
            response = client.post("/api/translate", json=data)
            return response.status_code, response.json

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(send, range(12)))
    assert all(code == 200 for code, _ in results)
    assert len({body["translation_id"] for _, body in results}) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("value", ["../book", "", 42, [], None])
def test_invalid_submission_id_is_rejected_before_worker(setup, value):
    make_app, _, _, calls = setup
    with make_app().test_client() as client:
        response = client.post("/api/translate", json={**payload(), "submission_id": value})
    assert response.status_code == 400
    assert calls == []


def test_dispatch_failure_stays_resumable_without_retry_dispatch(setup):
    make_app, state, manager, calls = setup

    def fail(run_id, config):
        assert manager.get_job(run_id)
        calls.append(run_id)
        raise RuntimeError("Synthetic thread creation failure")

    data = payload()
    with make_app(worker=fail).test_client() as client:
        first = client.post("/api/translate", json=data)
        replay = client.post("/api/translate", json=data)
    assert first.status_code == 503
    run_id = first.json["translation_id"]
    assert replay.status_code == 200
    assert replay.json["translation_id"] == run_id
    assert manager.get_job(run_id)["status"] == "error"
    assert state.get_translation(run_id)["status"] == "error"
    assert len(calls) == 1


def test_reserved_but_undispatched_job_is_recoverable_after_crash(setup):
    make_app, _, manager, calls = setup
    with make_app().test_client() as client:
        run_id = client.post('/api/translate', json=payload()).json['translation_id']
    assert run_id in manager.db.get_stale_active_job_ids('new-server-session')
    assert manager.db.reset_running_jobs('new-server-session') == 1
    assert run_id in {job['translation_id'] for job in manager.get_resumable_jobs()}
    assert len(calls) == 1


def test_duplicate_checkpoint_does_not_replace_preserved_source(setup, tmp_path):
    _, _, manager, _ = setup
    source = tmp_path / 'source.txt'
    source.write_text('The original book.')
    first = {'file_type': 'txt', 'model': 'original'}
    assert manager.start_job('book', 'txt', first, str(source))
    source.write_text('An unrelated book.')
    assert manager.start_job('book', 'txt', {'model': 'different'}, str(source)) is False
    assert manager.get_job('book')['config']['model'] == 'original'
    assert Path(first['preserved_input_path']).read_text() == 'The original book.'


def test_inline_text_resume_uses_saved_source(setup):
    make_app, _, manager, calls = setup
    data = payload()
    with make_app().test_client() as client:
        run_id = client.post('/api/translate', json=data).json['translation_id']
        assert client.post(f'/api/translation/{run_id}/interrupt').status_code == 200
        manager.mark_interrupted(run_id)
        restored = TranslationStateManager(manager)
        with make_app(restored).test_client() as after_restart:
            response = after_restart.post(f'/api/resume/{run_id}', json={})
    assert response.status_code == 200
    assert calls[-1][1]['text'] == data['text']
    assert calls[-1][1]['resume_from_index'] == 0
    assert calls[-1][1]['is_resume'] is True


@pytest.mark.asyncio
async def test_pause_during_worker_setup_preserves_reserved_job(setup, monkeypatch, tmp_path):
    from src.api import handlers
    from src.api.blueprints.translation_routes import _persist_manual_pause_request

    monkeypatch.chdir(tmp_path)
    make_app, state, manager, calls = setup
    with make_app().test_client() as client:
        run_id = client.post('/api/translate', json=payload()).json['translation_id']

    def pause(_config):
        _persist_manual_pause_request(state, run_id)
        state.set_interrupted(run_id, True)

    monkeypatch.setattr(handlers, '_ensure_layout_sanitizer_options', pause)
    monkeypatch.setattr(handlers, 'emit_update', lambda *_args, **_kwargs: None)
    await handlers.perform_actual_translation(run_id, calls[0][1], state, str(tmp_path / 'outputs'), None)
    assert manager.get_job(run_id)['config']['_manual_pause_requested']
    assert manager.get_job(run_id)['status'] == 'interrupted'
    assert state.get_translation(run_id)['status'] == 'interrupted'
    assert manager.load_checkpoint(run_id)['chunks'] == []


@pytest.mark.parametrize('status', ['queued', 'running', 'provider_wait', 'pricing_wait', 'rate_limited'])
def test_exact_submission_interrupt_persists_manual_intent_for_active_states(setup, status):
    make_app, state, manager, _ = setup
    with make_app().test_client() as client:
        run_id = client.post('/api/translate', json=payload()).json['translation_id']
        state.set_translation_field(run_id, 'status', status)
        result = client.post(f'/api/translation/{run_id}/interrupt')
    assert result.status_code == 200
    assert state.is_interrupted(run_id)
    assert manager.get_job(run_id)['config']['_manual_pause_requested']
