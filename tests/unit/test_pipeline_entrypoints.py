"""Characterize persisted role configuration at the HTTP/job boundaries."""

from copy import deepcopy

import pytest
from flask import Flask

from src.api.blueprints.translation_routes import (
    _apply_resume_overrides,
    create_translation_blueprint,
)
from src.core.pipeline_config import apply_pipeline, pipeline_preset, validate_pipeline


@pytest.fixture
def app_ctx():
    with Flask(__name__).app_context():
        yield


def saved_config():
    config = {
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "deepseek_api_key": "YOUR_DEEPSEEK_KEY_HERE",
        "openai_api_key": "YOUR_OPENAI_KEY_HERE",
        "prompt_options": {"target_locale": "es-ES"},
    }
    pipeline = pipeline_preset("compatible", config)
    pipeline["budget"] = {"limit_usd": "5", "max_calls": 100}
    apply_pipeline(config, pipeline)
    return config


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"preset": []},
    ],
)
def test_invalid_json_types_are_validation_errors(change):
    with pytest.raises(ValueError):
        validate_pipeline({**saved_config()["pipeline"], **change})


@pytest.mark.parametrize("field", ["provider", "model", "reasoning_effort"])
def test_invalid_role_json_types_are_validation_errors(field):
    pipeline = saved_config()["pipeline"]
    pipeline["roles"]["audit"][field] = []
    with pytest.raises(ValueError):
        validate_pipeline(pipeline)


def test_resume_rejects_endpoint_override_without_mutating_checkpoint(app_ctx):
    config = saved_config()
    before = deepcopy(config)
    result = _apply_resume_overrides(
        config, {"llm_api_endpoint": "https://example.invalid/v1"}
    )
    assert result is not None and result[1] == 400
    assert config == before


def test_resume_rejects_budget_reset_without_mutation(app_ctx):
    config = saved_config()
    before = deepcopy(config)
    override = deepcopy(config["pipeline"])
    override["budget"]["limit_usd"] = "6"
    assert _apply_resume_overrides(config, {"pipeline": override})[1] == 400
    assert config == before


def test_resume_explicit_primary_key_does_not_replace_secondary_key(
    app_ctx, monkeypatch
):
    config = saved_config()
    replacement = pipeline_preset("openai_equivalent", config)
    replacement["budget"] = config["pipeline"]["budget"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        _apply_resume_overrides(
            config,
            {
                "pipeline": replacement,
                "llm_provider": "openai",
                "model": "gpt-6-luna",
                "api_key": "YOUR_REPLACEMENT_KEY_HERE",
            },
        )
        is None
    )
    assert config["openai_api_key"] == "YOUR_REPLACEMENT_KEY_HERE"
    assert config["deepseek_api_key"] == "YOUR_DEEPSEEK_KEY_HERE"
    assert config["prompt_options"]["target_locale"] == "es-ES"


def test_resume_missing_secondary_key_does_not_partially_change_config(
    app_ctx, monkeypatch
):
    config = saved_config()
    config.pop("openai_api_key")
    before = deepcopy(config)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    override = deepcopy(config["pipeline"])
    override["roles"]["audit"] = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "reasoning_effort": "none",
        "max_output_tokens": 8192,
    }
    assert _apply_resume_overrides(config, {"pipeline": override})[1] == 400
    assert config == before


def test_resume_uses_saved_roles_instead_of_new_global_defaults(app_ctx, monkeypatch):
    config = saved_config()
    pipeline = deepcopy(config["pipeline"])
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-6.1-terra")
    assert _apply_resume_overrides(config, None) is None
    assert config["pipeline"] == pipeline
    assert config["llm_provider"] == "deepseek"


@pytest.mark.parametrize("has_secondary_key", [True, False])
def test_start_saves_effective_roles_and_validates_all_keys_before_worker(
    tmp_path, monkeypatch, has_secondary_key
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "src.api.blueprints.translation_routes.get_deepseek_pricing_status",
        lambda: SimpleNamespace(disabled=False),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    created, started = [], []
    state = SimpleNamespace(create_translation=lambda *args: created.append(args))
    app = Flask(__name__)
    app.register_blueprint(
        create_translation_blueprint(
            state, lambda *args: started.append(args), output_dir=tmp_path
        )
    )
    config = saved_config()
    config["pipeline"]["roles"]["audit"] = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "reasoning_effort": "low",
        "max_output_tokens": 4096,
    }
    request = {
        **config,
        "text": "The door was closed.",
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": "fixture.txt",
        "file_type": "txt",
    }
    if not has_secondary_key:
        request.pop("openai_api_key")
    response = app.test_client().post("/api/translate", json=request)
    if not has_secondary_key:
        assert response.status_code == 400
        assert created == started == []
        return
    assert response.status_code == 200
    assert len(started) == len(created) == 1
    effective = started[0][1]
    assert effective["pipeline"] == config["pipeline"]
    assert effective["prompt_options"]["target_locale"] == "es-ES"
    assert effective["openai_api_key"] == config["openai_api_key"]
    assert effective["deepseek_api_key"] == config["deepseek_api_key"]
    assert b"YOUR_" not in response.data
    assert response.get_json()["config_received"]["pipeline"] == effective["pipeline"]
