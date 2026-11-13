"""Public settings expose the shared role contract, never credentials."""

import pytest
from flask import Flask

from src.api.blueprints.config_routes import create_config_blueprint
from src.core.pipeline_config import ROLES, pipeline_preset


def client():
    app = Flask(__name__)
    app.register_blueprint(create_config_blueprint())
    return app.test_client()


@pytest.mark.parametrize("preset", [
    "compatible", "openai_equivalent", "deepseek_source_audited", "openai_source_audited",
])
def test_settings_resolve_the_same_core_presets(preset):
    baseline = {"llm_provider": "deepseek", "model": "deepseek-v4-flash"}
    response = client().post("/api/pipeline/configure", json={"preset": preset, "baseline": baseline})
    assert response.status_code == 200
    assert response.get_json()["pipeline"] == pipeline_preset(preset, baseline)
    assert response.headers["Cache-Control"] == "no-store"


def test_catalog_is_complete_and_has_no_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
    response = client().get("/api/pipeline/catalog")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["roles"] == list(ROLES)
    assert "gpt-6-luna" in payload["models"]["openai"]
    assert "deepseek-v4-pro" in payload["models"]["deepseek"]
    assert "YOUR_API_KEY_HERE" not in response.get_data(as_text=True)


@pytest.mark.parametrize("bad", [
    [], {"preset": "unknown"}, {"pipeline": {"api_key": "YOUR_API_KEY_HERE"}},
    {"preset": "compatible", "baseline": []},
])
def test_invalid_settings_fail_before_network_or_job_creation(bad):
    assert client().post("/api/pipeline/configure", json=bad).status_code == 400


def test_missing_role_key_is_an_explicit_nonbillable_preflight(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pipeline = pipeline_preset("openai_equivalent", {"llm_provider": "deepseek", "model": "deepseek-v4-pro"})
    response = client().post("/api/pipeline/preflight", json={"pipeline": pipeline})
    assert response.status_code == 200
    assert response.get_json() == {
        "status": "missing_credentials", "provider": "openai",
        "generation_tested": False, "balance": None,
    }


def test_preflight_uses_only_the_selected_providers_key(monkeypatch):
    from src.api.blueprints import pipeline_routes

    seen = []

    async def fake_openai(**kwargs):
        seen.append(kwargs)
        return {"status": "models_visible", "models": [], "generation_tested": False, "balance": None}

    monkeypatch.setattr(pipeline_routes, "preflight_openai", fake_openai)
    pipeline = pipeline_preset("openai_equivalent", {"llm_provider": "deepseek", "model": "deepseek-v4-pro"})
    response = client().post("/api/pipeline/preflight", json={
        "pipeline": pipeline, "openai_api_key": "YOUR_OPENAI_API_KEY_HERE",
        "deepseek_api_key": "YOUR_DEEPSEEK_API_KEY_HERE",
    })
    assert response.status_code == 200
    assert response.get_json()["status"] == "models_visible"
    assert seen[0]["api_key"] == "YOUR_OPENAI_API_KEY_HERE"
    assert set(seen[0]["models"]) == {"gpt-6-luna", "gpt-6.1-terra"}
    assert "API_KEY_HERE" not in response.get_data(as_text=True)


def test_profile_preflight_does_not_require_unused_translation_provider(monkeypatch):
    from src.api.blueprints import pipeline_routes

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []

    async def fake_deepseek(**kwargs):
        seen.append(kwargs)
        return {"status": "models_visible", "models": [], "generation_tested": False, "balance": None}

    async def unused_openai(**kwargs):
        pytest.fail("Profile preparation must not preflight translation-only models")

    monkeypatch.setattr(pipeline_routes, "preflight_deepseek", fake_deepseek)
    monkeypatch.setattr(pipeline_routes, "preflight_openai", unused_openai)
    baseline = {"llm_provider": "deepseek", "model": "deepseek-v4-pro"}
    pipeline = pipeline_preset("openai_equivalent", baseline)
    deepseek = pipeline_preset("compatible", baseline)
    selected = ["profile_prepare", "profile_review"]
    for role in selected:
        pipeline["roles"][role] = deepseek["roles"][role]
    response = client().post("/api/pipeline/preflight", json={
        "pipeline": pipeline, "roles": selected, "deepseek_api_key": "YOUR_API_KEY_HERE",
    })
    assert response.status_code == 200
    assert response.get_json()["status"] == "models_visible"
    assert response.get_json()["roles"] == selected
    assert len(seen) == 1
    assert set(seen[0]["models"]) == {pipeline["roles"][role]["model"] for role in selected}


@pytest.mark.parametrize("roles", [[], "translate", ["unknown"], ["translate", "translate"], [None], [[]]])
def test_preflight_rejects_invalid_scope_before_network(monkeypatch, roles):
    from src.api.blueprints import pipeline_routes

    async def unexpected(**kwargs):
        pytest.fail("Invalid scopes must not contact a provider")

    monkeypatch.setattr(pipeline_routes, "preflight_openai", unexpected)
    pipeline = pipeline_preset("openai_equivalent", {"llm_provider": "deepseek", "model": "deepseek-v4-pro"})
    response = client().post("/api/pipeline/preflight", json={
        "pipeline": pipeline, "roles": roles, "openai_api_key": "YOUR_API_KEY_HERE",
    })
    assert response.status_code == 400
