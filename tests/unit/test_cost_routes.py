from types import SimpleNamespace

from flask import Flask

from src.api.blueprints import cost_routes
from src.api.blueprints.cost_routes import create_cost_blueprint


def _client(tmp_path, monkeypatch, *, tier="peak", guard_enabled=True):
    status = SimpleNamespace(
        pricing_tier=tier,
        enabled=guard_enabled,
        source_url="https://api-docs.deepseek.com/quick_start/pricing/",
    )
    monkeypatch.setattr(cost_routes, "get_deepseek_pricing_status", lambda: status)
    app = Flask(__name__)
    app.register_blueprint(create_cost_blueprint(tmp_path))
    return app.test_client()


def test_pricing_defaults_exposes_two_current_deepseek_models(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        payload = client.get("/api/pricing/defaults").get_json()

    deepseek = payload["pricing"]["deepseek"]
    assert deepseek["deepseek-flash"]["input_cache_miss"] == 0.15
    assert deepseek["deepseek-v4-pro"]["output"] == 1.98
    assert payload["pricing_context"]["deepseek"] == {
        "current_tier": "peak",
        "effective_estimate_tier": "off_peak",
        "off_peak_guard_enabled": True,
        "source_url": "https://api-docs.deepseek.com/quick_start/pricing/",
    }


def test_deepseek_estimate_uses_off_peak_cache_rates_when_job_will_wait(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/cost/estimate",
            json={
                "provider": "deepseek",
                "model": "deepseek-flash",
                "text": "A complete sentence for translation. " * 80,
                "src_lang": "English",
                "tgt_lang": "Spanish",
                "options": {"refine": True},
            },
        )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["pricing_tier"] == "off_peak"
    assert payload["pricing_current_tier"] == "peak"
    assert payload["pricing_waits_for_off_peak"] is True
    assert payload["pricing_used"]["input_cache_hit_per_million"] == 0.003
    assert payload["pricing_used"]["input_cache_miss_per_million"] == 0.15
    assert payload["input_tokens"] == payload["input_tokens_per_pass"] * 2


def test_deepseek_estimate_uses_peak_rates_when_guard_is_disabled(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, tier="peak", guard_enabled=False) as client:
        response = client.post(
            "/api/cost/estimate",
            json={
                "provider": "deepseek",
                "model": "deepseek-v4-pro",
                "text": "A complete sentence for translation. " * 20,
            },
        )

    payload = response.get_json()
    assert payload["pricing_tier"] == "peak"
    assert payload["pricing_waits_for_off_peak"] is False
    assert payload["pricing_used"]["input_cache_miss_per_million"] == 1.32
    assert payload["pricing_used"]["output_per_million"] == 3.96
