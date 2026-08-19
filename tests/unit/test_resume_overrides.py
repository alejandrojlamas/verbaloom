"""Unit tests for the resume model/provider override logic (issue #183).

Covers `_apply_resume_overrides`: backward compatibility (empty body), field
merging, generic API-key routing through `_resolve_api_key`, and the
key/endpoint validation guards.
"""
import pytest
from flask import Flask

from src.api.blueprints.translation_routes import _apply_resume_overrides


@pytest.fixture
def app_ctx():
    """`_apply_resume_overrides` calls jsonify on failure, which needs a context."""
    app = Flask(__name__)
    with app.app_context():
        yield


def _base_config():
    return {
        'model': 'llama3',
        'llm_provider': 'ollama',
        'llm_api_endpoint': 'http://localhost:11434/api/generate',
    }


def test_empty_overrides_leaves_config_untouched(app_ctx):
    config = _base_config()
    snapshot = dict(config)
    assert _apply_resume_overrides(config, {}) is None
    assert config == snapshot


def test_none_overrides_is_noop(app_ctx):
    config = _base_config()
    assert _apply_resume_overrides(config, None) is None


def test_simple_model_provider_override(app_ctx, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-from-env')
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'model': 'anthropic/claude-sonnet-4',
        'llm_provider': 'OpenRouter',  # case-insensitive
    })
    assert err is None
    assert config['model'] == 'anthropic/claude-sonnet-4'
    assert config['llm_provider'] == 'openrouter'  # normalized


def test_api_key_routed_to_provider_field(app_ctx):
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': 'real-key-123',
    })
    assert err is None
    assert config['gemini_api_key'] == 'real-key-123'


def test_api_key_use_env_sentinel_resolves_from_env(app_ctx, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'env-gemini-key')
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
        'api_key': '__USE_ENV__',
    })
    assert err is None
    assert config['gemini_api_key'] == 'env-gemini-key'


def test_cloud_provider_without_key_is_rejected(app_ctx, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    config = _base_config()
    result = _apply_resume_overrides(config, {
        'llm_provider': 'gemini',
        'model': 'gemini-2.0-flash',
    })
    assert result is not None
    _response, status = result
    assert status == 400


def test_endpoint_provider_without_endpoint_is_rejected(app_ctx, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai')
    config = _base_config()
    config['llm_api_endpoint'] = ''  # cleared
    result = _apply_resume_overrides(config, {
        'llm_provider': 'openai',
        'model': 'gpt-4o',
    })
    assert result is not None
    _response, status = result
    assert status == 400


def test_invalid_context_window_is_rejected(app_ctx):
    config = _base_config()
    result = _apply_resume_overrides(config, {'context_window': 'not-a-number'})
    assert result is not None
    _response, status = result
    assert status == 400


def test_multi_key_string_is_preserved(app_ctx):
    """Comma-separated keys must reach the config unchanged for the rotation pool."""
    config = _base_config()
    err = _apply_resume_overrides(config, {
        'llm_provider': 'openrouter',
        'model': 'x',
        'api_key': 'sk-or-1,sk-or-2,sk-or-3',
    })
    assert err is None
    assert config['openrouter_api_key'] == 'sk-or-1,sk-or-2,sk-or-3'
