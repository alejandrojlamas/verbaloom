"""Provider selection must not redirect explicit CLI endpoints."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "provider, override, expected",
    [
        ("openai", None, "https://api.openai.com/v1/chat/completions"),
        ("ollama", None, "http://localhost:11434/api/generate"),
        ("openai", "http://localhost:8080/v1", "http://localhost:8080/v1"),
        ("openai", "http://localhost:8080/api/generate", "http://localhost:8080/api/generate"),
        ("ollama", "http://localhost:11435/api/generate", "http://localhost:11435/api/generate"),
    ],
)
def test_legacy_cli_resolves_endpoint_before_pipeline_config(
    monkeypatch, provider, override, expected
):
    from src import config
    from src.core import pipeline_cli

    monkeypatch.setattr(config, "API_ENDPOINT", "http://localhost:11434/api/generate")
    monkeypatch.setattr(config, "OPENAI_API_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    seen = {}

    class ConfigurationReached(Exception):
        pass

    def capture(args, job):
        seen.update(job)
        assert args.api_endpoint == job["llm_api_endpoint"]
        raise ConfigurationReached

    monkeypatch.setattr(pipeline_cli, "configure_cli_pipeline", capture)
    main = runpy.run_path(str(Path(__file__).resolve().parents[2] / "translate.py"))["main"]
    args = ["--input", "unused.txt", "--provider", provider]
    if override is not None:
        args.extend(["--api_endpoint", override])
    with pytest.raises(ConfigurationReached):
        main(args)
    assert seen["llm_api_endpoint"] == expected
    assert seen["llm_provider"] == provider


def test_openai_cli_uses_configured_compatible_server(monkeypatch):
    from src import config

    monkeypatch.setattr(config, "OPENAI_API_ENDPOINT", "http://localhost:8080/v1")
    resolve = runpy.run_path(str(Path(__file__).resolve().parents[2] / "translate.py"))["endpoint_for_provider"]
    assert resolve("openai", None) == "http://localhost:8080/v1"
