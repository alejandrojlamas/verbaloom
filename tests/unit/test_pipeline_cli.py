"""CLI role configuration must reach the same runtime and durable budget as HTTP."""

import json
import runpy
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.core.pipeline_config import apply_pipeline, pipeline_preset
from src.core.pipeline_runtime import active_pipeline
from src.core.quality_assurance import cli
from src.core.usage.context import get_usage_context


def saved_config():
    config = {
        "llm_provider": "deepseek", "model": "deepseek-flash",
        "source_language": "Czech", "target_language": "Spanish",
        "prompt_options": {"target_locale": "es-ES"},
    }
    pipeline = pipeline_preset("openai_equivalent", config)
    pipeline["budget"] = {"limit_usd": "5", "max_calls": 30}
    apply_pipeline(config, pipeline)
    return config


@pytest.mark.parametrize("preset", [
    "compatible", "openai_equivalent", "deepseek_source_audited", "openai_source_audited",
])
def test_strict_parser_exposes_presets_and_finite_budget(preset):
    args = cli.build_parser().parse_args([
        "translate", "source.txt", "--pipeline-preset", preset,
        "--budget-usd", "5.00", "--max-calls", "30",
    ])
    assert args.pipeline_preset == preset
    assert args.budget_usd == "5.00"
    assert args.max_calls == 30


@pytest.mark.asyncio
async def test_resume_invocation_activates_saved_roles_and_usage_identity(tmp_path, monkeypatch):
    from src.core.llm import preflight
    from src.core.usage import store

    usage = store.TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr(store, "_DEFAULT_STORE", usage)
    monkeypatch.setattr(preflight, "preflight_openai", AsyncMock(return_value={"status": "models_visible"}))
    monkeypatch.setattr(cli, "OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    config = saved_config()
    before = deepcopy(config)

    async def operation(**kwargs):
        runtime = active_pipeline.get()
        assert runtime is not None, "CLI dropped the saved pipeline"
        assert runtime.config == before["pipeline"]
        assert runtime.budget_scope_ids == ("job:saved",)
        assert runtime.credentials == {"openai": "YOUR_OPENAI_KEY_HERE"}
        assert get_usage_context()["translation_id"] == "saved"
        assert kwargs["resume_from_index"] == 4
        assert kwargs["prompt_options"]["target_locale"] == "es-ES"
        return True

    monkeypatch.setattr(cli, "translate_file", operation)
    assert await cli._invoke_translation(
        Mock(), "saved", config, tmp_path / "source.txt", tmp_path / "staged.txt", resume_from=4,
    )
    assert active_pipeline.get() is None
    assert config == before


def test_cli_does_not_invent_a_spanish_variant():
    args = cli.build_parser().parse_args(["translate", "source.txt", "--target-language", "French"])
    assert args.target_locale == ""


def test_cli_role_override_and_cost_alias_share_core_contract():
    from src.core.pipeline_cli import configure_cli_pipeline

    args = cli.build_parser().parse_args([
        "translate", "source.txt", "--pipeline-preset", "compatible",
        "--role", "audit", "openai", "gpt-6-luna", "low", "2048",
        "--review-policy", "source_audited_v1", "--max-cost", "5", "--max-calls", "30",
    ])
    config = {"llm_provider": "deepseek", "model": "deepseek-flash"}
    result = configure_cli_pipeline(args, config)
    assert result["roles"]["audit"] == {
        "provider": "openai", "model": "gpt-6-luna", "reasoning_effort": "low", "max_output_tokens": 2048,
    }
    assert result["roles"]["translate"]["provider"] == "deepseek"
    assert result["budget"] == {"limit_usd": "5", "max_calls": 30}
    assert result["preset"] == "custom"
    assert result["epub_repair_policy"] == "localized_v1"


@pytest.mark.parametrize("flags", [
    ["--budget-usd", "5"], ["--max-calls", "30"],
    ["--budget-usd", "NaN", "--max-calls", "30"],
    ["--budget-usd", "5", "--max-calls", "0"],
    ["--role", "translate", "openai", "deepseek-flash", "none", "8192"],
    ["--role", "translate", "openai", "gpt-6-luna", "high", "8192"],
    ["--role", "translate", "openai", "gpt-6-luna", "none", "invalid"],
    ["--reviewer-model", "deepseek-v4-pro", "--pipeline-preset", "compatible"],
])
def test_invalid_options_never_partially_mutate_job(flags):
    from src.core.pipeline_cli import configure_cli_pipeline

    args = cli.build_parser().parse_args(["translate", "source.txt", *flags])
    config = {"llm_provider": "deepseek", "model": "deepseek-flash"}
    before = deepcopy(config)
    with pytest.raises(ValueError):
        configure_cli_pipeline(args, config)
    assert config == before


def test_json_snapshot_preserves_explicit_refine_role_and_rejects_secrets(tmp_path):
    from src.core.pipeline_cli import configure_cli_pipeline

    pipeline = saved_config()["pipeline"]
    pipeline["roles"]["refine"]["model"] = "gpt-6.1-terra"
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(pipeline))
    args = cli.build_parser().parse_args(["translate", "source.txt", "--pipeline-config", str(path)])
    config = {"refine_only": True}
    configure_cli_pipeline(args, config)
    assert config["model"] == "gpt-6.1-terra"
    assert config["pipeline"] == pipeline
    path.write_text(json.dumps({**pipeline, "api_key": "YOUR_API_KEY_HERE"}))
    with pytest.raises(ValueError, match="credential-free") as error:
        configure_cli_pipeline(args, config)
    assert "YOUR_API_KEY_HERE" not in str(error.value)


@pytest.mark.asyncio
async def test_cli_resume_cannot_replenish_http_attempt_budget(tmp_path, monkeypatch):
    from src.core.llm import preflight
    from src.core.llm.providers.openai import OpenAICompatibleProvider
    from src.core.llm.request_deadline import await_llm_call
    from src.core.usage import store
    from src.core.usage.budget import BudgetLimitError, SpendBudget

    usage = store.TokenUsageStore(tmp_path / "usage.db")
    monkeypatch.setattr(store, "_DEFAULT_STORE", usage)
    monkeypatch.setattr(preflight, "preflight_openai", AsyncMock(return_value={"status": "models_visible"}))
    monkeypatch.setattr(cli, "OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    config = saved_config()
    config["pipeline"]["budget"]["max_calls"] = 1
    posts = []

    class Client:
        async def post(self, url, **kwargs):
            posts.append(kwargs)
            assert url == "https://api.openai.com/v1/chat/completions"
            assert kwargs["json"]["model"] == "gpt-6-luna"
            assert kwargs["json"]["reasoning_effort"] == "none"
            return httpx.Response(200, request=httpx.Request("POST", url), json={
                "model": "gpt-6-luna",
                "choices": [{"finish_reason": "stop", "message": {"content": "Complete candidate"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            })

    monkeypatch.setattr(OpenAICompatibleProvider, "_get_client", AsyncMock(return_value=Client()))

    async def operation(**kwargs):
        provider = active_pipeline.get().client_for("translate")
        result = await await_llm_call(provider.generate, "Synthetic source", provider=provider, role="translate")
        return bool(result.content)

    monkeypatch.setattr(cli, "translate_file", operation)
    values = (Mock(), "same-run", config, tmp_path / "source.txt", tmp_path / "staged.txt")
    assert await cli._invoke_translation(*values, resume_from=0)
    with pytest.raises(BudgetLimitError, match="attempt budget"):
        await cli._invoke_translation(*values, resume_from=1)
    assert len(posts) == len(usage.events()) == 1
    assert usage.events()[0]["translation_id"] == "same-run"
    assert SpendBudget(usage).snapshot("job:same-run")["calls"] == 1
    assert active_pipeline.get() is None
    assert get_usage_context().get("translation_id") != "same-run"


@pytest.mark.parametrize("entrypoint", ["strict", "legacy"])
def test_preflight_only_does_not_create_a_job_or_generate(tmp_path, monkeypatch, capsys, entrypoint):
    from src.core import adapters
    from src.core.llm import preflight

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Synthetic source")
    check = AsyncMock(return_value={"status": "models_visible", "balance": None, "generation_tested": False})
    monkeypatch.setattr(preflight, "preflight_openai", check)
    monkeypatch.setenv("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE")
    forbidden = Mock(side_effect=AssertionError("Preflight must not create a checkpoint"))
    monkeypatch.setattr(cli, "CheckpointManager", forbidden)
    monkeypatch.setattr("src.persistence.checkpoint_manager.CheckpointManager", forbidden)
    operation = AsyncMock(side_effect=AssertionError("Preflight must not generate"))
    monkeypatch.setattr(cli, "translate_file", operation)
    monkeypatch.setattr(adapters, "translate_file", operation)
    flags = ["--provider", "deepseek", "--model", "deepseek-v4-pro",
             "--pipeline-preset", "openai_equivalent", "--preflight-only"]
    if entrypoint == "strict":
        assert cli.main(["translate", str(source), *flags]) == 0
    else:
        path = Path(__file__).resolve().parents[2] / "translate.py"
        assert runpy.run_path(str(path))["main"](["-i", str(source), *flags]) == 0
    check.assert_awaited_once()
    operation.assert_not_awaited()
    forbidden.assert_not_called()
    output = capsys.readouterr().out
    assert '"generation_tested": false' in output
    assert "YOUR_OPENAI_KEY_HERE" not in output
