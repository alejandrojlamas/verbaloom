import os
import subprocess
import sys
from pathlib import Path

from src.core.translator import _source_aware_editorial_guard_mode, split_chunk_for_retry


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cloud_provider_default_without_override(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "deepseek",
        "CLOUD_MAX_TOKENS_PER_CHUNK": "1400",
    }
    value = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print(c.MAX_TOKENS_PER_CHUNK)"],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip()

    assert value == "1400"


def test_ollama_keeps_small_chunk_default(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "ollama",
    }
    value = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print(c.MAX_TOKENS_PER_CHUNK)"],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip()

    assert value == "450"


def test_explicit_chunk_override_still_wins(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "deepseek",
        "MAX_TOKENS_PER_CHUNK": "777",
    }
    value = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print(c.MAX_TOKENS_PER_CHUNK)"],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip()

    assert value == "777"


def test_legacy_450_env_does_not_block_cloud_default(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "deepseek",
        "MAX_TOKENS_PER_CHUNK": "450",
        "CLOUD_MAX_TOKENS_PER_CHUNK": "1400",
    }
    value = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print(c.MAX_TOKENS_PER_CHUNK)"],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip()

    assert value == "1400"


def test_legacy_450_can_be_forced_for_cloud_provider(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "deepseek",
        "MAX_TOKENS_PER_CHUNK": "450",
        "MAX_TOKENS_PER_CHUNK_FORCE": "true",
        "CLOUD_MAX_TOKENS_PER_CHUNK": "1400",
    }
    value = subprocess.check_output(
        [sys.executable, "-c", "import src.config as c; print(c.MAX_TOKENS_PER_CHUNK)"],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip()

    assert value == "450"


def test_runtime_chunk_default_uses_job_provider_even_when_env_provider_is_ollama(tmp_path):
    run_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "LLM_PROVIDER": "ollama",
        "CLOUD_MAX_TOKENS_PER_CHUNK": "1400",
    }
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import src.config as c; "
                "print(c.MAX_TOKENS_PER_CHUNK); "
                "print(c.max_tokens_per_chunk_for_provider('deepseek')); "
                "print(c.max_tokens_per_chunk_for_provider('ollama'))"
            ),
        ],
        cwd=tmp_path,
        env=run_env,
        text=True,
    ).strip().splitlines()

    assert output == ["450", "1400", "450"]


def test_source_aware_guard_defaults_to_alerted():
    assert _source_aware_editorial_guard_mode({}) == "alerted"
    assert _source_aware_editorial_guard_mode({"source_aware_editorial_guard_mode": "bogus"}) == "alerted"
    assert _source_aware_editorial_guard_mode({"source_aware_editorial_guard_mode": "always"}) == "always"
    assert _source_aware_editorial_guard_mode({"source_aware_editorial_guard_mode": "off"}) == "off"


def test_retry_split_prefers_sentence_boundary():
    text = (
        "Primera oración con valor 3.14 y una explicación suficiente. "
        "Segunda oración que debe quedar completa después del corte. "
        "Tercera oración para mantener margen de búsqueda."
    )

    first, second = split_chunk_for_retry(text, 0.45)

    assert first.endswith("suficiente.")
    assert second.startswith("Segunda oración")
    assert "3.14" in first
