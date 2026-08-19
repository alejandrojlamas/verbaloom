"""
Regression test for the MAX_TRANSLATION_ATTEMPTS default.

Raised from 2 to 3 now that every LLM provider shares a circuit breaker
(src/core/adapters/retry_manager.py, wired into src/core/llm/providers/*.py):
a sustained provider outage fails fast instead of paying for the extra
attempt on every remaining chunk of the book, so a slightly higher
per-chunk retry budget for genuine one-off blips is safe.
"""
import importlib
import os

from src import config


def test_max_translation_attempts_defaults_to_three(monkeypatch, tmp_path):
    """Isolated from this machine's real .env: config.py resolves its
    directory from the current working directory at import time
    (`_config_dir = Path.cwd()`), so a plain env-var delete is not enough
    -- a real local .env (like this repo's own) would still get re-read
    from disk on reload. Point cwd at an empty temp dir with no .env at
    all to test the pure code default."""
    original_cwd = os.getcwd()
    monkeypatch.delenv("MAX_TRANSLATION_ATTEMPTS", raising=False)
    monkeypatch.delenv("MAX_RETRIES", raising=False)

    try:
        os.chdir(tmp_path)
        importlib.reload(config)
        assert config.MAX_TRANSLATION_ATTEMPTS == 3
    finally:
        os.chdir(original_cwd)
        importlib.reload(config)
