"""The discounted tier must be an explicit installation choice."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, ["default", 600]),
        ({"OPENAI_SERVICE_TIER": " Flex ", "OPENAI_FLEX_COOLDOWN_SECONDS": "90"}, ["flex", 90]),
    ],
)
def test_service_tier_environment_without_reading_installed_env(tmp_path, settings, expected):
    env = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    env.update(settings)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", "import json; from src import config; "
         "print(json.dumps([config.OPENAI_SERVICE_TIER, config.OPENAI_FLEX_COOLDOWN_SECONDS]))"],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    assert json.loads(result.stdout) == expected
