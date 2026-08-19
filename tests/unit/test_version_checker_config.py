import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_version_checker_defaults_to_custom_fork(tmp_path):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import src.utils.version_checker as v; "
                "print(v.GITHUB_REPO_OWNER); "
                "print(v.GITHUB_REPO_NAME)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
    ).strip().splitlines()

    assert output == ["alejandrojlamas", "verbaloom"]


def test_version_checker_repo_can_be_overridden(tmp_path):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "VERBALOOM_UPDATE_REPO_OWNER": "example-owner",
        "VERBALOOM_UPDATE_REPO_NAME": "example-repo",
    }
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import src.utils.version_checker as v; "
                "print(v.GITHUB_REPO_OWNER); "
                "print(v.GITHUB_REPO_NAME)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
    ).strip().splitlines()

    assert output == ["example-owner", "example-repo"]


def test_version_checker_accepts_legacy_repo_environment_aliases(tmp_path):
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(REPO_ROOT),
        "TBL_UPDATE_REPO_OWNER": "legacy-owner",
        "TBL_UPDATE_REPO_NAME": "legacy-repo",
    }
    output = subprocess.check_output(
        [
            sys.executable,
            "-c",
            (
                "import src.utils.version_checker as v; "
                "print(v.GITHUB_REPO_OWNER); "
                "print(v.GITHUB_REPO_NAME)"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
    ).strip().splitlines()

    assert output == ["legacy-owner", "legacy-repo"]


def test_version_checker_handles_repo_without_releases(monkeypatch):
    from src.utils import version_checker

    version_checker.invalidate_cache()
    monkeypatch.setattr(version_checker, "get_current_version", lambda: "1.4.6")
    monkeypatch.setattr(
        version_checker,
        "_fetch_latest_from_github",
        lambda timeout=5.0: {
            "tag_name": "",
            "name": "",
            "html_url": "https://github.com/example/repo/releases",
            "body": "",
            "published_at": "",
            "source": "none",
        },
    )

    result = version_checker.check_for_update(force=True)

    assert result["current"] == "1.4.6"
    assert result["latest"] == ""
    assert result["update_available"] is False
    assert "error" not in result
