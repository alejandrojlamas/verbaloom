from __future__ import annotations

import os
import sys

from src.utils.branding import (
    DISPLAY_NAME,
    ENV_PREFIX,
    REPOSITORY_NAME,
    REPOSITORY_URL,
    ROUTE_PREFIX,
    TECHNICAL_PREFIX,
    env_value,
)


def test_canonical_brand_contract():
    assert DISPLAY_NAME == "VerbaLoom"
    assert TECHNICAL_PREFIX == "verbaloom"
    assert ENV_PREFIX == "VERBALOOM_"
    assert ROUTE_PREFIX == "/verbaloom"
    assert REPOSITORY_NAME == "verbaloom"
    assert REPOSITORY_URL == "https://github.com/alejandrojlamas/verbaloom"


def test_canonical_environment_value_precedes_legacy_alias(monkeypatch):
    monkeypatch.setenv("TBL_SAMPLE_SETTING", "legacy")
    monkeypatch.setenv("VERBALOOM_SAMPLE_SETTING", "canonical")
    assert env_value("SAMPLE_SETTING", "default") == "canonical"

    monkeypatch.setenv("VERBALOOM_SAMPLE_SETTING", "")
    assert env_value("SAMPLE_SETTING", "default") == ""


def test_legacy_environment_value_is_still_readable(monkeypatch):
    monkeypatch.delenv("VERBALOOM_SAMPLE_SETTING", raising=False)
    monkeypatch.setenv("TBL_SAMPLE_SETTING", "legacy")
    assert env_value("SAMPLE_SETTING", "default") == "legacy"


def test_usage_data_directory_prefers_canonical_and_accepts_legacy(monkeypatch, tmp_path):
    from src.core.usage.store import _default_data_dir

    canonical = tmp_path / "canonical"
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("TBL_DATA_DIR", str(legacy))
    monkeypatch.setenv("VERBALOOM_DATA_DIR", str(canonical))
    assert _default_data_dir() == canonical

    monkeypatch.delenv("VERBALOOM_DATA_DIR")
    assert _default_data_dir() == legacy


def test_generator_and_telemetry_emit_only_verbaloom():
    from src.config import GENERATOR_NAME, GENERATOR_SOURCE
    from src.utils.telemetry import TelemetryCollector

    collector = TelemetryCollector()
    payload = {
        "generator_name": GENERATOR_NAME,
        "generator_source": GENERATOR_SOURCE,
        "headers": collector.get_client_headers(),
        "metadata": collector.get_generation_metadata(),
    }
    serialized = str(payload)

    assert GENERATOR_NAME == "VerbaLoom"
    assert GENERATOR_SOURCE == REPOSITORY_URL
    assert "VerbaLoom" in serialized
    assert REPOSITORY_URL in serialized
    assert "TranslateBook" not in serialized
    assert "TBL" not in serialized


def test_srt_attribution_emits_only_verbaloom():
    from src.core.srt_processor import SRTProcessor

    output = SRTProcessor().reconstruct_srt(
        [
            {
                "number": "1",
                "start_time": "00:00:00,000",
                "end_time": "00:00:01,000",
                "text": "Hello.",
            }
        ]
    )

    assert "Translated with VerbaLoom" in output
    assert REPOSITORY_URL in output
    assert "TranslateBook" not in output
    assert "(TBL)" not in output


def test_docx_equation_parser_accepts_legacy_but_generates_canonical_markers():
    from src.core.docx.converter import (
        _EQ_MARKER_PREFIX,
        _EQ_MARKER_SUFFIX,
        _EQ_MARKER_REGEX,
        _LEGACY_EQ_MARKER_REGEX,
    )

    canonical = f"{_EQ_MARKER_PREFIX}7{_EQ_MARKER_SUFFIX}"
    assert canonical == "__VERBALOOMEQ7EQVERBALOOM__"
    assert _EQ_MARKER_REGEX.fullmatch(canonical)
    assert _LEGACY_EQ_MARKER_REGEX.fullmatch("__TTBLLEQ7EQTTBLL__")


def test_main_blueprint_exposes_canonical_route():
    from flask import Flask

    from src.api.blueprints.config_routes import create_config_blueprint

    app = Flask(__name__)
    app.register_blueprint(create_config_blueprint(server_session_id="1234567890"))
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    assert "/verbaloom" in routes
    assert "/verbaloom/" in routes


def test_launcher_migrates_legacy_bundle_data_directory(tmp_path, monkeypatch):
    import launcher

    executable = tmp_path / "VerbaLoom"
    executable.write_text("", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    legacy_dir = tmp_path / "TranslateBook_Data"
    legacy_dir.mkdir()
    (legacy_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    previous_cwd = os.getcwd()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_dir), raising=False)
    try:
        launcher.setup_working_directory()
    finally:
        os.chdir(previous_cwd)

    canonical_dir = tmp_path / "VerbaLoom_Data"
    assert (canonical_dir / "checkpoint.json").read_text(encoding="utf-8") == "{}"
    assert "# VerbaLoom Configuration" in (canonical_dir / ".env").read_text(
        encoding="utf-8"
    )
