import json

from src.core.quality_assurance import cli
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization import fake_llm


def test_strict_cli_translates_validates_reports_and_publishes_without_network(
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    fake_llm.install(monkeypatch)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    monkeypatch.setattr(cli, "CheckpointManager", lambda: manager)
    source = tmp_path / "book.txt"
    output = tmp_path / "book-fr.txt"
    report_root = tmp_path / "quality"
    source.write_text(
        "The quick brown fox jumps over the lazy dog. This document contains "
        "enough material and several sentences. Every paragraph preserves meaning.\n\n"
        "A translator runs carefully while the chunker split the document into "
        "one more paragraph with enough material for characterization.",
        encoding="utf-8",
    )

    exit_code = cli.main(
        [
            "--report-root",
            str(report_root),
            "translate",
            str(source),
            "--output",
            str(output),
            "--provider",
            "poe",
            "--model",
            "fake-echo",
            "--source-language",
            "English",
            "--target-language",
            "French",
            "--target-locale",
            "fr",
            "--run-id",
            "strict-e2e",
            "--allow-warnings",
        ]
    )

    assert exit_code == 0
    assert output.is_file()
    assert output.read_text(encoding="utf-8") != source.read_text(encoding="utf-8")
    report_path = report_root / "strict-e2e" / "translation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["publishable"] is True
    assert report["status"] in {"PASSED", "PASSED_WITH_WARNINGS"}
    assert len(report["quality_gates"]) == 10
    assert report["coverage"]["translatable"] == report["coverage"]["translated"]
    assert report["coverage"]["translatable"] == report["coverage"]["reviewed"]
    assert report["coverage"]["translatable"] == report["coverage"]["audited"]
    assert report["coverage"]["translatable"] == report["coverage"]["approved"]
    assert report["coverage"]["translatable"] == report["coverage"]["exported"]
    assert manager.get_job("strict-e2e")["status"] == "completed"
    persisted_options = manager.get_job("strict-e2e")["config"]["prompt_options"]
    assert "_fidelity_report" not in persisted_options
    assert "Error updating job config" not in capsys.readouterr().out
