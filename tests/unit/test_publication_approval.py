"""A publication approval applies to exact source/candidate bytes, not a path."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.core.quality_assurance import cli, publication, runner
from src.persistence.checkpoint_manager import CheckpointManager


def validated_pair(tmp_path):
    source = tmp_path / "source.txt"
    candidate = tmp_path / "candidate.txt"
    source.write_text("The traveler crossed the old bridge and returned to the house.")
    candidate.write_text("El viajero cruzo el puente antiguo y regreso a la casa.")
    quality = runner.run_quality_assurance(
        source_path=source, output_path=candidate, source_language="English",
        target_language="Spanish", run_id="fixture", report_root=tmp_path / "reports",
    )
    assert quality.publishable
    return source, candidate, quality


@pytest.mark.parametrize("mutation", [
    "candidate", "source", "missing_proof", "wrong_run", "truthy_flag", "stopped_job",
    "missing_checksum", "mismatched_checksum", "mismatched_status", "truthy_job_flag",
    "mismatched_generation", "missing_generation",
])
def test_export_rejects_stale_or_mismatched_approval(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(tmp_path)
    source, candidate, quality = validated_pair(tmp_path)
    prior = tmp_path / "delivered.txt"
    prior.write_text("Prior validated delivery")
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    close = Mock(wraps=manager.close)
    monkeypatch.setattr(manager, "close", close)
    monkeypatch.setattr(cli, "CheckpointManager", lambda: manager)
    if mutation == "candidate":
        candidate.write_text("A changed and unvalidated candidate")
    elif mutation == "source":
        source.write_text("A different original book")
    elif mutation == "missing_proof":
        quality.paths.export_report.unlink()
    elif mutation in {"wrong_run", "truthy_flag"}:
        report = json.loads(quality.paths.export_report.read_text())
        report["run_id" if mutation == "wrong_run" else "publishable"] = "different" if mutation == "wrong_run" else "false"
        quality.paths.export_report.write_text(json.dumps(report))
    elif mutation in {"missing_checksum", "mismatched_checksum", "mismatched_status", "mismatched_generation", "missing_generation"}:
        report = json.loads(quality.paths.export_report.read_text())
        if mutation == "missing_checksum":
            report.pop("validation_artifacts")
        elif mutation == "mismatched_checksum":
            report["output_sha256"] = "0" * 64
        elif mutation == "mismatched_status":
            report["status"] = "PASSED_WITH_WARNINGS"
        elif mutation == "mismatched_generation":
            report["generated_at"] = "2000-01-01T00:00:00+00:00"
        else:
            report.pop("generated_at")
        quality.paths.export_report.write_text(json.dumps(report))
    elif mutation in {"stopped_job", "truthy_job_flag"}:
        manager.start_job("fixture", "txt", {
            "qa_staging_output": str(candidate), "output_filepath": str(prior),
            "quality_assurance_result": {
                "publishable": "false" if mutation == "truthy_job_flag" else True,
                "status": "PASSED",
            },
        }, str(source))
        manager.update_progress("fixture", status="completed" if mutation == "truthy_job_flag" else "paused")
    try:
        assert cli.main(["--report-root", str(quality.paths.root.parent), "export", "fixture", "--output", str(prior)]) != 0
        assert prior.read_text() == "Prior validated delivery"
        assert not (tmp_path / "delivered (1).txt").exists()
        assert candidate.exists()
    finally:
        manager.close()


@pytest.mark.parametrize("with_job", [False, True])
def test_export_validated_bytes_preserves_previous_delivery_and_closes_manager(tmp_path, monkeypatch, capsys, with_job):
    monkeypatch.chdir(tmp_path)
    source, candidate, quality = validated_pair(tmp_path)
    prior = tmp_path / "delivered.txt"
    prior.write_text("Prior validated delivery")
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    if with_job:
        manager.start_job("fixture", "txt", {
            "qa_staging_output": str(candidate), "output_filepath": str(prior),
            "quality_assurance_result": {"publishable": True, "status": "PASSED"},
        }, str(source))
        manager.mark_completed("fixture", quality_gate_passed=True)
    close = Mock(wraps=manager.close)
    monkeypatch.setattr(manager, "close", close)
    monkeypatch.setattr(cli, "CheckpointManager", lambda: manager)
    capsys.readouterr()
    assert cli.main(["--report-root", str(quality.paths.root.parent), "export", "fixture", "--output", str(prior)]) == 0
    published = Path(capsys.readouterr().out.strip())
    assert published == tmp_path / "delivered (1).txt"
    assert published.read_bytes() == candidate.read_bytes()
    assert prior.read_text() == "Prior validated delivery"
    close.assert_called_once()
    candidate.write_text("A later repair")
    assert publication.artifact_sha256(published) == quality.validated_output_sha256


def test_updating_report_path_does_not_approve_new_bytes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, candidate, quality = validated_pair(tmp_path)
    expected = quality.validated_output_sha256
    candidate.write_text("A changed candidate")
    quality.update_output_path(candidate)
    report = json.loads(quality.paths.export_report.read_text())
    assert report["output_sha256"] == expected
    assert report["output_sha256"] != publication.artifact_sha256(candidate)
    manager = CheckpointManager(str(tmp_path / "jobs.db"))
    close = Mock(wraps=manager.close)
    monkeypatch.setattr(manager, "close", close)
    monkeypatch.setattr(cli, "CheckpointManager", lambda: manager)
    assert cli.main(["--report-root", str(quality.paths.root.parent), "export", "fixture", "--output", str(tmp_path / "final.txt")]) == 1
    assert not (tmp_path / "final.txt").exists()
    close.assert_called_once()


@pytest.mark.parametrize("changed", ["source", "candidate"])
def test_artifact_change_during_validation_cannot_be_approved(tmp_path, monkeypatch, changed):
    source, candidate, _ = validated_pair(tmp_path)
    validate = runner.validate_manifest

    def changing_validation(*args, **kwargs):
        result = validate(*args, **kwargs)
        (source if changed == "source" else candidate).write_text("Changed during validation")
        return result

    monkeypatch.setattr(runner, "validate_manifest", changing_validation)
    with pytest.raises(RuntimeError, match="changed during validation"):
        runner.run_quality_assurance(
            source_path=source, output_path=candidate, source_language="English",
            target_language="Spanish", run_id="race", report_root=tmp_path / "reports",
        )
    assert not (tmp_path / "reports/race/translation_report.json").exists()


def test_approved_copy_detects_a_change_after_copy_started(tmp_path, monkeypatch):
    _, candidate, quality = validated_pair(tmp_path)
    destination = tmp_path / "delivered.txt"
    original_copy = publication.shutil.copyfileobj

    def changed_copy(input_file, output_file, *args, **kwargs):
        candidate.write_text("Changed after the copy opened its source")
        return original_copy(input_file, output_file, *args, **kwargs)

    monkeypatch.setattr(publication.shutil, "copyfileobj", changed_copy)
    with pytest.raises(ValueError, match="validated artifact"):
        publication.atomic_publish(candidate, destination, expected_sha256=quality.validated_output_sha256)
    assert not destination.exists()
    assert not list(tmp_path.glob(".publishing-*"))
