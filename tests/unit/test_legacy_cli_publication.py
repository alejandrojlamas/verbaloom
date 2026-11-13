"""Legacy CLI must report success only after a validated atomic delivery."""

import asyncio
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from src.core import adapters
from src.core.quality_assurance import cli as quality_cli
from src.core.quality_assurance.publication import artifact_sha256
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils import file_utils, notifier, unified_logger

ENTRYPOINT = Path(__file__).resolve().parents[2] / "translate.py"


def invoke(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", [str(ENTRYPOINT), *argv])
    try:
        runpy.run_path(str(ENTRYPOINT), run_name="__main__")
    except SystemExit as exc:
        return exc.code
    return 0


@pytest.mark.parametrize("status", ["paused", "interrupted", "error", "review_required", "budget_exhausted"])
@pytest.mark.parametrize("strict", [False, True])
def test_reading_complete_checkpoint_preserves_stop_reason(tmp_path, status, strict):
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    try:
        manager.start_job("stopped", "txt", {"prompt_options": {"strict_quality_assurance": strict}})
        manager.save_checkpoint("stopped", 0, "Source", "Target", total_chunks=1, completed_chunks=1)
        manager.update_progress("stopped", status=status)
        checkpoint = manager.load_checkpoint("stopped")
        assert checkpoint["checkpoint_complete"] is True
        assert checkpoint["job"]["status"] == status
        assert manager.get_job("stopped")["status"] == status
        if status in {"paused", "interrupted", "error"}:
            assert manager.get_resumable_jobs()[0]["status"] == status
    finally:
        manager.close()


@pytest.mark.parametrize("status", ["paused", "interrupted", "error", "running", "validating", "completed"])
def test_repair_queue_keeps_a_stop_reason_and_retains_rejected_text(tmp_path, status):
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    try:
        manager.start_job("repair", "txt", {})
        manager.save_checkpoint("repair", 0, "Source", "Candidate", total_chunks=1, completed_chunks=1)
        manager.update_progress("repair", status=status)
        assert manager.mark_chunks_for_repair("repair", [0]) == 1
        expected = status if status in {"paused", "interrupted", "error"} else "partial"
        checkpoint = manager.load_checkpoint("repair")
        assert checkpoint["job"]["status"] == expected
        assert checkpoint["chunks"][0]["chunk_data"]["rejected_translation"] == "Candidate"
        assert checkpoint["failed_chunk_indices"] == [0]
    finally:
        manager.close()


@pytest.mark.parametrize("refine_only", [False, True])
@pytest.mark.parametrize(
    "outcome", ["pass", "false", "reject", "gate_error", "crash", "cancel", "pause", "missing", "changed_after_qa"]
)
def test_legacy_entrypoint_stages_gates_and_reports_truthfully(
    tmp_path, monkeypatch, refine_only, outcome
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("A complete passage with several sentences and all its original details.")
    destination = tmp_path / "delivery" / "book.txt"
    destination.parent.mkdir()
    destination.write_text("Prior validated delivery")
    database = tmp_path / "jobs.db"
    manager = CheckpointManager(str(database), uploads_dir=tmp_path / "uploads")
    close = Mock(wraps=manager.close)
    monkeypatch.setattr(manager, "close", close)
    monkeypatch.setattr("src.persistence.checkpoint_manager.CheckpointManager", lambda: manager)
    logger, notifications, observed = Mock(), Mock(), {}
    monkeypatch.setattr(unified_logger, "setup_cli_logger", lambda **_: logger)
    monkeypatch.setattr(notifier, "notify", notifications)
    tts = AsyncMock(return_value=(True, "", "audio.mp3"))
    monkeypatch.setattr(file_utils, "generate_tts_for_translation", tts)

    async def process(**kwargs):
        observed.update(kwargs)
        path = Path(kwargs["output_filepath"])
        path.parent.mkdir(parents=True, exist_ok=True)
        run_id = kwargs["translation_id"]
        # Characterize the old entrypoint too, which delegated job creation.
        if not manager.get_job(run_id):
            manager.start_job(run_id, "txt", {}, str(source))
        manager.save_checkpoint(
            run_id, 0, source.read_text(), "Complete candidate", {},
            total_chunks=1, completed_chunks=1, failed_chunks=0,
        )
        if outcome != "missing":
            path.write_text("Complete candidate")
        if outcome == "crash":
            raise RuntimeError("Injected worker failure")
        if outcome == "cancel":
            raise asyncio.CancelledError()
        if outcome == "pause":
            manager.mark_paused(run_id)
        return outcome != "false"

    monkeypatch.setattr(adapters, "translate_file", process)
    monkeypatch.setattr(adapters, "refine_file", process)
    quality = SimpleNamespace(
        publishable=outcome != "reject",
        report=SimpleNamespace(status=SimpleNamespace(value="PASSED"), model_usage={}),
        paths=SimpleNamespace(report_html=tmp_path / "report.html", root=tmp_path, manifest=tmp_path / "manifest.json"),
        repair_checkpoint_indices=[],
        update_output_path=Mock(),
    )

    def validate(**kwargs):
        observed["quality"] = kwargs
        assert destination.read_text() == "Prior validated delivery"
        assert list(destination.parent.glob("*.txt")) == [destination]
        assert Path(kwargs["output_path"]).parent.name.startswith(".publication-")
        if outcome == "gate_error":
            raise RuntimeError("Injected gate failure")
        quality.validated_output_sha256 = artifact_sha256(kwargs["output_path"])
        if outcome == "changed_after_qa":
            Path(kwargs["output_path"]).write_text("Unvalidated later candidate")
        return quality

    monkeypatch.setattr(quality_cli, "run_quality_assurance", validate)
    args = ["-i", str(source), "-o", str(destination), "--provider", "ollama", "-m", "fake",
            "-sl", "English", "-tl", "Spanish", "--tts", "--no-color"]
    if refine_only:
        args.append("--refine-only")
    try:
        code = invoke(monkeypatch, args)
    except asyncio.CancelledError:
        pytest.fail("CLI cancellation escaped without a persistent interrupted result")
    expected_code = (
        0 if outcome == "pass" else 130 if outcome == "cancel"
        else 1 if outcome in {"crash", "gate_error", "missing", "changed_after_qa"} else 2
    )
    assert code == expected_code, logger.error.call_args_list
    assert destination.read_text() == "Prior validated delivery"
    staged = Path(observed["output_filepath"])
    assert staged.parent.name.startswith(".publication-")
    assert close.call_count == 1
    with_manager = CheckpointManager(str(database), uploads_dir=tmp_path / "uploads")
    try:
        job = with_manager.get_job(observed["translation_id"])
    finally:
        with_manager.close()
    assert job["config"]["qa_staging_output"] == str(staged)
    assert job["config"]["refine_only"] is refine_only
    assert "api_key" not in str(job["config"]).lower()
    assert job["config"]["source_language"] == ("Spanish" if refine_only else "English")
    if "quality" in observed:
        assert observed["quality"]["source_language"] == ("Spanish" if refine_only else "English")
    if outcome == "pass":
        assert code == 0
        assert "quality" in observed
        assert job["status"] == "completed"
        published = Path(job["config"]["output_filepath"])
        assert published != destination and published.read_text() == "Complete candidate"
        tts.assert_awaited_once()
        assert tts.call_args.kwargs["translated_filepath"] == str(published)
        assert notifications.call_args.args[0] == notifier.EVENT_SUCCESS
    else:
        assert code == expected_code
        assert job["status"] == (
            "paused" if outcome == "pause" else "interrupted" if outcome == "cancel"
            else "error" if outcome in {"crash", "gate_error", "changed_after_qa"} else "partial"
        )
        assert list(destination.parent.glob("*.txt")) == [destination]
        assert not any(c.args[0] == notifier.EVENT_SUCCESS for c in notifications.call_args_list)
        tts.assert_not_awaited()
        assert staged.exists() is (outcome != "missing")


@pytest.mark.parametrize("refine_only", [False, True])
def test_legacy_cli_with_real_txt_adapter_and_quality_gates(tmp_path, monkeypatch, refine_only):
    from tests.characterization import fake_llm

    monkeypatch.chdir(tmp_path)
    fake_llm.install(monkeypatch)
    monkeypatch.setattr(notifier, "notify", Mock())
    source = tmp_path / "source.txt"
    original = (
        "The quick brown fox jumps over the lazy dog. This document contains "
        "enough material and several sentences. Every paragraph preserves meaning.\n\n"
        "A translator runs carefully while the chunker split the document into "
        "one more paragraph with enough material for characterization."
    )
    source.write_text(fake_llm._pseudo_translate_to_french(original) if refine_only else original)
    output = tmp_path / "book.txt"
    args = ["-i", str(source), "-o", str(output), "--provider", "poe", "--poe_api_key", "fixture",
            "-m", "fake-echo", "-sl", "English", "-tl", "French", "--no-color"]
    if refine_only:
        args.append("--refine-only")
    # No network TTS, even when the user's environment enables it globally.
    monkeypatch.setattr("src.tts.tts_config.TTS_ENABLED", False)
    assert invoke(monkeypatch, args) == 0
    import json

    reports = list((tmp_path / "data/quality_runs").glob("*/translation_report.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert report["publishable"] is True
    assert len(report["quality_gates"]) == 10
    assert output.is_file()
    assert str(output) == report["output"]["path"]
    assert output.read_text() != ""


@pytest.mark.asyncio
async def test_strict_resume_keeps_the_saved_monolingual_operation(tmp_path, monkeypatch):
    translate, refine = AsyncMock(return_value=True), AsyncMock(return_value=True)
    monkeypatch.setattr(quality_cli, "translate_file", translate)
    monkeypatch.setattr(quality_cli, "refine_file", refine)
    options = {"refine_only": True, "target_language": "French", "prompt_options": {"refine": False}}
    assert await quality_cli._invoke_translation(
        Mock(), "saved", options, tmp_path / "source.txt", tmp_path / "staged.txt", resume_from=3
    )
    translate.assert_not_awaited()
    refine.assert_awaited_once()
    assert refine.call_args.kwargs["resume_from_index"] == 3
    assert refine.call_args.kwargs["target_language"] == "French"
    assert "source_language" not in refine.call_args.kwargs
    assert "concurrency" not in refine.call_args.kwargs


@pytest.mark.parametrize("cancelled", [False, True])
def test_strict_resume_worker_failure_is_persisted_and_closed(tmp_path, monkeypatch, cancelled):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Le texte conserve son sens et ses personnages.")
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    manager.start_job("resume", "txt", {
        "refine_only": True, "source_language": "French", "target_language": "French",
        "prompt_options": {"strict_quality_assurance": True},
    }, str(source))
    manager.save_checkpoint("resume", 0, source.read_text(), source.read_text(), total_chunks=2, completed_chunks=1)
    manager.mark_paused("resume")
    close = Mock(wraps=manager.close)
    monkeypatch.setattr(manager, "close", close)
    monkeypatch.setattr(quality_cli, "CheckpointManager", lambda: manager)
    operation = AsyncMock(side_effect=asyncio.CancelledError() if cancelled else RuntimeError("Injected crash"))
    monkeypatch.setattr(quality_cli, "refine_file", operation)
    try:
        assert quality_cli.main(["resume", "resume"]) == (130 if cancelled else 1)
        assert close.call_count == 1
        assert manager.get_job("resume")["status"] == ("interrupted" if cancelled else "error")
        assert manager.load_checkpoint("resume")["actual_completed_chunks"] == 1
    finally:
        manager.close()


def test_duplicate_cli_id_does_not_replace_the_saved_source_or_state(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.txt"
    source.write_text("Original source")
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    manager.start_job("existing", "txt", {}, str(source))
    before = manager.get_job("existing")
    preserved = Path(before["config"]["preserved_input_path"])
    source.write_text("Different source under the same filename")
    monkeypatch.setattr(quality_cli, "CheckpointManager", lambda: manager)
    try:
        assert quality_cli.main(["translate", str(source), "--run-id", "existing", "--dry-run"]) == 1
        assert preserved.read_text() == "Original source"
        assert manager.get_job("existing") == before
    finally:
        manager.close()
