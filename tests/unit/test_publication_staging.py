"""Publication is invisible until validated and never overwrites another delivery."""

import asyncio
import errno
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.api import handlers
from src.api.services.file_service import FileService
from src.api.translation_state import TranslationStateManager
from src.core.quality_assurance import publication
from src.core.quality_assurance.cli import _atomic_publish
from src.persistence.checkpoint_manager import CheckpointManager


def test_concurrent_publishers_keep_every_validated_candidate(tmp_path, monkeypatch):
    destination = tmp_path / "delivered" / "book.txt"
    destination.parent.mkdir()
    sources = [tmp_path / f"candidate-{i}.txt" for i in range(4)]
    for i, source in enumerate(sources):
        source.write_text(f"Validated candidate {i}")
    barrier = threading.Barrier(len(sources))
    copy = shutil.copyfileobj

    def synchronized_copy(*args, **kwargs):
        barrier.wait(timeout=10)
        return copy(*args, **kwargs)

    monkeypatch.setattr(shutil, "copyfileobj", synchronized_copy)
    with ThreadPoolExecutor(max_workers=len(sources)) as pool:
        outputs = list(
            pool.map(lambda source: _atomic_publish(source, destination), sources)
        )
    assert len(set(outputs)) == len(sources)
    assert {p.read_text() for p in outputs} == {p.read_text() for p in sources}
    assert not list(destination.parent.glob(".*"))


@pytest.mark.parametrize("failure", ["copy", "fsync", "link"])
def test_failed_commit_preserves_source_and_previous_delivery(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "candidate.txt"
    destination = tmp_path / "delivered.txt"
    source.write_bytes(b"new validated candidate")
    destination.write_bytes(b"previous valid delivery")

    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "Injected filesystem failure")

    if failure == "copy":
        monkeypatch.setattr(publication.shutil, "copyfileobj", fail)
    else:
        monkeypatch.setattr(publication.os, failure, fail)
    with pytest.raises(OSError):
        _atomic_publish(source, destination)
    assert source.read_bytes() == b"new validated candidate"
    assert destination.read_bytes() == b"previous valid delivery"
    assert set(tmp_path.iterdir()) == {source, destination}


def test_broken_symlink_is_not_replaced_and_published_copy_is_independent(tmp_path):
    source = tmp_path / "candidate.txt"
    destination = tmp_path / "book.txt"
    source.write_bytes(b"validated")
    destination.symlink_to(tmp_path / "absent.txt")
    published = _atomic_publish(source, destination)
    source.write_bytes(b"later rejected repair")
    assert destination.is_symlink()
    assert not (tmp_path / "absent.txt").exists()
    assert published.name == "book (1).txt"
    assert published.read_bytes() == b"validated"


def test_staging_hides_drafts_and_reports_follow_committed_name(tmp_path):
    source = publication.create_staging_path(tmp_path / "book.epub")
    source.write_bytes(b"validated archive")
    source.with_name("book - publication gate.json").write_text('{"publishable": true}')
    source.with_name("book [native txt].txt").write_text("native intermediate")
    service = FileService(str(tmp_path))
    assert service.list_all_files() == []
    assert service.find_file(str(source.relative_to(tmp_path))) is None
    published = _atomic_publish(source, tmp_path / "different.epub")
    publication.publish_report_sidecars(source, published)
    assert {f["filename"] for f in service.list_all_files()} == {
        "different.epub",
        "different - publication gate.json",
    }
    assert source.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("converted", [False, True])
@pytest.mark.parametrize(
    "verdict",
    [
        "pass",
        "reject",
        "exception",
        "pause",
        "refine_failure",
        "crash",
        "cancel",
        "engine_error",
        "changed_after_qa",
    ],
)
@pytest.mark.parametrize('reserved_submission', [False, True])
async def test_web_worker_keeps_previous_delivery_until_all_gates_pass(
    tmp_path, monkeypatch, converted, verdict, reserved_submission
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("A complete source passage.")
    desired = output / ("book.docx" if converted else "book.txt")
    desired.write_bytes(b"previous validated delivery")
    config = {
        "file_type": "txt",
        "file_path": str(source),
        "source_language": "English",
        "target_language": "Spanish",
        "output_filename": desired.name,
        "_job_output_filename": desired.name,
        "output_format": "docx" if converted else "auto",
        "is_resume": not reserved_submission,
        "llm_provider": "deepseek",
        "model": "deepseek-v4-flash",
        "llm_api_endpoint": "https://api.deepseek.com",
        "prompt_options": {},
        "max_automatic_recoveries": 0,
        "refine_after": verdict == "refine_failure",
    }
    if reserved_submission:
        config['_submission_fingerprint'] = 'fixture-request-identity'
    manager = CheckpointManager(str(tmp_path / "jobs.db"))
    manager.start_job("publication", "txt", config, str(source))
    state = TranslationStateManager(manager)
    state.create_translation("publication", config)
    observations = []
    paths = []

    def observe():
        observations.append(
            (
                desired.read_bytes(),
                {f["filename"] for f in FileService(str(output)).list_all_files()},
            )
        )

    async def translate(**kwargs):
        manager.save_checkpoint(
            "publication",
            0,
            source.read_text(),
            "Translated fixture.",
            total_chunks=1,
            completed_chunks=1,
            failed_chunks=0,
        )
        path = Path(kwargs["output_filepath"])
        paths.append(path)
        path.write_bytes(b"new candidate")
        observe()
        if verdict == "crash":
            raise RuntimeError("Injected worker failure after assembly")
        if verdict == "cancel":
            raise asyncio.CancelledError()
        kwargs["stats_callback"](
            {"total_chunks": 1, "completed_chunks": 1, "failed_chunks": 0}
        )
        if verdict == "engine_error":
            state.set_translation_field("publication", "status", "error")
            state.set_translation_field(
                "publication", "error", "Permanent engine failure"
            )
            return False
        return True

    def convert(**kwargs):
        path = Path(kwargs["destination_path"])
        paths.append(path)
        path.write_bytes(b"new converted candidate")
        observe()

    quality = SimpleNamespace(
        publishable=verdict in {"pass", "pause", "refine_failure", "changed_after_qa"},
        paths=SimpleNamespace(
            root=tmp_path / "qa", report_html=tmp_path / "qa/report.html"
        ),
        report=SimpleNamespace(
            status=SimpleNamespace(value="PASS" if verdict == "pass" else "BLOCKED"),
            repair_units=[],
        ),
        update_output_path=Mock(),
    )

    def gate(**kwargs):
        paths.append(Path(kwargs["output_path"]))
        observe()
        if verdict == "exception":
            raise RuntimeError("Validator failed")
        if verdict == "pause":
            state.set_translation_field("publication", "interrupted", True)
        quality.validated_output_sha256 = publication.artifact_sha256(kwargs["output_path"])
        if verdict == "changed_after_qa":
            Path(kwargs["output_path"]).write_bytes(b"Unvalidated later candidate")
        return quality

    monkeypatch.setattr(handlers, "translate_file", translate)

    async def failed_refinement(**kwargs):
        observe()
        return False

    monkeypatch.setattr(handlers, "refine_file", failed_refinement)
    monkeypatch.setattr(handlers, "convert_output_file", convert)
    monkeypatch.setattr(handlers, "run_quality_assurance", gate)
    monkeypatch.setattr(
        handlers,
        "audit_and_clean_final_artifact",
        lambda *a, **k: SimpleNamespace(
            changed=False, unresolved_findings=[], reading_quality_warnings=[]
        ),
    )
    monkeypatch.setattr(handlers, "_final_source_sample_audit_enabled", lambda _: False)
    monkeypatch.setattr(handlers, "emit_update", Mock())
    monkeypatch.setattr(handlers, "notify", Mock())
    companions = Mock(return_value={"files": [], "summary": "Fixture companion"})
    monkeypatch.setattr(handlers, "audiobook_profile_enabled", lambda _: True)
    monkeypatch.setattr(handlers, "_create_audiobook_companion_outputs", companions)
    monkeypatch.setattr(
        handlers,
        "_schedule_failed_chunk_recovery",
        Mock(side_effect=AssertionError("Recovery disabled for fixture")),
    )
    try:
        if verdict == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await handlers.perform_actual_translation(
                    "publication", config, state, str(output), Mock()
                )
        else:
            await handlers.perform_actual_translation(
                "publication", config, state, str(output), Mock()
            )
        assert observations, state.get_translation_field("publication", "error")
        assert all(old == b"previous validated delivery" for old, _ in observations)
        assert all(files == {desired.name} for _, files in observations)
        assert desired.read_bytes() == b"previous validated delivery"
        assert all(path.parent != output for path in paths)
        if verdict in {"crash", "cancel", "changed_after_qa"}:
            if verdict == "changed_after_qa":
                assert paths[-1].read_bytes() == b"Unvalidated later candidate"
                assert state.get_translation_field("publication", "status") == "error"
                assert "validated artifact" in state.get_translation_field("publication", "error")
            else:
                assert paths[0].read_bytes() == b"new candidate"
            assert {
                f["filename"] for f in FileService(str(output)).list_all_files()
            } == {desired.name}
            assert manager.load_checkpoint("publication")["job"]["config"][
                "qa_staging_output"
            ]
            quality.update_output_path.assert_not_called()
            companions.assert_not_called()
            return
        published = Path(state.get_translation_field("publication", "output_filepath"))
        assert published.exists() and published.parent == output
        assert published != desired
        assert published.read_bytes() == (
            b"new converted candidate" if converted else b"new candidate"
        )
        if verdict == "pass":
            companions.assert_called_once()
            assert state.get_translation_field("publication", "status") == "completed"
            assert not published.name.startswith("[partial]")
            assert (
                manager.load_checkpoint("publication")["job"]["config"][
                    "_job_output_filename"
                ]
                == published.name
            )
        elif verdict == "engine_error":
            companions.assert_not_called()
            assert state.get_translation_field("publication", "status") == "error"
            assert (
                state.get_translation_field("publication", "error")
                == "Permanent engine failure"
            )
            assert published.name.startswith("[partial] ")
        else:
            companions.assert_not_called()
            assert state.get_translation_field("publication", "status") in {
                "partial",
                "interrupted",
            }
            assert published.name.startswith("[partial] ")
        if verdict not in {"exception", "refine_failure", "engine_error"}:
            quality.update_output_path.assert_called_once_with(str(published))
        assert source.read_text() == "A complete source passage."
    finally:
        manager.close()
