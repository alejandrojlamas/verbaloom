"""Resume safety net for EPUB.

The progress-system goldens cover full runs, but not interrupt/resume — which
is exactly the path the checkpoint `chunk_index` convention and the EPUB
cross-file accumulator drive. This test interrupts an EPUB translation partway,
resumes it from the checkpoint, and asserts the resumed output is byte-for-byte
identical (content-wise) to an uninterrupted run.

It is the prerequisite net for any change to the EPUB checkpoint/accumulator
logic: if resume stops producing the same output, this fails.
"""

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.execution_lock import ExecutionLockedError
from tests.characterization import fake_llm, fixtures
from tests.characterization.recorder import fingerprint_output

_COMMON = dict(
    source_language="English",
    target_language="French",
    model_name="fake-echo",
    llm_provider="poe",
    poe_api_key="UNUSED_FAKE_KEY",
    max_tokens_per_chunk=60,
    context_window=4096,
    auto_adjust_context=False,
    prompt_options={"disable_target_language_gate": True},
)


def _run(coro_factory, work_dir: Path):
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        asyncio.run(coro_factory())
    finally:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _patch_llm(monkeypatch):
    fake_llm.install(monkeypatch)


@pytest.mark.asyncio
async def test_native_job_registration_excludes_competing_worker_and_credentials(tmp_path, monkeypatch):
    source = fixtures.build_epub(tmp_path)
    manager = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    engine = AsyncMock(return_value=True)
    monkeypatch.setattr("src.core.epub.translator.translate_epub_file", engine)
    options = dict(_COMMON, prompt_options={
        "editorial": {"tone": "formal", "credentials": {"custom": "TOKEN_SENTINEL"}},
        "glossary_terms": {"token": "fiche"}, "_runtime_only": "TOKEN_SENTINEL",
    })

    async def run():
        return await translate_file(str(source), str(tmp_path / "out.epub"),
                                    checkpoint_manager=manager, translation_id="native", **options)

    try:
        with manager.execution_lock("native"), pytest.raises(ExecutionLockedError):
            await run()
        engine.assert_not_awaited()
        assert manager.get_job("native") is None
        assert await run()
        stored = manager.get_job("native")["config"]
        assert "TOKEN_SENTINEL" not in json.dumps(stored)
        assert "UNUSED_FAKE_KEY" not in json.dumps(stored)
        assert stored["prompt_options"]["editorial"] == {"tone": "formal"}
        assert stored["prompt_options"]["glossary_terms"] == {"token": "fiche"}
    finally:
        manager.close()


@pytest.mark.parametrize("registered", [False, True], ids=["public-api", "existing-job"])
def test_epub_interrupt_resume_matches_clean_run(tmp_path, registered):
    input_path = fixtures.build_epub(tmp_path)

    # 1) Clean, uninterrupted reference run.
    clean_out = tmp_path / "clean.epub"
    clean_cm = CheckpointManager(str(tmp_path / "jobs_clean.db"), uploads_dir=tmp_path / "uploads_clean")

    def clean_factory():
        return translate_file(
            input_filepath=str(input_path),
            output_filepath=str(clean_out),
            checkpoint_manager=clean_cm,
            translation_id="clean",
            **_COMMON,
        )

    _run(clean_factory, tmp_path)
    clean_fp = fingerprint_output(clean_out)

    # 2) Interrupted run: stop once a couple of chunks have been processed
    #    (file 0 checkpointed, into a later file).
    resume_out = tmp_path / "resumed.epub"
    cm = CheckpointManager(str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads")
    tid = "resume_job"
    if registered:
        assert cm.start_job(tid, "epub", {"existing_policy": "preserve"}, str(input_path))
        original_config = cm.get_job(tid)["config"]

    state = {"completed": 0, "stop": False}

    def stats_cb(stats):
        state["completed"] = stats.get("completed_chunks", 0)
        if state["completed"] >= 2:
            state["stop"] = True

    def interrupted_factory():
        return translate_file(
            input_filepath=str(input_path),
            output_filepath=str(resume_out),
            checkpoint_manager=cm,
            translation_id=tid,
            stats_callback=stats_cb,
            check_interruption_callback=lambda: state["stop"],
            **_COMMON,
        )

    _run(interrupted_factory, tmp_path)

    checkpoint = cm.load_checkpoint(tid)
    resume_from_index = checkpoint["resume_from_index"] if checkpoint else 0
    assert state["completed"] >= 2, "interrupt did not fire as expected"
    assert resume_from_index >= 1, "expected at least one file checkpointed before interrupt"
    saved_config = checkpoint["job"]["config"]
    assert Path(saved_config["preserved_input_path"]).read_bytes() == input_path.read_bytes()
    if registered:
        assert saved_config == original_config
    else:
        assert saved_config["target_language"] == "French" and saved_config["model"] == "fake-echo"
        assert saved_config["prompt_options"]["disable_target_language_gate"] is True
        assert "UNUSED_FAKE_KEY" not in json.dumps(saved_config)

    # 3) Resume to completion.
    def resume_factory():
        return translate_file(
            input_filepath=saved_config["preserved_input_path"],
            output_filepath=str(resume_out),
            checkpoint_manager=cm,
            translation_id=tid,
            resume_from_index=resume_from_index,
            check_interruption_callback=lambda: False,
            **_COMMON,
        )

    _run(resume_factory, tmp_path)
    resumed_fp = fingerprint_output(resume_out)

    assert resumed_fp == clean_fp, (
        "resumed EPUB output differs from the uninterrupted run"
    )
