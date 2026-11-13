"""Native EPUB pipeline: a failed chapter is deferred, then repaired alone.

Runs the real orchestration, checkpoint store and resume path against
synthetic HTTP. No validator is stubbed: publication gates are not part of
this scenario, only "does the book keep moving and never pay twice".
"""

import json
import zipfile
from types import SimpleNamespace

import httpx
import pytest
from lxml import etree

from src.core.adapters.translate_file import translate_file
from src.core.pipeline_runtime import activate_pipeline, close_pipeline, request_role
from src.core.usage.store import TokenUsageStore
from src.persistence.checkpoint_manager import CheckpointManager
from tests.characterization.fake_llm import _extract_source_block
from tests.unit.epub.test_native_pipeline_e2e import TRANSLATIONS, write_book
from tests.unit.test_benchmark_execution import CREDENTIALS
from tests.unit.test_source_audited_strategy import ASSESSMENT, config

CHAPTER_ONE = "The traveler watched the quiet sea"
CHAPTER_TWO = "The next day they returned to the quiet house."


@pytest.fixture
def faulty_http(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = {"break_chapter_one": True, "calls": []}

    class HTTP:
        async def post(self, url, **kwargs):
            payload = kwargs["json"]
            user = payload["messages"][-1]["content"]
            system = payload["messages"][0]["content"]
            role = request_role.get()
            state["calls"].append({"role": role, "user": user})
            if "COMBINED_QUALITY_AUDIT_JSON" in system:
                fidelity = json.loads(ASSESSMENT.split(">", 1)[1].split("</", 1)[0])
                answer = json.dumps({
                    "editorial": {
                        "decision": "accept", "confidence": 0.95,
                        "reason": "Synthetic fixture", "issues": [],
                        "missing_from_source": [], "added_not_in_source": [],
                        "weird_symbols": [], "structure_score": 1,
                    },
                    "fidelity": fidelity,
                    "profile": {
                        "scores": {}, "overall_decision": "pass", "issues": [],
                        "glossary_suggestions": [], "summary": "Synthetic fixture",
                    },
                })
            elif role == "audit":
                answer = ASSESSMENT
            elif state["break_chapter_one"] and CHAPTER_ONE in user:
                # A healthy HTTP exchange whose content is unusable: the unit
                # has no valid candidate, the provider itself is fine.
                answer = "I cannot format this passage."
            else:
                source = _extract_source_block(user)
                for original, translated in TRANSLATIONS.items():
                    source = source.replace(original, translated)
                answer = "<TRANSLATION>" + source + "</TRANSLATION>"
            return httpx.Response(
                200,
                request=httpx.Request("POST", url),
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": answer}}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 200},
                    "model": payload["model"],
                    "id": "synthetic-epub-response",
                },
            )

    async def client(_):
        return HTTP()

    monkeypatch.setattr("src.core.llm.base.LLMProvider._get_client", client)
    monkeypatch.setattr(
        "src.core.llm.providers.deepseek.get_deepseek_pricing_status",
        lambda: SimpleNamespace(disabled=False),
    )
    return state


async def _translate(tmp_path, settings, store, output, logs, *, resume_from_index=0):
    manager = CheckpointManager(
        str(tmp_path / "jobs.db"), uploads_dir=tmp_path / "uploads"
    )
    if resume_from_index == 0 and manager.get_job("deferred") is None:
        manager.start_job("deferred", "epub", settings, str(tmp_path / "source.epub"))
    token = activate_pipeline(
        settings["pipeline"], job_id="deferred", credentials=CREDENTIALS, store=store
    )
    try:
        result = await translate_file(
            input_filepath=str(tmp_path / "source.epub"),
            output_filepath=str(output),
            source_language="English",
            target_language="Spanish",
            model_name=settings["model"],
            llm_provider=settings["llm_provider"],
            deepseek_api_key=CREDENTIALS["deepseek"],
            openai_api_key=CREDENTIALS["openai"],
            prompt_options=settings["prompt_options"],
            max_tokens_per_chunk=450,
            checkpoint_manager=manager,
            translation_id="deferred",
            resume_from_index=resume_from_index,
            log_callback=lambda event, message, **data: logs.append((event, message)),
        )
        return result, manager.load_checkpoint("deferred")
    finally:
        await close_pipeline(token)
        manager.close()


@pytest.mark.asyncio
async def test_failed_chapter_is_deferred_then_repaired_without_repaying(
    tmp_path, faulty_http,
):
    write_book(tmp_path / "source.epub")
    output = tmp_path / "translated.epub"
    settings = config("compatible", refine=True, preserve_technical=True)
    settings["pipeline"]["budget"] = {"limit_usd": "5", "max_calls": 200}
    store = TokenUsageStore(tmp_path / "usage.db")
    logs = []

    # --- Pass 1: chapter one has no usable candidate ------------------------
    result, checkpoint = await _translate(tmp_path, settings, store, output, logs)

    events = [event for event, _message in logs]
    assert result is False
    assert not output.exists(), "an incomplete book must not be published"
    assert "epub_file_deferred" in events
    # The book kept moving: the chapter after the failed one was translated.
    assert any(
        CHAPTER_TWO in call["user"] and call["role"] != "audit"
        for call in faulty_http["calls"]
    )
    completed_rows = {
        row["chunk_data"]["last_file"]
        for row in checkpoint["chunks"]
        if row["status"] == "completed"
    }
    assert "two.xhtml" in completed_rows
    assert "one.xhtml" not in completed_rows
    # Resume starts at the failed chapter, never past it.
    assert checkpoint["resume_from_index"] == 1
    assert checkpoint["checkpoint_complete"] is False
    assert int(checkpoint["job"]["progress"]["failed_chunks"]) >= 1

    # --- Pass 2: the model answers properly; only chapter one is pending -----
    faulty_http["break_chapter_one"] = False
    calls_before = len(faulty_http["calls"])
    logs.clear()

    result, checkpoint = await _translate(
        tmp_path, settings, store, output, logs,
        resume_from_index=checkpoint["resume_from_index"],
    )

    assert result is True, "\n".join(f"{e}: {m}" for e, m in logs[-15:])
    second_pass = faulty_http["calls"][calls_before:]
    assert second_pass, "the pending chapter must be translated"
    assert any(CHAPTER_ONE in call["user"] for call in second_pass)
    # Nothing of the completed chapter is requested, or paid for, again.
    assert not any(
        CHAPTER_TWO in call["user"]
        or TRANSLATIONS[CHAPTER_TWO] in call["user"]
        for call in second_pass
    )
    assert checkpoint["checkpoint_complete"] is True
    assert int(checkpoint["job"]["progress"]["failed_chunks"]) == 0

    with zipfile.ZipFile(output) as translated:
        one = etree.fromstring(translated.read("OEBPS/one.xhtml"))
        two = etree.fromstring(translated.read("OEBPS/two.xhtml"))
    assert one.xpath('string(//*[@id="p1"])') == TRANSLATIONS[
        "The traveler watched the quiet sea and remembered his childhood."
    ]
    assert two.xpath('string(//*[@id="p4"])') == TRANSLATIONS[CHAPTER_TWO]
