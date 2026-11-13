"""C/D are strategy ablations, not measured model-quality claims."""

from copy import deepcopy

import pytest
from lxml import etree

from src.core.epub import xhtml_translator
from src.core.epub.unit_contract import mark_reviewed, mark_translated
from src.core.fidelity_supervisor import (
    FidelityDecision,
    FidelityIssue,
    _should_run_llm_audit,
    fidelity_supervisor_enabled,
)
from src.core.job_runtime_config import configure_editorial_guard_options
from src.core.llm.base import LLMResponse
from src.core.pipeline_config import apply_pipeline, pipeline_preset, validate_pipeline
from src.core.review_strategy import (
    OPTION,
    apply_review_strategy,
    needs_final_unit_audit,
    skip_universal_refinement,
)

SOURCE = "The traveler watched the quiet sea and remembered his childhood."
TARGET = "El viajero contemplaba el mar tranquilo y recordaba su infancia."
ASSESSMENT = (
    '<FIDELITY_AUDIT_JSON>{"verdict":"pass","confidence":0.95,"reason":"Source comparison complete",'
    '"issues":[],"missing_from_source":[],"added_not_in_source":[],"changed_facts":[],"censored_or_softened":[],'
    '"structure_issues":[],"evidence_source":[],"evidence_candidate":[]}</FIDELITY_AUDIT_JSON>'
)


def config(preset="deepseek_source_audited", kind="epub", **options):
    value = {
        "file_type": kind,
        "llm_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "source_language": "English",
        "target_language": "Spanish",
        "prompt_options": {"target_locale": "es-ES", **options},
    }
    apply_pipeline(value, pipeline_preset(preset, value))
    configure_editorial_guard_options(value)
    return value


class AuditClient:
    provider_type = "deepseek"

    def __init__(self, *args, **kwargs):
        self.calls = []

    async def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return LLMResponse(content=ASSESSMENT)

    async def close(self):
        pass


def test_abcd_hold_roles_and_locale_constant_within_each_provider_pair():
    variants = [
        config(preset)
        for preset in (
            "compatible",
            "openai_equivalent",
            "deepseek_source_audited",
            "openai_source_audited",
        )
    ]
    a, b, c, d = [v["pipeline"] for v in variants]
    assert a["roles"] == c["roles"]
    assert b["roles"] == d["roles"]
    assert a["review_policy"] == b["review_policy"] == "compatible"
    assert c["review_policy"] == d["review_policy"] == "source_audited_v1"
    assert c["epub_repair_policy"] == d["epub_repair_policy"] == "localized_v1"
    assert variants[2]["prompt_options"] == variants[3]["prompt_options"]
    assert all(v["prompt_options"]["target_locale"] == "es-ES" for v in variants)


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"refine_only": True}, "explicit_refinement"),
        ({"operation": "transform"}, "transformation_contract"),
        (
            {"prompt_options": {"text_transform_mode": "explain"}},
            "transformation_contract",
        ),
        ({"prompt_options": {"profile_strength": "strict"}}, "strict_profile"),
    ],
)
def test_required_editorial_operations_are_not_suppressed(updates, reason):
    value = config()
    value.update(updates)
    options = apply_review_strategy(value)
    assert options[OPTION]["reason"] == reason
    assert not skip_universal_refinement(options)

    from types import SimpleNamespace

    from src.core.pipeline_runtime import active_pipeline
    from src.core.review_strategy import bind_runtime_strategy

    token = active_pipeline.set(SimpleNamespace(config=value["pipeline"]))
    try:
        rebound = bind_runtime_strategy(options, file_type="txt")
        assert rebound[OPTION]["reason"] == reason
        assert not skip_universal_refinement(rebound)
    finally:
        active_pipeline.reset(token)


def test_strategy_is_stable_after_resume_and_does_not_mutate_requested_refinement():
    value = config(refine=True, audit_entire_book=True)
    before = deepcopy(value)
    assert configure_editorial_guard_options(value) == before["prompt_options"]
    assert value == before
    assert value["prompt_options"]["refine"] is True
    value["pipeline"]["review_policy"] = "compatible"
    apply_review_strategy(value)
    assert (
        OPTION not in value["prompt_options"]
        and value["prompt_options"]["refine"] is True
    )


def test_complete_semantic_coverage_cannot_be_disabled_by_legacy_off_or_sampling():
    options = config(
        kind="txt", fidelity_supervisor=False, fidelity_supervisor_mode="off"
    )["prompt_options"]
    decision = FidelityDecision(
        1, "translation", "test", True, source_chars=100, candidate_chars=100
    )
    assert fidelity_supervisor_enabled(options)
    assert _should_run_llm_audit(decision, options)
    options = config(kind="epub")["prompt_options"]
    assert not _should_run_llm_audit(decision, options)
    decision.issues = [FidelityIssue("numbers", "reject", "changed number")]
    assert _should_run_llm_audit(decision, options)
    decision.issues = []
    decision.phase = "final_epub_unit_audit"
    assert _should_run_llm_audit(decision, options)


def test_text_first_and_plain_routes_do_not_defer_to_a_nonexistent_final_stage():
    assert not needs_final_unit_audit(
        config(text_first_pipeline_active=True)["prompt_options"]
    )
    assert not needs_final_unit_audit(config(plain_text_mode=True)["prompt_options"])


@pytest.mark.parametrize("invalid", [[], {}, True, "unknown"])
def test_strategy_validation_is_strict(invalid):
    with pytest.raises(ValueError):
        validate_pipeline({**config()["pipeline"], "review_policy": invalid})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preset,refine,expected_rewrites",
    [
        ("compatible", True, 1),
        ("deepseek_source_audited", True, 0),
        ("deepseek_source_audited", False, 0),
    ],
)
async def test_real_xhtml_flow_audits_without_universal_rewrite(
    monkeypatch, preset, refine, expected_rewrites
):
    rewrites = []

    async def translate(**kwargs):
        candidates = []
        for chunk in kwargs["chunks"]:
            translated = chunk["text"].replace(SOURCE, TARGET)
            mark_translated(chunk["unit"], translated)
            candidates.append(translated)
        return candidates, kwargs["stats"], False

    async def review(**kwargs):
        rewrites.append(True)
        for chunk, candidate in zip(kwargs["chunks"], kwargs["translated_chunks"]):
            mark_reviewed(chunk["unit"], candidate, review={"status": "pass"})
        return kwargs["translated_chunks"]

    monkeypatch.setattr(
        xhtml_translator, "_translate_all_chunks_with_checkpoint", translate
    )
    monkeypatch.setattr(xhtml_translator, "_refine_epub_chunks", review)
    root = etree.fromstring(f"<html><body><p>{SOURCE}</p></body></html>")
    client = AuditClient()
    options = config(
        preset,
        refine=refine,
        audit_entire_book=True,
        fidelity_supervisor_invalid_json_retries=0,
    )["prompt_options"]
    success, _ = await xhtml_translator.translate_xhtml_simplified(
        root,
        "English",
        "Spanish",
        "deepseek-v4-pro",
        client,
        prompt_options=options,
        file_href="Text/chapter.xhtml",
    )
    assert success
    assert len(rewrites) == expected_rewrites
    assert len(client.calls) == 1
    assert root.xpath("string(//*[local-name()='p'])") == TARGET


@pytest.mark.asyncio
async def test_txt_approved_resume_and_exact_memory_hit_do_not_call_models(
    tmp_path, monkeypatch
):
    runner, calls, manager, output = txt_runner(tmp_path, monkeypatch)
    assert await runner("first")
    assert calls == {"translation": 1, "audit": 1, "rewrite": 0}
    assert output.read_text(encoding="utf-8-sig").strip() == TARGET
    saved = manager.load_checkpoint("first")["chunks"][0]["chunk_data"]
    assert (
        saved["fidelity_decision"]["review_reason"]
        == "source_audited:complete_semantic_coverage"
    )
    assert await runner("first")
    assert await runner("memory-hit")
    assert calls == {"translation": 1, "audit": 1, "rewrite": 0}


@pytest.mark.asyncio
async def test_txt_changed_source_never_reuses_an_old_unit_approval(
    tmp_path, monkeypatch
):
    from src.core.llm.exceptions import ProviderActionRequiredError

    runner, calls, _, output = txt_runner(tmp_path, monkeypatch)
    assert await runner("first")
    prior = output.read_bytes()
    (tmp_path / "source.txt").write_text(SOURCE.replace("childhood", "father"))
    with pytest.raises(ProviderActionRequiredError) as error:
        await runner("first")
    assert error.value.reason == "checkpoint_source_changed"
    assert calls == {"translation": 1, "audit": 1, "rewrite": 0}
    assert output.read_bytes() == prior


@pytest.mark.asyncio
async def test_changed_cached_candidate_is_audited_without_retranslation(
    tmp_path, monkeypatch
):
    runner, calls, _, output = txt_runner(tmp_path, monkeypatch)
    assert await runner("first")
    amended = TARGET.replace("contemplaba", "miraba")
    monkeypatch.setattr(
        "src.core.adapters.generic_translator.apply_profile_glossary_corrections",
        lambda *a, **k: amended,
    )
    assert await runner("memory-hit")
    assert output.read_text(encoding="utf-8-sig").strip() == amended
    assert calls == {"translation": 1, "audit": 2, "rewrite": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "preset,expected_audits", [("compatible", 0), ("deepseek_source_audited", 1)]
)
async def test_plain_structured_route_audits_final_candidate(
    monkeypatch, preset, expected_audits
):
    from src.core.common import plain_text_pipeline

    async def translate(**kwargs):
        return TARGET

    monkeypatch.setattr(plain_text_pipeline, "generate_translation_request", translate)
    client = AuditClient()
    result, stats, interrupted = await plain_text_pipeline.translate_paragraphs_plain(
        [SOURCE],
        "English",
        "Spanish",
        "deepseek-v4-pro",
        client,
        1000,
        prompt_options=config(preset, plain_text_mode=True)["prompt_options"],
    )
    assert result == [TARGET] and not interrupted
    assert stats.failed_chunks == 0
    assert len(client.calls) == expected_audits


@pytest.mark.asyncio
async def test_plain_structured_route_does_not_accept_a_malformed_audit(monkeypatch):
    from src.core.common import plain_text_pipeline
    from src.core.llm.exceptions import ProviderActionRequiredError

    async def translate(**kwargs):
        return TARGET

    class MalformedClient(AuditClient):
        async def generate(self, *args, **kwargs):
            return LLMResponse(content="not a judgment")

    monkeypatch.setattr(plain_text_pipeline, "generate_translation_request", translate)
    with pytest.raises(ProviderActionRequiredError):
        await plain_text_pipeline.translate_paragraphs_plain(
            [SOURCE],
            "English",
            "Spanish",
            "deepseek-v4-pro",
            MalformedClient(),
            1000,
            prompt_options=config(
                plain_text_mode=True, fidelity_supervisor_invalid_json_retries=0
            )["prompt_options"],
        )


def test_semantic_approval_uses_the_existing_hash_contract_and_accepts_noncritical_warnings():
    from src.core.fidelity_supervisor import assess_fidelity
    from src.core.review_strategy import valid_saved_semantic_approval

    decision = assess_fidelity(SOURCE, TARGET, chunk_index=1, phase="translation")
    decision.judge_decision = "warn"
    decision.judge_model = "deepseek-v4-pro"
    decision.accepted = True
    assert valid_saved_semantic_approval(SOURCE, TARGET, decision.to_dict())
    assert not valid_saved_semantic_approval(
        SOURCE, TARGET + " Changed.", decision.to_dict()
    )
    assert not valid_saved_semantic_approval(SOURCE, TARGET, ["invalid"])


def test_deepseek_named_preset_does_not_silently_select_openai():
    with pytest.raises(ValueError, match="DeepSeek"):
        pipeline_preset(
            "deepseek_source_audited",
            {"llm_provider": "openai", "model": "gpt-6.1-terra"},
        )


@pytest.mark.parametrize(
    "preset,skip", [("compatible", False), ("deepseek_source_audited", True)]
)
@pytest.mark.parametrize("inline", [True, False])
def test_chained_refinement_execution_progress_and_usage_agree(preset, skip, inline):
    from src.core.job_runtime_config import usage_process_type
    from src.core.review_strategy import refinement_plan

    value = config(preset, kind="txt", inline_refinement=inline)
    value["refine_after"] = True
    # Usage metadata is initialized before worker-side option resolution.
    value["prompt_options"].pop(OPTION, None)
    before = deepcopy(value)
    plan = refinement_plan(value)
    assert plan == {
        "refine_after": not inline and not skip,
        "inline_refinement": inline and not skip,
    }
    assert usage_process_type(value) == (
        "translation" if skip else "translation_with_refinement"
    )
    assert value == before


@pytest.mark.asyncio
async def test_plain_rejection_is_not_returned_as_translated_output(monkeypatch):
    from src.core.common import plain_text_pipeline
    from src.core.epub.exceptions import ChunkTranslationFailedError

    async def translate(**kwargs):
        return TARGET

    class RejectingClient(AuditClient):
        async def generate(self, *args, **kwargs):
            return LLMResponse(
                content=ASSESSMENT.replace('"pass"', '"fail"').replace(
                    '"changed_facts":[]',
                    '"changed_facts":["A factual statement was reversed"]',
                )
            )

    stats = []
    monkeypatch.setattr(plain_text_pipeline, "generate_translation_request", translate)
    with pytest.raises(ChunkTranslationFailedError) as error:
        await plain_text_pipeline.translate_paragraphs_plain(
            [SOURCE],
            "English",
            "Spanish",
            "deepseek-v4-pro",
            RejectingClient(),
            1000,
            prompt_options=config(plain_text_mode=True)["prompt_options"],
            stats_callback=stats.append,
        )
    assert error.value.reason == "fidelity_rejected"
    assert stats[-1]["failed_chunks"] == 1


def txt_runner(tmp_path, monkeypatch):
    from src.core.adapters import GenericTranslator, TxtAdapter
    from src.core.translation_memory import TranslationMemoryCache
    from src.persistence.checkpoint_manager import CheckpointManager

    monkeypatch.chdir(tmp_path)
    calls = {"translation": 0, "audit": 0, "rewrite": 0}
    source, output = tmp_path / "source.txt", tmp_path / "target.txt"
    source.write_text(SOURCE)
    manager = CheckpointManager(db_path=str(tmp_path / "checkpoints.db"))
    options = config(
        kind="txt", inline_refinement=True, fidelity_supervisor_invalid_json_retries=0
    )["prompt_options"]
    keys = []
    original_get, original_put = TranslationMemoryCache.get, TranslationMemoryCache.put

    def get(cache, request):
        keys.append((cache.build_key(request), deepcopy(request.prompt_options)))
        return original_get(cache, request)

    def put(cache, request, *args, **kwargs):
        differences = {
            k: (keys[-1][1].get(k), request.prompt_options.get(k))
            for k in keys[-1][1].keys() | request.prompt_options.keys()
            if k != "_fidelity_report"
            and keys[-1][1].get(k) != request.prompt_options.get(k)
        }
        assert cache.build_key(request) == keys[-1][0], str(differences)
        return original_put(cache, request, *args, **kwargs)

    monkeypatch.setattr(TranslationMemoryCache, "get", get)
    monkeypatch.setattr(TranslationMemoryCache, "put", put)

    class CountingClient(AuditClient):
        async def generate(self, *args, **kwargs):
            calls["audit"] += 1
            return await super().generate(*args, **kwargs)

    async def translate(**kwargs):
        calls["translation"] += 1
        return TARGET

    async def rewrite(**kwargs):
        calls["rewrite"] += 1
        raise AssertionError(
            "Experimental ordinary translation must not rewrite accepted text"
        )

    monkeypatch.setenv("TRANSLATION_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setattr("src.core.llm_client.LLMClient", CountingClient)
    monkeypatch.setattr("src.core.translator.generate_translation_request", translate)
    monkeypatch.setattr("src.core.translator._make_refinement_request", rewrite)

    async def run(job_id):
        logs = []
        adapter = TxtAdapter(str(source), str(output), {"max_tokens_per_chunk": 1000})
        translator = GenericTranslator(adapter, manager, job_id)
        result = await translator.translate(
            source_language="English",
            target_language="Spanish",
            model_name="deepseek-v4-pro",
            llm_provider="deepseek",
            prompt_options=deepcopy(options),
            log_callback=lambda event, message, **data: logs.append((event, message)),
        )
        assert result, "\n".join(f"{event}: {message}" for event, message in logs)
        return result

    return run, calls, manager, output
