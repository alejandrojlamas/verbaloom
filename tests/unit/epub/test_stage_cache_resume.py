"""Resume identities must separate request allowances from editorial meaning."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from lxml import etree

from src.core.epub import xhtml_translator as subject
from src.core.epub.unit_contract import (
    ensure_unit_records,
    invalidated_stages,
    mark_audited,
    mark_reviewed,
    prompt_versions,
    stage_fingerprints,
    text_sha256,
    with_profile_dependency,
)
from src.core.epub.xhtml_translation_state import XHTMLTranslationState
from src.core.llm.exceptions import ProviderActionRequiredError

SOURCE = "The traveler watched the sea."
TRANSLATION = "El viajero contemplaba el mar."
OPTIONS = {
    "refine": True,
    "audit_entire_book": True,
    "fidelity_adjudicate_rejections": False,
    "max_repair_rounds": 0,
}
# Captured from the production hash function before separating runtime limits.
LEGACY = {
    "translation": "51cfd6759409b9da1e018f934ae9f7f572f0a53e202ec31c5b36369ac43c139a",
    "review": "52704f7976e97d58342e29426aec6a398c41100efdc26f4f7188c9bbfa14a65f",
    "audit": "d368006107305ab619171dd9a4367ccee3ccfb8a9baac383f23821b27f35eb93",
}


def fingerprints(options=None, **overrides):
    return stage_fingerprints(
        **{
            "source_language": "English",
            "target_language": "Spanish",
            "model_name": "deepseek-v4-pro",
            "max_tokens_per_chunk": 450,
            "max_retries": 3,
            "prompt_options": OPTIONS if options is None else options,
            **overrides,
        }
    )


def root():
    return etree.fromstring(f"<html><body>{SOURCE}</body></html>")


def state(*, legacy=False, options=None):
    options = deepcopy(OPTIONS if options is None else options)
    stages = dict(LEGACY) if legacy else fingerprints(options)
    chunks = [{"text": SOURCE, "local_tag_map": {}, "global_indices": []}]
    record = ensure_unit_records(chunks, "Text/chapter.xhtml", fingerprints=stages)[0]
    mark_audited(
        record, TRANSLATION, review={"status": "pass"}, audit={"status": "pass"}
    )
    record.update(translation_attempts=2, review_attempts=1, audit_attempts=1)
    result = XHTMLTranslationState(
        file_path="Text/chapter.xhtml",
        file_href="Text/chapter.xhtml",
        translation_id="book-one",
        source_language="English",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=450,
        max_retries=3,
        chunks=chunks,
        global_tag_map={},
        placeholder_format=("[id", "]"),
        translated_chunks=[TRANSLATION],
        current_chunk_index=1,
        original_body_html="",
        doc_metadata={},
        stats={"total_chunks": 1, "processed_chunks": 1},
        created_at="2026-11-12T00:00:00Z",
        updated_at="2026-11-12T00:00:00Z",
        prompt_options=options,
        strict_contract=True,
        config_fingerprint=stages["translation"],
        stage_fingerprints=stages,
        prompt_versions_by_stage=prompt_versions(),
        source_document_hash=text_sha256(subject.extract_body_html(root())[0]),
    )
    assert result.validate()
    return result


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_repair_rounds", 4),
        ("epub_audit_repair_rounds", 3),
        ("epub_audit_max_attempts", 20),
        ("repair_until_pass", False),
        ("automatic_failure_recovery", True),
        ("automatic_failure_recovery_cycle", 15),
    ],
)
def test_operational_allowances_do_not_invalidate_completed_stages(name, value):
    assert (
        invalidated_stages(fingerprints(), fingerprints({**OPTIONS, name: value})) == ()
    )


def test_technical_retry_limit_does_not_change_editorial_identity():
    assert fingerprints(max_retries=1) == fingerprints(max_retries=5)


def test_nested_glossary_metadata_is_part_of_the_dependency_identity():
    base = {**OPTIONS, "glossary_term_metadata": {"Captain": {"policy": "translate"}}}
    changed = deepcopy(base)
    changed["glossary_term_metadata"]["Captain"]["policy"] = "preserve"
    assert invalidated_stages(fingerprints(base), fingerprints(changed)) == (
        "translation",
        "review",
        "audit",
    )


def test_output_token_allowance_is_semantic_not_a_secret():
    assert fingerprints({**OPTIONS, "max_output_tokens": 100}) != fingerprints(
        {
            **OPTIONS,
            "max_output_tokens": 1000,
        }
    )


def test_profile_content_change_is_not_hidden_by_same_profile_id():
    assert fingerprints(
        {**OPTIONS, "profile_id": "book", "_profile_content_hash": "old"}
    ) != fingerprints(
        {
            **OPTIONS,
            "profile_id": "book",
            "_profile_content_hash": "new",
        }
    )


def test_legacy_identity_matches_captured_hashes():
    assert fingerprints(_legacy_runtime_identity=True) == LEGACY


@pytest.mark.parametrize(
    "change", ["source", "model", "review_hash", "unit_hash", "options"]
)
def test_migration_rejects_unproven_state_without_mutation(change):
    saved = state(legacy=True)
    if change == "source":
        saved.source_document_hash = "other-document"
    elif change == "model":
        saved.model_name = "other-model"
    elif change == "review_hash":
        saved.stage_fingerprints["review"] = "corrupt"
    elif change == "unit_hash":
        saved.chunks[0]["unit"]["stage_fingerprints"]["audit"] = "corrupt"
    else:
        saved.prompt_options["custom_instructions"] = "Change the narrator."
    before = deepcopy(saved.to_dict())
    assert not saved.migrate_runtime_cache_identity(
        prompt_options=OPTIONS,
        current_stage_fingerprints=fingerprints(),
        source_document_hash=state().source_document_hash,
        current_max_retries=3,
    )
    assert saved.to_dict() == before


@pytest.mark.parametrize(
    "field,invalidated",
    [
        ("review_model", ["review", "audit"]),
        ("audit_model", ["audit"]),
    ],
)
def test_legacy_migration_preserves_narrow_editorial_invalidation(field, invalidated):
    saved = state(legacy=True)
    options = {**OPTIONS, field: "changed-model"}
    assert saved.migrate_runtime_cache_identity(
        prompt_options=options,
        current_stage_fingerprints=fingerprints(options),
        source_document_hash=saved.source_document_hash,
        current_max_retries=1,
    )
    record = saved.chunks[0]["unit"]
    assert record["cache_identity_migration"]["invalidated"] == invalidated
    assert record["translation"] == TRANSLATION
    assert record["translation_status"] == "COMPLETED"
    assert record["review_status"] == (
        "PENDING" if field == "review_model" else "COMPLETED"
    )
    assert record["audit_status"] == "PENDING"
    assert record["audit_attempts"] == 1


def test_changed_profile_reuses_text_as_unapproved_candidate():
    options = {**OPTIONS, "profile_id": "book", "_profile_content_hash": "old"}
    saved = state(options=options)
    current = {**options, "_profile_content_hash": "new"}
    assert saved.migrate_runtime_cache_identity(
        prompt_options=current,
        current_stage_fingerprints=fingerprints(current),
        source_document_hash=saved.source_document_hash,
        current_max_retries=3,
    )
    record = saved.chunks[0]["unit"]
    assert saved.translated_chunks == [TRANSLATION]
    assert record["review_status"] == record["audit_status"] == "PENDING"
    assert (
        record["cache_identity_migration"]["translation_retained_as_candidate"] is True
    )
    assert "profile_content_changed" in record["cache_identity_migration"]["reason"]


def test_profile_snapshot_covers_content_and_not_operational_limits(
    monkeypatch, tmp_path
):
    from dataclasses import replace

    from src.core.book_profiles import loader
    from src.core.book_profiles.models import BookProfile, ProfileGlossaryEntry

    profile = BookProfile(
        profile_id="book",
        name="Book",
        root=tmp_path,
        policy_text="Preserve rhythm.",
        raw_config={"max_repair_rounds": 2},
    )
    monkeypatch.setattr(loader, "load_book_profile", lambda *_: profile)
    base = with_profile_dependency({"profile_id": "book"})
    profile = replace(profile, max_repair_rounds=5, raw_config={"max_repair_rounds": 5})
    assert with_profile_dependency({"profile_id": "book"}) == base
    profile = replace(profile, policy_text="Preserve all narrator distinctions.")
    changed = with_profile_dependency({"profile_id": "book"})
    assert changed != base
    profile = replace(
        profile,
        glossary_entries=(
            ProfileGlossaryEntry("Captain", "capitan", status="approved"),
        ),
    )
    assert with_profile_dependency({"profile_id": "book"}) != changed
    assert with_profile_dependency(
        {"profile_id": "book", "_profile_content_hash": "forged"}
    ) != {
        "profile_id": "book",
        "_profile_content_hash": "forged",
    }


def test_nested_secrets_do_not_change_identity_but_glossary_values_do():
    first = {
        **OPTIONS,
        "provider_config": {"api_key": "YOUR_API_KEY_HERE", "max_output_tokens": 500},
    }
    second = deepcopy(first)
    second["provider_config"]["api_key"] = "<REDACTED>"
    assert fingerprints(first) == fingerprints(second)
    second["provider_config"]["max_output_tokens"] = 1000
    assert fingerprints(first) != fingerprints(second)


@pytest.mark.parametrize("term", ["key", "token", "secret", "password"])
def test_glossary_terms_named_like_credentials_are_still_dependencies(term):
    assert fingerprints({**OPTIONS, "glossary_terms": {term: "first"}}) != fingerprints(
        {
            **OPTIONS,
            "glossary_terms": {term: "second"},
        }
    )


async def resume(
    monkeypatch, saved, *, options=None, max_retries=3, saver=None, **kwargs
):
    calls = []

    async def unexpected_request(*args, **kw):
        calls.append(kw)
        raise ProviderActionRequiredError(
            "Unexpected duplicate request", reason="test_failure", provider="test"
        )

    monkeypatch.setattr(subject, "translate_chunk_with_fallback", unexpected_request)
    monkeypatch.setattr(subject, "supervise_fidelity", unexpected_request)
    client = SimpleNamespace(provider_type="deepseek", generate=unexpected_request)
    snapshots = []

    def save(job, href, value):
        snapshots.append(json.loads(json.dumps(value.to_dict())))
        return True if saver is None else saver(job, href, value)

    doc = root()
    success, _stats = await subject.translate_xhtml_simplified(
        doc,
        "English",
        "Spanish",
        "deepseek-v4-pro",
        client,
        max_tokens_per_chunk=450,
        max_retries=max_retries,
        prompt_options=dict(OPTIONS if options is None else options),
        resume_state=XHTMLTranslationState.from_dict(
            json.loads(json.dumps(saved.to_dict()))
        ),
        checkpoint_manager=SimpleNamespace(save_xhtml_partial_state=save),
        translation_id=kwargs.get("translation_id", "book-one"),
        file_href=kwargs.get("file_href", "Text/chapter.xhtml"),
    )
    assert success and doc.find("body").text == TRANSLATION
    assert calls == []
    return snapshots


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_entrypoint_preserves_approved_units_after_allowance_change(
    monkeypatch, legacy
):
    snapshots = await resume(
        monkeypatch,
        state(legacy=legacy),
        options={**OPTIONS, "max_repair_rounds": 2},
        max_retries=5,
    )
    record = snapshots[-1]["chunks"][0]["unit"]
    assert record["translation_attempts"] == 2
    assert record["review_attempts"] == 1
    assert record["audit_attempts"] == 1
    assert record["audit_status"] == "COMPLETED"


@pytest.mark.asyncio
async def test_legacy_migration_must_be_saved_before_reusing_approval(monkeypatch):
    with pytest.raises(ProviderActionRequiredError) as error:
        await resume(
            monkeypatch, state(legacy=True), max_retries=4, saver=lambda *_: False
        )
    assert error.value.reason == "checkpoint_write_failed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identity",
    [
        {"translation_id": "book-two"},
        {"file_href": "Text/other.xhtml"},
    ],
)
async def test_identical_text_cannot_reuse_another_job_or_document(
    monkeypatch, identity
):
    with pytest.raises(
        ProviderActionRequiredError, match="Unexpected duplicate request"
    ):
        await resume(monkeypatch, state(), **identity)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_real_checkpoint_resume_repairs_only_the_pending_candidate(
    monkeypatch, tmp_path, legacy
):
    from src.core.epub.exceptions import ChunkTranslationFailedError
    from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
    from src.core.llm.base import LLMResponse
    from src.persistence.checkpoint_manager import CheckpointManager

    monkeypatch.chdir(tmp_path)
    manager = CheckpointManager(db_path=str(tmp_path / "jobs.db"))
    saved = state(legacy=legacy)
    record = saved.chunks[0]["unit"]
    mark_reviewed(record, TRANSLATION)
    repaired = "El viajero observaba el mar."
    calls = []

    async def audit(source, candidate, **kwargs):
        calls.append(("audit", candidate))
        accepted = candidate == repaired
        return FidelityDecision(
            chunk_index=1,
            phase=kwargs["phase"],
            section="chapter",
            accepted=accepted,
            source_hash=text_sha256(source),
            candidate_hash=text_sha256(candidate),
            judge_decision="pass" if accepted else "fail",
            judge_changed_facts=[] if accepted else ["Mock localized defect"],
            issues=[]
            if accepted
            else [FidelityIssue("fidelity_judge_reject", "reject", "Defect")],
        ), LLMResponse(content="mock judgment")

    async def repair(**kwargs):
        calls.append(("repair", kwargs["candidate_text"]))
        return repaired

    async def forbidden(*args, **kwargs):
        pytest.fail("Saved translation and review must not be regenerated")

    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    monkeypatch.setattr(subject, "_repair_audited_candidate", repair)
    monkeypatch.setattr(subject, "translate_chunk_with_fallback", forbidden)
    client = SimpleNamespace(provider_type="deepseek", generate=forbidden)
    try:
        with pytest.raises(ChunkTranslationFailedError):
            await subject._audit_epub_chunks(
                saved.translated_chunks,
                saved.chunks,
                source_language="English",
                target_language="Spanish",
                model_name=saved.model_name,
                llm_client=client,
                placeholder_format=saved.placeholder_format,
                prompt_options=OPTIONS,
            )
        assert record["audit_repair"]["stop_reason"] == "repair_attempts_exhausted"
        assert manager.save_xhtml_partial_state("book-one", saved.file_href, saved)
        loaded = manager.load_xhtml_partial_state("book-one", saved.file_href)
        assert loaded is not None and loaded.validate()
        doc = root()
        success, _ = await subject.translate_xhtml_simplified(
            doc,
            "English",
            "Spanish",
            saved.model_name,
            client,
            max_tokens_per_chunk=450,
            max_retries=5,
            prompt_options={**OPTIONS, "max_repair_rounds": 1},
            resume_state=loaded,
            checkpoint_manager=manager,
            translation_id="book-one",
            file_href=saved.file_href,
        )
        assert success and doc.find("body").text == repaired
        assert calls == [
            ("audit", TRANSLATION),
            ("repair", TRANSLATION),
            ("audit", repaired),
        ]
        final = manager.load_xhtml_partial_state("book-one", saved.file_href)
        assert final is not None and final.validate()
        record = final.chunks[0]["unit"]
        assert record["translation_attempts"] == 2 and record["review_attempts"] == 1
        assert record["audit_attempts"] == 3 and record["audit_repair_attempts"] == 1
        assert record["audit_status"] == "COMPLETED"
        assert len(record["audit_repair"].get("contract_migrations", [])) == int(legacy)
    finally:
        manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refine", [False, True])
async def test_entrypoint_profile_edit_cannot_reuse_old_approval(
    monkeypatch, tmp_path, refine
):
    from dataclasses import replace

    from src.core.book_profiles import loader
    from src.core.book_profiles.models import BookProfile
    from src.core.epub.exceptions import ChunkTranslationFailedError
    from src.core.fidelity_supervisor import FidelityDecision
    from src.core.llm.base import LLMResponse

    profile = BookProfile(
        profile_id="book", name="Book", root=tmp_path, policy_text="Old policy"
    )
    monkeypatch.setattr(loader, "load_book_profile", lambda *_: profile)
    options = {**OPTIONS, "profile_id": "book", "refine": refine}
    saved = state(options=with_profile_dependency(options))
    profile = replace(profile, policy_text="New policy")
    calls = []

    async def forbidden(*args, **kwargs):
        pytest.fail("Profile edits must retain the previous translation as a candidate")

    async def review(**kwargs):
        calls.append("review")
        for chunk, text in zip(kwargs["chunks"], kwargs["translated_chunks"]):
            assert chunk["unit"]["review_status"] == "PENDING"
            assert chunk["unit"]["audit_status"] == "PENDING"
            mark_reviewed(chunk["unit"], text)
        return kwargs["translated_chunks"]

    async def audit(source, candidate, **kwargs):
        calls.append("audit")
        return FidelityDecision(
            chunk_index=1,
            phase=kwargs["phase"],
            section="chapter",
            accepted=True,
            source_hash=text_sha256(source),
            candidate_hash=text_sha256(candidate),
            judge_decision="pass",
        ), LLMResponse(content="mock judgment")

    monkeypatch.setattr(subject, "translate_chunk_with_fallback", forbidden)
    monkeypatch.setattr(subject, "_refine_epub_chunks", review)
    monkeypatch.setattr(subject, "supervise_fidelity", audit)
    doc = root()
    task = subject.translate_xhtml_simplified(
        doc,
        "English",
        "Spanish",
        saved.model_name,
        SimpleNamespace(provider_type="deepseek", generate=forbidden),
        max_tokens_per_chunk=450,
        prompt_options=options,
        resume_state=saved,
        translation_id="book-one",
        file_href=saved.file_href,
    )
    if refine:
        success, _ = await task
        assert success and calls == ["review", "audit"]
    else:
        with pytest.raises(ChunkTranslationFailedError) as caught:
            await task
        assert caught.value.reason == "profile_change_requires_review"
        assert doc.find("body").text == SOURCE and calls == []
