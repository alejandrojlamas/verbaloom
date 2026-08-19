from __future__ import annotations

from src.core.epub.unit_contract import (
    EPUB_PIPELINE_VERSION,
    TranslationUnitStatus,
    config_fingerprint,
    ensure_unit_records,
    invalidated_stages,
    invalidate_unit_stages,
    mark_attempt,
    mark_audited,
    mark_failed,
    mark_reviewed,
    mark_translated,
    stage_fingerprints,
    validate_publishable_units,
    validate_resume_prefix,
)


def _chunks():
    return [
        {"text": "[id0]Guten Morgen.[id1]", "local_tag_map": {}, "global_indices": []},
        {"text": "[id0]Weiter gingen wir.[id1]", "local_tag_map": {}, "global_indices": []},
    ]


def test_unit_ids_and_hashes_are_stable_and_file_scoped():
    first = _chunks()
    second = _chunks()

    first_records = ensure_unit_records(first, "Text/chapter.xhtml")
    second_records = ensure_unit_records(second, "Text/chapter.xhtml")

    assert [item["unit_id"] for item in first_records] == [item["unit_id"] for item in second_records]
    assert all(item["source_hash"] for item in first_records)
    assert len({item["unit_id"] for item in first_records}) == 2


def test_publish_contract_requires_every_unit_to_be_audited():
    chunks = _chunks()
    records = ensure_unit_records(chunks, "Text/chapter.xhtml")
    mark_attempt(records[0])
    mark_audited(records[0], "[id0]Buenos días.[id1]")
    mark_attempt(records[1])
    mark_failed(records[1], "provider_timeout")

    publishable, errors = validate_publishable_units(chunks)

    assert publishable is False
    assert any("FAILED" in error for error in errors)


def test_resume_prefix_rejects_non_audited_or_tampered_units():
    chunks = _chunks()
    records = ensure_unit_records(chunks, "Text/chapter.xhtml")
    translated = ["[id0]Buenos días.[id1]"]
    mark_audited(records[0], translated[0])

    assert validate_resume_prefix(chunks, translated, 1) == (True, "")

    records[0]["translation_hash"] = "stale"
    valid, reason = validate_resume_prefix(chunks, translated, 1)
    assert valid is False
    assert "translation hash" in reason


def test_config_fingerprint_changes_with_pipeline_relevant_inputs_only():
    base = config_fingerprint(
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=1400,
        max_retries=3,
        prompt_options={"target_locale": "es-MX", "deepseek_api_key": "secret-a"},
    )
    same_without_secret = config_fingerprint(
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=1400,
        max_retries=3,
        prompt_options={"target_locale": "es-MX", "deepseek_api_key": "secret-b"},
    )
    changed = config_fingerprint(
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=900,
        max_retries=3,
        prompt_options={"target_locale": "es-MX"},
    )

    assert EPUB_PIPELINE_VERSION
    assert base == same_without_secret
    assert base != changed
    assert TranslationUnitStatus.AUDITED.value == "AUDITED"


def _fingerprints(options):
    return stage_fingerprints(
        source_language="German",
        target_language="Spanish",
        model_name="deepseek-v4-pro",
        max_tokens_per_chunk=1400,
        max_retries=3,
        prompt_options=options,
    )


def test_stage_fingerprints_invalidate_only_the_required_suffix():
    base = _fingerprints({"target_locale": "es-MX", "review_model": "review-v1", "audit_model": "audit-v1"})
    review_changed = _fingerprints({"target_locale": "es-MX", "review_model": "review-v2", "audit_model": "audit-v1"})
    audit_changed = _fingerprints({"target_locale": "es-MX", "review_model": "review-v1", "audit_model": "audit-v2"})
    translation_changed = _fingerprints({"target_locale": "es-ES", "review_model": "review-v1", "audit_model": "audit-v1"})

    assert invalidated_stages(base, base) == ()
    assert invalidated_stages(base, review_changed) == ("review", "audit")
    assert invalidated_stages(base, audit_changed) == ("audit",)
    assert invalidated_stages(base, translation_changed) == ("translation", "review", "audit")


def test_unit_stage_invalidation_preserves_valid_earlier_results():
    chunks = _chunks()
    record = ensure_unit_records(chunks, "Text/chapter.xhtml")[0]
    mark_translated(record, "Buenos días.")
    mark_reviewed(record, "Muy buenos días.", review={"status": "pass"})
    mark_audited(record, "Muy buenos días.", audit={"status": "pass"})

    invalidate_unit_stages(record, ("audit",))
    assert record["translation"] == "Muy buenos días."
    assert record["review_status"] == "COMPLETED"
    assert record["audit_status"] == "PENDING"
    assert record["status"] == "REVIEWED"

    invalidate_unit_stages(record, ("review", "audit"))
    assert record["translation"] == "Muy buenos días."
    assert record["translation_status"] == "COMPLETED"
    assert record["review_status"] == "PENDING"
    assert record["status"] == "TRANSLATED"
