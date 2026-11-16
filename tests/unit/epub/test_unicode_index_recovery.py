"""Unicode index identities must survive translation and structural recovery."""

import unicodedata

import pytest

from src.core.epub import xhtml_translator
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.fidelity_supervisor import assess_fidelity


@pytest.mark.asyncio
@pytest.mark.parametrize("normalization", ["NFC", "NFD"])
@pytest.mark.parametrize("fallback", [False, True])
async def test_unicode_name_index_recovery_preserves_exact_source_without_model(
    monkeypatch, normalization, fallback
):
    source = unicodedata.normalize(
        normalization,
        "[id0]Canrobert, General François, 335[id1]"
        "Dvořák, Antonín, 45[id2]"
        "Łukasiewicz, Jan, 91[id3]",
    )
    tag_map = {
        "[id0]": '<p id="letter-c">',
        "[id1]": "</p><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }

    async def unexpected_model_call(*args, **kwargs):
        raise AssertionError("Source-backed index identities require no model")

    monkeypatch.setattr(
        xhtml_translator, "generate_translation_request", unexpected_model_call
    )
    monkeypatch.setattr(
        xhtml_translator, "supervise_fidelity", unexpected_model_call
    )
    arguments = dict(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={"_document_block_context": "index"},
        runtime_state={},
    )
    if fallback:
        result = await xhtml_translator._translate_structure_safe_fallback(
            **arguments
        )
    else:
        stats = TranslationMetrics(total_chunks=1)
        unit = {
            "status": "FAILED",
            "translation_status": "FAILED",
            "failure_reason": "target_language_gate",
        }
        result = await xhtml_translator.translate_chunk_with_fallback(
            **arguments,
            global_indices=[0, 1, 2, 3],
            stats=stats,
            unit_record=unit,
        )
        assert unit["translation_status"] == "COMPLETED"
        assert stats.processed_chunks == 1
    # Normalization is a classification aid, never an export transformation.
    assert result == source


@pytest.mark.parametrize("has_description", [False, True])
def test_language_gate_distinguishes_unicode_identities_from_descriptive_prose(
    has_description,
):
    lines = [
        "Canrobert, General François, 335",
        "Dvořák, Antonín, 45",
        "Łukasiewicz, Jan, 91",
    ]
    if has_description:
        lines.append("Elizabeth, Madame (sister to Louis XVI), 247")
    source = "\n".join(lines)
    decision = assess_fidelity(
        source,
        source,
        chunk_index=0,
        phase="translation_alignment_fallback",
        source_language="English",
        target_language="Spanish",
        prompt_options={"_document_block_context": "index"},
    )
    assert decision.accepted is not has_description
    if has_description:
        assert {issue.code for issue in decision.rejections} & {
            "untranslated_source",
            "target_language_missing",
        }
