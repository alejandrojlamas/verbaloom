from __future__ import annotations

import pytest

from src import config
from src.core.epub import xhtml_translator
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.structure_safe_fallback import (
    split_structure_safe_parts,
    structure_signature,
)
from src.core.epub.token_alignment_fallback import (
    TokenAlignmentFallback,
    UnsafeStructuralAlignmentError,
)


def _translate_marked_payload(text, translations):
    if text in translations:
        return translations[text]
    result = text
    for source, target in sorted(
        translations.items(), key=lambda item: len(item[0]), reverse=True
    ):
        result = result.replace(source, target)
    return result


def test_epub_errors_reach_active_job_log():
    events = []

    xhtml_translator._log_error(
        lambda event, message: events.append((event, message)),
        "phase2_error",
        "structural recovery failed",
    )

    assert events == [("phase2_error", "structural recovery failed")]


def test_document_context_detects_critical_apparatus_from_readable_text():
    source = (
        "[id0]Kenneth J. Harris, K. Michel Kacmer, Suzanne Zivnuska, and "
        'Jason D. Shaw (2007), “The Impact of Political Skill on Impression '
        'Management Effectiveness,” [id1]Journal of Applied Psychology, 92'
        "[id2] (1), 278–285.[id3]"
    )
    tag_map = {
        "[id0]": "<p>",
        "[id1]": "<i>",
        "[id2]": "</i>",
        "[id3]": "</p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "text/Notes.xhtml",
        )
        == "critical_apparatus"
    )


def test_document_context_detects_further_reading_filename_and_reference_class():
    source = (
        "[id0]Interpersonal Peacemaking: Confrontations and Third-Party "
        "Consultation.[id1] Reading, Mass.: Addison-Wesley, 1969.[id2]"
        "Weisbord, Marvin.[id3]Discovering Common Ground.[id4] "
        "San Francisco: Berrett-Koehler, 1992.[id5]"
    )
    tag_map = {
        "[id0]": '<p class="reference"><span class="italic">',
        "[id1]": "</span>",
        "[id2]": '</p><p class="reference">',
        "[id3]": '<span class="italic">',
        "[id4]": "</span>",
        "[id5]": "</p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "xhtml/FurtherReading.html",
        )
        == "critical_apparatus"
    )


def test_document_context_detects_legacy_comma_delimited_bibliography():
    source = (
        "[id0]S. A. Handford, Penguin, 1951[id1]"
        "Cameron, James, What a Way to Run the Tribe, Macmillan, 1968[id2]"
        "Churchill, Winston, My Early Life, Heinemann, 1930[id3]"
    )
    tag_map = {
        "[id0]": "<p>",
        "[id1]": "</p><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "main-13.xhtml",
        )
        == "critical_apparatus"
    )


def test_document_context_detects_unlabelled_analytical_index():
    source = (
        "[id0]Bramwell, James G., 595[id1]"
        "Bride, Harold, 435[id2]"
        "Buckingham, Duchess of (1620), 172[id3]"
        "Byron, George Gordon, Lord, 302[id4]"
        "Campbell, Sir Colin, 339,347[id5]"
    )
    tag_map = {
        "[id0]": "<p>",
        "[id1]": "</p><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p><p>",
        "[id4]": "</p><p>",
        "[id5]": "</p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "main-13.xhtml",
        )
        == "index"
    )


@pytest.mark.asyncio
async def test_structure_recovery_inherits_critical_apparatus_context(monkeypatch):
    source = (
        "[id0]S. A. Handford, Penguin, 1951[id1]"
        "Cameron, James, What a Way to Run the Tribe, Macmillan, 1968[id2]"
    )
    tag_map = {
        "[id0]": "<p>",
        "[id1]": "</p><p>",
        "[id2]": "</p>",
    }
    observed_options = []

    async def fake_request(text, **kwargs):
        observed_options.append(dict(kwargs.get("prompt_options") or {}))
        return text

    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        fake_request,
    )

    result = await xhtml_translator._translate_structure_safe_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={"_document_block_context": "critical_apparatus"},
        runtime_state={},
    )

    assert result == source
    assert observed_options
    assert all(
        options.get("_document_block_context") == "critical_apparatus"
        for options in observed_options
    )
    assert all(
        "BIBLIOGRAPHY/NOTES POLICY" in options.get("custom_instructions", "")
        for options in observed_options
    )


@pytest.mark.asyncio
async def test_structure_recovery_preserves_identity_only_index_without_model(
    monkeypatch,
):
    source = (
        "[id0]Bramwell, James G., 595[id1]"
        "Bride, Harold, 435[id2]"
        "Campbell, Sir Colin, 339,347[id3]"
    )
    tag_map = {
        "[id0]": "<p>",
        "[id1]": "</p><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }

    async def unexpected_request(*_args, **_kwargs):
        raise AssertionError("identity-only index entries must not call the model")

    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        unexpected_request,
    )

    result = await xhtml_translator._translate_structure_safe_fallback(
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

    assert result == source


def test_document_context_detects_index_from_xhtml_class_names():
    source = (
        "[id0]Granovetter, Mark[id1]Great Society[id2]"
        "Handy Dan (home improvement company)[id3]"
    )
    tag_map = {
        "[id0]": '<p class="index_p">',
        "[id1]": '</p><p class="index_p">',
        "[id2]": '</p><p class="index_dis">',
        "[id3]": "</p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "text/Index.xhtml",
        )
        == "index"
    )


def test_document_context_detects_glossary_from_lexical_records():
    source = (
        "[id0]Acastus[id1]a-kas´-tus[id2]): king of Dulichium.14.340."
        "[id3]Achaean[id4]a-kee´-an[id5]): inhabitants of Achaea.1.272."
        "[id6]Acheron[id7]a´-ker-on[id8]): a mythical river.10.516."
    )
    tag_map = {
        "[id0]": "<p>", "[id1]": "<b>", "[id2]": "</b></p>",
        "[id3]": "<p>", "[id4]": "<b>", "[id5]": "</b></p>",
        "[id6]": "<p>", "[id7]": "<b>", "[id8]": "</b></p>",
    }

    assert (
        xhtml_translator._structure_recovery_document_context(
            tag_map,
            source,
            "text/appendix.xhtml",
        )
        == "glossary"
    )


def test_glossary_translation_policy_is_bounded_and_not_duplicated():
    options = {"custom_instructions": "Keep source order."}

    first = xhtml_translator._translation_options_for_document_context(
        options,
        "glossary",
    )
    second = xhtml_translator._translation_options_for_document_context(
        first,
        "glossary",
    )

    assert first["_document_block_context"] == "glossary"
    assert "LEXICAL GLOSSARY POLICY" in first["custom_instructions"]
    assert second["custom_instructions"].count("LEXICAL GLOSSARY POLICY") == 1


def test_index_translation_policy_is_bounded_and_not_duplicated():
    options = {"custom_instructions": "Keep the source order."}

    first = xhtml_translator._translation_options_for_document_context(
        options,
        "index",
    )
    second = xhtml_translator._translation_options_for_document_context(
        first,
        "index",
    )

    assert first["_document_block_context"] == "index"
    assert "ANALYTICAL INDEX/CATALOG POLICY" in first["custom_instructions"]
    assert second["custom_instructions"].count(
        "ANALYTICAL INDEX/CATALOG POLICY"
    ) == 1


@pytest.mark.asyncio
async def test_translation_retry_changes_prompt_after_language_gate_rejection(
    monkeypatch,
):
    source = "[id0]The role description.[id1]"
    tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
    calls = []

    async def fake_request(text, **kwargs):
        options = kwargs["prompt_options"]
        calls.append(str(options.get("custom_instructions") or ""))
        if len(calls) == 1:
            options["_last_target_language_gate_rejection"] = {
                "issues": [{"code": "target_language_missing"}],
            }
            return None
        return "[id0]La descripción del puesto.[id1]"

    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        fake_request,
    )

    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        global_indices=[0, 1],
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=TranslationMetrics(total_chunks=1),
        max_retries=2,
        prompt_options={"fidelity_supervisor_mode": "off"},
        runtime_state={},
        unit_record={},
    )

    assert result == "[id0]La descripción del puesto.[id1]"
    assert calls[0] == ""
    assert "TARGET-LANGUAGE RECOVERY" in calls[1]
    assert "target_language_missing" in calls[1]


@pytest.mark.asyncio
async def test_structure_recovery_logs_gate_reason_and_block(monkeypatch):
    events = []

    async def fake_request(_text, **kwargs):
        kwargs["prompt_options"]["_last_target_language_gate_rejection"] = {
            "issues": [
                {
                    "code": "source_language_residual",
                    "detail": "residual phrase remained",
                }
            ]
        }
        return None

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)

    with pytest.raises(
        ValueError,
        match="empty or invalid structural block",
    ):
        await xhtml_translator._translate_structure_safe_fallback(
            chunk_text="[id0]Source paragraph.[id1]",
            local_tag_map={"[id0]": "<p>", "[id1]": "</p>"},
            source_language="English",
            target_language="Spanish",
            model_name="test",
            llm_client=object(),
            log_callback=lambda event, message: events.append((event, message)),
            context_manager=None,
            prompt_options={},
            runtime_state={},
        )

    rejection = next(
        message
        for event, message in events
        if event == "phase2_structure_safe_candidate_rejected"
    )
    assert "bloque(s) 1" in rejection
    assert "source_language_residual: residual phrase remained" in rejection
    assert "Source paragraph." in rejection


@pytest.mark.asyncio
async def test_structure_recovery_reserves_budget_for_later_batches(monkeypatch):
    source = "".join(
        f"[id{index}]Company {index}" for index in range(6)
    ) + "[id6]"
    tag_map = {
        f"[id{index}]": (
            "<table><tr><td>" if index == 0 else "</td></tr><tr><td>"
        )
        for index in range(6)
    }
    tag_map["[id6]"] = "</td></tr></table>"
    calls = []

    async def fake_request(text, **_kwargs):
        calls.append(text)
        if len(calls) == 1:
            return None
        return _translate_marked_payload(
            text,
            {f"Company {index}": f"Empresa {index}" for index in range(6)},
        )

    monkeypatch.setattr(config, "EPUB_STRUCTURE_RECOVERY_BATCH_SIZE", 2)
    monkeypatch.setattr(config, "EPUB_STRUCTURE_RECOVERY_MAX_CALLS", 2)
    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        fake_request,
    )

    result = await xhtml_translator._translate_structure_safe_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={},
        runtime_state={},
    )

    assert len(calls) == 5
    assert "Empresa 0" in result
    assert "Empresa 5" in result
    assert structure_signature(result, tag_map) == structure_signature(
        source,
        tag_map,
    )


@pytest.mark.asyncio
async def test_structure_recovery_rejects_marker_only_candidate(monkeypatch):
    events = []

    async def fake_request(_text, **_kwargs):
        return "[[[VERBALOOMBLOCK000]]]\n[[[/VERBALOOMBLOCK000]]]"

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)

    with pytest.raises(ValueError, match="empty or invalid"):
        await xhtml_translator._translate_structure_safe_fallback(
            chunk_text=(
                "[id0]This source paragraph contains enough words to prove that "
                "an empty marked response cannot be accepted as a translation "
                "without losing nearly all of the original content.[id1]"
            ),
            local_tag_map={"[id0]": "<p>", "[id1]": "</p>"},
            source_language="English",
            target_language="Spanish",
            model_name="test",
            llm_client=object(),
            log_callback=lambda event, message: events.append((event, message)),
            context_manager=None,
            prompt_options={},
            runtime_state={},
        )

    assert any(
        event == "phase2_structure_safe_empty_or_short_block"
        for event, _message in events
    )


def test_structure_parts_keep_block_boundaries_out_of_content_runs():
    text = "[id0]Title[id1]First paragraph.[id2]Second paragraph.[id3]"
    tag_map = {
        "[id0]": "<h1>",
        "[id1]": "</h1><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }

    parts = split_structure_safe_parts(text, tag_map)

    assert [part.kind for part in parts] == [
        "structure", "content", "structure", "content", "structure", "content", "structure"
    ]
    assert [part.text for part in parts if part.kind == "content"] == [
        "Title", "First paragraph.", "Second paragraph."
    ]


def test_semantic_audit_text_keeps_block_and_inline_boundaries_readable():
    source = (
        "GO TO NOTE REFERENCE IN TEXT[id0]"
        "They set their sights:[id1]OpenAI[id2]reference."
    )
    tag_map = {
        "[id0]": "</a><p>",
        "[id1]": "</p><p><a>",
        "[id2]": "</a>",
    }

    semantic = xhtml_translator._semantic_text_from_placeholder_stream(
        source,
        tag_map,
    )

    assert semantic == (
        "GO TO NOTE REFERENCE IN TEXT\n"
        "They set their sights:\n"
        "OpenAI reference."
    )
    assert "TEXTThey" not in semantic
    assert "sights:OpenAI" not in semantic


def test_audited_identity_repair_restores_only_project_names():
    decision = FidelityDecision(
        chunk_index=12,
        phase="final_epub_unit_audit",
        section="notes.xhtml",
        accepted=False,
        judge_reason=(
            "The project names were translated and must remain exact code names."
        ),
        judge_issues=["project_name_translated"],
        judge_changed_facts=[
            "'Crab Generation' translated as 'Generación de Cangrejo'",
            "'Crab Paraphrase' translated as 'Paráfrasis de Cangrejo'",
        ],
    )
    source = (
        "[id0]Review of Crab Generation instructions.[id1]"
        "[id2]Copy of Crab Paraphrase instructions.[id3]"
    )
    candidate = (
        "[id0]Revisión de las instrucciones de Generación de Cangrejo.[id1]"
        "[id2]Copia de las instrucciones de Paráfrasis de Cangrejo.[id3]"
    )

    repaired, restored = xhtml_translator._restore_audited_identity_names(
        source,
        candidate,
        decision,
    )

    assert repaired == (
        "[id0]Revisión de las instrucciones de Crab Generation.[id1]"
        "[id2]Copia de las instrucciones de Crab Paraphrase.[id3]"
    )
    assert restored == ["Crab Generation", "Crab Paraphrase"]
    assert "Review of" not in repaired
    assert "Copy of" not in repaired


def test_audited_identity_repair_restores_official_institute_name():
    decision = FidelityDecision(
        chunk_index=48,
        phase="final_epub_unit_audit",
        section="index.xhtml",
        accepted=False,
        judge_reason=(
            "The DAIR acronym expansion was translated, which alters the "
            "cited institute name."
        ),
        judge_issues=["institute name translated instead of preserved"],
        judge_changed_facts=[
            "DAIR expansion changed from 'Distributed AI Research Institute' "
            "to 'Instituto de Investigación de IA Distribuida'"
        ],
    )
    source = (
        "[id0]cybersecurity[id1]Cyc,[id2]"
        "DAIR (Distributed AI Research Institute),[id3]"
    )
    candidate = (
        "[id0]ciberseguridad[id1]Cyc,[id2]"
        "DAIR (Instituto de Investigación de IA Distribuida),[id3]"
    )

    repaired, restored = xhtml_translator._restore_audited_identity_names(
        source,
        candidate,
        decision,
    )

    assert repaired == (
        "[id0]ciberseguridad[id1]Cyc,[id2]"
        "DAIR (Distributed AI Research Institute),[id3]"
    )
    assert restored == ["Distributed AI Research Institute"]


def test_audited_identity_repair_restores_source_author_spelling():
    decision = FidelityDecision(
        chunk_index=10,
        phase="translation",
        section="notes.xhtml",
        accepted=False,
        judge_reason=(
            "The candidate changes the author name 'Any Cuddy' to 'Amy Cuddy', "
            "which is a factual error."
        ),
        judge_issues=["author name changed"],
        judge_changed_facts=["Author name 'Any Cuddy' changed to 'Amy Cuddy'"],
    )

    repaired, restored = xhtml_translator._restore_audited_identity_names(
        "[id0]Any Cuddy (2009), cited work.[id1]",
        "[id0]Amy Cuddy (2009), obra citada.[id1]",
        decision,
    )

    assert repaired == "[id0]Any Cuddy (2009), obra citada.[id1]"
    assert restored == ["Any Cuddy"]


def test_audited_identity_repair_does_not_reverse_place_localization():
    decision = FidelityDecision(
        chunk_index=1,
        phase="final_epub_unit_audit",
        section="chapter.xhtml",
        accepted=False,
        judge_reason="The place name was localized.",
        judge_changed_facts=["'United States' translated as 'Estados Unidos'"],
    )

    repaired, restored = xhtml_translator._restore_audited_identity_names(
        "She returned to the United States.",
        "Regresó a Estados Unidos.",
        decision,
    )

    assert repaired == "Regresó a Estados Unidos."
    assert restored == []


def test_audited_identity_repair_handles_proper_noun_wording():
    decision = FidelityDecision(
        chunk_index=12,
        phase="translation_alignment_fallback",
        section="index.xhtml",
        accepted=False,
        judge_reason=(
            "Great Society was translated even though it is a proper noun."
        ),
        judge_changed_facts=[
            "'Great Society' -> 'Gran Sociedad': proper noun translated"
        ],
    )

    repaired, restored = xhtml_translator._restore_audited_identity_names(
        "Grant, Adam Great Society Green Bay Packers",
        "Grant, Adam Gran Sociedad Green Bay Packers",
        decision,
    )

    assert repaired == "Grant, Adam Great Society Green Bay Packers"
    assert restored == ["Great Society"]


def test_audited_identity_prefix_repair_removes_only_source_proven_article():
    decision = FidelityDecision(
        chunk_index=12,
        phase="translation_alignment_fallback",
        section="index.xhtml",
        accepted=False,
        judge_added_not_in_source=["Added 'The' before 'Guardian'"],
    )

    repaired, removed = (
        xhtml_translator._strip_audited_identity_prefix_additions(
            "Gruenfeld, Deborah Guardian H",
            "Gruenfeld, Deborah The Guardian H",
            decision,
        )
    )

    assert repaired == "Gruenfeld, Deborah Guardian H"
    assert removed == ["The"]


def test_audited_url_repair_removes_only_source_proven_duplicate_labels():
    decision = FidelityDecision(
        chunk_index=9,
        phase="final_epub_unit_audit",
        section="notes.xhtml",
        accepted=False,
        judge_reason="The candidate damages two URLs.",
        judge_issues=[
            "structure_issues: URL 'openai openai.com/...' is damaged",
            "structure_issues: URL 'theguardian theguardian.com/...' is damaged",
        ],
    )
    source = (
        "OpenAI, [id0]openai.com/index/openai-and-journalism[id1] "
        "Guardian, [id2]theguardian.com/technology/article[id3]"
    )
    candidate = (
        "OpenAI, openai[id0]openai.com/index/openai-and-journalism[id1] "
        "Guardian, theguardian[id2]theguardian.com/technology/article[id3]"
    )

    repaired, removed = xhtml_translator._strip_audited_duplicate_domain_labels(
        source,
        candidate,
        decision,
    )

    assert repaired == source
    assert removed == ["openai", "theguardian"]


def test_audited_url_repair_requires_audit_and_source_evidence():
    decision = FidelityDecision(
        chunk_index=1,
        phase="final_epub_unit_audit",
        section="notes.xhtml",
        accepted=False,
        judge_reason="The candidate changes punctuation.",
    )
    source = "OpenAI, [id0]example.com/reference[id1]"
    candidate = "OpenAI[id0]openai.com/reference[id1]"

    repaired, removed = xhtml_translator._strip_audited_duplicate_domain_labels(
        source,
        candidate,
        decision,
    )

    assert repaired == candidate
    assert removed == []


@pytest.mark.asyncio
async def test_audited_candidate_repair_restores_identity_without_llm(monkeypatch):
    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("deterministic identity repair should avoid an LLM call")

    monkeypatch.setattr(
        xhtml_translator,
        "_generate_alert_repair",
        fail_if_called,
    )
    decision = FidelityDecision(
        chunk_index=12,
        phase="final_epub_unit_audit",
        section="notes.xhtml",
        accepted=False,
        judge_reason=(
            "The project names were translated and must remain exact code names."
        ),
        judge_issues=["project_name_translated"],
        judge_changed_facts=[
            "'Crab Generation' translated as 'Generación de Cangrejo'",
            "'Crab Paraphrase' translated as 'Paráfrasis de Cangrejo'",
        ],
    )
    source = (
        "[id0]Review of Crab Generation instructions.[id1]"
        "[id2]Copy of Crab Paraphrase instructions.[id3]"
    )
    candidate = (
        "[id0]Revisión de las instrucciones de Generación de Cangrejo.[id1]"
        "[id2]Copia de las instrucciones de Paráfrasis de Cangrejo.[id3]"
    )

    repaired = await xhtml_translator._repair_audited_candidate(
        source_text=source,
        candidate_text=candidate,
        decision=decision,
        chunk={
            "local_tag_map": {
                "[id0]": "<p>",
                "[id1]": "</p>",
                "[id2]": "<p>",
                "[id3]": "</p>",
            },
            "global_indices": [10, 11, 12, 13],
        },
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        prompt_options={},
        placeholder_format=("[id", "]"),
    )

    assert repaired == (
        "[id10]Revisión de las instrucciones de Crab Generation.[id11]"
        "[id12]Copia de las instrucciones de Crab Paraphrase.[id13]"
    )


@pytest.mark.asyncio
async def test_audited_candidate_repair_cleans_urls_before_semantic_repair(
    monkeypatch,
):
    captured = {}

    async def fake_repair(_client, prompt, _system_prompt, **_kwargs):
        captured["prompt"] = prompt
        return type(
            "Response",
            (),
            {
                "content": (
                    f"{config.TRANSLATE_TAG_IN}"
                    "[id0]Quien redactó el texto de la empresa:[id1] "
                    "[id2]openai.com/path[id3]"
                    f"{config.TRANSLATE_TAG_OUT}"
                )
            },
        )()

    monkeypatch.setattr(
        xhtml_translator,
        "_generate_alert_repair",
        fake_repair,
    )
    decision = FidelityDecision(
        chunk_index=9,
        phase="final_epub_unit_audit",
        section="notes.xhtml",
        accepted=False,
        judge_reason=(
            "The candidate changes author gender and damages a URL."
        ),
        judge_issues=[
            "changed_facts: neutral author becomes feminine",
            "structure_issues: URL 'openai openai.com/...' is damaged",
        ],
        judge_changed_facts=[
            "Source author is gender-neutral; candidate adds feminine gender."
        ],
    )

    repaired = await xhtml_translator._repair_audited_candidate(
        source_text=(
            "[id0]The author of the company's:[id1] "
            "[id2]openai.com/path[id3]"
        ),
        candidate_text=(
            "[id0]La autora de la empresa:[id1] "
            "openai[id2]openai.com/path[id3]"
        ),
        decision=decision,
        chunk={
            "local_tag_map": {
                "[id0]": "<p>",
                "[id1]": "</p>",
                "[id2]": "<a>",
                "[id3]": "</a>",
            },
            "global_indices": [20, 21, 22, 23],
        },
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        prompt_options={},
        placeholder_format=("[id", "]"),
    )

    assert "openai[id2]openai.com" not in captured["prompt"]
    assert "[id2]openai.com/path[id3]" in captured["prompt"]
    assert repaired == (
        "[id20]Quien redactó el texto de la empresa:[id21] "
        "[id22]openai.com/path[id23]"
    )


def test_proportional_alignment_refuses_block_level_tag_map():
    with pytest.raises(UnsafeStructuralAlignmentError):
        TokenAlignmentFallback().align_and_insert_placeholders(
            "[id0]Heading[id1]Body[id2]",
            "Título Cuerpo",
            ["[id0]", "[id1]", "[id2]"],
            tag_map={
                "[id0]": "<h1>",
                "[id1]": "</h1><p>",
                "[id2]": "</p>",
            },
        )


@pytest.mark.asyncio
async def test_structure_safe_translation_cannot_move_text_between_blocks(monkeypatch):
    source = "[id0]Title[id1]First paragraph.[id2]Second paragraph.[id3]"
    tag_map = {
        "[id0]": "<h1>",
        "[id1]": "</h1><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }
    translations = {
        "Title": "Título",
        "First paragraph.": "Primer párrafo.",
        "Second paragraph.": "Segundo párrafo.",
    }

    async def fake_request(text, **_kwargs):
        return _translate_marked_payload(text, translations)

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    result = await xhtml_translator._translate_structure_safe_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={},
        runtime_state={},
    )

    assert result == "[id0]Título[id1]Primer párrafo.[id2]Segundo párrafo.[id3]"
    assert structure_signature(result, tag_map) == structure_signature(source, tag_map)


def test_placeholder_cap_splits_front_matter_before_token_limit():
    paragraphs = [f"[id{i}]Paragraph {i}." for i in range(60)]
    text = "".join(paragraphs) + "[id60]"
    tag_map = {
        f"[id{i}]": ("<p>" if i == 0 else "</p><p>")
        for i in range(60)
    }
    tag_map["[id60]"] = "</p>"

    chunks = HtmlChunker(
        max_tokens=10_000,
        max_placeholders_per_chunk=24,
    ).chunk_html_with_placeholders(text, tag_map)

    assert len(chunks) >= 3
    assert max(len(chunk["local_tag_map"]) for chunk in chunks) <= 24


@pytest.mark.asyncio
async def test_full_fallback_path_never_redistributes_text_across_blocks(monkeypatch):
    source = "[id0]Title[id1]First paragraph.[id2]Second paragraph.[id3]"
    tag_map = {
        "[id0]": "<h1>",
        "[id1]": "</h1><p>",
        "[id2]": "</p><p>",
        "[id3]": "</p>",
    }
    translations = {
        "Title": "Título",
        "First paragraph.": "Primer párrafo.",
        "Second paragraph.": "Segundo párrafo.",
    }

    async def fake_request(text, **_kwargs):
        # Phase 1 simulates a provider response that dropped every placeholder.
        if text == source:
            return "Título Primer párrafo. Segundo párrafo."
        return _translate_marked_payload(text, translations)

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    stats = TranslationMetrics(total_chunks=1)
    unit = {}
    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        global_indices=[0, 1, 2, 3],
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=stats,
        max_retries=1,
        prompt_options={"fidelity_supervisor_mode": "off"},
        runtime_state={},
        unit_record=unit,
    )

    assert result == "[id0]Título[id1]Primer párrafo.[id2]Segundo párrafo.[id3]"
    assert unit["translation_method"] == "structure_safe_alignment"
    assert stats.token_alignment_used == 1
    assert stats.token_alignment_success == 1


@pytest.mark.asyncio
async def test_alignment_fallback_repairs_one_fidelity_rejection(monkeypatch):
    source = "[id0]The published paper title remains exact.[id1]"
    tag_map = {"[id0]": "<p>", "[id1]": "</p>"}
    requests = []
    audits = []

    async def fake_request(text, **kwargs):
        options = kwargs.get("prompt_options") or {}
        requests.append((text, str(options.get("custom_instructions") or "")))
        if text == source:
            return "El título publicado quedó mal."
        if "# FIDELITY RETRY" in requests[-1][1]:
            return (
                "[[[VERBALOOMBLOCK000]]]\n"
                "El título del artículo publicado se conserva exactamente.\n"
                "[[[/VERBALOOMBLOCK000]]]"
            )
        return (
            "[[[VERBALOOMBLOCK000]]]\n"
            "El título publicado se alteró y perdió contenido.\n"
            "[[[/VERBALOOMBLOCK000]]]"
        )

    async def fake_supervise(_source, _candidate, **kwargs):
        audits.append(kwargs["phase"])
        if len(audits) == 1:
            return (
                FidelityDecision(
                    chunk_index=0,
                    phase=kwargs["phase"],
                    section="",
                    accepted=False,
                    issues=[
                        FidelityIssue(
                            "fidelity_judge_reject",
                            "reject",
                            "El título publicado fue alterado",
                        )
                    ],
                ),
                None,
            )
        return (
            FidelityDecision(
                chunk_index=0,
                phase=kwargs["phase"],
                section="",
                accepted=True,
            ),
            None,
        )

    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        fake_request,
    )
    monkeypatch.setattr(
        xhtml_translator,
        "supervise_fidelity",
        fake_supervise,
    )
    stats = TranslationMetrics(total_chunks=1)
    unit = {}

    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        global_indices=[0, 1],
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=stats,
        max_retries=1,
        prompt_options={
            "fidelity_supervisor": True,
            "fidelity_supervisor_retry": True,
        },
        runtime_state={},
        unit_record=unit,
    )

    assert result == (
        "[id0]El título del artículo publicado se conserva exactamente.[id1]"
    )
    assert audits == [
        "translation_alignment_fallback",
        "translation_alignment_fallback_repair",
    ]
    assert any("# FIDELITY RETRY" in instructions for _text, instructions in requests)
    assert stats.token_alignment_success == 1


@pytest.mark.asyncio
async def test_table_context_reaches_alignment_fidelity_supervisor(monkeypatch):
    source = (
        "[id0]Nike[id1]Mary Kay Ash[id2]Mary Kay Cosmetics"
        "[id3]Virgin Records, Airlines, and others[id4]"
    )
    tag_map = {
        "[id0]": "<table><tr><td>",
        "[id1]": "</td></tr><tr><td>",
        "[id2]": "</td></tr><tr><td>",
        "[id3]": "</td></tr><tr><td>",
        "[id4]": "</td></tr></table>",
    }
    audited_options = []
    request_options = []

    async def fake_request(text, **kwargs):
        request_options.append(dict(kwargs.get("prompt_options") or {}))
        if text == source:
            return None
        return _translate_marked_payload(
            text,
            {
                "Nike": "Nike",
                "Mary Kay Ash": "Mary Kay Ash",
                "Mary Kay Cosmetics": "Mary Kay Cosmetics",
                "Virgin Records, Airlines, and others": (
                    "Virgin Records, aerolíneas y otras empresas"
                ),
            },
        )

    async def fake_supervise(_source, _candidate, **kwargs):
        audited_options.append(dict(kwargs.get("prompt_options") or {}))
        return (
            FidelityDecision(
                chunk_index=0,
                phase=kwargs["phase"],
                section="",
                accepted=True,
            ),
            None,
        )

    monkeypatch.setattr(
        xhtml_translator,
        "generate_translation_request",
        fake_request,
    )
    monkeypatch.setattr(
        xhtml_translator,
        "supervise_fidelity",
        fake_supervise,
    )

    result = await xhtml_translator.translate_chunk_with_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        global_indices=[0, 1, 2, 3, 4],
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        stats=TranslationMetrics(total_chunks=1),
        max_retries=1,
        prompt_options={"fidelity_supervisor": True},
        runtime_state={},
        unit_record={},
    )

    assert "aerolíneas y otras empresas" in result
    assert request_options
    assert request_options[0]["_document_block_context"] == "table"
    assert any(
        "Never expand a generic category into an inferred brand"
        in str(options.get("custom_instructions") or "")
        for options in request_options
    )
    assert audited_options
    assert audited_options[0]["_document_block_context"] == "table"


@pytest.mark.asyncio
async def test_structure_recovery_batches_dense_xhtml_with_a_hard_call_budget(monkeypatch):
    block_count = 20
    source = "".join(
        f"[id{index}]Paragraph {index}." for index in range(block_count)
    ) + f"[id{block_count}]"
    tag_map = {
        f"[id{index}]": ("<p>" if index == 0 else "</p><p>")
        for index in range(block_count)
    }
    tag_map[f"[id{block_count}]"] = "</p>"
    translations = {
        f"Paragraph {index}.": f"Párrafo {index}."
        for index in range(block_count)
    }
    calls = []

    async def fake_request(text, **_kwargs):
        calls.append(text)
        return _translate_marked_payload(text, translations)

    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)
    result = await xhtml_translator._translate_structure_safe_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={},
        runtime_state={},
    )

    assert len(calls) == 3
    assert "Párrafo 0." in result
    assert "Párrafo 19." in result
    assert structure_signature(result, tag_map) == structure_signature(source, tag_map)


@pytest.mark.asyncio
async def test_structure_recovery_presplits_oversized_batches(monkeypatch):
    block_count = 8
    source_blocks = {
        index: f"Paragraph {index}. " + ("Detailed source material. " * 32)
        for index in range(block_count)
    }
    source = "".join(
        f"[id{index}]{source_blocks[index]}" for index in range(block_count)
    ) + f"[id{block_count}]"
    tag_map = {
        f"[id{index}]": ("<p>" if index == 0 else "</p><p>")
        for index in range(block_count)
    }
    tag_map[f"[id{block_count}]"] = "</p>"
    translations = {
        value.strip(): (
            f"Párrafo {index}. " + ("Material fuente detallado. " * 20)
        ).strip()
        for index, value in source_blocks.items()
    }
    calls = []

    async def fake_request(text, **_kwargs):
        calls.append(text)
        return _translate_marked_payload(text, translations)

    monkeypatch.setattr(config, "EPUB_STRUCTURE_RECOVERY_BATCH_SIZE", 8)
    monkeypatch.setattr(config, "EPUB_STRUCTURE_RECOVERY_MAX_SOURCE_TOKENS", 180)
    monkeypatch.setattr(config, "EPUB_STRUCTURE_RECOVERY_MAX_CALLS", 12)
    monkeypatch.setattr(xhtml_translator, "generate_translation_request", fake_request)

    result = await xhtml_translator._translate_structure_safe_fallback(
        chunk_text=source,
        local_tag_map=tag_map,
        source_language="English",
        target_language="Spanish",
        model_name="test",
        llm_client=object(),
        log_callback=None,
        context_manager=None,
        prompt_options={},
        runtime_state={},
    )

    assert 1 < len(calls) <= block_count
    assert max(call.count("[[[VERBALOOMBLOCK") for call in calls) <= 2
    assert "Párrafo 0." in result
    assert "Párrafo 7." in result
    assert structure_signature(result, tag_map) == structure_signature(source, tag_map)
