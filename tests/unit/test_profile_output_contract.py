"""Exercise actual provider requests without a network client or a model."""

import json

import pytest
import yaml

from src.core import translator
from src.core.book_profiles.audit import run_profile_precheck, score_fields_for_profile
from src.core.book_profiles.loader import create_profile, load_book_profile
from src.core.book_profiles.profile_goals import declared_profile_goal
from src.core.llm.base import LLMResponse
from tests.unit.test_source_audited_strategy import ASSESSMENT, SOURCE, TARGET


class RecordingClient:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []

    async def generate(self, prompt, *, system_prompt, **_):
        self.messages.append((system_prompt, prompt))
        return LLMResponse(content=self.reply)

    async def make_request(self, prompt, model, *, system_prompt, **kwargs):
        return await self.generate(prompt, system_prompt=system_prompt, **kwargs)

    def extract_translation(self, response):
        return response.removeprefix("<TRANSLATION>").removesuffix("</TRANSLATION>")


def configure_profile(tmp_path, monkeypatch, data):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    root = create_profile("fixture", profiles_root=tmp_path)
    path = root / "profile.yml"
    config = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**config, **data}))
    (root / "editorial_policy.md").write_text("Retain the deliberate changes of narrator.")
    return {"editorial_mode": "book_profile", "profile_id": "fixture"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("declaration", "mode", "goal_key"), [
    ({"profile_goal": "faithful_translation"}, "", "faithful_translation"),
    ({"business_rules": {"goal": "explanatory_rewrite"}}, "", "explanatory_rewrite"),
    ({"audiobook": {"enabled": True}}, "", "audiobook"),
    ({"transform_mode": "modernize"}, "modernize", "modernization"),
    ({"profile_goal": "literary_polish"}, "literary_polish", "literary_polish"),
    ({"profile_goal": "explanatory_rewrite"}, "literary_polish", None),
    ({"source_name": "audiobook modern thesis.txt"}, "", None),
    ({"profile_goal": "house_voice"}, "literary_polish", None),
])
async def test_writing_and_refinement_receive_the_selected_policy_once(
    tmp_path, monkeypatch, declaration, mode, goal_key,
):
    options = configure_profile(tmp_path, monkeypatch, declaration)
    options.update(text_transform_mode=mode, refinement_instructions="Retain the opening pause.")
    source, language = (TARGET, "Spanish") if mode else (SOURCE, "English")
    client = RecordingClient(f"<TRANSLATION>{TARGET}</TRANSLATION>")

    assert await translator.generate_translation_request(
        source, "", "", "", source_language=language, target_language="Spanish",
        model="fixture", llm_client=client, prompt_options=options,
    ) == TARGET
    output, _ = await translator._make_refinement_request(
        TARGET, "", "", "", "Spanish", "fixture", client, None, False,
        prompt_options=options, source_text=source,
    )
    assert output == TARGET
    assert len(client.messages) == 2
    for system, user in client.messages:
        request = system + "\n" + user
        assert request.count("# BOOK EDITORIAL PROFILE") == 1
        assert request.count("Retain the deliberate changes of narrator.") == 1
        if goal_key:
            rules = declared_profile_goal({"profile_goal": goal_key}).output_rules
            for rule in rules:
                assert request.count(rule) == 1
        else:
            assert "Editorial goal:" not in request
        if mode != "modernize":
            assert "A high-modernization profile may require" not in request
        if mode:
            assert "faithful translation from" not in request
    assert "Retain the opening pause." in "\n".join(client.messages[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("generated", [True, False])
async def test_judges_and_local_repair_receive_the_same_editorial_criteria(tmp_path, monkeypatch, generated):
    options = configure_profile(tmp_path, monkeypatch, {
        "business_rules": {"goal": "explanatory_rewrite"}, "generated_profile": generated,
    })
    profile = load_book_profile("fixture")
    scores = dict.fromkeys(score_fields_for_profile(profile), 10)
    approved = {"scores": scores, "overall_decision": "pass", "issues": [], "summary": "OK", "glossary_suggestions": []}
    client = RecordingClient(f'<PROFILE_AUDIT_JSON>{json.dumps(approved)}</PROFILE_AUDIT_JSON>')
    result, _ = await translator._run_profile_audit(
        SOURCE, TARGET, profile_id="fixture", prompt_options=options, model="fixture", client=client,
    )
    assert result.overall_decision == "pass"
    editorial = {"decision": "accept", "confidence": 1, "reason": "OK", "issues": [],
                 "missing_from_source": [], "added_not_in_source": [], "weird_symbols": [], "structure_score": 1}
    fidelity = json.loads(ASSESSMENT.split(">", 1)[1].split("</", 1)[0])
    combined = {"editorial": editorial, "profile": approved, "fidelity": fidelity}
    client.reply = f'<COMBINED_QUALITY_AUDIT_JSON>{json.dumps(combined)}</COMBINED_QUALITY_AUDIT_JSON>'
    assert (await translator._run_combined_quality_audit(
        source_text=SOURCE, draft_text=TARGET, candidate_text=TARGET, chunk_index=1,
        section="Introduction", source_language="English", target_language="Spanish",
        phase="refinement", editorial_decision=None, prompt_options=options, model="fixture", client=client,
    ))[0]["profile"]["overall_decision"] == "pass"
    client.reply = f"<TRANSLATION>{TARGET}</TRANSLATION>"
    assert (await translator._make_profile_repair_request(
        SOURCE, TARGET, result, profile_id="fixture", prompt_options=options,
        target_language="Spanish", model="fixture", client=client,
    ))[0] == TARGET
    assert len(client.messages) == 3
    for idx, (system, user) in enumerate(client.messages):
        # Combined audits historically consume the whole saved policy. It can contain
        # a reader's amendments even if the original file was generated automatically.
        assert ("Retain the deliberate changes of narrator." in system + user) == (idx == 1 or not generated)
        for rule in declared_profile_goal(profile.raw_config).output_rules:
            assert (system + "\n" + user).count(rule) == 1


def test_modernization_residue_requires_an_active_modernization_and_its_strength(tmp_path, monkeypatch):
    options = configure_profile(tmp_path, monkeypatch, {
        "business_rules": {"goal": "modernization"}, "target_locale": "es-MX", "modernization_strength": "high",
    })
    passage = "Servíase e traíanle frutas. ¿Queréis ir a vuestros palacios? " * 3
    for mode, strength, should_flag in [("modernize", "high", True), ("literary_polish", "high", False),
                                         ("modernize", "light", False)]:
        options.update(text_transform_mode=mode, modernization_strength=strength)
        issues = run_profile_precheck(passage, passage, profile_id="fixture", prompt_options=options).issues
        assert any(issue.issue_type == "modernization_residue" for issue in issues) is should_flag


@pytest.mark.asyncio
async def test_epub_review_sends_glossary_matches_from_source_even_after_translation(tmp_path, monkeypatch):
    from src.core.epub.xhtml_translator import _refine_epub_chunks

    options = configure_profile(tmp_path, monkeypatch, {"profile_goal": "literary_polish"})
    path = tmp_path / "fixture" / "glossary" / "terms.yml"
    path.write_text(yaml.safe_dump({"entries": [{
        "source": "quiet sea", "target": "mar tranquilo", "type": "term",
        "status": "approved", "translation_policy": "translate",
    }]}))
    options["editorial_quality_guard"] = False
    original, candidate = "[id0]" + SOURCE, "[id0]" + TARGET
    refined = candidate.replace("contemplaba", "observaba")
    client = RecordingClient(f"<TRANSLATION>{refined}</TRANSLATION>")
    assert await _refine_epub_chunks(
        [candidate], [{"text": original, "global_indices": [0], "local_tag_map": {"[id0]": "<p>"}}],
        "Spanish", "fixture", client, None, ("[id", "]"), None, options, source_language="English",
    ) == [refined]
    assert client.messages
    system, user = client.messages[0]
    request = "\n".join((system, user))
    assert request.count("Editorial goal: literary_polish") == 1
    assert "quiet sea" in request
    assert "mar tranquilo" in request
    assert "[id0]" in user


def test_legacy_audio_preset_retains_shared_criteria_during_explicit_polishing(tmp_path, monkeypatch):
    options = configure_profile(tmp_path, monkeypatch, {"audit_dimensions": "translation_audiobook_full"})
    profile = load_book_profile("fixture")
    assert "oral_flow" in score_fields_for_profile(profile, options)
    options["text_transform_mode"] = "literary_polish"
    fields = score_fields_for_profile(profile, options)
    assert {"content_fidelity", "name_consistency", "authorial_voice"} <= set(fields)
    assert {"oral_flow", "note_intrusion_control", "terminology_translation"}.isdisjoint(fields)
    # An authored dimension list expresses the editor's own criteria.
    path = tmp_path / "fixture" / "profile.yml"
    config = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**config, "audit_score_fields": ["local_distinction", "oral_flow"]}))
    assert {"local_distinction", "oral_flow"} <= set(score_fields_for_profile(load_book_profile("fixture"), options))
