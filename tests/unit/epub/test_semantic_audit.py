from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core.epub.semantic_audit import adjudicate_semantic_audit, audit_epub_units_semantically


def _unit(unit_id, order, text):
    return SimpleNamespace(
        unit_id=unit_id,
        file_href="Text/chapter.xhtml",
        ordinal=order,
        dom_path=f"/html/body/p[{order + 1}]",
        spine_index=0,
        source_order=order,
        source_hash=f"source-{order}",
        text=text,
    )


class Provider:
    def __init__(self):
        self.calls = 0

    async def make_request(self, user, _model, system_prompt=""):
        self.calls += 1
        requested = json.loads(user)["requested_units"]
        units = [{
            "unit_id": item["unit_id"],
            "verdict": "pass",
            "confidence": 0.98,
            "reason": "faithful",
            "missing_from_source": [],
            "added_not_in_source": [],
            "changed_facts": [],
            "censored_or_softened": [],
            "terminology_errors": [],
            "structure_issues": [],
            "source_language_residual": False,
        } for item in requested]
        return SimpleNamespace(content="<UNIT_RESULTS_JSON>" + json.dumps({"units": units}) + "</UNIT_RESULTS_JSON>", was_truncated=False)


class GenerateOnlyProvider(Provider):
    make_request = None

    async def generate(self, user, system_prompt=""):
        return await Provider.make_request(self, user, "deepseek-v4-pro", system_prompt=system_prompt)


@pytest.mark.asyncio
async def test_semantic_audit_covers_every_aligned_unit_and_resumes_from_cache(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "Quelle eins"), _unit("u2", 1, "Quelle zwei")])
    output = SimpleNamespace(units=[_unit("u1", 0, "Texto uno"), _unit("u2", 1, "Texto dos")])
    provider = Provider()
    cache = tmp_path / "audit.json"

    first = await audit_epub_units_semantically(
        source, output, provider=provider, model="deepseek-v4-pro",
        source_language="German", target_language="Spanish", cache_path=cache,
    )
    first_calls = provider.calls
    second = await audit_epub_units_semantically(
        source, output, provider=provider, model="deepseek-v4-pro",
        source_language="German", target_language="Spanish", cache_path=cache,
    )

    assert first.complete
    assert first.passed == 2
    assert first_calls == 1
    assert provider.calls == first_calls
    assert second.cache_hits == 2


@pytest.mark.asyncio
async def test_semantic_audit_supports_direct_generate_provider(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "Quelle")])
    output = SimpleNamespace(units=[_unit("u1", 0, "Texto")])
    report = await audit_epub_units_semantically(
        source, output, provider=GenerateOnlyProvider(), model="deepseek-v4-pro",
        source_language="German", target_language="Spanish", cache_path=tmp_path / "generate.json",
    )
    assert report.complete


@pytest.mark.asyncio
async def test_semantic_audit_refuses_misaligned_units(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "Quelle")])
    broken = _unit("u1", 1, "Texto")
    output = SimpleNamespace(units=[broken])

    with pytest.raises(ValueError, match="alignment mismatch"):
        await audit_epub_units_semantically(
            source, output, provider=Provider(), model="deepseek-v4-pro",
            source_language="German", target_language="Spanish", cache_path=tmp_path / "audit.json",
        )


@pytest.mark.asyncio
async def test_adjudicator_can_reverse_a_self_contradictory_false_positive(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "Am frühen Nachmittag")])
    output = SimpleNamespace(units=[_unit("u1", 0, "A primera hora de la tarde")])
    report = SimpleNamespace(
        results=[{"unit_id": "u1", "accepted": False, "verdict": "fail", "reason": "The translation is correct; verdict should pass."}],
        requests=0,
        validation_failures=0,
        recursive_splits=0,
        cache_hits=0,
    )

    class Adjudicator(GenerateOnlyProvider):
        async def generate(self, user, system_prompt=""):
            requested = json.loads(user)["requested_units"]
            units = [{
                "unit_id": item["unit_id"], "verdict": "false_positive", "confidence": 0.99,
                "reason": "The candidate is an accurate Spanish rendering.",
                "objective_issues": [], "recommended_action": "accept",
            } for item in requested]
            return SimpleNamespace(content="<UNIT_RESULTS_JSON>" + json.dumps({"units": units}) + "</UNIT_RESULTS_JSON>", was_truncated=False)

    updated = await adjudicate_semantic_audit(
        source, output, report, provider=Adjudicator(), model="deepseek-v4-pro",
        source_language="German", target_language="Spanish",
        cache_path=tmp_path / "adjudication.json",
    )
    assert updated.results[0]["accepted"] is True
    assert updated.results[0]["acceptance_basis"] == "adjudicated_false_positive"


@pytest.mark.asyncio
async def test_adjudicator_normalizes_pass_alias_without_objective_issues(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "Quelle")])
    output = SimpleNamespace(units=[_unit("u1", 0, "Fuente")])
    report = SimpleNamespace(
        results=[{"unit_id": "u1", "accepted": False, "verdict": "fail", "reason": "self-contradictory"}],
        requests=0, validation_failures=0, recursive_splits=0, cache_hits=0,
    )

    class PassAlias(GenerateOnlyProvider):
        async def generate(self, user, system_prompt=""):
            requested = json.loads(user)["requested_units"]
            units = [{
                "unit_id": item["unit_id"], "verdict": "pass", "confidence": 0.95,
                "reason": "No objective error.", "objective_issues": [],
                "recommended_action": "false_positive",
            } for item in requested]
            return SimpleNamespace(content="<UNIT_RESULTS_JSON>" + json.dumps({"units": units}) + "</UNIT_RESULTS_JSON>", was_truncated=False)

    updated = await adjudicate_semantic_audit(
        source, output, report, provider=PassAlias(), model="deepseek-v4-pro",
        source_language="German", target_language="Spanish",
        cache_path=tmp_path / "pass-alias.json",
    )
    assert updated.results[0]["accepted"] is True


@pytest.mark.asyncio
async def test_adjudicator_does_not_accept_confirmed_issue_with_accept_action(tmp_path):
    source = SimpleNamespace(units=[_unit("u1", 0, "vier Falben")])
    output = SimpleNamespace(units=[_unit("u1", 0, "cuatro caballos overos")])
    report = SimpleNamespace(
        results=[{"unit_id": "u1", "accepted": False, "verdict": "warn", "reason": "changed fact"}],
        requests=0, validation_failures=0, recursive_splits=0, cache_hits=0,
    )

    class Confirmed(GenerateOnlyProvider):
        async def generate(self, user, system_prompt=""):
            requested = json.loads(user)["requested_units"]
            units = [{
                "unit_id": item["unit_id"], "verdict": "confirmed", "confidence": 0.99,
                "reason": "Horse colour changed.", "objective_issues": ["changed_facts"],
                "recommended_action": "accept",
            } for item in requested]
            return SimpleNamespace(content="<UNIT_RESULTS_JSON>" + json.dumps({"units": units}) + "</UNIT_RESULTS_JSON>", was_truncated=False)

    updated = await adjudicate_semantic_audit(
        source, output, report, provider=Confirmed(), model="deepseek-v4-pro",
        source_language="German", target_language="Spanish",
        cache_path=tmp_path / "confirmed.json",
    )
    assert updated.results[0]["accepted"] is False
