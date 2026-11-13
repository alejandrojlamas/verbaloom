"""A required remote judgment cannot silently become a local-only approval."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.core import translator
from src.core.book_profiles.audit import SCORE_FIELDS, parse_profile_audit_response
from src.core.combined_audit import parse_combined_quality_audit_response
from src.core.editorial_quality import parse_source_aware_guard_response
from src.core.fidelity_supervisor import (
    parse_fidelity_audit_response,
    supervise_fidelity,
)
from src.core.llm.base import LLMResponse
from src.core.llm.exceptions import ContentRiskError, ProviderActionRequiredError


def fidelity_payload():
    return {
        "verdict": "pass",
        "confidence": 0.98,
        "reason": "No material defects.",
        **{
            key: []
            for key in (
                "issues",
                "missing_from_source",
                "added_not_in_source",
                "changed_facts",
                "censored_or_softened",
                "structure_issues",
                "evidence_source",
                "evidence_candidate",
            )
        },
    }


def profile_payload():
    return {
        "scores": dict.fromkeys(SCORE_FIELDS, 9.5),
        "overall_decision": "pass",
        "issues": [],
        "glossary_suggestions": [],
        "summary": "No changes needed.",
    }


def editorial_payload():
    return {
        "decision": "accept",
        "confidence": 0.95,
        "reason": "Faithful refinement.",
        "issues": [],
        "missing_from_source": [],
        "added_not_in_source": [],
        "weird_symbols": [],
        "structure_score": 0.95,
    }


@pytest.mark.parametrize("field", list(editorial_payload()))
def test_editorial_audit_requires_every_requested_field(field):
    payload = editorial_payload()
    del payload[field]
    assert parse_source_aware_guard_response(json.dumps(payload)) is None


@pytest.mark.parametrize("axis", ["editorial", "fidelity", "profile"])
def test_combined_audit_does_not_fabricate_missing_axis_fields(axis):
    payload = {
        "editorial": editorial_payload(),
        "fidelity": fidelity_payload(),
        "profile": profile_payload(),
    }
    payload[axis] = {"decision": "pass", "verdict": "pass", "overall_decision": "pass"}
    assert parse_combined_quality_audit_response(json.dumps(payload)) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("combined", [True, False])
async def test_refinement_guard_does_not_accept_candidate_when_remote_audit_is_invalid(
    monkeypatch, combined
):
    request = AsyncMock(return_value=LLMResponse(content="not JSON"))
    monkeypatch.setattr(translator, "_generate_alert_repair", request)
    with pytest.raises(ProviderActionRequiredError) as caught:
        await translator._assess_refinement_with_editorial_guard(
            "La puerta estaba abierta.",
            "La puerta seguia abierta.",
            chunk_index=1,
            section="fixture",
            source_text="The door was open.",
            source_language="English",
            target_language="Spanish",
            model="deepseek-flash",
            client=object(),
            prompt_options={
                "source_aware_editorial_guard_mode": "always",
                "combined_quality_audit": combined,
            },
        )
    assert caught.value.reason == "audit_invalid_response"
    assert request.await_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"confidence": float("nan")},
        {"confidence": True},
        {"confidence": "0.9"},
        {"confidence": 1.1},
        {"reason": {}},
        {"issues": "none"},
        {"changed_facts": [42]},
        {"evidence_source": None},
    ],
)
def test_fidelity_rejects_malformed_fields(change):
    assert (
        parse_fidelity_audit_response(json.dumps({**fidelity_payload(), **change}))
        is None
    )


@pytest.mark.parametrize("field", list(fidelity_payload()))
def test_fidelity_requires_every_requested_field(field):
    payload = fidelity_payload()
    del payload[field]
    assert parse_fidelity_audit_response(json.dumps(payload)) is None


@pytest.mark.parametrize(
    "change",
    [
        {"scores": {}},
        {"scores": dict.fromkeys(SCORE_FIELDS, float("nan"))},
        {"scores": dict.fromkeys(SCORE_FIELDS, True)},
        {"overall_decision": "maybe"},
        {"issues": "none"},
        {"issues": [{"type": "omission", "severity": "high"}]},
        {"glossary_suggestions": {}},
        {"summary": []},
    ],
)
def test_profile_rejects_malformed_fields(change):
    assert (
        parse_profile_audit_response(json.dumps({**profile_payload(), **change}))
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        LLMResponse(content="not JSON"),
        LLMResponse(content=json.dumps(fidelity_payload()), was_truncated=True),
        LLMResponse(content=json.dumps(fidelity_payload()), status="refused"),
    ],
)
async def test_required_fidelity_audit_stops_after_bounded_format_attempts(response):
    client = SimpleNamespace(make_request=AsyncMock(return_value=response))
    with pytest.raises(ProviderActionRequiredError) as caught:
        await supervise_fidelity(
            "The door was open.",
            "La puerta estaba abierta.",
            chunk_index=1,
            phase="final_epub_unit_audit",
            primary_provider="deepseek",
            client=client,
            prompt_options={"fidelity_supervisor_mode": "strict_full"},
        )
    assert caught.value.reason in {"audit_invalid_response", "provider_refusal"}
    assert caught.value.retryable is False
    assert client.make_request.await_count <= 3


@pytest.mark.asyncio
async def test_legacy_refusal_is_an_actionable_stop_without_local_approval():
    refusal = ContentRiskError("Provider refused this request", provider="deepseek")
    client = SimpleNamespace(make_request=AsyncMock(side_effect=refusal))
    with pytest.raises(ProviderActionRequiredError) as caught:
        await supervise_fidelity(
            "The door was open.",
            "La puerta estaba abierta.",
            chunk_index=1,
            phase="translation",
            primary_provider="deepseek",
            client=client,
            prompt_options={"fidelity_supervisor_mode": "always"},
        )
    assert caught.value.reason == "provider_refusal"
    assert client.make_request.await_count == 1


@pytest.mark.asyncio
async def test_required_remote_audit_without_client_cannot_approve_locally():
    with pytest.raises(ProviderActionRequiredError) as caught:
        await supervise_fidelity(
            "The door was open.",
            "La puerta estaba abierta.",
            chunk_index=1,
            phase="translation",
            primary_provider="deepseek",
            client=None,
            prompt_options={"fidelity_supervisor_mode": "always"},
        )
    assert caught.value.reason == "audit_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        LLMResponse(content="not JSON"),
        LLMResponse(content=json.dumps({"scores": {}, "overall_decision": "pass"})),
        LLMResponse(content=json.dumps(profile_payload()), was_truncated=True),
    ],
)
async def test_required_profile_audit_never_defaults_to_perfect_scores(
    monkeypatch, response
):
    profile = SimpleNamespace(
        profile_id="fixture", raw_config={}, min_dimension_score=8.5
    )
    monkeypatch.setattr(translator, "load_book_profile", lambda _: profile)
    monkeypatch.setattr(translator, "build_profile_glossary_block", lambda *a, **k: "")
    monkeypatch.setattr(
        translator, "build_profile_audit_prompt", lambda *a, **k: ("audit", "source")
    )
    request = AsyncMock(return_value=response)
    monkeypatch.setattr(translator, "_generate_alert_repair", request)
    with pytest.raises(ProviderActionRequiredError) as caught:
        await translator._run_profile_audit(
            "The door was open.",
            "La puerta estaba abierta.",
            profile_id="fixture",
            prompt_options={
                "editorial_mode": "book_profile",
                "profile_id": "fixture",
                "profile_audit_enabled": True,
                "profile_local_precheck_enabled": False,
            },
            model="deepseek-flash",
            client=object(),
        )
    assert caught.value.reason == "audit_invalid_response"
    assert request.await_count == 1


def test_complete_no_change_judgments_remain_valid():
    assert parse_fidelity_audit_response(json.dumps(fidelity_payload()))
    assert parse_profile_audit_response(json.dumps(profile_payload()))


def test_active_profile_dimensions_cannot_be_omitted_from_standalone_or_combined_audit():
    fields = (*SCORE_FIELDS, "authorial_voice")
    profile = profile_payload()
    assert (
        parse_profile_audit_response(json.dumps(profile), score_fields=fields) is None
    )
    combined = {
        "editorial": editorial_payload(),
        "fidelity": fidelity_payload(),
        "profile": profile,
    }
    assert (
        parse_combined_quality_audit_response(
            json.dumps(combined), profile_score_fields=fields
        )
        is None
    )
    profile["scores"]["authorial_voice"] = 9.5
    assert parse_profile_audit_response(json.dumps(profile), score_fields=fields)
    assert parse_combined_quality_audit_response(
        json.dumps(combined), profile_score_fields=fields
    )
