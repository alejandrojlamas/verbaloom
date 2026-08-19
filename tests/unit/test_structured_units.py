from __future__ import annotations

import json

import pytest

from src.core.structured_units import (
    RESULT_TAG_IN,
    RESULT_TAG_OUT,
    StructuredUnit,
    StructuredBatchStats,
    StructuredUnitResponseError,
    bisect_structured_units,
    build_structured_unit_prompt,
    parse_structured_unit_response,
    request_structured_units,
)


def _wrapped(units):
    return RESULT_TAG_IN + json.dumps({"units": units}) + RESULT_TAG_OUT


def test_exact_id_response_accepts_out_of_order_results_but_returns_source_order():
    result = parse_structured_unit_response(
        _wrapped([
            {"unit_id": "u2", "text": "Dos"},
            {"unit_id": "u1", "text": "Uno"},
        ]),
        requested_ids=["u1", "u2"],
        required_fields=["text"],
    )
    assert list(result) == ["u1", "u2"]
    assert result["u1"]["text"] == "Uno"


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("", "empty_response"),
        (RESULT_TAG_IN + '{"units":[]}', "truncated_response"),
        ('{"units":[', "truncated_json"),
        ('not json', "invalid_json"),
        (_wrapped([{"unit_id": "u1", "text": "Uno"}]), "missing_ids"),
        (_wrapped([{"unit_id": "u1", "text": "Uno"}, {"unit_id": "u1", "text": "Otro"}]), "duplicate_id"),
        (_wrapped([{"unit_id": "u1", "text": "Uno"}, {"unit_id": "u3", "text": "Tres"}]), "unknown_id"),
        (_wrapped([{"unit_id": "u1", "text": ""}, {"unit_id": "u2", "text": "Dos"}]), "empty_field"),
    ],
)
def test_rejects_incomplete_or_ambiguous_batch_responses(content, code):
    with pytest.raises(StructuredUnitResponseError) as exc:
        parse_structured_unit_response(
            content,
            requested_ids=["u1", "u2"],
            required_fields=["text"],
        )
    assert exc.value.code == code


def test_rejects_provider_truncation_and_placeholder_loss_or_duplication():
    with pytest.raises(StructuredUnitResponseError) as truncated:
        parse_structured_unit_response(
            _wrapped([{"unit_id": "u1", "text": "[id0]Texto[id1]"}]),
            requested_ids=["u1"],
            required_fields=["text"],
            response_truncated=True,
        )
    assert truncated.value.code == "truncated_response"

    with pytest.raises(StructuredUnitResponseError) as placeholders:
        parse_structured_unit_response(
            _wrapped([{"unit_id": "u1", "text": "[id0]Texto[id0]"}]),
            requested_ids=["u1"],
            required_fields=["text"],
            placeholder_manifest={"u1": ["[id0]", "[id1]"]},
        )
    assert placeholders.value.code == "placeholder_mismatch"


def test_prompt_keeps_stable_contract_separate_from_variable_units_and_bisects():
    units = [StructuredUnit("u1", {"source": "A"}), StructuredUnit("u2", {"source": "B"})]
    system, user = build_structured_unit_prompt(
        units,
        system_instructions="Audit faithfully.",
        response_fields=["verdict", "reason"],
    )
    left, right = bisect_structured_units(units)

    assert "u1" not in system
    assert "requested_units" in user
    assert [unit.unit_id for unit in left] == ["u1"]
    assert [unit.unit_id for unit in right] == ["u2"]


@pytest.mark.asyncio
async def test_invalid_batch_is_retried_then_recursively_split_without_losing_order():
    calls = []

    async def request(_system, user):
        ids = [item["unit_id"] for item in json.loads(user)["requested_units"]]
        calls.append(ids)
        if len(ids) > 1:
            return _wrapped([{"unit_id": ids[0], "text": "partial"}])
        return _wrapped([{"unit_id": ids[0], "text": f"result-{ids[0]}"}])

    async def no_sleep(_delay):
        return None

    units = [StructuredUnit(f"u{i}", {"source": str(i)}) for i in range(1, 4)]
    stats = StructuredBatchStats()
    result = await request_structured_units(
        units,
        request=request,
        system_instructions="Translate.",
        response_fields=["text"],
        required_fields=["text"],
        max_attempts=2,
        stats=stats,
        sleep=no_sleep,
    )

    assert list(result) == ["u1", "u2", "u3"]
    assert [result[key]["text"] for key in result] == ["result-u1", "result-u2", "result-u3"]
    assert stats.recursive_splits == 2
    assert stats.validation_failures >= 2
    assert calls[0] == ["u1", "u2", "u3"]
