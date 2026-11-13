"""Exact, atomic edits; these tests do not claim semantic correctness."""

import json
from copy import deepcopy

import pytest

from src.core.localized_repair import (
    InvalidRepairPatch,
    apply_repair_patch,
    build_repair_contract,
)

SOURCE = "[id0]She did not sell the violin.[id1]\n\n[id2]He waited.[id3]"
TARGET = "[id0]She sold the violin.[id1]\n\n[id2]He waited.[id3]"
TOKENS = ["[id0]", "[id1]", "[id2]", "[id3]"]


def contract(source=SOURCE, candidate=TARGET):
    return build_repair_contract(
        source,
        candidate,
        unit_id="chapter:unit-7",
        findings=[{"kind": "changed_facts", "detail": "The negation was reversed."}],
        protected_tokens=TOKENS,
    )


def reply(value):
    return {
        "schema": value["schema"],
        "unit_id": value["unit_id"],
        "source_sha256": value["source_sha256"],
        "candidate_sha256": value["candidate_sha256"],
        "contract_sha256": value["contract_sha256"],
        "patches": [
            {
                "finding_id": value["findings"][0]["id"],
                "block_id": value["candidate_blocks"][0]["id"],
                "operation": "replace",
                "anchor": "sold",
                "replacement": "did not sell",
                "source_block_id": value["source_blocks"][0]["id"],
                "source_quote": "did not sell",
            }
        ],
    }


def apply(value, data, source=SOURCE, candidate=TARGET):
    return apply_repair_patch(
        value,
        json.dumps(data),
        source_text=source,
        candidate_text=candidate,
        unit_id="chapter:unit-7",
    )


def test_patch_changes_only_anchored_text_and_reports_exact_offsets():
    value = contract()
    result = apply(value, reply(value))
    assert result.text == SOURCE
    assert result.edits[0]["before"] == "sold"
    assert result.edits[0]["after"] == "did not sell"
    assert TARGET[result.edits[0]["start"] : result.edits[0]["end"]] == "sold"
    assert result.candidate_sha256 != value["candidate_sha256"]


@pytest.mark.parametrize(
    "field", ["unit_id", "source_sha256", "candidate_sha256", "contract_sha256"]
)
def test_changed_or_cross_book_response_is_rejected(field):
    value = contract()
    data = reply(value)
    data[field] = "stale"
    with pytest.raises(InvalidRepairPatch):
        apply(value, data)


@pytest.mark.parametrize("side", ["source", "candidate"])
def test_runtime_source_or_candidate_changed_after_request(side):
    value = contract()
    with pytest.raises(InvalidRepairPatch):
        apply(
            value,
            reply(value),
            source=SOURCE + ("!" if side == "source" else ""),
            candidate=TARGET + ("!" if side == "candidate" else ""),
        )


def test_contract_tampering_is_not_a_new_authorization():
    value = contract()
    data = reply(value)
    value["candidate_blocks"][0]["text"] = "tampered"
    with pytest.raises(InvalidRepairPatch):
        apply(value, data)


@pytest.mark.parametrize(
    "change",
    [
        {"finding_id": "invented"},
        {"block_id": "another-unit"},
        {"source_block_id": "another-source"},
        {"source_quote": "not in the source"},
        {"anchor": "not in the candidate"},
        {"anchor": ""},
        {"operation": "rewrite_chapter"},
        {"replacement": "[id99]inserted"},
        {"replacement": "[[[TBLBLOCK999]]]inserted"},
        {"replacement": "first\n\nsecond"},
        {"replacement": "\u0000"},
        {"replacement": "\ud800"},
        {"replacement": "<img src='missing.png'/>"},
        {"confidence": 1.0},
    ],
)
def test_unsupported_unanchored_or_structural_edits_are_rejected(change):
    value = contract()
    data = reply(value)
    data["patches"][0].update(change)
    with pytest.raises(InvalidRepairPatch):
        apply(value, data)


def test_duplicate_anchor_requires_more_context_not_first_match():
    value = contract(candidate="[id0]She sold what he sold.[id1]")
    data = reply(value)
    with pytest.raises(InvalidRepairPatch, match="ambiguous"):
        apply(value, data, candidate="[id0]She sold what he sold.[id1]")


def test_all_edits_validate_before_any_are_applied():
    value = contract()
    data = reply(value)
    data["patches"].append({**data["patches"][0], "anchor": "She sold"})
    before = deepcopy(value)
    with pytest.raises(InvalidRepairPatch, match="overlap"):
        apply(value, data)
    assert value == before


def test_no_change_is_valid_without_inventing_a_patch():
    value = contract()
    data = reply(value)
    data["patches"] = []
    result = apply(value, data)
    assert result.text == TARGET and result.edits == []


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("insert_before", "She did not sell sold the violin."),
        ("insert_after", "She solddid not sell  the violin."),
    ],
)
def test_insertions_preserve_the_anchor_exactly(operation, expected):
    value = contract()
    data = reply(value)
    data["patches"][0].update(operation=operation, replacement="did not sell ")
    result = apply(value, data)
    assert result.text == TARGET.replace("She sold the violin.", expected)


def test_unicode_offsets_and_combining_forms_are_not_normalized():
    source, target = "[id0]Cafe\u0301: no.[id1]", "[id0]Cafe\u0301: yes.[id1]"
    value = contract(source, target)
    data = reply(value)
    data["patches"][0].update(anchor="yes", replacement="no", source_quote="no")
    assert apply(value, data, source, target).text == source


def test_blocks_protect_markers_paragraphs_and_inline_formatting():
    text = "[[[TBLBLOCK006]]]\n[id0]One[id1]two[id2]\n\nThree\r\n\r\nFour"
    value = contract(text, text)
    assert [b["text"] for b in value["candidate_blocks"]] == [
        "One",
        "two",
        "Three",
        "Four",
    ]
    assert all(b["id"].startswith("target-") for b in value["candidate_blocks"])
    assert value == contract(text, text)


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n{}\n```",
        "<TRANSLATION>complete rewrite</TRANSLATION>",
        '{"schema": "a", "schema": "b"}',
        '{"bad": NaN}',
        "[]",
    ],
)
def test_malformed_json_is_not_a_translation_fallback(raw):
    with pytest.raises(InvalidRepairPatch):
        apply_repair_patch(
            contract(),
            raw,
            source_text=SOURCE,
            candidate_text=TARGET,
            unit_id="chapter:unit-7",
        )


def test_patch_count_and_reply_size_have_finite_limits():
    value = contract()
    data = reply(value)
    data["patches"] *= 33
    with pytest.raises(InvalidRepairPatch):
        apply(value, data)
    with pytest.raises(InvalidRepairPatch):
        apply_repair_patch(
            value,
            " " * 1_000_001,
            source_text=SOURCE,
            candidate_text=TARGET,
            unit_id="chapter:unit-7",
        )
