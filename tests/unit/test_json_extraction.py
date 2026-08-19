from src.utils.json_extraction import (
    extract_first_json_value_text,
    extract_tagged_payload,
    loads_first_json_object,
    loads_first_json_value,
)


def test_extracts_tagged_payload_without_regex():
    assert extract_tagged_payload(
        "noise <TAG>{\"ok\": true}</TAG> tail",
        "<TAG>",
        "</TAG>",
    ) == '{"ok": true}'


def test_loads_first_json_object_from_noisy_response():
    parsed = loads_first_json_object(
        "Here is the result:\n```json\n{\"decision\":\"accept\",\"items\":[1,2]}\n```"
    )
    assert parsed == {"decision": "accept", "items": [1, 2]}


def test_loads_first_json_value_accepts_arrays_for_review_payloads():
    assert loads_first_json_value("prefix [{\"term\":\"A\"}] suffix") == [{"term": "A"}]


def test_incomplete_large_json_returns_none_quickly():
    payload = "prefix " + "{" + '"decision":"accept",' + ("x" * 100_000)
    assert extract_first_json_value_text(payload) is None
    assert loads_first_json_object(payload) is None
