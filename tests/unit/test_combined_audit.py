from src.core.combined_audit import (
    COMBINED_AUDIT_TAG_IN,
    COMBINED_AUDIT_TAG_OUT,
    build_combined_quality_audit_prompt,
    editorial_assessment_from_combined,
    fidelity_assessment_from_combined,
    parse_combined_quality_audit_response,
    profile_payload_from_combined,
)


def test_combined_audit_prompt_keeps_three_axis_contract():
    prompt = build_combined_quality_audit_prompt(
        source_text="Source text.",
        draft_text="Draft text.",
        candidate_text="Candidate text.",
        target_language="Spanish",
        profile_id="sample_profile",
        profile_policy="Use clear literary Spanish.",
        glossary_block="- Source => Fuente",
    )

    assert "editorial" in prompt.system
    assert "fidelity" in prompt.system
    assert "profile" in prompt.system
    assert "Source text." in prompt.user
    assert "Candidate text." in prompt.user
    assert "sample_profile" in prompt.user
    assert COMBINED_AUDIT_TAG_IN in prompt.system


def test_parse_combined_audit_normalizes_axis_payloads():
    raw = f"""
{COMBINED_AUDIT_TAG_IN}
{{
  "editorial": {{"verdict": "pass", "confidence": 0.91, "reason": "safe"}},
  "fidelity": {{"decision": "repair", "confidence": 0.8, "reason": "one omission"}},
  "profile": {{"decision": "warn", "scores": {{"content_fidelity": 9.2}}, "issues": []}}
}}
{COMBINED_AUDIT_TAG_OUT}
"""

    parsed = parse_combined_quality_audit_response(raw)

    assert parsed is not None
    assert editorial_assessment_from_combined(parsed)["decision"] == "accept"
    assert fidelity_assessment_from_combined(parsed)["verdict"] == "repair_needed"
    assert profile_payload_from_combined(parsed)["overall_decision"] == "warn"
    assert parsed["worst"] == "warn"
