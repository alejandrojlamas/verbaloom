from __future__ import annotations

import yaml

from src.core.book_profiles.loader import create_profile
from src.core.candidate_result import (
    CandidateIssue,
    CandidateResult,
    TokenCost,
    record_candidate_result,
)
from src.core.editorial_quality import QualityDecision, QualityIssue
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.llm.base import LLMResponse


def test_candidate_result_builds_language_script_scores_and_cost():
    response = LLMResponse(
        content="<TRANSLATION>Hola mundo.</TRANSLATION>",
        prompt_tokens=100,
        completion_tokens=20,
        prompt_cache_hit_tokens=70,
        prompt_cache_miss_tokens=30,
    )
    result = CandidateResult.build(
        "Hola mundo. Este es el resultado.",
        source_text="Hello world. This is the result.",
        phase="translation",
        chunk_index=3,
        source_language="English",
        target_language="Spanish",
        response=response,
    )

    assert result.decision == "accepted"
    assert result.detected_language == "Spanish"
    assert result.target_script == "latin"
    assert result.target_script_ratio > 0.9
    assert result.scores["length_ratio"] > 0
    assert result.token_cost == TokenCost(
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
        prompt_cache_hit_tokens=70,
        prompt_cache_miss_tokens=30,
    )


def test_candidate_result_repair_plan_for_target_language_rejection():
    result = CandidateResult.build(
        "μήνιν ἄειδε θεὰ",
        source_text="μήνιν ἄειδε θεὰ",
        phase="translation",
        chunk_index=1,
        source_language="Greek",
        target_language="Spanish",
        issues=[
            CandidateIssue(
                "target_script_mismatch",
                "reject",
                "Wrong target script.",
            )
        ],
    )

    assert result.decision == "retry"
    assert result.repair_plan.action == "retry"
    assert result.repair_plan.prompt_variant == "target_language_retry"


def test_candidate_result_normalizes_issue_severity():
    result = CandidateResult.build(
        "Texto devuelto sin traducir.",
        phase="translation",
        target_language="Spanish",
        issues=[{"code": "untranslated_source", "severity": "Reject"}],
    )

    assert result.issues[0].severity == "reject"
    assert result.decision == "retry"


def test_candidate_result_reject_issue_overrides_conflicting_accepted_decision():
    result = CandidateResult.build(
        "The source came back unchanged.",
        phase="translation",
        source_text="The source came back unchanged.",
        source_language="English",
        target_language="Spanish",
        decision="accepted",
        issues=[{"code": "untranslated_source", "severity": "reject"}],
    )

    assert result.decision == "retry"
    assert result.repair_plan.action == "retry"


def test_candidate_result_accepts_guard_scores():
    result = CandidateResult.build(
        "Texto limpio.",
        phase="translation",
        target_language="Spanish",
        extra_scores={"style_drift": 0.42, "bad_value": "ignored"},
    )

    assert result.scores["style_drift"] == 0.42
    assert "bad_value" not in result.scores


def test_candidate_result_adapts_quality_and_fidelity_decisions():
    quality = QualityDecision(
        chunk_index=4,
        section="Capitulo",
        accepted=False,
        issues=[
            QualityIssue(
                "artifact_glyphs_added",
                "reject",
                "Added artifact glyphs.",
            )
        ],
    )
    quality_result = CandidateResult.from_quality_decision(
        quality,
        text="Texto □ roto.",
        source_text="Texto limpio.",
        target_language="Spanish",
    )

    assert quality_result.phase == "editorial_guard"
    assert quality_result.decision == "repair"
    assert quality_result.issues[0].code == "artifact_glyphs_added"

    fidelity = FidelityDecision(
        chunk_index=5,
        phase="refinement",
        section="Capitulo",
        accepted=False,
        issues=[
            FidelityIssue(
                "untranslated_source",
                "reject",
                "Candidate is untranslated.",
            )
        ],
    )
    fidelity_result = CandidateResult.from_fidelity_decision(
        fidelity,
        text="The original text.",
        source_text="The original text.",
        source_language="English",
        target_language="Spanish",
    )

    assert fidelity_result.phase == "refinement"
    assert fidelity_result.decision == "retry"
    assert fidelity_result.repair_plan.action == "retry"


def test_record_candidate_result_is_bounded_and_metadata_only():
    options = {}
    for index in range(3):
        record_candidate_result(
            options,
            CandidateResult.build(
                f"Texto {index}",
                phase="translation",
                chunk_index=index,
                target_language="Spanish",
            ),
            limit=2,
        )

    records = options["_candidate_results"]
    assert len(records) == 2
    assert records[0]["chunk_index"] == 1
    assert "text" not in records[0]
    assert "source_text" not in records[0]


def test_record_text_candidate_adds_editorial_knowledge_issues(tmp_path, monkeypatch):
    from src.core import translator

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_contract_profile", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "source": "unconscious",
                        "target": "inconsciente",
                        "type": "technical_term",
                        "status": "approved",
                        "translation_policy": "translate_exact",
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    options = {
        "editorial_mode": "book_profile",
        "profile_id": "auto_contract_profile",
    }
    result = translator._record_text_candidate(
        options,
        text="El unconscious aparece de nuevo.",
        source_text="The unconscious appears again.",
        phase="translation",
        source_language="English",
        target_language="Spanish",
    )

    assert any(issue.code == "editorial_term_translation_missing" for issue in result.issues)
    assert result.scores["editorial_matched_terms"] == 1.0
    assert options["_candidate_results"][-1]["scores"]["editorial_matched_terms"] == 1.0
