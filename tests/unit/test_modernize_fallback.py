"""Tests for the modernize fallback policy and prompt architecture.

Covers the failure chain that made modernization jobs return the archaic
source untouched: revert-on-reject gates, unreachable score floors, and the
transformation directives being subordinated to a light-copyedit identity.
"""
import pytest

from src.core.editorial_quality import (
    QualityDecision,
    QualityIssue,
    soften_decision_for_modernize,
)
from src.core.fidelity_supervisor import FidelityDecision, FidelityIssue
from src.core.text_transform import (
    MODERNIZE_FIDELITY_HARD_REJECT_CODES,
    MODERNIZE_HARD_REJECT_CODES,
    apply_faithful_modernize_defaults,
    transform_fallback_mode,
)


def _decision(issues):
    return QualityDecision(
        chunk_index=0,
        section="cap1",
        accepted=not any(i.severity == "reject" for i in issues),
        issues=list(issues),
    )


class TestSoftenDecisionForModernize:
    def test_soft_rejects_become_warnings_and_decision_accepts(self):
        decision = _decision([
            QualityIssue(code="length_regression", severity="reject", message="m"),
            QualityIssue(code="paragraph_collapse", severity="reject", message="m"),
        ])
        out = soften_decision_for_modernize(decision, MODERNIZE_HARD_REJECT_CODES)
        assert out.accepted is True
        assert all(i.severity == "warning" for i in out.issues)
        assert all("softened for modernize" in i.detail for i in out.issues)

    def test_hard_corruption_codes_still_block(self):
        for code in sorted(MODERNIZE_HARD_REJECT_CODES):
            decision = _decision([
                QualityIssue(code=code, severity="reject", message="m"),
            ])
            out = soften_decision_for_modernize(
                decision, MODERNIZE_HARD_REJECT_CODES
            )
            assert out.accepted is False, code
            assert out.rejections and out.rejections[0].code == code

    def test_mixed_hard_and_soft_keeps_blocking(self):
        decision = _decision([
            QualityIssue(code="length_regression", severity="reject", message="m"),
            QualityIssue(code="empty_refinement", severity="reject", message="m"),
        ])
        out = soften_decision_for_modernize(decision, MODERNIZE_HARD_REJECT_CODES)
        assert out.accepted is False
        codes = {i.code: i.severity for i in out.issues}
        assert codes["length_regression"] == "warning"
        assert codes["empty_refinement"] == "reject"

    def test_warnings_untouched(self):
        decision = _decision([
            QualityIssue(code="style_drift", severity="warning", message="m"),
        ])
        out = soften_decision_for_modernize(decision, MODERNIZE_HARD_REJECT_CODES)
        assert out.accepted is True
        assert out.issues[0].severity == "warning"
        assert "softened" not in out.issues[0].detail


class TestTransformFallbackMode:
    def test_default_is_best_candidate(self):
        assert transform_fallback_mode(None) == "best_candidate"
        assert transform_fallback_mode({}) == "best_candidate"

    def test_explicit_source_restores_legacy_revert(self):
        assert transform_fallback_mode({"transform_fallback": "source"}) == "source"

    def test_garbage_falls_back_to_default(self):
        assert transform_fallback_mode({"transform_fallback": "??"}) == "best_candidate"


class TestModernizeFidelityFallback:
    def _fidelity_decision(self, code):
        return FidelityDecision(
            chunk_index=1,
            phase="refinement",
            section="Capitulo I",
            accepted=False,
            issues=[
                FidelityIssue(
                    code=code,
                    severity="reject",
                    message="m",
                )
            ],
        )

    def _profile_options(self):
        return {
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
            "transform_fallback": "best_candidate",
        }

    def test_profile_modernize_keeps_repairable_fidelity_reject_for_audit(self):
        from src.core.translator import _fidelity_decision_requires_source_fallback

        decision = self._fidelity_decision("fidelity_judge_reject")

        assert _fidelity_decision_requires_source_fallback(
            decision,
            self._profile_options(),
        ) is False

    def test_profile_modernize_still_falls_back_on_hard_corruption(self):
        from src.core.translator import _fidelity_decision_requires_source_fallback

        for code in sorted(MODERNIZE_FIDELITY_HARD_REJECT_CODES):
            assert _fidelity_decision_requires_source_fallback(
                self._fidelity_decision(code),
                self._profile_options(),
            ) is True

    def test_non_profile_modernize_keeps_legacy_fidelity_fallback(self):
        from src.core.translator import _fidelity_decision_requires_source_fallback

        assert _fidelity_decision_requires_source_fallback(
            self._fidelity_decision("fidelity_judge_reject"),
            {"text_transform_mode": "modernize"},
        ) is True


class TestModernizeDefaults:
    def _base_options(self):
        return {"text_transform_mode": "modernize"}

    def test_defaults_alerted_supervisor_and_two_repairs(self):
        options = self._base_options()
        apply_faithful_modernize_defaults(options)
        assert options["fidelity_supervisor_mode"] == "alerted"
        assert options["transform_repair_attempts"] == 2
        assert options["transform_fallback"] == "best_candidate"

    def test_user_choice_of_supervisor_mode_is_respected(self):
        options = self._base_options()
        options["fidelity_supervisor_mode"] = "strict_full"
        apply_faithful_modernize_defaults(options)
        assert options["fidelity_supervisor_mode"] == "strict_full"

    def test_explicit_supervisor_model_is_respected(self):
        options = self._base_options()
        options["fidelity_supervisor_model"] = "deepseek-v4-pro"
        apply_faithful_modernize_defaults(options)
        assert options["fidelity_supervisor_model"] == "deepseek-v4-pro"


class TestAsymmetricAuditSeverity:
    def test_low_style_scores_are_medium_low_fidelity_scores_are_high(self):
        from src.core.book_profiles.audit import _append_score_floor_issues

        scores = {
            "content_fidelity": 6.0,           # fidelity -> high
            "syntactic_modernization": 5.0,    # style -> medium
            "contemporary_naturalness": 6.5,   # style -> medium
        }
        issues = []
        _append_score_floor_issues(scores, issues, min_score=8.5)
        by_type = {i.issue_type: i.severity for i in issues}
        assert by_type.get("fidelity_error") == "high"
        assert by_type.get("syntactic_modernization_error") == "medium"
        assert by_type.get("contemporary_naturalness_error") == "medium"

    def test_scores_between_floor_and_seven_stay_medium(self):
        from src.core.book_profiles.audit import _append_score_floor_issues

        scores = {"content_fidelity": 7.8}
        issues = []
        _append_score_floor_issues(scores, issues, min_score=8.5)
        assert issues and issues[0].severity == "medium"


class TestTransformationPrompt:
    def _options(self, **extra):
        options = {
            "text_transform_mode": "modernize",
            "text_transform_profile": "faithful_current_spanish",
        }
        options.update(extra)
        return options

    def test_transform_mode_gets_dedicated_system_identity(self):
        from src.prompts.prompts import generate_refinement_prompt

        pair = generate_refinement_prompt(
            draft_translation="Texto de prueba.",
            target_language="Spanish",
            prompt_options=self._options(),
        )
        assert "active rewriting task" in pair.system
        assert "NOT a reason to skip" in pair.system
        # The light-copyedit identity must not leak into transform jobs.
        assert "Editorial restraint is part of the task" not in pair.system
        assert "TEXT TO TRANSFORM" in pair.user

    def test_plain_refinement_keeps_copyedit_identity(self):
        from src.prompts.prompts import generate_refinement_prompt

        pair = generate_refinement_prompt(
            draft_translation="Texto de prueba.",
            target_language="Spanish",
            prompt_options={},
        )
        assert "TEXT TO TRANSFORM" not in pair.user

    def test_high_strength_drops_conservative_rules(self):
        from src.prompts.prompts import build_text_transform_instructions

        conservative = build_text_transform_instructions(
            self._options(), "Spanish"
        )
        aggressive = build_text_transform_instructions(
            self._options(modernization_strength="high"), "Spanish"
        )
        assert "famous" in conservative
        assert "famous" not in aggressive
        assert "Actively restructure" in aggressive
