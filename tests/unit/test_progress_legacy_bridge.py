"""Tests for the legacy-stats -> ProgressSnapshot bridge (Step 2 seam).

Each case pins that the canonical ``percent``/``phase`` derived server-side
equals what the frontend's current ``progress-manager.js`` ``update()`` would
compute, so the Step 3 frontend swap is behavior-preserving.
"""

import pytest

from src.core.progress import Phase, snapshot_from_legacy_stats


def test_plain_translation_single_phase():
    snap = snapshot_from_legacy_stats(
        {"total_chunks": 4, "completed_chunks": 2, "failed_chunks": 0}
    )
    assert snap.phase is Phase.TRANSLATING
    assert snap.refinement_enabled is False
    assert snap.percent == 50.0


def test_refine_after_phase1_maps_to_first_half():
    snap = snapshot_from_legacy_stats(
        {
            "total_chunks": 4,
            "completed_chunks": 2,
            "enable_refinement": True,
            "current_phase": 1,
        }
    )
    assert snap.phase is Phase.TRANSLATING
    assert snap.refinement_enabled is True
    assert snap.percent == 25.0  # (2/4)*50


def test_refine_after_phase2_maps_to_second_half():
    snap = snapshot_from_legacy_stats(
        {
            "total_chunks": 4,
            "completed_chunks": 2,
            "enable_refinement": True,
            "current_phase": 2,
        }
    )
    assert snap.phase is Phase.REFINING
    assert snap.refinement_enabled is True
    assert snap.percent == 75.0  # 50 + (2/4)*50


def test_refine_only_is_single_phase_refining():
    snap = snapshot_from_legacy_stats(
        {"total_chunks": 4, "completed_chunks": 2, "refine_only": True}
    )
    assert snap.phase is Phase.REFINING
    assert snap.refinement_enabled is False
    assert snap.percent == 50.0


def test_single_phase_trusts_progress_percent_when_present():
    snap = snapshot_from_legacy_stats(
        {"total_chunks": 4, "completed_chunks": 1, "progress_percent": 42.0}
    )
    # Mirrors the frontend's "trust server progress_percent" branch.
    assert snap.percent == 42.0


def test_refine_after_ignores_progress_percent():
    # When two-phase, the frontend computes from chunks and ignores
    # progress_percent; the bridge must too.
    snap = snapshot_from_legacy_stats(
        {
            "total_chunks": 4,
            "completed_chunks": 4,
            "enable_refinement": True,
            "current_phase": 1,
            "progress_percent": 99.0,
        }
    )
    assert snap.percent == 50.0


def test_progress_percent_bool_is_not_trusted():
    snap = snapshot_from_legacy_stats(
        {"total_chunks": 2, "completed_chunks": 1, "progress_percent": True}
    )
    # True must not be read as 1.0%; falls back to chunk ratio.
    assert snap.percent == 50.0


def test_failed_units_split_and_quality_flag():
    snap = snapshot_from_legacy_stats(
        {"total_chunks": 4, "completed_chunks": 2, "failed_chunks": 1}
    )
    assert snap.units_done == 2
    assert snap.units_failed == 1
    assert snap.units_succeeded == 1
    assert snap.quality_degraded is True


def test_zero_total_no_error():
    snap = snapshot_from_legacy_stats({"total_chunks": 0, "completed_chunks": 0})
    assert snap.percent == 0.0


def test_empty_dict_defaults():
    snap = snapshot_from_legacy_stats({})
    assert snap.phase is Phase.TRANSLATING
    assert snap.percent == 0.0
    assert snap.units_total == 0


def test_to_dict_exposes_canonical_fields():
    d = snapshot_from_legacy_stats(
        {"total_chunks": 4, "completed_chunks": 2, "failed_chunks": 0}
    ).to_dict()
    assert d["phase"] == "translating"
    assert d["percent"] == 50.0
    assert d["units_done"] == 2 and d["units_total"] == 4


@pytest.mark.parametrize("garbage", [None, "x", {}, []])
def test_robust_against_non_numeric_counts(garbage):
    snap = snapshot_from_legacy_stats(
        {"total_chunks": garbage, "completed_chunks": garbage}
    )
    assert snap.percent == 0.0
    assert snap.units_total == 0
