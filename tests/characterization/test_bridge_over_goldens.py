"""Run the Step 2 legacy bridge over the *real* captured engine outputs.

The unit tests for the bridge use synthetic dicts. This test feeds every
stats dict actually emitted by every format (as captured in the golden
files) through ``snapshot_from_legacy_stats`` and asserts the derived
``percent`` is well-formed and monotonic — i.e. the seam handles the three
real legacy shapes without surprises.

It also pins the captured-bug behavior: ``refine_docx`` emits a single
``0/3`` callback that never advances, so its bridged percent stays at 0. When
the DOCX refine progress is fixed (Step 7) this expectation must be updated.
"""

import json
from pathlib import Path

import pytest

from src.core.progress import snapshot_from_legacy_stats

GOLDEN_DIR = Path(__file__).parent / "golden"

# Formats/modes whose captured run reaches completion (final percent 100).
_COMPLETES = {
    "translate_txt",
    "translate_srt",
    "translate_docx",
    "translate_epub",
    "refine_txt",
    "refine_srt",
    "refine_epub",
    # DOCX refine now emits per-chunk progress (fixed in Step 7); previously it
    # was stuck at a single 0/N callback.
    "refine_docx",
}

_ALL = sorted(_COMPLETES)


def _load_sequence(name: str):
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text(encoding="utf-8"))[
        "sequence"
    ]


@pytest.mark.parametrize("name", _ALL)
def test_bridge_percent_is_monotonic_and_bounded(name):
    percents = [
        snapshot_from_legacy_stats(stats).percent for stats in _load_sequence(name)
    ]
    assert percents == sorted(percents), f"{name}: bridged percent regressed"
    assert all(0.0 <= p <= 100.0 for p in percents), f"{name}: percent out of range"


@pytest.mark.parametrize("name", sorted(_COMPLETES))
def test_completing_runs_reach_100(name):
    final = snapshot_from_legacy_stats(_load_sequence(name)[-1]).percent
    assert final == 100.0, f"{name}: expected final percent 100, got {final}"
