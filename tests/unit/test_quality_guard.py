from types import SimpleNamespace

from src.core import quality_guard
from src.core.quality_guard import (
    _build_quality_alert_repair_instructions,
    _content_length_ratio_ok,
    _count_quality_alerts,
    _extract_source_text_for_guard,
    _normalize_guard_text,
    _should_run_source_aware_editorial_guard,
)


def test_extract_source_text_skips_refine_only_draft_alias():
    chunk = {
        "source_text": "  Texto traducido\nactual  ",
        "original_text": "Original source text",
    }

    assert (
        _extract_source_text_for_guard(chunk, "Texto traducido actual")
        == "Original source text"
    )


def test_source_aware_guard_policy_respects_modes_and_local_decision():
    clean = SimpleNamespace(accepted=True, warnings=[])
    warned = SimpleNamespace(accepted=True, warnings=["warning"])
    rejected = SimpleNamespace(accepted=False, warnings=[])

    assert not _should_run_source_aware_editorial_guard(
        clean,
        source_text="source",
        prompt_options={},
    )
    assert _should_run_source_aware_editorial_guard(
        warned,
        source_text="source",
        prompt_options={},
    )
    assert _should_run_source_aware_editorial_guard(
        rejected,
        source_text="source",
        prompt_options={},
    )
    assert _should_run_source_aware_editorial_guard(
        clean,
        source_text="source",
        prompt_options={"source_aware_editorial_guard_mode": "always"},
    )
    assert not _should_run_source_aware_editorial_guard(
        rejected,
        source_text="source",
        prompt_options={"source_aware_editorial_guard": False},
    )


def test_quality_alerts_and_repair_prompt_live_in_extracted_module():
    counts = _count_quality_alerts(
        "Chitón. ¡Oh! ustedes, mis agujas. _¡Así_, renuncio.",
        target_language="Spanish",
        prompt_options={"spanish_variant": "mexican"},
    )

    assert counts == {
        "chiton": 1,
        "markdown_emphasis_residue": 1,
        "awkward_oh_ustedes": 1,
    }
    instructions = _build_quality_alert_repair_instructions(
        counts,
        target_language="Spanish",
    )
    assert "regional register" in instructions
    assert "markdown" in instructions
    assert "alert repair pass" in instructions


def test_guard_normalization_and_length_ratio_boundaries():
    assert _normalize_guard_text("  uno\n\t dos  ") == "uno dos"
    assert _content_length_ratio_ok("a" * 200, "b" * 140)
    assert _content_length_ratio_ok("a" * 200, "b" * 290)
    assert not _content_length_ratio_ok("a" * 200, "b" * 139)
    assert not _content_length_ratio_ok("a" * 200, "b" * 291)
    assert not _content_length_ratio_ok("short", "")


def test_translator_keeps_legacy_private_imports_as_reexports():
    from src.core import translator

    assert translator._normalize_guard_text is quality_guard._normalize_guard_text
    assert translator._count_quality_alerts is quality_guard._count_quality_alerts
