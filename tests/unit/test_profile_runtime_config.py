from __future__ import annotations

from pathlib import Path

import yaml

from src.core.book_profiles import create_profile
from src.core.book_profiles.models import BookProfile
from src.core.job_runtime_config import (
    activate_profile_for_translation,
    configure_editorial_guard_options,
    profile_strength,
)


def _profile() -> BookProfile:
    return BookProfile(
        profile_id="book_profile",
        name="Book Profile",
        root=Path("."),
        target_locale="es-MX",
        min_dimension_score=8.5,
        max_repair_rounds=3,
    )


def test_translation_profile_defaults_to_balanced_local_validation():
    options = {}

    activate_profile_for_translation(options, _profile())

    assert profile_strength(options) == "balanced"
    assert options["profile_local_precheck_enabled"] is True
    assert options["profile_audit_enabled"] is False
    assert options["repair_until_pass"] is True
    assert options["max_repair_rounds"] == 1


def test_light_profile_strength_is_prompt_only():
    options = {"profile_strength": "light"}

    activate_profile_for_translation(options, _profile())

    assert options["profile_local_precheck_enabled"] is False
    assert options["profile_audit_enabled"] is False
    assert options["repair_until_pass"] is False
    assert options["max_repair_rounds"] == 0


def test_strict_profile_strength_enables_llm_audit():
    options = {"profile_strength": "strict"}

    activate_profile_for_translation(options, _profile())

    assert options["profile_local_precheck_enabled"] is True
    assert options["profile_audit_enabled"] is True
    assert options["repair_until_pass"] is True


def test_runtime_replaces_explicit_generated_profile_from_another_book(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    sources = {
        "auto_elise": "Elise - Ken Grimwood.epub",
        "auto_into_the_deep": "Into the deep - Ken Grimwood.epub",
    }
    for profile_id, source_name in sources.items():
        profile_dir = create_profile(profile_id, profiles_root=tmp_path)
        profile_path = profile_dir / "profile.yml"
        config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        config.update({
            "source_name": source_name,
            "generated_profile": True,
            "target_locale": "es-MX",
            "auto_detect": {"enabled": False},
        })
        profile_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    config = {
        "input_filename": "Into the deep - Ken Grimwood.epub",
        "output_filename": "Into the deep - Ken Grimwood (Spanish).epub",
        "source_language": "English",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "prompt_options": {
            "editorial_mode": "book_profile",
            "profile_id": "auto_elise",
        },
    }

    options = configure_editorial_guard_options(config)

    assert options["profile_id"] == "auto_into_the_deep"
    assert options["_profile_scope_corrected_from"] == "auto_elise"

    # Keep the internal migration marker across server restarts until the
    # corrected profile can invalidate review and audit in the XHTML state.
    options = configure_editorial_guard_options(config)
    assert options["profile_id"] == "auto_into_the_deep"
    assert options["_profile_scope_corrected_from"] == "auto_elise"
    assert options["_profile_scope_correction"] == "replaced"


def test_epub_translation_declares_its_existing_review_as_inline():
    config = {
        "file_type": "epub",
        "source_language": "English",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "refine_after": True,
        "refine_only": False,
        "prompt_options": {"refine": True},
    }

    options = configure_editorial_guard_options(config)

    assert options["inline_refinement"] is True


def test_strict_flash_epub_defers_duplicate_remote_audits_to_final_stage():
    config = {
        "file_type": "epub",
        "source_language": "English",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "operation": "translate",
        "refine_only": False,
        "prompt_options": {
            "refine": True,
            "strict_stage_contract": True,
            "audit_entire_book": True,
        },
    }

    options = configure_editorial_guard_options(config)

    assert options["defer_intermediate_fidelity_to_final_audit"] is True
    assert options["defer_intermediate_editorial_guard_to_final_audit"] is True
    assert options["quality_alert_model"] == "deepseek-flash"
    assert options["fidelity_supervisor_model"] == "deepseek-v4-pro"


def test_transform_profile_does_not_defer_its_distinct_quality_contract():
    config = {
        "file_type": "epub",
        "source_language": "Spanish",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "model": "deepseek-flash",
        "operation": "transform",
        "prompt_options": {
            "refine": True,
            "text_transform_mode": "modernize",
            "strict_stage_contract": True,
            "audit_entire_book": True,
        },
    }

    options = configure_editorial_guard_options(config)

    assert "defer_intermediate_fidelity_to_final_audit" not in options
    assert "defer_intermediate_editorial_guard_to_final_audit" not in options
    assert options["quality_alert_model"] == "deepseek-v4-pro"


def test_refine_only_epub_does_not_claim_translation_inline_refinement():
    config = {
        "file_type": "epub",
        "source_language": "Spanish",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "refine_only": True,
        "prompt_options": {"refine": True},
    }

    options = configure_editorial_guard_options(config)

    assert "inline_refinement" not in options


def test_inference_preserves_explicit_operation_and_the_book_purpose(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    published = {}
    for goal in ("faithful_translation", "audiobook", "modernization", "explanatory_rewrite", "literary_polish"):
        profile_dir = create_profile("auto_" + goal, profiles_root=tmp_path)
        path = profile_dir / "profile.yml"
        saved = yaml.safe_load(path.read_text())
        saved.update({
            "generated_profile": True, "source_name": "Shared title.txt",
            "source_language": "Spanish", "target_locale": "es-MX", "profile_goal": goal,
        })
        path.write_text(yaml.safe_dump(saved))
        published[goal] = profile_dir.name

    base = {
        "input_filename": "Shared title.txt", "source_language": "Spanish",
        "target_language": "Spanish", "operation": "translate",
    }
    for goal, profile_id in published.items():
        config = {**base, "prompt_options": {"profile_goal": goal}}
        options = configure_editorial_guard_options(config)
        assert options["profile_id"] == profile_id
        assert "text_transform_mode" not in options

    options = configure_editorial_guard_options({
        **base, "operation": "transform",
        "prompt_options": {"profile_goal": "explanatory_rewrite", "text_transform_mode": "simplify"},
    })
    assert options["profile_id"] == published["explanatory_rewrite"]
    assert options["text_transform_mode"] == "simplify"
    assert "profile_id" not in configure_editorial_guard_options({**base, "prompt_options": {}})



def test_french_goal_is_usable_with_or_without_a_regional_locale(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    for index, locale in enumerate(("", "fr-CA")):
        source_name = f"River essay number {index}.txt"
        profile_id = f"auto_river_{index}"
        profile_path = create_profile(profile_id, profiles_root=tmp_path) / "profile.yml"
        saved = yaml.safe_load(profile_path.read_text())
        saved.update({
            "generated_profile": True, "profile_goal": "literary_polish", "source_name": source_name,
            "source_language": "English", "target_language": "French", "target_locale": locale,
        })
        profile_path.write_text(yaml.safe_dump(saved))
        config = {"input_filename": source_name, "source_language": "English",
                  "target_language": "French", "llm_provider": "deepseek", "operation": "translate"}
        for options in ({}, {"profile_id": profile_id, "editorial_mode": "book_profile"}):
            applied = configure_editorial_guard_options({**config, "prompt_options": options})
            assert applied["profile_id"] == profile_id
            assert (applied.get("target_locale") or "") == locale
            assert "text_transform_mode" not in applied
