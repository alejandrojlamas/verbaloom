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
