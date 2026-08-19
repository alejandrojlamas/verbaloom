from src.api.blueprints import sample_routes
from src.api.blueprints.sample_routes import (
    _column_prompt_options,
    _glossary_block_for,
    _sample_glossary_purpose,
)


def _glossary_data():
    return {
        "terms_dict": {"señor": "usted"},
        "term_metadata": {"señor": {"category": "formality"}},
        "target_language": "Spanish",
    }


def test_sample_transformation_glossary_matches_rendered_side():
    block = _glossary_block_for(
        _glossary_data(),
        "Usted llegó tarde.",
        purpose="transformation",
    )

    assert "# GLOSSARY - REQUIRED TERM TREATMENT" in block
    assert "señor -> usted" in block
    assert "[formality]" in block


def test_sample_translation_glossary_keeps_source_side_matching():
    block = _glossary_block_for(
        _glossary_data(),
        "Usted llegó tarde.",
        purpose="translation",
    )

    assert block == ""


def test_sample_glossary_purpose_tracks_transform_and_refine_phases():
    assert _sample_glossary_purpose("translate", {"text_transform_mode": "modernize"}, "translate") == "transformation"
    assert _sample_glossary_purpose("translate_refine", {}, "refine") == "refinement"
    assert _sample_glossary_purpose("translate_refine", {}, "translate") == "translation"


def test_column_prompt_options_activates_selected_profile():
    opts = _column_prompt_options({}, {"profile_id": "auto_under_the_volcano_malcolm_lowry"})

    assert opts["editorial_mode"] == "book_profile"
    assert opts["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert opts["use_profile_glossary"] is True
    assert opts["auto_approve_glossary_suggestions"] is False
    assert opts["allow_cross_profile_glossary"] is False


def test_sample_glossary_block_passes_purpose_to_book_profile(monkeypatch):
    calls = []

    def fake_profile_block(text, prompt_options, *, purpose="translation", **_kwargs):
        calls.append((text, prompt_options, purpose))
        return "# ACTIVE BOOK GLOSSARY\n"

    monkeypatch.setattr(sample_routes, "build_profile_glossary_block", fake_profile_block)

    block = _glossary_block_for(
        None,
        "Usted llegó tarde.",
        prompt_options={"profile_id": "book"},
        purpose="transformation",
    )

    assert "# ACTIVE BOOK GLOSSARY" in block
    assert calls == [
        ("Usted llegó tarde.", {"profile_id": "book"}, "transformation"),
    ]
