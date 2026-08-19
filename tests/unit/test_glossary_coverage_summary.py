from src.api import handlers


def test_transform_profile_glossary_coverage_counts_target_side_matches(tmp_path, monkeypatch):
    import yaml

    from src.core.book_profiles import create_profile

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("profile_transform")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "señor",
                "target": "usted",
                "type": "address_form",
                "status": "approved",
                "confidence": 0.99,
                "translation_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        handlers,
        "extract_readable_text",
        lambda _path: "Lo vi a usted en la plaza.",
    )

    summary = handlers._profile_glossary_match_summary(
        {
            "file_type": "txt",
            "refine_only": True,
            "prompt_options": {
                "text_transform_mode": "modernize",
                "editorial_mode": "book_profile",
                "profile_id": "profile_transform",
                "use_profile_glossary": True,
            },
        },
        "/tmp/input.txt",
    )

    assert summary == {
        "profile_id": "profile_transform",
        "purpose": "transformation",
        "total_terms": 1,
        "matched_terms": 1,
        "rendered_terms": 1,
        "capped": False,
    }


def test_transform_glossary_coverage_counts_target_side_matches(monkeypatch):
    monkeypatch.setattr(
        handlers,
        "extract_readable_text",
        lambda _path: "Lo vi a usted en la plaza.",
    )

    summary = handlers._glossary_match_summary(
        {
            "file_type": "txt",
            "refine_only": True,
            "prompt_options": {
                "text_transform_mode": "modernize",
                "glossary_terms": {"señor": "usted"},
            },
        },
        "/tmp/input.txt",
    )

    assert summary == {
        "purpose": "transformation",
        "total_terms": 1,
        "matched_terms": 1,
        "capped": False,
    }


def test_translation_glossary_coverage_remains_source_side(monkeypatch):
    monkeypatch.setattr(
        handlers,
        "extract_readable_text",
        lambda _path: "Lo vi a usted en la plaza.",
    )

    summary = handlers._glossary_match_summary(
        {
            "file_type": "txt",
            "prompt_options": {"glossary_terms": {"señor": "usted"}},
        },
        "/tmp/input.txt",
    )

    assert summary == {
        "purpose": "translation",
        "total_terms": 1,
        "matched_terms": 0,
        "capped": False,
    }
