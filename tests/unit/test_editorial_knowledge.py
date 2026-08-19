from __future__ import annotations

from src.core.book_profiles.loader import create_profile, load_book_profile
from src.core.editorial_knowledge import (
    build_editorial_knowledge_base,
    contains_term,
)
from src.core.glossary.models import Glossary, GlossaryTerm


def test_editorial_knowledge_merges_profile_and_classic_glossary(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("film_profile", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        """
entries:
  - source: montage
    target: montaje
    type: technical_term
    status: approved
    confidence: 0.97
  - source: Hollywood
    target: Hollywood
    type: place
    status: approved
    translation_policy: preserve_exact
""",
        encoding="utf-8",
    )
    profile = load_book_profile("film_profile")
    glossary = Glossary(
        name="Manual film terms",
        target_language="Spanish",
        terms=[GlossaryTerm("shot", "plano", category="cinema")],
    )

    knowledge = build_editorial_knowledge_base(profile=profile, glossary=glossary)

    assert knowledge.prompt_diagnostics()["counts"]["translated_terms"] == 2
    assert any(term.source == "Hollywood" and term.should_preserve for term in knowledge.terms)


def test_editorial_knowledge_validates_missing_translation_target():
    glossary = Glossary(
        name="Terms",
        target_language="Spanish",
        terms=[GlossaryTerm("unconscious", "inconsciente", category="concept")],
    )
    knowledge = build_editorial_knowledge_base(glossary=glossary)

    result = knowledge.validate_candidate(
        "The unconscious appears again.",
        "El unconscious aparece otra vez.",
        purpose="translation",
    )

    assert result.clean is False
    assert any(issue.code == "editorial_term_translation_missing" for issue in result.issues)


def test_editorial_knowledge_preserve_policy_requires_source_form():
    glossary = Glossary(
        name="Names",
        target_language="Spanish",
        terms=[GlossaryTerm("Tereza", "Tereza", category="character")],
    )
    knowledge = build_editorial_knowledge_base(glossary=glossary)

    result = knowledge.validate_candidate(
        "Tereza arrived.",
        "Teresa llegó.",
        purpose="translation",
    )

    assert any(issue.code == "editorial_preserve_term_missing" for issue in result.issues)


def test_contains_term_is_accent_and_case_insensitive_with_boundaries():
    assert contains_term("Moctezuma habló.", "moctezuma")
    assert contains_term("MOTECUHZOMA habló.", "Motēcuhzōma")
    assert contains_term("montage", "tag") is False
