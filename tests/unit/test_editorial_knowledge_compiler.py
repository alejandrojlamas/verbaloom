from __future__ import annotations

from types import SimpleNamespace

import yaml

from src.core.book_profiles.loader import create_profile, load_book_profile
from src.core.book_profiles.rendering import build_profile_glossary_block
from src.core.book_profiles.models import ProfileGlossaryEntry
from src.core.editorial_knowledge_compiler import compile_profile_prompt_context


def test_compiler_prioritizes_translations_and_withholds_noisy_preserves():
    profile = SimpleNamespace(
        editorial_artifacts={
            "canonical_names": [
                {
                    "name": "Moctezuma",
                    "canonical": "Motecuhzoma",
                    "type": "canonical_proper_noun",
                    "occurrences": 12,
                    "confidence": 0.92,
                }
            ]
        }
    )
    entries = [
        ProfileGlossaryEntry(
            source="unconscious",
            target="inconsciente",
            entry_type="technical_term",
            status="approved",
            translation_policy="translate_exact",
            confidence=0.98,
            occurrences=20,
        ),
        ProfileGlossaryEntry(
            source="dream",
            target="dream",
            entry_type="technical_term",
            status="approved",
            translation_policy="preserve_exact",
            confidence=0.99,
            occurrences=40,
        ),
        ProfileGlossaryEntry(
            source="Tereza",
            target="Tereza",
            entry_type="character",
            status="approved",
            translation_policy="preserve_exact",
            confidence=0.95,
            occurrences=15,
        ),
    ]

    compiled = compile_profile_prompt_context(
        profile,
        "Moctezuma, Tereza and the unconscious dream are discussed.",
        entries,
        purpose="translation",
        max_entries=10,
    )

    sources = [entry.source for entry in compiled.entries]
    assert sources[:2] == ["unconscious", "Tereza"]
    assert "dream" not in sources
    assert any("low-signal preserve" in warning for warning in compiled.warnings)
    assert any("Moctezuma" in hint for hint in compiled.artifact_hints)


def test_profile_glossary_block_uses_compiled_prompt_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_compiler_profile", profiles_root=tmp_path)
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
                        "confidence": 0.99,
                    },
                    {
                        "source": "dream",
                        "target": "dream",
                        "type": "technical_term",
                        "status": "approved",
                        "translation_policy": "preserve_exact",
                        "confidence": 0.99,
                    },
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (profile_dir / "editorial").mkdir(exist_ok=True)
    (profile_dir / "editorial" / "editorial_map.yml").write_text(
        yaml.safe_dump(
            {
                "technical_cultural_terms": [
                    {
                        "source": "unconscious",
                        "target": "inconsciente",
                        "type": "concept",
                        "occurrences": 8,
                    }
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Force load once so corrupt/missing profile metadata would fail here.
    assert load_book_profile("auto_compiler_profile").profile_id == "auto_compiler_profile"

    block = build_profile_glossary_block(
        "The unconscious appears in the dream.",
        {"editorial_mode": "book_profile", "profile_id": "auto_compiler_profile"},
        purpose="translation",
    )

    assert "unconscious -> inconsciente" in block
    assert "dream -> keep exact spelling" not in block
