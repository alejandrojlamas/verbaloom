from __future__ import annotations

import asyncio
from pathlib import Path
import re
import threading
import time
import zipfile

import pytest
import yaml

from src.core.book_profiles import (
    apply_profile_exact_translation_corrections,
    apply_profile_glossary_corrections,
    build_profile_glossary_context,
    build_profile_impact_preview,
    build_profile_instruction_block,
    build_profile_glossary_block,
    build_profile_knowledge_base,
    create_profile,
    infer_profile_id_from_metadata,
    load_book_profile,
    migrate_generated_profile_glossaries,
    profile_glossary_match_summary,
)
from src.core.book_profiles.artifacts import (
    enrich_editorial_map_with_reviewed_terms,
    merge_llm_editorial_map,
    render_editorial_brief,
)
from src.core.book_profiles.audit import (
    ProfileAuditIssue,
    ProfileAuditResult,
    profile_audit_result_from_payload,
    run_profile_precheck,
    score_fields_for_profile,
)
from src.core.book_profiles.discovery import (
    GlossarySuggestion,
    merge_pending_suggestions,
    parse_glossary_discovery_payload,
    suggest_glossary_entries,
)
from src.core.book_profiles.preparation import (
    distributed_discovery_chunks,
    extract_profile_prep_text_from_bytes,
    full_coverage_discovery_chunks,
    prepare_book_profile_from_text,
)
from src.core.book_profiles.glossary_editor import (
    ProfileGlossaryConflictError,
    ProfileGlossaryEditError,
    apply_profile_glossary_action,
    merge_profile_glossary_entries,
)
from src.core.book_profiles.profile_goals import resolve_profile_goal
from src.core.book_profiles.loader import clear_book_profile_cache
from src.core.book_profiles.term_review import (
    review_profile_terms,
    reviewed_candidate_to_approved_entry,
    suspicious_preserve_entry,
    weakly_supported_generated_preserve_entry,
)
from src.core.editorial_quality import EditorialQualityReport
from src.core.output_formats import _write_epub
from src.core.text_transform import apply_faithful_modernize_defaults
from src.prompts.prompts import build_text_transform_instructions


ROOT = Path(__file__).resolve().parents[2]


def test_book_profile_cache_reuses_and_invalidates_profile_snapshot(tmp_path):
    clear_book_profile_cache()
    profile_dir = create_profile("cache_profile", profiles_root=tmp_path)

    first = load_book_profile("cache_profile", profiles_root=tmp_path)
    second = load_book_profile("cache_profile", profiles_root=tmp_path)
    assert second is first

    terms_path = profile_dir / "glossary" / "terms.yml"
    terms_path.write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "source term",
                "target": "target term",
                "status": "approved",
            }],
        }),
        encoding="utf-8",
    )

    reloaded = load_book_profile("cache_profile", profiles_root=tmp_path)
    assert reloaded is not first
    assert any(entry.source == "source term" for entry in reloaded.approved_entries)


def test_profile_prep_detects_misnamed_epub_by_content(tmp_path):
    epub_path = tmp_path / "book.epub"
    _write_epub(
        "Chapter One\n\nThe profile preparation should read this EPUB content.",
        epub_path,
        title="Book",
    )

    text = extract_profile_prep_text_from_bytes(
        epub_path.read_bytes(),
        "Games People Play - The Basics (z-library.sk, 1lib.sk, z-lib.sk)",
    )

    assert "Chapter One" in text
    assert "profile preparation should read this EPUB content" in text


def test_profile_prep_validates_misnamed_epub_before_extraction(tmp_path):
    epub_path = tmp_path / "compressed-book.bin"
    with zipfile.ZipFile(epub_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", "<container />")
        zf.writestr("OEBPS/chapter.xhtml", b"A" * 2_000_000)

    with pytest.raises(ValueError, match=r"compression ratio"):
        extract_profile_prep_text_from_bytes(
            epub_path.read_bytes(),
            "book-without-extension",
        )


def test_profile_prep_handles_deep_epub_markup_without_recursion(tmp_path):
    epub_path = tmp_path / "deep.epub"
    inner = "<p>Deep profile paragraph should initialize the job.</p>"
    for _ in range(1200):
        inner = f"<div>{inner}</div>"
    _write_profile_prep_epub(epub_path, inner)

    text = extract_profile_prep_text_from_bytes(
        epub_path.read_bytes(),
        "The Story of Film (Mark Cousins).epub",
    )

    assert "Deep profile paragraph should initialize the job" in text


def test_profile_prep_epub_uses_nonrecursive_fallback_on_recursion(tmp_path, monkeypatch):
    from src.core.book_profiles import preparation as prep_module

    epub_path = tmp_path / "fallback.epub"
    _write_profile_prep_epub(
        epub_path,
        "<p>The fallback extractor should still read this film book.</p>",
    )

    def raise_recursion(_path):
        raise RecursionError("forced recursion")

    monkeypatch.setattr(prep_module, "extract_readable_text", raise_recursion)

    text = extract_profile_prep_text_from_bytes(
        epub_path.read_bytes(),
        "The Story of Film (Mark Cousins).epub",
    )

    assert "fallback extractor should still read this film book" in text


def test_profile_prep_detects_readable_text_with_unknown_extension():
    text = extract_profile_prep_text_from_bytes(
        b"Plain text can arrive with a misleading .sk suffix.",
        "book-source.sk",
    )

    assert text == "Plain text can arrive with a misleading .sk suffix."


def _write_profile_prep_epub(path: Path, body_html: str) -> None:
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml">'
        f"<body>{body_html}</body>"
        "</html>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>""",
        )
        zf.writestr(
            "OEBPS/content.opf",
            """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <manifest>
    <item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chapter"/>
  </spine>
</package>""",
        )
        zf.writestr("OEBPS/chapter.xhtml", chapter)


def test_profile_prep_rejects_unknown_binary_content():
    with pytest.raises(ValueError, match=r"Unsupported file type"):
        extract_profile_prep_text_from_bytes(
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x00\x01\x02\x03\x04",
            "book-source.sk",
        )


@pytest.fixture
def quijote_profile_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("quijote_mx_contemporary", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config.update({
        "name": "Don Quijote MX",
        "target_locale": "es-MX",
        "business_rules": {"goal": "modernization"},
        "audit_score_fields": [
            "mexican_literary_spanish",
            "cervantine_voice",
        ],
        "allow_cross_profile_glossary": False,
        "min_dimension_score": 8.5,
        "max_repair_rounds": 2,
        "auto_detect": {
            "enabled": True,
            "all": ["quijote"],
            "any": ["cervantes", "mancha", "ingenioso hidalgo"],
        },
        "detectors": [{
            "code": "editorial_structure_error",
            "pattern": r"\.{8,}\s*\d+",
            "severity": "high",
            "message": "Dot-leader table-of-contents noise.",
            "applies_to": "candidate",
        }],
    })
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "prompts" / "modernize.txt").write_text(
        "Moderniza al espanol literario mexicano contemporaneo. Preserve what is merely voice.",
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "agora",
                "target": "ahora",
                "type": "lexical_archaism",
                "status": "approved",
                "confidence": 0.98,
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "treatments.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "vuestra merced",
                    "target_options": ["usted", "señor", "señora", "mantener tratamiento"],
                    "type": "address_form",
                    "status": "approved",
                    "confidence": 0.9,
                    "forbidden_default": "su merced",
                    "decision_rule": "Elegir por contexto según hablante, destinatario y tono.",
                },
                {
                    "source": "vuestras mercedes",
                    "target_options": ["ustedes", "señores", "mantener tratamiento"],
                    "type": "address_form",
                    "status": "approved",
                    "confidence": 0.9,
                    "forbidden_default": "sus mercedes",
                    "decision_rule": "Elegir por contexto según hablante, destinatario y tono.",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return tmp_path


def test_quijote_profile_loads_own_glossary(quijote_profile_root):
    profile = load_book_profile("quijote_mx_contemporary")

    assert profile.profile_id == "quijote_mx_contemporary"
    assert profile.target_locale == "es-MX"
    assert profile.allow_cross_profile_glossary is False
    assert any(entry.source == "agora" and entry.target == "ahora" for entry in profile.approved_entries)
    assert any(entry.source == "vuestra merced" for entry in profile.approved_entries)


def test_profile_glossary_editor_translates_and_preserves_profile_terms(quijote_profile_root):
    apply_profile_glossary_action(
        "quijote_mx_contemporary",
        source_file="treatments.yml",
        entry_index=0,
        action="translate",
        target="usted",
        review_rationale="Use Mexican contemporary treatment by context.",
    )
    profile = load_book_profile("quijote_mx_contemporary")
    treatment = next(entry for entry in profile.glossary_entries if entry.source == "vuestra merced")

    assert treatment.status == "approved"
    assert treatment.target == "usted"
    assert treatment.translation_policy == "translate_consistently"
    assert treatment.injection_policy == "canonical_translation"

    apply_profile_glossary_action(
        "quijote_mx_contemporary",
        source_file="terms.yml",
        entry_index=0,
        action="preserve",
        review_rationale="Force preserving this test term.",
    )
    profile = load_book_profile("quijote_mx_contemporary")
    term = next(entry for entry in profile.glossary_entries if entry.source == "agora")

    assert term.target == "agora"
    assert term.translation_policy == "preserve_exact"
    assert term.injection_policy == "preserve_exact"


def test_profile_glossary_preserve_action_cannot_be_overridden_by_updates(quijote_profile_root):
    result = apply_profile_glossary_action(
        "quijote_mx_contemporary",
        source_file="terms.yml",
        entry_index=0,
        action="preserve",
        expected_source="agora",
        updates={"target": "valor que no debe sobrevivir"},
    )

    assert result.entry is not None
    assert result.entry.target == "agora"
    assert result.entry.translation_policy == "preserve_exact"


def test_profile_glossary_editor_rejects_stale_row_identity(quijote_profile_root):
    with pytest.raises(ProfileGlossaryConflictError, match="changed after this row"):
        apply_profile_glossary_action(
            "quijote_mx_contemporary",
            source_file="terms.yml",
            entry_index=0,
            action="approve",
            expected_source="another term",
        )


def test_profile_glossary_edit_endpoint_returns_conflict_for_stale_row(quijote_profile_root):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    with app.test_client() as client:
        response = client.patch(
            "/api/book-profiles/quijote_mx_contemporary/glossary/entry",
            json={
                "source_file": "terms.yml",
                "entry_index": 0,
                "expected_source": "stale term",
                "action": "approve",
            },
        )

    assert response.status_code == 409
    assert response.get_json()["code"] == "glossary_edit_conflict"


def test_profile_glossary_editor_rejects_common_or_unknown_files(quijote_profile_root):
    with pytest.raises(ProfileGlossaryEditError, match="not editable"):
        apply_profile_glossary_action(
            "quijote_mx_contemporary",
            source_file="common.yml",
            entry_index=0,
            action="approve",
        )


def test_profile_glossary_editor_merges_entries_without_losing_source_variants(quijote_profile_root):
    result = merge_profile_glossary_entries(
        "quijote_mx_contemporary",
        source_file="treatments.yml",
        entry_indices=[0, 1],
        target="usted",
        rationale="Unify related address forms for this profile.",
        expected_sources={0: "vuestra merced", 1: "vuestras mercedes"},
    )

    assert result.entry is not None
    assert result.entry.status == "approved"
    assert result.entry.target == "usted"

    profile = load_book_profile("quijote_mx_contemporary")
    entries = [entry for entry in profile.glossary_entries if entry.source in {"vuestra merced", "vuestras mercedes"}]
    assert len(entries) == 2
    assert all(entry.status == "approved" for entry in entries)
    assert all(entry.target == "usted" for entry in entries)
    assert {entry.review_status for entry in entries} == {"merged", "merged_alias"}

    block = build_profile_glossary_block(
        "Vuestras mercedes pueden continuar.",
        {"editorial_mode": "book_profile", "profile_id": "quijote_mx_contemporary"},
    )
    assert "vuestras mercedes -> usted" in block


def test_profile_impact_preview_explains_translate_preserve_and_pending_terms(quijote_profile_root):
    profile_dir = Path(quijote_profile_root) / "quijote_mx_contemporary"
    (profile_dir / "glossary" / "phrases.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Dulcinea del Toboso",
                "target": "Dulcinea del Toboso",
                "type": "proper_noun",
                "status": "approved",
                "translation_policy": "preserve_exact",
                "injection_policy": "preserve_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "pending_suggestions.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "rocín",
                "target": "caballo flaco",
                "type": "lexical_archaism",
                "status": "pending",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    profile = load_book_profile("quijote_mx_contemporary")

    preview = build_profile_impact_preview(
        profile,
        "Agora vuestra merced mira a Dulcinea del Toboso y al rocín.",
        purpose="transformation",
    )

    assert preview["glossary"]["matched_approved"] >= 3
    assert any(item["source"] == "agora" for item in preview["terms_to_translate"])
    assert any(item["source"] == "Dulcinea del Toboso" for item in preview["terms_to_preserve"])
    assert any(item["source"] == "rocín" for item in preview["pending_suggestions"])
    assert any("pending" in risk for risk in preview["risks"])


def test_profile_impact_preview_matches_rendered_terms_for_transformation(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("target_side_impact", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "residual stream",
                "target": "flujo residual",
                "type": "technical_term",
                "status": "approved",
                "translation_policy": "translate_consistently",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    profile = load_book_profile("target_side_impact")

    preview = build_profile_impact_preview(
        profile,
        "El flujo residual conserva la información.",
        purpose="transformation",
    )

    assert preview["glossary"]["matched_approved"] == 1
    assert preview["glossary"]["matched_for_prompt"] == 1
    assert preview["terms_to_translate"][0]["source"] == "residual stream"


def test_profile_glossary_context_builds_block_and_summary_in_one_match_pass(
    quijote_profile_root,
    monkeypatch,
):
    from src.core.book_profiles import rendering

    calls = 0
    original = rendering._match_entries

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rendering, "_match_entries", counted)
    block, summary = build_profile_glossary_context(
        "Agora vuestra merced escucha.",
        {"editorial_mode": "book_profile", "profile_id": "quijote_mx_contemporary"},
        purpose="transformation",
    )

    assert calls == 1
    assert "agora" in block
    assert summary["matched_terms"] >= 2


def test_profile_glossary_quotes_symbol_bearing_name_in_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("symbol_names", profiles_root=tmp_path)
    terms_path = profile_dir / "glossary" / "terms.yml"
    terms_path.write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Ch*Tril",
                "target": "Ch*Tril",
                "type": "proper_noun",
                "status": "approved",
                "confidence": 0.99,
                "translation_policy": "preserve_exact",
                "injection_policy": "preserve_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    block = build_profile_glossary_block(
        "Ch*Tril entered the water.",
        {"editorial_mode": "book_profile", "profile_id": "symbol_names"},
    )

    assert "`Ch*Tril` -> keep exact spelling" in block


def test_symbol_bearing_name_restoration_is_generic_and_source_bounded():
    source = "Ch*Tril greeted Dj\\Tal before the dive."
    damaged = "ChTril saludó a Dj Tal antes de la inmersión."

    restored = apply_profile_glossary_corrections(
        damaged,
        {},
        source_text=source,
    )

    assert restored == "Ch*Tril saludó a Dj\\Tal antes de la inmersión."
    assert apply_profile_glossary_corrections(
        damaged,
        {},
        source_text="Another character greeted them.",
    ) == damaged


def test_quijote_profile_can_be_inferred_from_filename_metadata(quijote_profile_root):
    inferred = infer_profile_id_from_metadata({
        "output_filename": "Miguel de Cervantes El Ingenioso Hidalgo Don Quijote de la Mancha (Modernizar).epub",
    })

    assert inferred == "quijote_mx_contemporary"


def test_unrelated_book_does_not_infer_quijote_profile():
    inferred = infer_profile_id_from_metadata({
        "output_filename": "Sor Juana Ines de la Cruz poemas (Modernizar).epub",
    })

    assert inferred is None


def test_generated_profile_can_be_inferred_from_source_name(tmp_path):
    profile_dir = create_profile("auto_new_expedition", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["source_name"] = "New Expedition.txt"
    config["target_locale"] = "es-MX"
    config["generated_profile"] = True
    config["auto_detect"] = {"enabled": False}
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    inferred = infer_profile_id_from_metadata(
        {"input_filename": "New Expedition (Spanish).epub"},
        profiles_root=tmp_path,
    )

    assert inferred == "auto_new_expedition"


def test_generated_profile_inference_prefers_completed_profile_over_empty_duplicate(tmp_path):
    source_name = "Die Ringe des Saturn - Eine englische Wallfahrt.epub"
    completed = create_profile("auto_die_ringe_des_saturn", profiles_root=tmp_path)
    duplicate = create_profile("auto_die_ringe_des_saturn_20260711", profiles_root=tmp_path)
    for profile_dir in (completed, duplicate):
        profile_path = profile_dir / "profile.yml"
        config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        config.update({
            "source_name": source_name,
            "generated_profile": True,
            "auto_detect": {"enabled": False},
        })
        profile_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    (completed / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Wirklichkeit",
                "target": "realidad",
                "status": "approved",
            }],
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    inferred = infer_profile_id_from_metadata(
        {"input_filename": "Die Ringe des Saturn Eine englische Wallfahrt.epub"},
        profiles_root=tmp_path,
    )

    assert inferred == "auto_die_ringe_des_saturn"


def test_generated_profiles_do_not_match_another_book_by_author_only(tmp_path):
    books = {
        "auto_elise_ken_grimwood": "Elise - Ken Grimwood.epub",
        "auto_into_the_deep_ken_grimwood": "Into the deep - Ken Grimwood.epub",
    }
    for profile_id, source_name in books.items():
        profile_dir = create_profile(profile_id, profiles_root=tmp_path)
        profile_path = profile_dir / "profile.yml"
        config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
        config.update({
            "source_name": source_name,
            "generated_profile": True,
            "auto_detect": {"enabled": False},
        })
        profile_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    inferred = infer_profile_id_from_metadata(
        {"input_filename": "Into the deep - Ken Grimwood.epub"},
        profiles_root=tmp_path,
    )

    assert inferred == "auto_into_the_deep_ken_grimwood"


def test_generated_profile_inference_tolerates_mobile_filename_variants(tmp_path):
    profile_dir = create_profile("auto_under_the_volcano_malcolm_lowry", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["source_name"] = "Under the Volcano - Malcolm Lowry.epub"
    config["target_locale"] = "es-MX"
    config["generated_profile"] = True
    config["auto_detect"] = {"enabled": False}
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    mobile_names = [
        "under_the_volcano_malcolm_lowry_12345.epub",
        "Under the Volcano (Malcolm Lowry) (Z-Library).epub",
        "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
        "content://com.android.providers.media.documents/document/Under the Volcano Malcolm Lowry.epub",
        "Under the Volcano (Spanish).epub",
    ]

    for filename in mobile_names:
        assert infer_profile_id_from_metadata(
            {"input_filename": filename},
            profiles_root=tmp_path,
        ) == "auto_under_the_volcano_malcolm_lowry"


def test_quijote_modernize_request_autoloads_profile_defaults(quijote_profile_root):
    from src.api.handlers import _configure_editorial_guard_options

    config = {
        "source_language": "Spanish",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "output_filename": "Miguel de Cervantes El Ingenioso Hidalgo Don Quijote de la Mancha (Modernizar).epub",
        "prompt_options": {
            "text_transform_mode": "modernize",
            "transform_auditor_model": "deepseek-v4-flash",
            "fidelity_supervisor_mode": "alerted",
            "fidelity_supervisor_model": "deepseek-v4-flash",
            "source_aware_editorial_guard_mode": "always",
        },
    }

    options = _configure_editorial_guard_options(config)

    assert options["editorial_mode"] == "book_profile"
    assert options["profile_id"] == "quijote_mx_contemporary"
    assert options["transform_auditor_model"] == "deepseek-v4-pro"
    assert options["fidelity_supervisor_mode"] == "always"
    assert options["fidelity_supervisor_model"] == "deepseek-v4-pro"
    assert options["source_aware_editorial_guard_mode"] == "off"
    assert options["abort_on_profile_fail"] is True


def test_quijote_same_language_request_autoloads_modernize_profile_defaults(quijote_profile_root):
    from src.api.handlers import _configure_editorial_guard_options

    config = {
        "source_language": "autodetectar",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "output_filename": "Miguel de Cervantes El Ingenioso Hidalgo Don Quijote de la Mancha (Spanish).txt",
        "prompt_options": {
            "refine": True,
            "source_aware_editorial_guard_mode": "always",
        },
    }

    options = _configure_editorial_guard_options(config)

    assert options["text_transform_mode"] == "modernize"
    assert options["editorial_mode"] == "book_profile"
    assert options["profile_id"] == "quijote_mx_contemporary"
    assert options["fidelity_supervisor_mode"] == "always"
    assert options["fidelity_supervisor_model"] == "deepseek-v4-pro"
    assert options["source_aware_editorial_guard_mode"] == "off"


def test_unrelated_same_language_request_does_not_load_quijote_profile():
    from src.api.handlers import _configure_editorial_guard_options

    config = {
        "source_language": "Spanish",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "output_filename": "Manual de botanica (Spanish).txt",
        "prompt_options": {
            "refine": True,
        },
    }

    options = _configure_editorial_guard_options(config)

    assert "text_transform_mode" not in options
    assert "profile_id" not in options
    assert options["source_aware_editorial_guard_mode"] == "alerted"


def test_generated_profile_translation_request_uses_profile_without_modernizing(tmp_path, monkeypatch):
    from src.api.handlers import _configure_editorial_guard_options

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_new_expedition", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config_payload.update({
        "source_name": "New Expedition.txt",
        "target_locale": "es-MX",
        "generated_profile": True,
        "auto_detect": {"enabled": False},
    })
    profile_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    config = {
        "source_language": "English",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "input_filename": "New Expedition.txt",
        "output_filename": "New Expedition (Spanish).epub",
        "prompt_options": {
            "refine": True,
        },
    }

    options = _configure_editorial_guard_options(config)

    assert options["editorial_mode"] == "book_profile"
    assert options["profile_id"] == "auto_new_expedition"
    assert "text_transform_mode" not in options
    assert options["translation_profile_mode"] is True
    assert options["use_profile_glossary"] is True
    assert options["profile_audit_enabled"] is False
    assert options["source_aware_editorial_guard_mode"] == "alerted"
    assert options["fidelity_supervisor_mode"] == "alerted"


def test_generated_profile_mobile_translation_filename_activates_profile(tmp_path, monkeypatch):
    from src.api.handlers import _configure_editorial_guard_options

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_under_the_volcano_malcolm_lowry", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config_payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config_payload.update({
        "source_name": "Under the Volcano - Malcolm Lowry.epub",
        "target_locale": "es-MX",
        "generated_profile": True,
        "auto_detect": {"enabled": False},
    })
    profile_path.write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    config = {
        "source_language": "English",
        "target_language": "Spanish",
        "llm_provider": "deepseek",
        "input_filename": "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
        "output_filename": "Under the Volcano (Spanish).epub",
        "prompt_options": {
            "refine": True,
        },
    }

    options = _configure_editorial_guard_options(config)

    assert options["editorial_mode"] == "book_profile"
    assert options["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert "text_transform_mode" not in options
    assert options["translation_profile_mode"] is True
    assert options["use_profile_glossary"] is True


def test_profile_glossary_block_is_scoped_to_active_profile(quijote_profile_root):
    text = "Agora que es de dia, vuestra merced vera el camino."
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "quijote_mx_contemporary",
    }

    block = build_profile_glossary_block(text, options)

    assert "# ACTIVE BOOK GLOSSARY" in block
    assert "agora -> ahora" in block
    assert "vuestra merced" in block
    assert build_profile_glossary_block(text, {}) == ""

    rendered_block = build_profile_glossary_block(
        "Ahora que es de dia, usted vera el camino.",
        options,
        purpose="refinement",
    )
    assert "agora -> ahora" in rendered_block


def test_profile_glossary_block_suppresses_preserve_exact_fragments(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("under_noise_sample", profiles_root=tmp_path)
    terms_path = profile_dir / "glossary" / "terms.yml"
    terms_path.write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Yvonne and Hugh",
                    "target": "Yvonne and Hugh",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Looks like a recurring named entity for this book profile.",
                },
                {
                    "source": "Oh Hugh",
                    "target": "Oh Hugh",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Looks like a recurring named entity for this book profile.",
                },
                {
                    "source": "Yvonne's",
                    "target": "Yvonne's",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Looks like a recurring named entity for this book profile.",
                },
                {
                    "source": "Yvonne Griffaton",
                    "target": "Yvonne Griffaton",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Looks like a recurring named entity for this book profile.",
                },
                {
                    "source": "Quauhnahuac",
                    "target": "Quauhnahuac",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Canonical place spelling for this novel.",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    block = build_profile_glossary_block(
        "Oh Hugh met Yvonne and Hugh near Quauhnahuac. Yvonne's letter mentioned Yvonne Griffaton.",
        {"editorial_mode": "book_profile", "profile_id": "under_noise_sample"},
    )

    assert "Yvonne and Hugh" not in block
    assert "Oh Hugh" not in block
    assert "Yvonne's" not in block
    assert "Hugh -> keep exact spelling" in block
    assert "Yvonne -> keep exact spelling" in block
    assert "Yvonne Griffaton -> keep exact spelling" in block
    assert "Quauhnahuac -> keep exact spelling" in block
    assert "Translation policy:" not in block
    assert "Looks like a recurring named entity" not in block
    assert "Canonical place spelling" in block


def test_profile_glossary_suppresses_weak_generated_preserve_rules_and_stale_artifacts(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_dialogue_noise", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["generated_profile"] = True
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    generic = "Looks like a recurring named entity for this book profile."
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Jesus Christ",
                    "target": "Jesus Christ",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "reviewed_by": "deterministic_profile_term_review",
                    "rationale": generic,
                },
                {
                    "source": "West Coast",
                    "target": "West Coast",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "reviewed_by": "deterministic_profile_term_review",
                    "rationale": generic,
                },
                {
                    "source": "Ch*Tril",
                    "target": "Ch*Tril",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "reviewed_by": "deterministic_profile_term_review",
                    "rationale": generic,
                },
                {
                    "source": "Daniel Colter",
                    "target": "Daniel Colter",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "reviewed_by": "llm_profile_term_review:deepseek-v4-pro",
                    "rationale": "Full name of the recurring protagonist.",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "editorial" / "editorial_map.yml").write_text(
        yaml.safe_dump({
            "version": "editorial-prep-v3",
            "canonical_names": [
                {
                    "source": "Jesus Christ",
                    "target": "Jesus Christ",
                    "type": "proper_noun",
                    "policy": "preserve_exact",
                    "source_kind": "reviewed_glossary",
                    "rationale": generic,
                    "occurrences": 3,
                    "confidence": 0.9,
                }
            ],
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    options = {"editorial_mode": "book_profile", "profile_id": "auto_dialogue_noise"}
    text = "Jesus Christ spoke near the West Coast while Ch*Tril met Daniel Colter."
    block = build_profile_glossary_block(text, options)
    instruction = build_profile_instruction_block(
        options,
        phase="translation",
        target_language="Spanish",
    )

    assert "Jesus Christ -> keep exact spelling" not in block
    assert "Jesus Christ: keep/canonicalize" not in block
    assert "West Coast -> keep exact spelling" not in block
    assert "Jesus -> keep exact spelling" not in block
    assert "West -> keep exact spelling" not in block
    assert "`Ch*Tril` -> keep exact spelling" in block
    assert "Daniel Colter -> keep exact spelling" in block
    assert "Suppressed unsafe generated preserve rules: 2" in instruction


def test_profile_glossary_uses_editorial_map_entities_as_contextual_aliases(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("under_editorial_alias_sample", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({"entries": []}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "editorial" / "editorial_map.yml").write_text(
        yaml.safe_dump({
            "version": "editorial-prep-v3",
            "characters_entities": [
                {
                    "name": "Yvonne",
                    "type": "character",
                    "occurrences": 352,
                    "confidence": 0.98,
                },
                {
                    "name": "Hugh",
                    "type": "character",
                    "occurrences": 40,
                    "confidence": 0.98,
                },
                {
                    "name": "Mexico",
                    "type": "character",
                    "occurrences": 27,
                    "confidence": 0.98,
                },
            ],
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    options = {
        "editorial_mode": "book_profile",
        "profile_id": "under_editorial_alias_sample",
    }
    block = build_profile_glossary_block(
        "Yvonne walked with Hugh through Mexico.",
        options,
    )

    assert "Yvonne -> prefer Yvonne when context fits" in block
    assert "Hugh -> prefer Hugh when context fits" in block
    assert "Mexico" not in block
    assert "Derived from the profile editorial map." in block

    summary = profile_glossary_match_summary(
        "Yvonne walked with Hugh through Mexico.",
        options,
    )
    assert summary["total_terms"] == 2
    assert summary["matched_terms"] == 2


def test_profile_glossary_block_prioritizes_translations_when_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("under_priority_sample", profiles_root=tmp_path)
    terms_path = profile_dir / "glossary" / "terms.yml"
    entries = [
        {
            "source": f"PlaceName{i}",
            "target": f"PlaceName{i}",
            "type": "proper_noun",
            "status": "approved",
            "confidence": 0.9,
            "occurrences": i,
            "translation_policy": "preserve_exact",
            "injection_policy": "preserve",
        }
        for i in range(55)
    ]
    entries.append({
        "source": "anti-Semitism",
        "target": "antisemitismo",
        "type": "concept",
        "status": "approved",
        "confidence": 0.98,
        "occurrences": 1,
        "translation_policy": "translate_exact",
    })
    terms_path.write_text(
        yaml.safe_dump({"entries": entries}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    profile = load_book_profile("under_priority_sample")
    assert any(entry.source == "anti-Semitism" and entry.occurrences == 1 for entry in profile.approved_entries)

    text = "anti-Semitism " + " ".join(f"PlaceName{i}" for i in range(55))
    block = build_profile_glossary_block(
        text,
        {"editorial_mode": "book_profile", "profile_id": "under_priority_sample"},
    )

    rendered_entries = [line for line in block.splitlines() if line.startswith("- ")]
    assert len(rendered_entries) == 48
    assert rendered_entries[0].startswith("- anti-Semitism -> antisemitismo")
    assert "Showing 48 highest-signal matches out of 56 matched entries." in block


def test_profile_glossary_acronyms_do_not_match_lowercase_words(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("under_acronym_sample", profiles_root=tmp_path)
    terms_path = profile_dir / "glossary" / "terms.yml"
    terms_path.write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "TALK",
                    "target": "TALK",
                    "type": "acronym",
                    "status": "approved",
                    "confidence": 0.95,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                    "rationale": "Looks like a real acronym or identifier that should remain stable.",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {"editorial_mode": "book_profile", "profile_id": "under_acronym_sample"}

    assert build_profile_glossary_block("They talk in the cantina.", options) == ""
    assert "TALK -> keep exact spelling" in build_profile_glossary_block("The sign reads TALK.", options)


def test_transform_prompt_receives_profile_policy(quijote_profile_root):
    prompt = build_text_transform_instructions(
        {
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
        },
        "Spanish",
    )

    assert "BOOK EDITORIAL PROFILE" in prompt
    assert "quijote_mx_contemporary" in prompt
    assert "espanol literario mexicano contemporaneo" in prompt
    assert "merely" in prompt
    assert "If a passage is already good for this mode" not in prompt


def test_translation_profile_prompt_does_not_fall_back_to_modernize_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("sample_translation_profile", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    prompts_dir = profile_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "modernize.txt").write_text(
        "MODERNIZE_ONLY_PROFILE_PROMPT",
        encoding="utf-8",
    )
    config["prompts"] = {"modernize": "prompts/modernize.txt"}
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    block = build_profile_instruction_block(
        {
            "editorial_mode": "book_profile",
            "profile_id": "sample_translation_profile",
            "_source_language": "English",
        },
        phase="translation",
        target_language="Spanish",
    )

    assert "Translation use of this profile" in block
    assert "faithful translation from English to Spanish" in block
    assert "MODERNIZE_ONLY_PROFILE_PROMPT" not in block


def test_profile_instruction_block_includes_compact_knowledge_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("profile_with_knowledge", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Tenochtitlan",
                    "target": "Tenochtitlan",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                },
                {
                    "source": "steel mill",
                    "target": "acería",
                    "type": "technical_term",
                    "status": "approved",
                    "translation_policy": "translate",
                },
            ]
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "pending_suggestions.yml").write_text(
        yaml.safe_dump({
            "suggestions": [{
                "source": "factory floor",
                "target": "piso de producción",
                "type": "technical_term",
                "status": "pending",
            }]
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    block = build_profile_instruction_block(
        {
            "editorial_mode": "book_profile",
            "profile_id": "profile_with_knowledge",
            "_source_language": "English",
        },
        phase="translation",
        target_language="Spanish",
    )

    assert "PROFILE KNOWLEDGE SNAPSHOT" in block
    assert "1 translate, 1 preserve" in block
    assert "Source-equals-target approved entries: 1" in block
    assert "ordinary technical or cultural terms should remain untranslated" in block
    assert "Pending glossary suggestions: 1" in block


def test_quijote_treatment_detector_flags_suspicious_default(quijote_profile_root):
    result = run_profile_precheck(
        "No le dije yo a vuestra merced que mirase bien lo que hacia?",
        "No le dije yo a su merced que mirara bien lo que hacia?",
        profile_id="quijote_mx_contemporary",
    )

    assert result.overall_decision == "fail"
    assert any(issue.issue_type == "treatment_error" for issue in result.issues)


def test_quijote_treatment_detector_flags_suspicious_plural_default(quijote_profile_root):
    result = run_profile_precheck(
        "Non fuyan las vuestras mercedes.",
        "No huyan sus mercedes.",
        profile_id="quijote_mx_contemporary",
    )

    assert result.overall_decision == "fail"
    assert any(issue.issue_type == "treatment_error" for issue in result.issues)


def test_quijote_profile_flags_toc_dot_leader_noise(quijote_profile_root):
    result = run_profile_precheck(
        "Capítulo I Que trata de la condición. . . . . . . . . 23",
        "Capítulo I Que trata de la condición........................................ 23",
        profile_id="quijote_mx_contemporary",
    )

    assert result.overall_decision == "fail"
    assert any(
        issue.issue_type == "editorial_structure_error"
        for issue in result.issues
    )


def test_profile_audit_payload_cannot_pass_below_profile_threshold():
    payload = {
        "scores": {
            "content_fidelity": 10,
            "facts_names_order": 10,
            "orthographic_modernization": 10,
            "syntactic_modernization": 8.4,
            "contemporary_naturalness": 10,
            "mexican_literary_spanish": 10,
            "editorial_consistency": 10,
            "cervantine_voice": 10,
            "voice_differentiation": 10,
            "no_censorship_summary_omission": 10,
            "glossary_compliance": 10,
            "profile_isolation": 10,
        },
        "overall_decision": "pass",
        "issues": [],
    }

    result = profile_audit_result_from_payload(
        "quijote_mx_contemporary",
        payload,
        min_score=8.5,
        score_fields=tuple(payload["scores"]),
    )

    assert result.overall_decision == "warn"


def test_profile_audit_score_fields_follow_profile_goal(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("audiobook_profile", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["business_rules"] = {"goal": "audiobook"}
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    profile = load_book_profile("audiobook_profile")
    fields = score_fields_for_profile(profile)

    assert "listenability" in fields
    assert "caption_integration" in fields
    assert "content_fidelity" in fields
    assert "cervantine_voice" not in fields
    assert "mexican_literary_spanish" not in fields


def test_faithful_translation_audit_has_no_work_specific_dimensions(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("faithful_profile", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["business_rules"] = {"goal": "faithful_translation"}
    config["audit_dimensions"] = "editorial_full"
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    fields = score_fields_for_profile(load_book_profile("faithful_profile"))

    assert "content_fidelity" in fields
    assert "terminology_translation" in fields
    assert "authorial_voice" in fields
    assert "cervantine_voice" not in fields
    assert "mexican_literary_spanish" not in fields
    assert "syntactic_modernization" not in fields


def test_low_style_scores_warn_instead_of_hard_fail():
    payload = {
        "scores": {
            "content_fidelity": 10,
            "facts_names_order": 10,
            "orthographic_modernization": 8.0,
            "syntactic_modernization": 4.0,
            "contemporary_naturalness": 5.0,
            "mexican_literary_spanish": 5.0,
            "editorial_consistency": 6.0,
            "cervantine_voice": 8.0,
            "voice_differentiation": 8.0,
            "no_censorship_summary_omission": 10,
            "glossary_compliance": 8.0,
            "profile_isolation": 10,
        },
        "overall_decision": "warn",
        "issues": [],
    }

    result = profile_audit_result_from_payload(
        "quijote_mx_contemporary",
        payload,
        min_score=8.5,
        score_fields=tuple(payload["scores"]),
    )

    assert result.overall_decision == "warn"
    assert not any(issue.severity == "high" for issue in result.issues)


def test_profile_style_alert_does_not_require_source_fallback():
    from src.core.translator import (
        _profile_audit_failure_should_abort,
        _profile_audit_requires_source_fallback,
    )

    style_result = ProfileAuditResult(
        profile_id="quijote_mx_contemporary",
        scores={
            "content_fidelity": 10,
            "facts_names_order": 10,
            "orthographic_modernization": 8,
            "syntactic_modernization": 4,
            "contemporary_naturalness": 5,
            "mexican_literary_spanish": 5,
            "editorial_consistency": 6,
            "cervantine_voice": 8,
            "voice_differentiation": 8,
            "no_censorship_summary_omission": 10,
            "glossary_compliance": 8,
            "profile_isolation": 10,
        },
        overall_decision="fail",
        issues=[
            ProfileAuditIssue(
                issue_type="mexican_style_error",
                severity="high",
                reason="Style still needs repair.",
            )
        ],
    )
    fidelity_result = ProfileAuditResult(
        profile_id="quijote_mx_contemporary",
        scores={**style_result.scores, "content_fidelity": 6},
        overall_decision="fail",
        issues=[],
    )

    assert _profile_audit_requires_source_fallback(style_result) is False
    assert _profile_audit_requires_source_fallback(fidelity_result) is True
    assert _profile_audit_failure_should_abort(
        style_result,
        {
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
            "abort_on_profile_fail": True,
        },
    ) is False
    assert _profile_audit_failure_should_abort(
        fidelity_result,
        {
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
            "abort_on_profile_fail": True,
        },
    ) is True
    assert _profile_audit_failure_should_abort(
        style_result,
        {
            "text_transform_mode": "modernize",
            "editorial_mode": "book_profile",
            "profile_id": "quijote_mx_contemporary",
            "abort_on_profile_fail": False,
        },
    ) is False


def test_profile_precheck_flags_residual_old_spanish_in_high_modernization(quijote_profile_root):
    source = (
        "SERÍA el gran Moctezuma de edad de hasta cuarenta años, e cenceño e "
        "pocas carnes, y la color no muy moreno. Señor Moctezuma, bien podéis "
        "creer que si os queréis ir a vuestros palacios, traíanle frutas y "
        "servíase con barro de Cholula. " * 2
    )

    result = run_profile_precheck(
        source,
        source.replace("Montezuma", "Moctezuma"),
        profile_id="quijote_mx_contemporary",
    )

    issues = {issue.issue_type: issue for issue in result.issues}
    assert issues["modernization_residue"].severity == "high"
    assert result.scores["syntactic_modernization"] < 8.5
    assert result.scores["contemporary_naturalness"] < 8.5
    assert result.overall_decision == "fail"


def test_profile_modernize_uses_profile_audit_not_generic_source_guard():
    options = apply_faithful_modernize_defaults({
        "text_transform_mode": "modernize",
        "editorial_mode": "book_profile",
        "profile_id": "quijote_mx_contemporary",
    })

    assert options["source_aware_editorial_guard"] is False
    assert options["source_aware_editorial_guard_mode"] == "off"
    assert options["fidelity_supervisor"] is True
    assert options["fidelity_supervisor_mode"] == "always"


def test_editorial_report_can_include_profile_summary():
    report = EditorialQualityReport(
        document_name="sample.txt",
        target_language="Spanish",
        profile_summary={
            "profile_id": "quijote_mx_contemporary",
            "profile_name": "Don Quijote",
            "target_locale": "es-MX",
            "approved_glossary_entries": 14,
            "pending_glossary_suggestions": 0,
            "allow_common_glossary": True,
            "allow_cross_profile_glossary": False,
        },
    )

    markdown = report.to_markdown()

    assert "Perfil editorial activo" in markdown
    assert "quijote_mx_contemporary" in markdown


def test_new_profile_starts_empty_and_does_not_load_quijote(tmp_path):
    profile_dir = create_profile("other_book", profiles_root=tmp_path)
    profile = load_book_profile("other_book", profiles_root=tmp_path)

    assert profile_dir.name == "other_book"
    assert profile.approved_entries == ()
    assert not any(entry.source == "agora" for entry in profile.glossary_entries)


def test_glossary_discovery_writes_pending_not_approved(tmp_path, monkeypatch):
    create_profile("source_book", profiles_root=tmp_path)
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))

    suggestions = suggest_glossary_entries(
        "d'alguna forma d'alguna manera d'alguna cosa",
        profile_id="source_book",
        max_suggestions=10,
    )
    suggestions.append(GlossarySuggestion(
        source="formula recurrente",
        suggested_target="",
        scope="source_book",
        confidence=0.91,
    ))
    pending_path = merge_pending_suggestions("source_book", suggestions)
    payload = yaml.safe_load(pending_path.read_text(encoding="utf-8"))

    assert payload["suggestions"]
    assert all(item["status"] == "pending" for item in payload["suggestions"])


def test_glossary_discovery_parser_accepts_bounded_compact_map_items():
    payload = parse_glossary_discovery_payload(
        '<GLOSSARY_DISCOVERY_JSON>{'
        '"suggestions": [], '
        '"editorial_map": {'
        '"translatable_terms": ["Wirklichkeit"], '
        '"risks": ["German common nouns are capitalized"]'
        '}}</GLOSSARY_DISCOVERY_JSON>'
    )

    assert payload["valid_schema"] is True
    assert payload["editorial_map"]["translatable_terms"] == [
        {"source": "Wirklichkeit"}
    ]
    assert payload["editorial_map"]["risks"] == [
        {"label": "German common nouns are capitalized"}
    ]


def test_glossary_discovery_parser_rejects_wrong_schema_even_when_json_is_valid():
    payload = parse_glossary_discovery_payload('{"analysis": "free-form response"}')

    assert payload["valid_schema"] is False
    assert payload["suggestions"] == []
    assert payload["editorial_map"] == {}


def test_prepare_book_profile_runs_preflight_discovery_into_scoped_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))

    class FakeProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<GLOSSARY_DISCOVERY_JSON>{"suggestions": ['
                    '{"source": "Terminus Prime", "suggested_target": "Terminus Prime", '
                    '"type": "proper_noun", "confidence": 0.91, '
                    '"rationale": "Recurring place name for this book."},'
                    '{"source": "lost its way", "suggested_target": "", '
                    '"type": "idiom", "confidence": 0.83, '
                    '"rationale": "Recurring formula that may need consistent handling."},'
                    '{"source": "The narrator returned to Terminus Prime whenever the expedition lost its way. '
                    'The narrator returned to Terminus Prime whenever the expedition lost its way.", '
                    '"suggested_target": "", "type": "syntax_pattern", "confidence": 0.77, '
                    '"rationale": "This should be rejected because it copies prose."}'
                    "]}</GLOSSARY_DISCOVERY_JSON>"
                )
            return Response()

    text = (
        "Terminus Prime appeared in every chapter. The narrator returned to "
        "Terminus Prime whenever the expedition lost its way. "
    ) * 20
    progress_events = []

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="New Expedition.txt",
        language="English",
        target_locale="es-MX",
        llm_provider=FakeProvider(),
        provider_name="deepseek",
        model="deepseek-v4-flash",
        max_llm_chunks=2,
        llm_full_coverage=True,
        progress_callback=progress_events.append,
    ))

    profile = load_book_profile(result.profile_id)
    assert result.profile_id.startswith("auto_new_expedition")
    assert profile.raw_config["generated_profile"] is True
    assert profile.allow_cross_profile_glossary is False
    assert not any(entry.source == "Terminus Prime" for entry in profile.approved_entries)
    assert any(entry.source == "Terminus Prime" for entry in profile.pending_entries)
    assert any(entry.source == "lost its way" for entry in profile.pending_entries)
    assert not any("The narrator returned" in entry.source for entry in profile.pending_entries)
    assert not any(entry.source == "agora" for entry in profile.glossary_entries)
    assert result.coverage_mode == "full"
    assert (profile.root / "editorial" / "signal_index.yml").exists()
    reloaded = load_book_profile(result.profile_id)
    knowledge = build_profile_knowledge_base(reloaded).to_dict()
    assert knowledge["signal_index"]["available"] is True
    assert knowledge["signal_index"]["buckets"]["local_candidates"]["total"] >= 1
    assert knowledge["signal_index"]["coverage"]["mode"] == "full"
    assert progress_events
    assert progress_events[-1]["stage"] == "completed"
    assert any(event["stage"] == "llm_discovery" for event in progress_events)


def test_prepare_book_profile_reuses_existing_generated_profile_with_glossary(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    existing_dir = create_profile("auto_under_the_volcano_malcolm_lowry", profiles_root=tmp_path)
    profile_path = existing_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config.update({
        "generated_profile": True,
        "source_name": "Under the Volcano - Malcolm Lowry.epub",
        "profile_goal": "audiobook",
    })
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (existing_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Pacific",
                "target": "Pacífico",
                "type": "proper_noun",
                "status": "approved",
                "confidence": 0.99,
                "translation_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    result = asyncio.run(prepare_book_profile_from_text(
        "Under the Volcano readable source text. " * 20,
        source_name="Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
        profile_goal="audiobook",
        auto_approve_safe_terms=False,
    ))

    assert result.profile_id == "auto_under_the_volcano_malcolm_lowry"
    assert result.coverage_mode == "reused"
    assert result.approved_entries == 1
    assert "skipped regeneration" in result.warnings[0]
    assert not any(path.name.startswith("auto_under_the_volcano_malcolm_lowry_") for path in tmp_path.iterdir())


def test_prepare_book_profile_continues_when_discovery_chunk_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))

    class FlakyDiscoveryProvider:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, timeout=0, system_prompt=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary DeepSeek Flash timeout")

            class Response:
                content = (
                    '<GLOSSARY_DISCOVERY_JSON>{"suggestions": ['
                    '{"source": "Obsidian Compass", "suggested_target": "brújula de obsidiana", '
                    '"type": "technical_term", "confidence": 0.93, '
                    '"rationale": "Recurring concept in the book."}'
                    "]}</GLOSSARY_DISCOVERY_JSON>"
                )
            return Response()

    class NoopReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = '<PROFILE_TERM_REVIEW_JSON>{"terms": []}</PROFILE_TERM_REVIEW_JSON>'
            return Response()

    text = (
        "The investigator compares recurring behavior with case notes. "
        "Witness accounts describe a pattern across several years. "
    ) * 90 + (
        "Near the end, the field notes mention the Obsidian Compass as a singular clue. "
    )
    progress_events = []
    provider = FlakyDiscoveryProvider()

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Flaky Discovery.epub",
        language="English",
        target_locale="es-MX",
        llm_provider=provider,
        term_review_provider=NoopReviewProvider(),
        provider_name="deepseek",
        model="deepseek-v4-flash",
        term_review_model="deepseek-v4-pro",
        max_llm_chunks=2,
        llm_chunk_chars=1200,
        llm_full_coverage=True,
        progress_callback=progress_events.append,
    ))

    assert result.llm_chunks == 1
    assert any("Discovery chunk 1/" in warning for warning in result.warnings)
    assert any("fallo durante lectura LLM" in event.get("message", "") for event in progress_events)
    assert progress_events[-1]["stage"] == "completed"


def test_profile_prep_does_not_approve_translatable_english_terms_as_preserve(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    text = (
        "Scenario one uses Probability and Bayes' Theorem. "
        "The Approach explains the Formula. "
        "Scenario two uses Probability and the Formula again. "
    ) * 40

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Bayes Theorem Examples.epub",
        language="English",
        target_locale="es-MX",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    approved_same = {
        entry.source
        for entry in profile.approved_entries
        if entry.target.casefold() == entry.source.casefold()
    }
    assert "Probability" not in approved_same
    assert "Scenario" not in approved_same
    assert "Approach" not in approved_same
    assert "Bayes' Theorem" not in approved_same
    pending_sources = {entry.source for entry in profile.pending_entries}
    assert "Probability" in pending_sources
    assert result.term_review["reviewed_terms"] > 0


def test_profile_term_reviewer_can_approve_direct_technical_translation(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))

    class FakeReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    "{\"source\": \"Bayes' Theorem\", \"target\": \"teorema de Bayes\", "
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", "translation_policy": "translate_exact", '
                    '"confidence": 0.94, "rationale": "Direct technical translation in es-MX."},'
                    '{"source": "Probability", "target": "probabilidad", '
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", "translation_policy": "translate_exact", '
                    '"confidence": 0.93, "rationale": "Direct technical translation."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    text = (
        "CHAPTER I\n\n"
        "Bayes' Theorem uses Probability. "
        "Bayes' Theorem updates Probability in each Scenario. "
    ) * 40

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Bayes Theorem Examples.epub",
        language="English",
        target_locale="es-MX",
        llm_provider=FakeReviewProvider(),
        provider_name="deepseek",
        model="deepseek-v4-flash",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    terms = {entry.source: entry for entry in profile.approved_entries}
    assert terms["Bayes' Theorem"].target == "teorema de Bayes"
    assert terms["Bayes' Theorem"].translation_policy == "translate_exact"
    assert terms["Probability"].target == "probabilidad"
    block = build_profile_glossary_block(
        "Bayes' Theorem uses Probability.",
        {"editorial_mode": "book_profile", "profile_id": result.profile_id},
    )
    assert "Bayes' Theorem -> teorema de Bayes" in block
    assert "Probability -> probabilidad" in block
    artifacts = profile.editorial_artifacts
    assert artifacts["version"] == "editorial-prep-v3"
    assert any(
        item.get("source") == "Probability" and item.get("target") == "probabilidad"
        for item in artifacts.get("translatable_terms", [])
    )
    assert any(
        item.get("source") == "Bayes' Theorem" and item.get("target") == "teorema de Bayes"
        for item in artifacts.get("translatable_terms", [])
    )
    assert any(
        str(item.get("title") or "").startswith("CHAPTER I")
        for item in artifacts.get("chapters", [])
    )


def test_profile_term_reviewer_demotes_same_target_translatable_terms(monkeypatch):
    class FakeReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Residual Stream", "target": "Residual Stream", '
                    '"review_status": "preserve_exact", "entry_type": "proper_noun", '
                    '"injection_policy": "preserve", "translation_policy": "preserve_exact", '
                    '"confidence": 0.97, "rationale": "Incorrectly treated as a proper name."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    reviewed, summary = asyncio.run(review_profile_terms(
        [{
            "source": "Residual Stream",
            "category": "technical",
            "occurrences": 8,
            "confidence": 0.91,
            "contexts": ["The residual stream carries information across layers."],
        }],
        profile_id="auto_ai_book",
        language="English",
        target_locale="es-MX",
        llm_provider=FakeReviewProvider(),
        model="deepseek-v4-pro",
    ))

    assert summary.demoted_entries == 1
    item = reviewed[0]
    assert item["review_status"] == "pending_review"
    assert item["review_target"] == ""
    assert "source_equals_target_translatable" in item["review_demoted_reason"]


def test_profile_term_reviewer_does_not_treat_long_german_nouns_as_names():
    reviewed, summary = asyncio.run(review_profile_terms(
        [
            {
                "source": "Wirklichkeit",
                "category": "proper_noun",
                "occurrences": 14,
                "confidence": 0.94,
                "contexts": ["Die Wirklichkeit dieser Geschichte blieb verborgen."],
            },
            {
                "source": "Zerstörung",
                "category": "proper_noun",
                "occurrences": 7,
                "confidence": 0.91,
                "contexts": ["Die Zerstörung der Stadt dauerte viele Jahre."],
            },
            {
                "source": "Die Geschichte",
                "category": "proper_noun",
                "occurrences": 6,
                "confidence": 0.91,
                "contexts": ["Die Geschichte jedes Menschen verlief anders."],
            },
        ],
        profile_id="auto_german_book",
        source_language="German",
        language="Spanish",
        target_locale="es-MX",
    ))

    assert summary.auto_approved_preserve == 0
    assert {item["review_status"] for item in reviewed} == {"pending_review"}
    assert all(reviewed_candidate_to_approved_entry(item, profile_id="auto_german_book") is None for item in reviewed)


def test_profile_term_reviewer_requires_canonical_name_for_single_german_entity():
    class FakeReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Wirklichkeit", "target": "Wirklichkeit", '
                    '"review_status": "preserve_exact", "entry_type": "proper_noun", '
                    '"confidence": 0.98},'
                    '{"source": "Sebald", "target": "Sebald", '
                    '"review_status": "canonical_name", "entry_type": "proper_noun", '
                    '"confidence": 0.98}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    reviewed, _summary = asyncio.run(review_profile_terms(
        [
            {"source": "Wirklichkeit", "category": "proper_noun", "occurrences": 8, "confidence": 0.9},
            {"source": "Sebald", "category": "proper_noun", "occurrences": 8, "confidence": 0.9},
        ],
        profile_id="auto_german_book",
        source_language="German",
        language="Spanish",
        target_locale="es-MX",
        llm_provider=FakeReviewProvider(),
        model="deepseek-v4-pro",
    ))

    by_source = {item["source"]: item for item in reviewed}
    assert by_source["Wirklichkeit"]["review_status"] == "pending_review"
    assert by_source["Sebald"]["review_status"] == "canonical_name"
    approved = reviewed_candidate_to_approved_entry(
        by_source["Sebald"],
        profile_id="auto_german_book",
    )
    assert approved is not None
    assert approved["target"] == "Sebald"


def test_editorial_map_removes_unapproved_local_entity_guesses():
    editorial_map = {
        "entities": [
            {"name": "Wirklichkeit", "source_kind": "local"},
            {"name": "Sebald", "source_kind": "local"},
        ],
        "characters_entities": [
            {"name": "Wirklichkeit", "source_kind": "local"},
            {"name": "Sebald", "source_kind": "local"},
        ],
        "preserve_terms": [
            {"source": "Wirklichkeit", "source_kind": "local"},
            {"source": "Sebald", "source_kind": "local"},
        ],
    }
    reviewed = [
        {"source": "Wirklichkeit", "review_status": "pending_review", "review_confidence": 0.86},
        {"source": "Sebald", "review_status": "canonical_name", "review_confidence": 0.98},
    ]
    approved = [{
        "source": "Sebald",
        "target": "Sebald",
        "type": "canonical_proper_noun",
        "translation_policy": "canonical_name",
        "confidence": 0.98,
    }]

    enriched = enrich_editorial_map_with_reviewed_terms(
        editorial_map,
        reviewed_candidates=reviewed,
        approved_entries=approved,
        profile_id="auto_german_book",
    )

    assert all(item.get("name") != "Wirklichkeit" for item in enriched["entities"])
    assert all(item.get("source") != "Wirklichkeit" for item in enriched["preserve_terms"])
    assert any(item.get("name") == "Sebald" for item in enriched["canonical_names"])


def test_flash_editorial_map_cannot_create_binding_preserve_decisions():
    merged = merge_llm_editorial_map(
        {"profile_id": "auto_book"},
        {
            "entities": [{"name": "Sebald", "type": "person"}],
            "preserve_terms": [{"source": "Wirklichkeit"}],
            "canonical_names": [{"name": "Wirklichkeit", "canonical": "Wirklichkeit"}],
            "do_not_translate": [{"source": "Wirklichkeit"}],
        },
        profile_id="auto_book",
        chunk_index=1,
    )

    assert any(item.get("name") == "Sebald" for item in merged["entities"])
    assert not merged["preserve_terms"]
    assert any(item.get("name") == "Wirklichkeit" for item in merged["canonical_names"])
    assert not merged["do_not_translate"]


def test_prepare_profile_stores_distinct_source_and_target_languages(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    result = asyncio.run(prepare_book_profile_from_text(
        "Die Wirklichkeit dieser Geschichte blieb lange verborgen. " * 30,
        source_name="German Book.epub",
        source_language="German",
        language="Spanish",
        target_locale="es-MX",
        profile_goal="faithful_translation",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    assert profile.raw_config["source_language"] == "German"
    assert profile.raw_config["target_language"] == "Spanish"


def test_profile_discovery_repairs_invalid_json_once(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))

    class InvalidThenRepairProvider:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, timeout=0, system_prompt=None):
            self.calls += 1

            class Response:
                content = ""

            response = Response()
            if self.calls == 1:
                response.content = '<GLOSSARY_DISCOVERY_JSON>{"suggestions":[{"source":"Obsidian Zeichen"'
            elif "Obsidian Zeichen" in prompt:
                response.content = (
                    '<GLOSSARY_DISCOVERY_JSON>{"suggestions":['
                    '{"source":"Obsidian Zeichen","suggested_target":"signo de obsidiana",'
                    '"type":"concept","confidence":0.95,"rationale":"Recurring concept."}'
                    ']}</GLOSSARY_DISCOVERY_JSON>'
                )
            else:
                response.content = (
                    '<GLOSSARY_DISCOVERY_JSON>{"suggestions":[],"editorial_map":'
                    '{"risks":[{"reason":"Check recurring conceptual language."}]}}'
                    '</GLOSSARY_DISCOVERY_JSON>'
                )
            return response

    class NoopReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = '<PROFILE_TERM_REVIEW_JSON>{"terms":[]}</PROFILE_TERM_REVIEW_JSON>'
            return Response()

    provider = InvalidThenRepairProvider()
    events = []
    result = asyncio.run(prepare_book_profile_from_text(
        ("Das Obsidian Zeichen erschien zuerst. " + ("Die Wirklichkeit dieser Geschichte blieb verborgen. " * 100)),
        source_name="Repair Discovery.epub",
        source_language="German",
        language="Spanish",
        target_locale="es-MX",
        profile_goal="faithful_translation",
        llm_provider=provider,
        term_review_provider=NoopReviewProvider(),
        provider_name="deepseek",
        model="deepseek-v4-flash",
        max_llm_chunks=1,
        llm_chunk_chars=3500,
        progress_callback=events.append,
    ))

    assert provider.calls == 3
    assert result.llm_suggestions == 1
    assert any(event.get("json_repaired") and event.get("split_retry") for event in events)


def test_profile_term_reviewer_demotes_titlecase_technical_phrases(monkeypatch):
    class FakeReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Attention Mechanism", "target": "Attention Mechanism", '
                    '"review_status": "preserve_exact", "entry_type": "proper_noun", '
                    '"injection_policy": "preserve", "translation_policy": "preserve_exact", '
                    '"confidence": 0.96, "rationale": "Incorrectly treated as a name."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    reviewed, summary = asyncio.run(review_profile_terms(
        [{
            "source": "Attention Mechanism",
            "category": "proper_noun",
            "occurrences": 11,
            "confidence": 0.91,
            "contexts": ["The attention mechanism aligns tokens across the sequence."],
        }],
        profile_id="auto_ai_book",
        language="English",
        target_locale="es-MX",
        llm_provider=FakeReviewProvider(),
        model="deepseek-v4-pro",
    ))

    assert summary.demoted_entries == 1
    item = reviewed[0]
    assert item["review_status"] == "pending_review"
    assert item["review_target"] == ""
    assert "source_equals_target_translatable" in item["review_demoted_reason"]


def test_profile_term_reviewer_uses_goal_specific_review_budget(monkeypatch):
    class CountingReviewProvider:
        def __init__(self):
            self.prompts = []

        async def generate(self, prompt, timeout=0, system_prompt=None):
            self.prompts.append((system_prompt or "", prompt))

            class Response:
                content = '<PROFILE_TERM_REVIEW_JSON>{"terms": []}</PROFILE_TERM_REVIEW_JSON>'

            return Response()

    candidates = [
        {
            "source": f"Technical Concept {index}",
            "category": "technical",
            "occurrences": 3 + index,
            "confidence": 0.81,
            "contexts": [f"Technical Concept {index} appears in the method."],
        }
        for index in range(700)
    ]
    provider = CountingReviewProvider()
    events = []

    reviewed, summary = asyncio.run(review_profile_terms(
        candidates,
        profile_id="academic_ai_book",
        language="English",
        target_locale="es-MX",
        profile_goal="academic_translation",
        llm_provider=provider,
        model="deepseek-v4-pro",
        text_chars=900_000,
        progress_callback=events.append,
    ))

    academic_rules = resolve_profile_goal("academic_translation")
    expected_reviewed = academic_rules.review_terms_limit(
        text_chars=900_000,
        candidate_count=len(candidates),
    )
    assert len(reviewed) == len(candidates)
    assert summary.llm_calls == len(provider.prompts)
    assert any(event.get("terms_total") == expected_reviewed for event in events)
    assert "Business goal: Académico/técnico" in provider.prompts[0][0]


def test_profile_term_reviewer_reports_batch_progress_and_timeout():
    class FakeReviewProvider:
        def __init__(self):
            self.timeouts = []

        async def generate(self, prompt, timeout=0, system_prompt=None):
            self.timeouts.append(timeout)

            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Residual Stream", "target": "flujo residual", '
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", "translation_policy": "translate_exact", '
                    '"confidence": 0.97, "rationale": "Direct technical term."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    provider = FakeReviewProvider()
    events = []
    reviewed, summary = asyncio.run(review_profile_terms(
        [{
            "source": "Residual Stream",
            "category": "technical",
            "occurrences": 8,
            "confidence": 0.91,
            "contexts": ["The residual stream carries information across layers."],
        }],
        profile_id="auto_ai_book",
        language="English",
        target_locale="es-MX",
        llm_provider=provider,
        model="deepseek-v4-pro",
        progress_callback=events.append,
        request_timeout=37,
    ))

    assert provider.timeouts == [37]
    assert summary.llm_calls == 1
    assert reviewed[0]["review_status"] == "translate_exact"
    assert reviewed[0]["review_target"] == "flujo residual"
    assert any(event.get("term_batch_index") == 1 for event in events)
    assert any(event.get("term_batch_total") == 1 for event in events)
    assert any(event.get("chunk_index") == 1 and event.get("chunk_total") == 1 for event in events)


def test_profile_term_reviewer_splits_a_timed_out_batch_without_recursion():
    class SplitRecoveryProvider:
        def __init__(self):
            self.batch_sizes = []

        async def generate(self, prompt, timeout=0, system_prompt=None):
            batch_size = prompt.count('"source"')
            self.batch_sizes.append(batch_size)
            if batch_size > 16:
                raise asyncio.TimeoutError()
            source = re.search(r'"source":\s*"([^"]+)"', prompt).group(1)

            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    f'{{"source": "{source}", "target": "termino revisado", '
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", '
                    '"translation_policy": "translate_exact", "confidence": 0.95, '
                    '"rationale": "Direct technical term."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )

            return Response()

    provider = SplitRecoveryProvider()
    events = []
    candidates = [
        {
            "source": f"Technical Concept {index}",
            "category": "technical",
            "occurrences": index + 2,
            "confidence": 0.9,
            "contexts": [f"Technical Concept {index} appears in this chapter."],
        }
        for index in range(24)
    ]

    reviewed, summary = asyncio.run(review_profile_terms(
        candidates,
        profile_id="auto_resilient_profile",
        language="English",
        target_locale="es-MX",
        llm_provider=provider,
        model="deepseek-v4-pro",
        progress_callback=events.append,
        request_timeout=37,
    ))

    assert len(reviewed) == 24
    assert provider.batch_sizes == [24, 12, 12]
    assert summary.llm_calls == 3
    assert any(event.get("review_recovery") == "split" for event in events)


def test_profile_term_reviewer_split_recovery_shares_one_total_deadline():
    class HangingProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            await asyncio.sleep(10)

    candidates = [
        {
            "source": f"Technical Concept {index}",
            "category": "technical",
            "occurrences": index + 2,
            "confidence": 0.9,
        }
        for index in range(24)
    ]

    started = time.monotonic()
    reviewed, summary = asyncio.run(review_profile_terms(
        candidates,
        profile_id="auto_bounded_profile",
        language="English",
        target_locale="es-MX",
        llm_provider=HangingProvider(),
        model="deepseek-v4-pro",
        request_timeout=1,
    ))
    elapsed = time.monotonic() - started

    assert elapsed < 1.5
    assert len(reviewed) == 24
    assert summary.llm_calls == 3
    assert {item["review_status"] for item in reviewed} == {"pending_review"}


def test_profile_term_reviewer_keeps_contextual_target_guidance():
    class FakeReviewProvider:
        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Source-Mind", "target": "Mente de Origen", '
                    '"review_status": "translate_contextual", "entry_type": "concept", '
                    '"injection_policy": "contextual", '
                    '"translation_policy": "translate_contextual", '
                    '"confidence": 0.93, "rationale": "Recurring coined concept."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    reviewed, _summary = asyncio.run(review_profile_terms(
        [{
            "source": "Source-Mind",
            "category": "concept",
            "occurrences": 12,
            "confidence": 0.9,
            "contexts": ["The Source-Mind spoke through the Link."],
        }],
        profile_id="auto_literary_book",
        source_language="English",
        language="Spanish",
        target_locale="es-MX",
        llm_provider=FakeReviewProvider(),
        model="deepseek-v4-pro",
    ))

    assert reviewed[0]["review_status"] == "translate_contextual"
    assert reviewed[0]["review_target"] == "Mente de Origen"


def test_profile_exact_translation_corrections_only_apply_approved_exact_entries(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("exact_terms", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Source-Mind",
                    "target": "Mente de Origen",
                    "type": "concept",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
                {
                    "source": "Link-Talent",
                    "target": "Talento de Enlace",
                    "type": "concept",
                    "status": "pending",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    corrected, count = apply_profile_exact_translation_corrections(
        "The Source-Mind activated the Link-Talent.",
        {"editorial_mode": "book_profile", "profile_id": "exact_terms"},
    )

    assert corrected == "The Mente de Origen activated the Link-Talent."
    assert count == 1


def test_profile_exact_translation_corrections_prefilter_unmatched_pairs(monkeypatch):
    from src.core.book_profiles import rendering

    original = rendering._replace_profile_term_counted
    visited: list[str] = []

    def tracked(text, source, target, *, protected_terms=()):
        visited.append(source)
        return original(
            text,
            source,
            target,
            protected_terms=protected_terms,
        )

    monkeypatch.setattr(rendering, "_replace_profile_term_counted", tracked)
    pairs = tuple(
        [(f"Absent Term {index}", f"Término ausente {index}") for index in range(500)]
        + [("Machine Learning", "aprendizaje automático")]
    )

    corrected, count = rendering.apply_profile_exact_translation_corrections(
        "Machine Learning",
        None,
        exact_pairs=pairs,
    )

    assert corrected == "aprendizaje automático"
    assert count == 1
    assert visited == ["Machine Learning"]


def test_short_translation_rule_does_not_corrupt_protected_long_name(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("overlapping_name_terms", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Terry Southern",
                    "target": "Terry Southern",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Southern",
                    "target": "sur",
                    "type": "term",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
            ],
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "overlapping_name_terms",
    }

    name_only_block = build_profile_glossary_block(
        "Terry Southern wrote the screenplay.",
        options,
    )
    mixed_block = build_profile_glossary_block(
        "Terry Southern addressed a Southern audience.",
        options,
    )
    surname_block = build_profile_glossary_block(
        "Southern was wry about working in the movies.",
        options,
    )
    corrected, count = apply_profile_exact_translation_corrections(
        "Terry Southern addressed a Southern audience.",
        options,
    )
    from src.core.book_profiles.rendering import profile_exact_translation_pairs
    from src.core.fidelity_supervisor import assess_fidelity

    exact_pairs = profile_exact_translation_pairs(options)
    decision = assess_fidelity(
        "Terry Southern addressed a Southern audience after the screening.",
        "Terry Southern se dirigió a un público del sur después de la proyección.",
        chunk_index=1,
        phase="translation",
        source_language="English",
        target_language="Spanish",
        prompt_options=options,
    )

    assert "Terry Southern -> keep exact spelling" in name_only_block
    assert "Southern -> sur" not in name_only_block
    assert "Southern -> sur" not in mixed_block
    assert "Southern -> prefer Southern when context fits" in surname_block
    assert "Southern was wry" not in surname_block
    assert ("Southern", "sur") not in exact_pairs
    assert corrected == "Terry Southern addressed a Southern audience."
    assert count == 0
    assert "source_language_residual" not in {
        issue.code for issue in decision.rejections
    }


def test_exact_common_terms_do_not_corrupt_unlisted_compound_institution(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("compound_institution_terms", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "New York University",
                    "target": "New York University",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "University",
                    "target": "Universidad",
                    "type": "term",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
                {
                    "source": "College",
                    "target": "Universidad",
                    "type": "term",
                    "status": "approved",
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
            ],
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "compound_institution_terms",
    }
    source = (
        "Researchers at Peking University and University College London "
        "published the findings."
    )
    candidate = (
        "Investigadores de la Universidad de Pekín y de University College "
        "London publicaron los hallazgos."
    )

    block = build_profile_glossary_block(source, options)
    corrected = apply_profile_glossary_corrections(
        candidate,
        options,
        source_text=source,
    )
    exact_name, exact_name_count = apply_profile_exact_translation_corrections(
        "University College London",
        options,
    )
    exact_standalone, exact_standalone_count = apply_profile_exact_translation_corrections(
        "The College expanded.",
        options,
    )

    assert "University -> prefer University when context fits" not in block
    assert "College -> Universidad [term]" not in block
    assert (
        "College -> prefer Universidad in ordinary prose; when this term is "
        "part of a longer proper name"
    ) in block
    assert corrected == candidate
    assert exact_name == "University College London"
    assert exact_name_count == 0
    assert exact_standalone == "The Universidad expanded."
    assert exact_standalone_count == 1


def test_profile_term_reviewer_supports_providers_without_timeout_argument():
    class NoTimeoutReviewProvider:
        async def generate(self, prompt, system_prompt=None):
            class Response:
                content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    '{"source": "Bayes Theorem", "target": "teorema de Bayes", '
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", "translation_policy": "translate_exact", '
                    '"confidence": 0.95, "rationale": "Direct term."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            return Response()

    reviewed, summary = asyncio.run(review_profile_terms(
        [{
            "source": "Bayes Theorem",
            "category": "technical",
            "occurrences": 5,
            "confidence": 0.9,
        }],
        profile_id="auto_math_book",
        language="English",
        target_locale="es-MX",
        llm_provider=NoTimeoutReviewProvider(),
        model="deepseek-v4-pro",
        request_timeout=37,
    ))

    assert summary.llm_calls == 1
    assert reviewed[0]["review_status"] == "translate_exact"
    assert reviewed[0]["review_target"] == "teorema de Bayes"


def test_editorial_map_v3_preserves_llm_decision_metadata():
    data = merge_llm_editorial_map(
        {
            "profile_id": "auto_ai_book",
            "version": "editorial-prep-v3",
        },
        {
            "translatable_terms": [{
                "source": "residual stream",
                "target": "flujo residual",
                "type": "technical_term",
                "policy": "translate_exact",
                "confidence": 0.92,
                "reason": "Direct technical term.",
            }],
            "canonical_names": [{
                "name": "Montezuma",
                "canonical": "Moctezuma",
                "status": "pending",
                "confidence": 0.88,
            }],
            "blockers": [{
                "source": "Residual Stream",
                "code": "source_equals_target_translatable",
                "severity": "medium",
                "risks": ["Do not preserve this English technical term in Spanish."],
            }],
        },
        profile_id="auto_ai_book",
        chunk_index=3,
    )

    term = next(item for item in data["translatable_terms"] if item["source"] == "residual stream")
    assert term["target"] == "flujo residual"
    assert term["policy"] == "translate_exact"
    canonical = next(item for item in data["canonical_names"] if item["name"] == "Montezuma")
    assert canonical["canonical"] == "Moctezuma"
    blocker = next(item for item in data["blockers"] if item["source"] == "Residual Stream")
    assert blocker["risks"] == ["Do not preserve this English technical term in Spanish."]
    brief = render_editorial_brief(data)
    assert "flujo residual" in brief
    assert "Moctezuma" in brief


def test_prepare_book_profile_filters_sentence_initial_noise(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    text = (
        "Luego Motecuhzoma habló con los mensajeros. "
        "Luego Motecuhzoma volvió al palacio. "
    ) * 30

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Vision.txt",
        language="Spanish",
        target_locale="es-MX",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    sources = {entry.source for entry in profile.glossary_entries}
    assert "Motecuhzoma" in sources
    approved_sources = {entry.source for entry in profile.approved_entries}
    assert "Motecuhzoma" not in approved_sources
    do_not_sources = {
        item.get("source")
        for item in (profile.editorial_artifacts.get("do_not_translate") or [])
        if isinstance(item, dict)
    }
    assert "Motecuhzoma" not in do_not_sources
    assert "Luego" not in sources
    assert "Luego Motecuhzoma" not in sources


def test_generated_profile_glossary_migration_demotes_unsafe_preserve_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_bayes_sample", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["generated_profile"] = True
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Probability",
                    "target": "Probability",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                },
                {
                    "source": "UNAM",
                    "target": "UNAM",
                    "type": "acronym",
                    "status": "approved",
                    "confidence": 0.98,
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = migrate_generated_profile_glossaries(profiles_root=tmp_path)

    assert report["entries_demoted"] == 1
    profile = load_book_profile("auto_bayes_sample", profiles_root=tmp_path)
    assert any(entry.source == "UNAM" for entry in profile.approved_entries)
    assert not any(entry.source == "Probability" for entry in profile.approved_entries)
    assert any(entry.source == "Probability" for entry in profile.pending_entries)


def test_generated_profile_glossary_migration_demotes_literary_noise(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_under_sample", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["generated_profile"] = True
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "American",
                    "target": "American",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Mexican These",
                    "target": "Mexican These",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Suddenly",
                    "target": "Suddenly",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Suddenly the Consul",
                    "target": "Suddenly the Consul",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "November",
                    "target": "November",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Yvonne and Hugh",
                    "target": "Yvonne and Hugh",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Oh Hugh",
                    "target": "Oh Hugh",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Yvonne's",
                    "target": "Yvonne's",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "Yvonne Griffaton",
                    "target": "Yvonne Griffaton",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
                {
                    "source": "UNAM",
                    "target": "UNAM",
                    "type": "acronym",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "preserve_exact",
                    "injection_policy": "preserve",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    report = migrate_generated_profile_glossaries(profiles_root=tmp_path)

    assert report["entries_demoted"] == 8
    profile = load_book_profile("auto_under_sample", profiles_root=tmp_path)
    approved = {entry.source for entry in profile.approved_entries}
    pending = {entry.source for entry in profile.pending_entries}
    assert "UNAM" in approved
    assert "Yvonne Griffaton" in approved
    assert "American" not in approved
    assert "Mexican These" not in approved
    assert "Suddenly" not in approved
    assert "Suddenly the Consul" not in approved
    assert "November" not in approved
    assert "Yvonne and Hugh" not in approved
    assert "Oh Hugh" not in approved
    assert "Yvonne's" not in approved
    assert {
        "American",
        "Mexican These",
        "Suddenly",
        "Suddenly the Consul",
        "November",
        "Yvonne and Hugh",
        "Oh Hugh",
        "Yvonne's",
    } <= pending


def test_suspicious_preserve_entry_flags_source_equals_target_demonyms():
    assert suspicious_preserve_entry({
        "source": "American",
        "target": "American",
        "type": "proper_noun",
    })
    assert suspicious_preserve_entry({
        "source": "Consul The Americano No He",
        "target": "Consul The Americano No He",
        "type": "proper_noun",
    })
    assert suspicious_preserve_entry({
        "source": "Suddenly",
        "target": "Suddenly",
        "type": "proper_noun",
    })
    assert suspicious_preserve_entry({
        "source": "Suddenly the Consul",
        "target": "Suddenly the Consul",
        "type": "proper_noun",
    })
    assert suspicious_preserve_entry({
        "source": "November",
        "target": "November",
        "type": "proper_noun",
    })
    assert not suspicious_preserve_entry({
        "source": "UNAM",
        "target": "UNAM",
        "type": "acronym",
    })


def test_deterministic_term_review_sends_plain_title_case_entities_to_contextual_review():
    reviewed, summary = asyncio.run(review_profile_terms(
        [
            {
                "source": "Daniel Colter",
                "category": "character",
                "occurrences": 12,
                "confidence": 0.98,
            },
            {
                "source": "Ch*Tril",
                "category": "character",
                "occurrences": 24,
                "confidence": 0.98,
            },
        ],
        profile_id="auto_review_safety",
        source_language="English",
        language="Spanish",
        target_locale="es-MX",
        llm_provider=None,
    ))
    by_source = {item["source"]: item for item in reviewed}

    assert by_source["Daniel Colter"]["review_status"] == "pending_review"
    assert by_source["Daniel Colter"]["injection_policy"] == "contextual"
    assert by_source["Ch*Tril"]["review_status"] == "preserve_exact"
    assert summary.auto_approved_preserve == 1
    assert summary.pending_review == 1


def test_weak_generated_preserve_entry_requires_contextual_evidence():
    generic = "Looks like a recurring named entity for this book profile."
    assert weakly_supported_generated_preserve_entry({
        "source": "West Coast",
        "target": "West Coast",
        "type": "proper_noun",
        "translation_policy": "preserve_exact",
        "reviewed_by": "deterministic_profile_term_review",
        "rationale": generic,
    })
    assert not weakly_supported_generated_preserve_entry({
        "source": "Daniel Colter",
        "target": "Daniel Colter",
        "type": "proper_noun",
        "translation_policy": "preserve_exact",
        "reviewed_by": "llm_profile_term_review:deepseek-v4-pro",
        "rationale": "Full name of the recurring protagonist.",
    })
    assert not weakly_supported_generated_preserve_entry({
        "source": "Ch*Tril",
        "target": "Ch*Tril",
        "type": "proper_noun",
        "translation_policy": "preserve_exact",
        "reviewed_by": "deterministic_profile_term_review",
        "rationale": generic,
    })


def test_profile_canonical_glossary_correction_is_profile_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("historia_mx", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config.update({
        "target_locale": "es-MX",
        "glossary_files": {
            "terms": "glossary/terms.yml",
        },
    })
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Montezuma",
                "target": "Moctezuma",
                "type": "canonical_proper_noun",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": True,
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    text = "El gran Montezuma habló con Cortés. MONTEZUMA aparece en un título."
    corrected = apply_profile_glossary_corrections(
        text,
        {"editorial_mode": "book_profile", "profile_id": "historia_mx"},
        source_text="Montezuma habló.",
    )

    assert "El gran Moctezuma habló" in corrected
    assert "MOCTEZUMA aparece" in corrected
    unchanged = apply_profile_glossary_corrections(text, {}, source_text="Montezuma habló.")
    assert unchanged == text


def test_profile_canonical_correction_does_not_expand_existing_target(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("film_titles", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Atalante",
                "target": "L'Atalante",
                "type": "canonical_proper_noun",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": True,
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    corrected = apply_profile_glossary_corrections(
        "L’Atalante se estrenó después de Atalante.",
        {"editorial_mode": "book_profile", "profile_id": "film_titles"},
        source_text="Atalante appears twice.",
    )

    assert corrected == "L’Atalante se estrenó después de L'Atalante."
    assert "L’L'Atalante" not in corrected


def test_profile_exact_translation_is_applied_before_gate_when_source_contains_term(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("film_title_translation", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "The Grand Picture",
                "target": "La gran película",
                "type": "title",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": False,
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "film_title_translation",
    }

    corrected = apply_profile_glossary_corrections(
        "La escena de The Grand Picture termina aquí.",
        options,
        source_text="The scene from The Grand Picture ends here.",
    )
    unrelated = apply_profile_glossary_corrections(
        "La escena de The Grand Picture termina aquí.",
        options,
        source_text="A different film ends here.",
    )

    assert corrected == "La escena de La gran película termina aquí."
    assert unrelated == "La escena de The Grand Picture termina aquí."


def test_single_word_title_translation_is_contextual_without_mechanical_safe(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("ambiguous_title_translation", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Greed",
                "target": "Avaricia",
                "type": "title",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": False,
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "ambiguous_title_translation",
    }
    text = (
        "The Complete Greed of Erich von Stroheim aparece junto a la película Greed."
    )

    corrected = apply_profile_glossary_corrections(
        text,
        options,
        source_text=text,
    )
    exact_corrected, count = apply_profile_exact_translation_corrections(
        text,
        options,
    )
    block = build_profile_glossary_block(text, options)

    assert corrected == text
    assert exact_corrected == text
    assert count == 0
    assert "prefer Avaricia in ordinary prose" in block
    assert "inside a longer published title" in block


def test_exact_common_term_does_not_rewrite_nested_xhtml_work_title(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("nested_work_title", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Russian",
                    "target": "ruso",
                    "type": "term",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
                {
                    "source": "Biography",
                    "target": "Biografía",
                    "type": "term",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "translate_exact",
                    "injection_policy": "translate_exact",
                },
            ]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "nested_work_title",
    }
    source = (
        "See [id0]The Film Factory: Russian and Soviet Cinema in Documents[id1] "
        "and [id2]Luis Buñuel: A Critical Biography[id3]."
    )
    candidate = (
        "Véase [id0]The Film Factory: Russian and Soviet Cinema in Documents[id1] "
        "y [id2]Luis Buñuel: A Critical Biography[id3]."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        options,
        source_text=source,
    )
    block = build_profile_glossary_block(source, options)

    assert corrected == candidate
    assert "Russian -> prefer ruso in ordinary prose" in block
    assert "Biography -> prefer Biografía in ordinary prose" in block
    assert "Russian -> ruso [term]" not in block
    assert "Biography -> Biografía [term]" not in block
    assert block.count("inside a longer published title") == 2


def test_exact_profile_term_never_rewrites_url_or_placeholder_literal(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("literal_protection", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "non-profit",
                "target": "sin fines de lucro",
                "type": "technical_term",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": False,
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "literal_protection",
    }
    source = (
        "A non-profit published the report at "
        "[id8]example.org/news/non-profit-report[id9]."
    )
    candidate = (
        "Una non-profit publicó el informe en "
        "[id8]example.org/news/non-profit-report[id9]."
    )
    corrupted = (
        "Una non-profit publicó el informe en "
        "[id8]example.org/news/sin fines de lucro-report[id9]."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        options,
        source_text=source,
    )
    restored = apply_profile_glossary_corrections(
        corrupted,
        options,
        source_text=source,
    )
    exact_corrected, count = apply_profile_exact_translation_corrections(
        candidate,
        options,
    )

    assert corrected == (
        "Una sin fines de lucro publicó el informe en "
        "[id8]example.org/news/non-profit-report[id9]."
    )
    assert restored == corrected
    assert exact_corrected == corrected
    assert count == 1


def test_mixed_case_cited_title_protects_nested_profile_term(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("citation_title_protection", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "non-profit",
                "target": "sin fines de lucro",
                "type": "technical_term",
                "status": "approved",
                "confidence": 0.98,
                "mechanical_safe": False,
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    options = {
        "editorial_mode": "book_profile",
        "profile_id": "citation_title_protection",
    }
    source = (
        "Nina Haikara, “This U of T Alum Is Leading AI research at $1 Billion "
        "Non-profit Backed by Elon Musk,” U of T News, March 28, 2017, "
        "[id8]utoronto.ca/news/1-billion-non-profit-backed[id9]; "
        "[id10]doi.org/10.1145/3065386[id11]."
    )
    candidate = (
        "Nina Haikara, «This U of T Alum Is Leading AI research at $1 Billion "
        "Non-profit Backed by Elon Musk», U of T News, 28 de marzo de 2017, "
        "[id8]utoronto.ca/news/1-billion-non-profit-backed[id9]; "
        "[id10]doi.org/10.1145/3065386[id11]."
    )
    corrupted = candidate.replace(
        "$1 Billion Non-profit Backed",
        "$1 Billion sin fines de lucro Backed",
    ).replace(
        "1-billion-non-profit-backed",
        "1-billion-sin fines de lucro-backed",
    )

    corrected = apply_profile_glossary_corrections(
        corrupted,
        options,
        source_text=source,
    )
    block = build_profile_glossary_block(source, options)

    assert corrected == candidate
    assert "inside a longer published title" in block


def test_bibliographic_title_restoration_is_global_and_leaves_dialogue_alone():
    source = (
        "“Do Not Change My Answer,” she said. "
        "Alec Radford, “Language Models Are Unsupervised Multitask Learners,” "
        "preprint, OpenAI, February 14, 2019, "
        "cdn.openai.com/better-language-models/paper.pdf. "
        "Jared Kaplan, “Scaling Laws for Neural Language Models,” preprint, "
        "arXiv, January 23, 2020, doi.org/10.48550/arXiv.2001.08361."
    )
    candidate = (
        "«No cambies mi respuesta», dijo. "
        "Alec Radford, «Los Language Models son aprendices multitarea no "
        "supervisados», preprint, OpenAI, 14 de febrero de 2019, "
        "cdn.openai.com/better-language-models/paper.pdf. "
        "Jared Kaplan, «Leyes de escala para Language Models neuronal», "
        "preprint, arXiv, 23 de enero de 2020, "
        "doi.org/10.48550/arXiv.2001.08361."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        {},
        source_text=source,
    )

    assert "«No cambies mi respuesta»" in corrected
    assert (
        "«Language Models Are Unsupervised Multitask Learners»"
        in corrected
    )
    assert "«Scaling Laws for Neural Language Models»" in corrected
    assert "Los Language Models" not in corrected


def test_bibliographic_title_restoration_does_not_restore_social_post_body():
    source = (
        "Ilya Sutskever (@ilyasut), “it may be that today’s large neural "
        "networks are slightly conscious,” Twitter (now X), February 9, 2022, "
        "x.com/ilyasut/status/1491554478243258368. "
        "Nirit Weiss-Blatt, “What Ilya Sutskever Really Wants,” AI Panic, "
        "September 16, 2023, aipanic.news/p/what-ilya-sutskever-really-wants."
    )
    candidate = (
        "Ilya Sutskever (@ilyasut), «puede ser que las grandes redes neuronales "
        "actuales sean ligeramente conscientes», Twitter (ahora X), 9 de febrero "
        "de 2022, x.com/ilyasut/status/1491554478243258368. "
        "Nirit Weiss-Blatt, «Lo que Ilya Sutskever realmente quiere», AI Panic, "
        "16 de septiembre de 2023, aipanic.news/p/what-ilya-sutskever-really-wants."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        {},
        source_text=source,
    )

    assert "«puede ser que las grandes redes neuronales" in corrected
    assert "«What Ilya Sutskever Really Wants»" in corrected


def test_bibliographic_title_restoration_does_not_restore_sentence_case_social_post():
    source = (
        "Helen Toner (@hlntnr), "
        "“A statement from Helen Toner and Tasha McCauley:,” "
        "Twitter (now X), March 8, 2024, "
        "x.com/hlntnr/status/1766269137628590185."
    )
    candidate = (
        "Helen Toner (@hlntnr), "
        "«Una declaración de Helen Toner y Tasha McCauley:», "
        "Twitter (ahora X), 8 de marzo de 2024, "
        "x.com/hlntnr/status/1766269137628590185."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        {},
        source_text=source,
    )

    assert "«Una declaración de Helen Toner y Tasha McCauley:»" in corrected
    assert "A statement from Helen Toner" not in corrected


def test_bibliographic_title_restoration_detects_social_post_from_status_url():
    source = (
        "Barret Zoph (@barret_zoph), "
        "“I posted this note to OpenAI.,” September 25, 2024, "
        "x.com/barret_zoph/status/1839095143397515452. "
        "Dara Kerr, “How Memphis Became a Battleground,” NPR, "
        "September 11, 2024, npr.org/example."
    )
    candidate = (
        "Barret Zoph (@barret_zoph), "
        "«Publiqué esta nota para OpenAI», 25 de septiembre de 2024, "
        "x.com/barret_zoph/status/1839095143397515452. "
        "Dara Kerr, «Cómo Memphis se convirtió en un campo de batalla», NPR, "
        "11 de septiembre de 2024, npr.org/example."
    )

    corrected = apply_profile_glossary_corrections(
        candidate,
        {},
        source_text=source,
    )

    assert "«Publiqué esta nota para OpenAI»" in corrected
    assert "I posted this note to OpenAI" not in corrected
    assert "«How Memphis Became a Battleground»" in corrected


def test_exact_correction_does_not_rewrite_target_substring(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("nested_exact_terms", profiles_root=tmp_path)
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Mind",
                "target": "Source-Mind",
                "type": "concept",
                "status": "approved",
                "translation_policy": "translate_exact",
                "injection_policy": "translate_exact",
            }]
        }, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    corrected, count = apply_profile_exact_translation_corrections(
        "Source-Mind and Mind",
        {"editorial_mode": "book_profile", "profile_id": "nested_exact_terms"},
    )

    assert corrected == "Source-Mind and Source-Mind"
    assert count == 1


def test_prepare_book_profile_job_exposes_progress_and_autosaves(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    text = (
        "Terminus Prime appeared in every chapter. The narrator returned to "
        "Terminus Prime whenever the expedition lost its way. "
    ) * 20

    with app.test_client() as client:
        response = client.post("/api/book-profiles/prepare-jobs", json={
            "text": text,
            "source_name": "New Expedition.txt",
            "language": "English",
            "target_locale": "es-MX",
            "max_llm_chunks": 0,
        })
        assert response.status_code == 202
        prep_id = response.get_json()["prep_id"]

        data = {}
        for _ in range(50):
            status_response = client.get(f"/api/book-profiles/prepare-jobs/{prep_id}")
            assert status_response.status_code == 200
            data = status_response.get_json()
            if data["status"] == "completed":
                break
            time.sleep(0.05)

    assert data["status"] == "completed"
    assert data["progress"] == 100
    assert data["result"]["profile"]["profile_id"].startswith("auto_new_expedition")
    profile_id = data["result"]["profile"]["profile_id"]
    assert (tmp_path / profile_id / "profile.yml").exists()


def test_profile_prep_jobs_list_and_deduplicate_active_request(tmp_path, monkeypatch):
    from flask import Flask

    import src.api.blueprints.profile_routes as profile_routes

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    started = threading.Event()
    release = threading.Event()

    def hold_job(prep_id, _payload):
        profile_routes._update_profile_prep_job(
            prep_id,
            status="running",
            progress=10,
            current_stage="local_scan",
            message="Scanning.",
        )
        started.set()
        release.wait(timeout=3)

    monkeypatch.setattr(profile_routes, "_run_profile_prep_job", hold_job)
    with profile_routes._PROFILE_PREP_LOCK:
        profile_routes._PROFILE_PREP_JOBS.clear()

    app = Flask(__name__)
    app.register_blueprint(profile_routes.create_profile_blueprint())
    payload = {
        "text": ("Terminus Prime appears throughout the book. " * 20),
        "source_name": "Duplicate Profile.txt",
        "language": "English",
        "target_locale": "es-MX",
        "max_llm_chunks": 0,
    }

    try:
        with app.test_client() as client:
            first = client.post("/api/book-profiles/prepare-jobs", json=payload)
            assert first.status_code == 202
            assert started.wait(timeout=1)
            first_id = first.get_json()["prep_id"]

            duplicate = client.post("/api/book-profiles/prepare-jobs", json=payload)
            assert duplicate.status_code == 202
            assert duplicate.get_json()["prep_id"] == first_id
            assert duplicate.get_json()["recovered_existing"] is True

            listing = client.get(
                "/api/book-profiles/prepare-jobs?statuses=queued,running&limit=1"
            )
            assert listing.status_code == 200
            jobs = listing.get_json()["jobs"]
            assert len(jobs) == 1
            assert jobs[0]["prep_id"] == first_id
            assert "request_fingerprint" not in jobs[0]
    finally:
        release.set()
        with profile_routes._PROFILE_PREP_LOCK:
            profile_routes._PROFILE_PREP_JOBS.clear()


def test_profile_prep_route_uses_deepseek_pro_for_term_reviewer(tmp_path, monkeypatch):
    from flask import Flask

    import src.api.blueprints.profile_routes as profile_routes
    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    created_models = []

    class FakeProvider:
        def __init__(self, model):
            self.model = model

        async def generate(self, prompt, timeout=0, system_prompt=None):
            class Response:
                content = ""

            response = Response()
            if "PROFILE_TERM_REVIEW_JSON" in (system_prompt or ""):
                response.content = (
                    '<PROFILE_TERM_REVIEW_JSON>{"terms": ['
                    "{\"source\": \"Bayes' Theorem\", \"target\": \"teorema de Bayes\", "
                    '"review_status": "translate_exact", "entry_type": "technical_term", '
                    '"injection_policy": "translate_exact", "translation_policy": "translate_exact", '
                    '"confidence": 0.94, "rationale": "Direct technical translation."}'
                    "]}</PROFILE_TERM_REVIEW_JSON>"
                )
            elif "GLOSSARY_DISCOVERY_JSON" in (system_prompt or ""):
                response.content = '<GLOSSARY_DISCOVERY_JSON>{"suggestions": []}</GLOSSARY_DISCOVERY_JSON>'
            return response

        async def close(self):
            return None

    def fake_create_llm_provider(**kwargs):
        created_models.append(kwargs.get("model"))
        return FakeProvider(kwargs.get("model"))

    monkeypatch.setattr(profile_routes, "create_llm_provider", fake_create_llm_provider)
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    text = (
        "Bayes' Theorem uses Probability in each Scenario. "
        "Bayes' Theorem updates Probability with new evidence. "
    ) * 30

    with app.test_client() as client:
        response = client.post("/api/book-profiles/prepare", json={
            "text": text,
            "source_name": "Bayes Theorem Examples.epub",
            "language": "English",
            "target_locale": "es-MX",
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "max_llm_chunks": 1,
            "llm_chunk_chars": 2500,
        })

    assert response.status_code == 201
    profile_data = response.get_json()["profile"]
    assert created_models == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert profile_data["model"] == "deepseek-v4-flash"
    assert profile_data["review_model"] == "deepseek-v4-pro"
    profile = load_book_profile(profile_data["profile_id"], profiles_root=tmp_path)
    assert any(
        entry.source == "Bayes' Theorem" and entry.target == "teorema de Bayes"
        for entry in profile.approved_entries
    )


def test_distributed_discovery_chunks_are_bounded_and_spread():
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = distributed_discovery_chunks(text, chunk_chars=1200, max_chunks=5)

    assert 1 < len(chunks) <= 5
    assert all(len(chunk) <= 1400 for chunk in chunks)
    assert "word0" in chunks[0]
    assert "word1999" in chunks[-1]


def test_full_coverage_discovery_chunks_cover_complete_text():
    text = ". ".join(f"sentence {i} with useful material" for i in range(120)) + "."
    chunks = full_coverage_discovery_chunks(text, chunk_chars=350, max_chunks=20)
    clean = re.sub(r"\s+", " ", text).strip()

    assert len(chunks) > 1
    assert chunks[0].startswith("sentence 0")
    assert chunks[-1].endswith("useful material.")
    combined = re.sub(r"\s+", " ", " ".join(chunks)).strip()
    assert len(combined) >= int(len(clean) * 0.98)
    assert "sentence 60 with useful material" in combined


def test_profile_prep_limits_scale_for_long_books():
    from src.api.blueprints.profile_routes import _profile_prep_limits

    short_limits = _profile_prep_limits(
        "short text " * 80,
        profile_goal="faithful_translation",
        max_llm_chunks=None,
        llm_chunk_chars=None,
        max_local_terms=None,
    )
    long_limits = _profile_prep_limits(
        "long book text " * 70000,
        profile_goal="faithful_translation",
        max_llm_chunks=None,
        llm_chunk_chars=None,
        max_local_terms=None,
    )
    disabled_llm = _profile_prep_limits(
        "long book text " * 70000,
        profile_goal="faithful_translation",
        max_llm_chunks=0,
        llm_chunk_chars=None,
        max_local_terms=None,
    )

    assert long_limits["max_local_terms"] > short_limits["max_local_terms"]
    assert long_limits["max_llm_chunks"] > 8
    assert disabled_llm["max_llm_chunks"] == 0


def test_profile_prep_limits_differ_by_business_goal():
    from src.api.blueprints.profile_routes import _profile_prep_limits

    text = "conceptual material and named entities " * 30000
    faithful = _profile_prep_limits(
        text,
        profile_goal="faithful_translation",
        max_llm_chunks=None,
        llm_chunk_chars=None,
        max_local_terms=None,
    )
    academic = _profile_prep_limits(
        text,
        profile_goal="academic_translation",
        max_llm_chunks=None,
        llm_chunk_chars=None,
        max_local_terms=None,
    )
    audiobook = _profile_prep_limits(
        text,
        profile_goal="audiobook",
        max_llm_chunks=None,
        llm_chunk_chars=None,
        max_local_terms=None,
    )

    assert academic["max_local_terms"] > faithful["max_local_terms"]
    assert faithful["max_local_terms"] > audiobook["max_local_terms"]


def test_prepare_book_profile_records_business_goal_and_limits(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    text = (
        "This book discusses montage, sequence, captions, credits, and film language. "
        "A figure caption explains a recurring visual motif. "
    ) * 70

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Film for Listening.epub",
        language="English",
        target_locale="es-MX",
        transform_mode="audiobook",
        profile_goal="audiobook",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    assert result.profile_goal == "audiobook"
    assert result.profile_goal_label == "Audiolibro fiel"
    assert profile.raw_config["profile_goal"] == "audiobook"
    assert profile.raw_config["business_rules"]["goal"] == "audiobook"
    assert result.business_limits["approved_entries_limit"] > 0


def test_prepare_book_profile_direct_call_uses_goal_dynamic_local_terms(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    text = (
        "Bayes' Theorem, Markov chains, latent variables, and posterior inference "
        "appear throughout this academic source. "
    ) * 1200

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="Academic Source.txt",
        language="English",
        target_locale="es-MX",
        profile_goal="academic_translation",
        max_llm_chunks=0,
    ))

    expected = resolve_profile_goal("academic_translation").local_terms_limit(len(text))
    assert result.max_local_terms == expected
    assert result.max_local_terms > 160
    assert result.business_limits["max_local_terms"] == expected


def test_book_profile_list_exposes_glossary_management_flags(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("quijote_mx_contemporary")
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["name"] = "Don Quijote MX"
    config["target_locale"] = "es-MX"
    config["profile_goal"] = "modernization"
    profile_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    legacy_dir = create_profile("auto_legacy_profile")
    legacy_path = legacy_dir / "profile.yml"
    legacy_config = yaml.safe_load(legacy_path.read_text(encoding="utf-8")) or {}
    legacy_config["generated_profile"] = True
    legacy_config["source_name"] = "Legacy Book.epub"
    legacy_path.write_text(yaml.safe_dump(legacy_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.get("/api/book-profiles")

    assert response.status_code == 200
    profiles = response.get_json()["profiles"]
    quijote = next(profile for profile in profiles if profile["profile_id"] == "quijote_mx_contemporary")
    assert quijote["deletable"] is True
    assert quijote["in_use"] is False
    assert quijote["profile_goal"] == "modernization"
    legacy = next(profile for profile in profiles if profile["profile_id"] == "auto_legacy_profile")
    assert legacy["profile_goal"] == "faithful_translation"


def test_book_profile_list_caches_knowledge_summary_until_files_change(
    tmp_path,
    monkeypatch,
):
    from flask import Flask

    import src.api.blueprints.profile_routes as profile_routes

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("cached_profile")
    original = profile_routes.build_profile_knowledge_base
    calls = []

    def counted(profile):
        calls.append(profile.profile_id)
        return original(profile)

    profile_routes._profile_knowledge_summary_cached.cache_clear()
    monkeypatch.setattr(profile_routes, "build_profile_knowledge_base", counted)
    app = Flask(__name__)
    app.register_blueprint(profile_routes.create_profile_blueprint())

    with app.test_client() as client:
        assert client.get("/api/book-profiles").status_code == 200
        assert client.get("/api/book-profiles").status_code == 200
        profile_path = profile_dir / "profile.yml"
        profile_path.write_text(
            profile_path.read_text(encoding="utf-8") + "\nnotes: changed\n",
            encoding="utf-8",
        )
        assert client.get("/api/book-profiles").status_code == 200

    assert calls == ["cached_profile", "cached_profile"]


def test_book_profile_glossary_endpoint_exposes_entries(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_book_profile")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Terminus Prime",
                "target": "Terminus Prime",
                "type": "proper_noun",
                "status": "approved",
                "confidence": 0.91,
                "rationale": "Preserve this recurring place name.",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "pending_suggestions.yml").write_text(
        yaml.safe_dump({
            "suggestions": [{
                "source": "lost its way",
                "suggested_target": "",
                "type": "idiom",
                "status": "pending",
                "confidence": 0.83,
                "rationale": "Review this recurring phrase.",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.get("/api/book-profiles/auto_book_profile/glossary")

    assert response.status_code == 200
    data = response.get_json()
    assert data["counts"]["approved"] == 1
    assert data["counts"]["pending"] == 1
    assert data["counts"]["total"] == 2
    assert data["approved_count"] == 1
    assert data["pending_count"] == 1
    assert data["total"] == 2
    sources = {entry["source"]: entry for entry in data["entries"]}
    assert sources["Terminus Prime"]["source_file"] == "terms.yml"
    assert sources["lost its way"]["source_file"] == "pending_suggestions.yml"


def test_book_profile_impact_preview_endpoint_reports_profile_effect(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_impact_profile")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Lake House",
                "target": "Casa del Lago",
                "type": "place",
                "status": "approved",
                "translation_policy": "translate_consistently",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "pending_suggestions.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "blue hour",
                "target": "hora azul",
                "type": "motif",
                "status": "pending",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.post(
            "/api/book-profiles/auto_impact_profile/impact-preview",
            json={"text": "The Lake House appeared during the blue hour."},
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["glossary"]["matched_approved"] == 1
    assert data["glossary"]["matched_pending"] == 1
    assert data["terms_to_translate"][0]["source"] == "Lake House"
    assert data["pending_suggestions"][0]["source"] == "blue hour"


def test_book_profile_impact_preview_restricts_json_paths_to_managed_output(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints import profile_routes
    from src.api.blueprints.profile_routes import create_profile_blueprint

    profiles_root = tmp_path / "profiles"
    output_root = tmp_path / "managed-output"
    output_root.mkdir()
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(profiles_root))
    monkeypatch.setattr(profile_routes._config, "OUTPUT_DIR", str(output_root))
    create_profile("safe_impact_profile", profiles_root=profiles_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("Private host text that is not app managed.", encoding="utf-8")
    managed = output_root / "managed.txt"
    managed.write_text("Managed readable text.", encoding="utf-8")

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    with app.test_client() as client:
        rejected = client.post(
            "/api/book-profiles/safe_impact_profile/impact-preview",
            json={"file_path": str(outside)},
        )
        accepted = client.post(
            "/api/book-profiles/safe_impact_profile/impact-preview",
            json={"file_path": str(managed)},
        )

    assert rejected.status_code == 403
    assert accepted.status_code == 200
    assert accepted.get_json()["source_name"] == "managed.txt"


def test_profile_knowledge_base_unifies_glossary_and_editorial_map(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_knowledge_profile")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Pacific",
                    "target": "Pacífico",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "translate_consistently",
                },
                {
                    "source": "Terminus Prime",
                    "target": "Terminus Prime",
                    "type": "proper_noun",
                    "status": "approved",
                    "translation_policy": "preserve_exact",
                },
            ],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "pending_suggestions.yml").write_text(
        yaml.safe_dump({
            "suggestions": [{
                "source": "hard saying",
                "type": "idiom",
                "status": "pending",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (profile_dir / "editorial").mkdir(exist_ok=True)
    (profile_dir / "editorial" / "editorial_map.yml").write_text(
        yaml.safe_dump({
            "characters_entities": [
                {"name": f"Entity {idx}", "confidence": 0.5}
                for idx in range(180)
            ],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    profile = load_book_profile("auto_knowledge_profile")
    knowledge = build_profile_knowledge_base(profile).to_dict()

    assert knowledge["glossary"]["approved"] == 2
    assert knowledge["glossary"]["pending"] == 1
    assert knowledge["glossary"]["translated_terms"] == 1
    assert knowledge["glossary"]["source_equals_target_approved"] == 1
    assert "characters_entities" in knowledge["editorial_map"]["saturated_buckets"]
    assert knowledge["prompt_readiness"]["recommended_action"] in {
        "refresh_profile_with_full_signal_index",
        "review_source_equals_target_entries",
    }


def test_book_profile_knowledge_base_endpoint(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_book_profile")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [{
                "source": "Bayesian inference",
                "target": "inferencia bayesiana",
                "type": "technical_term",
                "status": "approved",
                "translation_policy": "translate_consistently",
            }],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.get("/api/book-profiles/auto_book_profile/knowledge-base")

    assert response.status_code == 200
    data = response.get_json()["knowledge_base"]
    assert data["profile_id"] == "auto_book_profile"
    assert data["glossary"]["translated_terms"] == 1
    assert data["prompt_readiness"]["ready"] is True


def test_book_profile_glossary_preview_matches_transformation_target_side(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_under_preview")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "Pacific",
                    "target": "Pacífico",
                    "type": "proper_noun",
                    "status": "approved",
                    "confidence": 0.99,
                    "translation_policy": "translate_exact",
                    "rationale": "Nombre propio de océano con traducción establecida al español.",
                },
                {
                    "source": "north-east",
                    "target": "noreste",
                    "type": "technical_term",
                    "status": "approved",
                    "confidence": 0.99,
                    "translation_policy": "translate_exact",
                    "rationale": "Punto cardinal. Traducción directa y estable al español.",
                },
                {
                    "source": "anti-Semitism",
                    "target": "antisemitismo",
                    "type": "concept",
                    "status": "approved",
                    "confidence": 0.98,
                    "translation_policy": "translate_exact",
                    "rationale": "Término con traducción directa en español.",
                },
                {
                    "source": "Suppress Me",
                    "target": "suprimirme",
                    "type": "concept",
                    "status": "approved",
                    "confidence": 0.99,
                    "injection_policy": "do_not_inject",
                    "translation_policy": "translate_exact",
                    "rationale": "Approved but intentionally excluded from prompt injection.",
                },
            ],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.post(
            "/api/book-profiles/auto_under_preview/glossary/preview-block",
            json={
                "text": "El Pacífico quedaba al noreste; el antisemitismo aparecía en la prensa.",
                "purpose": "transformation",
            },
        )

    assert response.status_code == 200
    data = response.get_json()
    assert data["matched_count"] == 3
    assert data["approved_count"] == 4
    assert data["total_terms"] == 3
    assert "Pacific -> Pacífico" in data["block"]
    assert "north-east -> noreste" in data["block"]
    assert "anti-Semitism -> antisemitismo" in data["block"]
    assert "Suppress Me" not in data["block"]


def test_profile_glossary_transformation_matches_inflected_target_side(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_inflected_preview")
    (profile_dir / "glossary" / "terms.yml").write_text(
        yaml.safe_dump({
            "entries": [
                {
                    "source": "British",
                    "target": "británico",
                    "type": "technical_term",
                    "status": "approved",
                    "confidence": 0.99,
                    "translation_policy": "translate_exact",
                },
                {
                    "source": "Fascist",
                    "target": "fascista",
                    "type": "technical_term",
                    "status": "approved",
                    "confidence": 0.99,
                    "translation_policy": "translate_exact",
                },
            ],
        }, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    block = build_profile_glossary_block(
        "La embajada británica y los grupos fascistas aparecen en el pasaje.",
        {"editorial_mode": "book_profile", "profile_id": "auto_inflected_preview"},
        purpose="transformation",
    )

    assert "British -> británico" in block
    assert "Fascist -> fascista" in block
    assert "canonical lemma" in block


def test_profile_glossary_purpose_for_audit_and_repair_uses_rendered_side():
    from src.core import translator
    from src.core import postprocess_repair

    assert translator._profile_glossary_purpose({}, "translation") == "translation"
    assert translator._profile_glossary_purpose({}, "profile_audit") == "refinement"
    assert translator._profile_glossary_purpose({}, "profile_repair") == "refinement"
    assert translator._profile_glossary_purpose({"text_transform_mode": "modernize"}, "profile_audit") == "transformation"
    assert postprocess_repair._profile_glossary_purpose({"text_transform_mode": "modernize"}) == "transformation"
    assert postprocess_repair._profile_glossary_purpose({}) == "refinement"


def test_corrupt_profile_glossary_still_lists_for_delete(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_corrupt_profile")
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config.update({
        "name": "Corrupt generated profile",
        "generated_profile": True,
        "source_name": "Corrupt Book.epub",
    })
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (profile_dir / "glossary" / "terms.yml").write_text(
        "entries:\n- source: Bad\n  rationale: \"unterminated\n",
        encoding="utf-8",
    )

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        list_response = client.get("/api/book-profiles")
        detail_response = client.get("/api/book-profiles/auto_corrupt_profile/glossary")

    assert list_response.status_code == 200
    profiles = {
        item["profile_id"]: item
        for item in list_response.get_json()["profiles"]
    }
    assert profiles["auto_corrupt_profile"]["load_error"]
    assert profiles["auto_corrupt_profile"]["deletable"] is True
    assert detail_response.status_code == 200
    detail = detail_response.get_json()
    assert detail["profile"]["load_error"]
    assert detail["entries"] == []


def test_book_profile_delete_allows_quijote_profile(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    create_profile("quijote_mx_contemporary")

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        response = client.delete("/api/book-profiles/quijote_mx_contemporary")

    assert response.status_code == 200
    assert response.get_json()["deleted"] is True
    assert not (tmp_path / "quijote_mx_contemporary").exists()


def test_book_profile_delete_blocks_protected_profile(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("audiobook_faithful")
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config["protected_profile"] = True
    profile_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())

    with app.test_client() as client:
        list_response = client.get("/api/book-profiles")
        delete_response = client.delete("/api/book-profiles/audiobook_faithful")

    assert delete_response.status_code == 403
    profiles = {
        item["profile_id"]: item
        for item in list_response.get_json()["profiles"]
    }
    assert profiles["audiobook_faithful"]["deletable"] is False
    assert (tmp_path / "audiobook_faithful").exists()


def test_book_profile_delete_blocks_active_profile(tmp_path, monkeypatch):
    from flask import Flask

    from src.api.blueprints.profile_routes import create_profile_blueprint

    class FakeStateManager:
        def get_all_translations(self):
            return {
                "trans_active": {
                    "status": "running",
                    "config": {
                        "prompt_options": {
                            "profile_id": "auto_book_profile",
                        },
                    },
                },
            }

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    create_profile("auto_book_profile")

    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint(state_manager=FakeStateManager()))

    with app.test_client() as client:
        response = client.delete("/api/book-profiles/auto_book_profile")

    assert response.status_code == 409
    assert (tmp_path / "auto_book_profile" / "profile.yml").exists()


def test_no_quijote_editorial_equivalences_are_hardcoded_in_src():
    banned = [
        "agora",
        "desta",
        "vuesa merced",
        "vuestra merced",
        "su merced",
        "ansí",
    ]
    offenders = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for term in banned:
            pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} contains {term!r}")

    assert not offenders
