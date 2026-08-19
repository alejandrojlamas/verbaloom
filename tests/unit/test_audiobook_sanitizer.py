import asyncio
from pathlib import Path

import yaml

from src.core.audiobook_sanitizer import (
    audiobook_profile_enabled,
    sanitize_for_audiobook,
)
from src.core.book_profiles import create_profile, load_book_profile
from src.core.book_profiles.preparation import prepare_book_profile_from_text
from src.core.book_profiles.rendering import build_profile_instruction_block
from src.core.output_formats import extract_readable_text, write_text_as_output


def test_audiobook_sanitizer_removes_links_notes_and_moves_apparatus():
    source = """Texto narrativo antes.[[23]](../Text/notas.xhtml#nt23)

[OceanofPDF.com](https://oceanofpdf.com)

Figure 12: Mark Cousins interviews a filmmaker beside a Steenbeck editing table

Photo courtesy of Example Archive

[23] Esta nota explica una referencia editorial.

References
Kundera, Milan. 1984. pp. 15-17.

Texto narrativo despues."""

    artifact = sanitize_for_audiobook(source, target_language="Spanish")

    assert "Texto narrativo antes." in artifact.main_text
    assert "Texto narrativo despues." in artifact.main_text
    assert "OceanofPDF" not in artifact.text
    assert "https://oceanofpdf.com" not in artifact.text
    assert "[[23]]" not in artifact.main_text
    assert "Descripcion de imagen: Mark Cousins interviews" in artifact.main_text
    assert "Photo courtesy of Example Archive" not in artifact.main_text
    assert "Apendice de notas" in artifact.appendix_text
    assert "Esta nota explica" in artifact.appendix_text
    assert "Kundera, Milan" in artifact.appendix_text
    assert artifact.report.captions_integrated == 1
    assert artifact.report.links_removed >= 1
    assert artifact.report.note_calls_removed >= 1


def test_audiobook_profile_loads_and_injects_translation_guidance():
    profile = load_book_profile("audiobook_faithful")

    assert profile is not None
    assert profile.raw_config["audiobook"]["enabled"] is True

    options = {
        "editorial_mode": "book_profile",
        "profile_id": "audiobook_faithful",
    }
    block = build_profile_instruction_block(options, phase="translation", target_language="Spanish")

    assert "Audiobook companion policy" in block
    assert "faithful" in block.lower()
    assert audiobook_profile_enabled(options) is True


def test_generated_audiobook_profile_can_enable_companion(monkeypatch, tmp_path):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile_dir = create_profile("auto_audio_book", profiles_root=tmp_path)
    profile_path = profile_dir / "profile.yml"
    config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    config.update({
        "profile_id": "auto_audio_book",
        "name": "Audio Book",
        "audiobook": {"enabled": True, "generate_companion": True},
    })
    profile_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    assert audiobook_profile_enabled({
        "editorial_mode": "book_profile",
        "profile_id": "auto_audio_book",
    }) is True


def test_profile_preparation_audiobook_purpose_marks_generated_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    text = (
        "Mark Cousins discusses montage, close-ups, and a captioned film still. "
        "Figure 1: A filmmaker stands beside a camera during production. "
    ) * 30

    result = asyncio.run(prepare_book_profile_from_text(
        text,
        source_name="The Story of Film.epub",
        language="English",
        target_locale="es-MX",
        transform_mode="audiobook",
        max_llm_chunks=0,
    ))

    profile = load_book_profile(result.profile_id)
    assert profile.raw_config["audiobook"]["enabled"] is True
    assert profile.raw_config["audiobook"]["notes_policy"] == "appendix"
    assert "Politica de audiolibro" in (profile.policy_text or "")
    assert audiobook_profile_enabled({
        "editorial_mode": "book_profile",
        "profile_id": result.profile_id,
    }) is True


def test_audiobook_text_can_be_written_as_epub_companion(tmp_path):
    artifact = sanitize_for_audiobook(
        "Capitulo uno\n\nTexto limpio para escuchar.",
        target_language="Spanish",
    )
    epub_path = tmp_path / "Libro (Audiolibro).epub"

    write_text_as_output(artifact.text, epub_path, "epub")
    extracted = extract_readable_text(Path(epub_path))

    assert "Capitulo uno" in extracted
    assert "Texto limpio para escuchar" in extracted
