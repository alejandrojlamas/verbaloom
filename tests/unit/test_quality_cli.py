from __future__ import annotations

import json

from src.core.quality_assurance.cli import _atomic_publish, build_parser, main


def test_cli_exposes_all_required_commands():
    parser = build_parser()
    help_text = parser.format_help()

    for command in ("inspect", "translate", "resume", "validate", "repair", "export", "report"):
        assert command in help_text


def test_cli_inspect_writes_stable_manifest(tmp_path, capsys):
    source = tmp_path / "book.md"
    source.write_text("Chapter One\n\nA complete source paragraph.", encoding="utf-8")

    result = main([
        "--report-root", str(tmp_path / "runs"),
        "inspect", str(source),
        "--source-language", "English",
        "--target-language", "Spanish",
        "--run-id", "inspect-test",
    ])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    manifest = tmp_path / "runs" / "inspect-test" / "translation_manifest.json"
    assert payload["manifest"] == str(manifest)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["source_format"] == "txt"
    assert data["counts"]["extracted"] == 2


def test_cli_validate_direct_pair_writes_report(tmp_path, capsys):
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text(
        "The traveler crossed the old bridge and returned to the house.",
        encoding="utf-8",
    )
    output.write_text(
        "El viajero cruzo el puente antiguo y regreso a la casa.",
        encoding="utf-8",
    )

    result = main([
        "--report-root", str(tmp_path / "runs"),
        "validate",
        "--source", str(source),
        "--output", str(output),
        "--source-language", "English",
        "--target-language", "Spanish",
        "--target-locale", "es-MX",
    ])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["publishable"] is True
    assert (tmp_path / "runs" / payload["run_id"] / "translation_report.html").exists()


def test_atomic_publish_never_overwrites_existing_valid_output(tmp_path):
    staged = tmp_path / "staged.txt"
    destination = tmp_path / "book.txt"
    staged.write_text("new validated book", encoding="utf-8")
    destination.write_text("previous valid book", encoding="utf-8")

    published = _atomic_publish(staged, destination)

    assert destination.read_text(encoding="utf-8") == "previous valid book"
    assert published != destination
    assert published.read_text(encoding="utf-8") == "new validated book"
