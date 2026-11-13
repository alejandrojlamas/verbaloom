"""Private output, actual source extraction, and synthetic calibration coverage."""

import json
from pathlib import Path

import pytest

from scripts.editorial_benchmark import (
    calibrate_local,
    main,
    private_directory,
    source_units,
    write_private,
)


def test_private_artifacts_are_atomic_exclusive_and_outside_source_control(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("scripts.editorial_benchmark.REPO", tmp_path / "repo")
    with pytest.raises(ValueError):
        private_directory(tmp_path / "repo/benchmark_results")
    folder = private_directory(tmp_path / "private")
    assert folder.stat().st_mode & 0o777 == 0o700
    target = folder / "corpus.json"
    write_private(target, {"source": "Private fixture"})
    assert target.stat().st_mode & 0o777 == 0o600
    previous = target.read_bytes()
    with pytest.raises(ValueError):
        write_private(target, {"source": "New fixture"})
    assert target.read_bytes() == previous
    assert list(folder.iterdir()) == [target]


def test_public_directory_and_symlinks_are_not_silently_reused(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.editorial_benchmark.REPO", tmp_path / "repo")
    folder = tmp_path / "public"
    folder.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        private_directory(folder)
    link = tmp_path / "link"
    link.symlink_to(folder)
    with pytest.raises(ValueError):
        private_directory(link)


def test_txt_extraction_preserves_paragraph_order_and_formatting(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("One *word*.\n\nSecond.\nA soft line.\n\nThird.", encoding="utf-8")
    units, languages = source_units(path)
    assert languages == []
    assert [u["text"] for u in units] == [
        "One *word*.",
        "Second.\nA soft line.",
        "Third.",
    ]
    assert [u["position"] for u in units] == [0, 1, 2]


def test_negative_controls_are_real_mutations_and_not_semantic_success_claims():
    result = calibrate_local()
    assert len(result["records"]) == 8
    assert {r["expected_kind"] for r in result["records"] if r["expected_kind"]} == {
        "negation",
        "omission",
        "name_changed",
        "duplication",
        "flattened_voice",
        "marker_damage",
    }
    assert len({r["candidate"] for r in result["records"]}) == 8
    assert result["paid_requests"] == 0
    assert not result["semantic_detection_measured"]
    assert all(r["expected_defect_confirmed"] is None for r in result["records"])


def test_cli_requires_profile_provenance_and_preserves_unresolved_locale(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("scripts.editorial_benchmark.REPO", tmp_path / "repo")
    path = tmp_path / "source.txt"
    path.write_text("\n\n".join(f"Paragraph {i}." for i in range(12)), encoding="utf-8")
    directory = tmp_path / "private"
    main(
        [
            "prepare",
            str(path),
            "--unresolved-profile",
            "--source-language",
            "Czech",
            "--target-language",
            "Spanish",
            "--tuning-start",
            "0",
            "--heldout-start",
            "7",
            "--sequence-length",
            "3",
            "--output",
            str(directory),
        ]
    )
    value = json.loads((directory / "corpus.json").read_text())
    assert value["book"]["target_locale"] == ""
    assert value["book"]["profile"]["status"] == "unresolved"
    assert len(value["units"]) == 6


def test_incomplete_corpus_is_not_created_after_source_parse_errors(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "src.core.epub.publication_gate.snapshot_epub",
        lambda p: SimpleNamespace(
            parse_errors=["malformed XML"],
            unsafe_paths=[],
            units=[],
            languages=["cs"],
        ),
    )
    with pytest.raises(ValueError, match="structural review"):
        source_units(Path("fixture.epub"))
