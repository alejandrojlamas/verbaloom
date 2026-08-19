from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

import pytest
from lxml import etree

from src.core.epub.unit_repairs import EpubUnitRepairError, apply_epub_unit_repairs


XHTML = b'''<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Test</title></head><body>
<p>Los cuatro caballos <em>overos</em> avanzaron.</p>
<p>Texto repetido. Texto repetido.</p>
</body></html>'''


def _epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("Text/chapter.xhtml", XHTML)
        archive.writestr("Images/cover.png", b"unchanged-image")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_exact_unit_repair_preserves_markup_resources_and_mimetype(tmp_path):
    epub = tmp_path / "book.epub"
    _epub(epub)
    with zipfile.ZipFile(epub) as archive:
        image_hash = _sha(archive.read("Images/cover.png"))

    report = apply_epub_unit_repairs(epub, [{
        "file_href": "Text/chapter.xhtml",
        "dom_path": "/*/*[2]/*[1]",
        "old_text": "overos",
        "new_text": "bayos",
        "reason": "correct horse colour",
    }])

    assert report.applied_repairs == 1
    with zipfile.ZipFile(epub) as archive:
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert _sha(archive.read("Images/cover.png")) == image_hash
        root = etree.fromstring(archive.read("Text/chapter.xhtml"))
        paragraph = root.xpath("/*/*[2]/*[1]")[0]
        assert "".join(paragraph.itertext()) == "Los cuatro caballos bayos avanzaron."
        assert paragraph.xpath("./*[local-name()='em']")


def test_unit_repair_is_atomic_when_precondition_is_missing(tmp_path):
    epub = tmp_path / "book.epub"
    _epub(epub)
    before = epub.read_bytes()

    with pytest.raises(EpubUnitRepairError, match="expected 1 exact occurrence"):
        apply_epub_unit_repairs(epub, [{
            "file_href": "Text/chapter.xhtml",
            "dom_path": "/*/*[2]/*[1]",
            "old_text": "does not exist",
            "new_text": "replacement",
        }])

    assert epub.read_bytes() == before


def test_unit_repair_rejects_ambiguous_occurrences(tmp_path):
    epub = tmp_path / "book.epub"
    _epub(epub)

    with pytest.raises(EpubUnitRepairError, match="found 2"):
        apply_epub_unit_repairs(epub, [{
            "file_href": "Text/chapter.xhtml",
            "dom_path": "/*/*[2]/*[2]",
            "old_text": "Texto repetido.",
            "new_text": "Texto corregido.",
        }])


def test_unit_repair_rejects_unexpected_epub_hash(tmp_path):
    epub = tmp_path / "book.epub"
    _epub(epub)

    with pytest.raises(EpubUnitRepairError, match="EPUB hash mismatch"):
        apply_epub_unit_repairs(
            epub,
            [{
                "file_href": "Text/chapter.xhtml",
                "dom_path": "/*/*[2]/*[1]",
                "old_text": "overos",
                "new_text": "bayos",
            }],
            expected_epub_sha256="0" * 64,
        )


def test_inline_marker_realigns_existing_nodes_without_changing_visible_text(tmp_path):
    epub = tmp_path / "book.epub"
    _epub(epub)
    visible = "Los cuatro caballos overos avanzaron."

    report = apply_epub_unit_repairs(epub, [{
        "file_href": "Text/chapter.xhtml",
        "dom_path": "/*/*[2]/*[1]",
        "operation": "realign_inline_markers",
        "inline_markers": ["caballos"],
        "expected_visible_sha256": hashlib.sha256(visible.encode("utf-8")).hexdigest(),
    }])

    assert report.applied_repairs == 1
    with zipfile.ZipFile(epub) as archive:
        root = etree.fromstring(archive.read("Text/chapter.xhtml"))
    paragraph = root.xpath("/*/*[2]/*[1]")[0]
    assert "".join(paragraph.itertext()) == visible
    assert paragraph[0].text == "caballos"
