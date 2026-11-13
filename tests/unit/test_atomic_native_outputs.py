"""Disk failures must not truncate a previous native-format delivery."""

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

from src.core.docx.translator import translate_docx_file
from src.core.epub.translator import _repackage_epub


@pytest.fixture
def epub_tree(tmp_path):
    folder = tmp_path / "unpacked"
    folder.mkdir()
    (folder / "mimetype").write_bytes(b"application/epub+zip")
    (folder / "chapter.xhtml").write_bytes(b"<html><body><p>Text</p></body></html>")
    (folder / "image.bin").write_bytes(b"original image bytes")
    return folder


@pytest.mark.parametrize("failure", ["archive_write", "fsync", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_epub_packaging_failure_preserves_destination(
    tmp_path, monkeypatch, epub_tree, failure, existing
):
    destination = tmp_path / "output.epub"
    if existing:
        destination.write_bytes(b"previous valid delivery")

    def broken(*args, **kwargs):
        raise OSError("injected disk failure")

    if failure == "archive_write":
        real_write = zipfile.ZipFile.write

        def fail_member(archive, filename, arcname, **kwargs):
            if arcname == "chapter.xhtml":
                broken()
            return real_write(archive, filename, arcname, **kwargs)

        monkeypatch.setattr(zipfile.ZipFile, "write", fail_member)
    elif failure == "fsync":
        monkeypatch.setattr("src.core.quality_assurance.models.os.fsync", broken)
    else:
        monkeypatch.setattr(Path, "replace", broken)
    with pytest.raises(OSError, match="injected disk failure"):
        _repackage_epub(str(epub_tree), str(destination))
    if existing:
        assert destination.read_bytes() == b"previous valid delivery"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob(".output.epub.*.tmp"))


def test_epub_is_complete_before_destination_changes(tmp_path, monkeypatch, epub_tree):
    destination = tmp_path / "output.epub"
    destination.write_bytes(b"previous valid delivery")
    real_write = zipfile.ZipFile.write

    def verify_during_write(archive, *args, **kwargs):
        assert destination.read_bytes() == b"previous valid delivery"
        return real_write(archive, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "write", verify_during_write)
    _repackage_epub(str(epub_tree), str(destination))
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        assert archive.read("image.bin") == b"original image bytes"
    assert not list(tmp_path.glob(".output.epub.*.tmp"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "fsync", "replace"])
async def test_docx_atomic_delivery(tmp_path, monkeypatch, failure):
    destination = tmp_path / "output.docx"
    destination.write_bytes(b"previous valid delivery")
    document = Document()
    document.add_paragraph("Translated text")
    payload = io.BytesIO()
    document.save(payload)

    async def translated(*args, **kwargs):
        return payload.getvalue(), SimpleNamespace(to_dict=dict)

    def broken(*args, **kwargs):
        raise OSError("injected disk failure")

    monkeypatch.setattr(
        "src.core.docx.translator.GenericTranslationOrchestrator.translate", translated
    )
    if failure == "fsync":
        monkeypatch.setattr("src.core.quality_assurance.models.os.fsync", broken)
    elif failure == "replace":
        monkeypatch.setattr(Path, "replace", broken)

    async def run():
        return await translate_docx_file(
            str(tmp_path / "source.docx"),
            str(destination),
            "English",
            "Spanish",
            "synthetic-model",
            SimpleNamespace(provider_type="synthetic"),
        )

    if failure:
        with pytest.raises(OSError, match="injected disk failure"):
            await run()
        assert destination.read_bytes() == b"previous valid delivery"
    else:
        result = await run()
        assert result["success"]
        assert Document(destination).paragraphs[0].text == "Translated text"
    assert not list(tmp_path.glob(".output.docx.*.tmp"))
