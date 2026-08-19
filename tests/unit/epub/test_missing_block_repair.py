from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import pytest

from src.core.epub.missing_block_repair import repair_epub_missing_blocks_with_llm


def _write_epub(path: Path, paragraph: str) -> None:
    chapter = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f"{paragraph}</body></html>"
    ).encode()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", b"application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr("Text/chapter.xhtml", chapter)


@dataclass
class _Response:
    content: str


class _Provider:
    async def generate(self, prompt, **kwargs):
        assert "I cold." in prompt
        assert kwargs["temperature"] == 0.1
        return _Response("Tenía frío.")


class _ProviderMustNotRun:
    async def generate(self, prompt, **kwargs):
        raise AssertionError("sanitized source artifacts must not call the LLM")


@pytest.mark.asyncio
async def test_selective_missing_block_repair_uses_source_context_and_patches_output(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(source, "<p>I cold.</p>")
    _write_epub(output, "<p></p>")

    report = await repair_epub_missing_blocks_with_llm(
        str(source),
        str(output),
        config={
            "source_language": "English",
            "target_language": "Spanish",
            "prompt_options": {"target_locale": "es-MX"},
        },
        provider=_Provider(),
    )

    assert report.clean
    assert report.found == 1
    assert report.repaired == 1
    with zipfile.ZipFile(output) as archive:
        assert "Tenía frío." in archive.read("Text/chapter.xhtml").decode()


@pytest.mark.asyncio
async def test_selective_missing_block_repair_skips_sanitized_artifacts(tmp_path):
    source = tmp_path / "source.epub"
    output = tmp_path / "output.epub"
    _write_epub(
        source,
        '<p><a href="https://example.com">www.example.com/author</a></p>',
    )
    _write_epub(
        output,
        '<p class="verbaloom-sanitized-artifact" style="display: none"><a></a></p>',
    )

    report = await repair_epub_missing_blocks_with_llm(
        str(source),
        str(output),
        config={
            "source_language": "English",
            "target_language": "Spanish",
        },
        provider=_ProviderMustNotRun(),
    )

    assert report.clean
    assert report.found == 0
    assert report.repaired == 0
