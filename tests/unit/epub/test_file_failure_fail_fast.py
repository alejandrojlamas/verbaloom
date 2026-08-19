from unittest.mock import AsyncMock

import pytest
from lxml import etree

from src.core.epub.translation_metrics import TranslationMetrics
from src.core.epub.translator import _process_all_content_files, translate_epub_file


@pytest.mark.asyncio
async def test_epub_stops_at_failed_file_and_counts_it_as_failed(monkeypatch, tmp_path):
    files = ["one.xhtml", "two.xhtml", "three.xhtml"]
    monkeypatch.setattr(
        "src.core.epub.translator._precount_chunks",
        AsyncMock(return_value=(3, [1, 1, 1])),
    )
    calls = []

    async def fake_translate(**kwargs):
        calls.append(kwargs["content_href"])
        if kwargs["content_href"] == "one.xhtml":
            return etree.Element("html"), True, TranslationMetrics()
        return None, False, None

    monkeypatch.setattr(
        "src.core.epub.translator._translate_single_xhtml_file",
        fake_translate,
    )

    result = await _process_all_content_files(
        content_files=files,
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="German",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        max_attempts=1,
        context_manager=None,
        translation_id=None,
        prompt_options={"abort_on_profile_fail": True},
    )

    assert calls == ["one.xhtml", "two.xhtml"]
    assert result["completed_files"] == 1
    assert result["completed_chunks"] == 1
    assert result["failed_files"] == 1
    assert result["failed_chunks"] == 1


@pytest.mark.asyncio
async def test_file_level_quality_failure_does_not_relabel_completed_siblings(monkeypatch, tmp_path):
    files = ["chapter.xhtml", "later.xhtml"]
    monkeypatch.setattr(
        "src.core.epub.translator._precount_chunks",
        AsyncMock(return_value=(22, [21, 1])),
    )

    async def fake_translate(**_kwargs):
        return None, False, None

    monkeypatch.setattr(
        "src.core.epub.translator._translate_single_xhtml_file",
        fake_translate,
    )

    result = await _process_all_content_files(
        content_files=files,
        opf_dir=str(tmp_path),
        temp_dir=str(tmp_path),
        source_language="English",
        target_language="Spanish",
        model_name="test-model",
        llm_client=object(),
        max_tokens_per_chunk=100,
        max_attempts=1,
        context_manager=None,
        translation_id=None,
        prompt_options={"abort_on_profile_fail": True},
    )

    assert result["completed_chunks"] == 0
    assert result["failed_files"] == 1
    assert result["failed_chunks"] == 1


@pytest.mark.asyncio
async def test_automatic_quality_failure_does_not_package_partial_epub(monkeypatch, tmp_path):
    input_path = tmp_path / "source.epub"
    output_path = tmp_path / "translated.epub"
    input_path.write_bytes(b"epub")

    monkeypatch.setattr(
        "src.core.epub.translator._create_llm_client",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "src.core.epub.translator._create_context_manager",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.epub.translator._extract_epub",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "src.core.epub.translator._parse_epub_manifest",
        lambda temp_dir, _log: {
            "content_files": ["one.xhtml", "two.xhtml"],
            "opf_dir": temp_dir,
            "opf_tree": etree.ElementTree(etree.Element("package")),
            "opf_path": str(tmp_path / "content.opf"),
        },
    )
    monkeypatch.setattr(
        "src.core.epub.translator._process_all_content_files",
        AsyncMock(return_value={
            "parsed_docs": {},
            "completed_files": 1,
            "failed_files": 1,
            "total_chunks": 2,
            "completed_chunks": 1,
            "failed_chunks": 1,
            "translation_stats": TranslationMetrics(),
            "was_interrupted": False,
        }),
    )
    monkeypatch.setattr(
        "src.core.epub.translator._save_translated_files",
        AsyncMock(),
    )
    repackage = AsyncMock()
    monkeypatch.setattr("src.core.epub.translator._repackage_epub", repackage)
    monkeypatch.setattr(
        "src.core.epub.translator.write_fidelity_report_from_options",
        lambda *_args, **_kwargs: None,
    )

    success = await translate_epub_file(
        str(input_path),
        str(output_path),
        source_language="English",
        target_language="Spanish",
    )

    assert success is False
    assert not output_path.exists()
    repackage.assert_not_awaited()
