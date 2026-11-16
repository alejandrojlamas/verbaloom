"""URI-encoded manifest paths must work throughout the EPUB lifecycle."""

from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote, unquote, urlsplit
import zipfile

from lxml import etree
import pytest

from src.core.epub import translator
from src.core.epub.metadata_localizer import _manifest_info
from src.core.epub.publication_gate import _broken_toc_targets, snapshot_epub
from src.core.epub.publication_structure import finish_epub_publication
from src.core.epub.resource_links import resource_uri_path
from src.core.epub.translation_metrics import TranslationMetrics


@pytest.mark.parametrize("name", [
    "Chapter 1.xhtml", "Chapter+1.xhtml", "Chapter%20.xhtml",
    "Chapter #1.xhtml", "Chapter ?1.xhtml", "Cap\u00edtulo 1.xhtml",
])
def test_uri_decodes_once_without_treating_plus_as_space(name):
    assert resource_uri_path(quote(name, safe="") + "?version=1#section") == name


def _book(tmp_path, name="Chapter 1.xhtml"):
    opf_dir = tmp_path / "OEBPS"
    (opf_dir / "Text").mkdir(parents=True)
    hrefs = ["Text/front.xhtml", "Text/" + quote(name, safe="")]
    opf = etree.fromstring(b'''<package xmlns="http://www.idpf.org/2007/opf"
      xmlns:dc="http://purl.org/dc/elements/1.1/" version="3.0" unique-identifier="book">
      <metadata><dc:identifier id="book">urn:test:encoded-paths</dc:identifier>
      <dc:title>Example</dc:title><dc:language>en</dc:language>
      <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta></metadata>
      <manifest/><spine/></package>''')
    manifest, spine = opf[-2:]
    for index, href in enumerate(hrefs):
        etree.SubElement(manifest, "{http://www.idpf.org/2007/opf}item",
                         id=f"chapter-{index}", href=href,
                         **{"media-type": "application/xhtml+xml"})
        etree.SubElement(spine, "{http://www.idpf.org/2007/opf}itemref", idref=f"chapter-{index}")
        paragraphs = ("<p>The traveler returned home and told the story.</p>"
                      * (1 if index == 0 else 120))
        (opf_dir / resource_uri_path(href)).write_text(
            '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Example</title></head>'
            f'<body><h1 id="opening">Chapter {index}</h1>{paragraphs}</body></html>',
            encoding="utf-8",
        )
    etree.ElementTree(opf).write(str(opf_dir / "content.opf"), encoding="utf-8")
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF/container.xml").write_text(
        '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">'
        '<rootfiles><rootfile full-path="OEBPS/content.opf" '
        'media-type="application/oebps-package+xml"/></rootfiles></container>', encoding="utf-8")
    (tmp_path / "mimetype").write_text("application/epub+zip", encoding="ascii")
    return translator._parse_epub_manifest(str(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Chapter 1.xhtml", "Chapter #1%20+?.xhtml", "Cap\u00edtulo 1.xhtml"])
async def test_manifest_planning_and_resume_use_decoded_files(tmp_path, monkeypatch, name):
    data = _book(tmp_path, name)
    hrefs, base = data["content_files"], data["opf_dir"]
    total, counts = await translator._precount_chunks(hrefs, base, 200)
    assert len(counts) == 2 and counts[0] == 1 and counts[1] > 1
    calls, ticks = [], []

    async def translate(**kwargs):
        calls.append(kwargs["content_href"])
        return etree.parse(kwargs["file_path"]).getroot(), True, TranslationMetrics()

    class Checkpoint:
        def get_job(self, _job):
            return {"progress": {"total_chunks": 1, "completed_chunks": 1}}

    monkeypatch.setattr(translator, "_translate_single_xhtml_file", translate)
    monkeypatch.setattr(translator, "_save_checkpoint", AsyncMock())
    first_path = str(Path(base) / resource_uri_path(hrefs[0]))
    result = await translator._process_all_content_files(
        content_files=hrefs, opf_dir=base, temp_dir=str(tmp_path),
        source_language="English", target_language="Spanish", model_name="test-model",
        llm_client=object(), max_tokens_per_chunk=200, max_attempts=1,
        context_manager=None, translation_id="job", resume_from_index=1,
        checkpoint_manager=Checkpoint(), stats_callback=ticks.append,
        restored_docs={first_path: etree.parse(first_path).getroot()},
        restore_attempted=True, precount_result=(total, counts),
    )
    assert calls == [hrefs[1]]
    assert result["failed_files"] == 0
    assert result["completed_chunks"] == result["total_chunks"] == total
    assert ticks[0]["completed_chunks"] == 1 and ticks[0]["total_chunks"] == total


@pytest.mark.parametrize("href", ["Text/missing.xhtml", "../outside.xhtml", "https://example.invalid/book.xhtml"])
def test_invalid_spine_fails_before_translation(tmp_path, href):
    data = _book(tmp_path)
    opf = data["opf_tree"]
    opf.xpath("//*[local-name()='item']")[1].set("href", href)
    opf.write(data["opf_path"], encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError), match="EPUB spine resource"):
        translator._parse_epub_manifest(str(tmp_path))


@pytest.mark.asyncio
async def test_precount_never_reports_missing_content_as_zero(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        await translator._precount_chunks(["missing.xhtml"], str(tmp_path), 200)


@pytest.mark.parametrize("name", ["Chapter 1.xhtml", "Chapter #1%20+?.xhtml", "Cap\u00edtulo 1.xhtml"])
def test_finishing_and_publication_snapshot_include_encoded_chapters(tmp_path, name):
    data = _book(tmp_path, name)
    base = Path(data["opf_dir"])
    docs = {str(base / resource_uri_path(href)): etree.parse(str(base / resource_uri_path(href))).getroot()
            for href in data["content_files"]}
    report = finish_epub_publication(
        opf_tree=data["opf_tree"], opf_dir=base, content_files=data["content_files"],
        parsed_xhtml_docs=docs, target_language="English", package_root=tmp_path,
    )
    assert report.chapter_headings == 2 and report.navigation_entries == 2
    for path, root in docs.items():
        etree.ElementTree(root).write(path, encoding="utf-8")
    data["opf_tree"].write(data["opf_path"], encoding="utf-8")
    nav = etree.parse(str(base / report.navigation_href))
    targets = nav.xpath("//*[local-name()='nav' and @*[local-name()='type']='toc']//*[local-name()='a']/@href")
    assert len(targets) == 2
    for target in targets:
        parsed = urlsplit(target)
        assert not parsed.query and " " not in target
        doc = etree.parse(str(base / resource_uri_path(target)))
        assert doc.xpath("//*[@id=$id]", id=unquote(parsed.fragment))
    archive_path = tmp_path.parent / "finished.epub"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in tmp_path.rglob("*"):
            if path.is_file():
                archive.write(path, str(path.relative_to(tmp_path)))
    snapshot = snapshot_epub(archive_path)
    assert not snapshot.parse_errors
    assert not _broken_toc_targets(snapshot)
    assert "OEBPS/Text/" + name in snapshot.spine
    href_by_id, _nav, _ncx = _manifest_info(data["opf_tree"].getroot(), "OEBPS/content.opf")
    assert href_by_id["chapter-1"] == "OEBPS/Text/" + name
