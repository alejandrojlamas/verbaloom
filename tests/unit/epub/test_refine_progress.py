import zipfile

import pytest

from src.core.refine import epub_refiner


def _write_test_epub(path):
    container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
    opf_xml = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Test EPUB</dc:title>
    <dc:language>es</dc:language>
  </metadata>
  <manifest>
    <item id="chap1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
    <item id="chap2" href="chap2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="chap2"/>
    <itemref idref="chap1"/>
  </spine>
</package>
"""
    xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <head><title>Chapter</title></head>
  <body><p>Texto de prueba.</p></body>
</html>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/content.opf", opf_xml)
        zf.writestr("OEBPS/chap1.xhtml", xhtml)
        zf.writestr("OEBPS/chap2.xhtml", xhtml)


@pytest.mark.asyncio
async def test_epub_refine_reports_real_chunks_in_spine_order(tmp_path, monkeypatch):
    input_epub = tmp_path / "input.epub"
    output_epub = tmp_path / "output.epub"
    _write_test_epub(input_epub)

    seen_hrefs = []
    progress = []
    chunks_by_href = {"chap2.xhtml": 2, "chap1.xhtml": 3}

    monkeypatch.setattr(
        epub_refiner,
        "build_refine_client",
        lambda **_kwargs: (object(), None),
    )

    def fake_count(file_path, _max_tokens_per_chunk):
        return chunks_by_href[file_path.rsplit("/", 1)[-1]]

    monkeypatch.setattr(epub_refiner, "_count_refine_chunks_for_xhtml", fake_count)

    async def fake_refine_one_xhtml(**kwargs):
        href = kwargs["prompt_options"]["_editorial_section_prefix"]
        seen_hrefs.append(href)
        for completed in range(1, chunks_by_href[href] + 1):
            kwargs["stats_callback"]({
                "total_chunks": chunks_by_href[href],
                "completed_chunks": completed,
                "failed_chunks": 0,
            })
        return True

    monkeypatch.setattr(epub_refiner, "_refine_one_xhtml", fake_refine_one_xhtml)

    ok = await epub_refiner.refine_epub_file(
        str(input_epub),
        str(output_epub),
        target_language="Spanish",
        stats_callback=progress.append,
        prompt_options={"editorial_quality_report": False},
    )

    assert ok is True
    assert output_epub.exists()
    assert seen_hrefs == ["chap2.xhtml", "chap1.xhtml"]
    assert progress[0]["total_chunks"] == 5
    assert progress[-1]["total_chunks"] == 5
    assert progress[-1]["completed_chunks"] == 5
