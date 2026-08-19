"""
Smoke tests for the Translator's Sampler EPUB fixture.

The sampler is built on the fly by scripts/build_test_epub.py via the
session-scoped fixtures defined in tests/conftest.py. These tests pin down
the shape of the fixture so downstream tests can rely on it.
"""

import zipfile

from lxml import etree


OPF_NS = {"opf": "http://www.idpf.org/2007/opf"}

EXPECTED_SPINE = [
    "cover", "title", "foreword",
    "ch01", "ch02", "ch03", "ch04", "ch05", "ch06",
    "glossary",
]


def _open(sampler_epub_path):
    return zipfile.ZipFile(sampler_epub_path)


def test_mimetype_is_first_and_stored(sampler_epub_path):
    with _open(sampler_epub_path) as z:
        first = z.infolist()[0]
        assert first.filename == "mimetype"
        assert first.compress_type == zipfile.ZIP_STORED
        assert z.read("mimetype") == b"application/epub+zip"


def test_spine_order_matches_expectation(sampler_epub_path):
    with _open(sampler_epub_path) as z:
        opf = etree.fromstring(z.read("OEBPS/content.opf"))
    spine = [i.get("idref") for i in opf.findall(".//opf:itemref", OPF_NS)]
    assert spine == EXPECTED_SPINE


def test_all_xhtml_documents_parse(sampler_epub_path):
    with _open(sampler_epub_path) as z:
        xhtml_names = [n for n in z.namelist() if n.endswith(".xhtml")]
        assert len(xhtml_names) == len(EXPECTED_SPINE) + 1  # + nav.xhtml
        for name in xhtml_names:
            etree.fromstring(z.read(name))  # raises on malformed XML


def test_cover_and_illustration_are_real_pngs(sampler_epub_path):
    png_sig = b"\x89PNG\r\n\x1a\n"
    with _open(sampler_epub_path) as z:
        assert z.read("OEBPS/images/cover.png").startswith(png_sig)
        assert z.read("OEBPS/images/illustration.png").startswith(png_sig)


def test_chunks_in_target_range(sampler_epub_path):
    """The sampler is calibrated for ~10-13 chunks at the default max_tokens."""
    from src.core.epub.tag_preservation import TagPreserver
    from src.core.epub.html_chunker import HtmlChunker

    chunker = HtmlChunker(max_tokens=450)
    total = 0
    with _open(sampler_epub_path) as z:
        opf = etree.fromstring(z.read("OEBPS/content.opf"))
        manifest = {
            item.get("id"): item.get("href")
            for item in opf.findall(".//opf:item", OPF_NS)
        }
        for idref in EXPECTED_SPINE:
            href = manifest[idref]
            raw = z.read(f"OEBPS/{href}").decode("utf-8")
            root = etree.fromstring(raw.encode("utf-8"))
            body = root.find("{http://www.w3.org/1999/xhtml}body")
            body_html = etree.tostring(body, encoding="unicode", method="xml")
            text, tag_map = TagPreserver().preserve_tags(body_html)
            total += len(chunker.chunk_html_with_placeholders(text, tag_map))

    assert 10 <= total <= 14, f"sampler produced {total} chunks (expected 10-14)"


def test_spine_docs_fixture_exposes_chapters(sampler_spine_docs):
    assert set(sampler_spine_docs).issuperset(
        {"cover.xhtml", "ch01.xhtml", "ch04.xhtml", "glossary.xhtml"}
    )
    # ch04 carries the EPUB3 footnote markup we want to exercise downstream.
    assert 'epub:type="noteref"' in sampler_spine_docs["ch04.xhtml"]
    assert 'epub:type="footnote"' in sampler_spine_docs["ch04.xhtml"]
    # ch06 carries the figure/figcaption block.
    assert "<figure>" in sampler_spine_docs["ch06.xhtml"]
    assert "<figcaption>" in sampler_spine_docs["ch06.xhtml"]


def test_bytes_fixture_matches_path_fixture(sampler_epub_bytes, sampler_epub_path):
    assert sampler_epub_bytes == sampler_epub_path.read_bytes()
