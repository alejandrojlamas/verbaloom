"""Navigation aliases must not create duplicate chapters or flatten the TOC."""
from pathlib import Path

from lxml import etree
import pytest

from src.core.epub.publication_structure import (
    _navigation_entries, _write_ncx, _write_navigation, _NavigationEntry,
    _opening_heading, _resolve_navigation_target,
)


def _doc(body):
    return etree.fromstring(f'<html xmlns="http://www.w3.org/1999/xhtml"><body id="start">{body}</body></html>'.encode())


def _ncx(points):
    return ('<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/"><docTitle><text>Book</text></docTitle>'
            '<navMap>' + points + '</navMap></ncx>').encode()


def _point(id, src, label, children=""):
    return f'<navPoint id="{id}"><navLabel><text>{label}</text></navLabel><content src="{src}"/>{children}</navPoint>'


def test_body_heading_alias_is_one_entry_preserves_target_and_text(tmp_path):
    doc = _doc('<section><h1 id="heading"><span id="page"/>CAPÍTULO UNO</h1><p>Texto.</p></section>')
    ncx = tmp_path / "toc.ncx"
    ncx.write_bytes(_ncx(_point("p1", "chapter.xhtml#start", "Chapter One") +
                        _point("p2", "chapter.xhtml#heading", "CAPÍTULO UNO")))
    opf = etree.fromstring(b'''<package><manifest><item id="c" href="chapter.xhtml" media-type="application/xhtml+xml"/>
        <item id="n" href="toc.ncx" media-type="application/x-dtbncx+xml"/></manifest><spine toc="n"/></package>''')
    text_before = "".join(doc.itertext())
    entries, count = _navigation_entries(opf_root=opf, opf_dir=tmp_path, content_files=["chapter.xhtml"],
        parsed_xhtml_docs={str(tmp_path / "chapter.xhtml"): doc}, package_root=tmp_path,
        cover_page_href="cover.xhtml", target_language="Spanish", epub3=False)
    assert count == 1 and len(entries) == 1
    assert entries[0].target == "chapter.xhtml#start"
    assert entries[0].label == "CAPÍTULO UNO"
    assert "chapter.xhtml#heading" in entries[0].aliases
    assert "".join(doc.itertext()) == text_before
    kwargs = dict(href="toc.ncx", title="Libro", identifier="urn:test", entries=entries)
    _write_ncx(ncx, **kwargs)
    before = ncx.read_bytes()
    _write_ncx(ncx, **kwargs)
    assert ncx.read_bytes() == before
    assert etree.parse(str(ncx)).xpath("//*[local-name()='content']/@src") == ["chapter.xhtml#start"]


@pytest.mark.parametrize("prefix", [
    "<p>A preface comes first.</p>",
    "<div><span/></div>Actual prose before the heading.",
    "<!-- a comment -->Visible prose.",
    '<img src="illustration.jpg" alt=""/>',
    '<div><svg/></div>',
])
def test_visible_content_before_heading_is_not_an_alias(prefix):
    doc = _doc(prefix + '<h1 id="heading">Chapter</h1>')
    body = doc.xpath("//*[local-name()='body']")[0]
    heading = doc.xpath("//*[local-name()='h1']")[0]
    assert _opening_heading(body, {heading}) is None


def test_empty_anchor_and_heading_anchor_are_equivalent_but_later_anchor_is_not():
    doc = _doc('<a id="empty"/><h1 id="heading">Chapter</h1><p id="later">Later</p>')
    heading = doc.xpath("//*[@id='heading']")[0]
    assert _opening_heading(doc.xpath("//*[@id='empty']")[0], {heading}) is heading
    assert _opening_heading(doc.xpath("//*[@id='later']")[0], {heading}) is None


def test_nested_ncx_survives_deduplication_with_both_child_branches(tmp_path):
    path = tmp_path / "toc.ncx"
    path.write_bytes(_ncx(
        _point("part", "c.xhtml#start", "Part", _point("child1", "c.xhtml#one", "One")) +
        _point("alias", "c.xhtml#heading", "Part translated", _point("child2", "c.xhtml#two", "Two"))))
    entries = [_NavigationEntry("Parte", "c.xhtml#start", aliases=("c.xhtml#heading",)),
               _NavigationEntry("Uno", "c.xhtml#one"), _NavigationEntry("Dos", "c.xhtml#two")]
    _write_ncx(path, href="toc.ncx", title="Libro", identifier="urn:test", entries=entries)
    tree = etree.parse(str(path))
    assert len(tree.xpath("//*[local-name()='navMap']/*[local-name()='navPoint']")) == 1
    assert len(tree.xpath("//*[local-name()='navMap']/*/*[local-name()='navPoint']")) == 2
    assert tree.xpath("//*[local-name()='content']/@src") == ["c.xhtml#start", "c.xhtml#one", "c.xhtml#two"]
    assert tree.xpath("//*[local-name()='navPoint']/@playOrder") == ["1", "2", "3"]


def test_same_label_different_targets_remain_distinct(tmp_path):
    path = tmp_path / "toc.ncx"
    path.write_bytes(_ncx(_point("one", "c.xhtml#a", "Interlude") + _point("two", "c.xhtml#b", "Interlude")))
    entries = [_NavigationEntry("Interludio", f"c.xhtml#{anchor}") for anchor in ("a", "b")]
    _write_ncx(path, href="toc.ncx", title="Libro", identifier="urn:test", entries=entries)
    assert len(etree.parse(str(path)).xpath("//*[local-name()='navPoint']")) == 2


def test_epub3_nav_aliases_coalesce_without_losing_children(tmp_path):
    path = tmp_path / "nav.xhtml"
    path.write_text('''<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
    <head/><body><nav epub:type="toc"><ol>
    <li><a href="c.xhtml#start">Chapter</a><ol><li><a href="c.xhtml#one">One</a></li></ol></li>
    <li><a href="c.xhtml#heading">CAPÍTULO</a><ol><li><a href="c.xhtml#two">Two</a></li></ol></li>
    </ol></nav></body></html>''')
    entries = [_NavigationEntry("Capítulo", "c.xhtml#start", aliases=("c.xhtml#heading",)),
               _NavigationEntry("Uno", "c.xhtml#one"), _NavigationEntry("Dos", "c.xhtml#two")]
    _write_navigation(path, href="nav.xhtml", title="Libro", entries=entries,
                      cover_page_href="cover.xhtml", language_code="es")
    tree = etree.parse(str(path))
    toc = tree.xpath("//*[local-name()='nav' and @*[local-name()='type']='toc']")[0]
    assert toc.xpath(".//*[local-name()='a']/@href") == ["c.xhtml#start", "c.xhtml#one", "c.xhtml#two"]
    assert len(toc.xpath("./*[local-name()='ol']/*[local-name()='li']")) == 1


@pytest.mark.parametrize(("base", "href", "expected"), [
    ("Text/nav.xhtml", "#section", "Text/nav.xhtml#section"),
    ("Text/nav.xhtml", "../chapter%201.xhtml#part%202", "chapter%201.xhtml#part%202"),
    ("Text/nav.xhtml", "../chapter%23one.xhtml#part%2520", "chapter%23one.xhtml#part%2520"),
    ("Text/nav.xhtml", "https://example.com", ""),
    ("Text/nav.xhtml", "//example.com/page", ""),
])
def test_navigation_target_resolution(base, href, expected):
    assert _resolve_navigation_target(base, href) == expected


def test_inline_heading_emphasis_does_not_introduce_word_spaces():
    from src.core.epub.publication_structure import _element_text
    heading = etree.fromstring("<h1>Capí<em>tulo</em> uno</h1>".encode())
    assert _element_text(heading) == "Capítulo uno"


def test_heading_label_includes_visible_breaks():
    from src.core.epub.publication_structure import _element_text
    assert _element_text(etree.XML(b"<h1>Chapter<br/>One</h1>")) == "Chapter One"


def test_ncx_page_list_shares_the_order_of_an_existing_target(tmp_path):
    path = tmp_path / "toc.ncx"
    root = etree.XML(_ncx(_point("chapter", "c.xhtml#start", "Chapter")))
    page_list = etree.SubElement(root, "{http://www.daisy.org/z3986/2005/ncx/}pageList")
    page = etree.SubElement(page_list, "{http://www.daisy.org/z3986/2005/ncx/}pageTarget",
                            id="page", playOrder="3")
    etree.SubElement(page, "{http://www.daisy.org/z3986/2005/ncx/}content", src="c.xhtml#start")
    path.write_bytes(etree.tostring(root))
    _write_ncx(path, href="toc.ncx", title="Book", identifier="urn:test",
               entries=[_NavigationEntry("Chapter", "c.xhtml#start")])
    tree = etree.parse(str(path))
    assert tree.xpath("//*[@playOrder]/@playOrder") == ["1", "1"]
