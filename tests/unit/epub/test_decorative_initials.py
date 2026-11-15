from lxml import etree
import pytest

from src.core.epub.body_serializer import extract_body_html, replace_body_content
from src.core.epub.dom_boundaries import compare_xhtml_boundaries
from src.core.epub.epub_translation_adapter import EpubTranslationAdapter


def test_heading_and_drop_cap_reach_the_model_as_whole_words_and_keep_markup():
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    original = etree.XML(
        b'<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        b'<h3><a id="page13"/>O<span class="small">BJECTIVE</span> '
        b'A<span class="small">CTIONS</span></h3>'
        b'<p><span class="drop-cap">W</span>hen the actor <em>moves</em>.</p></body></html>'
    )
    target = etree.fromstring(etree.tostring(original))
    adapter = EpubTranslationAdapter()
    html, context = adapter.extract_content(target, None)
    assert "OBJECTIVE ACTIONS" in html
    assert "When the actor" in html
    assert "O<span" in extract_body_html(original)[0]
    masked, tags, _ = adapter.preserve_structure(html, context, None)
    assert "OBJECTIVE ACTIONS" in masked
    assert "When the actor" in masked
    rendered = adapter.tag_preserver.restore_tags(
        masked.replace("OBJECTIVE ACTIONS", "ACCIONES OBJETIVAS").replace(
            "When the actor", "Cuando la actriz"
        ).replace("moves", "se mueve"),
        tags,
    )
    replace_body_content(context["body_element"], rendered)
    assert "".join(target.find(".//x:h3", ns).itertext()) == "ACCIONES OBJETIVAS"
    assert "".join(target.find(".//x:p", ns).itertext()) == "Cuando la actriz se mueve."
    assert target.find(".//x:a", ns).get("id") == "page13"
    assert target.find(".//x:em", ns).text == "se mueve"
    assert compare_xhtml_boundaries(original, target).clean


def test_does_not_join_intentional_spacing_or_move_semantic_markup():
    source = etree.XML(
        b'<html><body><h2>A <span class="small">REAL CHOICE</span></h2>'
        b'<h3>W<span class="small" id="anchor">ORD</span></h3>'
        b'<h3>W<span class="small">ORD</span> <em>emphasis</em></h3>'
        b'<p><span class="dropcap">A</span> choice.</p>'
        b'<p><span class="note">Q</span>uestion.</p></body></html>'
    )
    html, _ = extract_body_html(source, join_decorative_initials=True)
    assert 'A <span class="small">REAL CHOICE</span>' in html
    assert 'W<span class="small" id="anchor">ORD</span>' in html
    assert 'W<span class="small">ORD</span> <em>emphasis</em>' in html
    assert '<span class="dropcap"></span>A choice.' in html
    assert '<span class="note">Q</span>uestion.' in html


@pytest.mark.parametrize(
    "opening,source_text,target_text",
    [
        ('I<span class="smallcaps">N THE MORNING</span>', "IN THE MORNING", "POR LA MANANA"),
        ('I <span class="smallcaps">WAS AWAKE</span>', "I WAS AWAKE", "ESTABA DESPIERTO"),
        ('A <span class="small-caps">NEW DAY</span>', "A NEW DAY", "UN NUEVO DIA"),
        ('<a id="page1"/>W<span class="small2">HEN SHE ARRIVED</span>',
         "WHEN SHE ARRIVED", "CUANDO ELLA LLEGO"),
    ],
)
def test_paragraph_initial_and_smallcaps_share_editable_text(opening, source_text, target_text):
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    original = etree.XML(
        ('<html xmlns="http://www.w3.org/1999/xhtml"><body><p>' + opening
         + ', <em>quiet</em>.</p></body></html>').encode()
    )
    original_xml = etree.tostring(original)
    target = etree.fromstring(original_xml)
    adapter = EpubTranslationAdapter()
    html, context = adapter.extract_content(target, None)
    normalized = etree.XML(('<body xmlns="http://www.w3.org/1999/xhtml">' + html + '</body>').encode())
    assert ''.join(normalized.find('x:p', ns).itertext()) == ''.join(original.find('.//x:p', ns).itertext())
    assert etree.tostring(original) == original_xml
    masked, tags, _ = adapter.preserve_structure(html, context, None)
    assert source_text in masked
    rendered = adapter.tag_preserver.restore_tags(
        masked.replace(source_text, target_text).replace("quiet", "tranquilo"), tags,
    )
    replace_body_content(context["body_element"], rendered)
    paragraph = target.find('.//x:p', ns)
    assert ''.join(paragraph.itertext()) == target_text + ', tranquilo.'
    assert paragraph.find('x:span', ns).text == target_text
    assert paragraph.find('x:em', ns).text == 'tranquilo'
    if 'page1' in opening:
        assert paragraph.find('x:a', ns).get('id') == 'page1'
    assert compare_xhtml_boundaries(original, target).clean


@pytest.mark.parametrize("paragraph", [
    '<p>WORD<span class="smallcaps">FORM</span>.</p>',
    '<p>i<span class="smallcaps">TEM</span>.</p>',
    '<p>1<span class="smallcaps">ST</span>.</p>',
    '<p>I<span class="note">NTRO</span>.</p>',
    '<p>I<span class="smallcaps" id="semantic">NTRO</span>.</p>',
    '<p>I<span class="smallcaps" lang="en">NTRO</span>.</p>',
    '<p>I<span class="smallcaps"><em>NTRO</em></span>.</p>',
    '<p>I<span class="smallcaps special">NTRO</span>.</p>',
    '<pre><p>I<span class="smallcaps">NTRO</span>.</p></pre>',
    '<code><p>I<span class="smallcaps">NTRO</span>.</p></code>',
])
def test_paragraph_initial_normalization_leaves_semantic_and_noninitial_markup(paragraph):
    source = etree.XML(('<html><body>' + paragraph + '</body></html>').encode())
    assert extract_body_html(source, join_decorative_initials=True)[0] == extract_body_html(source)[0]
