from lxml import etree

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
