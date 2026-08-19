"""
Unit tests for the stray angle-bracket escape helper.

Source EPUBs (especially Korean webnovels) often use literal `<Word>` markers
in text content as stylistic quotation, e.g. `<Skill>` or `<ItemName>` for
status windows. Calibre encodes these as `&lt;Word&gt;` in the XHTML, but
once parsed by lxml the text node contains real `<` and `>` characters. The
LLM passes them through, and without escaping they corrupt the reinjected
document because the XML parser treats them as phantom HTML tags.
"""
from src.core.epub.xhtml_translator import _escape_stray_angle_brackets


def test_escapes_simple_angle_brackets():
    assert _escape_stray_angle_brackets("<Cloud>") == "&lt;Cloud&gt;"


def test_escapes_brackets_around_korean_text():
    assert _escape_stray_angle_brackets("<엔젤릭>") == "&lt;엔젤릭&gt;"


def test_escapes_brackets_around_spanish_translation():
    assert _escape_stray_angle_brackets("<Ángelico>") == "&lt;Ángelico&gt;"


def test_leaves_placeholder_brackets_untouched():
    text = "[id0]Hola [id1]<Skill>[id2]"
    assert _escape_stray_angle_brackets(text) == "[id0]Hola [id1]&lt;Skill&gt;[id2]"


def test_preserves_existing_entities():
    # We do NOT touch '&', so if the LLM already produced proper entities they
    # stay intact instead of being double-escaped to &amp;lt;.
    assert _escape_stray_angle_brackets("&lt;Cloud&gt;") == "&lt;Cloud&gt;"


def test_handles_mixed_text_with_multiple_brackets():
    raw = "Las clientes entraron en <Cloud> con paso firme hacia <Angelico>"
    expected = "Las clientes entraron en &lt;Cloud&gt; con paso firme hacia &lt;Angelico&gt;"
    assert _escape_stray_angle_brackets(raw) == expected


def test_empty_string_returns_empty():
    assert _escape_stray_angle_brackets("") == ""


def test_text_without_brackets_unchanged():
    assert _escape_stray_angle_brackets("Hola mundo") == "Hola mundo"
