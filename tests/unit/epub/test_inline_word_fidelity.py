import pytest

from src.core.epub.tag_preservation import TagPreserver, is_non_translatable
from src.prompts.prompts import _literary_style_contract_section


@pytest.mark.parametrize("word", ["did", "civil", "mix", "I", "vi", "liv"])
@pytest.mark.parametrize("technical_mode", [False, True])
def test_roman_looking_words_stay_visible_inside_emphasis(word, technical_mode):
    html = f"<p>They <em>{word}</em> speak.</p>"
    preserver = TagPreserver(protect_technical=technical_mode)
    preserve = (preserver.preserve_tags_and_technical_content if technical_mode
                else preserver.preserve_tags)
    protected, mapping = preserve(html)
    assert word in protected
    assert all(word != value for value in mapping.values())
    assert preserver.restore_tags(protected, mapping) == html


@pytest.mark.parametrize("value", ["", "  ", "12", "(19)", "2-4", " 7. ", "\u200b"])
def test_actual_numbers_and_empty_furniture_remain_protected(value):
    assert is_non_translatable(value)


def test_chapter_number_is_visible_and_can_preserve_its_format():
    html = "<h1><em>XIV</em></h1><p><em>I</em> ran.</p>"
    preserver = TagPreserver()
    protected, mapping = preserver.preserve_tags(html)
    assert "XIV" in protected and "I ran" not in protected
    translated = protected.replace("I[", "Yo[").replace(" ran.", " corr\u00ed.")
    assert preserver.restore_tags(translated, mapping) == "<h1><em>XIV</em></h1><p><em>Yo</em> corr\u00ed.</p>"


@pytest.mark.parametrize("target", ["Spanish", "French", "Japanese"])
def test_literary_fidelity_instructions_are_not_tied_to_a_book_or_locale(target):
    block = _literary_style_contract_section(target)
    assert "speaker, addressee" in block and "intentional repetitions" in block
    assert "emphasis marks meaning, not word boundaries" in block
    assert _literary_style_contract_section(target, {"literary_style_contract": False}) == ""
