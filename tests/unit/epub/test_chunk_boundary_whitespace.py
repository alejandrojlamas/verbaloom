import pytest

from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.text_splitter import TextSplitter
from src.core.epub.xhtml_translator import _reconstruct_html, PlaceholderManager


class CharacterCounter:
    def count_tokens(self, text):
        return len(text)


@pytest.mark.parametrize(
    'content',
    [
        'Alpha.  Beta?\n\tGamma! Delta. ' * 4,
        'One;  two:\tthree,\nfour,\u00a0five ' * 4,
        'Uno\n\nDos\r\nTres\n' * 7,
        'uno\t   dos tres\u00a0cinco\t' * 7,
        '水温は高い。別の段落。' * 8,
    ],
)
def test_every_split_strategy_preserves_all_source_characters(content):
    splitter = TextSplitter(19, CharacterCounter())
    chunks = splitter.split_oversized_segment(content)
    assert len(chunks) > 1
    assert ''.join(chunks) == content


def test_real_chunker_retains_sentence_separators_inside_and_between_requests():
    html = '<p>' + 'Winter ends. Spring begins. ' * 48 + '</p>'
    preserver = TagPreserver()
    text, tags = preserver.preserve_tags(html)
    chunks = HtmlChunker(max_tokens=35).chunk_html_with_placeholders(text, tags)
    assert len(chunks) > 1
    translated = [
        PlaceholderManager.restore_to_global(chunk['text'], chunk['global_indices'])
        for chunk in chunks
    ]
    assert ''.join(translated) == text
    expected = html.replace('Winter ends.', 'Termina el invierno.').replace('Spring begins.', 'Llega la primavera.')
    responses = [chunk.strip().replace('Winter ends.', 'Termina el invierno.').replace('Spring begins.', 'Llega la primavera.') for chunk in translated]
    assert _reconstruct_html(responses, tags, preserver, original_chunks=chunks) == expected


@pytest.mark.parametrize(
    ('source', 'responses', 'expected'),
    [
        (['Uno.\u00a0\t', 'Dos.'], ['One.', 'Two.'], 'One.\u00a0\tTwo.'),
        (['abcdef', 'ghijk'], ['abcdef', 'ghijk'], 'abcdefghijk'),
        (['東京', '大学'], ['東京', '大学'], '東京大学'),
        (['bien', ' colocado'], ['bien ', 'colocado'], 'bien colocado'),
    ],
)
def test_only_proven_missing_chunk_separators_are_reinserted(source, responses, expected):
    chunks = [{'text': item} for item in source]
    assert _reconstruct_html(responses, {}, TagPreserver(), original_chunks=chunks) == expected


def test_forced_cut_keeps_adjacent_inline_placeholder_atomic():
    # No spaces for the force-splitter to use in this CJK passage.
    text = '東' * 15 + '[id42]' + '京' * 32
    tags = {'[id42]': '<em>'}
    chunks = HtmlChunker(max_tokens=8).chunk_html_with_placeholders(text, tags)
    assert [i for chunk in chunks for i in chunk['global_indices']] == [42]
    assert ''.join(
        PlaceholderManager.restore_to_global(chunk['text'], chunk['global_indices'])
        for chunk in chunks
    ) == text


def test_source_backed_marks_survive_reconstruction_only_in_their_own_slots():
    preserver = TagPreserver()
    markup = "<p>He paused. ... .</p><p>First ■ then □.</p><p>Last.</p>"
    source, tags = preserver.preserve_tags(markup)
    chunks = HtmlChunker(max_tokens=500).chunk_html_with_placeholders(source, tags)
    candidate = source.replace("He paused", "Se detuvo").replace("First ■ then □.", "Primero ■ y luego.").replace("Last.", "Fin■. Final ..... ruido")
    assert _reconstruct_html([candidate], tags, preserver, original_chunks=chunks) == (
        "<p>Se detuvo. ... .</p><p>Primero ■ y luego.</p><p>Fin. Final ruido</p>"
    )
