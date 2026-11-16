"""Recover source paragraph breaks without accepting arbitrary text cuts."""

import json
from zipfile import ZipFile

from lxml import etree
import pytest

from src.core.epub import paragraph_alignment as alignment
from src.core.epub.dom_boundaries import (
    _BLOCK_XPATH, _parse_xhtml, _slot_value, _text_slots, audit_epub_dom_boundaries,
)
from tests.unit.epub.test_paragraph_alignment import Provider, _book


SOURCE = (
    "<p>The captain said, 'Please examine the revised diagrams before the meeting.'</p>"
    "<p>'I studied the diagrams already, and I found no errors in the design.'</p>"
    "<p>'Then tell the officers that the technical review can proceed tomorrow.'</p>"
)
TARGET = (
    "<p>El capitán dijo:</p>"
    "<p>—Revise los planos actualizados antes de la reunión.</p>"
    "<p>—Ya estudié los planos y no encontré errores en el diseño."
    "—Entonces informe a los oficiales que la revisión técnica puede continuar mañana.</p>"
)
ANCHORS = ["—Ya estudié los planos", "—Entonces informe a los oficiales"]


def _blocks(text):
    return _parse_xhtml(("<html><body>" + text + "</body></html>").encode()).xpath(_BLOCK_XPATH)


@pytest.mark.asyncio
@pytest.mark.parametrize("emphasized_start", [False, True])
async def test_recovers_glued_paragraph_and_replays_exact_result_without_more_requests(tmp_path, emphasized_start):
    source_text, target_text = SOURCE, TARGET
    if emphasized_start:
        source_text = source_text.replace("<p>'Then", "<p><em>'Then").replace("tomorrow.'</p>", "tomorrow.'</em></p>")
        target_text = target_text.replace("<p>—Ya", "<p><em>—Ya").replace("mañana.</p>", "mañana.</em></p>")
    source = _book(tmp_path / "source.epub", source_text)
    target = _book(tmp_path / "target.epub", target_text)
    original_zip = target.read_bytes()
    original_stream = alignment._flatten(_blocks(target_text))
    provider = Provider(json.dumps(ANCHORS))
    cache = tmp_path / "alignment.json"
    assert not audit_epub_dom_boundaries(source, target).clean
    assert await alignment.repair_epub_paragraph_alignment(source, target, config={}, provider=provider, cache_path=cache) == 1
    assert provider.calls == 1
    assert audit_epub_dom_boundaries(source, target).clean
    with ZipFile(target) as book:
        corrected_xml = book.read("chapter.xhtml")
        corrected = _parse_xhtml(corrected_xml).xpath(_BLOCK_XPATH)
        # Exact slice concatenation, including whitespace and punctuation.
        assert "".join(_slot_value(slot) for block in corrected for slot in _text_slots(block)) == original_stream
        assert "".join(corrected[1].itertext()).endswith("diseño.")
        assert "".join(corrected[2].itertext()).startswith("—Entonces informe")
        assert bool(corrected[2].find("em") is not None) == emphasized_start
        assert book.read("image.jpg") == b"opaque image"
    good_zip = target.read_bytes()
    assert await alignment.repair_epub_paragraph_alignment(source, target, config={}, provider=provider, cache_path=cache) == 0
    assert target.read_bytes() == good_zip and provider.calls == 1
    target.write_bytes(original_zip)
    assert await alignment.repair_epub_paragraph_alignment(source, target, config={}, provider=provider, cache_path=cache) == 1
    assert provider.calls == 1
    with ZipFile(target) as book:
        assert book.read("chapter.xhtml") == corrected_xml


@pytest.mark.parametrize("punctuation", [".", "?", "!", "…", ":", ";", "。", "？", "！", "：", "；"])
def test_unique_missing_space_anchor_requires_source_paragraph_start(punctuation):
    text = "Earlier passage" + punctuation + "—Next complete passage."
    response = json.dumps(["—Next complete passage."])
    with pytest.raises(ValueError):
        alignment._find_cuts(text, response, 1)
    assert alignment._find_cuts(text, response, 1, paragraph_starts={0}) == [0, text.index("—Next"), len(text)]


@pytest.mark.parametrize(("text", "anchor"), [
    ("Earlier.Next passage. Later Next passage.", "Next passage."),
    ("Earlier.Next passage. Later.Next passage.", "Next passage."),
    ("Someotherwords remain.", "otherwords remain."),
    ("Compound-otherwords remain.", "otherwords remain."),
    ("The result is 3.14 units.", "14 units."),
    ("Consult https://example.org/First.Last for details.", "Last for details."),
    ("Contact First.Last@example.org for details.", "Last@example.org"),
    ("Read ../First.Last for details.", "Last for details."),
])
def test_ambiguous_intraword_hyphen_decimal_and_literal_cuts_remain_rejected(text, anchor):
    with pytest.raises(ValueError):
        alignment._find_cuts(text, json.dumps([anchor]), 1, paragraph_starts={0})


@pytest.mark.asyncio
async def test_inline_emphasis_does_not_receive_paragraph_boundary_permission():
    source = _blocks(
        "<p>The scientist completed the earlier analysis and recorded every observation."
        "<em>The report includes the complete technical findings and the final recommendation.</em></p>"
    )
    target = _blocks(
        "<p>La científica terminó el análisis previo y registró todas las observaciones."
        "<em>El informe incluye los hallazgos técnicos completos y la recomendación final.</em></p>"
    )
    before = etree.tostring(target[0])
    with pytest.raises(ValueError):
        await alignment._realign(source, target, None, 30, cached_response=json.dumps(["El informe incluye"]))
    assert etree.tostring(target[0]) == before


@pytest.mark.asyncio
async def test_failed_validation_restores_every_original_slot():
    source, target = _blocks(SOURCE), _blocks(TARGET)
    before = [etree.tostring(block) for block in target]
    assert not await alignment._realign(source, target, None, 30, cached_response=json.dumps(ANCHORS), expected_result="invalid-result")
    assert [etree.tostring(block) for block in target] == before


@pytest.mark.asyncio
async def test_missing_passage_cannot_be_replaced_with_a_paragraph_boundary(tmp_path):
    source = _book(tmp_path / "source.epub", SOURCE)
    missing = TARGET.replace("—Ya estudié los planos y no encontré errores en el diseño.", "")
    target = _book(tmp_path / "target.epub", missing)
    before = target.read_bytes()
    assert await alignment.repair_epub_paragraph_alignment(source, target, config={}, provider=Provider(json.dumps(ANCHORS))) == 0
    assert target.read_bytes() == before
