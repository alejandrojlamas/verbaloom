from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from src.core.epub.dom_boundaries import audit_epub_dom_boundaries
from src.core.epub.paragraph_alignment import repair_epub_paragraph_alignment


SOURCE = (
    "<p>She nodded. I will need to buy a few things before I leave this city.</p>"
    "<p>Buy what you want. Charge it to the room and bring me the receipt.</p>"
    "<p>They waited for the overnight train. It was late again.</p>"
)
TARGET = (
    "<p>Asintio.</p>"
    "<p>Tendre que comprar algunas cosas antes de irme de esta ciudad.</p>"
    "<p>Compra lo que quieras. Cargalo a la habitacion y traeme el recibo. "
    "Esperaron el tren nocturno. Volvia a llegar tarde.</p>"
)


def _book(path, body):
    with ZipFile(path, "w") as book:
        book.writestr("chapter.xhtml", "<html><body>" + body + "</body></html>")
        book.writestr("image.jpg", b"opaque image")
    return path


class Provider:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    async def generate(self, *_args, **_kwargs):
        self.calls += 1
        return SimpleNamespace(content=self.answer)


@pytest.mark.asyncio
async def test_repairs_split_and_merge_without_rewriting_words(tmp_path):
    source = _book(tmp_path / "source.epub", SOURCE)
    target = _book(tmp_path / "target.epub", TARGET)
    llm = Provider('["Compra lo que quieras.", "Esperaron el tren nocturno."]')
    assert not audit_epub_dom_boundaries(source, target).clean
    assert await repair_epub_paragraph_alignment(source, target, config={}, provider=llm) == 1
    assert llm.calls == 1
    assert audit_epub_dom_boundaries(source, target).clean
    with ZipFile(target) as book:
        assert b"Asintio. Tendre que comprar" in book.read("chapter.xhtml")
        assert book.read("image.jpg") == b"opaque image"
    assert await repair_epub_paragraph_alignment(source, target, config={}, provider=llm) == 0
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [
    '["Invented replacement.", "Esperaron el tren nocturno."]',
    '["Esperaron el tren nocturno.", "Compra lo que quieras."]',
    '["Tendre que comprar", "Compra lo que quieras."]',
])
async def test_rejects_unverified_alignment_without_touching_epub(tmp_path, answer):
    source = _book(tmp_path / "source.epub", SOURCE)
    target = _book(tmp_path / "target.epub", TARGET)
    before = target.read_bytes()
    assert await repair_epub_paragraph_alignment(source, target, config={}, provider=Provider(answer)) == 0
    assert target.read_bytes() == before


@pytest.mark.asyncio
async def test_does_not_move_linked_or_identified_paragraph(tmp_path):
    source = _book(tmp_path / "source.epub", SOURCE.replace("<p>", '<p id="part">', 1))
    target = _book(tmp_path / "target.epub", TARGET.replace("<p>", '<p id="part">', 1))
    llm = Provider("[]")
    assert await repair_epub_paragraph_alignment(source, target, config={}, provider=llm) == 0
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_accepts_fenced_json_after_bounded_retry(tmp_path):
    source = _book(tmp_path / "source.epub", SOURCE)
    target = _book(tmp_path / "target.epub", TARGET)

    class RetryProvider(Provider):
        async def generate(self, prompt, **kwargs):
            if self.calls:
                assert "previous alignment was not accepted" in prompt
                self.answer = '```json\n["Compra lo que quieras.", "Esperaron el tren nocturno."]\n```'
            return await super().generate(prompt, **kwargs)

    llm = RetryProvider('["not an anchor"]')
    assert await repair_epub_paragraph_alignment(source, target, config={}, provider=llm) == 1
    assert llm.calls == 2
