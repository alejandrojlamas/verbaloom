"""Replaying an alignment must never turn actual omission into publication."""

from zipfile import ZipFile

import pytest
from lxml import etree

from src.core.epub.paragraph_alignment import repair_epub_paragraph_alignment
from src.core.epub.publication_gate import audit_epub_publication
from tests.unit.epub.test_paragraph_alignment import Provider, SOURCE, TARGET, _book
from tests.unit.epub.test_publication_gate import _write_epub


class NoModelCalls:
    async def generate(self, *_args, **_kwargs):
        raise AssertionError("Missing text must not be filled by cached alignment")


def _gate(source, output):
    # Structure and content checks are real; the external EPUBCheck subprocess
    # is stubbed, as in the existing publication-gate fixtures.
    return audit_epub_publication(
        source, output, source_language="English", target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )


def _omitted_paragraph_book(source, output):
    _write_epub(source, language="en", html_language="en", paragraphs=[
        "Mary refused the offer.",
        "John accepted the invitation and walked through the garden. He stopped "
        "beside the gate and watched the children playing near the old fountain.",
    ])
    _write_epub(output, language="es", html_language="es", paragraphs=[
        "",
        "Juan aceptó la invitación y atravesó el jardín. Se detuvo junto a la verja "
        "y observó a los niños que jugaban cerca de la vieja fuente.",
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_kind", ["none", "malformed", "unrelated"])
async def test_missing_paragraph_remains_unpublishable(tmp_path, cache_kind):
    source, output = tmp_path / "source.epub", tmp_path / "output.epub"
    cache = tmp_path / "alignment.json" if cache_kind != "none" else None
    if cache_kind == "malformed":
        cache.write_text('{"not": "an accepted alignment record"}')
    elif cache_kind == "unrelated":
        _book(source, SOURCE)
        _book(output, TARGET)
        provider = Provider('["Compra lo que quieras.", "Esperaron el tren nocturno."]')
        assert await repair_epub_paragraph_alignment(
            source, output, config={}, provider=provider, cache_path=cache,
        ) == 1
        assert provider.calls == 1
        assert cache.exists()

    _omitted_paragraph_book(source, output)
    before = output.read_bytes()
    assert not _gate(source, output).publishable
    assert await repair_epub_paragraph_alignment(
        source, output, config={}, provider=NoModelCalls(), cache_path=cache,
    ) == 0

    assert output.read_bytes() == before
    report = _gate(source, output)
    assert not report.publishable
    assert any("non-empty source block became empty" in error for error in report.errors)


@pytest.mark.asyncio
async def test_cached_alignment_cannot_fill_new_omission_in_same_source(tmp_path):
    source, output = tmp_path / "source.epub", tmp_path / "output.epub"
    cache = tmp_path / "alignment.json"
    source_paragraphs = ["".join(p.itertext()) for p in etree.fromstring("<body>" + SOURCE + "</body>")]
    target_paragraphs = ["".join(p.itertext()) for p in etree.fromstring("<body>" + TARGET + "</body>")]
    _write_epub(source, language="en", html_language="en", paragraphs=source_paragraphs)
    _write_epub(output, language="es", html_language="es", paragraphs=target_paragraphs)
    provider = Provider('["Compra lo que quieras.", "Esperaron el tren nocturno."]')
    assert await repair_epub_paragraph_alignment(
        source, output, config={}, provider=provider, cache_path=cache,
    ) == 1
    assert provider.calls == 1
    original_source = source.read_bytes()

    _write_epub(output, language="es", html_language="es", paragraphs=[
        "",
        "Compra lo que quieras. Cargalo a la habitacion y traeme el recibo.",
        "Esperaron el tren nocturno. Volvia a llegar tarde.",
    ])
    before = output.read_bytes()
    assert not _gate(source, output).publishable
    assert await repair_epub_paragraph_alignment(
        source, output, config={}, provider=NoModelCalls(), cache_path=cache,
    ) == 0
    assert source.read_bytes() == original_source
    assert output.read_bytes() == before
    assert not _gate(source, output).publishable


@pytest.mark.asyncio
async def test_declined_alignment_keeps_original_emphasis_and_text(tmp_path):
    source = _book(tmp_path / "source.epub", (
        "<p>The engineer opened the valve slowly. Water began to fill the lower tank at once.</p>"
        "<p>Nobody in the <em>control room</em> spoke. The gauges climbed and then settled again.</p>"
        "<p>By evening the whole system was running. They wrote the numbers in the log.</p>"
    ))
    output = _book(tmp_path / "output.epub", (
        "<p>Abrió la válvula.</p><p>El agua empezó a llenar el depósito inferior de inmediato. "
        "Nadie habló en la <em>sala de control</em>.</p><p>Los indicadores subieron y luego se "
        "estabilizaron. Al anochecer todo el sistema funcionaba. Anotaron las cifras en el registro.</p>"
    ))
    before = output.read_bytes()
    cache = tmp_path / "alignment.json"
    assert await repair_epub_paragraph_alignment(
        source, output, config={}, provider=Provider("[]"), cache_path=cache,
    ) == 0
    assert output.read_bytes() == before
    with ZipFile(output) as book:
        root = etree.fromstring(book.read("chapter.xhtml"))
    assert ["".join(node.itertext()) for node in root.iter("em")] == ["sala de control"]
