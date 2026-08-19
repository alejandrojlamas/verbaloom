import pytest

from src.core.adapters import TxtAdapter
from src.core.refine import txt_refiner
from src.utils.text_reader import read_text_file_with_fallbacks


def test_read_text_file_with_fallbacks_handles_cp1252(tmp_path):
    path = tmp_path / "book.txt"
    path.write_bytes("El café costó €5 y dijo “hola”.".encode("cp1252"))

    text, encoding = read_text_file_with_fallbacks(path)

    assert encoding == "cp1252"
    assert "café" in text
    assert "€5" in text
    assert "“hola”" in text


@pytest.mark.asyncio
async def test_txt_adapter_prepares_cp1252_input(tmp_path):
    source = tmp_path / "source.txt"
    output = tmp_path / "out.txt"
    source.write_bytes("Capítulo uno: café.\n\nSegundo párrafo.".encode("cp1252"))

    adapter = TxtAdapter(str(source), str(output), {"max_tokens_per_chunk": 500})

    assert await adapter.prepare_for_translation() is True
    assert adapter.input_encoding == "cp1252"
    units = adapter.get_translation_units()
    assert len(units) == 1
    assert "Capítulo uno" in units[0].content


@pytest.mark.asyncio
async def test_txt_refiner_reads_cp1252_input(tmp_path, monkeypatch):
    source = tmp_path / "translated.txt"
    output = tmp_path / "refined.txt"
    source.write_bytes("Texto ya traducido: café.".encode("cp1252"))
    seen = {}

    async def fake_refine_text_content(**kwargs):
        seen["translated_text"] = kwargs["translated_text"]
        output.write_text("ok", encoding="utf-8")
        return True

    monkeypatch.setattr(txt_refiner, "refine_text_content", fake_refine_text_content)

    ok = await txt_refiner.refine_txt_file(
        str(source),
        str(output),
        target_language="Spanish",
    )

    assert ok is True
    assert "café" in seen["translated_text"]
