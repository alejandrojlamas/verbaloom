"""Tests for PDF text extraction and the PDF adapter."""

from pathlib import Path

import pytest

from src.core.adapters import PdfAdapter, TranslationUnit
from src.core.pdf import extract_pdf_text, extract_pdf_text_from_bytes
from src.utils.file_detector import detect_file_type, detect_file_type_by_content
from src.utils.text_encoding import UTF8_BOM


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_minimal_pdf(path: Path, text: str) -> None:
    """Write a small valid text PDF that pypdf can extract from."""
    stream = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    content = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode("ascii") + obj + b"\nendobj\n"

    xref_offset = len(content)
    content += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets[1:]:
        content += f"{offset:010d} 00000 n \n".encode("ascii")
    content += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    path.write_bytes(content)


def test_extract_pdf_text_from_path_and_bytes(tmp_path):
    pdf_path = tmp_path / "book.pdf"
    write_minimal_pdf(pdf_path, "Hello from a PDF document.")

    assert "Hello from a PDF document." in extract_pdf_text(pdf_path)
    assert "Hello from a PDF document." in extract_pdf_text_from_bytes(pdf_path.read_bytes())


def test_detect_file_type_recognizes_pdf_extension_and_magic(tmp_path):
    pdf_path = tmp_path / "book.pdf"
    renamed_path = tmp_path / "book.upload"
    write_minimal_pdf(pdf_path, "Detectable PDF text.")
    renamed_path.write_bytes(pdf_path.read_bytes())

    assert detect_file_type(str(pdf_path)) == "pdf"
    assert detect_file_type_by_content(str(renamed_path)) == "pdf"
    assert detect_file_type(str(renamed_path)) == "pdf"


@pytest.mark.asyncio
async def test_pdf_adapter_prepares_units_and_reconstructs_text(tmp_path):
    pdf_path = tmp_path / "source.pdf"
    output_path = tmp_path / "translated.txt"
    write_minimal_pdf(pdf_path, "Hello from a PDF document.")

    adapter = PdfAdapter(
        input_file_path=str(pdf_path),
        output_file_path=str(output_path),
        config={"max_tokens_per_chunk": 500},
    )

    assert await adapter.prepare_for_translation() is True
    assert adapter.format_name == "pdf"

    units = adapter.get_translation_units()
    assert len(units) == 1
    assert all(isinstance(unit, TranslationUnit) for unit in units)
    assert "Hello from a PDF document." in units[0].content

    assert await adapter.save_unit_translation(units[0].unit_id, "Hola desde un PDF.")
    output_bytes = await adapter.reconstruct_output()
    assert output_bytes.startswith(UTF8_BOM)
    output_text = output_bytes.decode("utf-8-sig")
    assert output_text == "Hola desde un PDF."


@pytest.mark.asyncio
async def test_pdf_adapter_reports_no_extractable_text(tmp_path):
    pdf_path = tmp_path / "blank.pdf"
    output_path = tmp_path / "translated.txt"
    write_minimal_pdf(pdf_path, "")

    adapter = PdfAdapter(
        input_file_path=str(pdf_path),
        output_file_path=str(output_path),
        config={"max_tokens_per_chunk": 500},
    )

    assert await adapter.prepare_for_translation() is False
    assert adapter.get_translation_units() == []
    assert adapter.last_error
    assert "PDF" in adapter.last_error
