"""Text extraction for PDF inputs.

PDF translation in this app is text-first: we extract readable text and
produce a translated text output. Layout-preserving PDF reconstruction would
need a separate rendering/OCR pipeline and is intentionally not attempted here.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import BinaryIO

from src.core.document_structure import DocumentStructureReport, normalize_document_structure
from src.core.layout_sanitizer import sanitize_extracted_pages, sanitize_extracted_text
from src.utils.text_encoding import remove_pdf_toc_dot_leaders


class PDFTextExtractionError(ValueError):
    """Raised when a PDF cannot provide usable text."""


def _normalize_pdf_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    text, _report = sanitize_extracted_text(
        remove_pdf_toc_dot_leaders(text),
        source_type="pdf",
    )
    text, _structure_report = normalize_document_structure(text, source_type="pdf")
    return text.strip()


def _extract_from_stream(
    stream: BinaryIO,
    *,
    max_pages: int | None = None,
) -> tuple[str, DocumentStructureReport]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise PDFTextExtractionError(
            "PDF support requires the 'pypdf' package. Run: pip install pypdf"
        ) from exc

    try:
        reader = PdfReader(stream)
    except Exception as exc:
        raise PDFTextExtractionError(f"Could not read PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            if reader.decrypt("") == 0:
                raise PDFTextExtractionError("Encrypted PDF requires a password")
        except PDFTextExtractionError:
            raise
        except Exception as exc:
            raise PDFTextExtractionError(f"Could not decrypt PDF: {exc}") from exc

    parts: list[str] = []
    pages = reader.pages
    limit = len(pages) if max_pages is None else min(max_pages, len(pages))
    for index in range(limit):
        try:
            page_text = pages[index].extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            parts.append(page_text)

    text, _report = sanitize_extracted_pages(parts, source_type="pdf")
    text, structure_report = normalize_document_structure(text, source_type="pdf")
    text = text.strip()
    if not text:
        raise PDFTextExtractionError(
            "PDF has no extractable text. Scanned/image-only PDFs need OCR first."
        )
    return text, structure_report


def extract_pdf_text(pdf_path: str | Path, *, max_pages: int | None = None) -> str:
    """Extract readable text from a PDF file path."""
    path = Path(pdf_path)
    with path.open("rb") as fh:
        text, _structure_report = _extract_from_stream(fh, max_pages=max_pages)
        return text


def extract_pdf_text_with_structure(
    pdf_path: str | Path,
    *,
    max_pages: int | None = None,
) -> tuple[str, DocumentStructureReport]:
    """Extract readable PDF text and return a compact structure report."""
    path = Path(pdf_path)
    with path.open("rb") as fh:
        return _extract_from_stream(fh, max_pages=max_pages)


def extract_pdf_text_from_bytes(file_data: bytes, *, max_pages: int | None = None) -> str:
    """Extract readable text from PDF bytes."""
    text, _structure_report = _extract_from_stream(io.BytesIO(file_data), max_pages=max_pages)
    return text
