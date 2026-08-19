import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.apply import (
    apply_term_correction_to_file,
    apply_term_correction_to_text,
)


def test_text_correction_uses_word_boundaries():
    updated, count = apply_term_correction_to_text(
        "Adam optimizes the model. Madam Curie is unrelated. Adam wins.",
        "Adam",
        "Adan",
    )

    assert count == 2
    assert "Adan optimizes" in updated
    assert "Madam Curie" in updated


def test_text_file_correction_creates_copy_by_default(tmp_path):
    source = tmp_path / "book.txt"
    source.write_text("BLEU was translated as BLEU.", encoding="utf-8")

    result = apply_term_correction_to_file(source, "BLEU", "BLEU score")

    assert result["supported"] is True
    assert result["replacements"] == 2
    assert result["output_filename"] != source.name
    assert source.read_text(encoding="utf-8") == "BLEU was translated as BLEU."
    corrected = tmp_path / result["output_filename"]
    assert corrected.read_text(encoding="utf-8") == "BLEU score was translated as BLEU score."


def test_pdf_correction_is_reported_unsupported(tmp_path):
    source = tmp_path / "book.pdf"
    source.write_bytes(b"%PDF-1.4")

    result = apply_term_correction_to_file(source, "old", "new")

    assert result["supported"] is False
    assert result["replacements"] == 0
    assert "PDF" in result["error"]
