from src.core.post_processor import clean_translated_text
from src.core.pdf.text_extractor import _normalize_pdf_text
from src.utils.text_encoding import remove_pdf_toc_dot_leaders


def test_removes_compact_toc_dot_leaders_before_page_numbers():
    text = (
        "Capítulo XIV\n"
        "Donde se ponen los versos desesperados del difunto pastor..... 69\n"
        "Capítulo XV\n"
        "Donde se cuenta la desgraciada aventura........................................... 73"
    )

    cleaned = remove_pdf_toc_dot_leaders(text)

    assert "....." not in cleaned
    assert "................................" not in cleaned
    assert "difunto pastor 69" in cleaned
    assert "aventura 73" in cleaned


def test_removes_spaced_toc_dot_leaders_and_standalone_page_leader_lines():
    text = (
        ". . . . . . . . . . . . . . . . . . . . . 62\n"
        "Capítulo XIII\n"
        "Donde se da fin al cuento . . . . . . . . . . . . . . . . . . . . . . 66"
    )

    cleaned = remove_pdf_toc_dot_leaders(text)

    assert "62" not in cleaned.splitlines()[0]
    assert ". . ." not in cleaned
    assert "Capítulo XIII\nDonde se da fin al cuento 66" in cleaned


def test_preserves_normal_ellipsis_and_decimal_values():
    text = "Pensó... y calló.\nLa constante vale 3.14159 en este ejemplo."

    cleaned = remove_pdf_toc_dot_leaders(text)

    assert cleaned == text


def test_pdf_extraction_normalization_removes_toc_leaders():
    raw = "Capítulo XVI\nDe lo que le sucedió al hidalgo............... 76\n\nTexto real."

    cleaned = _normalize_pdf_text(raw)

    assert "........" not in cleaned
    assert "hidalgo 76" in cleaned


def test_post_processor_removes_llm_reintroduced_toc_leaders():
    raw = (
        "Capítulo XVII\n"
        "Sancho Panza pasaron en la venta........................\n"
        "</TRANSLATIONATION>"
    )

    cleaned = clean_translated_text(raw)

    assert "........................" not in cleaned
    assert "</TRANSLATION" not in cleaned
    assert "Sancho Panza pasaron en la venta" in cleaned
