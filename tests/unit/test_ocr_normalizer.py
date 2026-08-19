from src.utils.ocr_normalizer import analyze_ocr_artifacts, normalize_ocr_text


def test_detects_and_normalizes_scan_artifacts():
    text = """1
This paragraph was split in the mid-
dle of a sentence by a scanned source
and it keeps wrapping every line even
though it should read as one paragraph.

2
Another para-
graph has the same problem and also
has page numbers that should not stay
inside the final editable text."""

    is_scan, score, reasons = analyze_ocr_artifacts(text)
    result = normalize_ocr_text(text)

    assert is_scan is True
    assert score >= 3
    assert "hyphenated_line_breaks" in reasons
    assert result.is_likely_scan is True
    assert "middle of a sentence" in result.text
    assert "Another paragraph has" in result.text
    assert "\n1\n" not in f"\n{result.text}\n"
    assert "\n2\n" not in f"\n{result.text}\n"


def test_does_not_flatten_list_like_text():
    text = """- Keep the first item
- Keep the second item
- Keep the third item
- Keep the fourth item
- Keep the fifth item
- Keep the sixth item"""

    result = normalize_ocr_text(text)

    assert result.is_likely_scan is False
    assert "- Keep the first item\n- Keep the second item" in result.text


def test_preserves_likely_heading_as_separate_block():
    text = """5 Entrenamiento
Sección describe el régimen de entrenamiento para nuestros modelos
y el conjunto de datos utilizado en la evaluación.
También conserva el contexto sin convertirlo todo en un bloque."""

    result = normalize_ocr_text(text, force=True)

    assert "5 Entrenamiento\n\nSección describe" in result.text
    assert "Entrenamiento Sección describe" not in result.text


def test_preserves_formula_like_line_as_separate_block_and_removes_boxes():
    text = """Optimizador
Usamos el optimizador Adam con β1 = 0,9, β2 = 0,98 y ε = 10■■.
lrate = d_model^-0.5 * min(num_pasos^-0.5, num_pasos * warmup^-1.5)
Variamos la tasa de aprendizaje durante el entrenamiento."""

    result = normalize_ocr_text(text, force=True)

    assert "■" not in result.text
    assert "β1 = 0,9" in result.text
    assert "\n\nlrate = d_model^-0.5" in result.text
    assert "warmup^-1.5)\n\nVariamos" in result.text
