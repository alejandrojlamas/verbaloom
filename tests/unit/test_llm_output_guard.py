from src.core.llm_output_guard import guard_llm_output


def test_guard_removes_wrapper_and_prompt_leak_lines_idempotently():
    raw = """# TEXT TO TRANSLATE
<TRANSLATIONATION>
Hola mundo.
</TRANSLATIONATION>
Return only the translated text.
"""

    first = guard_llm_output(raw, phase="translation")
    second = guard_llm_output(first.text, phase="translation")

    assert first.text == "Hola mundo."
    assert second.text == first.text
    assert {issue.code for issue in first.issues} >= {
        "llm_protocol_leak_cleaned",
    }
    assert "non_idempotent_output_cleanup" not in {issue.code for issue in second.issues}


def test_guard_rejects_residual_prompt_meta_inside_text():
    result = guard_llm_output(
        "Texto correcto.\n\nAssistant: internal note still visible.",
        phase="refinement",
    )

    assert "Assistant:" not in result.text
    assert any(issue.code == "llm_protocol_leak_cleaned" for issue in result.issues)


def test_guard_keeps_content_after_meta_prefix():
    result = guard_llm_output("Here is the translation: Bonjour le monde")

    assert result.text == "Bonjour le monde"
    assert any(issue.code == "llm_protocol_leak_cleaned" for issue in result.issues)


def test_guard_keeps_content_after_spanish_meta_prefix():
    result = guard_llm_output("Aquí está la traducción: El mundo siguió su curso.")

    assert result.text == "El mundo siguió su curso."
    assert any(issue.code == "llm_protocol_leak_cleaned" for issue in result.issues)


def test_guard_preserves_legitimate_acknowledgement_words_in_book_prose():
    raw = (
        "[[[TBLBLOCK000]]]\n"
        "Por supuesto, Sullivan sabía que el amor era la verdadera prueba.\n"
        "Claro, dijo ella, pero nadie abandonó la sala.\n"
        "Certainly, he knew the risk and continued.\n"
        "[[[/TBLBLOCK000]]]"
    )

    result = guard_llm_output(raw, phase="translation")

    assert result.text == raw
    assert not result.issues


def test_guard_removes_repeated_image_description_labels_idempotently():
    raw = "Descripcion de imagen: Descripción de imagen: La lámpara seguía encendida."

    first = guard_llm_output(raw, phase="translation")
    second = guard_llm_output(first.text, phase="translation")

    assert first.text == "La lámpara seguía encendida."
    assert second.text == first.text
    assert first.scores["reader_artifact_labels_removed"] == 2.0
    assert any(issue.code == "reader_artifact_label_cleaned" for issue in first.issues)
    assert not second.issues


def test_guard_reports_style_drift_without_rewriting_content():
    reference = (
        "Ella caminó despacio por la calle. Miró la ventana. Esperó una respuesta. "
        "La tarde era tranquila. Nadie habló durante un largo minuto. "
    ) * 8
    candidate = (
        "Primero: la estructura general del fenómeno requiere una observación amplia; "
        "segundo: el sistema de relaciones, aunque funcional, se comporta de manera "
        "desigual; tercero: el conjunto permanece estable bajo condiciones externas. "
    ) * 8

    result = guard_llm_output(candidate, phase="refinement", style_reference=reference)

    assert result.text == candidate.strip()
    assert result.scores["style_drift"] >= 0.0
    assert any(issue.code == "possible_style_drift" for issue in result.issues)


def test_guard_repairs_duplicated_apostrophized_prefix_before_audit():
    raw = "Luego vino [id8]L'L'Atalante[id9], seguida por d’d’Artagnan."

    first = guard_llm_output(raw, phase="translation")
    second = guard_llm_output(first.text, phase="translation")

    assert first.text == "Luego vino [id8]L'Atalante[id9], seguida por d’Artagnan."
    assert second.text == first.text


def test_guard_preserves_normal_apostrophized_names_and_phrases():
    raw = "O'Connor habló de L’Atalante y del rock 'n' roll."

    result = guard_llm_output(raw, phase="translation")

    assert result.text == raw
