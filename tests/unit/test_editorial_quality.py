from src.core.editorial_quality import (
    EditorialQualityReport,
    apply_source_aware_guard_assessment,
    assess_refinement,
    build_source_aware_guard_prompt,
    editorial_report_path,
    infer_section_title,
    parse_source_aware_guard_response,
)


def test_accepts_small_editorial_improvement():
    draft = (
        "El modelo conserva la estructura original del documento.\n\n"
        "La revisión corrige puntuación menor sin cambiar el sentido."
    )
    refined = (
        "El modelo conserva la estructura original del documento.\n\n"
        "La revisión corrige la puntuación menor sin cambiar el sentido."
    )

    decision = assess_refinement(draft, refined, chunk_index=1, section="Capítulo 1")

    assert decision.accepted is True
    assert decision.rejections == []


def test_rejects_added_square_artifacts():
    decision = assess_refinement(
        "El texto ya estaba limpio y no contenía símbolos extraños.",
        "El texto ya estaba limpio■■■■ y no contenía símbolos extraños.",
        chunk_index=2,
    )

    assert decision.accepted is False
    assert "artifact_glyphs_added" in {issue.code for issue in decision.rejections}


def test_rejects_mojibake_regression():
    decision = assess_refinement(
        "La traducción automática conserva la atención del modelo.",
        "La traducciÃ³n automÃ¡tica conserva la atenciÃ³n del modelo.",
        chunk_index=3,
    )

    assert decision.accepted is False
    assert "mojibake_regression" in {issue.code for issue in decision.rejections}


def test_rejects_paragraph_collapse():
    draft = "Primer párrafo completo.\n\nSegundo párrafo completo.\n\nTercer párrafo completo."
    refined = "Primer párrafo completo. Segundo párrafo completo. Tercer párrafo completo."

    decision = assess_refinement(draft, refined, chunk_index=4)

    assert decision.accepted is False
    assert "paragraph_collapse" in {issue.code for issue in decision.rejections}


def test_rejects_lost_numbers_citations_and_formula_fragments():
    draft = (
        "Usamos Adam [18] con β1 = 0,9, β2 = 0,98 y ε = 10^-9.\n\n"
        "lrate = d_model^-0.5 * min(num_pasos^-0.5, num_pasos * warmup^-1.5)"
    )
    refined = "Usamos Adam con una tasa ajustada durante el entrenamiento."

    decision = assess_refinement(draft, refined, chunk_index=5)
    rejection_codes = {issue.code for issue in decision.rejections}

    assert decision.accepted is False
    assert "numbers_lost" in rejection_codes
    assert "citations_lost" in rejection_codes
    assert "formula_fragments_lost" in rejection_codes


def test_citations_are_not_treated_as_internal_placeholders():
    decision = assess_refinement(
        "El método se reportó previamente [18].",
        "El método se reportó previamente [19].",
        chunk_index=6,
    )
    issue_codes = {issue.code for issue in decision.issues}

    assert "citations_lost" in issue_codes
    assert "placeholder_mismatch" not in issue_codes


def test_rejects_lost_glossary_target_terms():
    decision = assess_refinement(
        "La autoatención conecta todas las posiciones.",
        "La atención propia conecta todas las posiciones.",
        chunk_index=7,
        glossary_terms={"self-attention": "autoatención"},
    )

    assert decision.accepted is False
    assert "glossary_terms_lost" in {issue.code for issue in decision.rejections}


def test_rejects_mexican_spanish_locale_regression():
    decision = assess_refinement(
        "Ustedes no se han rendido; desde sus confines, precipítense.",
        "Vosotras no os habéis rendido; desde vuestros confines, precipitaos.",
        chunk_index=8,
        target_language="Spanish",
        prompt_options={"spanish_variant": "mexican"},
    )

    assert decision.accepted is False
    assert "mexican_spanish_regression" in {issue.code for issue in decision.rejections}


def test_rejects_lost_asterisk_omission_marker():
    decision = assess_refinement(
        "No es mi intención entrar en la investigación. * * * Reina una confusión absoluta.",
        "No es mi intención entrar en la investigación. Reina una confusión absoluta.",
        chunk_index=9,
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "asterisk_omission_lost" in {issue.code for issue in decision.rejections}


def test_rejects_dialogue_opening_style_regression():
    decision = assess_refinement(
        "—Aparto mi cuerpo del sol. ¡Ah, Tashtego!",
        "“Aparto mi cuerpo del sol. ¡Ah, Tashtego!”",
        chunk_index=10,
        target_language="Spanish",
    )

    assert decision.accepted is False
    assert "dialogue_opening_style_regression" in {issue.code for issue in decision.rejections}


def test_inferrs_section_title_from_numbered_heading():
    text = "5 Entrenamiento\n\nSección describe el régimen de entrenamiento."

    assert infer_section_title(text) == "5 Entrenamiento"


def test_report_groups_decisions_by_section(tmp_path):
    report = EditorialQualityReport(document_name="paper.txt", target_language="Spanish")
    report.add(assess_refinement("Texto limpio.", "Texto limpio.", chunk_index=1, section="Resumen"))
    report.add(assess_refinement("Texto limpio.", "Texto■■ limpio.", chunk_index=2, section="Resumen"))

    report_path = report.write(editorial_report_path(tmp_path / "paper.txt"))
    markdown = report_path.read_text(encoding="utf-8")

    assert report_path.name == "paper - reporte editorial.md"
    assert "## Resumen" in markdown
    assert "Refinamientos rechazados: 1" in markdown
    assert "artifact_glyphs_added" in markdown


def test_source_aware_judge_can_accept_ocr_cleanup_rejected_locally():
    decision = assess_refinement(
        "EL LIBRO 01729*395\n\nSobrevivientes de una época dorada hablan.",
        "EL LIBRO\n\nSobrevivientes de una época dorada hablan.",
        chunk_index=1,
        source_text="EL LIBRO\n\nSurvivors of a golden age speak.",
        target_language="Spanish",
    )
    assert decision.accepted is False

    apply_source_aware_guard_assessment(
        decision,
        {
            "decision": "accept",
            "confidence": 0.91,
            "reason": "The removed numbers are OCR garbage not supported by the source.",
            "issues": [],
            "missing_from_source": [],
            "added_not_in_source": [],
            "weird_symbols": [],
            "structure_score": 0.9,
        },
        model="deepseek-v4-pro",
    )

    assert decision.accepted is True
    assert "source_aware_override_accept" in {issue.code for issue in decision.warnings}
    assert decision.judge_model == "deepseek-v4-pro"


def test_source_aware_judge_rejects_real_source_loss():
    decision = assess_refinement(
        "Barents murió una semana después, pero la mayoría sobrevivió.",
        "Barents murió una semana después.",
        chunk_index=2,
        source_text="Barents died a week later, but most of the others survived.",
        target_language="Spanish",
    )

    apply_source_aware_guard_assessment(
        decision,
        {
            "decision": "reject",
            "confidence": 0.95,
            "reason": "The refined text drops that most of the others survived.",
            "issues": ["content_loss"],
            "missing_from_source": ["most of the others survived"],
            "added_not_in_source": [],
            "weird_symbols": [],
            "structure_score": 0.8,
        },
        model="deepseek-v4-pro",
    )

    assert decision.accepted is False
    assert "source_aware_judge_reject" in {issue.code for issue in decision.rejections}
    assert decision.judge_missing_from_source == ["most of the others survived"]


def test_parse_source_aware_guard_response_from_wrapped_json():
    parsed = parse_source_aware_guard_response(
        """
        <EDITORIAL_GUARD_JSON>
        {
          "decision": "accept",
          "confidence": 0.88,
          "reason": "cleaner",
          "issues": [],
          "missing_from_source": [],
          "added_not_in_source": [],
          "weird_symbols": [],
          "structure_score": 0.92
        }
        </EDITORIAL_GUARD_JSON>
        """
    )

    assert parsed["decision"] == "accept"
    assert parsed["confidence"] == 0.88
    assert parsed["structure_score"] == 0.92


def test_parse_source_aware_guard_response_ignores_incomplete_large_json():
    parsed = parse_source_aware_guard_response(
        "<EDITORIAL_GUARD_JSON>"
        '{"decision":"accept","confidence":0.9,'
        + ("x" * 100_000)
    )

    assert parsed is None


def test_source_aware_guard_prompt_contains_three_way_inputs():
    prompt = build_source_aware_guard_prompt(
        "Original source",
        "Traducción inicial",
        "Revisión",
        source_language="English",
        target_language="Spanish",
        section="Capítulo 1",
    )

    assert "ORIGINAL SOURCE" in prompt.user
    assert "INITIAL TRANSLATION" in prompt.user
    assert "REFINED CANDIDATE" in prompt.user
    assert "<EDITORIAL_GUARD_JSON>" in prompt.system
