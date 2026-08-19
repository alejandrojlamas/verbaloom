from src.core.locale_quality import (
    build_mexican_spanish_repair_instructions,
    build_spanish_modernization_repair_instructions,
    collect_mexican_spanish_issue_examples,
    count_mexican_spanish_issues,
    count_spanish_modernization_residue,
    is_mexican_spanish_target,
    mexican_spanish_issue_catalog,
)


def test_detects_peninsular_forms_for_mexican_spanish_guard():
    text = "Chitón, vosotras no os habéis rendido; desde vuestros confines, precipitaos."

    counts = count_mexican_spanish_issues(text)

    assert counts["chiton"] == 1
    assert counts["vosotros"] == 1
    assert counts["os_pronoun"] == 1
    assert counts["habeis"] == 1
    assert counts["vuestro"] == 1
    assert counts["vosotros_imperative"] == 1


def test_preserves_conventional_historical_honorifics():
    text = (
        "Firmó la carta con tratamientos como Vuestra Alteza y Majestad, "
        "y luego habló de vuestros confines."
    )

    counts = count_mexican_spanish_issues(text)
    examples = collect_mexican_spanish_issue_examples(text)

    assert counts["vuestro"] == 1
    assert examples["vuestro"] == ["vuestros"]


def test_mexican_spanish_target_defaults_for_spanish_and_can_be_disabled():
    assert is_mexican_spanish_target("Spanish", None) is True
    assert is_mexican_spanish_target("Spanish", {"spanish_variant": "generic"}) is False
    assert is_mexican_spanish_target("French", {"spanish_variant": "mexican"}) is False


def test_repair_instructions_include_direct_conversions():
    instructions = build_mexican_spanish_repair_instructions({"vosotros": 2})

    assert "ustedes" in instructions
    assert "su/sus" in instructions
    assert "precipítense" in instructions
    assert "chitón" in instructions
    assert "Vuestra Alteza" in instructions


def test_detects_editorial_mexican_spanish_lexical_drift():
    text = (
        "Pidió un zumo en una localización cutre, mientras el chaval decía que "
        "su ordenador molaba."
    )

    counts = count_mexican_spanish_issues(text)
    examples = collect_mexican_spanish_issue_examples(text)
    catalog = mexican_spanish_issue_catalog()

    assert counts["zumo"] == 1
    assert counts["localizaciones_locaciones"] == 1
    assert counts["cutre"] == 1
    assert counts["chaval"] == 1
    assert counts["ordenador"] == 1
    assert counts["mola"] == 1
    assert examples["zumo"] == ["zumo"]
    assert "jugo" in catalog["zumo"]["suggestion"]
    assert "locaciones" in catalog["localizaciones_locaciones"]["suggestion"]


def test_locale_guard_filters_contextual_false_positives_and_catches_coger_inflections():
    text = (
        "Más vale prevenir. El proyectil móvil rodeó otros objetos móviles. "
        "La criatura dijo: «Vuestra Anciana está a salvo». "
        "Después cogió el teléfono, siguió cogiendo vasos y propuso cogerse de la mano. "
        "Su móvil recibió una llamada; al final preguntó: «¿Vale?»."
    )

    counts = count_mexican_spanish_issues(text)
    examples = collect_mexican_spanish_issue_examples(text)

    assert counts["coger_regional"] == 3
    assert counts["movil_phone"] == 1
    assert counts["hala_vale"] == 1
    assert "vuestro" not in counts
    assert "Más vale" not in examples.get("hala_vale", [])


def test_repair_instructions_include_mexican_editorial_lexicon_policy():
    instructions = build_mexican_spanish_repair_instructions(
        {"zumo": 1, "cutre": 1, "localizaciones_locaciones": 1},
        examples={"zumo": ["zumo"], "cutre": ["cutre"], "localizaciones_locaciones": ["localizaciones"]},
    )

    assert "zumo" in instructions
    assert "cutre" in instructions
    assert "localizaciones" in instructions
    assert "locaciones" in instructions
    assert "Mexican/LatAm" in instructions


def test_detects_old_spanish_residue_for_strong_modernization():
    text = (
        "SERÍA el gran Moctezuma, e cenceño e pocas carnes, y la color no muy "
        "moreno. Señor Moctezuma, bien podéis creer que si os queréis ir a "
        "vuestros palacios, traíanle frutas y servíase con barro de Cholula."
    )

    counts = count_spanish_modernization_residue(text)

    assert counts["archaic_e_conjunction"] >= 2
    assert counts["archaic_syntax_phrase"] >= 1
    assert counts["archaic_enclitic"] >= 2
    assert counts["vosotros_verb"] >= 2
    assert counts["os_pronoun"] == 1
    assert counts["vuestro"] == 1


def test_modernization_repair_instructions_are_editorial_not_summary():
    instructions = build_spanish_modernization_repair_instructions({
        "archaic_e_conjunction": 4,
        "vosotros_verb": 2,
    })

    assert "current editorial Spanish" in instructions
    assert "Do not flatten" in instructions
    assert "obsolete grammar" in instructions
