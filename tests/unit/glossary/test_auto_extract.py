import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.auto import extract_glossary_candidates
from src.core.glossary.lexical_policy import resolve_lexical_policy


def _by_source(candidates):
    return {item["source"]: item for item in candidates}


def test_local_extractor_finds_recurring_names_and_acronyms():
    text = """
    Ada Lovelace studied the Analytical Engine with Charles Babbage.
    Ada Lovelace wrote notes about the Analytical Engine.
    BLEU and WMT are used in machine translation evaluation.
    BLEU appears again in WMT reports.
    """

    candidates, warnings = extract_glossary_candidates(text, max_terms=20, min_occurrences=2)
    by_source = _by_source(candidates)

    assert warnings == []
    assert "Ada Lovelace" in by_source
    assert by_source["Ada Lovelace"]["category"] == "character"
    assert by_source["Ada Lovelace"]["target"] == "Ada Lovelace"
    assert by_source["Ada Lovelace"]["keep_source"] is True
    assert "BLEU" in by_source
    assert by_source["BLEU"]["category"] == "acronym"
    assert by_source["BLEU"]["occurrences"] == 2


def test_local_extractor_marks_lowercase_technical_phrases_for_translation():
    text = """
    Scaled dot-product attention improves sequence transduction.
    The model relies on scaled dot-product attention and multi-head attention.
    We compare multi-head attention with recurrent networks.
    """

    candidates, _warnings = extract_glossary_candidates(text, max_terms=20, min_occurrences=2)
    by_source = _by_source(candidates)

    assert "scaled dot-product attention" in by_source
    assert by_source["scaled dot-product attention"]["needs_translation"] is True
    assert by_source["scaled dot-product attention"]["target"] == ""


def test_local_extractor_marks_existing_glossary_terms():
    text = (
        "Transformer improves attention. Transformer uses attention. "
        "The Transformer architecture appears in this chapter. "
        "Transformer models are discussed with enough surrounding text "
        "to pass the reliability threshold for local extraction. "
    )

    candidates, _warnings = extract_glossary_candidates(
        text,
        max_terms=20,
        min_occurrences=2,
        existing_sources={"Transformer"},
    )
    by_source = _by_source(candidates)

    assert by_source["Transformer"]["already_in_glossary"] is True


def test_local_extractor_marks_titlecase_demonyms_as_translation_needed():
    text = (
        "American travelers met Mexican guides near the station. "
        "The American road and Mexican market were mentioned again. "
        "Mexican These was OCR-like fragment noise. "
        "Consul The Americano No He was another broken fragment. "
    ) * 20

    candidates, _warnings = extract_glossary_candidates(
        text,
        max_terms=50,
        min_occurrences=2,
    )
    by_source = _by_source(candidates)

    assert by_source["American"]["needs_translation"] is True
    assert by_source["American"]["target"] == ""
    assert by_source["Mexican"]["needs_translation"] is True
    assert by_source["Mexican"]["target"] == ""
    assert "Mexican These" not in by_source
    assert "Consul The Americano No He" not in by_source


def test_local_extractor_does_not_promote_sentence_starters_as_names():
    text = (
        "Suddenly the road bent toward town. "
        "Suddenly the consul stopped walking. "
        "Suddenly the Consul looked back. "
        "Suddenly the Consul crossed the street. "
        "Probably the cantina would be closed. "
        "Probably they would return later. "
        "Nevertheless the chapter continued. "
        "Nevertheless the repeated sentence starters were not names. "
    ) * 20

    candidates, _warnings = extract_glossary_candidates(
        text,
        max_terms=50,
        min_occurrences=2,
    )
    by_source = _by_source(candidates)

    assert "Suddenly" not in by_source
    assert "Suddenly the Consul" not in by_source
    assert "Probably" not in by_source
    assert "Nevertheless" not in by_source


def test_local_extractor_preserves_symbol_bearing_names_exactly():
    text = (
        "Ch*Tril called Ek*Tiq while Dj\\Tal waited nearby. "
        "Later Ch*Tril answered Ek*Tiq and Dj\\Tal returned. "
    ) * 10

    candidates, warnings = extract_glossary_candidates(
        text,
        max_terms=50,
        min_occurrences=2,
    )
    by_source = _by_source(candidates)

    assert warnings == []
    assert by_source["Ch*Tril"]["category"] == "character"
    assert by_source["Ch*Tril"]["target"] == "Ch*Tril"
    assert by_source["Dj\\Tal"]["keep_source"] is True
    assert "Ch Tril" not in by_source


def test_lexical_policy_does_not_leak_english_terms_into_german():
    english = resolve_lexical_policy("English")
    german = resolve_lexical_policy("German")

    assert "attention" in english.tech_nouns
    assert "attention" not in german.tech_nouns
    assert "suddenly" in english.single_title_stopwords
    assert "suddenly" not in german.single_title_stopwords
    assert german.common_noun_capitalization is True


def test_local_extractor_uses_only_the_active_language_technical_policy():
    text = (
        "scaled dot-product attention improves the model. "
        "scaled dot-product attention appears again in this explanation. "
    ) * 8

    english_candidates, _ = extract_glossary_candidates(
        text,
        max_terms=30,
        min_occurrences=2,
        source_language="English",
    )
    german_candidates, _ = extract_glossary_candidates(
        text,
        max_terms=30,
        min_occurrences=2,
        source_language="German",
    )

    assert "scaled dot-product attention" in _by_source(english_candidates)
    assert "scaled dot-product attention" not in _by_source(german_candidates)
