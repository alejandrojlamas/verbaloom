"""
Unit tests for filter_glossary.

Verifies per-chunk glossary filtering for Latin word boundaries, CJK substring
matches, longest-first ordering, capping behavior, and case sensitivity.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.filter import filter_glossary, filter_glossary_for_purpose
from src.core.glossary.models import GlossaryConfig


class TestFilterGlossary:
    """Tests for filter_glossary."""

    def test_empty_chunk(self):
        """Empty chunk returns empty dict and capped=False."""
        filtered, capped = filter_glossary("", {"Fan": "X"})
        assert filtered == {}
        assert capped is False

    def test_empty_glossary(self):
        """Empty glossary returns empty dict and capped=False."""
        filtered, capped = filter_glossary("Some text here", {})
        assert filtered == {}
        assert capped is False

    def test_both_empty(self):
        """Both empty inputs return empty dict and capped=False."""
        filtered, capped = filter_glossary("", {})
        assert filtered == {}
        assert capped is False

    def test_latin_word_boundary_no_match_inside_word(self):
        """Latin term 'Fan' must NOT match inside 'Fantasy'."""
        filtered, capped = filter_glossary("Fantasy is fun", {"Fan": "X"})
        assert filtered == {}
        assert capped is False

    def test_latin_word_boundary_match_standalone(self):
        """Latin term 'Fan' matches when it appears as a standalone word."""
        filtered, capped = filter_glossary("Mr. Fan said", {"Fan": "X"})
        assert filtered == {"Fan": "X"}
        assert capped is False

    def test_cjk_substring_match(self):
        """CJK term matches as substring (no word boundary in CJK)."""
        filtered, capped = filter_glossary("李凡来了", {"李凡": "Li Fan"})
        assert filtered == {"李凡": "Li Fan"}
        assert capped is False

    def test_mixed_latin_and_cjk(self):
        """Mixed Latin and CJK terms in a glossary, both present in chunk."""
        glossary = {"Fan": "Fan-Translated", "李凡": "Li Fan"}
        chunk = "Mr. Fan met 李凡来了 yesterday"
        filtered, capped = filter_glossary(chunk, glossary)
        assert "Fan" in filtered
        assert "李凡" in filtered
        assert filtered["Fan"] == "Fan-Translated"
        assert filtered["李凡"] == "Li Fan"
        assert capped is False

    def test_longest_first_ordering(self):
        """When multiple terms match independently, longest source comes first in dict order."""
        # Use a chunk where both terms appear as standalone words so word-boundary
        # matching catches them both. Insertion order in the source dict has
        # 'Li Fan' first, but the filter must reorder by length (longest first).
        glossary = {"Li Fan": "Short", "Li Fanqing": "Long"}
        chunk = "Li Fan met Li Fanqing yesterday"
        filtered, capped = filter_glossary(chunk, glossary)
        # Both should be present
        assert "Li Fan" in filtered
        assert "Li Fanqing" in filtered
        # Longest source must come first in iteration order
        keys = list(filtered.keys())
        assert keys[0] == "Li Fanqing"
        assert keys[1] == "Li Fan"
        assert capped is False

    def test_cap_applied_when_too_many_matches(self):
        """When matches exceed max_entries, result is capped and capped=True."""
        glossary = {f"Term{i}": f"Trans{i}" for i in range(5)}
        chunk = " ".join(glossary.keys())
        config = GlossaryConfig(max_entries=2)
        filtered, capped = filter_glossary(chunk, glossary, config)
        assert len(filtered) == 2
        assert capped is True

    def test_no_cap_when_below_limit(self):
        """When match count is below the cap, capped=False."""
        glossary = {"Alpha": "A", "Beta": "B"}
        chunk = "Alpha and Beta walk in"
        config = GlossaryConfig(max_entries=10)
        filtered, capped = filter_glossary(chunk, glossary, config)
        assert len(filtered) == 2
        assert capped is False

    def test_case_sensitive_default_no_match(self):
        """With case_sensitive=True (default), 'fan' does not match 'Fan'."""
        filtered, capped = filter_glossary("Mr. Fan said", {"fan": "X"})
        assert filtered == {}
        assert capped is False

    def test_case_insensitive_match(self):
        """With case_sensitive=False, 'fan' matches 'Fan'."""
        config = GlossaryConfig(case_sensitive=False)
        filtered, capped = filter_glossary("Mr. Fan said", {"fan": "X"}, config)
        assert filtered == {"fan": "X"}
        assert capped is False

    def test_term_with_no_word_chars_at_edges_uses_substring_branch(self):
        """A term with non-word chars at BOTH edges forces the substring branch."""
        # Both edges are non-word characters (parens), so _has_word_char_at_edge
        # returns False and the filter uses the plain substring branch.
        glossary = {"(test)": "(essai)"}
        chunk = "this is a (test) for sure"
        filtered, capped = filter_glossary(chunk, glossary)
        assert filtered == {"(test)": "(essai)"}
        assert capped is False

    def test_cyrillic_term_does_not_crash(self):
        """Cyrillic terms are handled without crashing (verifies no exception)."""
        glossary = {"Привет": "Hello"}
        chunk = "Привет мир"
        # Just verify it doesn't crash - exact match behavior depends on \w/\b
        # which in Python's default regex DOES match Cyrillic letters.
        filtered, capped = filter_glossary(chunk, glossary)
        assert isinstance(filtered, dict)
        assert isinstance(capped, bool)


class TestFilterGlossaryEdgeCases:
    """Edge case tests for filter_glossary."""

    def test_empty_source_term_skipped(self):
        """An empty source term key is skipped, not matched."""
        glossary = {"": "Empty", "Fan": "F"}
        chunk = "Mr. Fan said"
        filtered, capped = filter_glossary(chunk, glossary)
        assert "" not in filtered
        assert filtered == {"Fan": "F"}

    def test_no_terms_match_in_chunk(self):
        """No matches yields empty dict and capped=False."""
        glossary = {"Apple": "Pomme", "Banana": "Banane"}
        chunk = "There are oranges only here"
        filtered, capped = filter_glossary(chunk, glossary)
        assert filtered == {}
        assert capped is False

    def test_default_config_when_none_passed(self):
        """Filter works without an explicit config (uses defaults)."""
        glossary = {"Fan": "X"}
        chunk = "Mr. Fan said"
        filtered, capped = filter_glossary(chunk, glossary, None)
        assert filtered == {"Fan": "X"}
        assert capped is False


class TestFilterGlossaryAlternatives:
    """A source term may declare alternative inflected forms separated by '|'."""

    def test_alternatives_match_any_form(self):
        """A '|'-separated source matches when ANY alternative appears in the chunk."""
        glossary = {"Москва|Москве|Москвы|Москвой": "Moscou"}
        # Only the locative form is in the chunk.
        chunk = "Я живу в Москве с детства"
        filtered, capped = filter_glossary(chunk, glossary)
        assert filtered == {"Москва|Москве|Москвы|Москвой": "Moscou"}
        assert capped is False

    def test_alternatives_match_canonical_form(self):
        """The canonical form (first alternative) matches just like the inflected ones."""
        glossary = {"Москва|Москве|Москвы": "Moscou"}
        chunk = "Москва — столица России"
        filtered, capped = filter_glossary(chunk, glossary)
        assert "Москва|Москве|Москвы" in filtered

    def test_alternatives_no_match(self):
        """If none of the alternatives appear in the chunk, the entry is excluded."""
        glossary = {"Hund|Hundes|Hunden": "chien"}
        chunk = "Le chat dort sur le canapé"
        filtered, capped = filter_glossary(chunk, glossary)
        assert filtered == {}
        assert capped is False

    def test_alternatives_count_aggregated_for_cap(self):
        """Occurrences across alternatives are summed for cap selection."""
        # Two terms, one matched once, the other matched many times via alternatives.
        glossary = {
            "rare": "rare-tr",
            "Hund|Hundes|Hunden|Hündin": "chien",
        }
        # 'rare' appears 1×, the German alternatives appear 4× combined.
        chunk = "rare. Der Hund. Des Hundes. Den Hunden. Eine Hündin."
        config = GlossaryConfig(max_entries=1)
        filtered, capped = filter_glossary(chunk, glossary, config)
        assert capped is True
        # The high-frequency alternative entry must win the single slot.
        assert "Hund|Hundes|Hunden|Hündin" in filtered
        assert "rare" not in filtered

    def test_alternatives_whitespace_stripped(self):
        """Alternatives with surrounding whitespace are still matched."""
        glossary = {"Москва | Москве  |   Москвы": "Moscou"}
        chunk = "из Москвы в Париж"
        filtered, capped = filter_glossary(chunk, glossary)
        assert "Москва | Москве  |   Москвы" in filtered

    def test_alternatives_empty_pieces_skipped(self):
        """Empty pieces between '|' (e.g. trailing pipe) don't break matching."""
        glossary = {"Hund||Hundes|": "chien"}
        chunk = "Der Hund läuft"
        filtered, capped = filter_glossary(chunk, glossary)
        assert filtered == {"Hund||Hundes|": "chien"}

    def test_alternatives_word_boundary_still_enforced(self):
        """Latin alternatives still respect word boundaries (no infix matches)."""
        glossary = {"Hund|Hundes": "chien"}
        # 'Hundert' contains 'Hund' as a prefix but is not the same word.
        chunk = "Es gibt hundert Bäume"
        filtered, capped = filter_glossary(chunk, glossary)
        # Lowercase 'hundert' won't match case-sensitive 'Hund|Hundes' anyway,
        # but even with capital 'Hundert' below the word boundary must protect.
        assert filtered == {}

        chunk2 = "Hundert Bäume stehen dort"
        filtered2, capped2 = filter_glossary(chunk2, glossary)
        assert filtered2 == {}, "Word-boundary regex must not match 'Hund' inside 'Hundert'"

    def test_alternatives_mixed_with_plain_term(self):
        """Plain (no-pipe) terms and pipe-terms coexist in the same glossary."""
        glossary = {
            "Москва|Москве": "Moscou",
            "Paris": "Paris",
        }
        chunk = "из Москве в Paris"
        filtered, capped = filter_glossary(chunk, glossary)
        assert "Москва|Москве" in filtered
        assert "Paris" in filtered

    def test_alternatives_sorted_by_longest_alt(self):
        """Sort uses the LONGEST alternative, not the raw source string length."""
        # Raw string lengths: "A|VeryLongTerm" = 14, "Medium" = 6.
        # But longest alternative is "VeryLongTerm" = 12 vs "Medium" = 6.
        glossary = {"A|VeryLongTerm": "X", "Medium": "Y"}
        chunk = "VeryLongTerm and Medium together"
        filtered, capped = filter_glossary(chunk, glossary)
        keys = list(filtered.keys())
        # Longest-alt entry must come first.
        assert keys[0] == "A|VeryLongTerm"
        assert keys[1] == "Medium"


class TestFilterGlossaryForPurpose:
    """Purpose-aware filtering keeps diagnostics and runtime prompts aligned."""

    def test_translation_purpose_does_not_match_target_side(self):
        terms = {"señor": "usted"}
        config = GlossaryConfig(case_sensitive=False, accent_insensitive=True)

        filtered, capped = filter_glossary_for_purpose(
            "Lo vi a usted en la plaza.",
            terms,
            config,
            "translation",
        )

        assert filtered == {}
        assert capped is False

    def test_transformation_purpose_matches_target_side(self):
        terms = {"señor": "usted"}
        config = GlossaryConfig(case_sensitive=False, accent_insensitive=True)

        filtered, capped = filter_glossary_for_purpose(
            "Lo vi a usted en la plaza.",
            terms,
            config,
            "transformation",
        )

        assert filtered == {"señor": "usted"}
        assert capped is False

    def test_refinement_purpose_matches_source_and_target_side(self):
        terms = {"señor": "usted", "Consul": "Cónsul"}
        config = GlossaryConfig(case_sensitive=False, accent_insensitive=True)

        filtered, capped = filter_glossary_for_purpose(
            "El senor saludó al Cónsul.",
            terms,
            config,
            "refinement",
        )

        assert filtered == {"Consul": "Cónsul", "señor": "usted"}
        assert capped is False

    def test_transformation_purpose_matches_simple_target_inflections(self):
        terms = {"British": "británico", "Fascist": "fascista", "Jewish": "judío"}
        config = GlossaryConfig(case_sensitive=False, accent_insensitive=True)

        filtered, capped = filter_glossary_for_purpose(
            "La embajada británica, los fascistas y las familias judías aparecen en el pasaje.",
            terms,
            config,
            "transformation",
        )

        assert filtered == {
            "British": "británico",
            "Fascist": "fascista",
            "Jewish": "judío",
        }
        assert capped is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
