"""
Unit tests for build_glossary_block.

Verifies the glossary prompt block is rendered with the expected heading,
mandatory phrasing, and one ``  - source -> target`` line per entry,
preserving dict insertion order.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.glossary.injector import build_glossary_block


class TestBuildGlossaryBlock:
    """Tests for build_glossary_block."""

    def test_empty_dict_returns_empty_string(self):
        """An empty terms dict yields an empty string."""
        result = build_glossary_block({})
        assert result == ""

    def test_blank_targets_are_not_rendered(self):
        """Draft auto-extracted rows must not become source -> blank prompt rules."""
        result = build_glossary_block({"Transformer": ""})
        assert result == ""

    def test_non_empty_contains_heading_and_mandatory(self):
        """Non-empty dict produces a block with the GLOSSARY heading and MANDATORY phrasing."""
        result = build_glossary_block({"hello": "bonjour"})
        assert "GLOSSARY" in result
        assert "MANDATORY" in result

    def test_non_empty_contains_source_target_lines(self):
        """Each glossary entry appears as a 'source -> target' line."""
        terms = {"hello": "bonjour", "world": "monde"}
        result = build_glossary_block(terms)
        assert "hello -> bonjour" in result
        assert "world -> monde" in result

    def test_line_format_two_space_dash(self):
        """Lines use the format '  - {source} -> {target}'."""
        terms = {"cat": "chat"}
        result = build_glossary_block(terms)
        assert "  - cat -> chat" in result

    def test_order_matches_dict_iteration_order(self):
        """Lines appear in the dict's insertion order (Python 3.7+)."""
        terms = {"zebra": "zebre", "apple": "pomme", "mango": "mangue"}
        result = build_glossary_block(terms)
        # Locate each entry's position in the rendered block
        z_pos = result.find("zebra -> zebre")
        a_pos = result.find("apple -> pomme")
        m_pos = result.find("mango -> mangue")
        assert z_pos != -1
        assert a_pos != -1
        assert m_pos != -1
        # Insertion order: zebra, apple, mango
        assert z_pos < a_pos < m_pos


class TestBuildGlossaryBlockEdgeCases:
    """Edge case tests for build_glossary_block."""

    def test_single_entry(self):
        """A single entry still produces a complete block."""
        result = build_glossary_block({"key": "value"})
        assert "GLOSSARY" in result
        assert "MANDATORY" in result
        assert "  - key -> value" in result

    def test_unicode_source_and_target(self):
        """Unicode source/target strings are rendered correctly."""
        result = build_glossary_block({"李凡": "Li Fan"})
        assert "  - 李凡 -> Li Fan" in result


class TestBuildGlossaryBlockMetadata:
    """Category metadata rendering."""

    def test_no_metadata_keeps_old_format(self):
        result = build_glossary_block({"hello": "bonjour"})
        assert "  - hello -> bonjour" in result
        assert "[" not in result.split("MANDATORY")[1].split("\n  -")[1]

    def test_category_appended_when_present(self):
        result = build_glossary_block(
            {"李凡": "Li Fan"},
            term_metadata={"李凡": {"category": "character"}},
        )
        assert "  - 李凡 -> Li Fan  [character]" in result

    def test_missing_category_falls_back_to_plain_line(self):
        result = build_glossary_block(
            {"a": "b", "c": "d"},
            term_metadata={"a": {"category": "character"}},
        )
        assert "  - a -> b  [character]" in result
        assert "  - c -> d" in result
        # `c` line should not have brackets
        c_line = [l for l in result.splitlines() if "c -> d" in l][0]
        assert "[" not in c_line


class TestBuildGlossaryBlockAlternatives:
    """Pipe-separated alternative source forms are rendered as a comma list."""

    def test_pipe_alternatives_rendered_as_comma_list(self):
        result = build_glossary_block({"Москва|Москве|Москвы|Москвой": "Moscou"})
        assert "  - Москва, Москве, Москвы, Москвой -> Moscou" in result
        # The raw '|' must not leak into the LLM-facing block.
        assert "|" not in result

    def test_pipe_alternatives_with_category(self):
        result = build_glossary_block(
            {"Москва|Москве": "Moscou"},
            term_metadata={"Москва|Москве": {"category": "location"}},
        )
        assert "  - Москва, Москве -> Moscou  [location]" in result

    def test_alternatives_announcement_in_header(self):
        """The block intro mentions the comma-separated alternative convention."""
        result = build_glossary_block({"Hund|Hundes": "chien"})
        assert "comma-separated" in result.lower()

    def test_plain_term_unchanged_no_pipe(self):
        """A term without '|' is rendered exactly as before."""
        result = build_glossary_block({"hello": "bonjour"})
        assert "  - hello -> bonjour" in result
        # No spurious comma anywhere on the entry line.
        entry_line = [l for l in result.splitlines() if "hello -> bonjour" in l][0]
        assert "," not in entry_line


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
