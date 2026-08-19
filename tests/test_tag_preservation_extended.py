"""
Unit tests for TagPreserver with technical content protection

Tests the integration of technical content detection with tag preservation,
ensuring both HTML tags and technical content are protected using the same
placeholder system.
"""

import pytest
from src.core.epub.tag_preservation import TagPreserver


class TestTagPreserverBasicBackwardCompatibility:
    """Test backward compatibility when protect_technical=False (default)."""

    def test_default_behavior_unchanged(self):
        """Standard tag preservation works as before."""
        preserver = TagPreserver()
        text = "<p>Hello world</p>"
        result, tag_map = preserver.preserve_tags(text)

        assert result == "[id0]Hello world[id1]"
        assert tag_map["[id0]"] == "<p>"
        assert tag_map["[id1]"] == "</p>"

    def test_preserve_tags_method_still_works(self):
        """Original preserve_tags() method unchanged."""
        preserver = TagPreserver(protect_technical=False)
        text = "<p>The $V_{cm}$ voltage</p>"
        result, tag_map = preserver.preserve_tags(text)

        # Technical content NOT protected when using preserve_tags()
        assert "$V_{cm}$" in result
        assert len(tag_map) == 2  # Only opening and closing tags


class TestTechnicalContentProtectionBasic:
    """Test basic technical content protection."""

    def test_latex_inline_formula_protection(self):
        """LaTeX inline formulas are protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>The common mode voltage ($V_{cm}$) is measured.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Should have 3 placeholders: <p>, $V_{cm}$, </p>
        assert result == "[id0]The common mode voltage ([id1]) is measured.[id2]"
        assert tag_map["[id0]"] == "<p>"
        assert tag_map["[id1]"] == "$V_{cm}$"
        assert tag_map["[id2]"] == "</p>"

    def test_inline_code_protection(self):
        """Inline code with backticks is protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Use the `MAX1482` chip for this.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        assert "`MAX1482`" in tag_map.values()
        assert "`MAX1482`" not in result.replace("[id0]", "").replace("[id1]", "").replace("[id2]", "")

    def test_measurement_protection(self):
        """Technical measurements are protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Speed: 10 Mbps using 32 ULs</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Find placeholders for measurements
        assert "10 Mbps" in tag_map.values()
        assert "32 ULs" in tag_map.values()

    def test_technical_identifier_protection(self):
        """Technical identifiers like TIA/EIA-485-A are protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Complies with TIA/EIA-485-A standard.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        assert "TIA/EIA-485-A" in tag_map.values()

    def test_symbol_bearing_identifier_protection(self):
        """Model-sensitive suffixes are represented by exact placeholders."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>The Q* project uses C++ but not plain Q.</p>"

        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        assert "Q*" in tag_map.values()
        assert "C++" in tag_map.values()
        assert "Q*" not in result
        assert "C++" not in result


class TestMultilineBlockProtection:
    """Test protection of multiline code blocks and LaTeX display formulas."""

    def test_code_block_atomic_protection(self):
        """Code blocks are kept atomic (never split)."""
        preserver = TagPreserver(protect_technical=True)
        code_block = "```python\ndef calculate():\n    return 42\n```"
        text = f"<p>Example:</p>{code_block}<p>Done.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # The code block should be in a single placeholder (not split across multiple)
        code_in_single_placeholder = any(code_block in value for value in tag_map.values())
        assert code_in_single_placeholder, "Code block should be atomic in a single placeholder"

        # Verify translatable text is preserved
        assert "Example:" in result
        assert "Done." in result

    def test_latex_display_formula_protection(self):
        """LaTeX display formulas ($$...$$) are protected."""
        preserver = TagPreserver(protect_technical=True)
        formula = "$$E = mc^2$$"
        text = f"<p>Einstein's equation: {formula} is famous.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        assert formula in tag_map.values()

    def test_code_block_between_paragraphs(self):
        """Code blocks between paragraphs are protected."""
        preserver = TagPreserver(protect_technical=True)
        # Use backtick code blocks which our detector recognizes
        code = "```\ndef f():\n    pass\n```"
        text = f"<p>Example:</p>{code}<p>Done</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Check that translatable text is preserved
        assert "Example:" in result
        assert "Done" in result

        # Code block should be in its own placeholder
        code_in_placeholder = any(code in value for value in tag_map.values())
        assert code_in_placeholder, "Code block should be protected in a placeholder"


class TestFalsePositiveAvoidance:
    """Test that currency and other false positives are NOT protected."""

    def test_currency_not_protected(self):
        """Simple currency amounts are not treated as LaTeX."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Price: $5 or $10.50</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Currency should NOT be in placeholders (should be in result text)
        assert "$5" in result or "$10.50" in result

    def test_variable_names_not_protected(self):
        """Simple variable names in $ are not protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>The $price variable</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # $price should not be protected (no LaTeX indicators)
        assert "$price" in result


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_mixed_content(self):
        """Multiple types of technical content in one paragraph."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>The $V_{cm}$ voltage is 10 Mbps using `MAX1482` chip per TIA/EIA-485-A.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # All technical content should be protected
        values = list(tag_map.values())
        assert "$V_{cm}$" in values
        assert "10 Mbps" in values
        assert "`MAX1482`" in values
        assert "TIA/EIA-485-A" in values

        # Translatable text should remain
        translatable = result
        for placeholder in tag_map.keys():
            translatable = translatable.replace(placeholder, "")
        assert "voltage is" in translatable or "using" in translatable or "chip per" in translatable

    def test_nested_tags_with_technical(self):
        """Nested HTML tags with technical content."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p><span class='formula'>The $V_{cm}$ formula</span> is important.</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Formula should be protected
        assert "$V_{cm}$" in tag_map.values()

        # Translatable text preserved
        assert "formula" in result
        assert "is important." in result

    def test_empty_paragraphs_with_technical(self):
        """Empty paragraphs and numbers grouped correctly."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p> </p><p>1.</p><p>The `code` here</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Empty paragraphs and chapter number should be grouped
        # Technical content should be protected
        assert "`code`" in tag_map.values()


class TestPlaceholderFormat:
    """Test that placeholder format is consistent."""

    def test_sequential_numbering(self):
        """Placeholders are numbered sequentially."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>The $V_{cm}$ and $I_{max}$ values</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Should have sequential numbering
        assert "[id0]" in result
        assert "[id1]" in result
        assert "[id2]" in result
        assert "[id3]" in result

    def test_consistent_format_with_tags_and_technical(self):
        """Same [idN] format for both tags and technical content."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Formula $x^2$ here</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # All placeholders should use [idN] format
        for placeholder in tag_map.keys():
            assert placeholder.startswith("[id")
            assert placeholder.endswith("]")


class TestRestoration:
    """Test that content can be properly restored."""

    def test_restore_mixed_content(self):
        """Restore both tags and technical content."""
        preserver = TagPreserver(protect_technical=True)
        original = "<p>The $V_{cm}$ voltage is `MAX1482`</p>"

        # Preserve
        result, tag_map = preserver.preserve_tags_and_technical_content(original)

        # Restore
        restored = preserver.restore_tags(result, tag_map)

        assert restored == original

    def test_restore_after_translation_simulation(self):
        """Simulate translation and restore."""
        preserver = TagPreserver(protect_technical=True)
        original = "<p>The voltage $V_{cm}$ is measured.</p>"

        # Preserve
        result, tag_map = preserver.preserve_tags_and_technical_content(original)

        # Simulate translation (only translate the translatable text)
        # Replace "The voltage" with "La tension" and "is measured" with "est mesurée"
        translated = result.replace("The voltage", "La tension").replace("is measured", "est mesurée")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Should have French text but original tags and technical content
        assert "<p>" in restored
        assert "</p>" in restored
        assert "$V_{cm}$" in restored
        assert "La tension" in restored
        assert "est mesurée" in restored


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_no_technical_content_with_protection_enabled(self):
        """Text with no technical content works normally."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Just normal text here</p>"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Should only have tag placeholders
        assert result == "[id0]Just normal text here[id1]"
        assert len(tag_map) == 2

    def test_only_technical_no_tags(self):
        """Pure technical content without HTML tags."""
        preserver = TagPreserver(protect_technical=True)
        text = "The formula $V_{cm}$ is important"
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Formula should be protected
        assert "$V_{cm}$" in tag_map.values()
        assert "The formula" in result
        assert "is important" in result

    def test_empty_text(self):
        """Empty text doesn't crash."""
        preserver = TagPreserver(protect_technical=True)
        text = ""
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        assert result == ""
        assert len(tag_map) == 0

    def test_malformed_latex_not_protected(self):
        """Malformed LaTeX (unclosed $) is not protected."""
        preserver = TagPreserver(protect_technical=True)
        text = "<p>Price is $50 for item</p>"  # Unclosed $
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Unclosed $ should not be protected
        # Only tags should be in placeholders
        assert "$50" in result  # Currency in result, not protected


class TestPerformance:
    """Test performance characteristics."""

    def test_lazy_detector_loading(self):
        """Detector is only loaded when protect_technical=True."""
        preserver_off = TagPreserver(protect_technical=False)
        assert preserver_off._detector is None

        preserver_on = TagPreserver(protect_technical=True)
        assert preserver_on._detector is None  # Not loaded yet

        # Should load on first use
        preserver_on.preserve_tags_and_technical_content("<p>Test $x$</p>")
        assert preserver_on._detector is not None

    def test_large_text_processing(self):
        """Can handle reasonably large text."""
        preserver = TagPreserver(protect_technical=True)

        # Create large text with mixed content
        paragraphs = []
        for i in range(100):
            paragraphs.append(f"<p>Paragraph {i} with formula $V_{{{i}}}$ and measurement {i} MHz.</p>")

        text = "".join(paragraphs)
        result, tag_map = preserver.preserve_tags_and_technical_content(text)

        # Should complete without error
        assert len(tag_map) > 0
        assert "Paragraph" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
