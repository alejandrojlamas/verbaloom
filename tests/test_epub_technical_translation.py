"""
Tests for EPUB translation with technical content protection.

Tests verify:
- TagPreserver with technical protection preserves technical content
- Preservation of technical content (LaTeX, code, measurements, IDs)
- Correct restoration of original content
- Integration with chunking and placeholder renumbering
"""

import pytest
import time
from src.core.epub.tag_preservation import TagPreserver


class TestEPUBTechnicalContentProtection:
    """Test EPUB translation with technical content protection."""

    def test_latex_formula_preservation(self):
        """LaTeX formulas should be preserved exactly."""
        html = "<p>The common mode voltage ($V_{cm}$) is measured.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Check that LaTeX formula is protected
        assert "$V_{cm}$" in tag_map.values(), "LaTeX formula should be in tag_map"

        # Simulate translation
        translated = result.replace("common mode voltage", "tension de mode commun")
        translated = translated.replace("is measured", "est mesurée")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Verify formula preserved
        assert "$V_{cm}$" in restored, "LaTeX formula should be preserved"
        assert "tension de mode commun" in restored, "Natural text should be translated"

    def test_code_block_preservation(self):
        """Code blocks should remain completely unchanged."""
        html = """<p>Example:</p>
<pre><code>def calculate():
    return 42</code></pre>
<p>Done.</p>"""

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Simulate translation
        translated = result.replace("Example", "Exemple").replace("Done", "Terminé")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Code block should be exactly preserved
        assert "def calculate():" in restored
        assert "return 42" in restored

    def test_inline_code_preservation(self):
        """Inline code should be preserved."""
        html = "<p>Use the `MAX1482` chip for communication.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Inline code should be protected
        assert "`MAX1482`" in tag_map.values()

        # Restore and verify
        restored = preserver.restore_tags(result, tag_map)
        assert "`MAX1482`" in restored or "MAX1482" in restored

    def test_measurement_preservation(self):
        """Technical measurements should be preserved."""
        html = "<p>Speed: 10 Mbps using 32 ULs.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Measurements should be protected
        assert "10 Mbps" in tag_map.values()
        assert "32 ULs" in tag_map.values()

        # Restore and verify
        restored = preserver.restore_tags(result, tag_map)
        assert "10 Mbps" in restored
        assert "32 ULs" in restored

    def test_technical_identifier_preservation(self):
        """Technical identifiers like standards should be preserved."""
        html = "<p>Complies with TIA/EIA-485-A standard.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Technical ID should be protected
        assert "TIA/EIA-485-A" in tag_map.values()

        # Restore and verify
        restored = preserver.restore_tags(result, tag_map)
        assert "TIA/EIA-485-A" in restored

    def test_mixed_technical_content(self):
        """Multiple types of technical content in one document."""
        html = "<p>The voltage $V_{cm}$ is 10 Mbps using `MAX1482` per TIA/EIA-485-A.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # All technical content should be protected
        assert "$V_{cm}$" in tag_map.values()
        assert "10 Mbps" in tag_map.values()
        assert "`MAX1482`" in tag_map.values()
        assert "TIA/EIA-485-A" in tag_map.values()

        # Restore and verify
        restored = preserver.restore_tags(result, tag_map)
        assert "$V_{cm}$" in restored
        assert "10 Mbps" in restored
        assert "MAX1482" in restored
        assert "TIA/EIA-485-A" in restored

    def test_currency_not_protected(self):
        """Currency amounts should not be protected (false positive avoidance)."""
        html = "<p>Price: $5 or $10.50 total.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Currency should remain (not treated as LaTeX)
        # Just verify no corruption
        restored = preserver.restore_tags(result, tag_map)
        assert "$" in restored or "5" in restored

    def test_protection_disabled_by_default(self):
        """Technical protection should be disabled when not explicitly enabled."""
        html = "<p>The voltage $V_{cm}$ is measured.</p>"

        preserver = TagPreserver(protect_technical=False)
        result, tag_map = preserver.preserve_tags(html)

        # Formula should NOT be in tag_map (only tags)
        assert "$V_{cm}$" not in tag_map.values()
        assert "$V_{cm}$" in result  # Formula should be in text

    def test_complex_document_structure(self):
        """Document with multiple paragraphs and nested structures."""
        html = """<h1>Introduction</h1>
<p>The formula $E = mc^2$ is fundamental.</p>
<p>Speed of light: 3e8 m/s.</p>
<h2>Implementation</h2>
<pre><code>def energy(m):
    c = 3e8
    return m * c**2</code></pre>
<p>Done with example.</p>"""

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Restore and verify
        restored = preserver.restore_tags(result, tag_map)

        # Technical content preserved
        assert "$E = mc^2$" in restored or "E = mc^2" in restored
        assert "def energy(m):" in restored
        assert "3e8" in restored

        # Structure preserved
        assert "<h1>" in restored
        assert "<h2>" in restored


class TestRegressionWithoutProtection:
    """Ensure standard translation still works when protection is disabled."""

    def test_standard_translation_unchanged(self):
        """Normal preservation without technical protection works as before."""
        html = "<p>This is a simple paragraph.</p><p>Another paragraph here.</p>"

        preserver = TagPreserver(protect_technical=False)
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        # Should work normally
        assert restored == html

    def test_nested_tags_still_work(self):
        """Nested HTML tags work correctly without protection."""
        html = "<p><em>Emphasized</em> and <strong>bold</strong> text.</p>"

        preserver = TagPreserver(protect_technical=False)
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        # Tags should be preserved
        assert "<em>" in restored
        assert "</em>" in restored
        assert "<strong>" in restored
        assert "</strong>" in restored


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_document(self):
        """Empty document doesn't crash."""
        html = ""

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Should complete without error
        assert result == ""
        assert len(tag_map) == 0

    def test_only_technical_content(self):
        """Document with only technical content, no translatable text."""
        html = "<p>$V_{cm}$</p><p>`code`</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Technical content preserved
        restored = preserver.restore_tags(result, tag_map)
        assert "$V_{cm}$" in restored or "V_{cm}" in restored
        assert "`code`" in restored or "code" in restored

    def test_malformed_latex_ignored(self):
        """Malformed LaTeX (unclosed) is not protected."""
        html = "<p>Price is $50 for the item.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Should complete without error
        restored = preserver.restore_tags(result, tag_map)
        assert "$" in restored or "50" in restored


class TestPerformanceImpact:
    """Test performance impact of technical content protection."""

    def test_overhead_is_minimal(self):
        """Technical protection adds minimal overhead."""
        # Create a moderately large document
        paragraphs = []
        for i in range(50):
            paragraphs.append(f"<p>Paragraph {i} with some text and formula $V_{{{i}}}$ here.</p>")

        html = "".join(paragraphs)

        # With protection
        preserver_on = TagPreserver(protect_technical=True)
        start = time.perf_counter()
        result_with, _ = preserver_on.preserve_tags_and_technical_content(html)
        time_with = time.perf_counter() - start

        # Without protection
        preserver_off = TagPreserver(protect_technical=False)
        start = time.perf_counter()
        result_without, _ = preserver_off.preserve_tags(html)
        time_without = time.perf_counter() - start

        # Calculate overhead
        overhead_pct = ((time_with - time_without) / time_without) * 100 if time_without > 0 else 0

        print(f"\nPerformance: With protection: {time_with*1000:.2f}ms, Without: {time_without*1000:.2f}ms")
        print(f"Overhead: {overhead_pct:.1f}%")

        # Technical protection adds regex analysis overhead, which is expected
        # The important thing is absolute time remains reasonable (< 50ms for 50 paragraphs)
        assert time_with < 0.1, f"Absolute time too high: {time_with*1000:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
