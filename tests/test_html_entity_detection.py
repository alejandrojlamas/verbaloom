"""
Test HTML entity block detection in TechnicalContentDetector

This test ensures that blocks of HTML entities (escaped code examples in documentation)
are properly detected and protected from translation.
"""

import pytest
from src.core.epub.technical_content_detector import TechnicalContentDetector, PatternPriority


class TestHTMLEntityDetection:
    """Test HTML entity block detection."""

    def setup_method(self):
        """Set up detector for each test."""
        self.detector = TechnicalContentDetector()

    def test_simple_html_entity_block(self):
        """Test detection of a simple HTML entity block."""
        text = "Example: &lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt;"
        patterns = self.detector.find_all_technical_content(text)

        # Should find HTML entity block
        assert len(patterns) == 1
        assert patterns[0].pattern_name == "html_entity_block"
        assert patterns[0].priority == PatternPriority.HTML_ENTITY_BLOCK
        assert "&lt;section&gt;" in patterns[0].content

    def test_html_entity_with_mixed_content(self):
        """Test HTML entities mixed with regular text."""
        text = (
            "Here is an example:\n"
            "&lt;section epub:type=\"part\"&gt;\n"
            "    &lt;h1&gt;Part I&lt;/h1&gt;\n"
            "&lt;/section&gt;\n"
            "This shows the structure."
        )
        patterns = self.detector.find_all_technical_content(text)

        # Should find the HTML entity block
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) >= 1

        # The entity block should contain the escaped tags
        entity_content = html_entity_blocks[0].content
        assert "&lt;section" in entity_content or "&lt;h1&gt;" in entity_content

    def test_single_entity_not_detected(self):
        """Test that single HTML entities are not treated as blocks."""
        text = "The price is 5 &amp; the tax is 10%."
        patterns = self.detector.find_all_technical_content(text)

        # Should not detect single ampersand as HTML entity block
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) == 0

    def test_numeric_entities(self):
        """Test detection of numeric HTML entities."""
        text = "Code: &#60;div&#62;content&#60;/div&#62;"
        patterns = self.detector.find_all_technical_content(text)

        # Should find numeric entity block
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) == 1
        assert "&#60;" in html_entity_blocks[0].content

    def test_hex_entities(self):
        """Test detection of hexadecimal HTML entities."""
        text = "Code: &#x3C;span&#x3E;text&#x3C;/span&#x3E;"
        patterns = self.detector.find_all_technical_content(text)

        # Should find hex entity block
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) == 1
        assert "&#x3C;" in html_entity_blocks[0].content

    def test_complex_example_from_issue(self):
        """Test the exact example from the bug report."""
        text = (
            "&lt;section epub:type=\"part\"&gt;\n"
            "    &lt;h1&gt;Part I&lt;/h1&gt;\n"
            "\n"
            "    &lt;section epub:type=\"chapter\"&gt;\n"
            "        &lt;h2&gt;Chapter 1&lt;/h2&gt;\n"
            "        …\n"
            "    &lt;/section&gt;\n"
            "&lt;/section&gt;"
        )

        patterns = self.detector.find_all_technical_content(text)

        # Should find HTML entity blocks
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) >= 1

        # Should preserve the entire structure
        full_content = ''.join([p.content for p in html_entity_blocks])
        assert "&lt;section" in full_content
        assert "&lt;h1&gt;" in full_content

    def test_html_entity_with_placeholders(self):
        """Test that HTML entities work correctly when mixed with placeholders."""
        text = "[id0]Numbered headings[id1]Using an [id2]h1[id3] heading and &lt;section&gt;&lt;h1&gt;Title&lt;/h1&gt;&lt;/section&gt; example."
        patterns = self.detector.find_all_technical_content(text)

        # Should find HTML entity block separate from placeholders
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) >= 1

        # Should contain the escaped tags
        assert any("&lt;section&gt;" in p.content for p in html_entity_blocks)

    def test_priority_html_entity_vs_inline_code(self):
        """Test that HTML entity blocks have higher priority than inline code."""
        text = "`code` and &lt;tag&gt;&lt;/tag&gt; example"
        patterns = self.detector.find_all_technical_content(text)

        # Both should be detected
        pattern_names = [p.pattern_name for p in patterns]
        assert "inline_code" in pattern_names
        assert "html_entity_block" in pattern_names

        # HTML entity block should have priority 9, inline code priority 5
        html_entity = next(p for p in patterns if p.pattern_name == "html_entity_block")
        inline_code = next(p for p in patterns if p.pattern_name == "inline_code")
        assert html_entity.priority == PatternPriority.HTML_ENTITY_BLOCK
        assert inline_code.priority == PatternPriority.INLINE_CODE
        assert html_entity.priority > inline_code.priority

    def test_multiline_html_entity_block(self):
        """Test detection of multiline HTML entity blocks."""
        text = """Documentation shows:
&lt;section epub:type="part"&gt;
    &lt;h1&gt;Part I&lt;/h1&gt;
    &lt;section epub:type="chapter"&gt;
        &lt;h2&gt;Chapter 1&lt;/h2&gt;
    &lt;/section&gt;
&lt;/section&gt;
End of example."""

        patterns = self.detector.find_all_technical_content(text)

        # Should find HTML entity blocks
        html_entity_blocks = [p for p in patterns if p.pattern_name == "html_entity_block"]
        assert len(html_entity_blocks) >= 1

        # Should span multiple lines
        block = html_entity_blocks[0]
        assert '\n' in block.content or len(block.content) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
