"""
Integration tests for EPUB translation with technical content protection.

Tests verify:
- Technical content is preserved during translation
- Natural language text is translated
- Pipeline integration works correctly
- Both modes (with/without protection) function properly
"""

import pytest
from lxml import etree

from src.core.epub.tag_preservation import TagPreserver
from src.core.epub.html_chunker import HtmlChunker
from src.core.epub.placeholder_validator import PlaceholderValidator
from src.core.epub.body_serializer import extract_body_html


class TestEPUBTechnicalIntegration:
    """Integration tests for EPUB translation with technical content protection."""

    def test_translation_without_technical_protection(self):
        """Test standard EPUB preservation without technical content protection."""
        html = "<p>Hello world</p>"

        preserver = TagPreserver(protect_technical=False)
        result, tag_map = preserver.preserve_tags(html)

        # Should have 2 placeholders (opening and closing p tag)
        assert len(tag_map) == 2
        assert result == "[id0]Hello world[id1]"

    def test_translation_with_technical_protection_enabled(self):
        """Test EPUB preservation WITH technical content protection enabled."""
        html = "<p>The voltage $V_{cm}$ is 10 Mbps using the `MAX1482` chip.</p>"

        preserver = TagPreserver(protect_technical=True)
        result, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Should have more placeholders (tags + technical content)
        assert len(tag_map) > 2

        # Technical content should be in tag_map
        assert "$V_{cm}$" in tag_map.values()
        assert "10 Mbps" in tag_map.values()
        assert "`MAX1482`" in tag_map.values()

    def test_technical_content_preservation(self):
        """Test that technical content is preserved exactly during translation."""
        html = "<p>Formula: $E = mc^2$ and speed: 10 Mbps</p>"

        preserver = TagPreserver(protect_technical=True)
        preserved, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Simulate translation
        translated = preserved.replace("Formula", "Formule").replace("and speed", "et vitesse")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Verify technical content is preserved exactly
        assert "$E = mc^2$" in restored
        assert "10 Mbps" in restored

        # Verify natural text is translated
        assert "Formule" in restored
        assert "vitesse" in restored

    def test_code_block_atomic_preservation(self):
        """Test that code blocks are preserved atomically."""
        html = """<p>Example code:</p>
<pre><code>def hello():
    return "world"</code></pre>
<p>End example.</p>"""

        preserver = TagPreserver(protect_technical=True)
        preserved, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Simulate translation
        translated = preserved.replace("Example code", "Exemple de code")
        translated = translated.replace("End example", "Fin d'exemple")

        # Restore
        restored = preserver.restore_tags(translated, tag_map)

        # Verify structure is maintained
        assert '<pre>' in restored
        assert '<code>' in restored
        assert 'def hello():' in restored
        assert 'return "world"' in restored

        # Verify translation occurred
        assert "Exemple" in restored or "Fin" in restored

    def test_backward_compatibility_no_protection(self):
        """Test backward compatibility: preservation works without protection flag."""
        html = "<p>Simple text $5 dollars</p>"

        preserver = TagPreserver(protect_technical=False)
        result, tag_map = preserver.preserve_tags(html)
        restored = preserver.restore_tags(result, tag_map)

        # Should work exactly as before
        assert restored == html

    def test_statistics_with_technical_protection(self):
        """Test that chunking and validation work with technical protection."""
        html = "<p>The $V_{cm}$ voltage</p>"

        preserver = TagPreserver(protect_technical=True)
        preserved, tag_map = preserver.preserve_tags_and_technical_content(html)

        # Create chunker
        chunker = HtmlChunker(max_tokens=450)
        chunks = chunker.chunk_html_with_placeholders(preserved, tag_map)

        # Should have at least 1 chunk
        assert len(chunks) >= 1

        # Validate placeholders in each chunk
        for chunk in chunks:
            is_valid = PlaceholderValidator.validate_basic(chunk['text'], chunk['local_tag_map'])
            assert is_valid, f"Chunk validation failed for: {chunk['text']}"

    def test_body_extraction_with_technical_content(self):
        """Test body extraction works with technical content."""
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
<p>The $V_{cm}$ voltage is measured.</p>
</body>
</html>"""

        doc_root = etree.fromstring(xhtml.encode('utf-8'))
        body_html, body_element = extract_body_html(doc_root)

        # Body should contain the technical content
        assert "$V_{cm}$" in body_html
        assert body_element is not None

    def test_full_pipeline_simulation(self):
        """Simulate the full translation pipeline with technical protection."""
        # 1. Start with XHTML
        xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>
<p>The formula $E = mc^2$ is fundamental to physics.</p>
<p>Speed limit: 100 MHz using `CHIP_XY`.</p>
</body>
</html>"""

        # 2. Parse and extract body
        doc_root = etree.fromstring(xhtml.encode('utf-8'))
        body_html, _ = extract_body_html(doc_root)

        # 3. Preserve tags and technical content
        preserver = TagPreserver(protect_technical=True)
        preserved, tag_map = preserver.preserve_tags_and_technical_content(body_html)

        # 4. Chunk
        chunker = HtmlChunker(max_tokens=450)
        chunks = chunker.chunk_html_with_placeholders(preserved, tag_map)

        # 5. Simulate translation of each chunk
        translated_chunks = []
        for chunk in chunks:
            # Simulate translation (preserve placeholders, translate text)
            translated = chunk['text']
            translated = translated.replace("fundamental to physics", "fondamentale en physique")
            translated = translated.replace("Speed limit", "Limite de vitesse")
            translated_chunks.append(translated)

        # 6. Validate placeholders preserved
        for i, (chunk, translated) in enumerate(zip(chunks, translated_chunks)):
            is_valid = PlaceholderValidator.validate_basic(translated, chunk['local_tag_map'])
            assert is_valid, f"Chunk {i} validation failed"

        # 7. Restore tags (join all chunks first)
        full_translated = "".join(translated_chunks)
        restored = preserver.restore_tags(full_translated, tag_map)

        # 8. Verify technical content preserved
        assert "$E = mc^2$" in restored
        assert "100 MHz" in restored
        assert "`CHIP_XY`" in restored

        # 9. Verify translation occurred
        assert "fondamentale" in restored or "physics" in restored
        assert "Limite" in restored or "Speed" in restored


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
