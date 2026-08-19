"""
Unit tests for TranslationExtractor (issue #170 fixes)
"""
import pytest
from src.core.llm.utils.extraction import TranslationExtractor
from src.core.post_processor import clean_translated_text
from src.utils.text_encoding import clean_text_artifacts


class TestTranslationExtractor:
    def test_basic_extraction(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION>Hello world</TRANSLATION>")
        assert result == "Hello world"

    def test_extraction_with_whitespace(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("  <TRANSLATION>  Hello world  </TRANSLATION>  ")
        assert result == "Hello world"

    def test_think_blocks_removed(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract(
            "<think>Some reasoning</think><TRANSLATION>Hello</TRANSLATION>"
        )
        assert result == "Hello"

    def test_orphan_think_before_translation_is_stripped(self):
        """Orphan </think> before <TRANSLATION> should be stripped."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "Some reasoning...</think>\n<TRANSLATION>Hello</TRANSLATION>"
        result = extractor.extract(raw)
        assert result == "Hello"

    def test_orphan_think_inside_translation_is_preserved(self):
        """Issue #170 fix: orphan </think> inside translation must NOT destroy the tag."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "<TRANSLATION>\nHello world\n</think>"
        result = extractor.extract(raw)
        # The orphan remover should NOT strip because prefix contains <TRANSLATION>
        assert result is None  # extraction still fails (no closing tag), but content was NOT destroyed

    def test_orphan_think_inside_translation_content_preserved(self):
        """Ensure the raw content is still inspectable after failed extraction."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        raw = "<TRANSLATION>\nLe renard brun\n</think>"
        result = extractor.extract(raw)
        # Content should NOT be wiped by orphan remover
        assert result is None
        assert "<TRANSLATION>" in raw
        assert "Le renard brun" in raw

    def test_markdown_fence_stripping(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("```xml\n<TRANSLATION>Hello</TRANSLATION>\n```")
        assert result == "Hello"

    def test_no_tags_returns_none(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("Just some text without tags")
        assert result is None

    def test_partial_opening_tag_returns_none(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION> incomplete")
        assert result is None

    def test_fuzzy_closing_tag(self):
        """Gemini-style typo in closing tag (</TRANATION>)"""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATION>Hello</TRANATION>")
        assert result == "Hello"

    def test_fuzzy_duplicated_translation_wrapper(self):
        """DeepSeek-style duplicated TRANSLATION suffix should not leak."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract("<TRANSLATIONATION>Hello world</TRANSLATIONATION>")
        assert result == "Hello world"

    def test_fuzzy_wrapper_uses_last_closing_tag(self):
        """A stray malformed close in the middle should be stripped, not truncate."""
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract(
            "<TRANSLATIONATION>Hola. </TRANSLATIONATION> Seguimos.</TRANSLATIONATION>"
        )
        assert result == "Hola. Seguimos."

    def test_residual_malformed_closing_tag_is_stripped_from_content(self):
        extractor = TranslationExtractor("<TRANSLATION>", "</TRANSLATION>")
        result = extractor.extract(
            "<TRANSLATION>Hola. </TRANSLATIONATION> Seguimos.</TRANSLATION>"
        )
        assert result == "Hola. Seguimos."

    def test_clean_text_artifacts_strips_html_escaped_wrapper_tags(self):
        text = "Hola. &lt;/TRANSLATIONATION&gt; Seguimos."
        assert clean_text_artifacts(text) == "Hola. Seguimos."

    def test_clean_text_artifacts_strips_llm_markdown_emphasis(self):
        text = "No le hicieron el *per signum crucis* con un alfanje. * * *"
        assert clean_text_artifacts(text) == (
            "No le hicieron el per signum crucis con un alfanje. * * *"
        )

    def test_clean_text_artifacts_strips_standalone_source_link_artifact(self):
        text = (
            "Texto real.\n\n"
            "[OceanofPDF.com]\n"
            "(https://oceanofpdf.com)\n\n"
            "12.\n\n"
            "Sigue el libro."
        )

        cleaned = clean_text_artifacts(text)

        assert "Texto real." in cleaned
        assert "Sigue el libro." in cleaned
        assert "OceanofPDF" not in cleaned
        assert "https://oceanofpdf.com" not in cleaned
        assert "\n12.\n" not in f"\n{cleaned}\n"

    def test_clean_text_artifacts_strips_markdown_link_targets(self):
        cleaned = clean_text_artifacts(
            "Texto con [una nota](../Text/notas.xhtml#nt23) y [web](https://example.com)."
        )

        assert cleaned == "Texto con una nota y web."
        assert "notas.xhtml" not in cleaned
        assert "https://example.com" not in cleaned

    def test_clean_text_artifacts_removes_internal_numeric_note_links(self):
        cleaned = clean_text_artifacts(
            "Mucho antes de que ocurra.[[23]](../Text/notas.xhtml#nt23)\n\nSigue el texto."
        )

        assert cleaned == "Mucho antes de que ocurra.\n\nSigue el texto."
        assert "[23]" not in cleaned
        assert "notas.xhtml" not in cleaned

    def test_clean_text_artifacts_preserves_reconstruction_placeholder_before_parentheses(self):
        text = "Bandharrawuy[id319](Riratjingu)[id320]Miliritbi"

        assert clean_text_artifacts(text) == text

    def test_clean_text_artifacts_cleans_real_link_without_touching_placeholders(self):
        text = "[id7](Worora)[id8] y [sitio](https://example.com)."

        assert clean_text_artifacts(text) == "[id7](Worora)[id8] y sitio."

    def test_post_processor_strips_decoded_wrapper_tags(self):
        text = "Hola. </TRANSLATIONATION> Seguimos."
        assert clean_translated_text(text) == "Hola. Seguimos."
