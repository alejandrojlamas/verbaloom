"""
Tests for Unicode normalization passes applied at output file boundaries.

Covers the public API of src.utils.text_encoding and verifies that the
normalization layer:
    - preserves visible content unchanged across encode/decode roundtrips
    - is idempotent under repeated application
    - leaves placeholder-shaped tokens intact (tag preservation invariant)
    - never modifies SRT timestamp-shaped strings
    - strips legacy width-zero marks instead of adding hidden payloads
    - applies correctly at TXT, SRT, DOCX, and EPUB write boundaries
"""

import os
import re
import tempfile
import zipfile
from pathlib import Path

import pytest

from src.utils.text_encoding import (
    apply_normalization,
    apply_normalization_to_srt_cue,
    derive_identifier_suffix,
    derive_identifier_urn,
    extract_signature,
)


# Width-zero codepoints used by the module
_ZWNJ = "‌"
_ZWJ = "‍"
_ZWSP = "​"
_WJ = "⁠"
_ANY_WIDTH_ZERO_RE = re.compile(r"[​-‍⁠﻿]")


def _strip_width_zero(s: str) -> str:
    """Remove all width-zero codepoints for visible-content comparisons."""
    return _ANY_WIDTH_ZERO_RE.sub("", s)


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

class TestIdentifierHelpers:
    """Stable identifier helpers used in document metadata fields."""

    def test_suffix_is_12_char_hex(self):
        suffix = derive_identifier_suffix()
        assert isinstance(suffix, str)
        assert len(suffix) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", suffix)

    def test_urn_format(self):
        urn = derive_identifier_urn()
        assert urn.startswith("urn:tbl:")
        assert re.fullmatch(r"urn:tbl:[0-9a-f]{12}", urn)

    def test_suffix_is_stable_across_calls(self):
        assert derive_identifier_suffix() == derive_identifier_suffix()


# ---------------------------------------------------------------------------
# apply_normalization core behavior
# ---------------------------------------------------------------------------

class TestApplyNormalization:
    """Core behavior of the normalization pass on arbitrary text."""

    def test_empty_input_returns_unchanged(self):
        assert apply_normalization("") == ""

    def test_whitespace_only_returns_unchanged(self):
        assert apply_normalization("   \n\t  ") == "   \n\t  "

    def test_short_text_returns_without_hidden_payload(self):
        """Short text must not receive invisible payload characters."""
        short = "Hello."
        out = apply_normalization(short)
        assert out == short
        assert _ANY_WIDTH_ZERO_RE.search(out) is None

    def test_long_text_visible_content_preserved(self):
        """All visible characters of the input must survive normalization."""
        long_text = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        )
        out = apply_normalization(long_text)
        assert _strip_width_zero(out) == long_text

    def test_no_signature_embedded_short_input(self):
        out = apply_normalization("Hello.")
        assert extract_signature(out) is None

    def test_no_signature_embedded_long_input(self):
        long_text = (
            "This is a reasonably long paragraph used to verify that normalization "
            "does not add hidden characters that can render as visible boxes."
        )
        out = apply_normalization(long_text)
        assert out == long_text
        assert extract_signature(out) is None

    def test_metadata_identifier_not_embedded_in_visible_text(self):
        out = apply_normalization("A reasonably long sample paragraph for testing.")
        assert derive_identifier_suffix() not in out
        assert extract_signature(out) is None

    def test_idempotent_application(self):
        """Applying normalization twice yields the same output."""
        text = "A sample paragraph with enough words to test normalization."
        once = apply_normalization(text)
        twice = apply_normalization(once)
        assert twice == once
        assert _ANY_WIDTH_ZERO_RE.search(twice) is None

    def test_idempotent_visible_content_preserved(self):
        text = "Another sample paragraph for repeated normalization checks."
        once = apply_normalization(text)
        twice = apply_normalization(once)
        assert _strip_width_zero(twice) == text

    def test_multiline_text_preserves_newlines(self):
        text = "First paragraph line.\n\nSecond paragraph line follows here."
        out = apply_normalization(text)
        assert "\n\n" in out
        assert _strip_width_zero(out) == text

    def test_legacy_width_zero_marks_are_stripped(self):
        text = f"Clean{_ZWJ}{_ZWNJ}{_ZWSP}{_WJ} text."
        assert apply_normalization(text) == "Clean text."

    def test_replacement_box_artifacts_are_removed(self):
        text = "Texto■■■■\ncon fórmula β1 = 0,9 y ε = 10■■."
        out = apply_normalization(text)
        assert "■" not in out
        assert "β1 = 0,9" in out


# ---------------------------------------------------------------------------
# Placeholder safety
# ---------------------------------------------------------------------------

class TestPlaceholderSafety:
    """Defensive: placeholder-shaped tokens must remain intact and matchable."""

    def test_placeholders_preserved_as_substrings(self):
        text = "[id0] hello world [id1] this is [id2] a test [id3]"
        out = apply_normalization(text)
        for ph in ("[id0]", "[id1]", "[id2]", "[id3]"):
            assert ph in out, f"Placeholder {ph} broken in output"

    def test_placeholders_count_unchanged(self):
        text = "alpha [id0] beta [id1] gamma [id2] delta [id3] epsilon"
        out = apply_normalization(text)
        before = re.findall(r"\[id\d+\]", text)
        after = re.findall(r"\[id\d+\]", out)
        assert before == after

    def test_replace_substitution_still_works(self):
        """Critical for tag_preservation: text.replace('[idN]', tag) must work."""
        text = "alpha [id0] beta [id1] gamma [id2] delta epsilon zeta"
        out = apply_normalization(text)
        restored = out.replace("[id0]", "<p>").replace("[id1]", "</p>").replace("[id2]", "<br/>")
        assert "<p>" in restored
        assert "</p>" in restored
        assert "<br/>" in restored
        assert "[id0]" not in restored
        assert "[id1]" not in restored
        assert "[id2]" not in restored

    def test_uppercase_placeholders_also_skipped(self):
        text = "alpha [ID5] beta [ID6] gamma delta epsilon zeta"
        out = apply_normalization(text)
        assert "[ID5]" in out
        assert "[ID6]" in out


# ---------------------------------------------------------------------------
# SRT cue safety
# ---------------------------------------------------------------------------

class TestSrtCueNormalization:
    """apply_normalization_to_srt_cue: must never modify timestamp-shaped input."""

    def test_normal_cue_text_normalized(self):
        cue = "Hello, this is a regular subtitle line with enough words."
        out = apply_normalization_to_srt_cue(cue)
        assert out == cue
        assert _ANY_WIDTH_ZERO_RE.search(out) is None

    def test_timestamp_input_refused(self):
        """A pure timestamp string must come back unchanged."""
        ts = "00:00:01,000 --> 00:00:05,000"
        out = apply_normalization_to_srt_cue(ts)
        assert out == ts
        assert extract_signature(out) is None

    def test_timestamp_with_dot_separator_refused(self):
        ts = "00:00:01.000 --> 00:00:05.000"
        out = apply_normalization_to_srt_cue(ts)
        assert out == ts

    def test_multiline_cue_preserves_structure(self):
        cue = "First line of cue.\nSecond line of cue."
        out = apply_normalization_to_srt_cue(cue)
        assert "\n" in out
        assert _strip_width_zero(out) == cue


# ---------------------------------------------------------------------------
# extract_signature behavior
# ---------------------------------------------------------------------------

class TestExtractSignature:
    """Signature extraction from normalized and unmodified text."""

    def test_plain_text_returns_none(self):
        assert extract_signature("Hello, this is plain text.") is None

    def test_empty_input_returns_none(self):
        assert extract_signature("") is None

    def test_text_with_random_emoji_returns_none(self):
        """Emoji with their own ZWJ should not trigger false positives."""
        # 👨‍👩‍👧 contains ZWJ between codepoints
        text = "Family emoji 👨‍👩‍👧 in text"
        assert extract_signature(text) is None

    def test_normalization_does_not_create_signature_after_concat(self):
        normalized = apply_normalization("A sample paragraph used for normalization.")
        combined = "Prefix unrelated content " + normalized + " suffix unrelated content"
        assert extract_signature(combined) is None


# ---------------------------------------------------------------------------
# Integration: SRT processor
# ---------------------------------------------------------------------------

class TestSrtProcessorIntegration:
    """Integration: reconstruct_srt cleans cue text without touching structure."""

    def _build_subs(self):
        return [
            {
                "number": "1",
                "start_time": "00:00:01,000",
                "end_time": "00:00:04,000",
                "text": "Hello world, this is the first subtitle line.",
            },
            {
                "number": "2",
                "start_time": "00:00:05,000",
                "end_time": "00:00:08,000",
                "text": "Second subtitle here.",
            },
            {
                "number": "3",
                "start_time": "00:00:09,000",
                "end_time": "00:00:12,000",
                "text": "Third one.",
            },
        ]

    def test_timestamps_intact(self):
        from src.core.srt_processor import SRTProcessor
        out = SRTProcessor().reconstruct_srt(self._build_subs())
        ts_count = len(re.findall(
            r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}",
            out,
        ))
        assert ts_count == 3

    def test_no_hidden_signature_present_in_output(self):
        from src.core.srt_processor import SRTProcessor
        out = SRTProcessor().reconstruct_srt(self._build_subs())
        assert extract_signature(out) is None
        assert _ANY_WIDTH_ZERO_RE.search(out) is None

    def test_output_parses_back_into_subs(self):
        from src.core.srt_processor import SRTProcessor
        proc = SRTProcessor()
        out = proc.reconstruct_srt(self._build_subs())
        parsed = proc.parse_srt(out)
        assert len(parsed) == 3
        # Numbers and timestamps survive round-trip
        assert parsed[0]["number"] == "1"
        assert parsed[0]["start_time"] == "00:00:01,000"
        assert parsed[1]["number"] == "2"
        assert parsed[2]["number"] == "3"

    def test_all_cues_cleaned_without_width_zero_marks(self):
        from src.core.srt_processor import SRTProcessor
        subs = self._build_subs()
        subs[0]["text"] += "■■"
        subs[1]["text"] += "■■"
        SRTProcessor().reconstruct_srt(subs)
        assert "■" not in subs[0]["text"]
        assert "■" not in subs[1]["text"]
        assert _ANY_WIDTH_ZERO_RE.search(subs[0]["text"]) is None
        assert _ANY_WIDTH_ZERO_RE.search(subs[1]["text"]) is None
        assert _ANY_WIDTH_ZERO_RE.search(subs[2]["text"]) is None

    def test_visible_first_cue_text_preserved(self):
        from src.core.srt_processor import SRTProcessor
        subs = self._build_subs()
        original_first = subs[0]["text"]
        SRTProcessor().reconstruct_srt(subs)
        assert _strip_width_zero(subs[0]["text"]) == original_first

    def test_all_empty_cues_no_crash(self):
        from src.core.srt_processor import SRTProcessor
        subs = [
            {"number": "1", "start_time": "00:00:01,000",
             "end_time": "00:00:04,000", "text": ""},
            {"number": "2", "start_time": "00:00:05,000",
             "end_time": "00:00:08,000", "text": "   "},
        ]
        # Should not crash even when no cue is signable
        out = SRTProcessor().reconstruct_srt(subs)
        assert "00:00:01,000" in out


# ---------------------------------------------------------------------------
# Integration: DOCX writer
# ---------------------------------------------------------------------------

class TestDocxIntegration:
    """Integration: DOCX save paths must stamp core properties."""

    def test_minimal_docx_carries_signature_in_core_xml(self):
        """plain_extractor.build_minimal_docx must stamp lastModifiedBy."""
        from src.core.docx.plain_extractor import (
            build_minimal_docx, DocxPlainContent,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.docx")
            content = DocxPlainContent(
                paragraphs_text=["Hello.", "World."],
                paragraphs_style=["normal", "normal"],
                images_by_paragraph={},
            )
            build_minimal_docx(
                translated_paragraphs=["Bonjour.", "Monde."],
                content=content,
                output_path=out_path,
                bilingual=False,
            )
            assert os.path.exists(out_path)
            with zipfile.ZipFile(out_path, "r") as z:
                core_xml = z.read("docProps/core.xml").decode("utf-8")
            m = re.search(
                r"<cp:lastModifiedBy[^>]*>([^<]+)</cp:lastModifiedBy>", core_xml
            )
            assert m is not None
            value = m.group(1)
            assert "TranslateBookWithLLM" in value
            assert derive_identifier_suffix() in value

    def test_html_to_docx_converter_carries_signature(self):
        """DocxHtmlConverter.from_html must stamp lastModifiedBy on save."""
        from src.core.docx.converter import DocxHtmlConverter

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.docx")
            converter = DocxHtmlConverter()
            html = "<html><body><p>Hello.</p></body></html>"
            converter.from_html(html, {}, out_path)

            assert os.path.exists(out_path)
            with zipfile.ZipFile(out_path, "r") as z:
                core_xml = z.read("docProps/core.xml").decode("utf-8")
            m = re.search(
                r"<cp:lastModifiedBy[^>]*>([^<]+)</cp:lastModifiedBy>", core_xml
            )
            assert m is not None
            assert "TranslateBookWithLLM" in m.group(1)


# ---------------------------------------------------------------------------
# Integration: EPUB metadata
# ---------------------------------------------------------------------------

class TestEpubMetadataIntegration:
    """Integration: _update_epub_metadata must add urn:tbl identifier."""

    def _make_opf(self, tmpdir: Path) -> Path:
        opf_xml = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
            'unique-identifier="BookId">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:opf="http://www.idpf.org/2007/opf">'
            '<dc:title>Sample</dc:title>'
            '<dc:language>en</dc:language>'
            '<dc:identifier id="BookId">urn:isbn:0000000000</dc:identifier>'
            '</metadata>'
            '<manifest/><spine/>'
            '</package>'
        )
        path = tmpdir / "content.opf"
        path.write_text(opf_xml, encoding="utf-8")
        return path

    def test_urn_tbl_identifier_added(self):
        from lxml import etree
        from src.core.epub.translator import _update_epub_metadata

        with tempfile.TemporaryDirectory() as tmp:
            opf_path = self._make_opf(Path(tmp))
            tree = etree.parse(str(opf_path))
            _update_epub_metadata(tree, str(opf_path), "French")
            result = opf_path.read_text(encoding="utf-8")

            urns = re.findall(r"urn:tbl:[0-9a-f]+", result)
            assert len(urns) == 1
            assert urns[0] == derive_identifier_urn()

    def test_render_uid_attribute_set(self):
        from lxml import etree
        from src.core.epub.translator import _update_epub_metadata

        with tempfile.TemporaryDirectory() as tmp:
            opf_path = self._make_opf(Path(tmp))
            tree = etree.parse(str(opf_path))
            _update_epub_metadata(tree, str(opf_path), "French")
            result = opf_path.read_text(encoding="utf-8")

            assert 'id="render-uid"' in result

    def test_original_identifier_preserved(self):
        from lxml import etree
        from src.core.epub.translator import _update_epub_metadata

        with tempfile.TemporaryDirectory() as tmp:
            opf_path = self._make_opf(Path(tmp))
            tree = etree.parse(str(opf_path))
            _update_epub_metadata(tree, str(opf_path), "French")
            result = opf_path.read_text(encoding="utf-8")

            assert "urn:isbn:0000000000" in result
            assert 'id="BookId"' in result

    def test_language_updated(self):
        from lxml import etree
        from src.core.epub.translator import _update_epub_metadata

        with tempfile.TemporaryDirectory() as tmp:
            opf_path = self._make_opf(Path(tmp))
            tree = etree.parse(str(opf_path))
            _update_epub_metadata(tree, str(opf_path), "French")
            result = opf_path.read_text(encoding="utf-8")

            assert "<dc:language>fr</dc:language>" in result


# ---------------------------------------------------------------------------
# Integration: TXT writer
# ---------------------------------------------------------------------------

class TestTxtWriterIntegration:
    """Integration: TXT save path must clean text at write boundary."""

    @pytest.mark.asyncio
    async def test_txt_refiner_writes_no_hidden_signature(self):
        """Refine-only TXT path must not add hidden text payloads."""
        import aiofiles

        # Simulate the relevant portion of txt_refiner save logic
        final_text = (
            "Refined paragraph one with several words. "
            "Refined paragraph two with several words. "
            "Refined paragraph three with several words."
        )
        final_text = apply_normalization(final_text)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "out.txt")
            async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                await f.write(final_text)

            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert extract_signature(content) is None
            assert _ANY_WIDTH_ZERO_RE.search(content) is None
