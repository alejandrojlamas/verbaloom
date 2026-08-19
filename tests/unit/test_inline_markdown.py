"""Tests for the shared inline-markdown codec (src/common/inline_markdown.py)."""
import pytest

from src.common.inline_markdown import (
    InlineSegment,
    has_inline_markdown,
    parse_inline_markdown,
    segments_to_markdown,
    strip_inline_markdown,
)


class TestEncoding:
    def test_plain_text_passes_through(self):
        assert segments_to_markdown([InlineSegment("hola mundo")]) == "hola mundo"

    def test_bold_italic_and_both(self):
        segments = [
            InlineSegment("a "),
            InlineSegment("b", bold=True),
            InlineSegment(" c "),
            InlineSegment("d", italic=True),
            InlineSegment(" e "),
            InlineSegment("f", bold=True, italic=True),
        ]
        assert segments_to_markdown(segments) == "a **b** c *d* e ***f***"

    def test_adjacent_runs_with_same_format_are_merged(self):
        # Word splits runs arbitrarily; without merging we would emit
        # "**Ho****la**" or "**Ho** **la**".
        segments = [
            InlineSegment("Ho", bold=True),
            InlineSegment("la", bold=True),
            InlineSegment(" mundo"),
        ]
        assert segments_to_markdown(segments) == "**Hola** mundo"

    def test_whitespace_moved_outside_markers(self):
        segments = [InlineSegment("negrita ", bold=True), InlineSegment("normal")]
        assert segments_to_markdown(segments) == "**negrita** normal"

    def test_link_with_inner_emphasis(self):
        segments = [
            InlineSegment("ver "),
            InlineSegment("la guía", italic=True, href="https://x.io/a"),
        ]
        assert segments_to_markdown(segments) == "ver [*la guía*](https://x.io/a)"

    def test_empty_or_whitespace_only_emphasis_not_wrapped(self):
        assert segments_to_markdown([InlineSegment("   ", bold=True)]) == "   "


class TestDecoding:
    def test_round_trip_basic(self):
        text = "Hola **mundo** y *adiós* y ***ambos*** fin"
        segments = parse_inline_markdown(text)
        assert segments_to_markdown(segments) == text

    def test_link_round_trip(self):
        text = "ver [la nota](https://nota.mx/ref) ahora"
        segments = parse_inline_markdown(text)
        link = [s for s in segments if s.href]
        assert len(link) == 1
        assert link[0].text == "la nota"
        assert link[0].href == "https://nota.mx/ref"
        assert segments_to_markdown(segments) == text

    def test_link_text_with_literal_brackets_decodes(self):
        text = "ver [[23]](../Text/notas.xhtml#nt23) ahora"
        segments = parse_inline_markdown(text)
        link = [s for s in segments if s.href]
        assert len(link) == 1
        assert link[0].text == "[23]"
        assert link[0].href == "../Text/notas.xhtml#nt23"
        assert "".join(s.text for s in segments) == "ver [23] ahora"

    def test_unmatched_markers_stay_literal(self):
        for text in ("2 * 3 = 6", "*solo", "fin**", "a ** b", "nota [sin cerrar"):
            segments = parse_inline_markdown(text)
            assert all(not s.has_formatting for s in segments)
            assert "".join(s.text for s in segments) == text

    def test_no_markers_fast_path(self):
        segments = parse_inline_markdown("texto plano sin nada")
        assert segments == [InlineSegment(text="texto plano sin nada")]

    def test_strip(self):
        assert (
            strip_inline_markdown("Hola **mundo** [link](http://a.b)")
            == "Hola mundo link"
        )

    def test_has_inline_markdown(self):
        assert has_inline_markdown("con **negrita**")
        assert has_inline_markdown("con [link](http://a.b)")
        assert not has_inline_markdown("2 * 3 = 6")
        assert not has_inline_markdown("texto plano")
        assert not has_inline_markdown("")


class TestToleranceUnderLLMDamage:
    """The decoder must degrade gracefully when the LLM perturbs markers."""

    def test_marker_with_space_inside_is_literal(self):
        segments = parse_inline_markdown("malo ** espacio ** aqui")
        assert all(not s.bold for s in segments)

    def test_nested_emphasis_inside_link_text(self):
        segments = parse_inline_markdown("[**fuerte**](http://a.b)")
        assert len(segments) == 1
        assert segments[0].bold and segments[0].href == "http://a.b"

    def test_multiline_text_keeps_paragraph_newlines(self):
        text = "línea uno **b**\nlínea dos"
        joined = "".join(s.text for s in parse_inline_markdown(text))
        assert "\n" in joined

    def test_many_inline_markers_do_not_recurse(self):
        text = " ".join(["*film*"] * 1500)
        segments = parse_inline_markdown(text)
        assert "".join(s.text for s in segments).count("film") == 1500
        assert sum(1 for s in segments if s.italic) == 1500

    def test_many_malformed_brackets_are_literal(self):
        text = ("[" * 2000) + "texto sin link"
        segments = parse_inline_markdown(text)
        assert all(not s.has_formatting for s in segments)
        assert "".join(s.text for s in segments) == text
