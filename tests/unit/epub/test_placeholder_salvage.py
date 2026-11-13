"""Tests for reusing a paid candidate whose placeholders were damaged."""

from src.core.epub.placeholder_salvage import (
    is_formatting_group,
    keep_intact_block,
    normalize_placeholder_variants,
    plan_placeholder_salvage,
    realign_inline_placeholders,
    salvage_placeholder_candidate,
    verify_salvaged_stream,
)
from src.core.epub.placeholder_validator import PlaceholderValidator

# Two paragraphs; the second one carries an italic span.
TAG_MAP = {
    "[id0]": "<p>",
    "[id1]": "</p><p>",
    "[id2]": "<i>",
    "[id3]": "</i>",
    "[id4]": "</p>",
}
SOURCE = "[id0]First paragraph.[id1]He was [id2]very[id3] tired.[id4]"
VALID = "[id0]Primer párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]"


def _strictly_valid(text, tag_map=TAG_MAP):
    return PlaceholderValidator.validate_strict(text, tag_map)[0]


class TestExactRepairs:
    def test_valid_candidate_is_returned_unchanged(self):
        assert salvage_placeholder_candidate(SOURCE, VALID, TAG_MAP) == VALID

    def test_dropped_leading_and_trailing_block_placeholders_are_restored(self):
        candidate = "Primer párrafo.[id1]Estaba [id2]muy[id3] cansado."
        assert not _strictly_valid(candidate)

        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) == VALID

    def test_boundary_next_to_an_unreadable_block_is_restored(self):
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "</p>",
            "[id2]": "<div class=\"rule\">",
            "[id3]": "</div><p>",
            "[id4]": "</p>",
        }
        source = "[id0]One.[id1]\n[id2][id3]Two.[id4]"
        # The model dropped the empty decorative block and its whitespace.
        candidate = "[id0]Uno.[id1][id3]Dos.[id4]"

        result = salvage_placeholder_candidate(source, candidate, tag_map)

        assert result == "[id0]Uno.[id1]\n[id2][id3]Dos.[id4]"
        assert _strictly_valid(result, tag_map)

    def test_spelling_variants_are_normalized(self):
        candidate = "[ID0]Primer párrafo.[id 1]Estaba ［id2］muy[id03] cansado.[id4]"

        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) == VALID

    def test_variant_is_not_rewritten_into_a_duplicate(self):
        candidate = "[id0]Primer párrafo.[id1]Estaba [id2]muy[id3] cansado [ID3].[id4]"

        assert normalize_placeholder_variants(candidate, TAG_MAP) == candidate
        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) is None

    def test_stray_token_in_a_block_without_placeholders_is_removed(self):
        candidate = "[id0]Primer [id9] párrafo.[id1]Estaba [id2]muy[id3] cansado.[id4]"

        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) == VALID

    def test_intact_blocks_keep_model_word_order(self):
        tag_map = {
            "[id0]": "<p>",
            "[id1]": "<b>",
            "[id2]": "</b>",
            "[id3]": "<i>",
            "[id4]": "</i>",
            "[id5]": "</p>",
        }
        source = "[id0]A [id1]red[id2] [id3]car[id4].[id5]"
        # Legitimate reordering by the model, trailing boundary dropped.
        candidate = "[id0]Un [id3]coche[id4] [id1]rojo[id2]."

        result = salvage_placeholder_candidate(source, candidate, tag_map)

        assert result == candidate + "[id5]"


class TestDamagedRegions:
    def test_merged_paragraphs_are_never_split_heuristically(self):
        candidate = "[id0]Primer párrafo. Estaba [id2]muy[id3] cansado.[id4]"

        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) is None
        plan = plan_placeholder_salvage(SOURCE, candidate, TAG_MAP)
        assert plan.usable and not plan.complete
        assert [region.readable_blocks for region in plan.damaged] == [2]
        assert plan.damaged[0].source_text == (
            "First paragraph.[id1]He was [id2]very[id3] tired."
        )
        assert plan.damaged[0].interpolated is None
        assert plan.damaged_share == 1.0

    def test_missing_inline_placeholder_marks_only_its_block(self):
        candidate = "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado.[id4]"

        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) is None
        plan = plan_placeholder_salvage(SOURCE, candidate, TAG_MAP)
        assert plan.regions[0] == "" and plan.regions[1] == "Primer párrafo."
        assert plan.regions[2] is None
        assert len(plan.damaged) == 1
        region = plan.damaged[0]
        assert region.source_text == "He was [id2]very[id3] tired."
        assert region.readable_blocks == 1
        assert plan.damaged_share == 0.5
        # The surviving closing tag is an anchor; only the opening tag moves.
        assert region.interpolated == "Estaba [id2]muy[id3] cansado."

        filled = plan.assemble({region.candidate_region: region.interpolated})
        assert verify_salvaged_stream(SOURCE, filled, TAG_MAP) == VALID

    def test_interpolation_is_opt_in(self):
        candidate = "[id0]Primer párrafo.[id1]Estaba muy[id3] cansado.[id4]"

        result = salvage_placeholder_candidate(
            SOURCE, candidate, TAG_MAP, interpolate_inline=True,
        )

        assert result == VALID

    def test_duplicated_block_placeholder_is_not_reused(self):
        candidate = "[id0]Primer párrafo.[id1][id1]Estaba [id2]muy[id3] cansado.[id4]"

        plan = plan_placeholder_salvage(SOURCE, candidate, TAG_MAP)

        assert plan.reason == "block_placeholder_duplicated"
        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) is None

    def test_reordered_block_placeholders_are_not_reused(self):
        candidate = "[id1]Primer párrafo.[id0]Estaba [id2]muy[id3] cansado.[id4]"

        assert plan_placeholder_salvage(SOURCE, candidate, TAG_MAP).reason == (
            "block_placeholder_reordered"
        )

    def test_text_invented_between_blocks_is_not_reused(self):
        tag_map = {"[id0]": "<p>", "[id1]": "</p>", "[id2]": "<p>", "[id3]": "</p>"}
        source = "[id0]One.[id1]\n[id2]Two.[id3]"
        candidate = "[id0]Uno.[id1] Nota del traductor [id2]Dos.[id3]"

        assert plan_placeholder_salvage(source, candidate, tag_map).reason == (
            "text_outside_source_blocks"
        )

    def test_placeholder_debris_blocks_reuse(self):
        candidate = "[id0]Primer párrafo.[id1]Estaba [id2]muy(id3) cansado.[id4]"

        plan = plan_placeholder_salvage(SOURCE, candidate, TAG_MAP)

        assert not plan.complete
        assert plan.damaged[0].interpolated is None

    def test_empty_translation_of_a_readable_block_is_damaged(self):
        candidate = "[id0][id1]Estaba [id2]muy[id3] cansado.[id4]"

        plan = plan_placeholder_salvage(SOURCE, candidate, TAG_MAP)

        assert [region.source_text for region in plan.damaged] == ["First paragraph."]

    def test_empty_candidate_has_no_plan(self):
        assert plan_placeholder_salvage(SOURCE, "  ", TAG_MAP).reason == "empty_candidate"


class TestInlineRealignment:
    INLINE = {"[id2]": "<i>", "[id3]": "</i>"}

    def test_intact_block_is_untouched(self):
        block = "Estaba [id2]muy[id3] cansado."

        assert realign_inline_placeholders(
            "He was [id2]very[id3] tired.", block, self.INLINE,
        ) == block

    def test_survivors_are_never_moved(self):
        result = realign_inline_placeholders(
            "He was [id2]very[id3] tired.",
            "Muy[id3] cansado estaba.",
            self.INLINE,
        )

        assert result == "[id2]Muy[id3] cansado estaba."

    def test_position_critical_placeholder_is_not_interpolated(self):
        inline = {"[id2]": "<br/>", "[id3]": "</i>"}

        assert realign_inline_placeholders(
            "Line one[id2]line two[id3].", "Línea uno línea dos[id3].", inline,
        ) is None

    def test_later_placeholders_do_not_drift_to_the_end(self):
        inline = {
            "[id0]": "<i>", "[id1]": "</i>", "[id2]": "<b>", "[id3]": "</b>",
        }
        source = "[id0]Alpha[id1] beta gamma [id2]delta[id3] epsilon zeta eta theta."
        candidate = "Alfa beta gamma delta épsilon zeta eta zeta."

        result = realign_inline_placeholders(source, candidate, inline)

        assert result == "[id0]Alfa[id1] beta gamma [id2]delta[id3] épsilon zeta eta zeta."

    def test_drop_cap_stays_inside_the_first_word(self):
        inline = {"[id1]": "</span>"}

        result = realign_inline_placeholders(
            "T[id1]he night was long.", "La noche fue larga.", inline,
        )

        assert result == "L[id1]a noche fue larga."

    def test_duplicated_placeholder_is_reinserted_once(self):
        result = realign_inline_placeholders(
            "He was [id2]very[id3] tired.",
            "Estaba [id2]muy[id3] cansado[id3].",
            self.INLINE,
        )

        assert result is not None
        assert result.count("[id2]") == 1 and result.count("[id3]") == 1

    def test_keep_intact_block_rejects_missing_placeholder(self):
        assert keep_intact_block("a [id2]b[id3] c", "a b[id3] c") is None


class TestFormattingGroups:
    def test_formatting_tags(self):
        assert is_formatting_group("<i>")
        assert is_formatting_group('</i><span class="x">')
        assert is_formatting_group('<a href="#n1">')

    def test_content_and_block_tags_are_not_formatting(self):
        assert not is_formatting_group("<br/>")
        assert not is_formatting_group("<br>")
        assert not is_formatting_group('<img src="a.png"/>')
        assert not is_formatting_group("</p><p>")
        assert not is_formatting_group("<i>1</i>")
        assert not is_formatting_group("")


class TestVerification:
    def test_inline_placeholder_may_not_change_block(self):
        moved = "[id0]Primer [id2]párrafo.[id1]Estaba muy[id3] cansado.[id4]"

        assert _strictly_valid(moved)
        assert verify_salvaged_stream(SOURCE, moved, TAG_MAP) is None

    def test_readable_block_may_not_become_empty(self):
        emptied = "[id0][id1]Primer párrafo. Estaba [id2]muy[id3] cansado.[id4]"

        assert _strictly_valid(emptied)
        assert verify_salvaged_stream(SOURCE, emptied, TAG_MAP) is None

    def test_stream_without_placeholders(self):
        assert salvage_placeholder_candidate("Hello.", "Hola.", {}) == "Hola."
        assert salvage_placeholder_candidate("Hello.", "Hola [id0].", {}) == "Hola."


class TestSplicedTokens:
    """Removing a stray token must never create a placeholder by accident."""

    def test_token_nested_inside_another_is_not_reused(self):
        # Dropping the stray ``[id9]`` would splice ``[id`` and ``2]`` together.
        candidate = "[id0]Primer párrafo.[id1]Estaba [id[id9]2]muy[id3] cansado.[id4]"

        assert keep_intact_block(
            "He was [id2]very[id3] tired.", "Estaba [id[id9]2]muy[id3] cansado.",
        ) is None
        assert realign_inline_placeholders(
            "He was [id2]very[id3] tired.",
            "Estaba [id[id9]2]muy[id3] cansado.",
            {"[id2]": "<i>", "[id3]": "</i>"},
        ) is None
        assert salvage_placeholder_candidate(SOURCE, candidate, TAG_MAP) is None
        assert salvage_placeholder_candidate(
            SOURCE, candidate, TAG_MAP, interpolate_inline=True,
        ) is None
