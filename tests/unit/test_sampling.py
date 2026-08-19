"""Unit tests for the Sample & Compare pure sampling logic.

Covers `select_sample_indices` (uniform interior selection, edge cases,
deduplication) and `cap_chunk_text` (sentence-boundary capping that never
cuts mid-sentence). Both are pure functions — no I/O, no LLM calls.
"""

import pytest

from src.config import SENTENCE_TERMINATORS
from src.core.sampling import cap_chunk_text, select_sample_indices


class TestSelectSampleIndices:
    """select_sample_indices(total_chunks, n)."""

    def test_raises_when_document_empty(self):
        for total in (0,):
            with pytest.raises(ValueError):
                select_sample_indices(total, 3)

    def test_tiny_documents_return_available_chunks(self):
        assert select_sample_indices(1, 3) == [0]
        assert select_sample_indices(2, 3) == [0, 1]

    def test_raises_when_n_below_one(self):
        with pytest.raises(ValueError):
            select_sample_indices(10, 0)

    def test_excludes_first_and_last_chunk(self):
        # Never pick a title page (0) or trailing metadata (total-1).
        indices = select_sample_indices(20, 5)
        assert all(1 <= idx <= 18 for idx in indices)
        assert 0 not in indices
        assert 19 not in indices

    def test_returns_ascending_unique(self):
        indices = select_sample_indices(50, 8)
        assert indices == sorted(indices)
        assert len(indices) == len(set(indices))

    def test_returns_all_interior_when_interior_smaller_than_n(self):
        # total=5 → interior is [1, 2, 3]; asking for 10 yields exactly those.
        assert select_sample_indices(5, 10) == [1, 2, 3]

    def test_distributes_across_the_interior(self):
        # Spread, not clustered at one end.
        indices = select_sample_indices(100, 5)
        assert len(indices) == 5
        assert indices[0] < 30
        assert indices[-1] > 70

    def test_dedup_preserves_order_on_tight_documents(self):
        # Small interior where the rounding formula can collide: result must
        # still be unique and ascending.
        indices = select_sample_indices(6, 5)
        assert indices == sorted(set(indices))
        assert all(1 <= idx <= 4 for idx in indices)


class TestCapChunkText:
    """cap_chunk_text(text, max_chars)."""

    def test_raises_on_non_positive_budget(self):
        with pytest.raises(ValueError):
            cap_chunk_text("anything", 0)

    def test_short_text_returned_unchanged(self):
        text = "Just one sentence."
        capped, was_capped = cap_chunk_text(text, 1000)
        assert capped == text
        assert was_capped is False

    def test_cuts_at_sentence_boundary_within_budget(self):
        text = "First sentence. Second sentence that overflows the budget."
        capped, was_capped = cap_chunk_text(text, 20)
        assert was_capped is True
        assert capped == "First sentence."
        # Never ends mid-sentence: last char is a terminator.
        assert capped[-1] in {term.strip()[-1] for term in SENTENCE_TERMINATORS if term.strip()}

    def test_extends_forward_when_no_boundary_within_budget(self):
        # No terminator before max_chars → extend to the first one after it
        # rather than chopping mid-sentence (result may exceed the cap).
        text = "aaaaaaaaaaaaaaaaaaaa bbbbb. ccccc."
        capped, was_capped = cap_chunk_text(text, 5)
        assert was_capped is True
        assert capped == "aaaaaaaaaaaaaaaaaaaa bbbbb."

    def test_no_terminator_anywhere_returns_whole_text(self):
        text = "word " * 50  # long, no sentence terminator at all
        capped, was_capped = cap_chunk_text(text, 10)
        assert capped == text
        assert was_capped is False

    def test_capped_text_is_never_longer_than_source(self):
        text = "Alpha. Beta. Gamma. Delta. Epsilon."
        capped, _ = cap_chunk_text(text, 12)
        assert len(capped) <= len(text)
