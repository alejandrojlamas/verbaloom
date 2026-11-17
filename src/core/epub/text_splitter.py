"""
Hierarchical text splitting strategies for HTML content.

This module provides text splitting functionality that tries different strategies
in order of preference: sentences -> punctuation -> newlines -> force split.
"""
import re
from itertools import chain
from typing import List

from src.common.placeholder_format import PlaceholderFormat
from src.core.chunking.token_chunker import TokenChunker


class TextSplitter:
    """Splits text using hierarchical strategies (sentences -> punctuation -> newlines -> force)."""

    def __init__(self, max_tokens: int, token_chunker: TokenChunker):
        """
        Initialize the TextSplitter.

        Args:
            max_tokens: Maximum tokens per segment
            token_chunker: TokenChunker instance for counting tokens
        """
        self.max_tokens = max_tokens
        self.token_chunker = token_chunker

    def split_oversized_segment(self, segment: str) -> List[str]:
        """
        Split oversized segment hierarchically.

        Tries strategies in order:
        1. Sentence boundaries
        2. Punctuation (; : ,)
        3. Newlines
        4. Force split at token boundaries

        Args:
            segment: Text segment to split

        Returns:
            List of smaller segments that fit within max_tokens
        """
        # Try sentence splitting first
        sentence_segments = self._split_on_sentences(segment)

        result_segments = []
        for sent_seg in sentence_segments:
            sent_tokens = self.token_chunker.count_tokens(sent_seg)

            if sent_tokens <= self.max_tokens:
                result_segments.append(sent_seg)
            else:
                # Sentence is still too large, try punctuation split
                punct_segments = self._split_on_punctuation(sent_seg)

                for punct_seg in punct_segments:
                    punct_tokens = self.token_chunker.count_tokens(punct_seg)

                    if punct_tokens <= self.max_tokens:
                        result_segments.append(punct_seg)
                    else:
                        # Still too large, try newline split
                        newline_segments = self._split_on_newlines(punct_seg)

                        for nl_seg in newline_segments:
                            nl_tokens = self.token_chunker.count_tokens(nl_seg)

                            if nl_tokens <= self.max_tokens:
                                result_segments.append(nl_seg)
                            else:
                                # Last resort: force split at max_tokens
                                force_segments = self._force_split_at_tokens(nl_seg)
                                result_segments.extend(force_segments)

        return result_segments

    def _split_with_strategy(self, text: str, boundary: str) -> List[str]:
        """Group contiguous source slices, retaining the separator at each cut."""
        segments = []
        start = previous = 0
        ends = (match.end() for match in re.finditer(boundary, text))
        for end in chain(ends, [len(text)]):
            if (start < previous < end
                    and self.token_chunker.count_tokens(text[start:end]) > self.max_tokens):
                segments.append(text[start:previous])
                start = previous
            previous = end
        if start < len(text):
            segments.append(text[start:])
        return segments or [text]

    def _split_on_sentences(self, text: str) -> List[str]:
        return self._split_with_strategy(text, r"(?<=[.!?])(?:\s+(?=\S)|(?=\[\[))")

    def _split_on_punctuation(self, text: str) -> List[str]:
        return self._split_with_strategy(text, r"(?<=[;:,])(?:\s+(?=\S)|(?=\[\[))")

    def _split_on_newlines(self, text: str) -> List[str]:
        return self._split_with_strategy(text, r"\n")

    def _force_split_at_tokens(self, text: str) -> List[str]:
        """
        Force split text at max_tokens boundary as last resort.
        Tries to split at word boundaries when possible.

        Returns:
            List of force-split segments
        """
        segments = []
        remaining = text
        placeholders = iter(PlaceholderFormat.from_config().find_all(text))
        placeholder = next(placeholders, None)

        while remaining:
            # If remaining text fits, we're done
            if self.token_chunker.count_tokens(remaining) <= self.max_tokens:
                segments.append(remaining)
                break

            # Binary search for the largest prefix that fits
            left, right = 0, len(remaining)
            best_pos = left

            while left <= right:
                mid = (left + right) // 2
                prefix = remaining[:mid]
                token_count = self.token_chunker.count_tokens(prefix)

                if token_count <= self.max_tokens:
                    best_pos = mid
                    left = mid + 1
                else:
                    right = mid - 1

            # Find word boundary near best_pos
            split_pos = self._find_word_boundary_near(remaining, best_pos)

            # Avoid infinite loop - ensure we make progress
            if split_pos == 0:
                split_pos = max(1, best_pos)

            # A placeholder split across model requests cannot be renumbered or
            # restored. Keep it atomic even if the descriptor itself is large.
            offset = len(text) - len(remaining)
            boundary = offset + split_pos
            while placeholder and placeholder[1] <= boundary:
                placeholder = next(placeholders, None)
            if placeholder and placeholder[0] < boundary < placeholder[1]:
                split_pos = (placeholder[0] if placeholder[0] > offset else placeholder[1]) - offset

            segments.append(remaining[:split_pos])
            remaining = remaining[split_pos:]

        return segments

    def _find_word_boundary_near(self, text: str, pos: int) -> int:
        """
        Find a word boundary (space) near the given position.
        Looks backward first to avoid cutting words.

        Args:
            text: Text to search in
            pos: Position to search near

        Returns:
            Position of word boundary
        """
        if pos >= len(text):
            return len(text)

        # Keep the entire source slice within the token budget.
        for i in range(pos, max(0, pos - 50), -1):
            if text[i - 1].isspace():
                return i
        return pos
