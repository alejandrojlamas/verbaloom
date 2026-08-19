"""
Token-based text chunking with natural boundary preservation.

This module provides intelligent text chunking based on token counts
using tiktoken, while respecting natural text boundaries (paragraphs and sentences).
"""
import re
from typing import List, Dict
import tiktoken

from src.config import SENTENCE_TERMINATORS


_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_NUMERICISH_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|M|B|K))?|"
    r"[-+]?\d+(?:[.,]\d+)?\s*[+/-]\s*[.,]?\d+|"
    r"[-+]?\d+(?:[.,]\d+)?\s*(?:x|×|·)\s*10\^?[-+]?\d+|"
    r"[A-Za-z]{1,8}\d*|"
    r"[A-Za-z0-9_.-]+\s*=\s*.+"
    r")\s*$"
)
_STRUCTURED_LINE_CUE_RE = re.compile(
    r"\b(table|tabla|figure|figura|caption|leyenda|bleu|rouge|meteor|cider|nist|glue|mnli|wikisql)\b|[|=∑Σ√βγδ∆]",
    re.IGNORECASE,
)
_LOOSE_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\s*\d+)?")


class _ApproxEncoder:
    """Offline fallback encoder used when tiktoken cannot load its encoding.

    tiktoken downloads encoding files on first use; on an offline machine (or
    behind a firewall) that download fails and, without this fallback, every
    chunking call crashes. The approximation (~4 chars/token, floor of one
    token per whitespace word) is close enough for chunk sizing, which only
    needs relative budgets, not exact token counts.
    """

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        approx = max(len(text) // 4, len(text.split()))
        return [0] * approx


_SHARED_ENCODER = None
_SHARED_ENCODER_IS_FALLBACK = False


def _get_shared_encoder():
    """Load cl100k_base once; fall back to the offline approximation on failure."""
    global _SHARED_ENCODER, _SHARED_ENCODER_IS_FALLBACK
    if _SHARED_ENCODER is None:
        try:
            _SHARED_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _SHARED_ENCODER = _ApproxEncoder()
            _SHARED_ENCODER_IS_FALLBACK = True
    return _SHARED_ENCODER


def using_fallback_encoder() -> bool:
    """True when chunking runs on the offline approximation instead of tiktoken.

    Token counts (and therefore chunk boundaries) differ from real cl100k_base
    in that case; characterization tests pinned to golden chunk sequences
    should skip rather than fail spuriously.
    """
    _get_shared_encoder()
    return _SHARED_ENCODER_IS_FALLBACK


class TokenChunker:
    """
    Token-based text chunker that respects natural boundaries.

    Uses a soft limit approach: accumulates content until reaching ~80% of max tokens,
    then completes at the next natural boundary (paragraph or sentence).
    """

    def __init__(self, max_tokens: int = 800, soft_limit_ratio: float = 0.8):
        """
        Initialize the TokenChunker.

        Args:
            max_tokens: Maximum tokens per chunk (hard limit)
            soft_limit_ratio: Ratio at which to start looking for boundaries (default 0.8 = 80%)
        """
        self.max_tokens = max_tokens
        self.soft_limit = int(max_tokens * soft_limit_ratio)
        self.encoder = _get_shared_encoder()

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs using double newlines.

        Args:
            text: Input text

        Returns:
            List of paragraphs (preserving single newlines within)
        """
        # Split on double newlines (or more)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs but preserve whitespace-only ones as empty markers
        return [p for p in paragraphs if p.strip()]

    def split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """
        Split a paragraph into sentences for finer-grained chunking.

        Used when a single paragraph exceeds max_tokens.

        Args:
            paragraph: Input paragraph text

        Returns:
            List of sentences
        """
        # Create regex pattern from sentence terminators
        sorted_terminators = sorted(list(SENTENCE_TERMINATORS), key=len, reverse=True)
        escaped_terminators = [re.escape(t) for t in sorted_terminators]
        pattern = '|'.join(escaped_terminators)

        sentences = []
        last_end = 0

        for match in re.finditer(pattern, paragraph):
            end = match.end()
            sentence = paragraph[last_end:end].strip()
            if sentence:
                sentences.append(sentence)
            last_end = end

        # Add remaining text if any
        remaining = paragraph[last_end:].strip()
        if remaining:
            sentences.append(remaining)

        # If no sentences found (no terminators), return the whole paragraph
        if not sentences and paragraph.strip():
            sentences = [paragraph.strip()]

        return sentences

    def _is_markdown_table_block(self, text: str) -> bool:
        lines = [line for line in (text or "").splitlines() if line.strip()]
        table_rows = [line for line in lines if _MARKDOWN_TABLE_ROW_RE.match(line)]
        if len(table_rows) < 2:
            return False
        if any(_MARKDOWN_TABLE_SEPARATOR_RE.match(line) for line in lines[:5]):
            return True
        return len(table_rows) >= max(3, len(lines) // 2)

    def _is_structured_line_block(self, text: str) -> bool:
        if self._is_markdown_table_block(text):
            return True
        lines = [line for line in (text or "").splitlines() if line.strip()]
        if len(lines) < 5:
            return False
        numericish = sum(
            1
            for line in lines
            if _NUMERICISH_LINE_RE.match(line) or len(_LOOSE_NUMBER_RE.findall(line)) >= 2
        )
        return numericish >= max(4, len(lines) // 2) and bool(_STRUCTURED_LINE_CUE_RE.search(text))

    def _chunk_structured_lines(self, text: str) -> List[str]:
        """Split oversized structured blocks by full lines, never by decimals."""
        if self.count_tokens(text) <= self.max_tokens * 3:
            return [text]

        chunks: List[str] = []
        current_lines: List[str] = []
        current_tokens = 0
        for line in text.splitlines():
            line_tokens = self.count_tokens(line)
            separator_tokens = self.count_tokens("\n") if current_lines else 0
            if current_lines and current_tokens + separator_tokens + line_tokens > self.max_tokens:
                chunks.append("\n".join(current_lines))
                current_lines = [line]
                current_tokens = line_tokens
            else:
                current_lines.append(line)
                current_tokens += separator_tokens + line_tokens
        if current_lines:
            chunks.append("\n".join(current_lines))
        return chunks

    def _chunk_units(self, units: List[str], separator: str = "\n\n") -> List[str]:
        """
        Chunk a list of text units (paragraphs or sentences) into appropriately sized chunks.

        Args:
            units: List of text units to chunk
            separator: Separator to use when joining units

        Returns:
            List of chunk strings
        """
        # Minimum chunk size threshold - chunks smaller than this will be merged
        # with adjacent content rather than saved separately
        min_chunk_tokens = int(self.max_tokens * 0.25)  # 25% of max_tokens

        chunks = []
        current_units = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.count_tokens(unit)

            # If single unit exceeds max, we need to handle it specially
            if unit_tokens > self.max_tokens:
                # If current chunk is too small, don't save it separately
                # Instead, prepend it to the first sentence chunk
                prefix_units = []
                if current_units and current_tokens < min_chunk_tokens:
                    prefix_units = current_units
                    current_units = []
                    current_tokens = 0
                elif current_units:
                    # Current chunk is big enough, save it
                    chunks.append(separator.join(current_units))
                    current_units = []
                    current_tokens = 0

                if self._is_structured_line_block(unit):
                    structured_chunks = self._chunk_structured_lines(unit)
                    if prefix_units and structured_chunks:
                        prefix_text = separator.join(prefix_units)
                        if self.count_tokens(prefix_text) + self.count_tokens(structured_chunks[0]) <= self.max_tokens:
                            structured_chunks[0] = prefix_text + separator + structured_chunks[0]
                        else:
                            chunks.append(prefix_text)
                    elif prefix_units:
                        chunks.append(separator.join(prefix_units))
                    chunks.extend(structured_chunks)
                    continue

                # If it's a paragraph, try splitting into sentences
                sentences = self.split_paragraph_into_sentences(unit)
                if len(sentences) > 1:
                    # Recursively chunk sentences
                    sentence_chunks = self._chunk_units(sentences, separator=" ")

                    # Prepend small prefix to first sentence chunk if exists
                    if prefix_units and sentence_chunks:
                        prefix_text = separator.join(prefix_units)
                        prefix_tokens = self.count_tokens(prefix_text)
                        first_chunk_tokens = self.count_tokens(sentence_chunks[0])

                        # Only merge if combined size is reasonable
                        if prefix_tokens + first_chunk_tokens <= self.max_tokens:
                            sentence_chunks[0] = prefix_text + separator + sentence_chunks[0]
                        else:
                            # Prefix too big, save it separately
                            chunks.append(prefix_text)
                    elif prefix_units:
                        # No sentence chunks but have prefix
                        chunks.append(separator.join(prefix_units))

                    chunks.extend(sentence_chunks)
                else:
                    # Can't split further, prepend prefix if any
                    if prefix_units:
                        chunks.append(separator.join(prefix_units) + separator + unit)
                    else:
                        chunks.append(unit)
                continue

            # Check if adding this unit would exceed limits
            potential_tokens = current_tokens + unit_tokens
            if current_units:
                # Account for separator
                potential_tokens += self.count_tokens(separator)

            # If we're past soft limit, check if we should start a new chunk
            if current_tokens >= self.soft_limit and potential_tokens > self.max_tokens:
                # Save current chunk and start new one
                chunks.append(separator.join(current_units))
                current_units = [unit]
                current_tokens = unit_tokens
            elif potential_tokens > self.max_tokens:
                # Would exceed hard limit, start new chunk
                if current_units:
                    chunks.append(separator.join(current_units))
                current_units = [unit]
                current_tokens = unit_tokens
            else:
                # Add to current chunk
                current_units.append(unit)
                current_tokens = potential_tokens

        # Don't forget the last chunk
        if current_units:
            chunks.append(separator.join(current_units))

        return chunks

    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into chunks with context preservation.

        Main algorithm:
        1. Split into paragraphs
        2. Accumulate until soft_limit (~80%)
        3. If next paragraph would exceed max_tokens, finalize chunk
        4. If single paragraph > max_tokens, split into sentences
        5. Return chunks with context_before/main_content/context_after

        Args:
            text: Input text to chunk

        Returns:
            List of chunk dictionaries with keys:
            - context_before: Last paragraph of previous chunk (for context)
            - main_content: Main content to translate
            - context_after: First paragraph of next chunk (for context)
        """
        if not text or not text.strip():
            return []

        # Split into paragraphs
        paragraphs = self.split_into_paragraphs(text)

        if not paragraphs:
            return []

        # Chunk paragraphs
        raw_chunks = self._chunk_units(paragraphs, separator="\n\n")

        if not raw_chunks:
            return []

        # Build structured chunks with context
        structured_chunks = []

        for i, chunk_content in enumerate(raw_chunks):
            # Context before: last part of previous chunk
            if i > 0:
                prev_paragraphs = self.split_into_paragraphs(raw_chunks[i - 1])
                context_before = prev_paragraphs[-1] if prev_paragraphs else ""
            else:
                context_before = ""

            # Context after: first part of next chunk
            if i < len(raw_chunks) - 1:
                next_paragraphs = self.split_into_paragraphs(raw_chunks[i + 1])
                context_after = next_paragraphs[0] if next_paragraphs else ""
            else:
                context_after = ""

            structured_chunks.append({
                "context_before": context_before,
                "main_content": chunk_content,
                "context_after": context_after
            })

        return structured_chunks

    def get_stats(self, chunks: List[Dict[str, str]]) -> Dict:
        """
        Get statistics about the chunked text.

        Args:
            chunks: List of chunk dictionaries from chunk_text()

        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_tokens": 0,
                "min_tokens": 0,
                "max_tokens": 0,
                "chunks_in_range": 0,
                "compliance_rate": 0.0
            }

        token_counts = [self.count_tokens(c["main_content"]) for c in chunks]

        # Calculate how many are within acceptable range (soft_limit to max_tokens)
        in_range = sum(1 for t in token_counts if t <= self.max_tokens)

        return {
            "total_chunks": len(chunks),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "chunks_in_range": in_range,
            "compliance_rate": in_range / len(chunks) * 100
        }
