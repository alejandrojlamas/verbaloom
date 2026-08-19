"""
Sampling utilities for the Sample & Compare feature.

Selects N representative chunks from a chunked document (excluding the first
and last chunk so we never pick a title page or trailing metadata), and caps
each chunk's main_content at a max character budget at the closest sentence
boundary.

Pure logic — no I/O, no LLM calls. Easy to unit-test.
"""
from typing import List, Tuple

from src.config import SENTENCE_TERMINATORS


def select_sample_indices(total_chunks: int, n: int) -> List[int]:
    """
    Uniformly distribute up to N indices across [1, total_chunks - 2], i.e.
    excluding the first (0) and last (total_chunks - 1) chunks when the
    document is large enough to have interior chunks. For tiny documents,
    return every available chunk instead of failing the Sample UI.

    Formula: round((i + 1) * (T - 1) / (N + 1)) for i in [0, N - 1]
    Where T = total_chunks. The result is clamped to [1, T - 2] and deduplicated
    while preserving order.

    Args:
        total_chunks: Number of chunks in the source document (T).
        n: Number of samples to draw (N). Must be >= 1.

    Returns:
        A list of unique chunk indices, in ascending order.

    Raises:
        ValueError: If total_chunks < 1.
        ValueError: If n < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if total_chunks < 1:
        raise ValueError("document too small for sampling")
    if total_chunks < 3:
        return list(range(total_chunks))[:n]

    interior_lo, interior_hi = 1, total_chunks - 2

    # When the interior is smaller than n, return every interior index.
    interior_count = interior_hi - interior_lo + 1
    if interior_count <= n:
        return list(range(interior_lo, interior_hi + 1))

    raw = [round((i + 1) * (total_chunks - 1) / (n + 1)) for i in range(n)]
    clamped = [max(interior_lo, min(interior_hi, idx)) for idx in raw]

    seen = set()
    deduped: List[int] = []
    for idx in clamped:
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def cap_chunk_text(text: str, max_chars: int) -> Tuple[str, bool]:
    """
    Cap `text` near `max_chars` characters, NEVER cutting mid-sentence.

    Strategy:
      1. If `text` already fits, return as-is.
      2. Otherwise, cut at the last sentence boundary at or before `max_chars`.
      3. If no boundary exists within the budget, extend to the first boundary
         AFTER `max_chars` (the result may exceed the cap by one sentence).
      4. If the text contains no sentence boundary at all, return it whole.

    `max_chars` is therefore a soft target, not a hard ceiling — the user's
    rule is "no half sentences in samples". A sentence boundary is any element
    of `SENTENCE_TERMINATORS` (``.``, ``!``, ``?``, ``:``, plus quoted variants).

    Args:
        text: The chunk's main content.
        max_chars: Approximate character budget. Must be > 0.

    Returns:
        Tuple of (capped_text, was_capped). `was_capped` is True only when the
        result is strictly shorter than `text`.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    if len(text) <= max_chars:
        return text, False

    window = text[:max_chars]
    best_cut = -1
    for term in SENTENCE_TERMINATORS:
        idx = window.rfind(term)
        if idx >= 0:
            end = idx + len(term)
            if end > best_cut:
                best_cut = end

    if best_cut > 0:
        return window[:best_cut].rstrip(), True

    # No sentence boundary inside the budget — extend forward to the next one
    # rather than chopping mid-sentence.
    after = text[max_chars:]
    best_after = -1
    best_after_term_len = 0
    for term in SENTENCE_TERMINATORS:
        idx = after.find(term)
        if idx >= 0 and (best_after == -1 or idx < best_after):
            best_after = idx
            best_after_term_len = len(term)

    if best_after >= 0:
        end = max_chars + best_after + best_after_term_len
        return text[:end].rstrip(), True

    # Pathological input (no terminator anywhere): keep the whole thing.
    return text, False
