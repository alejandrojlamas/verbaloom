"""
Centralized HTTP 429 handling with API key rotation support.

Used by cloud LLM providers to deduplicate the rate-limit retry/backoff logic
that was previously copy-pasted across all providers (gemini, openai-compatible,
openrouter, mistral, deepseek, poe).

Behavior on 429:
    1. Compute wait time from response headers (Retry-After or X-RateLimit-Reset)
    2. Mark the failed key as throttled in the pool
    3. If another key is available: rotate without sleeping (caller's next
       acquire() returns the new key)
    4. Else if attempts remain: sleep until the next key becomes available
    5. Else: raise RateLimitError to trigger upstream auto-pause
"""

import asyncio
import time
from typing import Callable, Mapping, Optional

from .exceptions import RateLimitError
from .key_pool import KeyPool


def compute_wait_time(headers: Mapping[str, str], attempt: int) -> int:
    """Derive a wait time in seconds from a 429 response.

    Priority:
        1. Retry-After header (seconds) — standard
        2. X-RateLimit-Reset header (UTC ms timestamp) — used by OpenRouter
        3. Exponential backoff fallback: min(2 ** (attempt + 2), 60)

    httpx returns case-insensitive headers, but we check both casings to stay
    robust against plain dict mocks in tests.

    Returns:
        Wait time in seconds, always >= 1.
    """
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return max(int(retry_after), 1)
        except (ValueError, TypeError):
            pass

    reset_ms = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    if reset_ms:
        try:
            wait = int((int(reset_ms) - time.time() * 1000) / 1000) + 1
            return max(min(wait, 65), 1)
        except (ValueError, TypeError):
            pass

    return min(2 ** (attempt + 2), 60)


async def handle_rate_limit(
    pool: KeyPool,
    failed_key: str,
    response_headers: Mapping[str, str],
    attempt: int,
    max_attempts: int,
    log_callback: Optional[Callable] = None,
) -> None:
    """Mark a key throttled, rotate if possible, sleep if needed, raise if exhausted.

    Caller pattern:

        if response.status_code == 429:
            await handle_rate_limit(
                self._key_pool, current_key, response.headers,
                attempt, MAX_TRANSLATION_ATTEMPTS, self.log_callback,
            )
            continue

    Args:
        pool: KeyPool to update.
        failed_key: The key that just received a 429.
        response_headers: Headers from the 429 response.
        attempt: Current attempt index (0-based).
        max_attempts: Max total attempts (typically MAX_TRANSLATION_ATTEMPTS).
        log_callback: Optional structured logging callback (key, message).

    Returns:
        None — caller should `continue` the retry loop after this returns.

    Raises:
        RateLimitError: If `attempt == max_attempts - 1` AND no other key is
            currently available.
    """
    wait_time = compute_wait_time(response_headers, attempt)
    await pool.mark_throttled(failed_key, time.monotonic() + wait_time)

    provider = pool.provider_name
    failed_idx = pool.index_of(failed_key)
    pool_size = pool.size

    # Fast path: another key is ready, rotate without sleeping.
    if pool_size > 1 and await pool.has_available():
        _log(
            log_callback,
            "llm_key_rotated",
            f"🔄 [{provider}] key #{failed_idx}/{pool_size} rate-limited "
            f"(retry-after {wait_time}s), rotating to next key",
        )
        return

    # All keys throttled (or single-key pool).
    if attempt < max_attempts - 1:
        remaining = await pool.time_until_next_available()
        sleep_for = max(int(remaining) if remaining > 0 else wait_time, 1)
        _log(
            log_callback,
            "llm_rate_limit",
            f"⚠️ [{provider}] all {pool_size} key(s) rate-limited "
            f"(attempt {attempt + 1}/{max_attempts}), waiting {sleep_for}s...",
        )
        await asyncio.sleep(sleep_for)
        return

    raise RateLimitError(
        f"{provider} rate limit exceeded after {max_attempts} attempts "
        f"(all {pool_size} key(s) exhausted)",
        retry_after=wait_time,
        provider=provider,
    )


def _log(log_callback: Optional[Callable], event: str, message: str) -> None:
    """Best-effort logging: callback if provided, else stdout via print."""
    if log_callback:
        try:
            log_callback(event, message)
            return
        except Exception:
            pass
    print(message)
