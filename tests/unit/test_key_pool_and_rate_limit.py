"""
Unit tests for the API key pool and centralized rate-limit handler.

Covers:
    - KeyPool: construction, dedup, round-robin, throttle skip, exhaustion
    - compute_wait_time: Retry-After / X-RateLimit-Reset / fallback
    - handle_rate_limit: rotation no-sleep, single-key sleep+raise, all-throttled
    - LLMProvider.api_key backwards-compat property
    - base.normalize_api_keys: comma/newline parsing
"""
import asyncio
import time

import pytest

from src.core.llm.base import normalize_api_keys
from src.core.llm.exceptions import RateLimitError
from src.core.llm.key_pool import KeyPool
from src.core.llm.rate_limit_handler import compute_wait_time, handle_rate_limit


# ---------------------------------------------------------------------------
# KeyPool
# ---------------------------------------------------------------------------

class TestKeyPool:
    def test_single_key(self):
        pool = KeyPool("only-key", provider_name="x")
        assert pool.size == 1
        assert pool.peek() == "only-key"

    def test_dedup_and_filter(self):
        pool = KeyPool(["a", "a", "", "b", " "], provider_name="x")
        # Note: " " (space) is truthy as a string and survives the empty filter,
        # but the dedup keeps order: ["a", "b", " "]. Acceptable: a stray space
        # is the user's input mistake; we don't aggressively trim here.
        # We document this in the test so a future refactor doesn't surprise.
        assert pool.size == 3
        assert pool.peek() == "a"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            KeyPool([], provider_name="x")
        with pytest.raises(ValueError):
            KeyPool(["", ""], provider_name="x")

    @pytest.mark.asyncio
    async def test_round_robin(self):
        pool = KeyPool(["a", "b", "c"], provider_name="x")
        seq = [await pool.acquire() for _ in range(7)]
        assert seq == ["a", "b", "c", "a", "b", "c", "a"]

    @pytest.mark.asyncio
    async def test_throttle_skip(self):
        pool = KeyPool(["a", "b", "c"], provider_name="x")
        await pool.mark_throttled("b", time.monotonic() + 60)
        seq = [await pool.acquire() for _ in range(5)]
        assert "b" not in seq

    @pytest.mark.asyncio
    async def test_all_throttled_returns_soonest(self):
        pool = KeyPool(["a", "b"], provider_name="x")
        now = time.monotonic()
        await pool.mark_throttled("a", now + 100)
        await pool.mark_throttled("b", now + 50)
        # Both throttled — acquire should still return one (the one expiring first)
        key = await pool.acquire()
        assert key == "b"

    @pytest.mark.asyncio
    async def test_has_available(self):
        pool = KeyPool(["a", "b"], provider_name="x")
        assert await pool.has_available() is True
        await pool.mark_throttled("a", time.monotonic() + 60)
        assert await pool.has_available() is True  # b still free
        await pool.mark_throttled("b", time.monotonic() + 60)
        assert await pool.has_available() is False

    @pytest.mark.asyncio
    async def test_time_until_next_available(self):
        pool = KeyPool(["a"], provider_name="x")
        assert await pool.time_until_next_available() == 0.0
        await pool.mark_throttled("a", time.monotonic() + 30)
        remaining = await pool.time_until_next_available()
        assert 28 <= remaining <= 31


# ---------------------------------------------------------------------------
# compute_wait_time
# ---------------------------------------------------------------------------

class TestComputeWaitTime:
    def test_retry_after(self):
        assert compute_wait_time({"Retry-After": "15"}, attempt=0) == 15
        assert compute_wait_time({"retry-after": "7"}, attempt=0) == 7

    def test_retry_after_invalid_falls_back(self):
        # Non-numeric retry-after → ignored, fall through to backoff
        assert compute_wait_time({"Retry-After": "junk"}, attempt=0) == 4

    def test_x_ratelimit_reset(self):
        future_ms = int((time.time() + 30) * 1000)
        w = compute_wait_time({"X-RateLimit-Reset": str(future_ms)}, attempt=0)
        assert 28 <= w <= 32

    def test_x_ratelimit_reset_capped_at_65(self):
        far_future_ms = int((time.time() + 600) * 1000)
        w = compute_wait_time({"X-RateLimit-Reset": str(far_future_ms)}, attempt=0)
        assert w == 65

    def test_exp_backoff_fallback(self):
        assert compute_wait_time({}, attempt=0) == 4
        assert compute_wait_time({}, attempt=1) == 8
        assert compute_wait_time({}, attempt=2) == 16
        # Capped at 60
        assert compute_wait_time({}, attempt=10) == 60

    def test_at_least_one_second(self):
        assert compute_wait_time({"Retry-After": "0"}, attempt=0) == 1


# ---------------------------------------------------------------------------
# handle_rate_limit
# ---------------------------------------------------------------------------

class TestHandleRateLimit:
    @pytest.mark.asyncio
    async def test_single_key_sleeps_then_raises(self):
        pool = KeyPool(["only"], provider_name="prov")
        # First call: not last attempt, should sleep ~1s and return
        start = time.monotonic()
        await handle_rate_limit(
            pool, "only", {"Retry-After": "1"},
            attempt=0, max_attempts=2,
        )
        elapsed = time.monotonic() - start
        assert 0.9 <= elapsed <= 1.5, f"expected ~1s sleep, got {elapsed:.2f}"

        # Second call: last attempt, should raise
        with pytest.raises(RateLimitError) as exc_info:
            await handle_rate_limit(
                pool, "only", {"Retry-After": "1"},
                attempt=1, max_attempts=2,
            )
        assert exc_info.value.provider == "prov"
        assert exc_info.value.retry_after == 1

    @pytest.mark.asyncio
    async def test_multi_key_rotates_no_sleep(self):
        pool = KeyPool(["k1", "k2", "k3"], provider_name="prov")
        captured = []
        def cb(event, message):
            captured.append((event, message))

        start = time.monotonic()
        await handle_rate_limit(
            pool, "k1", {"Retry-After": "30"},
            attempt=0, max_attempts=2, log_callback=cb,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, "rotation should not sleep"
        assert any(event == "llm_key_rotated" for event, _ in captured)

    @pytest.mark.asyncio
    async def test_all_throttled_last_attempt_raises(self):
        pool = KeyPool(["k1", "k2"], provider_name="prov")
        # Manually throttle both keys with long retry-after
        now = time.monotonic()
        await pool.mark_throttled("k1", now + 60)
        await pool.mark_throttled("k2", now + 60)

        with pytest.raises(RateLimitError) as exc_info:
            await handle_rate_limit(
                pool, "k1", {"Retry-After": "60"},
                attempt=1, max_attempts=2,
            )
        msg = str(exc_info.value)
        assert "all 2 key(s)" in msg
        assert exc_info.value.provider == "prov"


# ---------------------------------------------------------------------------
# LLMProvider.api_key backwards-compat property
# ---------------------------------------------------------------------------

class TestProviderApiKeyCompat:
    def test_property_returns_first_key(self):
        from src.core.llm.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key="single", model="gemini-2.0-flash")
        assert p.api_key == "single"

    def test_property_with_multi_key(self):
        from src.core.llm.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key=["k1", "k2"], model="gemini-2.0-flash")
        assert p.api_key == "k1"
        assert p._key_pool.size == 2

    def test_no_key_pool_is_none(self):
        from src.core.llm.providers.openai import OpenAICompatibleProvider
        p = OpenAICompatibleProvider(
            api_endpoint="http://localhost:11434/v1/chat/completions",
            model="llama3",
            api_key=None,
        )
        assert p.api_key is None
        assert p._key_pool is None


# ---------------------------------------------------------------------------
# base.normalize_api_keys
# ---------------------------------------------------------------------------

class TestNormalizeApiKeys:
    """The helper always returns list[str] — pool construction handles the rest.

    This is the canonical entry point now used by LLMProvider.__init__, so any
    provider built with a raw comma/newline string ends up with a multi-key pool
    automatically — the cause of the silent rotation failure fixed alongside
    these tests.
    """

    def test_none(self):
        assert normalize_api_keys(None) == []

    def test_empty_string(self):
        assert normalize_api_keys("") == []

    def test_single_key(self):
        assert normalize_api_keys("just-one-key") == ["just-one-key"]

    def test_csv_to_list(self):
        assert normalize_api_keys("k1,k2,k3") == ["k1", "k2", "k3"]

    def test_csv_with_whitespace(self):
        assert normalize_api_keys("  k1 , k2  ,k3 ") == ["k1", "k2", "k3"]

    def test_newline_separated(self):
        assert normalize_api_keys("k1\nk2\nk3") == ["k1", "k2", "k3"]

    def test_mixed_csv_newline(self):
        assert normalize_api_keys("k1,k2\nk3 , k4") == ["k1", "k2", "k3", "k4"]

    def test_trailing_empty_fragments_dropped(self):
        # "k1,," drops the empty trailing fragments → single-key pool
        assert normalize_api_keys("k1,,") == ["k1"]

    def test_only_separators_returns_empty(self):
        assert normalize_api_keys(",  ,  ,") == []

    def test_list_passes_through(self):
        assert normalize_api_keys(["a", "b"]) == ["a", "b"]

    def test_provider_built_from_csv_string_splits(self):
        """The actual bug regression: a provider given a comma-string must
        produce a multi-key pool, not a single key that contains commas."""
        from src.core.llm.providers.gemini import GeminiProvider
        p = GeminiProvider(api_key="k1,k2,k3")
        assert p._key_pool.size == 3
        assert p._key_pool.peek() == "k1"
