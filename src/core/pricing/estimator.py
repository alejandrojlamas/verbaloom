"""Cost estimator for translation jobs.

The preflight count uses the app's tokenizer as an estimate. Completed calls
always use the provider-reported usage ledger instead.
"""
from typing import Optional

from src.core.chunking.token_chunker import TokenChunker


SYSTEM_PROMPT_TOKENS = 500
USER_TEMPLATE_TOKENS = 60
PREV_TRANSLATION_CONTEXT_TOKENS = 50

LANGUAGE_RATIOS = {
    ("english", "french"):  (1.15, 1.50),
    ("english", "spanish"): (1.15, 1.50),
    ("english", "italian"): (1.15, 1.50),
    ("english", "portuguese"): (1.15, 1.50),
    ("english", "german"):  (1.10, 1.40),
    ("english", "dutch"):   (1.05, 1.30),
    ("english", "russian"): (1.10, 1.45),
    ("english", "japanese"): (1.30, 1.80),
    ("english", "chinese"): (0.50, 0.80),
    ("english", "korean"):  (1.10, 1.50),
    ("english", "arabic"):  (1.10, 1.45),
    ("french", "english"):  (0.75, 0.95),
    ("german", "english"):  (0.80, 1.00),
    ("spanish", "english"): (0.80, 1.00),
    ("japanese", "english"): (1.50, 2.50),
    ("chinese", "english"): (1.50, 2.50),
    ("chinese", "french"):  (1.80, 2.80),
    ("chinese", "german"):  (1.80, 2.80),
    ("japanese", "french"): (1.80, 2.80),
}

DEFAULT_RATIO = (0.95, 1.30)


def get_output_ratio(src_lang: str, tgt_lang: str) -> tuple[float, float]:
    if not src_lang or not tgt_lang:
        return DEFAULT_RATIO
    key = (src_lang.lower().strip(), tgt_lang.lower().strip())
    return LANGUAGE_RATIOS.get(key, DEFAULT_RATIO)


class CostEstimator:
    """
    Estimate translation cost in USD based on text length, chunking, pricing,
    and language pair. Returns a min/max range.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        pricing: dict,
        max_tokens_per_chunk: int = 450,
    ):
        """
        Args:
            provider: e.g. "openai"
            model: e.g. "gpt-4o-mini"
            pricing: {"input": float, "output": float} per 1M tokens (USD)
            max_tokens_per_chunk: chunker setting
        """
        self.provider = provider
        self.model = model
        self.pricing = pricing
        self.chunker = TokenChunker(max_tokens=max_tokens_per_chunk)

    def estimate(
        self,
        text: str,
        src_lang: str = "",
        tgt_lang: str = "",
        options: Optional[dict] = None,
    ) -> dict:
        options = options or {}

        if not text or not text.strip():
            return self._empty_result()

        chunks = self.chunker.chunk_text(text)
        n_chunks = len(chunks)

        if n_chunks == 0:
            return self._empty_result()

        main_tokens = sum(self.chunker.count_tokens(c["main_content"]) for c in chunks)

        per_chunk_overhead = SYSTEM_PROMPT_TOKENS + USER_TEMPLATE_TOKENS
        prev_context_tokens = max(0, n_chunks - 1) * PREV_TRANSLATION_CONTEXT_TOKENS
        total_input_tokens = main_tokens + n_chunks * per_chunk_overhead + prev_context_tokens

        ratio_min, ratio_max = get_output_ratio(src_lang, tgt_lang)
        output_min = int(main_tokens * ratio_min)
        output_max = int(main_tokens * ratio_max)

        passes = 1
        if options.get("refine"):
            passes += 1
        if options.get("text_cleanup"):
            passes += 1

        in_per_million = self.pricing.get("input", 0.0)
        input_cache_hit_per_million = self.pricing.get("input_cache_hit")
        input_cache_miss_per_million = self.pricing.get("input_cache_miss", in_per_million)
        out_per_million = self.pricing.get("output", 0.0)

        all_pass_input_tokens = total_input_tokens * passes
        all_pass_output_min = output_min * passes
        all_pass_output_max = output_max * passes

        input_cost = (all_pass_input_tokens / 1_000_000) * input_cache_miss_per_million
        input_cost_min = input_cost
        cacheable_input_tokens = 0
        if input_cache_hit_per_million is not None and n_chunks > 1:
            # DeepSeek bills repeated prefixes at the cache-hit rate. The exact
            # hit/miss split is only known after the provider response, but the
            # system prompt and prompt template are stable across chunks. Use
            # this as the optimistic side of the estimate and keep all-miss as
            # the conservative side.
            cacheable_input_tokens = max(0, n_chunks - 1) * per_chunk_overhead * passes
            miss_tokens_min = max(0, all_pass_input_tokens - cacheable_input_tokens)
            input_cost_min = (
                (miss_tokens_min / 1_000_000) * input_cache_miss_per_million
                + (cacheable_input_tokens / 1_000_000) * input_cache_hit_per_million
            )
        output_cost_min = (all_pass_output_min / 1_000_000) * out_per_million
        output_cost_max = (all_pass_output_max / 1_000_000) * out_per_million

        return {
            "model": self.model,
            "provider": self.provider,
            "n_chunks": n_chunks,
            "passes": passes,
            "input_tokens": all_pass_input_tokens,
            "input_tokens_per_pass": total_input_tokens,
            "main_text_tokens": main_tokens,
            "estimated_output_tokens_min": all_pass_output_min,
            "estimated_output_tokens_max": all_pass_output_max,
            "input_cost": round(input_cost, 4),
            "input_cost_min": round(input_cost_min, 4),
            "output_cost_min": round(output_cost_min, 4),
            "output_cost_max": round(output_cost_max, 4),
            "total_cost_min": round(input_cost_min + output_cost_min, 4),
            "total_cost_max": round(input_cost + output_cost_max, 4),
            "currency": "USD",
            "token_count_source": "tokenizer_estimate",
            "pricing_used": {
                "input_per_million": in_per_million,
                "input_cache_hit_per_million": input_cache_hit_per_million,
                "input_cache_miss_per_million": input_cache_miss_per_million,
                "output_per_million": out_per_million,
            },
            "cache_assumption": {
                "cacheable_input_tokens": cacheable_input_tokens,
                "min_uses_cache_hits": input_cache_hit_per_million is not None,
                "max_uses_all_cache_miss": input_cache_hit_per_million is not None,
            },
            "ratio_used": {"min": ratio_min, "max": ratio_max},
        }

    def _empty_result(self) -> dict:
        return {
            "model": self.model,
            "provider": self.provider,
            "n_chunks": 0,
            "passes": 1,
            "input_tokens": 0,
            "input_tokens_per_pass": 0,
            "main_text_tokens": 0,
            "estimated_output_tokens_min": 0,
            "estimated_output_tokens_max": 0,
            "input_cost": 0.0,
            "input_cost_min": 0.0,
            "output_cost_min": 0.0,
            "output_cost_max": 0.0,
            "total_cost_min": 0.0,
            "total_cost_max": 0.0,
            "currency": "USD",
            "token_count_source": "tokenizer_estimate",
            "pricing_used": {
                "input_per_million": self.pricing.get("input", 0.0),
                "input_cache_hit_per_million": self.pricing.get("input_cache_hit"),
                "input_cache_miss_per_million": self.pricing.get("input_cache_miss", self.pricing.get("input", 0.0)),
                "output_per_million": self.pricing.get("output", 0.0),
            },
            "cache_assumption": {
                "cacheable_input_tokens": 0,
                "min_uses_cache_hits": False,
                "max_uses_all_cache_miss": False,
            },
            "ratio_used": {"min": DEFAULT_RATIO[0], "max": DEFAULT_RATIO[1]},
        }
