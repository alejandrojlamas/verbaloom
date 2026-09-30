"""
Base classes and data structures for LLM providers.

This module defines the abstract base class that all LLM providers must implement,
as well as common data structures like LLMResponse.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union
import httpx

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT, REQUEST_TIMEOUT, LLM_CONNECT_TIMEOUT
from src.utils.telemetry import get_telemetry_headers
from src.core.llm.utils.extraction import TranslationExtractor
from src.core.llm.key_pool import KeyPool

# retry_manager is intentionally NOT imported at module level: src.core.adapters's
# __init__ eagerly imports generic_translator.py, which imports
# fidelity_supervisor.py, which imports src.core.llm.exceptions, which
# triggers src.core.llm/__init__ -> this very module -- a module-level
# import here closes that cycle and breaks whichever side happens to start
# initializing first (observed as "cannot import name 'FidelityDecision'
# from partially initialized module" when a test imports src.core.epub
# before anything has warmed up src.core.adapters). Deferring the import
# into __init__ (below) sidesteps it: by the time any provider is actually
# instantiated, module initialization has long finished.


def _build_provider_retry_manager():
    from src.core.adapters.retry_manager import RetryConfig, RetryManager, RetryStrategy

    # Shared backoff shape for the "plain transient network/protocol failure"
    # retry decision (timeout, connection error, malformed/empty response,
    # generic 5xx) in every provider's generate() loop. Deliberately NOT used
    # for rate limits (handled separately via key rotation in
    # rate_limit_handler.py), content-policy refusals, context overflow, or
    # authentication failures -- those raise src.core.llm.exceptions types
    # that translator.py/handlers.py catch by type to drive chunk-splitting,
    # auto-pause, and fail-fast behavior. Routing them through this
    # manager's circuit breaker would conflate "is this provider's
    # connectivity healthy" with unrelated per-request/per-account
    # conditions.
    return RetryManager(
        default_config=RetryConfig(
            initial_delay=2.0,
            max_delay=20.0,
            backoff_factor=2.0,
            jitter=0.1,
            strategy=RetryStrategy.EXPONENTIAL,
        ),
        enable_circuit_breaker=True,
    )


def normalize_api_keys(raw: Optional[Union[str, Iterable[str]]]) -> List[str]:
    """Split comma/newline-separated key strings into a clean list.

    Accepts a single key, a "k1,k2,k3" string (the documented multi-key
    format used by .env, the Web UI input, and the CLI), or an iterable of
    keys. Whitespace and empty fragments are trimmed; order is preserved
    for round-robin rotation.

    Returns an empty list when no usable key is provided.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        return [k for k in raw if k]
    if "," not in raw and "\n" not in raw:
        return [raw] if raw else []
    parts = [p.strip() for p in raw.replace("\n", ",").split(",")]
    return [p for p in parts if p]


@dataclass
class LLMResponse:
    """Response from LLM with token usage information"""
    content: str
    prompt_tokens: int = 0  # Number of tokens in the prompt
    completion_tokens: int = 0  # Number of tokens in the response
    prompt_cache_hit_tokens: int = 0  # Input tokens billed at provider cache-hit rate
    prompt_cache_miss_tokens: int = 0  # Input tokens billed at provider cache-miss rate
    total_tokens: int = 0  # Provider-reported request total, when available
    reasoning_tokens: int = 0  # Included in completion_tokens by reasoning providers
    context_used: int = 0  # Total context used (prompt + completion)
    context_limit: int = 0  # Context limit that was set for this request
    was_truncated: bool = False  # True if response was truncated due to context limit
    was_fallback: bool = False  # True if raw response was used because tag extraction failed

    @property
    def prompt_cache_total_tokens(self) -> int:
        return int(self.prompt_cache_hit_tokens or 0) + int(self.prompt_cache_miss_tokens or 0)

    @property
    def prompt_cache_hit_ratio(self) -> float:
        total = self.prompt_cache_total_tokens
        if total <= 0:
            return 0.0
        return float(self.prompt_cache_hit_tokens or 0) / total


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, Iterable[str]]] = None,
        provider_name: str = "",
    ):
        """
        Initialize the LLM provider.

        Args:
            model: Model name/identifier.
            api_keys: A single API key or an iterable of keys. When given,
                wrapped in a KeyPool that supports rotation on HTTP 429.
                Pass None for providers that don't use API keys (e.g. Ollama).
            provider_name: Logical name used for log messages and rate-limit
                error reports. Defaults to the empty string ("unknown" in logs).
        """
        self.model = model
        self._extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)
        self._client = None
        self._key_pool: Optional[KeyPool] = None
        keys_iter = normalize_api_keys(api_keys)
        if keys_iter:
            self._key_pool = KeyPool(keys_iter, provider_name=provider_name or "unknown")
        # One circuit breaker per provider instance, spanning that
        # instance's whole lifetime (typically one whole translation job).
        # If the provider's plain connectivity is down for a sustained run
        # of chunks, this trips and subsequent chunks fail fast instead of
        # each independently retrying with backoff for the full
        # REQUEST_TIMEOUT -- without it, a book with thousands of remaining
        # chunks would spend the whole outage hammering a dead endpoint.
        self._retry_manager = _build_provider_retry_manager()

    @property
    def api_key(self) -> Optional[str]:
        """The current 'primary' API key (first non-throttled if a pool is used).

        Kept for backwards compatibility with code that reads the key directly
        (e.g. listing models, context detection). Translation paths should
        always use the pool's `acquire()` so rotation can happen.
        """
        return self._key_pool.peek() if self._key_pool else None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling"""
        if self._client is None:
            # Add client identification headers to all requests
            telemetry_headers = get_telemetry_headers()
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                # A real LLM response can legitimately take minutes, so
                # read/write/pool stay at the long REQUEST_TIMEOUT. The
                # TCP-connect phase never needs that long: a server that is
                # unreachable at the network level should fail fast so the
                # retry loop gets a chance to run instead of blocking the
                # chunk for up to REQUEST_TIMEOUT before the first retry.
                timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=LLM_CONNECT_TIMEOUT),
                headers=telemetry_headers
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None) -> Optional["LLMResponse"]:
        """
        Generate text from prompt.

        Args:
            prompt: The user prompt (content to process)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse object with content and token usage info, or None if failed
        """
        pass

    def extract_translation(self, response: str) -> Optional[str]:
        """
        Extract translation from response using configured tags with strict validation.

        Returns the content between TRANSLATE_TAG_IN and TRANSLATE_TAG_OUT.
        Prefers responses where tags are at exact boundaries for better reliability.

        NOTE: This method completely ignores content within <think></think> tags,
        as these are used by certain LLMs for internal reasoning and should not
        be searched for translation tags.

        Args:
            response: Raw text response from the LLM

        Returns:
            Extracted translation text, or None if extraction fails
        """
        return self._extractor.extract(response)

    async def translate_text(self, prompt: str) -> Optional[str]:
        """Complete translation workflow: request + extraction"""
        from .request_deadline import await_llm_call

        response = await await_llm_call(
            self.generate,
            prompt,
            provider=self,
        )
        if response:
            return self.extract_translation(response.content)
        return None
