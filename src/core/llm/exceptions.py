"""
LLM-specific exceptions.

This module defines all custom exceptions used in the LLM provider system.
"""


class ContextOverflowError(Exception):
    """
    Raised when the input text exceeds the model's context window.

    This typically occurs when a chunk is too large for the model to process
    in a single request.
    """
    pass


class RepetitionLoopError(Exception):
    """
    Raised when the model enters a repetition loop.

    This can occur with "thinking" models that get stuck repeating the same
    phrase or pattern, indicating the model has likely exceeded its effective
    context window or encountered an issue.
    """
    pass


class ContentRiskError(Exception):
    """Raised when a provider refuses otherwise valid book content.

    Retrying the identical payload wastes requests because this is not a network
    failure. Translation orchestration may recover by reducing the semantic
    unit, while quality stages can record a transparent local-only audit when
    every deterministic fidelity check passes.
    """

    def __init__(self, message: str, provider: str = None):
        super().__init__(message)
        self.provider = provider
        self.retryable_with_smaller_unit = True


class RateLimitError(Exception):
    """
    Raised when the API returns HTTP 429 (Too Many Requests) and all retry
    attempts with backoff have been exhausted.

    This signals the translation pipeline to auto-pause and save a checkpoint
    so the user can resume later.

    Attributes:
        retry_after: Suggested wait time in seconds (from Retry-After header),
                     or None if not provided by the API.
        provider: Name of the LLM provider that was rate-limited.
    """

    def __init__(self, message: str, retry_after: int = None, provider: str = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.provider = provider
        self.retryable = True


class InsufficientCreditsError(RateLimitError):
    """Raised for a provider billing/quota block that waiting cannot fix.

    It subclasses :class:`RateLimitError` so every existing translation layer
    preserves and propagates it to the job boundary instead of converting it to
    a missing model response. ``retryable`` prevents the auto-resume loop from
    spending repeated requests while the account has no balance.
    """

    def __init__(self, message: str, provider: str = None):
        super().__init__(message, retry_after=None, provider=provider)
        self.retryable = False


class DeepSeekPeakPricingError(RateLimitError):
    """Raised before a paid DeepSeek request during an official peak window."""

    def __init__(
        self,
        *,
        retry_after: int,
        next_available_at_utc: str,
        next_available_at_local: str,
        display_timezone: str,
        source_url: str,
    ):
        super().__init__(
            "DeepSeek generation is disabled during its high-price window.",
            retry_after=max(1, int(retry_after)),
            provider="deepseek",
        )
        self.pause_reason = "deepseek_peak_pricing"
        self.next_available_at_utc = next_available_at_utc
        self.next_available_at_local = next_available_at_local
        self.display_timezone = display_timezone
        self.source_url = source_url
