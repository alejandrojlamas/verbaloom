"""
DeepSeek LLM Provider.

This module provides the DeepSeekProvider class for interacting with
DeepSeek's API, which offers cost-effective models with strong capabilities.

Features:
    - DeepSeek V3 (chat) and V4 (flash/pro) models
    - OpenAI-compatible API format
    - Cost-effective pricing (~5-10x cheaper than OpenAI)
    - 1M context window for V4 models
    - Auto-disables V4 reasoning by default (translation-friendly)
"""

from typing import Any, List, Mapping, Optional, Union
import httpx
import asyncio
import json
import time

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS, TEMPERATURE
from ..base import LLMProvider, LLMResponse
from src.core.deepseek_pricing import (
    get_deepseek_pricing_status,
    is_official_deepseek_endpoint,
)
from ..exceptions import (
    ContextOverflowError,
    DeepSeekPeakPricingError,
    InsufficientCreditsError,
    ProviderActionRequiredError,
    RateLimitError,
)
from ..rate_limit_handler import handle_rate_limit
from ..attempts import observe_attempt, reserve_attempt


class DeepSeekProvider(LLMProvider):
    """
    Provider for DeepSeek API.

    DeepSeek provides powerful language models with excellent price/performance:
        - deepseek-v4-pro: Recommended high-quality model for translation
        - deepseek-flash: DeepSeek V4.1 Flash, faster and economical

    Configuration:
        endpoint: https://api.deepseek.com/chat/completions
        model: Model identifier (e.g., "deepseek-v4-pro")
        api_key: DeepSeek API key

    Example:
        >>> provider = DeepSeekProvider(
        ...     api_key="your-api-key",
        ...     model="deepseek-v4-pro"
        ... )
        >>> response = await provider.generate("Translate: Hello")
    """

    API_URL = "https://api.deepseek.com/chat/completions"
    supports_attempt_reservations = True
    MODELS_URL = "https://api.deepseek.com/models"

    MODEL_CONTEXT_SIZES = {
        "deepseek-flash": 1_000_000,
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4-flash": 1_000_000,
        "deepseek-chat": 1_000_000,
        "deepseek-reasoner": 1_000_000,
        "deepseek-coder": 16000,
        "deepseek-v4": 1_000_000,
    }

    FALLBACK_MODELS = [
        "deepseek-flash",
        "deepseek-v4-pro",
    ]

    LEGACY_MODEL_ALIASES = {
        "deepseek-v4-flash": "deepseek-flash",
        "deepseek-v4-flash-vision-exp": "deepseek-flash",
        "deepseek-chat": "deepseek-flash",
        "deepseek-reasoner": "deepseek-flash",
    }

    MODEL_METADATA = {
        "deepseek-flash": {
            "name": "DeepSeek V4.1 Flash",
            "description": "Fast, economical, multimodal model",
        },
        "deepseek-v4-pro": {
            "name": "DeepSeek V4 Pro",
            "description": "Highest-quality text model",
        },
    }

    THINKING_MODELS = ["deepseek-reasoner", "deepseek-r1"]
    THINKING_BY_DEFAULT_MODELS = ["deepseek-flash", "deepseek-v4"]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "deepseek-v4-pro",
        api_endpoint: Optional[str] = None,
        disable_thinking: bool = True,
        max_output_tokens: Optional[int] = None,
    ):
        """
        Initialize the DeepSeek provider.

        Args:
            api_key: DeepSeek API key
            model: Model identifier (default: deepseek-v4-pro)
            api_endpoint: Optional custom API endpoint
            disable_thinking: For models that think by default (V4 family),
                inject ``thinking={"type":"disabled"}`` to skip reasoning tokens.
        """
        resolved_endpoint = api_endpoint or self.API_URL
        normalized_model = model
        if is_official_deepseek_endpoint(resolved_endpoint):
            normalized_model = self.LEGACY_MODEL_ALIASES.get(
                str(model or "").strip().lower(),
                model,
            )
        super().__init__(normalized_model, api_keys=api_key, provider_name="deepseek")
        self.api_endpoint = resolved_endpoint
        self.disable_thinking = disable_thinking
        self.configured_model = model
        if max_output_tokens is not None and (
            type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 384_000
        ):
            raise ValueError("DeepSeek output limit must be an integer in 1..384000")
        self.max_output_tokens = max_output_tokens

    def _get_context_limit(self) -> int:
        """
        Determine context limit based on model name.

        Returns:
            Context limit in tokens
        """
        model_lower = self.model.lower()
        for prefix, limit in self.MODEL_CONTEXT_SIZES.items():
            if prefix in model_lower:
                return limit
        return 1_000_000  # Default for current DeepSeek V4-compatible models

    def _is_thinking_model(self) -> bool:
        """Check if the current model uses thinking mode."""
        return any(tm in self.model.lower() for tm in self.THINKING_MODELS)

    def _thinking_enabled_by_default(self) -> bool:
        """True for models (V4 family) that think unless `thinking.type=disabled` is sent."""
        model_lower = self.model.lower()
        return any(tm in model_lower for tm in self.THINKING_BY_DEFAULT_MODELS)

    async def get_available_models(self) -> list:
        """
        Fetch available DeepSeek models from API.

        Returns:
            List of model dicts with id, name, and context_length
        """
        if not self.api_key:
            return self._get_fallback_models()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            client = await self._get_client()

            response = await client.get(
                self.MODELS_URL,
                headers=headers,
                timeout=15
            )
            response.raise_for_status()

            models_data = response.json().get("data", [])
            filtered_models = []

            for model in models_data:
                model_id = model.get("id", "")
                # DeepSeek exposes exactly two current public chat models. Do
                # not reintroduce retired aliases if an older gateway lists
                # them alongside the canonical IDs.
                if model_id not in self.FALLBACK_MODELS:
                    continue
                context_length = model.get("max_context_length")
                if not context_length:
                    context_length = self._get_context_limit_for_model(model_id)

                metadata = self.MODEL_METADATA.get(model_id, {})
                filtered_models.append({
                    "id": model_id,
                    "name": metadata.get("name", model_id),
                    "description": metadata.get("description", ""),
                    "context_length": context_length,
                })

            preferred_order = {name: index for index, name in enumerate(self.FALLBACK_MODELS)}
            filtered_models.sort(
                key=lambda item: (
                    preferred_order.get(item["id"], len(preferred_order)),
                    item["id"],
                )
            )

            if len(filtered_models) < 1:
                return self._get_fallback_models()

            return filtered_models

        except Exception as e:
            print(f"⚠️ Failed to fetch DeepSeek models: {e}")
            return self._get_fallback_models()

    def _get_fallback_models(self) -> list:
        """Return fallback models list when API fetch fails."""
        return [
            {
                "id": m,
                "name": self.MODEL_METADATA.get(m, {}).get("name", m),
                "description": self.MODEL_METADATA.get(m, {}).get("description", ""),
                "context_length": self._get_context_limit_for_model(m),
            }
            for m in self.FALLBACK_MODELS
        ]

    @staticmethod
    def _usage_counts(usage: Mapping[str, Any] | None) -> dict[str, int]:
        """Normalize DeepSeek's authoritative usage and cache breakdown."""

        usage = usage or {}

        def count(value: Any) -> int:
            try:
                return max(0, int(value or 0))
            except (TypeError, ValueError):
                return 0

        prompt = count(usage.get("prompt_tokens"))
        completion = count(usage.get("completion_tokens"))
        total = max(count(usage.get("total_tokens")), prompt + completion)
        details = usage.get("prompt_tokens_details")
        details = details if isinstance(details, Mapping) else {}
        cache_hit = count(usage.get("prompt_cache_hit_tokens"))
        if not cache_hit:
            cache_hit = count(details.get("cached_tokens"))
        cache_hit = min(cache_hit, prompt)

        raw_cache_miss = usage.get("prompt_cache_miss_tokens")
        cache_miss = count(raw_cache_miss)
        if raw_cache_miss is None and prompt:
            cache_miss = max(0, prompt - cache_hit)
        cache_miss = min(cache_miss, max(0, prompt - cache_hit))

        completion_details = usage.get("completion_tokens_details")
        completion_details = completion_details if isinstance(completion_details, Mapping) else {}
        reasoning = min(count(completion_details.get("reasoning_tokens")), completion)
        return {
            "prompt": prompt,
            "completion": completion,
            "total": total,
            "cache_hit": cache_hit,
            "cache_miss": cache_miss,
            "reasoning": reasoning,
        }

    def _get_context_limit_for_model(self, model_name: str) -> int:
        """Get context limit for a specific model name."""
        model_lower = model_name.lower()
        for prefix, limit in self.MODEL_CONTEXT_SIZES.items():
            if prefix in model_lower:
                return limit
        return 1_000_000

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[LLMResponse]:
        """
        Generate text using DeepSeek API.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed

        Raises:
            ContextOverflowError: If input exceeds model's context window
        """
        if is_official_deepseek_endpoint(self.api_endpoint):
            pricing = get_deepseek_pricing_status()
            if pricing.disabled:
                # Add a small boundary buffer so the resumed request cannot land
                # inside the peak window because of clock or scheduler rounding.
                raise DeepSeekPeakPricingError(
                    retry_after=pricing.seconds_until_available + 2,
                    next_available_at_utc=pricing.next_available_at_utc or "",
                    next_available_at_local=pricing.next_available_at_local or "",
                    display_timezone=pricing.display_timezone,
                    source_url=pricing.source_url,
                )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": TEMPERATURE if temperature is None else float(temperature),
            "stream": False
        }

        if self._thinking_enabled_by_default() and self.disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            if not self._retry_manager.circuit_allows_attempt():
                print(
                    "⛔ DeepSeek: circuit breaker open after repeated connectivity "
                    "failures; failing fast without a network call."
                )
                return None

            current_key = await self._key_pool.acquire()
            headers = {
                "Authorization": f"Bearer {current_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            reservation_id = reserve_attempt(endpoint=self.api_endpoint, payload=payload,
                                             attempt=attempt + 1)
            response = None
            response_record = None
            attempt_status = "error"
            error_type = ""
            error_reason = ""
            started_at = time.time()
            started = time.perf_counter()
            http_finished = started
            try:
                try:
                    response = await client.post(
                        self.api_endpoint, headers=headers, json=payload, timeout=timeout,
                    )
                finally:
                    http_finished = time.perf_counter()

                if response.status_code == 401:
                    raise ProviderActionRequiredError(
                        "DeepSeek credentials require attention.", reason="authentication_error", provider="deepseek",
                    )

                if response.status_code == 429:
                    await handle_rate_limit(
                        self._key_pool, current_key, response.headers,
                        attempt, MAX_TRANSLATION_ATTEMPTS,
                    )
                    continue

                if response.status_code == 402:
                    raise InsufficientCreditsError(
                        "DeepSeek rejected the request because the account has insufficient credits.",
                        provider="deepseek",
                    )

                response.raise_for_status()
                result = response.json()
                usage = self._usage_counts(result.get("usage"))
                response_record = LLMResponse(
                    content="", prompt_tokens=usage["prompt"], completion_tokens=usage["completion"],
                    prompt_cache_hit_tokens=usage["cache_hit"], prompt_cache_miss_tokens=usage["cache_miss"],
                    total_tokens=usage["total"], reasoning_tokens=usage["reasoning"],
                    context_used=usage["total"], context_limit=self._get_context_limit(),
                    raw_usage=result.get("usage") or {}, request_id=response.headers.get("x-request-id", ""),
                    response_id=str(result.get("id") or ""), response_model=str(result.get("model") or self.model),
                    latency_ms=int((http_finished-started)*1000), status="incomplete",
                )

                if "choices" not in result or len(result["choices"]) == 0:
                    # An empty/malformed body can be a transient provider
                    # hiccup, not a permanent failure -- give it the same
                    # retry treatment as timeouts/5xx/JSON errors instead of
                    # spending the whole request on a single attempt.
                    print(f"⚠️ DeepSeek: Unexpected response format (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {result}")
                    self._retry_manager.record_attempt_result(False)
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                        continue
                    return None

                choice = result["choices"][0]
                response_text = choice.get("message", {}).get("content", "")
                finish_reason = str(choice.get("finish_reason") or "").strip().lower()
                was_truncated = finish_reason in {
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                }
                response_record.content = response_text if isinstance(response_text, str) else ""
                response_record.finish_reason = finish_reason
                response_record.was_truncated = was_truncated
                refusal = choice.get("message", {}).get("refusal")
                if refusal or finish_reason == "content_filter":
                    attempt_status = "refused"
                    response_record.status = "refused"
                    response_record.refusal = str(refusal or "")
                    raise ProviderActionRequiredError(
                        "DeepSeek refused this passage; review is required.",
                        reason="provider_refusal", provider="deepseek", response=response_record,
                    )
                if not isinstance(response_text, str) or (not response_text.strip() and not was_truncated):
                    raise ValueError("Empty completion content")
                if is_official_deepseek_endpoint(self.api_endpoint) and not finish_reason:
                    raise ValueError("Official DeepSeek response is missing finish_reason")
                if finish_reason and finish_reason not in {"stop", "length", "max_tokens", "max_output_tokens"}:
                    raise ValueError("Unexpected completion finish reason")

                prompt_tokens = usage["prompt"]
                completion_tokens = usage["completion"]
                cache_hit_tokens = usage["cache_hit"]
                cache_miss_tokens = usage["cache_miss"]

                cache_note = ""
                cache_total = cache_hit_tokens + cache_miss_tokens
                if cache_total:
                    cache_ratio = cache_hit_tokens / cache_total * 100
                    cache_note = f" · cache {cache_hit_tokens}/{cache_total} hit ({cache_ratio:.0f}%)"
                print(f"💬 DeepSeek: {prompt_tokens}+{completion_tokens} tokens{cache_note}")
                self._retry_manager.record_attempt_result(True)

                response_record.status = "incomplete" if was_truncated else "completed"
                attempt_status = "incomplete" if was_truncated else "ok"
                return response_record

            except (RateLimitError, ContextOverflowError) as exc:
                error_type = type(exc).__name__
                error_reason = getattr(exc, "reason", "")
                raise
            except asyncio.CancelledError:
                error_type = "cancelled_in_flight"
                raise

            except httpx.TimeoutException as e:
                error_type = "ambiguous_timeout"
                print(f"DeepSeek API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            except httpx.HTTPStatusError as e:
                error_type = "http_error"
                error_body = ""
                error_message = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    error_message = f"{e} - {error_body}"

                if e.response.status_code in {403, 404}:
                    error_reason = "permission_denied" if e.response.status_code == 403 else "model_unavailable"
                    raise ProviderActionRequiredError("DeepSeek request requires configuration changes.",
                                                      reason=error_reason, provider="deepseek")
                elif e.response.status_code == 401:
                    print("❌ DeepSeek: Invalid API key!")
                elif e.response.status_code == 402:
                    print("❌ DeepSeek: Insufficient credits!")
                else:
                    print(f"DeepSeek API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                context_overflow_keywords = [
                    "context_length_exceeded", "maximum context length",
                    "too many tokens", "context window exceeded",
                ]
                if e.response.status_code == 400 and any(keyword in error_message.lower() for keyword in context_overflow_keywords):
                    raise ContextOverflowError(f"DeepSeek context overflow: {error_message}")

                content_risk_keywords = (
                    "content exists risk",
                    "content risk",
                    "content policy",
                    "unsafe content",
                )
                if e.response.status_code == 400 and any(
                    keyword in error_message.lower() for keyword in content_risk_keywords
                ):
                    error_reason = "provider_refusal"
                    attempt_status = "refused"
                    raise ProviderActionRequiredError(
                        "DeepSeek refused this passage; review is required.",
                        reason=error_reason, provider="deepseek",
                    )
                if 400 <= e.response.status_code < 500 and e.response.status_code not in {408, 409}:
                    error_reason = "invalid_request"
                    raise ProviderActionRequiredError("DeepSeek rejected the request configuration.",
                                                      reason=error_reason, provider="deepseek")

                # Reached only for a plain/unexpected HTTP error (not the
                # context-overflow or content-risk cases raised above,
                # which propagate to translator.py's dedicated handling
                # instead of hitting the circuit breaker): a real
                # connectivity/provider-health signal.
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            except json.JSONDecodeError as e:
                error_type = "invalid_json"
                print(f"DeepSeek API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            except Exception as e:
                error_type = type(e).__name__
                print(f"DeepSeek API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            finally:
                http_status = response.status_code if response is not None else None
                observe_attempt(response=response_record, status=attempt_status, metadata={
                    "attempt": attempt + 1, "started_at": started_at,
                    "elapsed_ms": int((http_finished-started)*1000),
                    "budget_reservation_id": reservation_id, "http_status": http_status,
                    "request_id": response.headers.get("x-request-id", "") if response is not None else "",
                    "error": error_type, "reason": error_reason,
                    "configured_model": self.configured_model,
                    "effective_model": self.model,
                    "usage_unknown": not bool(response_record and response_record.raw_usage),
                    "billing_uncertain": not bool(response_record and response_record.raw_usage) and (
                        http_status is None or http_status >= 500 or http_status == 200),
                    "parameters": {key: payload[key] for key in (
                        "model", "max_tokens", "temperature", "thinking", "stream") if key in payload},
                })

        return None
