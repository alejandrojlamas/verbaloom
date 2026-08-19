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

from typing import List, Optional, Union
import httpx
import asyncio
import json

from src.config import REQUEST_TIMEOUT, MAX_TRANSLATION_ATTEMPTS, TEMPERATURE
from ..base import LLMProvider, LLMResponse
from ..exceptions import ContentRiskError, ContextOverflowError, InsufficientCreditsError
from ..rate_limit_handler import handle_rate_limit


class DeepSeekProvider(LLMProvider):
    """
    Provider for DeepSeek API.

    DeepSeek provides powerful language models with excellent price/performance:
        - deepseek-v4-pro: Recommended high-quality model for translation
        - deepseek-v4-flash: Faster economical model

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
    MODELS_URL = "https://api.deepseek.com/models"

    MODEL_CONTEXT_SIZES = {
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4-flash": 1_000_000,
        "deepseek-chat": 1_000_000,
        "deepseek-reasoner": 1_000_000,
        "deepseek-coder": 16000,
        "deepseek-v4": 1_000_000,
    }

    FALLBACK_MODELS = [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ]

    THINKING_MODELS = ["deepseek-reasoner", "deepseek-r1"]
    THINKING_BY_DEFAULT_MODELS = ["deepseek-v4"]

    def __init__(
        self,
        api_key: Union[str, List[str]],
        model: str = "deepseek-v4-pro",
        api_endpoint: Optional[str] = None,
        disable_thinking: bool = True
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
        super().__init__(model, api_keys=api_key, provider_name="deepseek")
        self.api_endpoint = api_endpoint or self.API_URL
        self.disable_thinking = disable_thinking

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
                # Skip deepseek-reasoner: always thinks, no toggle. V4 models
                # are kept (they think by default but we override that).
                if "reasoner" in model_id.lower():
                    continue
                if "deepseek" in model_id.lower():
                    context_length = model.get("max_context_length")
                    if not context_length:
                        for prefix, size in self.MODEL_CONTEXT_SIZES.items():
                            if prefix in model_id.lower():
                                context_length = size
                                break
                        if not context_length:
                            context_length = 1_000_000

                    filtered_models.append({
                        "id": model_id,
                        "name": model_id,
                        "context_length": context_length
                    })

            filtered_models.sort(key=lambda x: x["name"])

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
                "name": m,
                "context_length": self._get_context_limit_for_model(m)
            }
            for m in self.FALLBACK_MODELS
        ]

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
            try:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 401:
                    # Permanent failure: an invalid/revoked key will not
                    # start working on retry. Fail immediately instead of
                    # raising and falling into the generic `except
                    # Exception` handler below, which used to retry it like
                    # a transient network error and waste the one extra
                    # attempt MAX_TRANSLATION_ATTEMPTS budgets.
                    print("❌ DeepSeek: Invalid API key (401) — not retrying, this cannot succeed without a valid key.")
                    return None

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

                usage = result.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cache_hit_tokens = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
                cache_miss_tokens = int(usage.get("prompt_cache_miss_tokens", 0) or 0)

                cache_note = ""
                cache_total = cache_hit_tokens + cache_miss_tokens
                if cache_total:
                    cache_ratio = cache_hit_tokens / cache_total * 100
                    cache_note = f" · cache {cache_hit_tokens}/{cache_total} hit ({cache_ratio:.0f}%)"
                print(f"💬 DeepSeek: {prompt_tokens}+{completion_tokens} tokens{cache_note}")
                self._retry_manager.record_attempt_result(True)

                return LLMResponse(
                    content=response_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    prompt_cache_hit_tokens=cache_hit_tokens,
                    prompt_cache_miss_tokens=cache_miss_tokens,
                    context_used=prompt_tokens + completion_tokens,
                    context_limit=self._get_context_limit(),
                    was_truncated=was_truncated
                )

            except InsufficientCreditsError:
                raise

            except httpx.TimeoutException as e:
                print(f"DeepSeek API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            except httpx.HTTPStatusError as e:
                error_body = ""
                error_message = str(e)
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    error_message = f"{e} - {error_body}"

                if e.response.status_code == 404:
                    print(f"❌ DeepSeek: Model '{self.model}' not found!")
                    print(f"   Check available models at https://platform.deepseek.com/")
                elif e.response.status_code == 401:
                    print(f"❌ DeepSeek: Invalid API key!")
                elif e.response.status_code == 402:
                    print(f"❌ DeepSeek: Insufficient credits!")
                else:
                    print(f"DeepSeek API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    print(f"Response details: Status {e.response.status_code}, Body: {error_body}...")

                context_overflow_keywords = [
                    "context_length", "maximum context", "token limit",
                    "too many tokens", "reduce the length", "max_tokens",
                    "context window", "exceeds"
                ]
                if any(keyword in error_message.lower() for keyword in context_overflow_keywords):
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
                    raise ContentRiskError(
                        "DeepSeek refused the submitted book passage because of its content filter.",
                        provider="deepseek",
                    )

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
                print(f"DeepSeek API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

            except Exception as e:
                print(f"DeepSeek API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    await asyncio.sleep(self._retry_manager.delay_for_attempt(attempt + 1))
                    continue
                return None

        return None
