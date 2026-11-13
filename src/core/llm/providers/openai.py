"""
OpenAI-compatible provider implementation.

This module provides the OpenAICompatibleProvider class for interacting with
OpenAI API and compatible endpoints (llama.cpp, LM Studio, vLLM, OpenAI, etc.).
"""

from typing import List, Optional, Callable, Union
import asyncio
import json
import time
from urllib.parse import urlparse
import httpx

from ..base import LLMProvider, LLMResponse
from ..exceptions import ContextOverflowError, InsufficientCreditsError, ProviderActionRequiredError, RateLimitError
from .openai_contract import EDITORIAL_MODELS, text_parameters, normalize_completion, normalize_usage
from src.utils.provider_security import endpoint_origin
from ..rate_limit_handler import handle_rate_limit
from ..utils.context_detection import ContextDetector
from ..attempts import observe_attempt, reserve_attempt

from src.config import (
    REQUEST_TIMEOUT,
    OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS
)


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI-compatible API provider (works with llama.cpp, LM Studio, vLLM, OpenAI, etc.)"""

    supports_attempt_reservations = True

    def __init__(self, api_endpoint: str, model: str,
                 api_key: Optional[Union[str, List[str]]] = None,
                 context_window: int = OLLAMA_NUM_CTX, log_callback: Optional[Callable] = None,
                 provider_name: str = "openai-compatible",
                 reasoning_effort: str = "none", max_output_tokens: int = 8192):
        # Skip pool creation if no key (local servers like llama.cpp don't need one)
        super().__init__(model, api_keys=api_key, provider_name=provider_name)
        self.api_endpoint = self._normalize_endpoint(api_endpoint)
        self.context_window = context_window
        self.log_callback = log_callback
        self._detected_context_size: Optional[int] = None
        self._context_detector = ContextDetector()
        self.provider_name = provider_name
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        if self._is_official_openai_endpoint():
            text_parameters(model, effort=reasoning_effort, max_output_tokens=max_output_tokens)

    def _is_official_openai_endpoint(self) -> bool:
        """Check if the endpoint is the official OpenAI API."""
        return endpoint_origin(self.api_endpoint) == "https://api.openai.com"

    def _is_local_endpoint(self) -> bool:
        """Check if the endpoint is a local server (llama.cpp, vLLM, LM Studio, etc.)."""
        return urlparse(self.api_endpoint).hostname in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _normalize_endpoint(endpoint: str) -> str:
        """
        Normalize API endpoint URL for OpenAI-compatible APIs.
        
        Automatically adds '/chat/completions' if the URL ends with '/v1' or '/v1/'
        but not with the full path. This handles common user mistakes like:
        - http://localhost:11434/v1 -> http://localhost:11434/v1/chat/completions
        - https://api.example.com/v1/ -> https://api.example.com/v1/chat/completions
        
        Args:
            endpoint: Raw endpoint URL provided by user
            
        Returns:
            Normalized endpoint URL with complete path
        """
        if not endpoint:
            return endpoint
        
        # Remove trailing slash for consistent processing
        endpoint = endpoint.rstrip('/')
        
        # If already ends with /v1/chat/completions, keep as-is
        if endpoint.endswith('/v1/chat/completions'):
            return endpoint
        
        # If ends with /v1, append /chat/completions
        if endpoint.endswith('/v1'):
            return endpoint + '/chat/completions'
        
        # Otherwise return as-is (user provided custom path)
        return endpoint

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT,
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None) -> Optional[LLMResponse]:
        """
        Generate text using an OpenAI compatible API.

        Args:
            prompt: The user prompt (content to translate)
            timeout: Request timeout in seconds
            system_prompt: Optional system prompt (role/instructions)

        Returns:
            LLMResponse with content and token usage info, or None if failed
        """
        # Build messages array with optional system prompt
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self._is_official_openai_endpoint():
            payload.update(text_parameters(
                self.model, effort=self.reasoning_effort,
                max_output_tokens=self.max_output_tokens, temperature=temperature,
            ))

        # Only add thinking-disable params for local servers (llama.cpp, vLLM, LM Studio)
        # Skip for official OpenAI API and cloud providers (NVIDIA NIM, etc.)
        if not self._is_official_openai_endpoint() and self._is_local_endpoint():
            payload["thinking"] = False
            payload["enable_thinking"] = False
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            if not self._retry_manager.circuit_allows_attempt():
                if self.log_callback:
                    self.log_callback("llm_circuit_open",
                        "⛔ Circuit breaker open after repeated connectivity "
                        "failures; failing fast without a network call.")
                else:
                    print("⛔ OpenAI-compatible: circuit breaker open; failing fast without a network call.")
                return None

            current_key = await self._key_pool.acquire() if self._key_pool else None
            headers = {"Content-Type": "application/json"}
            if current_key:
                headers["Authorization"] = f"Bearer {current_key}"
            result = None
            response = None
            attempt_status = "error"
            error_type = ""
            error_reason = ""
            reservation_id = reserve_attempt(endpoint=self.api_endpoint, payload=payload,
                                             attempt=attempt + 1)
            started_at = time.time()
            started = time.perf_counter()
            http_finished = started
            try:
                try:
                    response = await client.post(
                        self.api_endpoint, json=payload, headers=headers, timeout=timeout,
                    )
                finally:
                    http_finished = time.perf_counter()
                response.raise_for_status()
                body = response.json()
                result = LLMResponse(content="", status="incomplete", **normalize_usage(body),
                                     request_id=response.headers.get("x-request-id", ""),
                                     response_id=str(body.get("id") or ""),
                                     response_model=str(body.get("model") or ""),
                                     service_tier=str(body.get("service_tier") or ""))
                result = normalize_completion(
                    body, request_id=response.headers.get("x-request-id", ""),
                    latency_ms=int((time.perf_counter()-started)*1000),
                    context_limit=self.context_window,
                )
                if self._is_official_openai_endpoint() and not result.finish_reason:
                    raise ValueError("Official OpenAI response is missing finish_reason")
                if result.status == "refused":
                    attempt_status = "refused"
                    raise ProviderActionRequiredError(
                        "The provider refused this passage; review is required.",
                        reason="provider_refusal", provider=self.provider_name, response=result,
                    )
                if not result.content.strip() and not result.was_truncated:
                    raise ValueError("Empty completion content")
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                context_used = result.context_used

                if self.log_callback and (prompt_tokens or completion_tokens):
                    self.log_callback("token_usage",
                        f"Tokens: prompt={prompt_tokens}, response={completion_tokens}, "
                        f"total={context_used}")

                self._retry_manager.record_attempt_result(True)
                attempt_status = "incomplete" if result.was_truncated else "ok"
                return result

            except (RateLimitError, ContextOverflowError) as e:
                error_type = type(e).__name__
                error_reason = getattr(e, "reason", "")
                raise
            except asyncio.CancelledError:
                error_type = "cancelled_in_flight"
                raise
            except httpx.TimeoutException as e:
                error_type = "ambiguous_timeout"
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_timeout",
                        f"{YELLOW}⚠️ LLM request timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Possible causes:{RESET}\n"
                        f"{YELLOW}   - llama.cpp/LM Studio server crashed or became unresponsive{RESET}\n"
                        f"{YELLOW}   - Server overloaded or out of memory{RESET}\n"
                        f"{YELLOW}   - Network connectivity issues{RESET}\n"
                        f"{YELLOW}   - Model too large for available VRAM{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")

                self._retry_manager.record_transport_failure("timeout")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    delay = self._retry_manager.delay_for_attempt(attempt + 1)
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                # All retry attempts exhausted
                if self.log_callback:
                    self.log_callback("llm_timeout_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Translation failed - unable to reach LLM server{RESET}\n"
                        f"{RED}   Recommendations:{RESET}\n"
                        f"{RED}   1. Check if llama.cpp/LM Studio server is running{RESET}\n"
                        f"{RED}   2. Verify server is accessible at: {self.api_endpoint}{RESET}\n"
                        f"{RED}   3. Check server console/logs for crashes or OOM errors{RESET}\n"
                        f"{RED}   4. Try reducing context size or chunk size{RESET}\n"
                        f"{RED}   5. Ensure model fits in available VRAM/RAM{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except httpx.HTTPStatusError as e:
                error_type = "http_error"
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                error_message = str(e)
                error_body = ""
                error_code = ""
                if hasattr(e, 'response') and hasattr(e.response, 'text'):
                    error_body = e.response.text[:500]
                    try:
                        # Try to parse JSON error for better messages
                        error_json = e.response.json()
                        if "error" in error_json:
                            if isinstance(error_json["error"], dict):
                                error_code = str(error_json["error"].get("code") or error_json["error"].get("type") or "")
                                error_message = error_json["error"].get("message", str(e))
                            else:
                                error_message = str(error_json.get("error", e))
                    except Exception:
                        error_message = f"{e} - {error_body}"

                if e.response.status_code == 402 or error_code in {
                    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
                }:
                    raise InsufficientCreditsError(
                        "Provider quota or billing requires attention.", provider=self.provider_name,
                    )
                if e.response.status_code in {401, 403, 404}:
                    reason = {401: "authentication_error", 403: "permission_denied", 404: "model_unavailable"}[e.response.status_code]
                    raise ProviderActionRequiredError(
                        f"Provider request blocked: {reason}.", reason=reason, provider=self.provider_name,
                    )

                # Handle rate limiting (429) — rotate key or sleep, raise if exhausted
                if (hasattr(e, 'response') and e.response is not None
                        and e.response.status_code == 429 and self._key_pool):
                    await handle_rate_limit(
                        self._key_pool, current_key, e.response.headers,
                        attempt, MAX_TRANSLATION_ATTEMPTS, self.log_callback,
                    )
                    continue

                # Detect context overflow errors
                context_overflow_keywords = ["context_length_exceeded", "maximum context length",
                                             "context window exceeded", "too many tokens"]
                if e.response.status_code == 400 and (
                    error_code == "context_length_exceeded" or
                    any(keyword in error_message.lower() for keyword in context_overflow_keywords)
                ):
                    RED = '\033[91m'
                    YELLOW = '\033[93m'
                    RESET = '\033[0m'

                    if self.log_callback:
                        self.log_callback("llm_context_overflow",
                            f"{RED}❌ Context size exceeded!{RESET}\n"
                            f"{RED}   Prompt is too large for model's context window{RESET}\n"
                            f"{RED}   Current context window: {self.context_window} tokens{RESET}\n"
                            f"{RED}   Error: {error_message}{RESET}\n"
                            f"{YELLOW}   Solutions:{RESET}\n"
                            f"{YELLOW}   1. Reduce max_tokens_per_chunk (current chunk may be too large){RESET}\n"
                            f"{YELLOW}   2. Increase context size in server configuration{RESET}\n"
                            f"{YELLOW}   3. Use a model with larger context window{RESET}\n"
                            f"{YELLOW}   4. For llama.cpp: increase -c/--ctx-size parameter{RESET}")
                    else:
                        print(f"{RED}Context size exceeded: {error_message}{RESET}")
                    raise ContextOverflowError(error_message)

                if e.response.status_code == 429:
                    raise RateLimitError("Provider rate limit reached.", provider=self.provider_name)
                if 400 <= e.response.status_code < 500 and e.response.status_code not in {408, 409}:
                    raise ProviderActionRequiredError(
                        "Provider rejected the request configuration.",
                        reason="invalid_request", provider=self.provider_name,
                    )

                # Handle other HTTP errors with detailed information
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    status_code = e.response.status_code if e.response else 'unknown'
                    self.log_callback("llm_http_error",
                        f"{YELLOW}⚠️ HTTP error from LLM server (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Status: {e.response.status_code if e.response else 'unknown'}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error: {error_message}{RESET}")
                    if error_body:
                        self.log_callback("llm_http_error_detail", f"{YELLOW}   Response: {error_body[:200]}...{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")
                    if error_body:
                        print(f"{YELLOW}   Response: {error_body[:200]}...{RESET}")

                # Reached only for a plain/unexpected HTTP error (the 429
                # and context-overflow cases above already `continue`d or
                # raised): a real connectivity/provider-health signal.
                self._retry_manager.record_transport_failure("http_error")
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    delay = self._retry_manager.delay_for_attempt(attempt + 1)
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                # All retries exhausted
                if self.log_callback:
                    self.log_callback("llm_http_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   HTTP error persists - translation failed{RESET}\n"
                        f"{RED}   Status: {e.response.status_code if e.response else 'unknown'}{RESET}\n"
                        f"{RED}   Check server logs for more details{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except json.JSONDecodeError as e:
                error_type = "invalid_json"
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_json_error",
                        f"{YELLOW}⚠️ Invalid JSON response from LLM (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}\n"
                        f"{YELLOW}   This may indicate:{RESET}\n"
                        f"{YELLOW}   - Server returned malformed response{RESET}\n"
                        f"{YELLOW}   - llama.cpp server crashed mid-response{RESET}\n"
                        f"{YELLOW}   - API endpoint incompatibility{RESET}\n"
                        f"{YELLOW}   - Server configuration issues{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}{RESET}")

                self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    delay = self._retry_manager.delay_for_attempt(attempt + 1)
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                if self.log_callback:
                    self.log_callback("llm_json_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Unable to parse LLM response - translation failed{RESET}\n"
                        f"{RED}   Verify server is running and endpoint is correct{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None
            except Exception as e:
                error_type = type(e).__name__
                RED = '\033[91m'
                YELLOW = '\033[93m'
                RESET = '\033[0m'

                if self.log_callback:
                    self.log_callback("llm_unexpected_error",
                        f"{YELLOW}⚠️ Unexpected error during LLM request (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}){RESET}\n"
                        f"{YELLOW}   Endpoint: {self.api_endpoint}{RESET}\n"
                        f"{YELLOW}   Model: {self.model}{RESET}\n"
                        f"{YELLOW}   Error type: {type(e).__name__}{RESET}\n"
                        f"{YELLOW}   Error: {str(e)}{RESET}")
                else:
                    print(f"{YELLOW}⚠️ OpenAI-compatible API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {type(e).__name__}: {e}{RESET}")

                if isinstance(e, httpx.HTTPError):
                    # Connection reset, DNS failure, protocol error.
                    self._retry_manager.record_transport_failure("connection_error")
                else:
                    self._retry_manager.record_attempt_result(False)
                if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                    delay = self._retry_manager.delay_for_attempt(attempt + 1)
                    if self.log_callback:
                        self.log_callback("llm_retry", f"   Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    continue

                if self.log_callback:
                    self.log_callback("llm_unexpected_error_fatal",
                        f"{RED}❌ All {MAX_TRANSLATION_ATTEMPTS} retry attempts exhausted{RESET}\n"
                        f"{RED}   Unexpected error persists - translation failed{RESET}\n"
                        f"{RED}   Please report this issue with the error details above{RESET}")
                else:
                    print(f"{RED}❌ All retry attempts exhausted. Translation failed.{RESET}")

                return None

            finally:
                http_status = response.status_code if response is not None else None
                observe_attempt(
                    response=result, status=attempt_status,
                    metadata={
                        "attempt": attempt + 1,
                        "elapsed_ms": int((http_finished - started) * 1000),
                        "started_at": started_at,
                        "budget_reservation_id": reservation_id,
                        "http_status": http_status,
                        "request_id": response.headers.get("x-request-id", "") if response is not None else "",
                        "error": error_type,
                        "reason": error_reason,
                        "usage_unknown": not bool(result and result.raw_usage),
                        "billing_uncertain": not bool(result and result.raw_usage) and (
                            http_status is None or http_status >= 500 or http_status == 200
                        ),
                        "parameters": {key: payload[key] for key in (
                            "model", "reasoning_effort", "max_completion_tokens", "temperature", "stream",
                            "service_tier", "prompt_cache_options",
                        ) if key in payload},
                    },
                )

        return None

    async def get_model_context_size(self) -> int:
        """Query server to get model's context size using ContextDetector."""
        if self._is_official_openai_endpoint() and self.model in EDITORIAL_MODELS:
            return 1_050_000
        if self._detected_context_size:
            return self._detected_context_size

        client = await self._get_client()
        ctx = await self._context_detector.detect(
            client=client,
            model=self.model,
            endpoint=self.api_endpoint,
            api_key=self.api_key,
            log_callback=self.log_callback
        )

        self._detected_context_size = ctx
        return ctx
