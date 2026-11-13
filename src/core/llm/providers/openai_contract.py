"""Official OpenAI text-model parameters and normalized response contract.

Verified 2026-11-12 against developers.openai.com/api/docs/models and
guides/latest-model. Compatible/local servers do not inherit these parameters.
"""

from src.core.llm.base import LLMResponse

EDITORIAL_MODELS = {"gpt-6-luna": "GPT-6 Luna", "gpt-6.1-terra": "GPT-6.1 Terra"}
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
MAX_OUTPUT_TOKENS = 128_000
MAX_CONTEXT_TOKENS = 1_050_000


def text_parameters(model, *, effort, max_output_tokens, temperature=None):
    if model not in EDITORIAL_MODELS:
        return {}
    if effort not in REASONING_EFFORTS:
        raise ValueError("Unsupported OpenAI reasoning effort")
    if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int):
        raise TypeError("OpenAI output limit must be an integer")
    if not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
        raise ValueError("OpenAI output limit is outside the supported range")
    result = {"reasoning_effort": effort, "max_completion_tokens": max_output_tokens,
              "service_tier": "default", "prompt_cache_options": {"ttl": "30m"}}
    if effort == "none" and temperature is not None:
        result["temperature"] = temperature
    return result


def normalize_completion(body, *, request_id="", latency_ms=0, context_limit=0):
    if not isinstance(body, dict):
        raise TypeError("Invalid completion envelope")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Completion has no choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("Completion has no message")
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise TypeError("Completion content is not text")
    reason = str(choice.get("finish_reason") or "").lower()
    refusal = str(message.get("refusal") or "")
    truncated = reason in {"length", "max_tokens", "max_output_tokens"}
    status = "refused" if refusal or reason == "content_filter" else (
        "incomplete" if truncated else "completed")
    if reason and reason not in {"stop", "length", "max_tokens", "max_output_tokens", "content_filter"}:
        raise ValueError("Unexpected completion finish reason")
    usage_fields = normalize_usage(body)
    return LLMResponse(
        content=content, **usage_fields, context_limit=context_limit,
        was_truncated=truncated, status=status, finish_reason=reason, refusal=refusal,
        request_id=request_id, response_id=str(body.get("id") or ""),
        response_model=str(body.get("model") or ""), latency_ms=latency_ms,
        service_tier=str(body.get("service_tier") or ""),
    )


def normalize_usage(body):
    """Extract observed usage even when a successful envelope has no valid choice."""
    usage = body.get("usage") or {}
    prompt = max(0, int(usage.get("prompt_tokens") or 0))
    output = max(0, int(usage.get("completion_tokens") or 0))
    details = usage.get("prompt_tokens_details") or {}
    cached = min(prompt, max(0, int(details.get("cached_tokens") or 0)))
    written = min(prompt-cached, max(0, int(details.get("cache_write_tokens") or 0)))
    reasoning = min(output, max(0, int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)))
    return {"prompt_tokens": prompt, "completion_tokens": output,
            "prompt_cache_hit_tokens": cached, "prompt_cache_miss_tokens": prompt-cached-written,
            "prompt_cache_write_tokens": written, "total_tokens": max(prompt+output, int(usage.get("total_tokens") or 0)),
            "reasoning_tokens": reasoning, "context_used": prompt+output, "raw_usage": usage}
