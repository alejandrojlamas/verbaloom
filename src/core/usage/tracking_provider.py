"""Provider proxy that records token/cost usage for every generate() call."""

from __future__ import annotations

import time
import inspect
from typing import Optional

from src.config import REQUEST_TIMEOUT
from src.core.llm.base import LLMResponse
from src.core.llm.request_deadline import await_llm_call

from .context import get_usage_context
from .store import default_usage_store


class UsageTrackingProvider:
    """Thin proxy over an LLM provider.

    It keeps the provider API intact while appending a metadata-only usage event
    after each request.
    """

    def __init__(self, wrapped, provider_name: str):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "provider_name", (provider_name or "").lower())
        object.__setattr__(self, "model", getattr(wrapped, "model", ""))

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    @property
    def __class__(self):
        """Expose the wrapped provider class for isinstance/introspection.

        The tracker is intentionally a transparent proxy. Some tests and
        integrations assert against the concrete provider type returned by the
        factory; forwarding ``__class__`` keeps those contracts intact while
        preserving usage tracking.
        """
        return self._wrapped.__class__

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name.startswith("_") or name in {"provider_name"}:
            return
        wrapped = object.__getattribute__(self, "_wrapped") if "_wrapped" in self.__dict__ else None
        if wrapped is not None and hasattr(wrapped, name):
            setattr(wrapped, name, value)

    async def close(self):
        close = getattr(self._wrapped, "close", None)
        if close:
            return await close()
        return None

    def extract_translation(self, response: str) -> Optional[str]:
        return self._wrapped.extract_translation(response)

    async def translate_text(self, prompt: str) -> Optional[str]:
        response = await self.generate(prompt)
        if response:
            return self.extract_translation(response.content)
        return None

    async def generate(
        self,
        prompt: str,
        timeout: int = REQUEST_TIMEOUT,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> Optional[LLMResponse]:
        started = time.perf_counter()
        try:
            kwargs = {"timeout": timeout, "system_prompt": system_prompt}
            if _supports_temperature(self._wrapped.generate):
                kwargs["temperature"] = temperature
            response = await await_llm_call(
                self._wrapped.generate,
                prompt,
                provider=self._wrapped,
                request_timeout=timeout,
                **kwargs,
            )
        except Exception as exc:
            self._record(
                prompt=prompt,
                system_prompt=system_prompt or "",
                response=None,
                status="error",
                metadata={"error": type(exc).__name__, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
            )
            raise

        self._record(
            prompt=prompt,
            system_prompt=system_prompt or "",
            response=response,
            status="ok" if response is not None else "empty_response",
            metadata={"elapsed_ms": int((time.perf_counter() - started) * 1000)},
        )
        return response

    def _record(
        self,
        *,
        prompt: str,
        system_prompt: str,
        response: Optional[LLMResponse],
        status: str,
        metadata: dict,
    ) -> None:
        try:
            context = get_usage_context()
            if not context.get("phase") or context.get("phase") == "llm_call":
                context["phase"] = _infer_phase(prompt, system_prompt)
            default_usage_store().record_call(
                provider=self.provider_name,
                model=getattr(self._wrapped, "model", self.model),
                prompt=prompt or "",
                system_prompt=system_prompt or "",
                response_content=getattr(response, "content", "") if response else "",
                prompt_tokens=int(getattr(response, "prompt_tokens", 0) or 0) if response else 0,
                completion_tokens=int(getattr(response, "completion_tokens", 0) or 0) if response else 0,
                prompt_cache_hit_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0) if response else 0,
                prompt_cache_miss_tokens=int(getattr(response, "prompt_cache_miss_tokens", 0) or 0) if response else 0,
                total_tokens=int(getattr(response, "total_tokens", 0) or 0) if response else 0,
                status=status,
                context=context,
                metadata={
                    **metadata,
                    "reasoning_tokens": int(getattr(response, "reasoning_tokens", 0) or 0) if response else 0,
                },
            )
        except Exception as exc:
            # Usage tracking must never break translation.
            print(f"⚠️ Token usage tracking failed: {exc}")


def _infer_phase(prompt: str, system_prompt: str = "") -> str:
    """Best-effort phase labels for cost attribution.

    This is intentionally heuristic and metadata-only. It does not change the
    prompt or provider behavior; it just prevents every cost event from landing
    in a generic "llm_call" bucket.
    """

    text = f"{system_prompt or ''}\n{prompt or ''}".lower()[:12000]
    if not text.strip():
        return "llm_call"

    if (
        "fidelity supervisor" in text
        or "audita esta modernización" in text
        or "audita esta traducción" in text
        or '"overall_decision"' in text
        or '"content_fidelity"' in text
    ):
        return "fidelity_audit"

    if (
        "corrige la modernización usando la auditoría" in text
        or "corrige la traducción usando la auditoría" in text
        or "resolve los problemas marcados" in text
        or "suggested_fix" in text
        or "quality alert repair" in text
    ):
        return "repair"

    if (
        "descubrimiento léxico" in text
        or "descubrimiento lexico" in text
        or "glossary_discovery" in text
        or "pending_suggestions" in text
        or '"suggestions"' in text and "glosario" in text
    ):
        return "glossary_discovery"

    if (
        "editor final de voz" in text
        or "restauración de voz" in text
        or "restauracion de voz" in text
        or "voice restoration" in text
    ):
        return "voice_restoration"

    if (
        "modernización intralingüística" in text
        or "modernizacion intralinguistica" in text
        or "moderniza todo lo que" in text
        or "español literario mexicano contemporáneo" in text
        or "espanol literario mexicano contemporaneo" in text
    ):
        return "modernization"

    if (
        "revisión editorial" in text
        or "revision editorial" in text
        or "refinamiento editorial" in text
        or "refina el texto" in text
        or "corrige estilo" in text
    ):
        return "editorial_refinement"

    if (
        "placeholder" in text
        or "marcadores de formato" in text
        or "correction request" in text
        or "corrige los marcadores" in text
    ):
        return "format_repair"

    if (
        "traduce" in text
        or "translate" in text
        or "<translation>" in text
        or "</translation>" in text
    ):
        return "translation"

    return "llm_call"


def _supports_temperature(fn) -> bool:
    try:
        return "temperature" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
