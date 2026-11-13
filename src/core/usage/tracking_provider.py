"""Provider proxy that records token/cost usage for every generate() call."""

from __future__ import annotations

import time
import inspect
import uuid
from typing import Optional

from src.config import REQUEST_TIMEOUT
from src.core.llm.base import LLMResponse
from src.core.llm.request_deadline import await_llm_call
from src.core.llm.attempts import attempt_observer, attempt_reserver

from .context import get_usage_context
from .store import default_usage_store
from .budget import BudgetLimitError, SpendBudget
from .request_budget import request_bound
from src.core.pipeline_runtime import active_pipeline, request_role


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
        response = await await_llm_call(self.generate, prompt, provider=self, role="translate")
        if response and not response.was_truncated and response.status == "completed":
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
        operation_id = uuid.uuid4().hex
        attempts = 0
        context = get_usage_context()
        runtime = active_pipeline.get()
        role = request_role.get()
        selected_provider, selected_name = self._wrapped, self.provider_name
        if runtime is not None:
            selected_provider, selected = runtime.provider_for(role)
            selected_name = selected["provider"]
        store = runtime.usage_store if runtime is not None else None
        scope_ids = list(dict.fromkeys([*(context.get("budget_scope_ids") or []),
                                       *(runtime.budget_scope_ids if runtime else [])]))
        selected_model = getattr(selected_provider, "model", self.model)
        common_metadata = {"effective_provider": selected_name, "effective_model": selected_model,
                           "role": role, "operation_id": operation_id}
        common_metadata.update({key: context[key] for key in (
            "unit_id", "profile_hash", "glossary_hash", "prompt_version", "reason",
        ) if key in context})
        if scope_ids and not getattr(selected_provider, "supports_attempt_reservations", False):
            raise BudgetLimitError("This provider does not support per-attempt reservations",
                                   reason="budget_configuration")

        def reserve(*, endpoint, payload, attempt):
            upper, metadata = request_bound(selected_name, endpoint, payload)
            metadata.update({"operation_id": operation_id, "attempt": attempt,
                             **{key: context[key] for key in (
                                 "translation_id", "process_id", "phase", "unit_id",
                                 "prompt_version", "profile_hash", "glossary_hash",
                             ) if key in context}})
            try:
                return SpendBudget(store or default_usage_store()).reserve(scope_ids, upper_usd=upper,
                                                                  metadata=metadata)
            except BudgetLimitError:
                raise
            except Exception as exc:
                raise BudgetLimitError("Unable to persist the pre-request budget reservation",
                                       reason="budget_ledger_unavailable") from exc

        def record_attempt(*, response, status, metadata):
            nonlocal attempts
            attempts += 1
            self._record(prompt=prompt, system_prompt=system_prompt or "", response=response,
                         status=status, store=store, metadata={**metadata, **common_metadata,
                                                  "scope": "http_attempt"})

        observer_token = attempt_observer.set(record_attempt)
        reserver_token = attempt_reserver.set(reserve if scope_ids else None)
        try:
            kwargs = {"timeout": timeout, "system_prompt": system_prompt}
            if _supports_temperature(selected_provider.generate):
                kwargs["temperature"] = temperature
            response = await await_llm_call(
                selected_provider.generate,
                prompt,
                provider=selected_provider,
                request_timeout=timeout,
                **kwargs,
            )
        except Exception as exc:
            if not attempts and not scope_ids:
                self._record(
                    prompt=prompt,
                    system_prompt=system_prompt or "",
                    response=getattr(exc, "response", None),
                    status="error",
                    store=store,
                    metadata={**common_metadata, "error": type(exc).__name__,
                              "reason": getattr(exc, "reason", ""),
                              "elapsed_ms": int((time.perf_counter() - started) * 1000)},
                )
            raise
        finally:
            attempt_observer.reset(observer_token)
            attempt_reserver.reset(reserver_token)

        if not attempts:
            self._record(
                prompt=prompt,
                system_prompt=system_prompt or "",
                response=response,
                status=("incomplete" if response.was_truncated else "ok") if response is not None else "empty_response",
                store=store,
                metadata={**common_metadata, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
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
        store=None,
    ) -> None:
        try:
            context = get_usage_context()
            if metadata.get("role"):
                context["phase"] = metadata["role"]
            elif not context.get("phase") or context.get("phase") == "llm_call":
                context["phase"] = _infer_phase(prompt, system_prompt)
            (store or default_usage_store()).record_call(
                provider=metadata.get("effective_provider") or self.provider_name,
                model=metadata.get("effective_model") or getattr(self._wrapped, "model", self.model),
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
                    "raw_usage": getattr(response, "raw_usage", {}) if response else {},
                    "service_tier": getattr(response, "service_tier", "") if response else "",
                    "request_id": (getattr(response, "request_id", "") if response else "") or metadata.get("request_id", ""),
                    "response_id": getattr(response, "response_id", "") if response else "",
                    "response_model": getattr(response, "response_model", "") if response else "",
                    "finish_reason": getattr(response, "finish_reason", "") if response else "",
                    "usage_observed": bool(response and (response.prompt_tokens or response.completion_tokens)),
                },
            )
        except Exception as exc:
            if metadata.get("budget_reservation_id"):
                if metadata.get("error") == "cancelled_in_flight":
                    # Cancellation wins; the durable pending allowance is not released.
                    return
                raise BudgetLimitError("Unable to persist request usage; allowance remains reserved",
                                       reason="budget_ledger_unavailable") from exc
            # Unbudgeted legacy calls retain best-effort telemetry behavior.
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
        "<combined_quality_audit_json>" in text
        or "independent bilingual quality auditor" in text
    ):
        return "combined_quality_audit"

    if (
        "fidelity supervisor" in text
        or "independent bilingual fidelity auditor" in text
        or "<fidelity_audit_json>" in text
        or "audita esta modernización" in text
        or "audita esta traducción" in text
        or '"overall_decision"' in text
        or '"content_fidelity"' in text
        or '"censored_or_softened"' in text
    ):
        return "fidelity_audit"

    if (
        "<editorial_guard_json>" in text
        or "source-aware editorial guard" in text
        or "juez fuente-aware" in text
    ):
        return "editorial_audit"

    if (
        "corrige la modernización usando la auditoría" in text
        or "corrige la traducción usando la auditoría" in text
        or "resolve los problemas marcados" in text
        or "suggested_fix" in text
        or "quality alert repair" in text
    ):
        return "repair"

    if (
        "technical placeholder correction specialist" in text
        or "placeholders were corrupted" in text
        or "correction request" in text
        or "corrige los marcadores" in text
        or "repair the placeholder" in text
        or "restore every placeholder" in text
    ):
        return "format_repair"

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
        or "editorial refinement" in text
        or "# draft to refine" in text
        or "final editorial review" in text
        or "refina el texto" in text
        or "corrige estilo" in text
    ):
        return "editorial_refinement"

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
