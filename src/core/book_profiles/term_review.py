"""Review and classify profile glossary candidates before approval."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import re
import time
from collections.abc import Callable
from typing import Any, Iterable, Mapping

from src.utils.json_extraction import (
    extract_tagged_payload,
    loads_first_json_value,
)
from src.core.glossary.lexical_policy import LexicalPolicy, resolve_lexical_policy
from src.core.llm.request_deadline import await_llm_call

from .profile_goals import ProfileGoalRules, resolve_profile_goal


TERM_REVIEW_TAG_IN = "<PROFILE_TERM_REVIEW_JSON>"
TERM_REVIEW_TAG_OUT = "</PROFILE_TERM_REVIEW_JSON>"

_REVIEW_BATCH_SIZE = 80
_MIN_RECOVERY_BATCH_SIZE = 16
_MIN_REVIEW_AUTO_APPROVE_CONFIDENCE = 0.88
_TERM_REVIEW_TIMEOUT_SECONDS = max(30, int(os.getenv("PROFILE_TERM_REVIEW_TIMEOUT", "240")))
_TERM_REVIEW_HEARTBEAT_SECONDS = max(10, int(os.getenv("PROFILE_TERM_REVIEW_HEARTBEAT_SECONDS", "25")))

_TRANSLATABLE_REVIEW_CATEGORIES = {
    "concept",
    "glossary",
    "idiom",
    "key_term",
    "phrase",
    "syntax_pattern",
    "technical",
    "technical_term",
    "term",
}
_PRESERVE_STATUSES = {"preserve_exact", "canonical_name"}
_SAFE_ACRONYM_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ]{2,6}[0-9A-ZÁÉÍÓÚÜÑ-]*$")
_SAFE_CODE_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ]{1,4}[0-9][0-9A-ZÁÉÍÓÚÜÑ-]{1,8}$")
_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]")
_SYMBOL_BEARING_ENTITY_RE = re.compile(r"[\\*]")
_GENERIC_DETERMINISTIC_PRESERVE_RATIONALES = {
    "looks like a recurring named entity for this book profile.",
    "looks like a recurring named entity for this profile.",
}


@dataclass(frozen=True)
class TermReviewSummary:
    reviewed_terms: int = 0
    auto_approved_preserve: int = 0
    auto_approved_translations: int = 0
    pending_review: int = 0
    rejected_noise: int = 0
    demoted_entries: int = 0
    llm_calls: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "reviewed_terms": self.reviewed_terms,
            "auto_approved_preserve": self.auto_approved_preserve,
            "auto_approved_translations": self.auto_approved_translations,
            "pending_review": self.pending_review,
            "rejected_noise": self.rejected_noise,
            "demoted_entries": self.demoted_entries,
            "llm_calls": self.llm_calls,
        }


async def review_profile_terms(
    candidates: Iterable[Mapping[str, Any]],
    *,
    profile_id: str,
    source_name: str = "",
    source_language: str = "",
    language: str = "",
    target_locale: str = "",
    transform_mode: str = "",
    llm_provider: Any = None,
    model: str = "",
    profile_goal: str = "",
    goal_rules: ProfileGoalRules | None = None,
    text_chars: int = 0,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_base: int = 36,
    progress_span: int = 3,
    request_timeout: int | None = None,
) -> tuple[list[dict[str, Any]], TermReviewSummary]:
    """Classify glossary candidates before anything becomes approved.

    The deterministic pass is intentionally conservative. When an LLM reviewer
    is available, it can promote high-confidence technical translations, but
    deterministic vetoes still prevent English-as-English technical rules from
    entering the approved glossary.
    """
    rules = goal_rules or resolve_profile_goal(
        profile_goal,
        transform_mode=transform_mode,
        source_name=source_name,
    )
    reviewed = [
        _with_deterministic_review(
            item,
            transform_mode=transform_mode,
            target_locale=target_locale,
            source_language=source_language,
            goal_rules=rules,
        )
        for item in candidates
    ]
    llm_calls = 0

    if llm_provider is not None:
        review_limit = rules.review_terms_limit(
            text_chars=text_chars,
            candidate_count=len(reviewed),
        )
        batch_size = max(20, int(rules.review_batch_size or _REVIEW_BATCH_SIZE))
        reviewable = [
            item for item in sorted(reviewed, key=_review_priority, reverse=True)
            if item.get("review_status") not in {"preserve_exact", "reject_noise"}
        ][:review_limit]
        total_batches = (len(reviewable) + batch_size - 1) // batch_size
        if total_batches:
            _emit_term_review_progress(
                progress_callback,
                progress=progress_base,
                message=(
                    f"Revisando terminos con {model or 'reviewer'}: "
                    f"{len(reviewable)} candidatos para {rules.label.lower()} en {total_batches} lotes."
                ),
                term_batch_index=0,
                term_batch_total=total_batches,
                terms_reviewed=0,
                terms_total=len(reviewable),
                review_model=model,
                profile_goal=rules.key,
            )
        timeout = int(request_timeout or _TERM_REVIEW_TIMEOUT_SECONDS)
        for batch_number, batch_start in enumerate(range(0, len(reviewable), batch_size), start=1):
            batch = reviewable[batch_start:batch_start + batch_size]
            if not batch:
                continue
            _emit_term_review_progress(
                progress_callback,
                progress=_term_review_progress(progress_base, progress_span, batch_number - 1, total_batches),
                message=(
                    f"Revisando lote {batch_number}/{total_batches} con "
                    f"{model or 'reviewer'} ({len(batch)} terminos)."
                ),
                term_batch_index=batch_number,
                term_batch_total=total_batches,
                terms_reviewed=batch_start,
                terms_total=len(reviewable),
                review_model=model,
                profile_goal=rules.key,
            )
            decisions, batch_calls, batch_warnings = await _review_term_batch_resilient(
                batch,
                llm_provider=llm_provider,
                profile_id=profile_id,
                source_name=source_name,
                source_language=source_language,
                language=language,
                target_locale=target_locale,
                transform_mode=transform_mode,
                goal_rules=rules,
                timeout=timeout,
                progress_callback=progress_callback,
                progress=_term_review_progress(
                    progress_base,
                    progress_span,
                    batch_number - 1,
                    total_batches,
                ),
                batch_number=batch_number,
                total_batches=total_batches,
                batch_start=batch_start,
                total_terms=len(reviewable),
                model=model,
            )
            llm_calls += batch_calls
            if decisions:
                _merge_llm_decisions(
                    reviewed,
                    decisions,
                    model=model or "llm",
                    target_locale=target_locale,
                    source_language=source_language,
                    goal_rules=rules,
                )
            _emit_term_review_progress(
                progress_callback,
                progress=_term_review_progress(progress_base, progress_span, batch_number, total_batches),
                message=(
                    f"Lote {batch_number}/{total_batches} revisado: "
                    f"{len(decisions)} decisiones recibidas."
                ),
                term_batch_index=batch_number,
                term_batch_total=total_batches,
                terms_reviewed=min(batch_start + len(batch), len(reviewable)),
                terms_total=len(reviewable),
                review_model=model,
                review_decisions=len(decisions),
                review_warning="; ".join(batch_warnings[:3]) if batch_warnings else None,
                profile_goal=rules.key,
            )

    summary = _summarize(reviewed, llm_calls=llm_calls)
    return reviewed, summary


async def _review_term_batch_resilient(
    batch: list[dict[str, Any]],
    *,
    llm_provider: Any,
    profile_id: str,
    source_name: str,
    source_language: str,
    language: str,
    target_locale: str,
    transform_mode: str,
    goal_rules: ProfileGoalRules,
    timeout: float,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress: int,
    batch_number: int,
    total_batches: int,
    batch_start: int,
    total_terms: int,
    model: str,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Review one logical batch with bounded, iterative split recovery.

    Provider retry loops historically applied ``timeout`` to every network
    attempt, so one 70-term batch could occupy the profile worker for eight or
    more minutes.  A total deadline now bounds each submitted request.  When a
    large request times out or returns malformed structured output, only that
    request is split into smaller independent batches; deterministic decisions
    remain available for any irrecoverable leaf.
    """
    queue: list[tuple[list[dict[str, Any]], int]] = [(list(batch), batch_start)]
    decisions: list[dict[str, Any]] = []
    warnings: list[str] = []
    calls = 0
    logical_deadline = time.monotonic() + max(0.01, float(timeout))

    while queue:
        current, current_start = queue.pop(0)
        remaining_budget = logical_deadline - time.monotonic()
        if remaining_budget <= 0:
            unresolved = [(current, current_start), *queue]
            unresolved_count = sum(len(items) for items, _ in unresolved)
            warning = (
                f"{unresolved_count} terminos conservaron revision deterministica "
                "porque se agoto el presupuesto total del lote"
            )
            warnings.append(warning)
            _emit_term_review_progress(
                progress_callback,
                progress=progress,
                message=warning,
                term_batch_index=batch_number,
                term_batch_total=total_batches,
                terms_reviewed=min(
                    total_terms,
                    max(start + len(items) for items, start in unresolved),
                ),
                terms_total=total_terms,
                review_model=model,
                review_warning="presupuesto total agotado",
            )
            break

        pending_requests = len(queue) + 1
        if calls == 0:
            # Reserve part of the logical budget for smaller recovery calls.
            attempt_timeout = min(
                remaining_budget,
                max(0.01, float(timeout) * 0.60),
            )
        else:
            attempt_timeout = max(
                0.01,
                remaining_budget / pending_requests,
            )
        system, user = build_term_review_prompt(
            current,
            profile_id=profile_id,
            source_name=source_name,
            source_language=source_language,
            language=language,
            target_locale=target_locale,
            transform_mode=transform_mode,
            goal_rules=goal_rules,
        )
        failure = ""
        response = None
        calls += 1
        try:
            response = await _generate_term_review_batch(
                llm_provider,
                user,
                system,
                timeout=attempt_timeout,
                provider_timeout=timeout,
                progress_callback=progress_callback,
                progress=progress,
                batch_number=batch_number,
                total_batches=total_batches,
                batch_start=current_start,
                batch_size=len(current),
                total_terms=total_terms,
                model=model,
            )
        except asyncio.TimeoutError:
            failure = f"deadline de {attempt_timeout:.1f}s agotado"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"

        parsed = parse_term_review_payload(
            getattr(response, "content", "") if response is not None else ""
        )
        if response is not None and not parsed:
            failure = "respuesta estructurada vacia o invalida"

        if parsed:
            decisions.extend(parsed)
            continue

        if len(current) > _MIN_RECOVERY_BATCH_SIZE:
            split_at = max(1, len(current) // 2)
            left = current[:split_at]
            right = current[split_at:]
            queue[0:0] = [
                (left, current_start),
                (right, current_start + len(left)),
            ]
            _emit_term_review_progress(
                progress_callback,
                progress=progress,
                message=(
                    f"El lote {batch_number}/{total_batches} no respondio de forma util "
                    f"({failure or 'sin respuesta'}); se recupera en sublotes de "
                    f"{len(left)} y {len(right)} terminos."
                ),
                term_batch_index=batch_number,
                term_batch_total=total_batches,
                terms_reviewed=current_start,
                terms_total=total_terms,
                review_model=model,
                review_recovery="split",
            )
            continue

        warning = (
            f"{len(current)} terminos conservaron revision deterministica "
            f"({failure or 'sin respuesta'})"
        )
        warnings.append(warning)
        _emit_term_review_progress(
            progress_callback,
            progress=progress,
            message=warning,
            term_batch_index=batch_number,
            term_batch_total=total_batches,
            terms_reviewed=current_start + len(current),
            terms_total=total_terms,
            review_model=model,
            review_warning=failure or "sin respuesta",
        )

    return decisions, calls, warnings


async def _generate_term_review_batch(
    llm_provider: Any,
    user: str,
    system: str,
    *,
    timeout: float,
    provider_timeout: float | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
    progress: int,
    batch_number: int,
    total_batches: int,
    batch_start: int,
    batch_size: int,
    total_terms: int,
    model: str,
) -> Any:
    heartbeat_task: asyncio.Task[None] | None = None
    if progress_callback is not None:
        heartbeat_task = asyncio.create_task(_term_review_heartbeat(
            progress_callback,
            progress=progress,
            batch_number=batch_number,
            total_batches=total_batches,
            batch_start=batch_start,
            batch_size=batch_size,
            total_terms=total_terms,
            model=model,
        ))
    try:
        # ``timeout`` on providers is commonly per network attempt.  The outer
        # deadline is deliberately the same value so retries cannot multiply a
        # 240-second UI wait into 8-12 minutes.
        return await await_llm_call(
            llm_provider.generate,
            user,
            provider=llm_provider,
            request_timeout=provider_timeout or timeout,
            deadline=timeout,
            system_prompt=system,
        )
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass


async def _term_review_heartbeat(
    progress_callback: Callable[[dict[str, Any]], None],
    *,
    progress: int,
    batch_number: int,
    total_batches: int,
    batch_start: int,
    batch_size: int,
    total_terms: int,
    model: str,
) -> None:
    elapsed = 0
    while True:
        await asyncio.sleep(_TERM_REVIEW_HEARTBEAT_SECONDS)
        elapsed += _TERM_REVIEW_HEARTBEAT_SECONDS
        _emit_term_review_progress(
            progress_callback,
            progress=progress,
            message=(
                f"DeepSeek sigue revisando lote {batch_number}/{total_batches} "
                f"({elapsed}s sin respuesta nueva)."
            ),
            term_batch_index=batch_number,
            term_batch_total=total_batches,
            terms_reviewed=batch_start,
            terms_total=total_terms,
            review_model=model,
            review_batch_terms=batch_size,
        )


def _term_review_progress(base: int, span: int, completed_batches: int, total_batches: int) -> int:
    if total_batches <= 0:
        return base
    completed_batches = max(0, min(completed_batches, total_batches))
    return int(round(base + (span * completed_batches / total_batches)))


def _emit_term_review_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    *,
    progress: int,
    message: str,
    term_batch_index: int,
    term_batch_total: int,
    terms_reviewed: int,
    terms_total: int,
    review_model: str = "",
    **extra: Any,
) -> None:
    if progress_callback is None:
        return
    event = {
        "stage": "term_review",
        "progress": progress,
        "message": message,
        "term_batch_index": term_batch_index,
        "term_batch_total": term_batch_total,
        "chunk_index": term_batch_index,
        "chunk_total": term_batch_total,
        "terms_reviewed": terms_reviewed,
        "terms_total": terms_total,
        "review_model": review_model,
    }
    event.update({key: value for key, value in extra.items() if value is not None})
    try:
        progress_callback(event)
    except Exception:
        pass


def build_term_review_prompt(
    candidates: Iterable[Mapping[str, Any]],
    *,
    profile_id: str,
    source_name: str = "",
    source_language: str = "",
    language: str = "",
    target_locale: str = "",
    transform_mode: str = "",
    goal_rules: ProfileGoalRules | None = None,
) -> tuple[str, str]:
    rules = goal_rules or resolve_profile_goal(transform_mode, transform_mode=transform_mode, source_name=source_name)
    system = f"""
You are a profile-specific glossary review agent for a book translation system.

Classify candidate glossary terms before they are approved. The active profile is {profile_id}.
Do not create global rules. Do not preserve source-language common or technical terms when they have a normal direct translation in the target locale.

{rules.prompt_brief()}

Return only valid JSON wrapped in {TERM_REVIEW_TAG_IN} and {TERM_REVIEW_TAG_OUT}.

For each candidate return:
{{
  "source": "...",
  "review_status": "preserve_exact | canonical_name | translate_exact | translate_contextual | pending_review | reject_noise",
  "target": "...",
  "entry_type": "proper_noun | acronym | technical_term | concept | title | term",
  "confidence": 0.0,
  "injection_policy": "preserve | translate_exact | contextual | do_not_inject",
  "translation_policy": "preserve_exact | translate_exact | translate_contextual | pending_review | reject_noise",
  "rationale": "..."
}}

Rules:
- preserve_exact only for real proper names, acronyms, codes, identifiers, works, places, or entities that should remain as written.
- translate_exact for technical terms with a stable direct translation.
- translate_contextual when a concept needs consistency but not a mechanical replacement. Always return a non-empty recommended target/base rendering for this status; the pipeline uses it as contextual guidance, not as a global replacement.
- Recurring coined compounds, titles, or capitalized concepts with one stable target-language rendering should use translate_exact. Capitalization or hyphenation alone does not make a term a proper name.
- reject_noise for headings, OCR fragments, generic instructions, all-caps running text, or copied prose.
- If source and target would be identical for a translatable English technical term, use pending_review or translate_contextual, not preserve_exact.
- Source language is {source_language or "unknown/auto"}; target language is {language or target_locale or "unknown"}.
- In languages such as German that capitalize common nouns, capitalization alone is never evidence of a proper name.
- For a one-word person, place, work, or institution name, use canonical_name only when the supplied context clearly proves it is an entity.
- Use {target_locale or language or "the target locale"} as the translation target.
""".strip()
    user = {
        "profile_id": profile_id,
        "source_name": source_name,
        "source_language": source_language,
        "target_language": language,
        "language": language,
        "target_locale": target_locale,
        "transform_mode": transform_mode,
        "profile_goal": rules.key,
        "business_rules": rules.to_config(),
        "candidates": [_compact_candidate(item) for item in candidates],
    }
    return system, json.dumps(user, ensure_ascii=False, separators=(",", ":"))


def parse_term_review_payload(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    payload = (
        extract_tagged_payload(text, TERM_REVIEW_TAG_IN, TERM_REVIEW_TAG_OUT)
        or text
    )
    parsed = loads_first_json_value(payload)
    if parsed is None:
        return []
    if isinstance(parsed, Mapping):
        raw = parsed.get("terms") or parsed.get("decisions") or parsed.get("reviews") or []
    elif isinstance(parsed, list):
        raw = parsed
    else:
        raw = []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def reviewed_candidate_to_approved_entry(
    item: Mapping[str, Any],
    *,
    profile_id: str,
) -> dict[str, Any] | None:
    source = str(item.get("source") or "").strip()
    if not source:
        return None
    status = str(item.get("review_status") or "").strip().lower()
    confidence = _as_float(item.get("review_confidence") or item.get("confidence"), default=0.0)
    target = str(item.get("review_target") or item.get("target") or "").strip()
    entry_type = str(item.get("review_entry_type") or item.get("category") or item.get("type") or "term").strip().lower()
    lexical_policy = _active_lexical_policy(str(item.get("source_language") or ""))

    if status in {"preserve_exact", "canonical_name"}:
        target = target or source
    if confidence < _MIN_REVIEW_AUTO_APPROVE_CONFIDENCE:
        return None
    if status == "preserve_exact" and not _preserve_exact_allowed(
        source,
        entry_type,
        lexical_policy=lexical_policy,
    ):
        return None
    if item.get("review_demoted_reason"):
        return None
    if status == "translate_exact" and (not target or target.casefold() == source.casefold()):
        return None
    if status == "canonical_name" and not target:
        return None
    if status not in {"preserve_exact", "translate_exact", "canonical_name"}:
        return None

    normalized_type = _entry_type_for_review(status, entry_type)
    return {
        "source": source,
        "target": target,
        "type": normalized_type,
        "scope": profile_id,
        "status": "approved",
        "confidence": round(confidence, 3),
        "occurrences": _as_int(item.get("occurrences"), default=0),
        "mechanical_safe": bool(status in {"preserve_exact", "canonical_name"} and normalized_type in {"proper_noun", "acronym", "canonical_proper_noun"}),
        "translation_policy": status,
        "injection_policy": _injection_policy_for_review(status),
        "review_status": status,
        "review_confidence": round(confidence, 3),
        "reviewed_by": str(item.get("reviewed_by") or "profile_term_review"),
        "source_language": str(item.get("source_language") or "").strip(),
        "review_rationale": str(item.get("review_rationale") or item.get("rationale") or "").strip(),
        "rationale": str(item.get("review_rationale") or item.get("rationale") or "").strip()
        or "Profile term reviewer approved this entry for the active book profile.",
        "examples": list(item.get("examples") or [])[:2],
    }


def review_to_pending_suggestion(item: Mapping[str, Any], *, profile_id: str) -> dict[str, Any] | None:
    source = str(item.get("source") or "").strip()
    if not source:
        return None
    status = str(item.get("review_status") or "").strip().lower()
    if status == "reject_noise":
        return None
    target = str(item.get("review_target") or item.get("target") or "").strip()
    confidence = _as_float(item.get("review_confidence") or item.get("confidence"), default=0.0)
    risks = list(item.get("risks") or ["Do not approve automatically without checking translation policy."])
    if item.get("review_demoted_reason"):
        risks.insert(0, str(item.get("review_demoted_reason")))
    return {
        "source": source,
        "suggested_target": target if target.casefold() != source.casefold() else "",
        "type": _entry_type_for_review(status, str(item.get("category") or item.get("type") or "term")),
        "scope": profile_id,
        "status": "pending",
        "confidence": round(confidence, 3),
        "occurrences": _as_int(item.get("occurrences"), default=0),
        "translation_policy": status or "pending_review",
        "injection_policy": _injection_policy_for_review(status),
        "review_status": status or "pending_review",
        "review_confidence": round(confidence, 3),
        "reviewed_by": str(item.get("reviewed_by") or "profile_term_review"),
        "source_language": str(item.get("source_language") or "").strip(),
        "review_rationale": str(item.get("review_rationale") or item.get("rationale") or "").strip(),
        "rationale": str(item.get("review_rationale") or item.get("rationale") or "").strip()
        or "Profile term reviewer left this entry pending for human or later LLM review.",
        "examples": list(item.get("examples") or [])[:2],
        "risks": risks[:4],
    }


def suspicious_preserve_entry(entry: Mapping[str, Any]) -> bool:
    source = str(entry.get("source") or "").strip()
    target = str(entry.get("target") or entry.get("suggested_target") or "").strip()
    if not source or not target or source.casefold() != target.casefold():
        return False
    entry_type = str(entry.get("type") or entry.get("entry_type") or "").strip().lower()
    lexical_policy = _active_lexical_policy(str(entry.get("source_language") or ""))
    if (
        _single_word_reject_noise(source, lexical_policy)
        or _looks_fragment_noise(source, lexical_policy)
    ):
        return True
    if _source_equals_target_translatable_word(source, lexical_policy):
        return True
    if weakly_supported_generated_preserve_entry(
        entry,
        lexical_policy=lexical_policy,
    ):
        return True
    if entry_type == "acronym":
        return not _safe_acronym(source, lexical_policy)
    return _looks_translatable_or_noise(
        source,
        entry_type,
        lexical_policy=lexical_policy,
    )


def weakly_supported_generated_preserve_entry(
    entry: Mapping[str, Any],
    *,
    lexical_policy: LexicalPolicy | None = None,
) -> bool:
    """Identify generated source==target rules that lack entity evidence.

    A title-cased span is not enough proof that a phrase is a proper name. The
    local extractor can join adjacent dialogue fragments (for example an
    interjection followed by a speaker name) and the old deterministic review
    promoted those spans without sending them to the LLM reviewer. Explicit
    human/LLM decisions remain valid; this only distrusts generic generated
    approvals with no contextual rationale.
    """
    source = str(entry.get("source") or "").strip()
    target = str(entry.get("target") or entry.get("suggested_target") or "").strip()
    if not source or not target or source.casefold() != target.casefold():
        return False
    lexical_policy = lexical_policy or _active_lexical_policy(
        str(entry.get("source_language") or "")
    )
    policy = str(
        entry.get("injection_policy") or entry.get("translation_policy") or ""
    ).strip().lower()
    if policy and policy not in {"preserve", "preserve_exact"}:
        return False
    entry_type = str(entry.get("type") or entry.get("entry_type") or "").strip().lower()
    if entry_type == "acronym" and _safe_acronym(source, lexical_policy):
        return False
    if _SAFE_CODE_RE.fullmatch(source) or _SYMBOL_BEARING_ENTITY_RE.search(source):
        return False

    reviewer = str(entry.get("reviewed_by") or "").strip().lower()
    source_kind = str(entry.get("source_kind") or "").strip().lower()
    rationale = re.sub(
        r"\s+",
        " ",
        str(entry.get("review_rationale") or entry.get("rationale") or "").strip(),
    ).casefold()
    generic_generated = (
        reviewer == "deterministic_profile_term_review"
        or (
            source_kind == "reviewed_glossary"
            and rationale in _GENERIC_DETERMINISTIC_PRESERVE_RATIONALES
        )
    )
    if not generic_generated:
        return False
    return entry_type in {
        "canonical_proper_noun",
        "character",
        "location",
        "organization",
        "proper_noun",
        "title",
    }


def _with_deterministic_review(
    item: Mapping[str, Any],
    *,
    transform_mode: str = "",
    target_locale: str = "",
    source_language: str = "",
    goal_rules: ProfileGoalRules | None = None,
) -> dict[str, Any]:
    rules = goal_rules or resolve_profile_goal(transform_mode, transform_mode=transform_mode)
    lexical_policy = _active_lexical_policy(source_language)
    data = dict(item)
    source = str(data.get("source") or "").strip()
    category = str(data.get("category") or data.get("type") or "term").strip().lower()
    confidence = _as_float(data.get("confidence"), default=0.0)

    if not source or _looks_reject_noise(source, category, lexical_policy):
        status = "reject_noise"
        review_confidence = max(confidence, 0.9)
        policy = "do_not_inject"
        target = ""
        rationale = "Rejected as a heading, generic instruction, all-caps prose, or extraction noise."
    elif category == "acronym" and _safe_acronym(source, lexical_policy):
        status = "preserve_exact"
        review_confidence = max(confidence, 0.9)
        policy = "preserve"
        target = source
        rationale = "Looks like a real acronym or identifier that should remain stable."
    elif _looks_translatable_or_noise(
        source,
        category,
        lexical_policy=lexical_policy,
        goal_rules=rules,
    ):
        status = "pending_review"
        review_confidence = min(max(confidence, 0.72), 0.86)
        policy = "contextual"
        target = ""
        rationale = "Looks like a translatable technical term or heading; do not preserve the source language by default."
    elif _capitalized_common_noun_language_risk(source, lexical_policy):
        status = "pending_review"
        review_confidence = min(max(confidence, 0.72), 0.86)
        policy = "contextual"
        target = ""
        rationale = (
            "The source language capitalizes common nouns; a capitalized candidate "
            "needs contextual entity review before it can be preserved."
        )
    elif (
        category in set(rules.preserve_categories)
        and _looks_named_entity(
            source,
            lexical_policy=lexical_policy,
            goal_rules=rules,
        )
        and _has_strong_deterministic_entity_evidence(
            source,
            category,
            lexical_policy,
        )
    ):
        status = "preserve_exact"
        review_confidence = max(confidence, 0.9)
        policy = "preserve"
        target = source
        rationale = "Looks like a recurring named entity for this book profile."
    elif (
        category in set(rules.preserve_categories)
        and _looks_named_entity(
            source,
            lexical_policy=lexical_policy,
            goal_rules=rules,
        )
    ):
        status = "pending_review"
        review_confidence = min(max(confidence, 0.72), 0.86)
        policy = "contextual"
        target = ""
        rationale = (
            "Capitalization suggests an entity, but contextual review is required "
            "before source wording can be preserved exactly."
        )
    else:
        status = "pending_review"
        review_confidence = min(max(confidence, 0.62), 0.84)
        policy = "contextual"
        target = ""
        rationale = "Useful candidate, but it needs contextual review before approval."

    data.update({
        "review_status": status,
        "review_confidence": round(review_confidence, 3),
        "review_target": target,
        "review_entry_type": _entry_type_for_review(status, category),
        "injection_policy": policy,
        "translation_policy": status,
        "review_rationale": rationale,
        "reviewed_by": "deterministic_profile_term_review",
        "source_language": source_language,
    })
    if (
        data["review_status"] == "preserve_exact"
        and _needs_canonical_review_before_preserve(
            source,
            category,
            transform_mode=transform_mode,
            target_locale=target_locale,
        )
    ):
        data.update({
            "review_status": "pending_review",
            "review_confidence": min(_as_float(data.get("review_confidence"), default=0.0), 0.86),
            "review_target": "",
            "injection_policy": "contextual",
            "translation_policy": "pending_review",
            "review_rationale": (
                "Single-word historical name in Spanish modernization needs canonical review "
                "before being preserved."
            ),
        })
    if _same_target_translatable_term(
        source,
        category=category,
        entry_type=str(data.get("review_entry_type") or ""),
        status=str(data.get("review_status") or ""),
        target=str(data.get("review_target") or source),
        target_locale=target_locale,
        source_language=source_language,
        goal_rules=rules,
    ):
        data.update(_demote_source_equals_target_review(data))
    return data


def _merge_llm_decisions(
    reviewed: list[dict[str, Any]],
    decisions: list[Mapping[str, Any]],
    *,
    model: str,
    target_locale: str = "",
    source_language: str = "",
    goal_rules: ProfileGoalRules | None = None,
) -> None:
    rules = goal_rules or resolve_profile_goal("")
    lexical_policy = _active_lexical_policy(source_language)
    by_key = {str(item.get("source") or "").casefold(): item for item in reviewed}
    for decision in decisions:
        source = str(decision.get("source") or "").strip()
        if not source:
            continue
        item = by_key.get(source.casefold())
        if item is None:
            continue
        status = str(decision.get("review_status") or decision.get("status") or "").strip().lower()
        if status not in {"preserve_exact", "canonical_name", "translate_exact", "translate_contextual", "pending_review", "reject_noise"}:
            continue
        target = str(decision.get("target") or decision.get("suggested_target") or "").strip()
        confidence = max(0.0, min(1.0, _as_float(decision.get("confidence"), default=0.0)))
        entry_type = str(decision.get("entry_type") or decision.get("type") or item.get("review_entry_type") or "term").strip().lower()

        if (
            status == "preserve_exact"
            and entry_type in {"proper_noun", "character", "location", "organization", "title"}
            and _capitalized_common_noun_language_risk(source, lexical_policy)
        ):
            status = "pending_review"
            target = ""
            confidence = min(confidence, 0.86)
            item["review_demoted_reason"] = (
                "capitalized_common_noun_risk: capitalization alone does not prove a "
                "source candidate is a proper name."
            )

        if status == "preserve_exact" and not _preserve_exact_allowed(
            source,
            entry_type,
            lexical_policy=lexical_policy,
            goal_rules=rules,
        ):
            status = "pending_review"
            target = ""
            confidence = min(confidence, 0.86)
            if (
                _looks_translatable_or_noise(
                    source,
                    entry_type,
                    lexical_policy=lexical_policy,
                    goal_rules=rules,
                )
                or _looks_translatable_or_noise(
                    source,
                    str(item.get("category") or item.get("type") or ""),
                    lexical_policy=lexical_policy,
                    goal_rules=rules,
                )
            ):
                item["review_demoted_reason"] = (
                    "source_equals_target_translatable: the reviewer tried to preserve a "
                    "term that appears directly translatable in the target language."
                )
        if _same_target_translatable_term(
            source,
            category=str(item.get("category") or item.get("type") or ""),
            entry_type=entry_type,
            status=status,
            target=target or source,
            target_locale=target_locale,
            source_language=source_language,
            goal_rules=rules,
        ):
            status = "pending_review"
            target = ""
            confidence = min(confidence, 0.86)
            item["review_demoted_reason"] = (
                "source_equals_target_translatable: the reviewer tried to preserve a "
                "term that appears directly translatable in the target language."
            )
        else:
            if status != "pending_review":
                item.pop("review_demoted_reason", None)
        if status == "translate_exact" and (not target or target.casefold() == source.casefold()):
            status = "pending_review"
            confidence = min(confidence, 0.86)
            item["review_demoted_reason"] = (
                "source_equals_target_translatable: translate_exact cannot use an identical target."
            )
        if status in {"canonical_name", "translate_exact", "translate_contextual"} and target:
            item["review_target"] = target
        elif status == "preserve_exact":
            item["review_target"] = source

        injection_policy = str(decision.get("injection_policy") or _injection_policy_for_review(status))
        translation_policy = str(decision.get("translation_policy") or status)
        if status == "pending_review" and item.get("review_demoted_reason"):
            injection_policy = "contextual"
            translation_policy = "pending_review"
        item.update({
            "review_status": status,
            "review_confidence": round(confidence, 3),
            "review_entry_type": _entry_type_for_review(status, entry_type),
            "injection_policy": injection_policy,
            "translation_policy": translation_policy,
            "review_rationale": str(decision.get("rationale") or decision.get("review_rationale") or "").strip(),
            "reviewed_by": f"llm_profile_term_review:{model}",
        })


def _summarize(reviewed: list[Mapping[str, Any]], *, llm_calls: int) -> TermReviewSummary:
    approved_preserve = 0
    approved_translate = 0
    pending = 0
    rejected = 0
    demoted = 0
    for item in reviewed:
        status = str(item.get("review_status") or "").lower()
        confidence = _as_float(item.get("review_confidence"), default=0.0)
        source = str(item.get("source") or "")
        target = str(item.get("review_target") or item.get("target") or "")
        if status == "reject_noise":
            rejected += 1
        elif status == "preserve_exact" and confidence >= _MIN_REVIEW_AUTO_APPROVE_CONFIDENCE:
            approved_preserve += 1
        elif status in {"translate_exact", "canonical_name"} and confidence >= _MIN_REVIEW_AUTO_APPROVE_CONFIDENCE:
            approved_translate += 1
        else:
            pending += 1
        if item.get("review_demoted_reason"):
            demoted += 1
        elif source and target and source.casefold() == target.casefold() and status != "preserve_exact":
            demoted += 1
    return TermReviewSummary(
        reviewed_terms=len(reviewed),
        auto_approved_preserve=approved_preserve,
        auto_approved_translations=approved_translate,
        pending_review=pending,
        rejected_noise=rejected,
        demoted_entries=demoted,
        llm_calls=llm_calls,
    )


def _review_priority(item: Mapping[str, Any]) -> tuple[float, int, int]:
    status = str(item.get("review_status") or "")
    priority = 1 if status in {"pending_review", "translate_contextual"} else 0
    return (
        priority,
        _as_int(item.get("occurrences"), default=0),
        len(str(item.get("source") or "")),
    )


def _compact_candidate(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source": str(item.get("source") or "")[:120],
        "category": str(item.get("category") or item.get("type") or ""),
        "occurrences": _as_int(item.get("occurrences"), default=0),
        "confidence": _as_float(item.get("confidence"), default=0.0),
        "review_status": str(item.get("review_status") or ""),
        "contexts": [str(x)[:220] for x in (item.get("contexts") or [])[:2]],
    }


def _looks_reject_noise(
    source: str,
    category: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    words = source.split()
    if not words:
        return True
    if (
        _single_word_reject_noise(source, lexical_policy)
        or _looks_fragment_noise(source, lexical_policy)
    ):
        return True
    if len(words) > 1 and words[0].casefold() in lexical_policy.leading_noise_words:
        return True
    if category == "acronym" and not _safe_acronym(source, lexical_policy):
        return True
    if len(words) >= 3 and all(
        word[:1].isupper() or word.casefold() in lexical_policy.connectors
        for word in words
    ):
        lowered = source.casefold()
        if any(
            hint in lowered
            for hint in lexical_policy.translatable_single_words
        ):
            return True
    return False


def _looks_translatable_or_noise(
    source: str,
    category: str,
    *,
    lexical_policy: LexicalPolicy,
    goal_rules: ProfileGoalRules | None = None,
) -> bool:
    folded = source.casefold()
    words = folded.split()
    if _source_equals_target_translatable_word(source, lexical_policy):
        return True
    if goal_rules is not None and category in set(goal_rules.translatable_categories):
        return True
    if len(words) == 1 and folded in lexical_policy.translatable_single_words:
        return True
    if category in {"technical", "concept", "technical_term"}:
        return True
    return any(
        hint in words or hint in folded
        for hint in lexical_policy.translatable_phrase_hints
    )


def _same_target_translatable_term(
    source: str,
    *,
    category: str,
    entry_type: str,
    status: str,
    target: str,
    target_locale: str = "",
    source_language: str = "",
    goal_rules: ProfileGoalRules | None = None,
) -> bool:
    """Return True when a source==target preservation looks like a translatable term.

    This is deliberately profile-agnostic. It does not translate anything; it
    only prevents an approved glossary entry from instructing the model to keep
    a source-language technical/common term unchanged when the active target
    language should receive a normal translation.
    """
    source = str(source or "").strip()
    target = str(target or "").strip()
    lexical_policy = _active_lexical_policy(source_language)
    if not source:
        return False
    if target and target.casefold() != source.casefold():
        return False
    if status not in _PRESERVE_STATUSES and status != "translate_exact":
        return False
    if _safe_acronym(source, lexical_policy) or _SAFE_CODE_RE.fullmatch(source):
        return False

    category = str(category or "").strip().lower()
    entry_type = str(entry_type or "").strip().lower()
    if goal_rules is not None:
        if category in set(goal_rules.translatable_categories) or entry_type in set(goal_rules.translatable_categories):
            return True
    if category in _TRANSLATABLE_REVIEW_CATEGORIES or entry_type in _TRANSLATABLE_REVIEW_CATEGORIES:
        return True
    if (
        _looks_translatable_or_noise(
            source,
            category,
            lexical_policy=lexical_policy,
            goal_rules=goal_rules,
        )
        or _looks_translatable_or_noise(
            source,
            entry_type,
            lexical_policy=lexical_policy,
            goal_rules=goal_rules,
        )
    ):
        return True

    folded = source.casefold()
    words = folded.split()
    if len(words) >= 2 and any(
        word in lexical_policy.translatable_phrase_hints
        for word in words
    ):
        return True
    if len(words) == 1 and folded in lexical_policy.translatable_single_words:
        return True

    # If the source language is English and the target is a Spanish locale,
    # Title Case alone is not enough evidence of a proper name for technical
    # phrases such as "Residual Stream" or "Attention Mechanism".
    target_folded = str(target_locale or "").casefold()
    source_folded = str(source_language or "").casefold()
    if target_folded in {"spanish", "es", "es-mx", "es_mx", "es-es", "es_es"} and source_folded.startswith("english"):
        if len(words) >= 2 and not any(ch in source for ch in "ÁÉÍÓÚÜÑáéíóúüñ"):
            if any(
                word in lexical_policy.translatable_phrase_hints
                for word in words
            ):
                return True
    return False


def _demote_source_equals_target_review(item: Mapping[str, Any]) -> dict[str, Any]:
    confidence = min(_as_float(item.get("review_confidence"), default=0.0), 0.86)
    return {
        "review_status": "pending_review",
        "review_confidence": round(confidence, 3),
        "review_target": "",
        "injection_policy": "contextual",
        "translation_policy": "pending_review",
        "review_demoted_reason": (
            "source_equals_target_translatable: preserving this source term would likely "
            "force untranslated terminology into the target-language output."
        ),
        "review_rationale": (
            "The candidate appears translatable in the target language, so source==target "
            "must remain pending instead of being approved as preserve_exact."
        ),
    }


def _looks_named_entity(
    source: str,
    *,
    lexical_policy: LexicalPolicy,
    goal_rules: ProfileGoalRules | None = None,
) -> bool:
    words = source.split()
    if not words:
        return False
    if (
        _single_word_reject_noise(source, lexical_policy)
        or _source_equals_target_translatable_word(source, lexical_policy)
    ):
        return False
    if _looks_fragment_noise(source, lexical_policy):
        return False
    if len(words) >= 2 and all(
        word[:1].isupper() or word.casefold() in lexical_policy.connectors
        for word in words
    ):
        return not _looks_translatable_or_noise(
            source,
            "term",
            lexical_policy=lexical_policy,
            goal_rules=goal_rules,
        )
    # A long or accented word is not necessarily a name. Single-word entities
    # require internal capitalization (for example, iPhone/McDonald) or an LLM
    # canonical-name decision backed by context.
    return any(ch.isupper() for ch in source[1:])


def _has_strong_deterministic_entity_evidence(
    source: str,
    category: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    """Return evidence strong enough to bypass the contextual LLM reviewer."""
    if str(category or "").strip().lower() == "acronym":
        return _safe_acronym(source, lexical_policy)
    return bool(_SAFE_CODE_RE.fullmatch(source) or _SYMBOL_BEARING_ENTITY_RE.search(source))


def _capitalized_common_noun_language_risk(
    source: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    if not lexical_policy.common_noun_capitalization:
        return False
    words = _source_words(source)
    if not words or source.isupper():
        return False
    # German capitalizes every common noun, so neither a one-word candidate nor
    # a title-cased phrase is safe to preserve without contextual entity proof.
    # Internal capitals remain a useful identifier signal; acronyms/codes were
    # already handled before this gate.
    has_capitalized_word = any(word[:1].isupper() for word in re.findall(r"[^\W\d_]+", source))
    has_internal_capital = any(any(ch.isupper() for ch in word[1:]) for word in source.split())
    return has_capitalized_word and not has_internal_capital


def _safe_acronym(
    source: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    source = source.strip()
    if source.casefold() in lexical_policy.uppercase_word_noise:
        return False
    if len(source) > 8 and source.isalpha():
        return False
    return bool(_SAFE_ACRONYM_RE.fullmatch(source) or _SAFE_CODE_RE.fullmatch(source))


def _fold_source_token(value: str) -> str:
    return (
        str(value or "")
        .strip(" \t\r\n.,;:!?()[]{}<>\"'“”‘’")
        .replace("’", "'")
        .casefold()
    )


def _source_words(value: str) -> list[str]:
    return [
        _fold_source_token(word)
        for word in re.split(r"\s+", str(value or "").strip())
        if _fold_source_token(word)
    ]


def _active_lexical_policy(source_language: str) -> LexicalPolicy:
    """Resolve a scoped policy without applying English rules to explicit auto.

    Older direct helper calls did not carry a source language and historically
    operated on English test/input data, so only an actually empty value keeps
    that backwards-compatible default. Live profile preparation always passes
    the detected or user-selected source language.
    """
    raw = str(source_language or "").strip()
    return resolve_lexical_policy(
        raw,
        default_language="english" if not raw else "",
    )


def _single_word_reject_noise(
    source: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    words = _source_words(source)
    return (
        len(words) == 1
        and words[0] in lexical_policy.single_word_reject_noise
    )


def _source_equals_target_translatable_word(
    source: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    words = _source_words(source)
    if len(words) != 1:
        return False
    word = words[0]
    if word.endswith("'s"):
        word = word[:-2]
    return word in lexical_policy.source_equals_target_translatable_words


def _looks_fragment_noise(
    source: str,
    lexical_policy: LexicalPolicy,
) -> bool:
    words = _source_words(source)
    if len(words) < 2:
        return False
    if (
        words[0] in lexical_policy.leading_noise_words
        or words[0] in lexical_policy.single_word_reject_noise
    ):
        return True
    if words[-1] in lexical_policy.trailing_fragment_words:
        return True
    fragment_noise = (
        lexical_policy.fragment_noise_words
        | lexical_policy.single_word_reject_noise
    )
    return sum(1 for word in words if word in fragment_noise) >= 2


def _preserve_exact_allowed(
    source: str,
    entry_type: str,
    *,
    lexical_policy: LexicalPolicy,
    goal_rules: ProfileGoalRules | None = None,
) -> bool:
    entry_type = str(entry_type or "").lower()
    if entry_type == "acronym":
        return _safe_acronym(source, lexical_policy)
    if _looks_translatable_or_noise(
        source,
        entry_type,
        lexical_policy=lexical_policy,
        goal_rules=goal_rules,
    ):
        return False
    preserve_categories = set(goal_rules.preserve_categories) if goal_rules is not None else {
        "proper_noun",
        "character",
        "location",
        "organization",
        "title",
    }
    return entry_type in preserve_categories or _looks_named_entity(
        source,
        lexical_policy=lexical_policy,
        goal_rules=goal_rules,
    )


def _needs_canonical_review_before_preserve(
    source: str,
    category: str,
    *,
    transform_mode: str = "",
    target_locale: str = "",
) -> bool:
    mode = str(transform_mode or "").strip().lower()
    if mode not in {"modernize", "modernizar", "contemporize", "faithful_current_spanish"}:
        return False
    if str(target_locale or "").strip().lower() not in {"es-mx", "spanish", "es"}:
        return False
    if category not in {"character", "location", "organization", "proper_noun", "title"}:
        return False
    words = str(source or "").split()
    if len(words) != 1:
        return False
    if source.isupper():
        return False
    if any(ch in source for ch in "ÁÉÍÓÚÜÑáéíóúüñ"):
        return False
    return bool(_WORD_RE.search(source or ""))


def _entry_type_for_review(status: str, category: str) -> str:
    category = str(category or "").lower()
    if status == "canonical_name":
        return "canonical_proper_noun"
    if category == "acronym":
        return "acronym"
    if category in {"technical", "technical_term"}:
        return "technical_term"
    if category in {"concept"}:
        return "concept"
    if category in {"character", "location", "organization", "proper_noun", "title"}:
        return "proper_noun" if category != "title" else "title"
    return "term"


def _injection_policy_for_review(status: str) -> str:
    return {
        "preserve_exact": "preserve",
        "canonical_name": "translate_exact",
        "translate_exact": "translate_exact",
        "translate_contextual": "contextual",
        "pending_review": "contextual",
        "reject_noise": "do_not_inject",
    }.get(str(status or "").lower(), "contextual")


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
