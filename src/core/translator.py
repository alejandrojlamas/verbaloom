"""
Translation module for LLM communication
"""
import asyncio
import inspect
import json
import time
import re
from typing import Any, List, Dict, Tuple, Optional

from tqdm.auto import tqdm

from src.config import (
    DEFAULT_MODEL, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT, SENTENCE_TERMINATORS,
    THINKING_MODELS, ADAPTIVE_CONTEXT_INITIAL_THINKING, temperature_for_phase
)
from src.prompts.prompts import (
    build_text_transform_instructions,
    generate_translation_prompt,
    generate_subtitle_block_prompt,
    generate_refinement_prompt,
)
from src.prompts.examples import ensure_example_ready, has_example_for_pair, PLACEHOLDER_EXAMPLES
from .llm_client import default_client, LLMClient, create_llm_client, LLMResponse
from .llm import ContentRiskError, ContextOverflowError, RepetitionLoopError, RateLimitError
from .post_processor import clean_translated_text
from .context_optimizer import (
    AdaptiveContextManager,
    validate_configuration,
    INITIAL_CONTEXT_SIZE,
    CONTEXT_STEP
)
from .candidate_result import CandidateIssue, CandidateResult, record_candidate_result
from .combined_audit import (
    build_combined_quality_audit_prompt,
    combined_quality_audit_enabled,
    editorial_assessment_from_combined,
    fidelity_assessment_from_combined,
    parse_combined_quality_audit_response,
    profile_payload_from_combined,
)
from .progress_tracker import TokenProgressTracker
from .chunking.token_chunker import TokenChunker
from .editorial_quality import (
    QualityIssue,
    apply_source_aware_guard_assessment,
    assess_refinement,
    build_source_aware_guard_prompt,
    infer_section_title,
    parse_source_aware_guard_response,
    soften_decision_for_modernize,
)
from .fidelity_supervisor import (
    apply_fidelity_audit_assessment,
    assess_fidelity,
    fidelity_supervisor_enabled,
    supervise_fidelity,
    target_language_gate_issues,
)
from .literary_continuity import (
    build_literary_continuity_block,
    observe_literary_continuity,
)
from .llm.request_deadline import await_llm_call
from .llm_output_guard import guard_llm_output, merge_guard_issues
from .locale_quality import is_mexican_spanish_target
from .style_continuity import build_style_continuity_hint
from .source_invariant_repair import (
    repair_source_invariants,
    summarize_source_invariant_repairs,
)
from .text_transform import (
    MODERNIZE_FIDELITY_HARD_REJECT_CODES,
    MODERNIZE_HARD_REJECT_CODES,
    apply_faithful_modernize_defaults,
    block_repair_instructions,
    is_faithful_modernize,
    prompt_bool,
    protect_meaningful_blocks,
    restore_protected_blocks,
    transform_fallback_mode,
)
from .quality_guard import (
    _QUALITY_ALERT_MODEL_OFF_VALUES,
    _QUALITY_ALERT_MODEL_SAME_VALUES,
    _build_quality_alert_repair_instructions,
    _content_length_ratio_ok,
    _count_quality_alerts,
    _extract_source_text_for_guard,
    _format_quality_alerts,
    _normalize_guard_text,
    _quality_alert_guard_enabled,
    _resolve_quality_alert_model,
    _resolve_source_aware_editorial_guard_model,
    _should_run_source_aware_editorial_guard,
    _source_aware_editorial_guard_mode,
)
from .book_profiles import (
    apply_profile_glossary_corrections,
    build_editorial_knowledge_base,
    build_profile_glossary_block,
    build_profile_glossary_context,
    build_profile_report_summary,
    load_book_profile,
    profile_enabled,
)
from .book_profiles.audit import (
    CRITICAL_ISSUE_TYPES,
    CRITICAL_SCORE_FIELDS,
    ProfileAuditResult,
    SCORE_FIELDS,
    build_profile_audit_prompt,
    parse_profile_audit_response,
    profile_audit_result_from_payload,
    run_profile_precheck,
    score_repair_guidance,
    score_fields_for_profile,
)
from src.utils.text_encoding import clean_text_artifacts


def _call_supports_kwarg(fn, name: str) -> bool:
    """Return whether a callable accepts a keyword argument.

    Long-lived tests and some lightweight client adapters implement older
    ``generate``/``make_request`` signatures. Feature additions such as
    phase-specific temperature should degrade gracefully for those clients.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _target_language_gate_rejections(
    source_text: str,
    candidate_text: str,
    *,
    source_language: str = "",
    target_language: str = "",
    phase: str = "",
    prompt_options: Optional[dict] = None,
) -> list:
    return [
        issue for issue in target_language_gate_issues(
            source_text,
            candidate_text,
            source_language=source_language,
            target_language=target_language,
            phase=phase,
            prompt_options=prompt_options,
        )
        if issue.severity == "reject"
    ]


def _log_target_language_gate_rejection(
    log_callback,
    *,
    event: str,
    phase: str,
    issues: list,
) -> None:
    if not log_callback or not issues:
        return
    summary = "; ".join(
        f"{issue.code}: {issue.detail or issue.message}"
        for issue in issues[:3]
    )
    log_callback(
        event,
        f"⚠️ Chunk rechazado por gate de idioma/script en {phase}: {summary}",
    )


def _record_candidate(
    prompt_options: Optional[dict],
    result: CandidateResult,
    *,
    log_callback=None,
) -> None:
    record_candidate_result(prompt_options, result)
    if log_callback and (prompt_options or {}).get("candidate_result_logging"):
        log_callback(
            "candidate_result",
            f"Candidate {result.phase} chunk {result.chunk_index or '?'} -> {result.decision}",
            data={"type": "candidate_result", "candidate": result.to_dict(include_text=False)},
        )


def _record_text_candidate(
    prompt_options: Optional[dict],
    *,
    text: str,
    source_text: str,
    phase: str,
    chunk_index: int = 0,
    section: str = "",
    source_language: str = "",
    target_language: str = "",
    issues: list | None = None,
    extra_scores: Optional[dict] = None,
    response: Optional[LLMResponse] = None,
    decision: str = "",
    model: str = "",
    source: str = "candidate",
    log_callback=None,
) -> CandidateResult:
    issue_list = list(issues or [])
    editorial_issues, editorial_scores = _editorial_knowledge_contract_issues(
        source_text,
        text,
        prompt_options,
        phase=phase,
    )
    if editorial_issues:
        issue_list = merge_guard_issues(issue_list, editorial_issues)
    if editorial_scores:
        merged_scores = dict(extra_scores or {})
        merged_scores.update(editorial_scores)
        extra_scores = merged_scores
    result = CandidateResult.build(
        text,
        source_text=source_text,
        phase=phase,
        chunk_index=chunk_index,
        section=section,
        source_language=source_language,
        target_language=target_language,
        issues=issue_list,
        response=response,
        decision=decision,
        model=model,
        source=source,
        extra_scores=extra_scores,
    )
    _record_candidate(prompt_options, result, log_callback=log_callback)
    return result


def _editorial_knowledge_contract_issues(
    source_text: str,
    candidate_text: str,
    prompt_options: Optional[dict],
    *,
    phase: str,
) -> tuple[list[CandidateIssue], dict[str, float]]:
    """Validate profile/manual glossary expectations for CandidateResult.

    This is metadata-first: it records whether the candidate honored active
    editorial knowledge, but callers keep their existing accept/repair routing.
    That lets us tighten the contract without destabilizing long jobs.
    """
    if not prompt_options or not source_text or not candidate_text:
        return [], {}

    issues: list[CandidateIssue] = []
    scores: dict[str, float] = {}
    purpose = _profile_glossary_purpose(prompt_options, phase)

    profile_id = str(prompt_options.get("profile_id") or "").strip()
    if profile_enabled(prompt_options) and profile_id:
        try:
            profile = load_book_profile(profile_id)
            validation = build_editorial_knowledge_base(profile=profile).validate_candidate(
                source_text,
                candidate_text,
                purpose=purpose,
            )
            issues.extend(validation.issues)
            scores.update({
                "editorial_matched_terms": float(validation.matched_terms),
                "editorial_translated_terms": float(validation.translated_terms),
                "editorial_preserved_terms": float(validation.preserved_terms),
                "editorial_pending_terms": float(validation.pending_terms),
            })
        except Exception:
            # Candidate recording must never break a translation job.
            pass

    manual_terms = prompt_options.get("glossary_terms") or {}
    if isinstance(manual_terms, dict) and manual_terms:
        matched = translated = 0
        for source, target in manual_terms.items():
            source_term = str(source or "").strip()
            target_term = str(target or "").strip()
            if not source_term or not target_term:
                continue
            if not re.search(r"(?<!\w)" + re.escape(source_term) + r"(?!\w)", source_text, re.I):
                continue
            matched += 1
            if re.search(r"(?<!\w)" + re.escape(target_term) + r"(?!\w)", candidate_text, re.I):
                translated += 1
                continue
            issues.append(CandidateIssue(
                "manual_glossary_term_missing",
                "warning",
                "An active manual glossary term was not rendered with its target form.",
                detail=f"{source_term} -> {target_term}",
                source="editorial_knowledge",
            ))
        if matched:
            scores["manual_glossary_matched_terms"] = float(matched)
            scores["manual_glossary_rendered_terms"] = float(translated)

    return issues, scores


def _guard_generated_text(
    text: str,
    *,
    phase: str,
    style_reference: str = "",
    log_callback=None,
) -> tuple[str, list[CandidateIssue], dict[str, float]]:
    result = guard_llm_output(
        text or "",
        phase=phase,
        style_reference=style_reference,
    )
    if log_callback and result.issues:
        codes = ", ".join(issue.code for issue in result.issues[:4])
        log_callback(
            "llm_output_guard",
            f"⚠️ Output guard flagged {phase}: {codes}",
            data={
                "type": "llm_output_guard",
                "phase": phase,
                "issues": [issue.to_dict() for issue in result.issues],
                "scores": result.scores,
                "changed": result.changed,
            },
        )
    return result.text, result.issues, result.scores


def _profile_glossary_purpose(prompt_options: Optional[dict], phase: str = "translation") -> str:
    options = prompt_options or {}
    if options.get("text_transform_mode"):
        return "transformation"
    normalized = (phase or "translation").strip().lower()
    if normalized in {"transform", "transformation", "modernize", "same-language"}:
        return "transformation"
    if normalized in {
        "refine",
        "refinement",
        "translation_refinement",
        "profile_audit",
        "profile_repair",
        "repair",
        "audit",
    }:
        return "refinement"
    return "translation"


async def _client_generate(
    client,
    prompt: str,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
) -> LLMResponse | None:
    kwargs = {"system_prompt": system_prompt}
    if _call_supports_kwarg(client.generate, "temperature"):
        kwargs["temperature"] = temperature
    return await await_llm_call(
        client.generate,
        prompt,
        provider=client,
        **kwargs,
    )


async def _client_make_request(
    client,
    prompt: str,
    model: str | None = None,
    *,
    system_prompt: str | None = None,
    temperature: float | None = None,
    timeout: int | None = None,
) -> LLMResponse | None:
    kwargs = {"system_prompt": system_prompt}
    if timeout is not None and _call_supports_kwarg(client.make_request, "timeout"):
        kwargs["timeout"] = timeout
    if _call_supports_kwarg(client.make_request, "temperature"):
        kwargs["temperature"] = temperature
    return await await_llm_call(
        client.make_request,
        prompt,
        model,
        provider=client,
        request_timeout=timeout,
        **kwargs,
    )


# Configuration for context overflow recovery
MAX_CHUNK_REDUCTION_ATTEMPTS = 3
CHUNK_REDUCTION_FACTOR = 0.6  # Reduce to 60% of original size each attempt
MIN_CHUNK_CHARACTERS = 200  # Minimum chunk size to attempt translation
MAX_CONTENT_RISK_REDUCTION_ATTEMPTS = 8
MIN_CONTENT_RISK_CHARACTERS = 8

def _profile_repair_instructions(
    source_text: str,
    audit_result: ProfileAuditResult,
) -> str:
    issues = [issue.to_dict() for issue in audit_result.issues[:8]]
    return f"""
PROFILE-AWARE REPAIR REQUIRED

The active book profile audit found issues or scores below the required target.
Repair the candidate without adding, omitting, censoring, summarizing, or
changing scene/order/voice beyond the active profile's rules.

Use only the active profile and approved glossary. Do not apply any other book's
rules. Resolve these issues:
{json.dumps(issues, ensure_ascii=False)}

Current audit scores:
{json.dumps(audit_result.scores, ensure_ascii=False)}

Source text for fidelity comparison:
{source_text}
""".strip()


def _profile_audit_enabled(prompt_options: Optional[dict]) -> bool:
    options = prompt_options or {}
    return (
        profile_enabled(options)
        and options.get("profile_audit_enabled", True) is not False
    )


def _resolve_profile_audit_model(model: str, prompt_options: Optional[dict]) -> str:
    options = prompt_options or {}
    configured = (
        options.get("profile_audit_model")
        or options.get("profile_editorial_judge_model")
    )
    if configured is not None:
        value = str(configured).strip()
        normalized = value.lower()
        if normalized in _QUALITY_ALERT_MODEL_OFF_VALUES:
            return model
        if normalized in _QUALITY_ALERT_MODEL_SAME_VALUES:
            return model
        return value
    if "deepseek" in (model or "").lower():
        return "deepseek-v4-pro"
    return model


def _resolve_profile_repair_model(model: str, prompt_options: Optional[dict]) -> str:
    options = prompt_options or {}
    configured = (
        options.get("profile_repair_model")
        or options.get("profile_audit_model")
        or options.get("profile_editorial_judge_model")
    )
    if configured is not None:
        value = str(configured).strip()
        normalized = value.lower()
        if normalized in _QUALITY_ALERT_MODEL_OFF_VALUES:
            return model
        if normalized in _QUALITY_ALERT_MODEL_SAME_VALUES:
            return model
        return value
    return _resolve_profile_audit_model(model, options)


def _resolve_combined_quality_audit_model(model: str, prompt_options: Optional[dict]) -> str:
    options = prompt_options or {}
    configured = (
        options.get("combined_quality_audit_model")
        or options.get("source_aware_editorial_guard_model")
        or options.get("profile_audit_model")
        or options.get("fidelity_supervisor_model")
        or options.get("profile_editorial_judge_model")
    )
    if configured is not None:
        value = str(configured).strip()
        normalized = value.lower()
        if normalized in _QUALITY_ALERT_MODEL_OFF_VALUES:
            return model
        if normalized in _QUALITY_ALERT_MODEL_SAME_VALUES:
            return model
        return value
    return _resolve_source_aware_editorial_guard_model(model, options)


def _combined_audit_key(chunk_index: int, phase: str) -> str:
    return f"{phase or 'refinement'}:{chunk_index}"


def _combined_audit_records(prompt_options: Optional[dict]) -> dict:
    if prompt_options is None:
        return {}
    records = prompt_options.setdefault("_combined_quality_audit_records", {})
    if not isinstance(records, dict):
        records = {}
        prompt_options["_combined_quality_audit_records"] = records
    return records


def _store_combined_audit_record(
    prompt_options: Optional[dict],
    *,
    chunk_index: int,
    phase: str,
    payload: dict,
    model: str,
) -> None:
    if prompt_options is None:
        return
    _combined_audit_records(prompt_options)[_combined_audit_key(chunk_index, phase)] = {
        "payload": payload,
        "model": model,
    }


def _get_combined_audit_record(
    prompt_options: Optional[dict],
    *,
    chunk_index: int,
    phase: str,
) -> Optional[dict]:
    if prompt_options is None:
        return None
    record = _combined_audit_records(prompt_options).get(_combined_audit_key(chunk_index, phase))
    return record if isinstance(record, dict) else None


async def _run_combined_quality_audit(
    *,
    source_text: str,
    draft_text: str,
    candidate_text: str,
    chunk_index: int,
    section: str,
    source_language: str,
    target_language: str,
    phase: str,
    editorial_decision,
    prompt_options: dict,
    model: str,
    client,
    log_callback=None,
) -> tuple[Optional[dict], Optional[LLMResponse]]:
    """Run the one-call quality auditor and store its parsed payload."""

    if not source_text or not combined_quality_audit_enabled(prompt_options):
        return None, None

    fidelity_decision = assess_fidelity(
        source_text,
        candidate_text,
        chunk_index=chunk_index,
        phase=phase,
        section=section,
        source_language=source_language,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    profile_id = str((prompt_options or {}).get("profile_id") or "").strip()
    profile = None
    profile_result = None
    profile_policy = ""
    profile_audit_prompt = ""
    glossary_block = ""
    if profile_id and profile_enabled(prompt_options):
        try:
            profile = load_book_profile(profile_id)
            profile_policy = profile.policy_text or ""
            profile_audit_prompt = profile.prompt_texts.get("audit") or ""
            glossary_block = build_profile_glossary_block(
                source_text,
                prompt_options,
                purpose=_profile_glossary_purpose(prompt_options, "profile_audit"),
            )
            if (prompt_options or {}).get("profile_local_precheck_enabled") is False:
                profile_result = ProfileAuditResult(
                    profile_id=profile_id,
                    scores={field: 10.0 for field in SCORE_FIELDS},
                    overall_decision="pass",
                    issues=[],
                    summary="Profile local precheck disabled by profile_strength.",
                )
            else:
                profile_result = run_profile_precheck(
                    source_text,
                    candidate_text,
                    profile_id=profile_id,
                )
        except Exception:
            profile = None
            profile_result = None

    prompt_pair = build_combined_quality_audit_prompt(
        source_text=source_text,
        draft_text=draft_text,
        candidate_text=candidate_text,
        source_language=source_language,
        target_language=target_language,
        section=section,
        phase=phase,
        editorial_decision=editorial_decision,
        fidelity_decision=fidelity_decision,
        profile_result=profile_result,
        profile_id=profile_id,
        profile_policy=profile_policy,
        profile_audit_prompt=profile_audit_prompt,
        glossary_block=glossary_block,
    )
    audit_model = _resolve_combined_quality_audit_model(model, prompt_options)
    if log_callback:
        log_callback(
            "combined_quality_audit_request",
            f"🔎 Combined quality audit for chunk {chunk_index} with {audit_model}",
            data={
                "type": "combined_quality_audit_request",
                "system_prompt": prompt_pair.system,
                "user_prompt": prompt_pair.user,
                "model": audit_model,
                "primary_model": model,
                "phase": phase,
            },
        )
    response = await _generate_alert_repair(
        client,
        prompt_pair.user,
        prompt_pair.system,
        primary_model=model,
        alert_model=audit_model,
        phase="audit",
    )
    if not response:
        return None, None
    payload = parse_combined_quality_audit_response(response.content)
    if not payload:
        if log_callback:
            log_callback(
                "combined_quality_audit_parse_failed",
                f"⚠️ Combined quality audit returned invalid JSON for chunk {chunk_index}.",
            )
        return None, response
    _store_combined_audit_record(
        prompt_options,
        chunk_index=chunk_index,
        phase=phase,
        payload=payload,
        model=audit_model,
    )
    if log_callback:
        log_callback(
            "combined_quality_audit_decision",
            f"🔎 Combined quality audit for chunk {chunk_index}: {payload.get('worst', 'warn')}",
            data={
                "type": "combined_quality_audit_decision",
                "model": audit_model,
                "primary_model": model,
                "phase": phase,
                "payload": payload,
            },
        )
    return payload, response


async def _make_profile_repair_request(
    source_text: str,
    candidate_text: str,
    audit_result: ProfileAuditResult,
    *,
    profile_id: str,
    prompt_options: dict,
    target_language: str,
    model: str,
    client,
    log_callback=None,
    chunk_label: str = "",
) -> Tuple[Optional[str], Optional[LLMResponse]]:
    profile = load_book_profile(profile_id)
    repair_prompt = (profile.prompt_texts.get("repair") or "").strip()
    glossary_block = build_profile_glossary_block(
        source_text,
        prompt_options,
        purpose=_profile_glossary_purpose(prompt_options, "profile_repair"),
    )
    score_guidance = score_repair_guidance(
        audit_result.scores,
        min_score=profile.min_dimension_score,
    )
    system_prompt = f"""
You are a profile-specific literary repair editor.

{repair_prompt}

# OUTPUT FORMAT

Return only the repaired text between {TRANSLATE_TAG_IN} and {TRANSLATE_TAG_OUT}.
Do not include notes, explanations, markdown fences, or comments.
""".strip()
    user_prompt = f"""
# ACTIVE PROFILE
{profile.profile_id}

# PROFILE GLOSSARY
{glossary_block or '(No matching approved entries for this chunk.)'}

# SOURCE TEXT
{source_text}

# CURRENT CANDIDATE
{candidate_text}

# AUDIT RESULT TO FIX
{json.dumps(audit_result.to_dict(), ensure_ascii=False)}

# SCORE-BASED REPAIR PRIORITIES
{chr(10).join(f"- {item}" for item in score_guidance) or "- No score-specific priorities."}

# ISSUE-LOCAL REPAIR CONTRACT
Repair only the sentences, lines, table cells, or short spans needed to resolve
the audit issues. Keep every unaffected sentence unchanged unless a minimal
boundary edit is necessary for grammar after the local fix. Do not reroll the
whole chunk, change correct terminology, reorder paragraphs, or polish
already-good prose just because this is a repair pass.

Repair the candidate now for {target_language}. Preserve all content from the source while satisfying the active profile.
""".strip()
    repair_model = _resolve_profile_repair_model(model, prompt_options)
    if log_callback:
        label = f" {chunk_label}" if chunk_label else ""
        log_callback(
            "profile_repair_request",
            f"🛠️ Active profile repair{label} with {repair_model}",
            data={
                "type": "profile_repair_request",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "model": repair_model,
                "primary_model": model,
            },
        )
    response = await _generate_alert_repair(
        client,
        user_prompt,
        system_prompt,
        primary_model=model,
        alert_model=repair_model,
        phase="repair",
    )
    if not response:
        return None, None
    repaired = client.extract_translation(response.content)
    if repaired:
        return clean_text_artifacts(repaired), response
    fallback = (response.content or "").strip()
    if fallback and candidate_text not in fallback:
        return clean_text_artifacts(fallback), response
    return None, response


def _profile_audit_quality_key(audit_result: ProfileAuditResult) -> tuple:
    scores = list(audit_result.scores.values())
    min_score = min(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
    decision_rank = {"fail": 0, "warn": 1, "pass": 2}.get(
        audit_result.overall_decision,
        0,
    )
    high_issues = sum(1 for issue in audit_result.issues if issue.severity == "high")
    medium_issues = sum(1 for issue in audit_result.issues if issue.severity == "medium")
    return (
        decision_rank,
        -high_issues,
        min_score,
        avg_score,
        -medium_issues,
        -len(audit_result.issues),
    )


def _profile_audit_requires_source_fallback(audit_result: ProfileAuditResult) -> bool:
    """Return True only for profile failures that make the candidate unsafe.

    Style/profile misses should drive repair and reporting. Falling back to the
    unmodernized source for those misses is exactly the failure mode that made
    Quijote modernization look unchanged.
    """
    if any(
        issue.severity == "high" and issue.issue_type in CRITICAL_ISSUE_TYPES
        for issue in audit_result.issues
    ):
        return True
    return any(
        float(audit_result.scores.get(field, 10.0)) < 7.0
        for field in CRITICAL_SCORE_FIELDS
    )


def _profile_audit_failure_should_abort(
    audit_result: ProfileAuditResult,
    prompt_options: Optional[dict],
) -> bool:
    if not profile_enabled(prompt_options):
        return False
    if not prompt_bool(prompt_options, "abort_on_profile_fail", False):
        return False
    if audit_result.overall_decision != "fail":
        return False
    # Only fidelity-critical failures should stop a long book job. Style and
    # modernization misses are repaired/reported and the best audited candidate
    # continues; otherwise one imperfect chunk can waste hours of progress.
    return _profile_audit_requires_source_fallback(audit_result)


def _quality_decision_requires_source_fallback(
    decision,
    prompt_options: Optional[dict],
) -> bool:
    """Decide whether a local/editorial rejection should revert the chunk."""
    if not is_faithful_modernize(prompt_options):
        return True
    if transform_fallback_mode(prompt_options) == "source":
        return True
    return any(
        issue.code in MODERNIZE_HARD_REJECT_CODES
        for issue in decision.rejections
    )


def _fidelity_decision_requires_source_fallback(
    decision,
    prompt_options: Optional[dict],
) -> bool:
    """Decide whether a fidelity rejection should revert the chunk.

    In book-profile modernization, the fidelity supervisor is a safety auditor,
    not the final style gate. Non-corruption findings such as pagination,
    dot-leader cleanup, length shifts from OCR cleanup, or a judge disagreement
    should be repaired and audited by the active profile instead of restoring
    the archaic/source text.
    """
    if not is_faithful_modernize(prompt_options):
        return True
    if transform_fallback_mode(prompt_options) == "source":
        return True
    if not profile_enabled(prompt_options):
        return True
    return any(
        issue.code in MODERNIZE_FIDELITY_HARD_REJECT_CODES
        for issue in decision.rejections
    )


async def _run_profile_audit(
    source_text: str,
    candidate_text: str,
    *,
    profile_id: str,
    prompt_options: dict,
    model: str,
    client,
    log_callback=None,
    chunk_label: str = "",
) -> Tuple[ProfileAuditResult, Optional[LLMResponse]]:
    if (prompt_options or {}).get("profile_local_precheck_enabled") is False:
        local_result = ProfileAuditResult(
            profile_id=profile_id,
            scores={field: 10.0 for field in SCORE_FIELDS},
            overall_decision="pass",
            issues=[],
            summary="Profile local precheck disabled by profile_strength.",
        )
    else:
        local_result = run_profile_precheck(
            source_text,
            candidate_text,
            profile_id=profile_id,
        )
    if not _profile_audit_enabled(prompt_options):
        return local_result, None

    try:
        profile = load_book_profile(profile_id)
    except Exception:
        return local_result, None

    glossary_block = build_profile_glossary_block(
        source_text,
        prompt_options,
        purpose=_profile_glossary_purpose(prompt_options, "profile_audit"),
    )
    audit_system, audit_user = build_profile_audit_prompt(
        source_text,
        candidate_text,
        profile=profile,
        glossary_block=glossary_block,
        local_issues=local_result.issues,
    )
    audit_model = _resolve_profile_audit_model(model, prompt_options)
    if log_callback:
        label = f" {chunk_label}" if chunk_label else ""
        log_callback(
            "profile_audit_request",
            f"🔎 Active profile audit{label} with {audit_model}",
            data={
                "type": "profile_audit_request",
                "system_prompt": audit_system,
                "user_prompt": audit_user,
                "model": audit_model,
                "primary_model": model,
            },
        )

    audit_response = await _generate_alert_repair(
        client,
        audit_user,
        audit_system,
        primary_model=model,
        alert_model=audit_model,
        phase="audit",
    )
    if not audit_response:
        return local_result, None

    parsed = parse_profile_audit_response(audit_response.content)
    if not parsed:
        if log_callback:
            label = f" {chunk_label}" if chunk_label else ""
            log_callback(
                "profile_audit_parse_failed",
                f"⚠️ Active profile audit{label} returned invalid JSON; using local precheck only."
            )
        return local_result, audit_response

    result = profile_audit_result_from_payload(
        profile_id,
        parsed,
        local_issues=local_result.issues,
        min_score=profile.min_dimension_score,
        score_fields=score_fields_for_profile(profile),
    )
    if log_callback:
        low_scores = {
            key: value for key, value in result.scores.items()
            if value < profile.min_dimension_score
        }
        label = f" {chunk_label}" if chunk_label else ""
        log_callback(
            "profile_audit_decision",
            f"🔎 Active profile audit{label}: {result.overall_decision}"
            + (f" low={low_scores}" if low_scores else ""),
            data={
                "type": "profile_audit_decision",
                "model": audit_model,
                "primary_model": model,
                "decision": result.overall_decision,
                "scores": result.scores,
                "issues": [issue.to_dict() for issue in result.issues],
            },
        )
    return result, audit_response


def _append_profile_audit_to_report(
    report,
    *,
    raw_draft_text: str,
    refined_text: str,
    source_text: str,
    audit_result: ProfileAuditResult,
    chunk_index: int,
    section: str,
    target_language: str,
    prompt_options: dict,
) -> None:
    if report is None or not audit_result.issues:
        return
    decision = assess_refinement(
        raw_draft_text,
        refined_text,
        chunk_index=chunk_index,
        section=section,
        source_text=source_text,
        target_language=target_language,
        source_language=prompt_options.get('_source_language', ''),
        prompt_options=prompt_options,
    )
    if _profile_audit_requires_source_fallback(audit_result):
        decision.accepted = False
    for issue in audit_result.issues:
        severity = (
            "reject"
            if issue.severity == "high" and issue.issue_type in CRITICAL_ISSUE_TYPES
            else "warning"
        )
        detail = issue.suggested_fix or issue.candidate_excerpt or issue.source_excerpt
        decision.issues.append(QualityIssue(
            f"profile_{issue.issue_type}",
            severity,
            issue.reason or issue.issue_type,
            detail,
        ))
    report.add(decision)


def _build_chunk_glossary_block(
    chunk_content: str,
    prompt_options: Optional[dict],
    log_callback=None,
    runtime_state: Optional[dict] = None,
    phase: str = "translation",
) -> str:
    """
    Filter the active glossary against the current chunk and render a prompt block.

    Reads `glossary_terms` (dict source -> target) and optional `glossary_config`
    (GlossaryConfig) from prompt_options. Returns "" when no glossary is active
    or no terms match this chunk.

    In refinement and same-language transformation phases, a term is also
    included when the draft already contains the right-hand form. This keeps
    established glossary renderings visible to the model after the first pass
    has replaced the source wording.

    When the per-chunk cap is hit and `warn_on_cap` is enabled, logs a single
    warning per job. The dedupe flag lives in `runtime_state` (a transient dict
    owned by the caller) so it never leaks into the persisted prompt_options
    snapshot. If runtime_state is None, a fresh local dict is used (warning
    won't be deduped across calls — fine for ad-hoc uses).
    """
    if not prompt_options:
        return ""

    def _normalize_glossary_purpose() -> str:
        return _profile_glossary_purpose(prompt_options, phase)

    def _glossary_config_for_purpose(glossary_config_cls, purpose: str):
        explicit_config = prompt_options.get("glossary_config")
        if explicit_config is not None:
            if isinstance(explicit_config, dict):
                allowed = {
                    "max_entries",
                    "case_sensitive",
                    "accent_insensitive",
                    "warn_on_cap",
                }
                return glossary_config_cls(
                    **{
                        key: value
                        for key, value in explicit_config.items()
                        if key in allowed
                    }
                )
            return explicit_config
        if purpose in {"refinement", "transformation"}:
            return glossary_config_cls(
                case_sensitive=False,
                accent_insensitive=True,
            )
        return glossary_config_cls()

    if runtime_state is None:
        runtime_state = {}
    visibility = runtime_state.setdefault("prompt_context_visibility", {
        "chunks_seen": 0,
        "manual_chunks_with_matches": 0,
        "profile_chunks_with_matches": 0,
        "manual_matched_terms": 0,
        "profile_matched_terms": 0,
        "profile_rendered_terms": 0,
        "logged_chunks": 0,
    })
    visibility["chunks_seen"] = int(visibility.get("chunks_seen") or 0) + 1

    blocks: list[str] = []
    manual_summary = None
    terms = prompt_options.get("glossary_terms")
    if terms:
        try:
            from src.core.glossary import filter_glossary_for_purpose, build_glossary_block, GlossaryConfig
        except ImportError:
            pass
        else:
            glossary_purpose = _normalize_glossary_purpose()
            config = _glossary_config_for_purpose(GlossaryConfig, glossary_purpose)
            filtered, capped = filter_glossary_for_purpose(
                chunk_content,
                terms,
                config,
                glossary_purpose,
            )

            if capped and config.warn_on_cap and not runtime_state.get("glossary_cap_warned"):
                runtime_state["glossary_cap_warned"] = True
                if log_callback:
                    log_callback(
                        "glossary_capped",
                        f"⚠️ Glossary cap reached: more than {config.max_entries} terms matched in a single chunk. "
                        f"Excess entries are dropped — increase `max_entries` if you need full coverage."
                    )

            if filtered:
                manual_summary = {
                    "matched_terms": len(filtered),
                    "total_terms": len(terms),
                    "capped": bool(capped),
                    "purpose": glossary_purpose,
                }
                visibility["manual_chunks_with_matches"] = int(visibility.get("manual_chunks_with_matches") or 0) + 1
                visibility["manual_matched_terms"] = int(visibility.get("manual_matched_terms") or 0) + len(filtered)
                metadata = prompt_options.get("glossary_term_metadata") or None
                blocks.append(
                    build_glossary_block(
                        filtered,
                        term_metadata=metadata,
                        purpose=glossary_purpose,
                    )
                )

    profile_purpose = _normalize_glossary_purpose()
    profile_block, profile_summary = build_profile_glossary_context(
        chunk_content,
        prompt_options,
        purpose=profile_purpose,
    )
    if profile_block:
        if profile_summary and profile_summary.get("matched_terms"):
            visibility["profile_chunks_with_matches"] = int(visibility.get("profile_chunks_with_matches") or 0) + 1
            visibility["profile_matched_terms"] = (
                int(visibility.get("profile_matched_terms") or 0)
                + int(profile_summary.get("matched_terms") or 0)
            )
            visibility["profile_rendered_terms"] = (
                int(visibility.get("profile_rendered_terms") or 0)
                + int(profile_summary.get("rendered_terms") or 0)
            )
        blocks.append(profile_block)

    if log_callback and (manual_summary or (profile_summary and profile_summary.get("matched_terms"))):
        logged = int(visibility.get("logged_chunks") or 0)
        chunk_no = int(visibility.get("chunks_seen") or 0)
        if logged < 8 or chunk_no % 50 == 0:
            visibility["logged_chunks"] = logged + 1
            parts = []
            if profile_summary and profile_summary.get("matched_terms"):
                parts.append(
                    "profile "
                    f"{profile_summary.get('matched_terms', 0)}/{profile_summary.get('total_terms', 0)} "
                    f"(rendered {profile_summary.get('rendered_terms', 0)})"
                )
            if manual_summary:
                parts.append(
                    "manual "
                    f"{manual_summary.get('matched_terms', 0)}/{manual_summary.get('total_terms', 0)}"
                )
            message = f"🧩 Chunk {chunk_no}: injected glossary context: {', '.join(parts)}."
            payload = {
                "type": "prompt_context",
                "chunk_sequence": chunk_no,
                "profile_glossary": profile_summary or {},
                "manual_glossary": manual_summary or {},
            }
            try:
                log_callback("chunk_prompt_context", message, data=payload)
            except TypeError:
                log_callback("chunk_prompt_context", message)

    return "\n\n".join(block.strip() for block in blocks if block and block.strip())


def split_chunk_for_retry(main_content: str, target_ratio: float = 0.5) -> Tuple[str, str]:
    """
    Split a chunk into two parts for retry after context overflow.

    Tries to split at a sentence boundary near the target ratio.

    Args:
        main_content: The text content to split
        target_ratio: Target position for split (0.5 = middle)

    Returns:
        Tuple of (first_half, second_half)
    """
    if not main_content.strip():
        return main_content, ""

    text = main_content
    target_pos = max(1, min(len(text) - 1, int(len(text) * target_ratio)))
    min_pos = max(1, int(len(text) * 0.12))
    max_pos = min(len(text) - 1, int(len(text) * 0.88))
    window = max(240, int(len(text) * 0.28))
    search_start = max(min_pos, target_pos - window)
    search_end = min(max_pos, target_pos + window)

    candidates: list[tuple[int, int]] = []

    def add_candidate(pos: int, priority: int) -> None:
        if search_start <= pos <= search_end:
            candidates.append((pos, priority))

    # Prefer paragraph boundaries, then sentence boundaries, then clean line
    # boundaries. Whitespace is only a final fallback. Scan only the useful
    # window instead of using regex over the full text; retry splitting can run
    # on malformed PDF/OCR chunks and must not monopolize the GIL.
    i = search_start
    while i < search_end:
        char = text[i]
        if char == "\n":
            j = i + 1
            while j < search_end and text[j] in " \t\r":
                j += 1
            if j < search_end and text[j] == "\n":
                while j < search_end and text[j].isspace():
                    j += 1
                add_candidate(j, 0)
            while j < search_end and text[j] == "\n":
                j += 1
            add_candidate(j, 2)
            i = max(j, i + 1)
            continue

        if char in ".!?;:…":
            if char == ".":
                prev_char = text[i - 1] if i > 0 else ""
                next_char = text[i + 1] if i + 1 < len(text) else ""
                if prev_char.isdigit() and next_char.isdigit():
                    i += 1
                    continue
            j = i + 1
            while j < search_end and text[j] in "\"'”’)]}":
                j += 1
            if j < search_end and text[j].isspace():
                while j < search_end and text[j].isspace():
                    j += 1
                add_candidate(j, 1)
                i = max(j, i + 1)
                continue

        if char.isspace():
            j = i + 1
            while j < search_end and text[j].isspace():
                j += 1
            add_candidate(j, 3)
            i = j
            continue

        i += 1

    if candidates:
        best_priority = min(priority for _pos, priority in candidates)
        best_split, _priority = min(
            (item for item in candidates if item[1] == best_priority),
            key=lambda item: abs(item[0] - target_pos),
        )
    else:
        best_split = target_pos

    return text[:best_split].strip(), text[best_split:].strip()


async def _assess_refinement_with_editorial_guard(
    draft_text: str,
    refined_text: str,
    *,
    chunk_index: int,
    section: str,
    source_text: str = "",
    source_language: str = "",
    target_language: str,
    model: str,
    client,
    log_callback=None,
    prompt_options: Optional[dict] = None,
) -> Tuple[object, Optional[LLMResponse]]:
    """Run local guard plus optional source-aware LLM assessment."""
    decision = assess_refinement(
        draft_text,
        refined_text,
        chunk_index=chunk_index,
        section=section,
        source_text=source_text,
        glossary_terms=(prompt_options or {}).get("glossary_terms"),
        target_language=target_language,
        source_language=source_language,
        prompt_options=prompt_options,
    )

    if not _should_run_source_aware_editorial_guard(
        decision,
        source_text=source_text,
        prompt_options=prompt_options,
    ):
        _record_candidate(
            prompt_options,
            CandidateResult.from_quality_decision(
                decision,
                text=refined_text,
                source_text=source_text or draft_text,
                source_language=source_language,
                target_language=target_language,
            ),
            log_callback=log_callback,
        )
        return decision, None

    guard_model = _resolve_source_aware_editorial_guard_model(model, prompt_options)
    if combined_quality_audit_enabled(prompt_options):
        combined_payload, combined_response = await _run_combined_quality_audit(
            source_text=source_text,
            draft_text=draft_text,
            candidate_text=refined_text,
            chunk_index=chunk_index,
            section=section,
            source_language=source_language,
            target_language=target_language,
            phase="refinement",
            editorial_decision=decision,
            prompt_options=prompt_options or {},
            model=model,
            client=client,
            log_callback=log_callback,
        )
        if combined_payload:
            assessment = editorial_assessment_from_combined(combined_payload)
            if assessment:
                combined_model = str(
                    (
                        _get_combined_audit_record(
                            prompt_options,
                            chunk_index=chunk_index,
                            phase="refinement",
                        )
                        or {}
                    ).get("model")
                    or guard_model
                )
                decision = apply_source_aware_guard_assessment(
                    decision,
                    assessment,
                    model=combined_model,
                )
                if log_callback:
                    log_callback(
                        "source_aware_editorial_guard_decision",
                        "🔎 Combined source-aware editorial guard: "
                        f"chunk {chunk_index} -> {decision.judge_decision} "
                        f"({decision.judge_confidence:.2f})",
                        data={
                            "type": "source_aware_editorial_guard_decision",
                            "model": combined_model,
                            "primary_model": model,
                            "decision": decision.judge_decision,
                            "confidence": decision.judge_confidence,
                            "accepted": decision.accepted,
                            "reason": decision.judge_reason,
                            "combined": True,
                        },
                    )
                _record_candidate(
                    prompt_options,
                    CandidateResult.from_quality_decision(
                        decision,
                        text=refined_text,
                        source_text=source_text or draft_text,
                        source_language=source_language,
                        target_language=target_language,
                        response=combined_response,
                    ),
                    log_callback=log_callback,
                )
                return decision, combined_response

    prompt_pair = build_source_aware_guard_prompt(
        source_text=source_text,
        draft_text=draft_text,
        refined_text=refined_text,
        source_language=source_language,
        target_language=target_language,
        section=section,
        local_decision=decision,
    )

    if log_callback:
        log_callback(
            "source_aware_editorial_guard_request",
            f"🔎 Source-aware editorial guard for chunk {chunk_index} with {guard_model}",
            data={
                "type": "source_aware_editorial_guard_request",
                "system_prompt": prompt_pair.system,
                "user_prompt": prompt_pair.user,
                "model": guard_model,
                "primary_model": model,
            },
        )

    guard_response = await _generate_alert_repair(
        client,
        prompt_pair.user,
        prompt_pair.system,
        primary_model=model,
        alert_model=guard_model,
        phase="audit",
    )
    if not guard_response:
        _record_candidate(
            prompt_options,
            CandidateResult.from_quality_decision(
                decision,
                text=refined_text,
                source_text=source_text or draft_text,
                source_language=source_language,
                target_language=target_language,
            ),
            log_callback=log_callback,
        )
        return decision, None

    assessment = parse_source_aware_guard_response(guard_response.content)
    if not assessment:
        if log_callback:
            log_callback(
                "source_aware_editorial_guard_parse_failed",
                f"⚠️ Source-aware editorial guard returned invalid JSON for chunk {chunk_index}; "
                "falling back to local guard.",
            )
        _record_candidate(
            prompt_options,
            CandidateResult.from_quality_decision(
                decision,
                text=refined_text,
                source_text=source_text or draft_text,
                source_language=source_language,
                target_language=target_language,
                response=guard_response,
            ),
            log_callback=log_callback,
        )
        return decision, guard_response

    decision = apply_source_aware_guard_assessment(
        decision,
        assessment,
        model=guard_model,
    )

    if log_callback:
        log_callback(
            "source_aware_editorial_guard_decision",
            "🔎 Source-aware editorial guard: "
            f"chunk {chunk_index} -> {decision.judge_decision} "
            f"({decision.judge_confidence:.2f})",
            data={
                "type": "source_aware_editorial_guard_decision",
                "model": guard_model,
                "primary_model": model,
                "decision": decision.judge_decision,
                "confidence": decision.judge_confidence,
                "accepted": decision.accepted,
                "reason": decision.judge_reason,
            },
        )

    _record_candidate(
        prompt_options,
        CandidateResult.from_quality_decision(
            decision,
            text=refined_text,
            source_text=source_text or draft_text,
            source_language=source_language,
            target_language=target_language,
            response=guard_response,
        ),
        log_callback=log_callback,
    )
    return decision, guard_response


async def _generate_alert_repair(
    client,
    prompt: str,
    system_prompt: str,
    *,
    primary_model: str,
    alert_model: str,
    phase: str = "repair",
) -> Optional[LLMResponse]:
    """Generate a repair with a temporary alert model, then restore the primary model."""
    temperature = temperature_for_phase(phase)
    if not alert_model or alert_model == primary_model or not hasattr(client, "make_request"):
        return await _client_generate(
            client,
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

    provider = None
    original_model = None
    try:
        if hasattr(client, "_get_provider"):
            provider = client._get_provider()
            original_model = getattr(provider, "model", None)
        return await _client_make_request(
            client,
            prompt,
            alert_model,
            system_prompt=system_prompt,
            temperature=temperature,
        )
    finally:
        if provider is not None and original_model:
            provider.model = original_model


def _merge_llm_usage(primary: Optional[LLMResponse], extra: Optional[LLMResponse]) -> Optional[LLMResponse]:
    if not extra:
        return primary
    if not primary:
        return extra
    primary.prompt_tokens += extra.prompt_tokens
    primary.completion_tokens += extra.completion_tokens
    primary.prompt_cache_hit_tokens += getattr(extra, "prompt_cache_hit_tokens", 0) or 0
    primary.prompt_cache_miss_tokens += getattr(extra, "prompt_cache_miss_tokens", 0) or 0
    primary.total_tokens += getattr(extra, "total_tokens", 0) or 0
    primary.reasoning_tokens += getattr(extra, "reasoning_tokens", 0) or 0
    primary.context_used += extra.context_used
    primary.context_limit = max(primary.context_limit, extra.context_limit)
    primary.was_truncated = primary.was_truncated or extra.was_truncated
    return primary


async def _stitch_content_risk_fragments(
    fragments: List[str],
    *,
    target_language: str,
    client: Any,
    log_callback: Optional[Any],
    prompt_options: Optional[dict],
) -> Tuple[str, Optional[LLMResponse]]:
    """Join target-language microfragments without resending filtered source.

    Content-policy recovery can require clause- or word-sized requests. Their
    translations are complete but may have artificial seam grammar. This pass
    receives only the already translated target text, and is accepted only
    when the deterministic editorial guard confirms that structural markers,
    numbers, references, and content volume remain intact.
    """
    combined = "\n".join(part for part in fragments if part)
    if len(fragments) < 2 or not combined.strip():
        return combined, None

    system_prompt = (
        f"You are a seam editor for {target_language}. The passage below was translated "
        "in adjacent microfragments. Join it into continuous, natural target-language "
        "prose. Preserve every fact, name, number, quotation, sentence, and placeholder; "
        "do not summarize, omit, add, censor, explain, or change meaning. Change only "
        "grammar, word order, punctuation, and artificial line breaks caused by the seams. "
        f"Return only the complete passage inside {TRANSLATE_TAG_IN} and {TRANSLATE_TAG_OUT}."
    )
    user_prompt = (
        "# ADJACENT TARGET-LANGUAGE FRAGMENTS\n"
        f"{TRANSLATE_TAG_IN}\n{combined}\n{TRANSLATE_TAG_OUT}"
    )
    try:
        response = await _client_generate(
            client,
            user_prompt,
            system_prompt=system_prompt,
            temperature=temperature_for_phase("refinement"),
        )
    except ContentRiskError:
        if log_callback:
            log_callback(
                "content_risk_stitch_filtered",
                "Provider also filtered the target-only seam pass; preserving every "
                "translated microfragment unchanged.",
            )
        return combined, None

    if not response:
        return combined, None
    stitched = client.extract_translation(response.content)
    if not stitched:
        if log_callback:
            log_callback(
                "content_risk_stitch_extraction_failed",
                "Target-only seam pass returned no valid translation wrapper; preserving "
                "the complete microfragment sequence.",
            )
        return combined, response

    decision = assess_refinement(
        combined,
        stitched,
        chunk_index=0,
        section="content_filter_seam",
        source_text=combined,
        target_language=target_language,
        source_language=target_language,
        prompt_options=prompt_options,
    )
    if not decision.accepted:
        if log_callback:
            rejected = ", ".join(issue.code for issue in decision.issues if issue.severity == "reject")
            log_callback(
                "content_risk_stitch_rejected",
                "Target-only seam pass changed protected content; preserving the complete "
                f"microfragment sequence ({rejected or 'deterministic guard'}).",
            )
        return combined, response

    if log_callback:
        log_callback(
            "content_risk_stitch_accepted",
            "Reassembled provider-filtered microfragments into continuous target-language prose.",
        )
    return stitched, response


async def _maybe_repair_quality_alerts(
    text: str,
    *,
    target_language: str,
    model: str,
    client,
    log_callback=None,
    prompt_options: Optional[dict] = None,
    has_placeholders: bool = False,
    placeholder_format: Optional[Tuple[str, str]] = None,
    phase: str = "translation",
) -> Tuple[str, Optional[LLMResponse]]:
    """Run one cheap alert-repair pass only when local quality alerts are found."""
    if not text or not _quality_alert_guard_enabled(prompt_options):
        return text, None

    issue_counts = _count_quality_alerts(
        text,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    original_issue_total = sum(issue_counts.values())
    if original_issue_total == 0:
        return text, None

    repair_options = dict(prompt_options or {})
    if is_mexican_spanish_target(target_language, prompt_options):
        repair_options["spanish_variant"] = "mexican"
    alert_model = _resolve_quality_alert_model(model, repair_options)
    repair_prompt = generate_refinement_prompt(
        draft_translation=text,
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=repair_options,
        placeholder_format=placeholder_format,
        additional_instructions=_build_quality_alert_repair_instructions(
            issue_counts,
            target_language=target_language,
        ),
    )

    if log_callback:
        log_callback(
            "quality_alert_repair_request",
            f"🔎 Quality alert repair for {phase} with {alert_model}: {_format_quality_alerts(issue_counts)}",
            data={
                "type": "quality_alert_repair_request",
                "system_prompt": repair_prompt.system,
                "user_prompt": repair_prompt.user,
                "model": alert_model,
                "primary_model": model,
                "issues": issue_counts,
            },
        )

    repair_response = await _generate_alert_repair(
        client,
        repair_prompt.user,
        repair_prompt.system,
        primary_model=model,
        alert_model=alert_model,
        phase="repair",
    )
    if not repair_response:
        return text, None

    repaired = client.extract_translation(repair_response.content)
    if not repaired:
        if log_callback:
            log_callback(
                "quality_alert_repair_failed",
                "⚠️ Quality alert repair did not return valid translation tags; keeping original output.",
            )
        return text, repair_response

    repaired, repair_guard_issues, _repair_guard_scores = _guard_generated_text(
        clean_text_artifacts(repaired),
        phase=f"{phase}_quality_alert_repair",
        style_reference=text,
        log_callback=log_callback,
    )
    if any(issue.severity == "reject" for issue in repair_guard_issues):
        if log_callback:
            reason = "; ".join(issue.code for issue in repair_guard_issues if issue.severity == "reject")
            log_callback(
                "quality_alert_repair_guard_rejected",
                "⚠️ Quality alert repair still contains reader-visible prompt protocol; keeping original output."
                + (f" ({reason})" if reason else ""),
            )
        return text, repair_response
    repaired = apply_profile_glossary_corrections(
        repaired,
        prompt_options,
        source_text=text,
    )
    repaired_counts = _count_quality_alerts(
        repaired,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    repaired_issue_total = sum(repaired_counts.values())
    improved = repaired_issue_total < original_issue_total
    quality_decision = assess_refinement(
        text,
        repaired,
        chunk_index=0,
        section=f"{phase} alert repair",
        glossary_terms=(prompt_options or {}).get("glossary_terms"),
        target_language=target_language,
        prompt_options=prompt_options,
    )

    if improved and _content_length_ratio_ok(text, repaired) and quality_decision.accepted:
        if log_callback:
            log_callback(
                "quality_alert_repair_accepted",
                "🔎 Quality alert repair accepted: "
                f"{_format_quality_alerts(issue_counts)} -> "
                f"{_format_quality_alerts(repaired_counts)}",
                data={
                    "type": "quality_alert_repair_accepted",
                    "model": alert_model,
                    "primary_model": model,
                    "before": issue_counts,
                    "after": repaired_counts,
                },
            )
        return repaired, repair_response

    if log_callback:
        rejection_reason = "; ".join(issue.code for issue in quality_decision.rejections)
        if not rejection_reason and not improved:
            rejection_reason = "alerts_not_reduced"
        elif not rejection_reason and not _content_length_ratio_ok(text, repaired):
            rejection_reason = "length_ratio"
        log_callback(
            "quality_alert_repair_rejected",
            "⚠️ Quality alert repair did not safely improve the output; keeping original. "
            f"{_format_quality_alerts(issue_counts)} -> "
            f"{_format_quality_alerts(repaired_counts)}"
            + (f" ({rejection_reason})" if rejection_reason else ""),
            data={
                "type": "quality_alert_repair_rejected",
                "model": alert_model,
                "primary_model": model,
                "before": issue_counts,
                "after": repaired_counts,
                "rejection_reason": rejection_reason,
            },
        )
    return text, repair_response


_ALL_CAPS_PROSE_RUN_RE = re.compile(
    r"[A-ZÀ-Þ][A-ZÀ-Þ0-9\s,.;:!?…'’\-]+",
    flags=re.UNICODE,
)


def _normalize_all_caps_prose_for_translation(text: str) -> str:
    """Sentence-case long all-caps prose while leaving short labels untouched."""
    def normalize_run(match: re.Match) -> str:
        run = match.group(0)
        letters = [char for char in run if char.isalpha()]
        words = re.findall(r"[^\W\d_]+", run, flags=re.UNICODE)
        if (
            len(letters) < 24
            or len(words) < 5
            or sum(char.isupper() for char in letters) / max(1, len(letters)) < 0.9
        ):
            return run
        normalized = run.lower()
        for index, char in enumerate(normalized):
            if char.isalpha():
                return normalized[:index] + char.upper() + normalized[index + 1:]
        return run

    return _ALL_CAPS_PROSE_RUN_RE.sub(normalize_run, text or "")


_RESIDUAL_DETAIL_SPAN_RE = re.compile(
    r"(?:^|;\s*)(.+?)\s+\((?:0(?:\.\d+)?|1(?:\.0+)?)\)(?=;\s*|$)"
)
_QUOTE_OPENERS = {'"', "'", "“", "‘", "«", "‹"}
_QUOTE_CLOSERS = {'"', "'", "”", "’", "»", "›"}


def _quoted_residual_spans(source_text: str, candidate_text: str, issues: list) -> list[str]:
    """Extract actionable copied quotations from structured gate details.

    Full-candidate repair is intentionally conservative, but it can lose an
    unrelated number or placeholder while changing one quoted phrase. A copied
    multiword quotation is safe to repair as a bounded span because punctuation
    remains outside the replacement and the complete candidate is audited again.
    """
    spans: list[str] = []
    for issue in issues or []:
        if str(getattr(issue, "code", "")) != "source_language_residual":
            continue
        detail = str(getattr(issue, "detail", "") or "")
        _prefix, marker, payload = detail.partition("residuos=")
        if not marker:
            continue
        for match in _RESIDUAL_DETAIL_SPAN_RE.finditer(payload.strip()):
            normalized_span = match.group(1).strip()
            words = re.findall(r"[^\W\d_]+", normalized_span, flags=re.UNICODE)
            if len(words) < 3:
                continue
            span_pattern = re.compile(
                r"(?<!\w)" + r"[^\w]+".join(
                    re.escape(word) for word in words
                ) + r"(?!\w)",
                flags=re.IGNORECASE | re.UNICODE,
            )
            source_match = span_pattern.search(source_text)
            candidate_match = span_pattern.search(candidate_text)
            if source_match is None or candidate_match is None:
                continue
            before = source_text[:source_match.start()].rstrip()
            after = source_text[source_match.end():].lstrip()
            if (
                not before
                or before[-1] not in _QUOTE_OPENERS
                or not after
                or after[0] not in _QUOTE_CLOSERS
            ):
                continue
            span = candidate_match.group(0)
            if span not in spans:
                spans.append(span)
    return spans[:3]


def _strip_wrapping_quotes(text: str) -> str:
    value = str(text or "").strip()
    if (
        len(value) >= 2
        and value[0] in _QUOTE_OPENERS
        and value[-1] in _QUOTE_CLOSERS
    ):
        return value[1:-1].strip()
    return value


async def _repair_quoted_residual_spans(
    source_text: str,
    candidate_text: str,
    gate_issues: list,
    *,
    source_language: str,
    target_language: str,
    model: str,
    client,
    phase: str,
    section: str,
    prompt_options: Optional[dict],
    log_callback=None,
) -> Tuple[str, Optional[LLMResponse], list]:
    """Translate only copied quoted spans, then re-run every normal gate."""
    spans = _quoted_residual_spans(source_text, candidate_text, gate_issues)
    if not spans:
        return candidate_text, None, list(gate_issues or [])

    working = candidate_text
    combined_response: Optional[LLMResponse] = None
    repair_model = _resolve_quality_alert_model(model, prompt_options)
    for span in spans:
        system_prompt = f"""You are a precision literary translator.

Translate the supplied quoted prose into {target_language}. The quotation may
be written in a third language embedded inside a {source_language} source; it
must still be translated. Preserve meaning and register. Return only the
translated words between {TRANSLATE_TAG_IN} and {TRANSLATE_TAG_OUT}, without
quotation marks, notes, JSON, or any surrounding sentence."""
        user_prompt = f"""Quoted span to translate:
{span}

Return its complete {target_language} translation now."""
        if log_callback:
            log_callback(
                "target_language_gate_span_repair_request",
                "🔄 Traduciendo una cita residual sin reescribir el resto del fragmento.",
                data={
                    "type": "target_language_gate_span_repair_request",
                    "phase": phase,
                    "model": repair_model,
                    "primary_model": model,
                },
            )
        response = await _generate_alert_repair(
            client,
            user_prompt,
            system_prompt,
            primary_model=model,
            alert_model=repair_model,
            phase="repair",
        )
        combined_response = _merge_llm_usage(combined_response, response)
        replacement = (
            client.extract_translation(response.content)
            if response is not None
            else None
        )
        replacement = _strip_wrapping_quotes(replacement or "")
        if (
            not replacement
            or replacement.casefold() == span.casefold()
            or "[id" in replacement
            or len(replacement) > max(80, len(span) * 4)
        ):
            continue
        working = working.replace(span, replacement, 1)

    if working == candidate_text:
        return candidate_text, combined_response, list(gate_issues or [])

    repaired, output_scores, remaining_gate_issues, repair_issues, safe = (
        _evaluate_target_language_repair_candidate(
            working,
            candidate_text=candidate_text,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            phase=f"{phase}_quoted_span_repair",
            section=section,
            prompt_options=prompt_options,
            log_callback=log_callback,
        )
    )
    _record_text_candidate(
        prompt_options,
        text=repaired,
        source_text=source_text,
        phase=f"{phase}_quoted_span_repair",
        section=section,
        source_language=source_language,
        target_language=target_language,
        issues=repair_issues,
        extra_scores=output_scores,
        response=combined_response,
        decision="accepted" if safe else "retry",
        model=repair_model,
        source="target_language_gate_quoted_span_repair",
        log_callback=log_callback,
    )
    if safe:
        if log_callback:
            log_callback(
                "target_language_gate_span_repair_accepted",
                "✓ La cita residual se tradujo sin alterar el resto del fragmento.",
            )
        return repaired, combined_response, []
    if log_callback:
        rejection_codes = ", ".join(
            issue.code for issue in repair_issues if issue.severity == "reject"
        ) or "quality_gate"
        log_callback(
            "target_language_gate_span_repair_rejected",
            "⚠️ La cita se tradujo, pero el fragmento aún requiere reparación "
            f"por: {rejection_codes}.",
        )
    return candidate_text, combined_response, list(
        remaining_gate_issues or gate_issues
    )


def _evaluate_target_language_repair_candidate(
    repaired: str,
    *,
    candidate_text: str,
    source_text: str,
    source_language: str,
    target_language: str,
    phase: str,
    section: str,
    prompt_options: Optional[dict],
    log_callback=None,
) -> Tuple[str, dict, list, list, bool]:
    repaired, output_issues, output_scores = _guard_generated_text(
        clean_text_artifacts(repaired),
        phase=phase,
        style_reference=candidate_text,
        log_callback=log_callback,
    )
    repaired = apply_profile_glossary_corrections(
        repaired,
        prompt_options,
        source_text=source_text,
    )
    remaining_gate_issues = _target_language_gate_rejections(
        source_text,
        repaired,
        source_language=source_language,
        target_language=target_language,
        phase=phase,
        prompt_options=prompt_options,
    )
    fidelity = assess_fidelity(
        source_text,
        repaired,
        chunk_index=0,
        phase=phase,
        section=section or "Documento",
        source_language=source_language,
        target_language=target_language,
        prompt_options=prompt_options,
    )
    repair_issues = merge_guard_issues(
        output_issues,
        remaining_gate_issues,
        fidelity.rejections,
    )
    safe = (
        not any(issue.severity == "reject" for issue in repair_issues)
        and _content_length_ratio_ok(candidate_text, repaired)
    )
    return repaired, output_scores, remaining_gate_issues, repair_issues, safe


async def _maybe_repair_target_language_gate(
    source_text: str,
    candidate_text: str,
    gate_issues: list,
    *,
    source_language: str,
    target_language: str,
    model: str,
    client,
    log_callback=None,
    prompt_options: Optional[dict] = None,
    has_placeholders: bool = False,
    placeholder_format: Optional[Tuple[str, str]] = None,
    phase: str = "translation",
    section: str = "",
) -> Tuple[str, Optional[LLMResponse], list]:
    """Repair a candidate rejected for source-language residue before retrying."""
    if not candidate_text or not gate_issues:
        return candidate_text, None, list(gate_issues or [])

    quoted_candidate, quoted_response, quoted_issues = (
        await _repair_quoted_residual_spans(
            source_text,
            candidate_text,
            gate_issues,
            source_language=source_language,
            target_language=target_language,
            model=model,
            client=client,
            phase=phase,
            section=section,
            prompt_options=prompt_options,
            log_callback=log_callback,
        )
    )
    if quoted_candidate != candidate_text and not quoted_issues:
        return quoted_candidate, quoted_response, []

    normalized_source = _normalize_all_caps_prose_for_translation(source_text)
    caps_response: Optional[LLMResponse] = None
    if normalized_source != source_text:
        caps_options = dict(prompt_options or {})
        caps_options["custom_instructions"] = "\n\n".join(
            part for part in (
                str(caps_options.get("custom_instructions") or "").strip(),
                (
                    "The source contains promotional or running prose whose letter case was "
                    "normalized only to prevent ALL-CAPS text from being mistaken for a proper "
                    "name. Translate that prose fully. Preserve actual titles, people, places, "
                    "organizations, publication names, numbers, quotations, and placeholders."
                ),
            ) if part
        )
        caps_prompt = generate_translation_prompt(
            normalized_source,
            "",
            "",
            "",
            source_language,
            target_language,
            has_placeholders=has_placeholders,
            prompt_options=caps_options,
            placeholder_format=placeholder_format,
        )
        repair_model = _resolve_quality_alert_model(model, prompt_options)
        if log_callback:
            log_callback(
                "target_language_gate_all_caps_retry",
                "🔄 Reintentando prosa en mayúsculas con caja normalizada para evitar que "
                "el modelo la confunda con nombres propios.",
            )
        caps_response = await _generate_alert_repair(
            client,
            caps_prompt.user,
            caps_prompt.system,
            primary_model=model,
            alert_model=repair_model,
            phase="translation",
        )
        caps_candidate = client.extract_translation(caps_response.content) if caps_response else None
        if caps_candidate:
            (
                caps_candidate,
                caps_output_scores,
                caps_remaining_gate_issues,
                caps_repair_issues,
                caps_safe,
            ) = _evaluate_target_language_repair_candidate(
                caps_candidate,
                candidate_text=candidate_text,
                source_text=source_text,
                source_language=source_language,
                target_language=target_language,
                phase=f"{phase}_all_caps_retranslation",
                section=section,
                prompt_options=prompt_options,
                log_callback=log_callback,
            )
            _record_text_candidate(
                prompt_options,
                text=caps_candidate,
                source_text=source_text,
                phase=f"{phase}_all_caps_retranslation",
                section=section,
                source_language=source_language,
                target_language=target_language,
                issues=caps_repair_issues,
                extra_scores=caps_output_scores,
                response=caps_response,
                decision="accepted" if caps_safe else "retry",
                model=repair_model,
                source="target_language_gate_all_caps_retranslation",
                log_callback=log_callback,
            )
            if caps_safe:
                if log_callback:
                    log_callback(
                        "target_language_gate_all_caps_accepted",
                        "✓ La prosa en mayúsculas se tradujo completamente y volvió a pasar fidelidad.",
                    )
                return caps_candidate, caps_response, []

    issue_payload = [
        {
            "code": str(getattr(issue, "code", "target_language_gate")),
            "message": str(getattr(issue, "message", "")),
            "detail": str(getattr(issue, "detail", "")),
        }
        for issue in gate_issues[:8]
    ]
    instructions = f"""TARGET-LANGUAGE GATE REPAIR

The candidate contains ordinary words, phrases, or titles left in the source
language. Repair only that defect while preserving every fact, name, number,
date, citation, paragraph boundary, and placeholder.

- Source language: {source_language or 'unknown'}
- Target language: {target_language or 'unknown'}
- Translate all ordinary source-language residue into the target language.
- Do not treat capitalization alone as evidence of a proper name.
- Preserve actual personal/place/organization/publisher names and acronyms.
- Translate quoted dialogue, song lyrics, slogans, and lexical sound effects;
  they are not proper names merely because they are quoted or stylized.
- Preserve only genuinely non-lexical vocalizations or items explicitly marked
  for preservation by the active profile or approved glossary.
- Render translatable work titles and descriptive labels naturally in the
  target language, using the approved active-profile glossary when available.
- Do not summarize, omit, add explanations, or rewrite already-correct prose.
- Return the complete corrected candidate, not only the changed span.

Detected issues:
{json.dumps(issue_payload, ensure_ascii=False)}

Source text for fidelity comparison:
{source_text}
""".strip()
    repair_prompt = generate_refinement_prompt(
        draft_translation=candidate_text,
        target_language=target_language,
        has_placeholders=has_placeholders,
        prompt_options=dict(prompt_options or {}),
        placeholder_format=placeholder_format,
        additional_instructions=instructions,
    )
    repair_model = _resolve_quality_alert_model(model, prompt_options)
    if log_callback:
        log_callback(
            "target_language_gate_repair_request",
            f"🔄 Reparando residuos de {source_language or 'idioma fuente'} en {phase} con {repair_model}.",
            data={
                "type": "target_language_gate_repair_request",
                "phase": phase,
                "model": repair_model,
                "primary_model": model,
                "issues": issue_payload,
            },
        )
    response = await _generate_alert_repair(
        client,
        repair_prompt.user,
        repair_prompt.system,
        primary_model=model,
        alert_model=repair_model,
        phase="repair",
    )
    if not response:
        return candidate_text, None, list(gate_issues)
    repaired = client.extract_translation(response.content)
    if not repaired:
        return candidate_text, response, list(gate_issues)

    repaired, output_scores, remaining_gate_issues, repair_issues, safe = (
        _evaluate_target_language_repair_candidate(
            repaired,
            candidate_text=candidate_text,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            phase=f"{phase}_target_language_repair",
            section=section,
            prompt_options=prompt_options,
            log_callback=log_callback,
        )
    )
    response = _merge_llm_usage(response, caps_response)
    response = _merge_llm_usage(response, quoted_response)
    _record_text_candidate(
        prompt_options,
        text=repaired,
        source_text=source_text,
        phase=f"{phase}_target_language_repair",
        section=section,
        source_language=source_language,
        target_language=target_language,
        issues=repair_issues,
        extra_scores=output_scores,
        response=response,
        decision="accepted" if safe else "retry",
        model=repair_model,
        source="target_language_gate_repair",
        log_callback=log_callback,
    )
    if safe:
        if log_callback:
            log_callback(
                "target_language_gate_repair_accepted",
                "✓ Reparación de idioma aceptada; el candidato volvió a pasar fidelidad.",
            )
        return repaired, response, []

    if log_callback:
        reason = "; ".join(
            str(getattr(issue, "code", "quality_reject"))
            for issue in repair_issues
            if getattr(issue, "severity", "") == "reject"
        ) or "length_ratio"
        log_callback(
            "target_language_gate_repair_rejected",
            f"⚠️ La reparación de idioma no fue segura ({reason}); se descartó.",
        )
    return candidate_text, response, list(remaining_gate_issues or gate_issues)





async def _make_llm_request_with_adaptive_context(
    main_content: str,
    context_before: str,
    context_after: str,
    previous_translation_context: str,
    source_language: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    placeholder_format: Optional[Tuple[str, str]] = None,
    runtime_state: Optional[dict] = None,
) -> Tuple[Optional[str], str, Optional[LLMResponse]]:
    """
    Make LLM request with adaptive context sizing.

    This function uses the AdaptiveContextManager to:
    1. Start with a small context
    2. Retry with larger context if needed
    3. Return token usage info for the manager to learn from

    Args:
        main_content: Text to translate
        context_before: Context before main content
        context_after: Context after main content
        previous_translation_context: Previous translation for consistency
        source_language: Source language
        target_language: Target language
        model: LLM model name
        llm_client: LLM client instance
        log_callback: Logging callback function
        has_placeholders: If True, includes placeholder preservation instructions (for EPUB HTML tags)
        prompt_options: Optional dict with prompt customization options
        context_manager: AdaptiveContextManager for context sizing

    Returns:
        Tuple of (translated_text or None, actual_content_translated, LLMResponse)
    """
    current_content = main_content
    remaining_content = ""
    all_translations = []
    reduction_attempt = 0
    last_response: Optional[LLMResponse] = None
    content_risk_micro_split = False

    while current_content.strip():
        try:
            # Build the per-chunk glossary block (empty if no glossary configured)
            glossary_block = _build_chunk_glossary_block(
                current_content, prompt_options, log_callback=log_callback,
                runtime_state=runtime_state,
                phase="translation",
            )
            current_section = infer_section_title(current_content, context_before=context_before, fallback="")
            continuity_block = build_literary_continuity_block(
                prompt_options=prompt_options,
                runtime_state=runtime_state,
                current_text=current_content,
                source_language=source_language,
                target_language=target_language,
                section=current_section,
                log_callback=log_callback,
            )
            style_hint = build_style_continuity_hint(
                previous_translation_context,
                prompt_options=prompt_options,
            )
            if style_hint:
                continuity_block = "\n\n".join(
                    part for part in (continuity_block, style_hint) if part
                )

            # Generate prompts
            prompt_pair = generate_translation_prompt(
                current_content,
                context_before,
                context_after,
                previous_translation_context,
                source_language,
                target_language,
                has_placeholders=has_placeholders,
                prompt_options=prompt_options,
                placeholder_format=placeholder_format,
                glossary_block=glossary_block,
                continuity_block=continuity_block,
            )

            # Log the request
            if log_callback and reduction_attempt == 0:
                log_callback("llm_request", "Sending request to LLM", data={
                    'type': 'llm_request',
                    'system_prompt': prompt_pair.system,
                    'user_prompt': prompt_pair.user,
                    'model': model
                })

            start_time = time.time()
            client = llm_client or default_client

            # Set context from manager if available
            if context_manager and hasattr(client, 'context_window'):
                new_ctx = context_manager.get_context_size()
                if client.context_window != new_ctx:
                    if log_callback:
                        log_callback("context_update",
                            f"📐 Updating context window: {client.context_window} → {new_ctx}")
                    else:
                        tqdm.write(f"\n📐 Context: {client.context_window} → {new_ctx}")
                client.context_window = new_ctx

            llm_response = await _client_generate(
                client,
                prompt_pair.user,
                system_prompt=prompt_pair.system,
                temperature=temperature_for_phase("translation"),
            )
            execution_time = time.time() - start_time

            if not llm_response:
                return None, main_content, None

            last_response = llm_response
            full_raw_response = llm_response.content

            # Check if we should retry with larger context (adaptive strategy)
            if llm_response.was_truncated:
                if context_manager and context_manager.should_retry_with_larger_context(
                    True, llm_response.context_used
                ):
                    context_manager.increase_context()
                    continue  # Retry with larger context
                if log_callback:
                    log_callback(
                        "provider_truncation_rejected",
                        "⚠️ El proveedor cortó la respuesta; el fragmento se reintentará sin aceptar texto parcial.",
                    )
                return None, main_content, last_response

            # Log the response
            if log_callback:
                log_callback("llm_response", "LLM Response received", data={
                    'type': 'llm_response',
                    'response': full_raw_response,
                    'execution_time': execution_time,
                    'model': model,
                    'tokens': {
                        'prompt': llm_response.prompt_tokens,
                        'completion': llm_response.completion_tokens,
                        'total': llm_response.context_used,
                        'limit': llm_response.context_limit
                    }
                })

            # Extract translation
            translated_text = client.extract_translation(full_raw_response)
            if translated_text:
                translated_text, guard_issues, guard_scores = _guard_generated_text(
                    translated_text,
                    phase="translation",
                    style_reference=previous_translation_context,
                    log_callback=log_callback,
                )
                translated_text, locale_repair_response = await _maybe_repair_quality_alerts(
                    translated_text,
                    target_language=target_language,
                    model=model,
                    client=client,
                    log_callback=log_callback,
                    prompt_options=prompt_options,
                    has_placeholders=has_placeholders,
                    placeholder_format=placeholder_format,
                    phase="translation",
                )
                last_response = _merge_llm_usage(last_response, locale_repair_response)
                translated_text = apply_profile_glossary_corrections(
                    translated_text,
                    prompt_options,
                    source_text=current_content,
                )
                gate_rejections = _target_language_gate_rejections(
                    current_content,
                    translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    phase="translation",
                    prompt_options=prompt_options,
                )
                if gate_rejections:
                    translated_text, language_repair_response, gate_rejections = (
                        await _maybe_repair_target_language_gate(
                            current_content,
                            translated_text,
                            gate_rejections,
                            source_language=source_language,
                            target_language=target_language,
                            model=model,
                            client=client,
                            log_callback=log_callback,
                            prompt_options=prompt_options,
                            has_placeholders=has_placeholders,
                            placeholder_format=placeholder_format,
                            phase="translation",
                            section=current_section,
                        )
                    )
                    last_response = _merge_llm_usage(last_response, language_repair_response)
                candidate_issues = merge_guard_issues(guard_issues, gate_rejections)
                _record_text_candidate(
                    prompt_options,
                    text=translated_text,
                    source_text=current_content,
                    phase="translation",
                    section=current_section,
                    source_language=source_language,
                    target_language=target_language,
                    issues=candidate_issues,
                    extra_scores=guard_scores,
                    response=last_response,
                    decision="retry" if any(issue.severity == "reject" for issue in candidate_issues) else "accepted",
                    model=model,
                    source="translation",
                    log_callback=log_callback,
                )
                if any(issue.severity == "reject" for issue in candidate_issues):
                    if prompt_options is not None and gate_rejections:
                        prompt_options["_last_target_language_gate_rejection"] = {
                            "phase": "translation",
                            "source_language": source_language,
                            "target_language": target_language,
                            "issues": [
                                {
                                    "code": str(getattr(issue, "code", "target_language_gate")),
                                    "detail": str(getattr(issue, "detail", "")),
                                }
                                for issue in gate_rejections[:8]
                            ],
                        }
                    _log_target_language_gate_rejection(
                        log_callback,
                        event="target_language_gate_rejected",
                        phase="translation",
                        issues=[issue for issue in candidate_issues if issue.severity == "reject"],
                    )
                    return None, main_content, last_response

                if prompt_options is not None:
                    prompt_options.pop("_last_target_language_gate_rejection", None)

            if translated_text:
                all_translations.append(translated_text)
            else:
                # Extraction failed - tags not found or malformed
                _record_text_candidate(
                    prompt_options,
                    text=full_raw_response,
                    source_text=current_content,
                    phase="translation_extraction",
                    section=current_section,
                    source_language=source_language,
                    target_language=target_language,
                    issues=[CandidateIssue(
                        "translation_extraction_failed",
                        "reject",
                        "Failed to extract translation tags from LLM response.",
                    )],
                    response=last_response,
                    decision="retry",
                    model=model,
                    source="translation_extraction",
                    log_callback=log_callback,
                )
                if log_callback:
                    log_callback("translation_extraction_failed",
                        "⚠️ WARNING: Failed to extract translation (tags not found or malformed)")
                    log_callback("translation_extraction_failed_preview",
                        f"Response preview (first 300 chars): {full_raw_response[:300]}")

                # Implicit truncation detection: model started <TRANSLATION> but hit EOS before </TRANSLATION>
                stripped_response = full_raw_response.strip()
                if (stripped_response.startswith(TRANSLATE_TAG_IN) and not stripped_response.endswith(TRANSLATE_TAG_OUT)):
                    if context_manager and context_manager.should_retry_with_larger_context(True, llm_response.context_used):
                        if log_callback:
                            log_callback("implicit_truncation_retry",
                                "🔄 Model stopped before closing tag. Retrying with larger context...")
                        context_manager.increase_context()
                        continue  # Retry with larger context

                # For EPUB with placeholders, failing to extract is CRITICAL
                # because using the raw response would include <TRANSLATION> tags in the HTML
                if has_placeholders:
                    if log_callback:
                        log_callback("epub_extraction_critical_fail",
                            "CRITICAL: Cannot use raw response for EPUB (would corrupt HTML structure)")
                    return None, main_content, last_response

                # For plain text, try fallback to raw response (legacy behavior)
                if current_content not in full_raw_response:
                    if log_callback:
                        log_callback("using_raw_response_fallback",
                            "Using raw response as fallback (plain text mode)")
                    fallback_text, guard_issues, guard_scores = _guard_generated_text(
                        full_raw_response.strip(),
                        phase="translation_fallback",
                        style_reference=previous_translation_context,
                        log_callback=log_callback,
                    )
                    fallback_text = apply_profile_glossary_corrections(
                        fallback_text,
                        prompt_options,
                        source_text=current_content,
                    )
                    gate_rejections = _target_language_gate_rejections(
                        current_content,
                        fallback_text,
                        source_language=source_language,
                        target_language=target_language,
                        phase="translation_fallback",
                        prompt_options=prompt_options,
                    )
                    candidate_issues = merge_guard_issues(guard_issues, gate_rejections)
                    _record_text_candidate(
                        prompt_options,
                        text=fallback_text,
                        source_text=current_content,
                        phase="translation_fallback",
                        section=current_section,
                        source_language=source_language,
                        target_language=target_language,
                        issues=candidate_issues,
                        extra_scores=guard_scores,
                        response=last_response,
                        decision="retry" if any(issue.severity == "reject" for issue in candidate_issues) else "accepted",
                        model=model,
                        source="translation_fallback",
                        log_callback=log_callback,
                    )
                    if any(issue.severity == "reject" for issue in candidate_issues):
                        _log_target_language_gate_rejection(
                            log_callback,
                            event="target_language_gate_fallback_rejected",
                            phase="translation fallback",
                            issues=[issue for issue in candidate_issues if issue.severity == "reject"],
                        )
                        return None, main_content, last_response
                    all_translations.append(fallback_text)
                    if last_response:
                        last_response.was_fallback = True
                else:
                    # Response contains input - this is an error
                    _record_text_candidate(
                        prompt_options,
                        text=full_raw_response,
                        source_text=current_content,
                        phase="translation_response_echo",
                        section=current_section,
                        source_language=source_language,
                        target_language=target_language,
                        issues=[CandidateIssue(
                            "llm_prompt_in_response",
                            "reject",
                            "LLM response appears to contain the input.",
                        )],
                        response=last_response,
                        decision="retry",
                        model=model,
                        source="translation_response_echo",
                        log_callback=log_callback,
                    )
                    if log_callback:
                        log_callback("llm_prompt_in_response_warning",
                            "WARNING: LLM response seems to contain input. Discarded.")
                    return None, main_content, last_response

            # If we had remaining content from a previous split, translate it
            if remaining_content.strip():
                current_content = remaining_content
                remaining_content = ""
                # Update context for continuity
                if all_translations:
                    words = all_translations[-1].split()
                    previous_translation_context = " ".join(words[-25:]) if len(words) > 25 else all_translations[-1]
                reduction_attempt = 0  # Reset for new content
                continue

            # Success - combine all translations
            combined = "\n".join(all_translations) if all_translations else None
            if combined and content_risk_micro_split:
                combined, stitch_response = await _stitch_content_risk_fragments(
                    all_translations,
                    target_language=target_language,
                    client=client,
                    log_callback=log_callback,
                    prompt_options=prompt_options,
                )
                last_response = _merge_llm_usage(last_response, stitch_response)
            if combined:
                observe_literary_continuity(
                    runtime_state=runtime_state,
                    source_text=main_content,
                    translated_text=combined,
                    section=current_section,
                    phase="translation",
                )
            return combined, main_content, last_response

        except RepetitionLoopError as e:
            # Repetition loop detected - this typically happens with thinking models
            # when context window is too small. Try increasing context.
            if context_manager:
                old_context = context_manager.get_context_size()
                # Force a larger context increase for repetition loops
                context_manager.increase_context()
                context_manager.increase_context()  # Double increase for repetition loops
                new_context = context_manager.get_context_size()

                if new_context > old_context:
                    if log_callback:
                        log_callback("repetition_loop_retry",
                            f"🔄 Repetition loop detected! Increasing context from {old_context} to {new_context} tokens")
                    else:
                        tqdm.write(f"\n🔄 Repetition loop - increasing context to {new_context}")
                    continue  # Retry with larger context

            # No context manager or can't increase further
            if log_callback:
                log_callback("repetition_loop_fatal",
                    f"⚠️ Repetition loop detected and cannot recover. "
                    f"Try manually increasing OLLAMA_NUM_CTX. Error: {e}")
            else:
                tqdm.write(f"\n⚠️ Repetition loop detected - increase OLLAMA_NUM_CTX")
            return None, main_content, last_response

        except ContentRiskError as e:
            reduction_attempt += 1
            if reduction_attempt > MAX_CONTENT_RISK_REDUCTION_ATTEMPTS:
                if log_callback:
                    log_callback(
                        "content_risk_split_fatal",
                        "Provider content filter still rejected the passage after "
                        f"{MAX_CONTENT_RISK_REDUCTION_ATTEMPTS} bounded splits: {e}",
                    )
                return None, main_content, last_response

            first_part, second_part = split_chunk_for_retry(current_content, 0.5)
            if (
                len(first_part.strip()) < MIN_CONTENT_RISK_CHARACTERS
                or not second_part.strip()
                or first_part == current_content
            ):
                if log_callback:
                    log_callback(
                        "content_risk_split_fatal",
                        "Provider content filter rejected the smallest safe semantic unit; "
                        "the checkpoint remains resumable.",
                    )
                return None, main_content, last_response

            if len(first_part.strip()) < MIN_CHUNK_CHARACTERS:
                content_risk_micro_split = True

            if log_callback:
                log_callback(
                    "content_risk_split_retry",
                    "Provider content filter rejected this passage; retrying as smaller "
                    f"sentence-aligned units ({len(first_part)} + {len(second_part)} chars).",
                )
            current_content = first_part
            remaining_content = second_part + ("\n" + remaining_content if remaining_content else "")

        except ContextOverflowError as e:
            # If we have a context manager, try increasing context
            if context_manager and context_manager.should_retry_with_larger_context(True, 0):
                context_manager.increase_context()
                continue  # Retry with larger context

            reduction_attempt += 1

            if reduction_attempt > MAX_CHUNK_REDUCTION_ATTEMPTS:
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Context overflow: Max reduction attempts ({MAX_CHUNK_REDUCTION_ATTEMPTS}) "
                        f"exceeded. Original error: {e}")
                else:
                    tqdm.write(f"\n⚠️ Context overflow after {MAX_CHUNK_REDUCTION_ATTEMPTS} reduction attempts")
                return None, main_content, last_response

            # Calculate new reduction factor
            reduction_factor = CHUNK_REDUCTION_FACTOR ** reduction_attempt

            if log_callback:
                log_callback("context_overflow_retry",
                    f"⚠️ Context overflow detected! Reducing chunk to {reduction_factor*100:.0f}% "
                    f"(attempt {reduction_attempt}/{MAX_CHUNK_REDUCTION_ATTEMPTS})")
            else:
                tqdm.write(f"\n⚠️ Context overflow - reducing chunk (attempt {reduction_attempt})")

            # Split the content
            first_part, second_part = split_chunk_for_retry(current_content, reduction_factor)

            if len(first_part) < MIN_CHUNK_CHARACTERS and not all_translations:
                # Can't reduce further without losing too much content
                if log_callback:
                    log_callback("context_overflow_fatal",
                        f"⚠️ Cannot reduce chunk further (min size: {MIN_CHUNK_CHARACTERS} chars)")
                return None, main_content, last_response

            current_content = first_part
            # Accumulate remaining content for later
            if second_part.strip():
                remaining_content = second_part + ("\n" + remaining_content if remaining_content else "")

    # Shouldn't reach here normally
    return "\n".join(all_translations) if all_translations else None, main_content, last_response


# Legacy wrapper for backward compatibility

async def generate_translation_request(main_content, context_before, context_after, previous_translation_context,
                                       source_language="English", target_language="Chinese", model=DEFAULT_MODEL,
                                       llm_client=None, log_callback=None, has_placeholders=False,
                                       prompt_options=None, context_manager: AdaptiveContextManager = None,
                                       placeholder_format: Optional[Tuple[str, str]] = None,
                                       runtime_state: Optional[dict] = None):
    """
    Generate translation request to LLM API with automatic context overflow handling.

    Args:
        main_content (str): Text to translate
        context_before (str): Context before main content
        context_after (str): Context after main content
        previous_translation_context (str): Previous translation for consistency
        source_language (str): Source language
        target_language (str): Target language
        model (str): LLM model name
        llm_client: LLM client instance
        log_callback (callable): Logging callback function
        has_placeholders (bool): If True, includes placeholder preservation instructions
        prompt_options (dict): Optional dict with prompt customization options
        context_manager (AdaptiveContextManager): Optional context manager for adaptive retry on overflow
        placeholder_format (Tuple[str, str]): Optional tuple of (prefix, suffix) for placeholders.
            e.g., ('[', ']') for [0] format or ('[[', ']]') for [[0]] format
        runtime_state: Optional transient per-job state for glossary/continuity memory

    Returns:
        str: Translated text or None if failed
    """
    # Skip LLM translation for single character or empty chunks
    if len(main_content.strip()) <= 1:
        if log_callback:
            log_callback("skip_translation", f"Skipping LLM for single/empty character: '{main_content}'")
        return main_content

    if prompt_options is not None:
        prompt_options.pop("_last_target_language_gate_rejection", None)

    # Use the adaptive context handler
    translated_text, _, _ = await _make_llm_request_with_adaptive_context(
        main_content=main_content,
        context_before=context_before,
        context_after=context_after,
        previous_translation_context=previous_translation_context,
        source_language=source_language,
        target_language=target_language,
        model=model,
        llm_client=llm_client,
        log_callback=log_callback,
        has_placeholders=has_placeholders,
        prompt_options=prompt_options,
        context_manager=context_manager,
        placeholder_format=placeholder_format,
        runtime_state=runtime_state,
    )

    if translated_text:
        translated_text, invariant_repairs = repair_source_invariants(
            main_content,
            translated_text,
        )
        if invariant_repairs and log_callback:
            log_callback(
                "source_invariant_repaired",
                "Restored source identifier(s) before quality audit: "
                + summarize_source_invariant_repairs(invariant_repairs),
            )
        return translated_text
    else:
        gate_rejection = (prompt_options or {}).get(
            "_last_target_language_gate_rejection"
        )
        if gate_rejection:
            err_msg = (
                "Candidate rejected by the target-language quality gate; "
                "the provider request itself succeeded"
            )
            event = "target_language_candidate_rejected"
        else:
            err_msg = "ERROR: LLM API request failed"
            event = "llm_api_error"
        if log_callback:
            log_callback(event, err_msg)
        else:
            tqdm.write(f"\n{err_msg}")
        return None



async def _make_refinement_request(
    draft_translation: str,
    context_before: str,
    context_after: str,
    previous_refined_context: str,
    target_language: str,
    model: str,
    llm_client,
    log_callback,
    has_placeholders: bool,
    prompt_options: dict = None,
    context_manager: AdaptiveContextManager = None,
    runtime_state: Optional[dict] = None,
    section: str = "",
    source_text: str = "",
) -> Tuple[Optional[str], Optional[LLMResponse]]:
    """
    Make LLM request for refinement pass.

    Similar to translation request but uses the refinement prompt.

    Args:
        draft_translation: First-pass translation to refine
        context_before: Previously refined text for context
        context_after: Text appearing after for context
        previous_refined_context: Last refined text for consistency
        target_language: Target language
        model: LLM model name
        llm_client: LLM client instance
        log_callback: Logging callback function
        has_placeholders: If True, includes placeholder preservation instructions
        prompt_options: Optional dict with prompt customization options
        context_manager: AdaptiveContextManager for context sizing

    Returns:
        Tuple of (refined_text or None, LLMResponse)
    """
    # Extract refinement instructions from prompt_options
    refinement_instructions = prompt_options.get('refinement_instructions', '') if prompt_options else ''
    transform_instructions = build_text_transform_instructions(prompt_options, target_language)
    if transform_instructions:
        refinement_instructions = "\n\n".join(
            part for part in (transform_instructions, refinement_instructions) if part
        )

    # Filter glossary/profile entries against both the source chunk and the
    # draft. Source terms disappear after translation, but they are still the
    # canonical triggers for source->target consistency during refinement.
    glossary_match_text = "\n\n".join(
        part for part in (source_text, draft_translation) if part and part.strip()
    ) or draft_translation
    glossary_block = _build_chunk_glossary_block(
        glossary_match_text, prompt_options, log_callback=log_callback,
        runtime_state=runtime_state,
        phase="refinement",
    )
    continuity_block = build_literary_continuity_block(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        current_text=draft_translation,
        source_language=target_language,
        target_language=target_language,
        section=section,
        log_callback=log_callback,
    )
    style_hint = build_style_continuity_hint(
        previous_refined_context or context_before,
        prompt_options=prompt_options,
    )
    if style_hint:
        continuity_block = "\n\n".join(
            part for part in (continuity_block, style_hint) if part
        )

    # Generate refinement prompts
    prompt_pair = generate_refinement_prompt(
        draft_translation=draft_translation,
        context_before=context_before,
        context_after=context_after,
        previous_refined_context=previous_refined_context,
        target_language=target_language,
        has_placeholders=False,
        prompt_options=prompt_options,
        additional_instructions=refinement_instructions,
        glossary_block=glossary_block,
        continuity_block=continuity_block,
    )

    client = llm_client or default_client
    last_response: Optional[LLMResponse] = None

    # Retry loop with adaptive context (mirrors translation logic)
    while True:
        try:
            # Log the request
            if log_callback:
                log_callback("refinement_request", "Sending refinement request to LLM", data={
                    'type': 'refinement_request',
                    'system_prompt': prompt_pair.system,
                    'user_prompt': prompt_pair.user,
                    'model': model
                })

            start_time = time.time()

            # Set context from manager if available
            if context_manager and hasattr(client, 'context_window'):
                new_ctx = context_manager.get_context_size()
                if client.context_window != new_ctx:
                    if log_callback:
                        log_callback("context_update",
                            f"📐 Refinement context window: {client.context_window} → {new_ctx}")
                    client.context_window = new_ctx

            llm_response = await _client_make_request(
                client,
                prompt_pair.user,
                model,
                system_prompt=prompt_pair.system,
                temperature=temperature_for_phase("refinement"),
            )
            execution_time = time.time() - start_time

            if not llm_response:
                return None, None

            last_response = llm_response

            # Check if we should retry with larger context (adaptive strategy)
            if llm_response.was_truncated:
                if context_manager and context_manager.should_retry_with_larger_context(
                    True, llm_response.context_used
                ):
                    context_manager.increase_context()
                    continue  # Retry with larger context
                if log_callback:
                    log_callback(
                        "refinement_truncation_rejected",
                        "⚠️ El proveedor cortó la revisión; se conserva el candidato completo anterior.",
                    )
                return None, last_response

            full_raw_response = llm_response.content

            # Log the response
            if log_callback:
                log_callback("refinement_response", "Refinement response received", data={
                    'type': 'refinement_response',
                    'response': full_raw_response,
                    'execution_time': execution_time,
                    'model': model,
                    'tokens': {
                        'prompt': llm_response.prompt_tokens,
                        'completion': llm_response.completion_tokens,
                        'total': llm_response.context_used,
                        'limit': llm_response.context_limit
                    }
                })

            # Extract refined text
            refined_text = client.extract_translation(full_raw_response)

            if refined_text:
                style_reference = previous_refined_context or context_before
                refined_text, guard_issues, guard_scores = _guard_generated_text(
                    refined_text,
                    phase="refinement",
                    style_reference=style_reference,
                    log_callback=log_callback,
                )
                refined_text = apply_profile_glossary_corrections(
                    refined_text,
                    prompt_options,
                    source_text=source_text or draft_translation,
                )
                refined_text, invariant_repairs = repair_source_invariants(
                    source_text or draft_translation,
                    refined_text,
                )
                if invariant_repairs and log_callback:
                    log_callback(
                        "source_invariant_repaired",
                        "Restored source identifier(s) after editorial review: "
                        + summarize_source_invariant_repairs(invariant_repairs),
                    )
                refined_text, locale_repair_response = await _maybe_repair_quality_alerts(
                    refined_text,
                    target_language=target_language,
                    model=model,
                    client=client,
                    log_callback=log_callback,
                    prompt_options=prompt_options,
                    has_placeholders=has_placeholders,
                    phase="refinement",
                )
                llm_response = _merge_llm_usage(llm_response, locale_repair_response)
                refined_text, post_repair_guard_issues, post_repair_guard_scores = _guard_generated_text(
                    refined_text,
                    phase="refinement_post_repair",
                    style_reference=style_reference,
                    log_callback=log_callback,
                )
                guard_issues = merge_guard_issues(guard_issues, post_repair_guard_issues)
                guard_scores.update(post_repair_guard_scores)
                gate_rejections = _target_language_gate_rejections(
                    source_text or draft_translation,
                    refined_text,
                    source_language=(prompt_options or {}).get("_source_language", ""),
                    target_language=target_language,
                    phase="refinement",
                    prompt_options=prompt_options,
                )
                if gate_rejections:
                    refined_text, language_repair_response, gate_rejections = (
                        await _maybe_repair_target_language_gate(
                            source_text or draft_translation,
                            refined_text,
                            gate_rejections,
                            source_language=(prompt_options or {}).get("_source_language", ""),
                            target_language=target_language,
                            model=model,
                            client=client,
                            log_callback=log_callback,
                            prompt_options=prompt_options,
                            has_placeholders=has_placeholders,
                            phase="refinement",
                            section=section,
                        )
                    )
                    llm_response = _merge_llm_usage(llm_response, language_repair_response)
                candidate_issues = merge_guard_issues(guard_issues, gate_rejections)
                _record_text_candidate(
                    prompt_options,
                    text=refined_text,
                    source_text=source_text or draft_translation,
                    phase="refinement",
                    section=section,
                    source_language=(prompt_options or {}).get("_source_language", ""),
                    target_language=target_language,
                    issues=candidate_issues,
                    extra_scores=guard_scores,
                    response=llm_response,
                    decision="retry" if any(issue.severity == "reject" for issue in candidate_issues) else "accepted",
                    model=model,
                    source="refinement",
                    log_callback=log_callback,
                )
                if any(issue.severity == "reject" for issue in candidate_issues):
                    _log_target_language_gate_rejection(
                        log_callback,
                        event="refinement_output_guard_rejected",
                        phase="refinement",
                        issues=[issue for issue in candidate_issues if issue.severity == "reject"],
                    )
                    return None, llm_response
                return refined_text, llm_response
            else:
                # Fallback to raw response if no tags found
                if draft_translation not in full_raw_response:
                    fallback_text, guard_issues, guard_scores = _guard_generated_text(
                        full_raw_response.strip(),
                        phase="refinement_fallback",
                        style_reference=previous_refined_context or context_before,
                        log_callback=log_callback,
                    )
                    fallback_text = apply_profile_glossary_corrections(
                        fallback_text,
                        prompt_options,
                        source_text=source_text or draft_translation,
                    )
                    fallback_text, invariant_repairs = repair_source_invariants(
                        source_text or draft_translation,
                        fallback_text,
                    )
                    if invariant_repairs and log_callback:
                        log_callback(
                            "source_invariant_repaired",
                            "Restored source identifier(s) in refinement fallback: "
                            + summarize_source_invariant_repairs(invariant_repairs),
                        )
                    gate_rejections = _target_language_gate_rejections(
                        source_text or draft_translation,
                        fallback_text,
                        source_language=(prompt_options or {}).get("_source_language", ""),
                        target_language=target_language,
                        phase="refinement_fallback",
                        prompt_options=prompt_options,
                    )
                    candidate_issues = merge_guard_issues(guard_issues, gate_rejections)
                    _record_text_candidate(
                        prompt_options,
                        text=fallback_text,
                        source_text=source_text or draft_translation,
                        phase="refinement_fallback",
                        section=section,
                        source_language=(prompt_options or {}).get("_source_language", ""),
                        target_language=target_language,
                        issues=candidate_issues,
                        extra_scores=guard_scores,
                        response=llm_response,
                        decision="retry" if any(issue.severity == "reject" for issue in candidate_issues) else "accepted",
                        model=model,
                        source="refinement_fallback",
                        log_callback=log_callback,
                    )
                    if any(issue.severity == "reject" for issue in candidate_issues):
                        return None, llm_response
                    return fallback_text, llm_response
                else:
                    if log_callback:
                        log_callback("refinement_warning",
                            "WARNING: Refinement response contains input. Using original.")
                    return None, llm_response

        except RepetitionLoopError as e:
            # Repetition loop detected - try increasing context (double increase)
            if context_manager:
                old_context = context_manager.get_context_size()
                context_manager.increase_context()
                context_manager.increase_context()  # Double increase for repetition loops
                new_context = context_manager.get_context_size()

                if new_context > old_context:
                    if log_callback:
                        log_callback("refinement_repetition_retry",
                            f"🔄 Refinement repetition loop! Increasing context from {old_context} to {new_context} tokens")
                    continue  # Retry with larger context

            # No context manager or can't increase further
            if log_callback:
                log_callback("refinement_error",
                    f"⚠️ Refinement repetition loop, cannot recover: {e}")
            return None, last_response

        except ContextOverflowError as e:
            # Context overflow - try increasing context
            if context_manager and context_manager.should_retry_with_larger_context(True, 0):
                context_manager.increase_context()
                if log_callback:
                    log_callback("refinement_overflow_retry",
                        f"⚠️ Refinement context overflow! Retrying with context {context_manager.get_context_size()}")
                continue  # Retry with larger context

            # Can't increase further
            if log_callback:
                log_callback("refinement_error",
                    f"⚠️ Refinement context overflow, cannot recover: {e}")
            return None, last_response


async def refine_chunks(
    translated_chunks: List[str],
    original_chunks: List[Dict],
    target_language: str,
    model_name: str,
    api_endpoint: str,
    log_callback=None,
    stats_callback=None,
    check_interruption_callback=None,
    llm_provider="ollama",
    gemini_api_key=None,
    openai_api_key=None,
    openrouter_api_key=None,
    mistral_api_key=None,
    deepseek_api_key=None,
    poe_api_key=None,
    nim_api_key=None,
    context_window=2048,
    auto_adjust_context=True,
    prompt_options=None,
    checkpoint_callback=None,
    chunk_index_offset: int = 0,
) -> List[str]:
    """
    Refine translated chunks with a second pass for literary quality improvement.

    This function takes already-translated chunks and runs them through a
    refinement prompt that focuses on improving literary quality, natural flow,
    and stylistic excellence.

    Args:
        translated_chunks: List of translated text strings from first pass
        original_chunks: Original chunk dictionaries (for context structure)
        target_language: Target language name
        model_name: LLM model name
        api_endpoint: API endpoint        log_callback: Logging callback
        stats_callback: Statistics update callback
        check_interruption_callback: Interruption check callback
        llm_provider: LLM provider name
        gemini_api_key: Gemini API key
        openai_api_key: OpenAI API key
        openrouter_api_key: OpenRouter API key
        context_window: Initial context window size
        auto_adjust_context: Enable adaptive context adjustment
        prompt_options: Optional dict with prompt customization options

    Returns:
        List of refined text strings
    """
    total_chunks = len(translated_chunks)
    refined_parts = []
    last_refined_context = ""
    current_section = "Documento"
    # Transient per-job state (e.g. glossary cap warning dedupe) — never persisted.
    runtime_state: dict = {}
    prompt_options = dict(prompt_options or {})
    apply_faithful_modernize_defaults(prompt_options)
    faithful_modernize = is_faithful_modernize(prompt_options)
    preserve_transform_blocks = faithful_modernize and prompt_bool(
        prompt_options,
        "preserve_block_structure",
        True,
    )
    transform_repair_attempts = int(prompt_options.get("transform_repair_attempts") or 0)
    editorial_quality_guard = prompt_options.get('editorial_quality_guard', True)
    editorial_quality_report = prompt_options.get('_editorial_quality_report')
    if editorial_quality_report is not None and profile_enabled(prompt_options):
        try:
            editorial_quality_report.profile_summary = build_profile_report_summary(prompt_options)
        except Exception as profile_summary_error:
            if log_callback:
                log_callback(
                    "profile_summary_error",
                    f"⚠️ Could not attach active profile summary to editorial report: {profile_summary_error}"
                )

    def _checkpoint(local_index: int, draft_text: str, output_text: Optional[str], failed: bool) -> None:
        if not checkpoint_callback:
            return
        try:
            checkpoint_callback(
                chunk_index_offset + local_index,
                draft_text,
                None if failed else output_text,
                progress_tracker.get_stats().to_dict(),
            )
        except Exception as exc:
            if log_callback:
                log_callback(
                    "refinement_checkpoint_error",
                    f"⚠️ Could not save refinement checkpoint at chunk "
                    f"{chunk_index_offset + local_index + 1}: {exc}"
                )

    # Single-phase refinement tracker (the workflow phase, when this runs as the
    # second pass of a translate→refine job, is tagged at the handler seam).
    progress_tracker = TokenProgressTracker()
    progress_tracker.start()
    token_counter = TokenChunker(max_tokens=800)
    for chunk_text in translated_chunks:
        token_count = token_counter.count_tokens(chunk_text)
        progress_tracker.register_chunk(token_count)

    if log_callback:
        log_callback("refinement_start", f"✨ Starting refinement pass ({total_chunks} chunks)...")

    # Determine if model is a thinking model for initial context sizing
    is_known_thinking_model = any(tm in model_name.lower() for tm in THINKING_MODELS)

    # Refinement needs MORE context than translation because:
    # - The prompt includes the already-translated text (input)
    # - Plus context before/after
    # - Plus instructions
    # So we start with at least 4096 or the user's context_window, whichever is larger
    REFINEMENT_MIN_CONTEXT = 4096

    if auto_adjust_context:
        if is_known_thinking_model:
            initial_context = max(ADAPTIVE_CONTEXT_INITIAL_THINKING, REFINEMENT_MIN_CONTEXT)
        else:
            initial_context = max(INITIAL_CONTEXT_SIZE * 2, REFINEMENT_MIN_CONTEXT)
    else:
        initial_context = max(context_window, REFINEMENT_MIN_CONTEXT)

    # Create LLM client
    llm_client = create_llm_client(
        llm_provider, gemini_api_key, api_endpoint, model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        context_window=initial_context, log_callback=log_callback
    )

    # Create adaptive context manager for Ollama
    context_manager = None
    if llm_provider == "ollama" and auto_adjust_context:
        from .context_optimizer import MAX_CONTEXT_SIZE
        context_manager = AdaptiveContextManager(
            initial_context=initial_context,
            context_step=CONTEXT_STEP,
            max_context=MAX_CONTEXT_SIZE,
            log_callback=log_callback
        )
        if log_callback:
            log_callback("refinement_context", f"📐 Refinement context: starting at {initial_context} tokens (min for refinement: {REFINEMENT_MIN_CONTEXT})")

    # Detect thinking model status
    if llm_client and llm_provider == "ollama":
        await llm_client.detect_thinking_model()

    try:
        iterator = tqdm(
            enumerate(translated_chunks),
            total=total_chunks,
            desc=f"Refining {target_language} translation",
            unit="seg"
        ) if not log_callback else enumerate(translated_chunks)

        for i, draft_text in iterator:
            # Check for interruption
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback("refinement_interrupted",
                        f"Refinement interrupted at chunk {i+1}/{total_chunks}")
                else:
                    tqdm.write(f"\nRefinement interrupted at chunk {i+1}/{total_chunks}")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                break

            # Progress update (token-based)
            # Measure refinement time for this chunk
            chunk_start_time = time.time()

            # Skip empty chunks
            if not draft_text.strip():
                refined_parts.append(draft_text)
                chunk_elapsed = time.time() - chunk_start_time
                progress_tracker.mark_completed(i, chunk_elapsed)
                _checkpoint(i, draft_text, draft_text, failed=False)
                if stats_callback:
                    stats_callback(progress_tracker.get_stats().to_dict())
                continue

            # Skip very short content
            if len(draft_text.strip()) <= 1:
                refined_parts.append(draft_text)
                chunk_elapsed = time.time() - chunk_start_time
                progress_tracker.mark_completed(i, chunk_elapsed)
                _checkpoint(i, draft_text, draft_text, failed=False)
                if stats_callback:
                    stats_callback(progress_tracker.get_stats().to_dict())
                continue

            # Get context from original chunks if available
            context_before = ""
            context_after = ""
            if i < len(original_chunks):
                context_before = original_chunks[i].get("context_before", "")
                context_after = original_chunks[i].get("context_after", "")

            raw_draft_text = draft_text
            block_protection = protect_meaningful_blocks(
                raw_draft_text,
                enabled=preserve_transform_blocks,
            )
            request_draft_text = block_protection.protected_text
            source_text = _extract_source_text_for_guard(
                original_chunks[i] if i < len(original_chunks) else None,
                raw_draft_text,
            )
            if faithful_modernize and not source_text:
                source_text = raw_draft_text

            current_section = infer_section_title(
                raw_draft_text,
                context_before=context_before,
                fallback=current_section,
            )

            # Make refinement request
            try:
                refined_text, llm_response = await _make_refinement_request(
                    draft_translation=request_draft_text,
                    context_before=context_before,
                    context_after=context_after,
                    previous_refined_context=last_refined_context,
                    target_language=target_language,
                    model=model_name,
                    llm_client=llm_client,
                    log_callback=log_callback,
                    has_placeholders=False,
                    prompt_options=prompt_options,
                    context_manager=context_manager,
                    runtime_state=runtime_state,
                    section=current_section,
                    source_text=source_text,
                )
            except RateLimitError as e:
                if log_callback:
                    retry_msg = f" (retry after ~{e.retry_after}s)" if e.retry_after else ""
                    log_callback("rate_limit_pause",
                        f"⏸️ Rate limited by {e.provider or 'API'}{retry_msg}. "
                        f"Auto-pausing refinement at chunk {i+1}/{total_chunks}...")
                # Add remaining unrefined chunks as-is
                for remaining in translated_chunks[i:]:
                    refined_parts.append(remaining)
                raise  # Re-raise to handlers.py

            # Record success in context manager
            if refined_text is not None and llm_response and context_manager:
                context_manager.record_success(
                    prompt_tokens=llm_response.prompt_tokens,
                    completion_tokens=llm_response.completion_tokens,
                    context_limit=llm_response.context_limit
                )

            chunk_elapsed = time.time() - chunk_start_time

            if refined_text is not None:
                if block_protection.active:
                    block_ok, restored_text, block_error = restore_protected_blocks(
                        refined_text,
                        block_protection,
                    )
                    retry_count = 0
                    while (
                        not block_ok
                        and retry_count < transform_repair_attempts
                    ):
                        retry_count += 1
                        retry_options = dict(prompt_options)
                        retry_options["refinement_instructions"] = "\n\n".join(
                            part for part in (
                                retry_options.get("refinement_instructions", ""),
                                block_repair_instructions(block_error),
                            ) if part
                        )
                        if log_callback:
                            log_callback(
                                "modernize_block_retry",
                                f"Modernize chunk {i+1}/{total_chunks}: retrying block structure repair ({block_error})."
                            )
                        retry_text, retry_response = await _make_refinement_request(
                            draft_translation=request_draft_text,
                            context_before=context_before,
                            context_after=context_after,
                            previous_refined_context=last_refined_context,
                            target_language=target_language,
                            model=model_name,
                            llm_client=llm_client,
                            log_callback=log_callback,
                            has_placeholders=False,
                            prompt_options=retry_options,
                            context_manager=context_manager,
                            runtime_state=runtime_state,
                            section=current_section,
                            source_text=source_text,
                        )
                        llm_response = _merge_llm_usage(llm_response, retry_response)
                        refined_text = retry_text
                        if refined_text is None:
                            block_ok = False
                            block_error = "retry returned empty response"
                            break
                        block_ok, restored_text, block_error = restore_protected_blocks(
                            refined_text,
                            block_protection,
                        )

                    if not block_ok:
                        kept_text = clean_translated_text(raw_draft_text)
                        refined_parts.append(kept_text)
                        progress_tracker.mark_completed(i, chunk_elapsed)
                        last_refined_context = kept_text
                        if editorial_quality_report is not None:
                            decision = assess_refinement(
                                raw_draft_text,
                                kept_text,
                                chunk_index=chunk_index_offset + i + 1,
                                section=current_section,
                                source_text=raw_draft_text,
                                target_language=target_language,
                                source_language=prompt_options.get('_source_language', ''),
                                prompt_options=prompt_options,
                            )
                            decision.accepted = False
                            decision.issues.append(QualityIssue(
                                "modernize_block_marker_mismatch",
                                "reject",
                                "La modernizacion cambio, perdio o reordeno marcadores de bloque",
                                block_error,
                            ))
                            editorial_quality_report.add(decision)
                        observe_literary_continuity(
                            runtime_state=runtime_state,
                            source_text=raw_draft_text,
                            translated_text=kept_text,
                            section=current_section,
                            phase="refinement",
                        )
                        if log_callback:
                            log_callback(
                                "modernize_block_rejected",
                                f"⚠️ Modernize chunk {i+1}/{total_chunks} rejected: {block_error}. Keeping source chunk."
                            )
                        if stats_callback:
                            stats_callback(progress_tracker.get_stats().to_dict())
                        _checkpoint(i, raw_draft_text, kept_text, failed=False)
                        continue

                    refined_text = restored_text

                if editorial_quality_guard:
                    decision, guard_response = await _assess_refinement_with_editorial_guard(
                        draft_text=raw_draft_text,
                        refined_text=refined_text,
                        chunk_index=i + 1,
                        section=current_section,
                        source_text=source_text,
                        source_language=prompt_options.get('_source_language', ''),
                        target_language=target_language,
                        model=model_name,
                        client=llm_client,
                        log_callback=log_callback,
                        prompt_options=prompt_options,
                    )
                    llm_response = _merge_llm_usage(llm_response, guard_response)
                    if (
                        faithful_modernize
                        and transform_fallback_mode(prompt_options) == "best_candidate"
                    ):
                        decision = soften_decision_for_modernize(
                            decision,
                            MODERNIZE_HARD_REJECT_CODES,
                        )
                    if editorial_quality_report is not None:
                        editorial_quality_report.add(decision)

                    if not decision.accepted and _quality_decision_requires_source_fallback(
                        decision,
                        prompt_options,
                    ):
                        kept_text = clean_translated_text(raw_draft_text)
                        refined_parts.append(kept_text)
                        progress_tracker.mark_completed(i, chunk_elapsed)
                        last_refined_context = kept_text
                        observe_literary_continuity(
                            runtime_state=runtime_state,
                            source_text=draft_text,
                            translated_text=kept_text,
                            section=current_section,
                            phase="refinement",
                        )
                        if log_callback:
                            reason = "; ".join(
                                issue.code for issue in decision.rejections
                            ) or "quality_guard"
                            log_callback(
                                "refinement_quality_rejected",
                                f"⚠️ Refinement for chunk {i+1}/{total_chunks} rejected by quality guard: {reason}. Keeping draft."
                        )
                        if stats_callback:
                            stats_callback(progress_tracker.get_stats().to_dict())
                        _checkpoint(i, raw_draft_text, kept_text, failed=False)
                        continue
                    if not decision.accepted and log_callback:
                        reason = "; ".join(
                            issue.code for issue in decision.rejections
                        ) or "quality_guard"
                        log_callback(
                            "refinement_quality_warn_keep_candidate",
                            f"⚠️ Refinement for chunk {i+1}/{total_chunks} had repairable quality warnings: "
                            f"{reason}. Keeping the candidate for fidelity/profile audit instead of reverting."
                        )

                # Clean only after the raw refinement has passed the guard.
                refined_text = clean_translated_text(refined_text)
                refined_text = apply_profile_glossary_corrections(
                    refined_text,
                    prompt_options,
                    source_text=source_text or raw_draft_text,
                )
                if (
                    fidelity_supervisor_enabled(prompt_options)
                    and source_text
                    and _normalize_guard_text(source_text) != _normalize_guard_text(refined_text)
                ):
                    combined_record = _get_combined_audit_record(
                        prompt_options,
                        chunk_index=i + 1,
                        phase="refinement",
                    )
                    combined_fidelity = fidelity_assessment_from_combined(
                        combined_record.get("payload", {}) if combined_record else {}
                    )
                    if combined_fidelity:
                        fidelity_decision = assess_fidelity(
                            source_text,
                            refined_text,
                            chunk_index=chunk_index_offset + i + 1,
                            phase="refinement",
                            section=current_section,
                            source_language=prompt_options.get('_source_language', ''),
                            target_language=target_language,
                            prompt_options=prompt_options,
                        )
                        combined_model = str(combined_record.get("model") or model_name)
                        fidelity_decision = apply_fidelity_audit_assessment(
                            fidelity_decision,
                            combined_fidelity,
                            model=combined_model,
                            provider=llm_provider,
                            primary_model=model_name,
                            primary_provider=llm_provider,
                        )
                        report = (prompt_options or {}).get("_fidelity_report")
                        if report is not None and hasattr(report, "add"):
                            report.add(fidelity_decision)
                        record_candidate_result(
                            prompt_options,
                            CandidateResult.from_fidelity_decision(
                                fidelity_decision,
                                text=refined_text,
                                source_text=source_text,
                                source_language=prompt_options.get('_source_language', ''),
                                target_language=target_language,
                            ),
                        )
                        if log_callback:
                            log_callback(
                                "fidelity_supervisor_decision",
                                "Fidelity supervisor reused combined audit: refinement chunk "
                                f"{i+1}/{total_chunks} -> "
                                f"{'accepted' if fidelity_decision.accepted else 'rejected'}"
                                + (
                                    f" ({fidelity_decision.judge_decision}, "
                                    f"{fidelity_decision.judge_confidence:.2f})"
                                    if fidelity_decision.judge_decision else ""
                                ),
                                data={
                                    "type": "fidelity_supervisor_decision",
                                    "accepted": fidelity_decision.accepted,
                                    "phase": "refinement",
                                    "chunk_index": chunk_index_offset + i + 1,
                                    "issues": [issue.code for issue in fidelity_decision.issues],
                                    "judge_decision": fidelity_decision.judge_decision,
                                    "confidence": fidelity_decision.judge_confidence,
                                    "combined": True,
                                },
                            )
                    else:
                        fidelity_decision, _fidelity_response = await supervise_fidelity(
                            source_text,
                            refined_text,
                            chunk_index=chunk_index_offset + i + 1,
                            phase="refinement",
                            section=current_section,
                            source_language=prompt_options.get('_source_language', ''),
                            target_language=target_language,
                            primary_model=model_name,
                            primary_provider=llm_provider,
                            client=llm_client,
                            prompt_options=prompt_options,
                            log_callback=log_callback,
                        )
                    if (
                        not fidelity_decision.accepted
                        and _fidelity_decision_requires_source_fallback(
                            fidelity_decision,
                            prompt_options,
                        )
                    ):
                        kept_text = clean_translated_text(raw_draft_text)
                        refined_parts.append(kept_text)
                        progress_tracker.mark_completed(i, chunk_elapsed)
                        last_refined_context = kept_text
                        observe_literary_continuity(
                            runtime_state=runtime_state,
                            source_text=draft_text,
                            translated_text=kept_text,
                            section=current_section,
                            phase="refinement",
                        )
                        if log_callback:
                            reason = "; ".join(
                                issue.code for issue in fidelity_decision.rejections
                            ) or "fidelity_supervisor"
                            log_callback(
                                "refinement_fidelity_rejected",
                                f"⚠️ Refinement for chunk {i+1}/{total_chunks} rejected by fidelity supervisor: {reason}. Keeping draft."
                        )
                        if stats_callback:
                            stats_callback(progress_tracker.get_stats().to_dict())
                        _checkpoint(i, raw_draft_text, kept_text, failed=False)
                        continue
                    if not fidelity_decision.accepted and log_callback:
                        reason = "; ".join(
                            issue.code for issue in fidelity_decision.rejections
                        ) or "fidelity_supervisor"
                        log_callback(
                            "refinement_fidelity_warn_keep_candidate",
                            f"⚠️ Fidelity supervisor flagged chunk {i+1}/{total_chunks}: "
                            f"{reason}. Keeping candidate for active profile audit/repair instead of reverting."
                        )

                if profile_enabled(prompt_options) and source_text:
                    profile_id = str(prompt_options.get("profile_id") or "").strip()
                    max_profile_repairs = int(
                        prompt_options.get("max_repair_rounds")
                        or prompt_options.get("profile_max_repair_rounds")
                        or 0
                    )
                    if not prompt_options.get("repair_until_pass", True):
                        max_profile_repairs = 0

                    profile_audit = None
                    best_profile_audit = None
                    best_profile_text = refined_text
                    best_profile_key = None
                    for repair_round in range(max_profile_repairs + 1):
                        try:
                            audit_response = None
                            combined_profile = None
                            combined_record = _get_combined_audit_record(
                                prompt_options,
                                chunk_index=i + 1,
                                phase="refinement",
                            )
                            if repair_round == 0 and combined_quality_audit_enabled(prompt_options):
                                if not combined_record:
                                    local_editorial_decision = assess_refinement(
                                        raw_draft_text,
                                        refined_text,
                                        chunk_index=chunk_index_offset + i + 1,
                                        section=current_section,
                                        source_text=source_text,
                                        glossary_terms=(prompt_options or {}).get("glossary_terms"),
                                        target_language=target_language,
                                        source_language=prompt_options.get('_source_language', ''),
                                        prompt_options=prompt_options,
                                    )
                                    combined_payload, audit_response = await _run_combined_quality_audit(
                                        source_text=source_text,
                                        draft_text=raw_draft_text,
                                        candidate_text=refined_text,
                                        chunk_index=i + 1,
                                        section=current_section,
                                        source_language=prompt_options.get('_source_language', ''),
                                        target_language=target_language,
                                        phase="refinement",
                                        editorial_decision=local_editorial_decision,
                                        prompt_options=prompt_options,
                                        model=model_name,
                                        client=llm_client,
                                        log_callback=log_callback,
                                    )
                                    if combined_payload:
                                        combined_record = _get_combined_audit_record(
                                            prompt_options,
                                            chunk_index=i + 1,
                                            phase="refinement",
                                        )
                                combined_profile = profile_payload_from_combined(
                                    combined_record.get("payload", {}) if combined_record else {}
                                )
                            if combined_profile:
                                try:
                                    active_profile = load_book_profile(profile_id)
                                    if (prompt_options or {}).get("profile_local_precheck_enabled") is False:
                                        local_profile_issues = []
                                    else:
                                        local_profile_issues = run_profile_precheck(
                                            source_text,
                                            refined_text,
                                            profile_id=profile_id,
                                        ).issues
                                    profile_audit = profile_audit_result_from_payload(
                                        profile_id,
                                        combined_profile,
                                        local_issues=local_profile_issues,
                                        min_score=active_profile.min_dimension_score,
                                        score_fields=score_fields_for_profile(active_profile),
                                    )
                                    if log_callback:
                                        low_scores = {
                                            key: value for key, value in profile_audit.scores.items()
                                            if value < active_profile.min_dimension_score
                                        }
                                        log_callback(
                                            "profile_audit_decision",
                                            "🔎 Active profile audit reused combined audit "
                                            f"chunk {i+1}/{total_chunks}: {profile_audit.overall_decision}"
                                            + (f" low={low_scores}" if low_scores else ""),
                                            data={
                                                "type": "profile_audit_decision",
                                                "model": (combined_record or {}).get("model") or model_name,
                                                "primary_model": model_name,
                                                "decision": profile_audit.overall_decision,
                                                "scores": profile_audit.scores,
                                                "issues": [issue.to_dict() for issue in profile_audit.issues],
                                                "combined": True,
                                            },
                                        )
                                except Exception:
                                    combined_profile = None
                            if not combined_profile:
                                profile_audit, audit_response = await _run_profile_audit(
                                    source_text,
                                    refined_text,
                                    profile_id=profile_id,
                                    prompt_options=prompt_options,
                                    model=model_name,
                                    client=llm_client,
                                    log_callback=log_callback,
                                    chunk_label=f"chunk {i+1}/{total_chunks}",
                                )
                            llm_response = _merge_llm_usage(llm_response, audit_response)
                        except Exception as audit_error:
                            if log_callback:
                                log_callback(
                                    "profile_audit_error",
                                    f"⚠️ Profile audit failed for chunk {i+1}/{total_chunks}: {audit_error}"
                                )
                            break

                        runtime_state.setdefault("profile_audit_records", []).append(
                            profile_audit.to_dict()
                        )
                        current_key = _profile_audit_quality_key(profile_audit)
                        if best_profile_key is None or current_key > best_profile_key:
                            best_profile_key = current_key
                            best_profile_audit = profile_audit
                            best_profile_text = refined_text
                        if profile_audit.overall_decision == "pass" or not profile_audit.issues:
                            break
                        if repair_round >= max_profile_repairs:
                            break

                        if log_callback:
                            log_callback(
                                "profile_repair_retry",
                                f"Profile audit flagged chunk {i+1}/{total_chunks}; repair round {repair_round + 1}/{max_profile_repairs}."
                            )
                        repaired_text, repaired_response = await _make_profile_repair_request(
                            source_text,
                            refined_text,
                            profile_audit,
                            profile_id=profile_id,
                            prompt_options=prompt_options,
                            target_language=target_language,
                            model=model_name,
                            client=llm_client,
                            log_callback=log_callback,
                            chunk_label=f"chunk {i+1}/{total_chunks}",
                        )
                        llm_response = _merge_llm_usage(llm_response, repaired_response)
                        if not repaired_text:
                            break
                        refined_text = clean_translated_text(clean_text_artifacts(repaired_text))
                        refined_text, repair_guard_issues, repair_guard_scores = _guard_generated_text(
                            refined_text,
                            phase="profile_repair",
                            style_reference=last_refined_context,
                            log_callback=log_callback,
                        )
                        if repair_guard_issues:
                            _record_text_candidate(
                                prompt_options,
                                text=refined_text,
                                source_text=source_text or raw_draft_text,
                                phase="profile_repair",
                                chunk_index=chunk_index_offset + i + 1,
                                section=current_section,
                                source_language=prompt_options.get('_source_language', ''),
                                target_language=target_language,
                                issues=repair_guard_issues,
                                extra_scores=repair_guard_scores,
                                response=repaired_response,
                                decision="repair",
                                model=model_name,
                                source="profile_repair_guard",
                                log_callback=log_callback,
                            )
                        if any(issue.severity == "reject" for issue in repair_guard_issues):
                            break
                        refined_text = apply_profile_glossary_corrections(
                            refined_text,
                            prompt_options,
                            source_text=source_text or raw_draft_text,
                        )
                        gate_rejections = _target_language_gate_rejections(
                            source_text or raw_draft_text,
                            refined_text,
                            source_language=prompt_options.get('_source_language', ''),
                            target_language=target_language,
                            phase="profile_repair",
                            prompt_options=prompt_options,
                        )
                        if gate_rejections:
                            _record_text_candidate(
                                prompt_options,
                                text=refined_text,
                                source_text=source_text or raw_draft_text,
                                phase="profile_repair",
                                chunk_index=chunk_index_offset + i + 1,
                                section=current_section,
                                source_language=prompt_options.get('_source_language', ''),
                                target_language=target_language,
                                issues=gate_rejections,
                                response=repaired_response,
                                decision="retry",
                                model=model_name,
                                source="profile_repair",
                                log_callback=log_callback,
                            )
                            _log_target_language_gate_rejection(
                                log_callback,
                                event="target_language_gate_profile_repair_rejected",
                                phase="profile repair",
                                issues=gate_rejections,
                            )
                            break

                    if best_profile_audit is not None and profile_audit is not None:
                        if best_profile_audit is not profile_audit:
                            refined_text = best_profile_text
                            profile_audit = best_profile_audit
                            if log_callback:
                                log_callback(
                                    "profile_repair_best_candidate",
                                    f"Profile repair for chunk {i+1}/{total_chunks} kept best audited candidate "
                                    f"({profile_audit.overall_decision}) instead of the last repair."
                                )

                    if profile_audit is not None:
                        _append_profile_audit_to_report(
                            editorial_quality_report,
                            raw_draft_text=raw_draft_text,
                            refined_text=refined_text,
                            source_text=source_text,
                            audit_result=profile_audit,
                            chunk_index=chunk_index_offset + i + 1,
                            section=current_section,
                            target_language=target_language,
                            prompt_options=prompt_options,
                        )
                        if _profile_audit_failure_should_abort(
                            profile_audit,
                            prompt_options,
                        ):
                            reason = "; ".join(
                                issue.issue_type for issue in profile_audit.issues
                            ) or profile_audit.summary or "profile_audit_fail"
                            if log_callback:
                                log_callback(
                                    "profile_audit_abort",
                                    f"⛔ Chunk {i+1}/{total_chunks} failed active book profile audit "
                                    f"after repair attempts: {reason}. Aborting job."
                                )
                            progress_tracker.mark_failed(i)
                            _checkpoint(i, raw_draft_text, refined_text, failed=True)
                            raise RuntimeError(
                                "Active book profile audit failed after repair attempts "
                                f"for chunk {i+1}/{total_chunks}: {reason}"
                            )
                        if _profile_audit_requires_source_fallback(profile_audit):
                            kept_text = clean_translated_text(raw_draft_text)
                            refined_parts.append(kept_text)
                            progress_tracker.mark_completed(i, chunk_elapsed)
                            last_refined_context = kept_text
                            observe_literary_continuity(
                                runtime_state=runtime_state,
                                source_text=draft_text,
                                translated_text=kept_text,
                                section=current_section,
                                phase="refinement",
                            )
                            if log_callback:
                                reason = "; ".join(
                                    issue.issue_type for issue in profile_audit.issues
                                    if issue.severity == "high"
                                ) or "profile_audit"
                                log_callback(
                                    "profile_audit_rejected",
                                    f"⚠️ Chunk {i+1}/{total_chunks} rejected by active book profile audit: {reason}. Keeping draft."
                                )
                            if stats_callback:
                                stats_callback(progress_tracker.get_stats().to_dict())
                            _checkpoint(i, raw_draft_text, kept_text, failed=False)
                            continue
                        if profile_audit.overall_decision != "pass" and log_callback:
                            repairable = "; ".join(
                                issue.issue_type for issue in profile_audit.issues
                                if issue.issue_type not in CRITICAL_ISSUE_TYPES
                            ) or profile_audit.overall_decision
                            log_callback(
                                "profile_audit_repairable_kept",
                                f"⚠️ Chunk {i+1}/{total_chunks} still has repairable profile issues "
                                f"({repairable}); keeping best audited candidate instead of reverting."
                            )
                gate_rejections = _target_language_gate_rejections(
                    source_text or raw_draft_text,
                    refined_text,
                    source_language=prompt_options.get('_source_language', ''),
                    target_language=target_language,
                    phase="refinement",
                    prompt_options=prompt_options,
                )
                if gate_rejections:
                    _record_text_candidate(
                        prompt_options,
                        text=refined_text,
                        source_text=source_text or raw_draft_text,
                        phase="refinement",
                        chunk_index=chunk_index_offset + i + 1,
                        section=current_section,
                        source_language=prompt_options.get('_source_language', ''),
                        target_language=target_language,
                        issues=gate_rejections,
                        response=llm_response,
                        decision="retry",
                        model=model_name,
                        source="refinement",
                        log_callback=log_callback,
                    )
                    _log_target_language_gate_rejection(
                        log_callback,
                        event="target_language_gate_refinement_rejected",
                        phase="refinement",
                        issues=gate_rejections,
                    )
                    refined_text = clean_translated_text(raw_draft_text)
                refined_parts.append(refined_text)
                progress_tracker.mark_completed(i, chunk_elapsed)
                observe_literary_continuity(
                    runtime_state=runtime_state,
                    source_text=raw_draft_text,
                    translated_text=refined_text,
                    section=current_section,
                    phase="refinement",
                )

                # Update context for next chunk
                words = refined_text.split()
                if len(words) > 25:
                    last_refined_context = " ".join(words[-25:])
                else:
                    last_refined_context = refined_text
                _checkpoint(i, raw_draft_text, refined_text, failed=False)
            else:
                # Keep original translation if refinement fails
                if log_callback:
                    log_callback("refinement_chunk_failed",
                        f"Refinement failed for chunk {i+1}, keeping original translation")
                refined_parts.append(raw_draft_text)
                progress_tracker.mark_failed(i)
                last_refined_context = ""
                observe_literary_continuity(
                    runtime_state=runtime_state,
                    source_text=raw_draft_text,
                    translated_text=raw_draft_text,
                    section=current_section,
                    phase="refinement",
                )
                _checkpoint(i, raw_draft_text, raw_draft_text, failed=True)

            if stats_callback:
                stats_callback(progress_tracker.get_stats().to_dict())

    finally:
        if llm_client:
            await llm_client.close()

    stats = progress_tracker.get_stats()
    if log_callback:
        log_callback("refinement_complete",
            f"✨ Refinement complete: {stats.completed_chunks} refined, {stats.failed_chunks} kept original")

    return refined_parts


# Subtitle translation functions moved to subtitle_translator.py
