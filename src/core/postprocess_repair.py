"""Post-run repair queue for suspicious text-first chunks.

The normal chunk pipeline must keep moving on long books. This module adds a
late, targeted pass for chunks that local heuristics can prove are suspicious:
near-identical modernizations, leaked wrapper tags, mojibake, profile detector
hits, extreme length drift, or other clear local errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence

from src.config import REQUEST_TIMEOUT, TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.core.book_profiles import build_profile_glossary_block, profile_enabled
from src.core.book_profiles.audit import run_profile_precheck
from src.core.book_profiles.detectors import (
    active_profile_from_options,
    has_profile_modernization_signal,
    lost_profile_proper_nouns,
)
from src.core.llm import RateLimitError
from src.core.llm.request_deadline import await_llm_call
from src.core.llm_client import LLMClient, create_llm_client
from src.core.llm_output_guard import guard_llm_output
from src.core.locale_quality import (
    build_spanish_modernization_repair_instructions,
    collect_spanish_modernization_residue_examples,
    count_spanish_modernization_residue,
)
from src.core.post_processor import clean_translated_text
from src.core.text_transform import is_faithful_modernize
from src.utils.text_encoding import clean_text_artifacts


_WRAPPER_ARTIFACT_RE = re.compile(
    r"(?:<|&lt;)\s*/?\s*(?:TRANSLATION|SOURCE|DRAFT|PROFILE_AUDIT_JSON)"
    r"(?:ATION)*\s*(?:>|&gt;)",
    re.I,
)
_MOJIBAKE_RE = re.compile(r"Ã.|Â.|�|[■□▮▯]")
_INDEX_NOISE_RE = re.compile(
    r"\.{8,}|(?:\.\s*){8,}|"
    r"(?:^|\n)\s*(?:CAP[IÍ]TULO|Cap[ií]tulo|PARTE|Parte|PR[OÓ]LOGO|Pr[oó]logo|DEDICATORIA|Dedicatoria)"
    r"\b[^\n]{80,}[ \t]+(?:CAP[IÍ]TULO|Cap[ií]tulo|PARTE|Parte|PR[OÓ]LOGO|Pr[oó]logo|DEDICATORIA|Dedicatoria)\b",
    re.M,
)
_PENINSULAR_OR_ARCHAIC_RE = re.compile(
    r"\b(?:vosotros|vosotras|sois|est[aá]is|hab[eé]is|habr[eé]is|"
    r"vuestro|vuestra|vuestros|vuestras|deteneos|rendeos|callaos|"
    r"marchaos|apresuraos|guardaos|entregaos|poneos|haceos|quedaos|"
    r"acercaos|alejaos|quitaos|sentaos|levantaos|preparaos|armaos|"
    r"largaos)\b",
    re.I,
)
_OS_PRONOUN_RE = re.compile(r"\bos\b", re.I)
_ARCHAIC_RE = re.compile(
    r"\b(?:non|fuy[a-záéíóúñ]*|facer[a-záéíóúñ]*|fasta|maguer|"
    r"home|fazañ[a-záéíóúñ]*|joeces|cerbelo|vues[ao]|"
    r"receb[a-záéíóúñ]*|fermos[a-záéíóúñ]*|mesm[a-záéíóúñ]*)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"\b\d+\b|\b[IVXLCDM]{1,8}\b")


def _profile_glossary_purpose(prompt_options: Mapping[str, Any]) -> str:
    if (prompt_options or {}).get("text_transform_mode"):
        return "transformation"
    return "refinement"


@dataclass(frozen=True)
class RepairIssue:
    code: str
    severity: str
    reason: str = ""


@dataclass
class RepairCandidate:
    prompt_id: str
    text: str
    issues: list[RepairIssue] = field(default_factory=list)
    score: float = 0.0
    failure: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.text.strip()) and not any(
            issue.severity in {"critical", "high"} for issue in self.issues
        )


@dataclass
class PostprocessRepairResult:
    parts: list[str]
    flagged_indices: list[int]
    repaired_indices: list[int]
    kept_indices: list[int]
    failed_indices: list[int]
    attempts: dict[int, list[RepairCandidate]] = field(default_factory=dict)


def postprocess_repair_enabled(prompt_options: Optional[Mapping[str, Any]]) -> bool:
    """Return whether the late repair queue should run for this job."""
    options = prompt_options or {}
    if options.get("postprocess_repair_enabled") is False:
        return False
    if options.get("postprocess_repair_enabled") is True:
        return True
    return bool(
        is_faithful_modernize(options)
        and (profile_enabled(options) or str(options.get("profile_id") or "").strip())
    )


def detect_repair_issues(
    source_text: str,
    candidate_text: str,
    *,
    chunk_index: int,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> list[RepairIssue]:
    """Detect local, token-free reasons a chunk needs post-run repair."""
    del chunk_index
    options = prompt_options or {}
    source = source_text or ""
    candidate = candidate_text or ""
    issues: list[RepairIssue] = []
    active_profile = active_profile_from_options(options)

    if not candidate.strip():
        issues.append(RepairIssue("empty_candidate", "critical", "No output text."))
        return issues

    if _WRAPPER_ARTIFACT_RE.search(candidate):
        issues.append(RepairIssue("wrapper_artifact", "critical", "Internal LLM wrapper tag leaked."))
    if _MOJIBAKE_RE.search(candidate):
        issues.append(RepairIssue("mojibake_or_artifact", "critical", "Encoding or artifact glyphs remain."))
    if _INDEX_NOISE_RE.search(candidate):
        issues.append(RepairIssue("structure_noise", "high", "PDF/index structure remains flattened or noisy."))
    if _PENINSULAR_OR_ARCHAIC_RE.search(candidate):
        issues.append(RepairIssue("peninsular_or_archaic_form", "medium", "Residual archaic/Peninsular form."))
    if _OS_PRONOUN_RE.search(candidate):
        issues.append(RepairIssue("os_pronoun", "medium", "Residual second-person plural pronoun."))
    if _ARCHAIC_RE.search(candidate):
        issues.append(RepairIssue("lexical_archaism", "medium", "Residual lexical archaism."))
    if is_faithful_modernize(options):
        residue_counts = count_spanish_modernization_residue(candidate)
        residue_total = sum(residue_counts.values())
        if residue_total >= 6:
            issues.append(
                RepairIssue(
                    "modernization_residue",
                    "high",
                    f"Too many old-Spanish forms remain: {residue_counts}",
                )
            )
        elif residue_total:
            issues.append(
                RepairIssue(
                    "modernization_residue",
                    "medium",
                    f"Old-Spanish forms remain: {residue_counts}",
                )
            )

    if source.strip():
        ratio = len(candidate) / max(1, len(source))
        if ratio < 0.72 or ratio > 1.55:
            issues.append(RepairIssue("length_extreme", "critical", f"Length ratio {ratio:.2f}."))

        similarity = difflib.SequenceMatcher(None, source, candidate).ratio()
        source_residue = count_spanish_modernization_residue(source)
        candidate_residue = count_spanish_modernization_residue(candidate)
        source_has_modernizable_forms = bool(
            _ARCHAIC_RE.search(source)
            or _PENINSULAR_OR_ARCHAIC_RE.search(source)
            or _OS_PRONOUN_RE.search(source)
            or source_residue
            or has_profile_modernization_signal(source, profile=active_profile)
        )
        candidate_has_residual_forms = bool(
            _ARCHAIC_RE.search(candidate)
            or _PENINSULAR_OR_ARCHAIC_RE.search(candidate)
            or _OS_PRONOUN_RE.search(candidate)
            or candidate_residue
        )
        if (
            is_faithful_modernize(options)
            and len(source) > 350
            and similarity > 0.965
            and source_has_modernizable_forms
            and candidate_has_residual_forms
        ):
            issues.append(
                RepairIssue(
                    "near_identical_modernization",
                    "high",
                    f"Candidate is too close to source for a modernization ({similarity:.3f}).",
                )
            )

        lost_names = set(
            lost_profile_proper_nouns(
                source,
                candidate,
                profile=active_profile,
            )
        )
        if lost_names:
            issues.append(RepairIssue("lost_proper_name", "critical", ", ".join(sorted(lost_names))))
        lost_numbers = set(_NUMBER_RE.findall(source)) - set(_NUMBER_RE.findall(candidate))
        if len(lost_numbers) >= 3:
            issues.append(
                RepairIssue("lost_numbers", "high", ", ".join(sorted(lost_numbers)[:8]))
            )

    profile_id = str(options.get("profile_id") or "").strip()
    if profile_id:
        try:
            precheck = run_profile_precheck(source, candidate, profile_id=profile_id)
            for issue in precheck.issues:
                severity = "high" if issue.severity == "high" else "medium"
                issues.append(
                    RepairIssue(
                        f"profile_{issue.issue_type}",
                        severity,
                        issue.reason or issue.suggested_fix,
                    )
                )
        except Exception:
            pass

    return _dedupe_issues(issues)


async def repair_flagged_chunks(
    *,
    refined_parts: Sequence[str],
    structured_chunks: Sequence[Mapping[str, Any]],
    target_language: str,
    model_name: str,
    api_endpoint: str,
    llm_provider: str,
    prompt_options: Mapping[str, Any],
    log_callback: Optional[Callable[[str, str], None]] = None,
    check_interruption_callback: Optional[Callable[[], bool]] = None,
    checkpoint_callback: Optional[Callable[[int, str, str], None]] = None,
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: Optional[int] = None,
    llm_client: Optional[LLMClient] = None,
) -> PostprocessRepairResult:
    """Repair suspicious chunks after the main pass has completed."""
    options = dict(prompt_options or {})
    parts = list(refined_parts)
    flagged: list[int] = []
    initial_issues: dict[int, list[RepairIssue]] = {}
    for idx, candidate in enumerate(parts):
        source = _chunk_source(structured_chunks, idx, candidate)
        issues = detect_repair_issues(
            source,
            candidate,
            chunk_index=idx,
            prompt_options=options,
        )
        if issues:
            flagged.append(idx)
            initial_issues[idx] = issues

    if not flagged:
        if log_callback:
            log_callback("postprocess_repair_clean", "🧹 No chunks needed post-run repair.")
        return PostprocessRepairResult(parts, [], [], [], [])

    if log_callback:
        log_callback(
            "postprocess_repair_start",
            f"🧹 Post-run repair queued {len(flagged)} suspicious chunk(s) for best-of-3 repair.",
        )

    owns_client = llm_client is None
    client = llm_client or create_llm_client(
        llm_provider,
        gemini_api_key,
        api_endpoint,
        model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        context_window=context_window,
        log_callback=log_callback,
    )
    if client is None:
        return PostprocessRepairResult(parts, flagged, [], flagged[:], flagged[:])

    repaired: list[int] = []
    kept: list[int] = []
    failed: list[int] = []
    attempts_by_index: dict[int, list[RepairCandidate]] = {}

    try:
        for idx in flagged:
            if check_interruption_callback and check_interruption_callback():
                if log_callback:
                    log_callback(
                        "postprocess_repair_interrupted",
                        f"Post-run repair interrupted before chunk {idx + 1}.",
                    )
                kept.extend(i for i in flagged if i not in repaired and i not in kept)
                break

            source = _chunk_source(structured_chunks, idx, parts[idx])
            current = parts[idx]
            issues = initial_issues[idx]
            prompt_variants = _build_prompt_variants(
                source_text=source,
                current_text=current,
                context_before=_context_before(structured_chunks, parts, idx),
                context_after=_context_after(structured_chunks, parts, idx),
                target_language=target_language,
                prompt_options=options,
                issues=issues,
                chunk_index=idx,
            )
            base_candidate = RepairCandidate(
                prompt_id="current",
                text=current,
                issues=issues,
                score=_candidate_score(source, current, issues, options),
            )
            best = base_candidate
            attempts: list[RepairCandidate] = []

            for prompt_id, system_prompt, user_prompt in prompt_variants:
                if check_interruption_callback and check_interruption_callback():
                    break
                try:
                    response = await await_llm_call(
                        client.make_request,
                        user_prompt,
                        provider=client,
                        model=model_name,
                        request_timeout=REQUEST_TIMEOUT,
                        system_prompt=system_prompt,
                    )
                except RateLimitError:
                    raise
                except Exception as exc:
                    attempts.append(RepairCandidate(prompt_id, "", failure=str(exc)))
                    continue

                extracted = None
                if response and response.content:
                    extracted = client.extract_translation(response.content)
                if not extracted:
                    attempts.append(RepairCandidate(prompt_id, "", failure="missing_translation_tags"))
                    continue

                guarded = guard_llm_output(
                    extracted,
                    phase=f"postprocess_repair_{prompt_id}",
                    style_reference=current,
                )
                text = clean_translated_text(clean_text_artifacts(guarded.text))
                candidate_issues = detect_repair_issues(
                    source,
                    text,
                    chunk_index=idx,
                    prompt_options=options,
                )
                for issue in guarded.issues:
                    severity = "critical" if issue.severity == "reject" else "medium"
                    candidate_issues.append(
                        RepairIssue(
                            f"llm_guard_{issue.code}",
                            severity,
                            issue.detail or issue.message,
                        )
                    )
                candidate = RepairCandidate(
                    prompt_id=prompt_id,
                    text=text,
                    issues=candidate_issues,
                    score=_candidate_score(source, text, candidate_issues, options),
                )
                attempts.append(candidate)
                if candidate.score > best.score:
                    best = candidate
                if candidate.passed and candidate.score >= base_candidate.score:
                    best = candidate
                    break

            attempts_by_index[idx] = attempts
            if best is not base_candidate and best.text.strip():
                parts[idx] = best.text
                repaired.append(idx)
                if checkpoint_callback:
                    checkpoint_callback(idx, source, best.text)
                if log_callback:
                    log_callback(
                        "postprocess_repair_applied",
                        f"🧹 Repaired chunk {idx + 1} with {best.prompt_id} "
                        f"(score {base_candidate.score:.1f} → {best.score:.1f}).",
                    )
            else:
                kept.append(idx)
                if any(attempt.failure for attempt in attempts):
                    failed.append(idx)
                if log_callback:
                    log_callback(
                        "postprocess_repair_kept",
                        f"⚠️ Post-run repair kept chunk {idx + 1}; no safer candidate improved it.",
                    )
    finally:
        if owns_client:
            await client.close()

    return PostprocessRepairResult(parts, flagged, repaired, kept, failed, attempts_by_index)


def _dedupe_issues(issues: list[RepairIssue]) -> list[RepairIssue]:
    seen: set[str] = set()
    result: list[RepairIssue] = []
    for issue in issues:
        key = f"{issue.code}:{issue.severity}"
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def _chunk_source(
    structured_chunks: Sequence[Mapping[str, Any]],
    idx: int,
    fallback: str,
) -> str:
    if idx < len(structured_chunks):
        chunk = structured_chunks[idx]
        for key in ("_source_text", "main_content"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return fallback


def _context_before(
    structured_chunks: Sequence[Mapping[str, Any]],
    parts: Sequence[str],
    idx: int,
) -> str:
    local = ""
    if idx < len(structured_chunks):
        value = structured_chunks[idx].get("context_before")
        if isinstance(value, str):
            local = value
    previous = parts[idx - 1] if idx > 0 else ""
    return _trim_context("\n\n".join(part for part in (local, previous) if part))


def _context_after(
    structured_chunks: Sequence[Mapping[str, Any]],
    parts: Sequence[str],
    idx: int,
) -> str:
    local = ""
    if idx < len(structured_chunks):
        value = structured_chunks[idx].get("context_after")
        if isinstance(value, str):
            local = value
    following = parts[idx + 1] if idx + 1 < len(parts) else ""
    return _trim_context("\n\n".join(part for part in (local, following) if part))


def _trim_context(text: str, limit: int = 1400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = text[: limit // 2].rstrip()
    tail = text[-limit // 2 :].lstrip()
    return f"{head}\n[...]\n{tail}"


def _candidate_score(
    source_text: str,
    candidate_text: str,
    issues: Sequence[RepairIssue],
    prompt_options: Mapping[str, Any],
) -> float:
    score = 100.0
    penalties = {"critical": 45.0, "high": 25.0, "medium": 8.0, "low": 3.0}
    for issue in issues:
        score -= penalties.get(issue.severity, 8.0)
    if source_text.strip() and candidate_text.strip():
        ratio = len(candidate_text) / max(1, len(source_text))
        score -= min(25.0, abs(1.0 - ratio) * 35.0)
        similarity = difflib.SequenceMatcher(None, source_text, candidate_text).ratio()
        source_residue = count_spanish_modernization_residue(source_text)
        candidate_residue = count_spanish_modernization_residue(candidate_text)
        if is_faithful_modernize(prompt_options) and (_ARCHAIC_RE.search(source_text) or source_residue):
            if similarity > 0.95:
                score -= 20.0
            elif similarity < 0.12:
                score -= 8.0
            if candidate_residue:
                score -= min(35.0, sum(candidate_residue.values()) * 2.0)
            if source_residue and sum(candidate_residue.values()) >= sum(source_residue.values()) * 0.75:
                score -= 20.0
    return max(0.0, score)


def _build_prompt_variants(
    *,
    source_text: str,
    current_text: str,
    context_before: str,
    context_after: str,
    target_language: str,
    prompt_options: Mapping[str, Any],
    issues: Sequence[RepairIssue],
    chunk_index: int,
) -> list[tuple[str, str, str]]:
    profile_id = str(prompt_options.get("profile_id") or "active_profile").strip()
    active_profile = active_profile_from_options(prompt_options)
    glossary = (
        build_profile_glossary_block(
            source_text,
            prompt_options,
            purpose=_profile_glossary_purpose(prompt_options),
        )
        if profile_enabled(prompt_options)
        else ""
    )
    profile_policy = ""
    if active_profile is not None:
        policy_parts = [
            active_profile.policy_text,
            active_profile.prompt_texts.get("modernize", ""),
            active_profile.prompt_texts.get("repair", ""),
        ]
        profile_policy = _trim_context(
            "\n\n".join(part.strip() for part in policy_parts if part and part.strip()),
            limit=2400,
        )
    issue_payload = json.dumps(
        [{"code": i.code, "severity": i.severity, "reason": i.reason} for i in issues],
        ensure_ascii=False,
    )
    modernization_counts = count_spanish_modernization_residue(current_text)
    modernization_examples = collect_spanish_modernization_residue_examples(current_text)
    modernization_instructions = ""
    if is_faithful_modernize(prompt_options) and modernization_counts:
        modernization_instructions = (
            "\n\n# Modernization residue instructions\n"
            + build_spanish_modernization_repair_instructions(
                modernization_counts,
                examples=modernization_examples,
            )
        )
    common_payload = f"""
# Chunk
{chunk_index + 1}

# Active profile
{profile_id}

# Profile editorial policy
{profile_policy or '(No profile policy text was available; follow the local issues and approved glossary only.)'}

# Target language and locale
{target_language}; {prompt_options.get('target_locale') or ''}

# Local issues to fix
{issue_payload}

# Issue-local repair contract
Fix only the sentences, lines, table cells, or short spans that are needed to
resolve the listed issues. Keep every unaffected sentence unchanged unless a
minimal boundary edit is necessary for grammar after the local fix. Do not
reroll the whole chunk, change correct terminology, reorder paragraphs, or
polish already-good prose just because this is a repair pass.

# Approved glossary excerpt
{glossary or '(No profile glossary entries matched this chunk.)'}
{modernization_instructions}

# Previous context
{context_before or '(none)'}

# Source chunk
{source_text}

# Current flawed candidate
{current_text}

# Following context
{context_after or '(none)'}
""".strip()
    output_rule = (
        f"Return only the repaired text between {TRANSLATE_TAG_IN} and "
        f"{TRANSLATE_TAG_OUT}. No notes, no markdown fences, no explanations."
    )
    variants = [
        (
            "postprocess_source_first",
            "You are a source-faithful literary modernization repair editor.",
            f"""
Repair this failed chunk from the source, not from the flawed candidate.

Goal: a faithful transformation for the active profile and target locale.
Preserve every fact, name, relation, number, scene, order and voice. Do not
summarize, omit, censor, explain, expand, or flatten the style.

The current candidate is evidence of what went wrong. Use it only to avoid
repeating the same failure.

{common_payload}

{output_rule}
""".strip(),
        ),
        (
            "postprocess_contextual_recast",
            "You are a continuity editor repairing a single chunk inside a long book.",
            f"""
Rewrite only the source chunk so it fits smoothly between the provided previous
and following context. The repair must keep the same editorial objective as the
active profile, but use a different strategy: recast the syntax directly from
the source instead of polishing the old candidate.

Resolve all listed local issues. If the source uses archaic treatment, choose a
natural contemporary treatment by context and by the active profile. Keep
authorial and character voice without preserving obsolete surface forms unless
the profile explicitly asks for them.

{common_payload}

{output_rule}
""".strip(),
        ),
        (
            "postprocess_voice_and_fidelity",
            "You are the final literary editor for a profile-scoped modernization.",
            f"""
Produce the safest corrected version of this chunk. Balance two priorities:
1. absolute content fidelity to the source;
2. natural literary syntax and diction for the active target locale.

Do not preserve a bad candidate just because it is close to the source. Do not
make it generic or school-like. Keep the authorial effects, tone, register,
relationships, and voice differences described by the active profile. Remove
leaked tags, OCR/encoding artifacts, residual obsolete forms, and profile
violations unless the active profile explicitly requires preserving them.

{common_payload}

{output_rule}
""".strip(),
        ),
    ]
    return variants
