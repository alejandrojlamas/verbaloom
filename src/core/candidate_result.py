"""Unified quality contract for generated chunk candidates.

Existing pipeline stages still return their historical values for backwards
compatibility, but every guard can also describe its output as a CandidateResult.
That gives translation, refinement, audit, repair, and resume code one compact
shape for deciding whether to accept, retry, repair, fallback, pause, or fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import hashlib
import re
from typing import Any, Iterable, Mapping, Optional


_SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "latin": re.compile(r"[A-Za-z\u00c0-\u024f]"),
    "greek": re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]"),
    "cyrillic": re.compile(r"[\u0400-\u052f]"),
    "cjk": re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "hangul": re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]"),
    "hebrew": re.compile(r"[\u0590-\u05ff]"),
    "devanagari": re.compile(r"[\u0900-\u097f]"),
}
_LANGUAGE_SCRIPT_ALIASES = {
    "spanish": "latin",
    "espanol": "latin",
    "español": "latin",
    "es": "latin",
    "english": "latin",
    "en": "latin",
    "french": "latin",
    "fr": "latin",
    "german": "latin",
    "de": "latin",
    "italian": "latin",
    "it": "latin",
    "portuguese": "latin",
    "pt": "latin",
    "czech": "latin",
    "cs": "latin",
    "greek": "greek",
    "griego": "greek",
    "el": "greek",
    "russian": "cyrillic",
    "ru": "cyrillic",
    "ukrainian": "cyrillic",
    "uk": "cyrillic",
    "chinese": "cjk",
    "zh": "cjk",
    "japanese": "cjk",
    "ja": "cjk",
    "korean": "hangul",
    "ko": "hangul",
    "arabic": "arabic",
    "ar": "arabic",
    "hebrew": "hebrew",
    "he": "hebrew",
    "hindi": "devanagari",
    "hi": "devanagari",
}
_SPANISH_MARKERS = re.compile(
    r"\b(?:el|la|los|las|un|una|de|del|que|en|para|con|por|se|no|su|al|"
    r"es|son|era|fue|hab[ií]a|pero|pues|cuando|entonces|tambi[eé]n|"
    r"m[aá]s|como|desde|hasta|sobre)\b",
    re.IGNORECASE,
)
_ENGLISH_MARKERS = re.compile(
    r"\b(?:the|and|of|to|in|that|is|was|for|with|as|on|by|from|this|it|"
    r"be|are|were|or|an|at|which|not|have|has|had|but|they|their)\b",
    re.IGNORECASE,
)
_FRENCH_MARKERS = re.compile(
    r"\b(?:le|la|les|des|du|de|et|que|qui|dans|pour|avec|sur|est|sont|"
    r"une|un|ce|cette|par|pas|plus|mais|comme)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[\wÁÉÍÓÚÜÑáéíóúüñ'-]+", re.UNICODE)
_RETRY_CODES = {
    "target_language_missing",
    "target_script_mismatch",
    "untranslated_source",
    "translation_extraction_failed",
    "missing_translation_tags",
}
_FAIL_CODES = {
    "placeholder_mismatch",
    "modernize_block_marker_mismatch",
    "epub_extraction_critical_fail",
}
_DECISIONS = {
    "accepted",
    "accepted_with_warnings",
    "retry",
    "repair",
    "fallback",
    "pause",
    "fail",
}
_SIMILARITY_COMPARE_CHARS = 8000


@dataclass(frozen=True)
class CandidateIssue:
    code: str
    severity: str = "warning"
    message: str = ""
    detail: str = ""
    source: str = ""

    @classmethod
    def from_any(cls, value: Any, *, source: str = "") -> "CandidateIssue":
        if isinstance(value, CandidateIssue):
            return cls(
                code=value.code,
                severity=normalize_label(value.severity) or "warning",
                message=value.message,
                detail=value.detail,
                source=value.source or source,
            )
        if isinstance(value, Mapping):
            return cls(
                code=str(value.get("code") or value.get("issue_type") or value.get("type") or "issue"),
                severity=normalize_label(value.get("severity") or "warning") or "warning",
                message=str(value.get("message") or value.get("reason") or ""),
                detail=str(value.get("detail") or value.get("suggested_fix") or ""),
                source=str(value.get("source") or source),
            )
        return cls(
            code=str(getattr(value, "code", None) or getattr(value, "issue_type", None) or "issue"),
            severity=normalize_label(getattr(value, "severity", None) or "warning") or "warning",
            message=str(getattr(value, "message", None) or getattr(value, "reason", None) or ""),
            detail=str(getattr(value, "detail", None) or getattr(value, "suggested_fix", None) or ""),
            source=source,
        )

    def to_dict(self) -> dict[str, str]:
        data = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.detail:
            data["detail"] = self.detail
        if self.source:
            data["source"] = self.source
        return data


@dataclass(frozen=True)
class TokenCost:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0

    @classmethod
    def from_response(cls, response: Any) -> "TokenCost":
        if response is None:
            return cls()
        prompt = int(getattr(response, "prompt_tokens", 0) or 0)
        completion = int(getattr(response, "completion_tokens", 0) or 0)
        total = int(getattr(response, "context_used", 0) or 0) or prompt + completion
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            prompt_cache_hit_tokens=int(getattr(response, "prompt_cache_hit_tokens", 0) or 0),
            prompt_cache_miss_tokens=int(getattr(response, "prompt_cache_miss_tokens", 0) or 0),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cache_hit_tokens": self.prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self.prompt_cache_miss_tokens,
        }


@dataclass(frozen=True)
class RepairPlan:
    action: str = "none"
    reasons: tuple[str, ...] = ()
    max_rounds: int = 0
    prompt_variant: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "action": self.action,
            "reasons": list(self.reasons),
            "max_rounds": self.max_rounds,
        }
        if self.prompt_variant:
            data["prompt_variant"] = self.prompt_variant
        if self.notes:
            data["notes"] = self.notes
        return data


@dataclass
class CandidateResult:
    text: str
    phase: str
    chunk_index: int = 0
    section: str = ""
    source_text: str = ""
    source_language: str = ""
    target_language: str = ""
    detected_language: str = ""
    dominant_script: str = ""
    target_script: str = ""
    target_script_ratio: float = 0.0
    script_ratios: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)
    issues: list[CandidateIssue] = field(default_factory=list)
    token_cost: TokenCost = field(default_factory=TokenCost)
    decision: str = "accepted"
    repair_plan: RepairPlan = field(default_factory=RepairPlan)
    model: str = ""
    provider: str = ""
    source: str = "candidate"

    @classmethod
    def build(
        cls,
        text: str,
        *,
        source_text: str = "",
        phase: str,
        chunk_index: int = 0,
        section: str = "",
        source_language: str = "",
        target_language: str = "",
        issues: Iterable[Any] = (),
        accepted: Optional[bool] = None,
        response: Any = None,
        decision: str = "",
        model: str = "",
        provider: str = "",
        source: str = "candidate",
        extra_scores: Optional[Mapping[str, float]] = None,
    ) -> "CandidateResult":
        normalized_issues = [CandidateIssue.from_any(issue, source=source) for issue in issues]
        script_ratios = script_profile(text)
        dominant_script, _dominant_ratio = dominant_script_ratio(script_ratios)
        target_script = script_for_language(target_language)
        target_script_ratio = script_ratios.get(target_script, 0.0) if target_script else 0.0
        scores = build_scores(source_text, text, target_script_ratio=target_script_ratio)
        if extra_scores:
            for key, value in extra_scores.items():
                try:
                    scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
        is_accepted = bool(accepted) if accepted is not None else not any(
            issue.severity == "reject" for issue in normalized_issues
        )
        chosen_decision = normalize_decision(decision)
        if not chosen_decision or _decision_conflicts_with_issues(chosen_decision, normalized_issues):
            chosen_decision = decide_candidate(normalized_issues, accepted=is_accepted)
        repair_plan = build_repair_plan(normalized_issues, decision=chosen_decision, phase=phase)
        return cls(
            text=text or "",
            source_text=source_text or "",
            phase=phase,
            chunk_index=int(chunk_index or 0),
            section=section or "",
            source_language=source_language or "",
            target_language=target_language or "",
            detected_language=detect_language(text),
            dominant_script=dominant_script,
            target_script=target_script,
            target_script_ratio=target_script_ratio,
            script_ratios=script_ratios,
            scores=scores,
            issues=normalized_issues,
            token_cost=TokenCost.from_response(response),
            decision=chosen_decision,
            repair_plan=repair_plan,
            model=model or str(getattr(response, "model", "") or ""),
            provider=provider,
            source=source,
        )

    @classmethod
    def from_quality_decision(
        cls,
        decision: Any,
        *,
        text: str,
        source_text: str = "",
        target_language: str = "",
        source_language: str = "",
        response: Any = None,
    ) -> "CandidateResult":
        return cls.build(
            text,
            source_text=source_text or str(getattr(decision, "source_snippet", "") or ""),
            phase="editorial_guard",
            chunk_index=int(getattr(decision, "chunk_index", 0) or 0),
            section=str(getattr(decision, "section", "") or ""),
            source_language=source_language,
            target_language=target_language,
            issues=list(getattr(decision, "issues", []) or []),
            accepted=bool(getattr(decision, "accepted", False)),
            response=response,
            model=str(getattr(decision, "judge_model", "") or ""),
            source="editorial_guard",
        )

    @classmethod
    def from_fidelity_decision(
        cls,
        decision: Any,
        *,
        text: str = "",
        source_text: str = "",
        source_language: str = "",
        target_language: str = "",
        response: Any = None,
    ) -> "CandidateResult":
        return cls.build(
            text or str(getattr(decision, "candidate_snippet", "") or ""),
            source_text=source_text or str(getattr(decision, "source_snippet", "") or ""),
            phase=str(getattr(decision, "phase", "") or "fidelity"),
            chunk_index=int(getattr(decision, "chunk_index", 0) or 0),
            section=str(getattr(decision, "section", "") or ""),
            source_language=source_language,
            target_language=target_language,
            issues=list(getattr(decision, "issues", []) or []),
            accepted=bool(getattr(decision, "accepted", False)),
            response=response,
            model=str(getattr(decision, "judge_model", "") or ""),
            provider=str(getattr(decision, "judge_provider", "") or ""),
            source="fidelity_supervisor",
        )

    @property
    def rejected(self) -> bool:
        return self.decision in {"retry", "repair", "fallback", "pause", "fail"}

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "phase": self.phase,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "detected_language": self.detected_language,
            "dominant_script": self.dominant_script,
            "target_script": self.target_script,
            "target_script_ratio": round(self.target_script_ratio, 4),
            "script_ratios": {key: round(value, 4) for key, value in self.script_ratios.items()},
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "issues": [issue.to_dict() for issue in self.issues],
            "token_cost": self.token_cost.to_dict(),
            "decision": self.decision,
            "repair_plan": self.repair_plan.to_dict(),
            "model": self.model,
            "provider": self.provider,
            "source": self.source,
            "text_hash": stable_hash(self.text),
            "source_hash": stable_hash(self.source_text),
            "text_snippet": snippet(self.text),
            "source_snippet": snippet(self.source_text),
        }
        if include_text:
            data["text"] = self.text
            data["source_text"] = self.source_text
        return data


def record_candidate_result(
    prompt_options: Optional[Mapping[str, Any]],
    result: CandidateResult,
    *,
    limit: int = 240,
) -> None:
    """Append a bounded, metadata-first CandidateResult to prompt_options."""
    if not isinstance(prompt_options, dict):
        return
    records = prompt_options.setdefault("_candidate_results", [])
    if not isinstance(records, list):
        records = []
        prompt_options["_candidate_results"] = records
    records.append(result.to_dict(include_text=False))
    overflow = len(records) - max(1, int(limit or 240))
    if overflow > 0:
        del records[:overflow]


def decide_candidate(issues: list[CandidateIssue], *, accepted: bool) -> str:
    if accepted and not any(issue.severity == "reject" for issue in issues):
        return "accepted"
    codes = {issue.code for issue in issues}
    if codes & _FAIL_CODES:
        return "fail"
    if codes & _RETRY_CODES:
        return "retry"
    if any(issue.severity == "reject" for issue in issues):
        return "repair"
    if issues:
        return "accepted_with_warnings"
    return "accepted"


def normalize_decision(value: str) -> str:
    normalized = normalize_label(value).replace("-", "_").replace(" ", "_")
    return normalized if normalized in _DECISIONS else ""


def _decision_conflicts_with_issues(decision: str, issues: list[CandidateIssue]) -> bool:
    if decision not in {"accepted", "accepted_with_warnings"}:
        return False
    return any(issue.severity == "reject" for issue in issues)


def build_repair_plan(
    issues: list[CandidateIssue],
    *,
    decision: str,
    phase: str,
) -> RepairPlan:
    reasons = tuple(dict.fromkeys(issue.code for issue in issues if issue.code))
    if decision == "accepted":
        return RepairPlan()
    if decision == "accepted_with_warnings":
        return RepairPlan(action="monitor", reasons=reasons)
    if decision == "retry":
        return RepairPlan(
            action="retry",
            reasons=reasons,
            max_rounds=1,
            prompt_variant="target_language_retry",
            notes="Retry with stricter target-language and fidelity instructions.",
        )
    if decision == "fail":
        return RepairPlan(
            action="fail",
            reasons=reasons,
            notes="Do not silently accept this candidate.",
        )
    return RepairPlan(
        action="repair",
        reasons=reasons,
        max_rounds=2 if phase in {"refinement", "profile_repair", "editorial_guard"} else 1,
        prompt_variant="focused_repair",
        notes="Repair only the flagged issues and preserve accepted content.",
    )


def script_for_language(language: str) -> str:
    normalized = normalize_label(language)
    if "(" in normalized:
        normalized = normalized.split("(", 1)[0].strip()
    return _LANGUAGE_SCRIPT_ALIASES.get(normalized, "")


def script_profile(text: str) -> dict[str, float]:
    counts: dict[str, int] = {}
    total = 0
    for script, pattern in _SCRIPT_PATTERNS.items():
        count = len(pattern.findall(text or ""))
        if count:
            counts[script] = count
            total += count
    if total <= 0:
        return {}
    return {script: count / total for script, count in counts.items()}


def dominant_script_ratio(profile: Mapping[str, float]) -> tuple[str, float]:
    if not profile:
        return "", 0.0
    script, ratio = max(profile.items(), key=lambda item: item[1])
    return script, float(ratio)


def detect_language(text: str) -> str:
    profile = script_profile(text)
    dominant, ratio = dominant_script_ratio(profile)
    if dominant and dominant != "latin" and ratio >= 0.35:
        return dominant
    value = text or ""
    marker_scores = {
        "Spanish": len(_SPANISH_MARKERS.findall(value)),
        "English": len(_ENGLISH_MARKERS.findall(value)),
        "French": len(_FRENCH_MARKERS.findall(value)),
    }
    best, score = max(marker_scores.items(), key=lambda item: item[1])
    return best if score > 0 else (dominant or "unknown")


def build_scores(source_text: str, text: str, *, target_script_ratio: float) -> dict[str, float]:
    source_norm = normalize_text(source_text)
    text_norm = normalize_text(text)
    source_chars = len(re.sub(r"\s+", "", source_text or ""))
    text_chars = len(re.sub(r"\s+", "", text or ""))
    scores = {
        "target_script_ratio": target_script_ratio,
        "length_ratio": text_chars / max(1, source_chars) if source_chars else 0.0,
    }
    if source_norm and text_norm:
        scores["source_similarity"] = difflib.SequenceMatcher(
            None,
            source_norm[:_SIMILARITY_COMPARE_CHARS],
            text_norm[:_SIMILARITY_COMPARE_CHARS],
        ).ratio()
    return scores


def normalize_label(value: str) -> str:
    return (
        str(value or "")
        .strip()
        .lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
        .replace("ñ", "n")
    )


def normalize_text(value: str) -> str:
    words = _WORD_RE.findall((value or "").casefold())
    return " ".join(words)


def stable_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


def snippet(value: str, *, max_chars: int = 240) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    return text[:max_chars].rstrip()
