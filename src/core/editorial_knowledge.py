"""Unified editorial knowledge primitives.

This module gives book profiles and classic glossaries one operational shape:
term policies.  The historical systems still own storage and UI compatibility;
the unified view is used by guards, reports, and prompt diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional

from src.core.candidate_result import CandidateIssue


_TRANSLATE_POLICIES = {
    "translate",
    "translate_exact",
    "translate_contextual",
    "translate_consistently",
    "canonical_translation",
    "canonical_name",
}
_PRESERVE_POLICIES = {
    "preserve",
    "preserve_exact",
    "do_not_translate",
    "keep_source",
    "source_form",
}
_ACTIVE_STATES = {"approved", "active"}
_PENDING_STATES = {"pending", "needs_review", "review"}


@dataclass(frozen=True)
class EditorialTermPolicy:
    """A term-level rule that can come from a profile or a manual glossary."""

    source: str
    target: str = ""
    policy: str = "translate"
    status: str = "approved"
    category: str = ""
    scope: str = ""
    confidence: float = 0.0
    occurrences: int = 0
    origin: str = ""
    rationale: str = ""
    target_options: tuple[str, ...] = ()
    applies_to: Mapping[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status in _ACTIVE_STATES

    @property
    def pending(self) -> bool:
        return self.status in _PENDING_STATES

    @property
    def should_translate(self) -> bool:
        normalized = normalize_policy(self.policy)
        if normalized in _TRANSLATE_POLICIES:
            return True
        return bool(self.target and fold_text(self.source) != fold_text(self.target))

    @property
    def should_preserve(self) -> bool:
        normalized = normalize_policy(self.policy)
        if normalized in _PRESERVE_POLICIES:
            return True
        if not self.target:
            return True
        return fold_text(self.source) == fold_text(self.target)

    @property
    def render_target(self) -> str:
        if self.target:
            return self.target
        if self.target_options:
            return " | ".join(self.target_options)
        return ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "policy": self.policy,
            "status": self.status,
            "category": self.category,
            "scope": self.scope,
            "confidence": self.confidence,
            "occurrences": self.occurrences,
            "origin": self.origin,
        }
        if self.rationale:
            data["rationale"] = self.rationale
        if self.target_options:
            data["target_options"] = list(self.target_options)
        if self.applies_to:
            data["applies_to"] = dict(self.applies_to)
        return {key: value for key, value in data.items() if value not in ("", 0, 0.0, [], {})}


@dataclass(frozen=True)
class EditorialKnowledgeValidation:
    issues: tuple[CandidateIssue, ...] = ()
    matched_terms: int = 0
    translated_terms: int = 0
    preserved_terms: int = 0
    pending_terms: int = 0

    @property
    def clean(self) -> bool:
        return not any(issue.severity == "reject" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "matched_terms": self.matched_terms,
            "translated_terms": self.translated_terms,
            "preserved_terms": self.preserved_terms,
            "pending_terms": self.pending_terms,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class EditorialKnowledgeBase:
    """Unified, read-only operational view of editorial knowledge."""

    scope_id: str = ""
    name: str = ""
    target_locale: str = ""
    terms: tuple[EditorialTermPolicy, ...] = ()
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    readiness: Mapping[str, Any] = field(default_factory=dict)

    @property
    def active_terms(self) -> tuple[EditorialTermPolicy, ...]:
        return tuple(term for term in self.terms if term.active)

    @property
    def pending_terms(self) -> tuple[EditorialTermPolicy, ...]:
        return tuple(term for term in self.terms if term.pending)

    @property
    def translated_terms(self) -> tuple[EditorialTermPolicy, ...]:
        return tuple(term for term in self.active_terms if term.should_translate)

    @property
    def preserved_terms(self) -> tuple[EditorialTermPolicy, ...]:
        return tuple(term for term in self.active_terms if term.should_preserve)

    def validate_candidate(
        self,
        source_text: str,
        candidate_text: str,
        *,
        purpose: str = "translation",
    ) -> EditorialKnowledgeValidation:
        """Check active term policies against one source/candidate pair."""

        issues: list[CandidateIssue] = []
        matched = translated = preserved = pending = 0
        for term in self.terms:
            if not term.active and not term.pending:
                continue
            if not contains_term(source_text, term.source):
                continue
            matched += 1
            if term.pending:
                pending += 1
                continue
            if term.should_translate:
                translated += 1
                _validate_translation_term(term, candidate_text, issues, purpose=purpose)
            elif term.should_preserve:
                preserved += 1
                _validate_preserve_term(term, candidate_text, issues)

        return EditorialKnowledgeValidation(
            issues=tuple(issues),
            matched_terms=matched,
            translated_terms=translated,
            preserved_terms=preserved,
            pending_terms=pending,
        )

    def prompt_diagnostics(self, *, max_terms: int = 12) -> dict[str, Any]:
        """Return a compact prompt-readiness summary for UI/reporting."""

        risky_source_equals_target = [
            term for term in self.translated_terms
            if term.target and fold_text(term.source) == fold_text(term.target)
        ]
        warnings: list[str] = []
        if not self.terms and not self.artifacts:
            warnings.append("empty_editorial_knowledge")
        if risky_source_equals_target:
            warnings.append("translated_terms_preserve_source_form")
        if len(self.pending_terms) > len(self.active_terms):
            warnings.append("pending_terms_dominate")
        return {
            "scope_id": self.scope_id,
            "name": self.name,
            "target_locale": self.target_locale,
            "counts": {
                "terms": len(self.terms),
                "active_terms": len(self.active_terms),
                "pending_terms": len(self.pending_terms),
                "translated_terms": len(self.translated_terms),
                "preserved_terms": len(self.preserved_terms),
                "source_equals_target_translations": len(risky_source_equals_target),
            },
            "warnings": warnings,
            "sample_terms": [term.to_dict() for term in self.active_terms[:max_terms]],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "name": self.name,
            "target_locale": self.target_locale,
            "terms": [term.to_dict() for term in self.terms],
            "artifacts": dict(self.artifacts or {}),
            "readiness": dict(self.readiness or {}),
            "diagnostics": self.prompt_diagnostics(),
        }


def build_editorial_knowledge_base(
    *,
    profile: Any = None,
    glossary: Any = None,
) -> EditorialKnowledgeBase:
    """Create a unified knowledge base from an optional profile and glossary."""

    terms: list[EditorialTermPolicy] = []
    scope_id = ""
    name = ""
    target_locale = ""
    artifacts: Mapping[str, Any] = {}
    readiness: Mapping[str, Any] = {}

    if profile is not None:
        scope_id = str(getattr(profile, "profile_id", "") or "")
        name = str(getattr(profile, "name", "") or "")
        target_locale = str(getattr(profile, "target_locale", "") or "")
        artifacts = getattr(profile, "editorial_artifacts", {}) or {}
        for entry in getattr(profile, "glossary_entries", ()) or ():
            policy = _policy_from_profile_entry(entry)
            terms.append(EditorialTermPolicy(
                source=str(getattr(entry, "source", "") or ""),
                target=str(getattr(entry, "target", "") or ""),
                target_options=tuple(getattr(entry, "target_options", ()) or ()),
                policy=policy,
                status=str(getattr(entry, "status", "") or "pending").lower(),
                category=str(getattr(entry, "entry_type", "") or ""),
                scope=str(getattr(entry, "scope", "") or scope_id),
                confidence=float(getattr(entry, "confidence", 0.0) or 0.0),
                occurrences=int(getattr(entry, "occurrences", 0) or 0),
                origin="profile",
                rationale=str(getattr(entry, "rationale", "") or getattr(entry, "review_rationale", "") or ""),
                applies_to=getattr(entry, "applies_to", {}) or {},
            ))

    if glossary is not None:
        glossary_id = getattr(glossary, "id", None)
        if not scope_id and glossary_id is not None:
            scope_id = f"glossary:{glossary_id}"
        if not name:
            name = str(getattr(glossary, "name", "") or "")
        if not target_locale:
            target_locale = str(getattr(glossary, "target_language", "") or "")
        for term in getattr(glossary, "terms", []) or []:
            source = str(getattr(term, "source_term", "") or "")
            target = str(getattr(term, "translated_term", "") or "")
            if not source:
                continue
            policy = (
                "preserve_exact"
                if target and fold_text(source) == fold_text(target)
                else "translate"
            )
            terms.append(EditorialTermPolicy(
                source=source,
                target=target,
                policy=policy if target else "pending",
                status="approved" if target else "pending",
                category=str(getattr(term, "category", "") or ""),
                scope=scope_id,
                confidence=1.0 if target else 0.0,
                origin="glossary",
            ))

    terms = _dedupe_terms(terms)
    return EditorialKnowledgeBase(
        scope_id=scope_id,
        name=name,
        target_locale=target_locale,
        terms=tuple(terms),
        artifacts=artifacts,
        readiness=readiness,
    )


def normalize_policy(value: str) -> str:
    return re.sub(r"[\s-]+", "_", str(value or "").strip().lower())


def fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", without_marks).strip().casefold()


def contains_term(text: str, term: str) -> bool:
    folded_text = fold_text(text)
    folded_term = fold_text(term)
    if not folded_text or not folded_term:
        return False
    alternatives = [part.strip() for part in folded_term.split("|") if part.strip()]
    for alternative in alternatives or [folded_term]:
        if _term_boundary_match(folded_text, alternative):
            return True
    return False


def _term_boundary_match(text: str, term: str) -> bool:
    if not term:
        return False
    if _has_word_edge(term):
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        return re.search(pattern, text, flags=re.IGNORECASE) is not None
    return term in text


def _has_word_edge(term: str) -> bool:
    return bool(term and (term[0].isalnum() or term[-1].isalnum()))


def _policy_from_profile_entry(entry: Any) -> str:
    raw = (
        getattr(entry, "translation_policy", "")
        or getattr(entry, "injection_policy", "")
        or getattr(entry, "review_status", "")
        or ""
    )
    normalized = normalize_policy(raw)
    if normalized in _TRANSLATE_POLICIES | _PRESERVE_POLICIES:
        return normalized
    source = str(getattr(entry, "source", "") or "")
    target = str(getattr(entry, "target", "") or "")
    if target and fold_text(source) != fold_text(target):
        return "translate"
    return "preserve_exact"


def _validate_translation_term(
    term: EditorialTermPolicy,
    candidate_text: str,
    issues: list[CandidateIssue],
    *,
    purpose: str,
) -> None:
    target = term.target
    if not target:
        issues.append(CandidateIssue(
            "editorial_term_missing_target",
            "warning",
            "An active translatable term has no approved target.",
            detail=term.source,
            source="editorial_knowledge",
        ))
        return
    candidate_has_target = contains_term(candidate_text, target)
    candidate_has_source = contains_term(candidate_text, term.source)
    if candidate_has_target:
        return
    severity = "reject" if candidate_has_source and purpose == "translation" else "warning"
    issues.append(CandidateIssue(
        "editorial_term_translation_missing",
        severity,
        "An approved editorial term was not rendered with its target form.",
        detail=f"{term.source} -> {target}",
        source="editorial_knowledge",
    ))


def _validate_preserve_term(
    term: EditorialTermPolicy,
    candidate_text: str,
    issues: list[CandidateIssue],
) -> None:
    expected = term.target or term.source
    if contains_term(candidate_text, expected):
        return
    issues.append(CandidateIssue(
        "editorial_preserve_term_missing",
        "warning",
        "A term marked for preservation is missing from the candidate.",
        detail=expected,
        source="editorial_knowledge",
    ))


def _dedupe_terms(terms: Iterable[EditorialTermPolicy]) -> list[EditorialTermPolicy]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[EditorialTermPolicy] = []
    for term in terms:
        if not term.source:
            continue
        key = (
            fold_text(term.source),
            fold_text(term.target),
            normalize_policy(term.policy),
            term.origin,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped
