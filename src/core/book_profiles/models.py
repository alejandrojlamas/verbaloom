"""Data models for book-scoped editorial profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


APPROVED_STATES = {"approved"}
PENDING_STATES = {"pending"}
REJECTED_STATES = {"rejected", "superseded"}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class ProfileGlossaryExample:
    source_excerpt: str = ""
    target_excerpt: str = ""
    recommended_modernization: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProfileGlossaryExample":
        return cls(
            source_excerpt=str(data.get("source_excerpt") or ""),
            target_excerpt=str(data.get("target_excerpt") or ""),
            recommended_modernization=str(data.get("recommended_modernization") or ""),
        )


@dataclass(frozen=True)
class ProfileGlossaryEntry:
    source: str
    target: str = ""
    target_options: tuple[str, ...] = ()
    entry_type: str = "term"
    scope: str = ""
    status: str = "pending"
    confidence: float = 0.0
    occurrences: int = 0
    rationale: str = ""
    examples: tuple[ProfileGlossaryExample, ...] = ()
    restrictions: tuple[str, ...] = ()
    do_not_apply_if: tuple[str, ...] = ()
    applies_to: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""
    forbidden_default: str = ""
    decision_rule: str = ""
    mechanical_safe: bool = False
    review_status: str = ""
    review_confidence: float = 0.0
    injection_policy: str = ""
    translation_policy: str = ""
    review_rationale: str = ""
    reviewed_by: str = ""
    source_language: str = ""
    source_file: str = ""

    @property
    def approved(self) -> bool:
        return self.status in APPROVED_STATES

    @property
    def pending(self) -> bool:
        return self.status in PENDING_STATES

    @property
    def rejected(self) -> bool:
        return self.status in REJECTED_STATES

    @property
    def render_target(self) -> str:
        if self.target:
            return self.target
        if self.target_options:
            return " | ".join(self.target_options)
        return ""

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        default_scope: str = "",
        source_file: str = "",
    ) -> Optional["ProfileGlossaryEntry"]:
        source = str(data.get("source") or data.get("term") or "").strip()
        if not source:
            return None
        examples = tuple(
            ProfileGlossaryExample.from_dict(item)
            for item in data.get("examples", []) or []
            if isinstance(item, Mapping)
        )
        try:
            confidence = float(data.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        try:
            review_confidence = float(data.get("review_confidence") or 0.0)
        except (TypeError, ValueError):
            review_confidence = 0.0
        try:
            occurrences = int(data.get("occurrences") or 0)
        except (TypeError, ValueError):
            occurrences = 0
        return cls(
            source=source,
            target=str(data.get("target") or data.get("suggested_target") or "").strip(),
            target_options=tuple(_as_list(data.get("target_options"))),
            entry_type=str(data.get("type") or data.get("entry_type") or "term").strip(),
            scope=str(data.get("scope") or default_scope or "").strip(),
            status=str(data.get("status") or "pending").strip().lower(),
            confidence=confidence,
            occurrences=occurrences,
            rationale=str(data.get("rationale") or "").strip(),
            examples=examples,
            restrictions=tuple(_as_list(data.get("restrictions"))),
            do_not_apply_if=tuple(
                _as_list(data.get("do_not_apply_if") or data.get("no_aplicar_si"))
            ),
            applies_to=_as_dict(data.get("applies_to") or data.get("aplica_a")),
            notes=str(data.get("notes") or data.get("notas_editoriales") or "").strip(),
            forbidden_default=str(data.get("forbidden_default") or "").strip(),
            decision_rule=str(data.get("decision_rule") or "").strip(),
            mechanical_safe=bool(data.get("mechanical_safe") or False),
            review_status=str(data.get("review_status") or "").strip(),
            review_confidence=review_confidence,
            injection_policy=str(data.get("injection_policy") or "").strip(),
            translation_policy=str(data.get("translation_policy") or "").strip(),
            review_rationale=str(data.get("review_rationale") or "").strip(),
            reviewed_by=str(data.get("reviewed_by") or "").strip(),
            source_language=str(data.get("source_language") or "").strip(),
            source_file=source_file,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "type": self.entry_type,
            "scope": self.scope,
            "status": self.status,
            "confidence": self.confidence,
        }
        if self.target:
            data["target"] = self.target
        if self.target_options:
            data["target_options"] = list(self.target_options)
        if self.rationale:
            data["rationale"] = self.rationale
        if self.occurrences:
            data["occurrences"] = self.occurrences
        if self.examples:
            data["examples"] = [example.__dict__ for example in self.examples]
        if self.restrictions:
            data["restrictions"] = list(self.restrictions)
        if self.do_not_apply_if:
            data["do_not_apply_if"] = list(self.do_not_apply_if)
        if self.applies_to:
            data["applies_to"] = dict(self.applies_to)
        if self.notes:
            data["notes"] = self.notes
        if self.forbidden_default:
            data["forbidden_default"] = self.forbidden_default
        if self.decision_rule:
            data["decision_rule"] = self.decision_rule
        if self.mechanical_safe:
            data["mechanical_safe"] = True
        if self.review_status:
            data["review_status"] = self.review_status
        if self.review_confidence:
            data["review_confidence"] = self.review_confidence
        if self.injection_policy:
            data["injection_policy"] = self.injection_policy
        if self.translation_policy:
            data["translation_policy"] = self.translation_policy
        if self.review_rationale:
            data["review_rationale"] = self.review_rationale
        if self.reviewed_by:
            data["reviewed_by"] = self.reviewed_by
        if self.source_language:
            data["source_language"] = self.source_language
        return data


@dataclass(frozen=True)
class ProfileDetector:
    code: str
    pattern: str
    severity: str = "warning"
    message: str = ""
    applies_to: str = "candidate"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Optional["ProfileDetector"]:
        code = str(data.get("code") or "").strip()
        pattern = str(data.get("pattern") or "").strip()
        if not code or not pattern:
            return None
        return cls(
            code=code,
            pattern=pattern,
            severity=str(data.get("severity") or "warning").strip().lower(),
            message=str(data.get("message") or code).strip(),
            applies_to=str(data.get("applies_to") or "candidate").strip().lower(),
        )


@dataclass(frozen=True)
class BookProfile:
    profile_id: str
    name: str
    root: Path
    target_locale: str = ""
    editorial_mode: str = "book_profile"
    modernization_strength: str = "high"
    preserve_author_voice: bool = True
    allow_common_glossary: bool = True
    allow_cross_profile_glossary: bool = False
    min_dimension_score: float = 8.5
    max_repair_rounds: int = 2
    policy_text: str = ""
    prompt_texts: Mapping[str, str] = field(default_factory=dict)
    glossary_entries: tuple[ProfileGlossaryEntry, ...] = ()
    detectors: tuple[ProfileDetector, ...] = ()
    editorial_artifacts: Mapping[str, Any] = field(default_factory=dict)
    editorial_signal_index: Mapping[str, Any] = field(default_factory=dict)
    raw_config: Mapping[str, Any] = field(default_factory=dict)

    @property
    def approved_entries(self) -> tuple[ProfileGlossaryEntry, ...]:
        return tuple(entry for entry in self.glossary_entries if entry.approved)

    @property
    def pending_entries(self) -> tuple[ProfileGlossaryEntry, ...]:
        return tuple(entry for entry in self.glossary_entries if entry.pending)

    @property
    def approved_count(self) -> int:
        return len(self.approved_entries)

    @property
    def pending_count(self) -> int:
        return len(self.pending_entries)
