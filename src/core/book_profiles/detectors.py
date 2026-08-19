"""Profile-scoped detector helpers.

The generic engine can ask a profile whether a chunk violates that profile, but
it must not know the book's literary vocabulary. This module is the narrow
bridge between generic pipeline code and profile-owned detector/glossary data.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Optional

from .loader import BookProfileError, load_book_profile
from .models import BookProfile, ProfileDetector, ProfileGlossaryEntry


_MODERNIZABLE_ENTRY_TYPES = {
    "address_form",
    "idiom",
    "lexical_archaism",
    "orthographic_variant",
    "phrase",
    "cultural_term",
    "sensitive_term",
    "syntax_pattern",
    "term",
}
_PROPER_NOUN_ENTRY_TYPES = {
    "canonical_proper_noun",
    "proper_noun",
    "character",
    "place",
    "location",
    "organization",
}


@dataclass(frozen=True)
class ProfileDetectorHit:
    code: str
    severity: str
    message: str
    excerpt: str = ""
    applies_to: str = "candidate"
    source: str = "detector"


def active_profile_from_options(prompt_options: Optional[Mapping[str, Any]]) -> Optional[BookProfile]:
    options = prompt_options or {}
    profile_id = str(options.get("profile_id") or "").strip()
    if not profile_id:
        return None
    try:
        return load_book_profile(profile_id)
    except BookProfileError:
        if options.get("profile_required") is not True:
            return None
        raise


def detector_hits(
    source_text: str,
    candidate_text: str,
    detectors: tuple[ProfileDetector, ...],
) -> list[ProfileDetectorHit]:
    haystacks = {
        "source": source_text or "",
        "candidate": candidate_text or "",
        "both": f"{source_text or ''}\n{candidate_text or ''}",
    }
    hits: list[ProfileDetectorHit] = []
    for detector in detectors:
        text = haystacks.get(detector.applies_to, candidate_text or "")
        try:
            match = re.search(detector.pattern, text, flags=re.I)
        except re.error:
            continue
        if not match:
            continue
        hits.append(
            ProfileDetectorHit(
                code=detector.code,
                severity=_normalized_severity(detector.severity),
                message=detector.message,
                excerpt=_snippet(text, match.start(), match.end()),
                applies_to=detector.applies_to,
                source="detector",
            )
        )
    return hits


def glossary_residual_hits(
    source_text: str,
    candidate_text: str,
    entries: tuple[ProfileGlossaryEntry, ...],
) -> list[ProfileDetectorHit]:
    """Return profile-owned glossary violations useful for repair triage."""
    source = source_text or ""
    candidate = candidate_text or ""
    hits: list[ProfileDetectorHit] = []
    for entry in entries:
        if not entry.approved:
            continue
        if entry.forbidden_default and contains_profile_term(candidate, entry.forbidden_default):
            hits.append(
                ProfileDetectorHit(
                    code="forbidden_default",
                    severity="high",
                    message=(
                        "Candidate uses a forbidden default from the active "
                        "profile glossary."
                    ),
                    excerpt=entry.forbidden_default,
                    source="glossary",
                )
            )
            continue

        if not _is_modernizable_entry(entry):
            continue
        if not contains_profile_term(source, entry.source):
            continue
        if not contains_profile_term(candidate, entry.source):
            continue
        if entry.target and _same_term(entry.source, entry.target):
            continue
        hits.append(
            ProfileDetectorHit(
                code="glossary_residual",
                severity="high" if entry.entry_type == "address_form" else "medium",
                message=(
                    "Candidate still contains a source form that the active "
                    "profile glossary marks for contextual modernization."
                ),
                excerpt=entry.source,
                source="glossary",
            )
        )
    return hits


def has_profile_modernization_signal(
    source_text: str,
    *,
    profile: Optional[BookProfile] = None,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> bool:
    profile = profile or active_profile_from_options(prompt_options)
    if profile is None:
        return False
    source = source_text or ""
    for entry in profile.approved_entries:
        if _is_modernizable_entry(entry) and contains_profile_term(source, entry.source):
            return True
    for hit in detector_hits(source, "", tuple(
        detector for detector in profile.detectors
        if detector.applies_to in {"source", "both"}
    )):
        if hit:
            return True
    return False


def lost_profile_proper_nouns(
    source_text: str,
    candidate_text: str,
    *,
    profile: Optional[BookProfile] = None,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    profile = profile or active_profile_from_options(prompt_options)
    if profile is None:
        return []
    lost: list[str] = []
    for entry in profile.approved_entries:
        if entry.entry_type not in _PROPER_NOUN_ENTRY_TYPES:
            continue
        expected = entry.target if entry.target else entry.source
        if contains_profile_term(source_text, entry.source) and not contains_profile_term(candidate_text, expected):
            lost.append(expected)
    return sorted(set(lost), key=str.casefold)


def contains_profile_term(text: str, term: str) -> bool:
    folded_term = fold_profile_match_text(term).strip()
    if not folded_term:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(folded_term) + r"(?!\w)", fold_profile_match_text(text)))


def fold_profile_match_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    stripped = "".join(
        char for char in decomposed
        if not unicodedata.combining(char)
    )
    return stripped.casefold()


def _is_modernizable_entry(entry: ProfileGlossaryEntry) -> bool:
    if entry.entry_type not in _MODERNIZABLE_ENTRY_TYPES:
        return False
    if entry.mechanical_safe:
        return False
    return bool(entry.target or entry.target_options or entry.decision_rule or entry.forbidden_default)


def _same_term(left: str, right: str) -> bool:
    return (left or "").casefold().strip() == (right or "").casefold().strip()


def _normalized_severity(value: str) -> str:
    severity = str(value or "").strip().lower()
    if severity in {"critical", "high", "medium", "low"}:
        return severity
    if severity in {"error", "fail", "failure"}:
        return "high"
    if severity in {"warn", "warning"}:
        return "medium"
    return "medium"


def _snippet(text: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    value = re.sub(r"\s+", " ", text[left:right]).strip()
    if left > 0:
        value = "..." + value
    if right < len(text):
        value += "..."
    return value
