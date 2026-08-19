"""Helpers for same-language text transformations.

The transformation tab intentionally allows larger edits than ordinary
copyediting.  This module contains the extra safety rails for conservative
literary modernization, where structure and source fidelity matter more than
showing visible change.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Optional


_OFF_VALUES = {"", "0", "false", "no", "off", "disabled", "none"}
_BLOCK_MARKER_RE = re.compile(r"\[id9\d{5}\]")


@dataclass(frozen=True)
class BlockProtection:
    original_text: str
    protected_text: str
    markers: tuple[str, ...]

    @property
    def active(self) -> bool:
        return bool(self.markers)


def text_transform_mode(prompt_options: Optional[Mapping[str, Any]]) -> str:
    return str((prompt_options or {}).get("text_transform_mode") or "").strip().lower()


def is_faithful_modernize(prompt_options: Optional[Mapping[str, Any]]) -> bool:
    return text_transform_mode(prompt_options) == "modernize"


def prompt_bool(
    prompt_options: Optional[Mapping[str, Any]],
    key: str,
    default: bool = False,
) -> bool:
    if prompt_options is None or key not in prompt_options:
        return default
    value = prompt_options.get(key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _OFF_VALUES


def apply_faithful_modernize_defaults(prompt_options: dict[str, Any]) -> dict[str, Any]:
    """Mutate and return prompt_options with safe defaults for Modernizar."""
    if not is_faithful_modernize(prompt_options):
        return prompt_options

    uses_profile = _uses_book_profile(prompt_options)

    prompt_options.setdefault("text_transform_profile", "faithful_current_spanish")
    prompt_options.setdefault("preserve_block_structure", True)
    prompt_options.setdefault("transform_guard", "strict")
    prompt_options.setdefault(
        "transform_auditor_model",
        "deepseek-v4-pro" if uses_profile else "deepseek-v4-flash",
    )
    prompt_options.setdefault("transform_repair_attempts", 2)
    prompt_options.setdefault("suppress_attribution_footer", True)
    # Default transformation fallback is the best audited candidate. Reverting
    # to source is reserved for hard corruption; otherwise a modernize job can
    # punish an imperfect modernization by returning the unmodernized source.
    prompt_options.setdefault("transform_fallback", "best_candidate")

    # Modernization has meaningful changes, but they must still be source-safe.
    prompt_options["editorial_quality_guard"] = True
    prompt_options["fidelity_supervisor"] = True
    # Book-profile modernization has its own style audit/repair loop. The
    # fidelity supervisor still audits every chunk against the source, but it
    # must not become the style gate that sends a valid modernization back to
    # the archaic source text.
    prompt_options.setdefault("fidelity_supervisor_mode", "always" if uses_profile else "alerted")
    prompt_options["fidelity_supervisor_model"] = str(
        prompt_options.get("fidelity_supervisor_model")
        or prompt_options.get("transform_auditor_model")
        or "deepseek-v4-flash"
    )
    prompt_options.setdefault("fidelity_supervisor_retry", True)

    if uses_profile:
        # The active profile has its own source-aware audit and repair loop.
        # The generic guard is intentionally profile-agnostic and can mistake
        # approved intralingual modernization for unsafe stylistic drift.
        prompt_options.setdefault("source_aware_editorial_guard", False)
        prompt_options.setdefault("source_aware_editorial_guard_mode", "off")
    return prompt_options


# Editorial-guard issue codes that indicate real candidate corruption in a
# modernize job. Every other reject code can be reported and repaired without
# treating the archaic/source text as the better deliverable.
MODERNIZE_HARD_REJECT_CODES = frozenset({
    "empty_refinement",
    "mojibake_regression",
    "artifact_glyphs_added",
    "placeholder_mismatch",
    "modernize_block_count_changed",
    "source_aware_weird_symbols",
    "source_aware_judge_reject",
})

# Fidelity rejections that still require falling back to the source/draft in a
# same-language modernization. Other fidelity alerts should flow into the
# profile audit/repair loop; otherwise the system can punish an imperfect but
# repairable modernization by restoring the unmodernized source.
MODERNIZE_FIDELITY_HARD_REJECT_CODES = frozenset({
    "empty_candidate",
    "mojibake_regression",
    "artifact_glyphs_added",
    "placeholder_mismatch",
})


def transform_fallback_mode(prompt_options: Optional[Mapping[str, Any]]) -> str:
    raw = str((prompt_options or {}).get("transform_fallback") or "").strip().lower()
    return raw if raw in {"best_candidate", "source"} else "best_candidate"


def meaningful_block_count(text: str) -> int:
    return len(_split_meaningful_blocks(text or ""))


def protect_meaningful_blocks(
    text: str,
    *,
    enabled: bool = True,
    marker_base: int = 900000,
) -> BlockProtection:
    """Prefix paragraph-like blocks with temporary [idNNNNNN] markers."""
    original = text or ""
    if not enabled:
        return BlockProtection(original, original, ())

    blocks = _split_meaningful_blocks(original)
    if len(blocks) < 2:
        return BlockProtection(original, original, ())

    markers = tuple(f"[id{marker_base + i}]" for i in range(len(blocks)))
    protected = "\n\n".join(f"{marker}\n{block}" for marker, block in zip(markers, blocks))
    return BlockProtection(original, protected, markers)


def restore_protected_blocks(candidate_text: str, protection: BlockProtection) -> tuple[bool, str, str]:
    """Validate and strip block markers from a candidate."""
    candidate = candidate_text or ""
    if not protection.active:
        return True, candidate, ""

    found = tuple(_BLOCK_MARKER_RE.findall(candidate))
    if found != protection.markers:
        return (
            False,
            candidate,
            "block marker sequence changed "
            f"(expected {len(protection.markers)}, found {len(found)})",
        )

    blocks: list[str] = []
    search_from = 0
    for index, marker in enumerate(protection.markers):
        marker_pos = candidate.find(marker, search_from)
        if marker_pos < 0:
            return False, candidate, f"missing marker {marker}"
        content_start = marker_pos + len(marker)
        if index + 1 < len(protection.markers):
            next_pos = candidate.find(protection.markers[index + 1], content_start)
            if next_pos < 0:
                return False, candidate, f"missing marker {protection.markers[index + 1]}"
            raw_block = candidate[content_start:next_pos]
            search_from = next_pos
        else:
            raw_block = candidate[content_start:]
            search_from = len(candidate)
        block = raw_block.strip()
        if not block:
            return False, candidate, f"empty block after marker {marker}"
        blocks.append(block)

    return True, "\n\n".join(blocks).strip(), ""


def block_repair_instructions(reason: str) -> str:
    return f"""# BLOCK STRUCTURE REPAIR

Your previous answer failed structural validation: {reason or 'block markers changed'}.
Return the same text again, but preserve every [idNNNNNN] block marker exactly once,
in the same order. Do not add, remove, rename, reorder, merge, or split block markers.
Each marker identifies one source paragraph or heading; keep that block separate.
"""


def modernize_guard_findings(
    draft_text: str,
    refined_text: str,
    *,
    prompt_options: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, str]]:
    """Return Modernizar-specific deterministic guard findings.

    Keep this intentionally language- and book-agnostic.  Semantic questions
    such as whether an archaic phrase should be changed, or whether a pronoun
    shift changes a quoted address, belong to the source-aware LLM auditor.
    """
    if not is_faithful_modernize(prompt_options):
        return []

    draft = draft_text or ""
    refined = refined_text or ""
    findings: list[dict[str, str]] = []

    if prompt_bool(prompt_options, "preserve_block_structure", True):
        before = meaningful_block_count(draft)
        after = meaningful_block_count(refined)
        if before >= 2 and before != after:
            findings.append({
                "code": "modernize_block_count_changed",
                "severity": "reject",
                "message": "La modernizacion cambio la cantidad de bloques/parrafos",
                "detail": f"{before} -> {after}",
            })

    return findings


def _uses_book_profile(prompt_options: Mapping[str, Any]) -> bool:
    return (
        str(prompt_options.get("editorial_mode") or "").strip().lower() == "book_profile"
        and bool(str(prompt_options.get("profile_id") or "").strip())
    )


def _split_meaningful_blocks(text: str) -> list[str]:
    stripped = (text or "").strip()
    if not stripped:
        return []
    return [
        block.strip()
        for block in re.split(r"\n\s*\n+", stripped)
        if block.strip()
    ]
