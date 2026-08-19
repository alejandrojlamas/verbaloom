"""Pure policy and text helpers for editorial quality guards."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from .locale_quality import (
    build_mexican_spanish_repair_instructions,
    count_mexican_spanish_issues,
    format_mexican_spanish_issues,
    is_mexican_spanish_target,
    mexican_spanish_issue_codes,
)


_MARKDOWN_EMPHASIS_RESIDUE_RE = re.compile(r"(?<!\w)_[^_\n]{1,80}_(?!\w)")
_AWKWARD_OH_USTEDES_RE = re.compile(r"¡Oh!\s+ustedes\b", re.IGNORECASE)
_QUALITY_ALERT_MODEL_OFF_VALUES = {"", "none", "off", "false", "disabled"}
_QUALITY_ALERT_MODEL_SAME_VALUES = {"same", "primary", "main", "default"}
_SOURCE_AWARE_GUARD_OFF_VALUES = {"", "none", "off", "false", "disabled"}


def _mexican_spanish_guard_enabled(
    target_language: str,
    prompt_options: Optional[dict],
) -> bool:
    if not is_mexican_spanish_target(target_language, prompt_options):
        return False
    options = prompt_options or {}
    return (
        options.get("spanish_locale_guard", True) is not False
        and options.get("mexican_spanish_guard", True) is not False
    )


def _quality_alert_guard_enabled(prompt_options: Optional[dict]) -> bool:
    options = prompt_options or {}
    return options.get("quality_alert_guard", True) is not False


def _literary_quality_alert_guard_enabled(prompt_options: Optional[dict]) -> bool:
    options = prompt_options or {}
    return (
        _quality_alert_guard_enabled(options)
        and options.get("literary_quality_alert_guard", True) is not False
    )


def _resolve_quality_alert_model(model: str, prompt_options: Optional[dict]) -> str:
    """Pick the cheap model used only for alert-triggered repair passes."""
    options = prompt_options or {}
    configured = (
        options.get("quality_alert_model")
        or options.get("alert_review_model")
        or options.get("locale_repair_model")
        or options.get("mexican_spanish_repair_model")
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


def _source_aware_editorial_guard_enabled(prompt_options: Optional[dict]) -> bool:
    options = prompt_options or {}
    return options.get("source_aware_editorial_guard", True) is not False


def _source_aware_editorial_guard_mode(prompt_options: Optional[dict]) -> str:
    options = prompt_options or {}
    mode = str(options.get("source_aware_editorial_guard_mode") or "alerted").strip().lower()
    if mode in _SOURCE_AWARE_GUARD_OFF_VALUES:
        return "off"
    if mode not in {"always", "alerted"}:
        return "alerted"
    return mode


def _resolve_source_aware_editorial_guard_model(
    model: str,
    prompt_options: Optional[dict],
) -> str:
    options = prompt_options or {}
    configured = (
        options.get("source_aware_editorial_guard_model")
        or options.get("editorial_guard_model")
    )
    if configured is not None:
        value = str(configured).strip()
        normalized = value.lower()
        if normalized in _SOURCE_AWARE_GUARD_OFF_VALUES:
            return model
        if normalized in _QUALITY_ALERT_MODEL_SAME_VALUES:
            return model
        return value

    if "deepseek" in (model or "").lower():
        return "deepseek-v4-pro"
    return model


def _normalize_guard_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_source_text_for_guard(
    chunk: Optional[Dict],
    draft_text: str,
) -> str:
    if not isinstance(chunk, dict):
        return ""
    for key in ("source_text", "_source_text", "original_text", "main_content", "text", "content"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            # Refine-only chunks often expose the translated draft as source.
            if _normalize_guard_text(value) == _normalize_guard_text(draft_text):
                continue
            return value
    return ""


def _should_run_source_aware_editorial_guard(
    decision: Any,
    *,
    source_text: str,
    prompt_options: Optional[dict],
) -> bool:
    if not source_text.strip():
        return False
    if not _source_aware_editorial_guard_enabled(prompt_options):
        return False
    mode = _source_aware_editorial_guard_mode(prompt_options)
    if mode == "off":
        return False
    if mode == "always":
        return True
    return (not decision.accepted) or bool(decision.warnings)


def _count_literary_quality_alerts(
    text: str,
    prompt_options: Optional[dict],
) -> dict[str, int]:
    if not _literary_quality_alert_guard_enabled(prompt_options):
        return {}

    counts: Dict[str, int] = {}
    markdown_residue = _MARKDOWN_EMPHASIS_RESIDUE_RE.findall(text or "")
    if markdown_residue:
        counts["markdown_emphasis_residue"] = len(markdown_residue)

    awkward_direct_address = _AWKWARD_OH_USTEDES_RE.findall(text or "")
    if awkward_direct_address:
        counts["awkward_oh_ustedes"] = len(awkward_direct_address)

    return counts


def _count_quality_alerts(
    text: str,
    *,
    target_language: str,
    prompt_options: Optional[dict],
) -> dict[str, int]:
    if not _quality_alert_guard_enabled(prompt_options):
        return {}

    counts: Dict[str, int] = {}
    if _mexican_spanish_guard_enabled(target_language, prompt_options):
        counts.update(count_mexican_spanish_issues(text))
    counts.update(_count_literary_quality_alerts(text, prompt_options))
    return counts


def _format_quality_alerts(issue_counts: Dict[str, int]) -> str:
    return format_mexican_spanish_issues(issue_counts)


def _build_quality_alert_repair_instructions(
    issue_counts: Dict[str, int],
    *,
    target_language: str,
) -> str:
    sections = []

    regional_keys = mexican_spanish_issue_codes()
    regional_counts = {
        key: value
        for key, value in issue_counts.items()
        if key in regional_keys and value
    }
    if regional_counts:
        sections.append(build_mexican_spanish_repair_instructions(regional_counts))

    if issue_counts.get("markdown_emphasis_residue"):
        sections.append(
            """
The draft contains visible markdown/Project Gutenberg emphasis markers.
- Remove stray underscore emphasis markers such as `_¡Así_` or `_palabra_`.
- Keep the word and punctuation; do not add replacement symbols.
- Do not remove underscores inside technical identifiers if any appear.
""".strip()
        )

    if issue_counts.get("awkward_oh_ustedes"):
        sections.append(
            """
The draft contains awkward literal direct address such as "¡Oh! ustedes".
- Rewrite it as elevated natural Mexican Spanish.
- Prefer structures like "¡Oh, mis...!" or "¡Oh, ustedes..." only if it truly sounds natural.
- Preserve the vocative force and the literary tone without using Peninsular forms.
""".strip()
        )

    if not sections:
        sections.append(
            f"Lightly copyedit the {target_language} draft only where needed; preserve meaning and structure."
        )

    sections.append(
        """
This is an alert repair pass, not a rewrite.
Make the smallest edit that clears the alerts.
Do not summarize, omit content, reorder paragraphs, or make the prose more ornate just to show improvement.
""".strip()
    )
    return "\n\n".join(sections)


def _content_length_ratio_ok(before: str, after: str) -> bool:
    before_chars = len(re.sub(r"\s+", "", before or ""))
    after_chars = len(re.sub(r"\s+", "", after or ""))
    if after_chars == 0:
        return False
    if before_chars < 120:
        return True
    ratio = after_chars / max(1, before_chars)
    return 0.70 <= ratio <= 1.45
