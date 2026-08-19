"""Maintenance helpers for generated book profiles."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml

from .loader import resolve_profiles_root
from .term_review import suspicious_preserve_entry

_PRESERVE_FRAGMENT_START_WORDS = {
    "ah",
    "but",
    "god",
    "no",
    "now",
    "oh",
    "so",
    "well",
    "why",
    "yes",
}
_PRESERVE_FRAGMENT_ANY_WORDS = {
    "and",
    "but",
    "or",
}
_PRESERVE_FRAGMENT_TRAILING_WORDS = {
    "and",
    "but",
    "or",
    "so",
}
_PROFILE_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:'[A-Za-z]+)?")


def migrate_generated_profile_glossaries(
    *,
    profiles_root: str | Path | None = None,
    dry_run: bool = False,
    profile_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Move suspect preserve-as-source rules in generated profiles to pending.

    This migration is intentionally conservative and lossless: every demoted
    entry is copied into pending_suggestions.yml with its original metadata.
    """
    root = resolve_profiles_root(profiles_root)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "profiles_scanned": 0,
        "profiles_changed": 0,
        "entries_demoted": 0,
        "profiles": [],
    }
    if not root.exists():
        return report
    selected_profiles = {
        str(item).strip()
        for item in (profile_ids or [])
        if str(item).strip()
    }

    for profile_path in sorted(root.glob("*/profile.yml")):
        profile_dir = profile_path.parent
        profile_id = profile_dir.name
        if selected_profiles and profile_id not in selected_profiles:
            continue
        if profile_id.startswith("_") or profile_id == "common":
            continue
        config = _read_yaml_mapping(profile_path)
        if not (config.get("generated_profile") is True or profile_id.startswith("auto_")):
            continue
        report["profiles_scanned"] += 1
        terms_path = profile_dir / "glossary" / "terms.yml"
        pending_path = profile_dir / "glossary" / "pending_suggestions.yml"
        terms_payload = _read_yaml_mapping(terms_path)
        entries = [
            dict(item) for item in (terms_payload.get("entries") or [])
            if isinstance(item, Mapping)
        ]
        keep: list[dict[str, Any]] = []
        demoted: list[dict[str, Any]] = []
        for entry in entries:
            if (
                str(entry.get("status") or "").lower() == "approved"
                and _should_demote_generated_preserve_entry(entry)
            ):
                demoted.append(_demoted_entry(entry, profile_id=profile_id))
            else:
                keep.append(entry)
        if not demoted:
            continue

        profile_report = {
            "profile_id": profile_id,
            "terms_path": str(terms_path.relative_to(root)),
            "demoted": len(demoted),
            "sources": [item["source"] for item in demoted],
        }
        report["profiles"].append(profile_report)
        report["profiles_changed"] += 1
        report["entries_demoted"] += len(demoted)

        if dry_run:
            continue

        _write_yaml(terms_path, {"entries": keep})
        _merge_pending(pending_path, demoted)
        _write_yaml(profile_dir / "glossary" / "migration_report.yml", profile_report)

    return report


def _should_demote_generated_preserve_entry(entry: Mapping[str, Any]) -> bool:
    return suspicious_preserve_entry(entry) or _preserve_entry_looks_like_fragment(entry)


def _preserve_entry_looks_like_fragment(entry: Mapping[str, Any]) -> bool:
    source = str(entry.get("source") or "").strip()
    target = str(entry.get("target") or "").strip()
    if not source or not target or source.casefold() != target.casefold():
        return False
    policy = str(
        entry.get("injection_policy") or entry.get("translation_policy") or ""
    ).strip().lower()
    if policy and policy not in {"preserve", "preserve_exact"}:
        return False
    entry_type = str(entry.get("type") or entry.get("entry_type") or "").strip().lower()
    if entry_type not in {
        "proper_noun",
        "character",
        "location",
        "organization",
        "title",
    }:
        return False

    words = [word.casefold() for word in _PROFILE_WORD_RE.findall(source)]
    if not words:
        return False
    if any(word.endswith("'s") for word in words):
        return True
    if len(words) < 2:
        return False
    if words[0] in _PRESERVE_FRAGMENT_START_WORDS:
        return True
    if words[-1] in _PRESERVE_FRAGMENT_TRAILING_WORDS:
        return True
    if any(word in _PRESERVE_FRAGMENT_ANY_WORDS for word in words):
        return True
    if len(words) >= 3 and "the" in words[1:-1]:
        return True
    return False


def _demoted_entry(entry: Mapping[str, Any], *, profile_id: str) -> dict[str, Any]:
    source = str(entry.get("source") or "").strip()
    target = str(entry.get("target") or "").strip()
    data = dict(entry)
    data["suggested_target"] = "" if target.casefold() == source.casefold() else target
    data.pop("target", None)
    data["scope"] = str(data.get("scope") or profile_id)
    data["status"] = "pending"
    data["translation_policy"] = "pending_review"
    data["injection_policy"] = "contextual"
    data["review_status"] = "pending_review"
    data["reviewed_by"] = "generated_profile_glossary_migration"
    data["migration_reason"] = (
        "Demoted from approved because preserving source text looked unsafe for translation; "
        "review before approving for this profile."
    )
    data["rationale"] = data.get("rationale") or data["migration_reason"]
    risks = list(data.get("risks") or [])
    risks.append("May force the model to keep a directly translatable term in the source language.")
    data["risks"] = risks[:4]
    return data


def _merge_pending(path: Path, entries: list[dict[str, Any]]) -> None:
    payload = _read_yaml_mapping(path)
    existing = [
        dict(item) for item in (payload.get("suggestions") or payload.get("entries") or [])
        if isinstance(item, Mapping)
    ]
    seen = {str(item.get("source") or "").casefold() for item in existing}
    for entry in entries:
        source = str(entry.get("source") or "").strip()
        if source and source.casefold() not in seen:
            existing.append(entry)
            seen.add(source.casefold())
    _write_yaml(path, {"suggestions": existing})


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.safe_load(text) if text.strip() else {}
    return dict(data) if isinstance(data, Mapping) else {}


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
