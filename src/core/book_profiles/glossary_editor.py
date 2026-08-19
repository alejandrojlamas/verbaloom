"""Safe mutations for book-profile glossary YAML files.

This module intentionally edits only glossary files declared by the active book
profile. Common glossaries and other profiles are read-only from this path, so
profile-specific editorial choices cannot leak globally.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tempfile
import threading
from typing import Any, Mapping

import yaml

from .loader import BookProfileError, load_book_profile, resolve_profiles_root
from .models import ProfileGlossaryEntry


class ProfileGlossaryEditError(RuntimeError):
    """Raised when a profile glossary mutation is invalid or unsafe."""


class ProfileGlossaryConflictError(ProfileGlossaryEditError):
    """Raised when a UI action targets a row that changed after it was loaded."""


_GLOSSARY_EDIT_LOCK = threading.RLock()


@dataclass(frozen=True)
class ProfileGlossaryEditResult:
    profile_id: str
    source_file: str
    entry_index: int
    entry: ProfileGlossaryEntry | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "source_file": self.source_file,
            "entry_index": self.entry_index,
            "entry": self.entry.to_dict() if self.entry is not None else None,
        }


_ALLOWED_UPDATE_KEYS = {
    "source",
    "target",
    "target_options",
    "type",
    "entry_type",
    "scope",
    "status",
    "confidence",
    "rationale",
    "occurrences",
    "examples",
    "restrictions",
    "do_not_apply_if",
    "applies_to",
    "notes",
    "forbidden_default",
    "decision_rule",
    "mechanical_safe",
    "review_status",
    "review_confidence",
    "injection_policy",
    "translation_policy",
    "review_rationale",
    "reviewed_by",
    "superseded_by",
}


def editable_profile_glossary_files(
    profile_id: str,
    *,
    profiles_root: str | Path | None = None,
) -> dict[str, Path]:
    """Return basename -> absolute path for glossary files owned by a profile."""
    profile = load_book_profile(profile_id, profiles_root=profiles_root)
    if profile is None:
        raise ProfileGlossaryEditError(f"Book profile not found: {profile_id}")
    profile_root = profile.root.resolve()
    glossary_files = profile.raw_config.get("glossary_files") or profile.raw_config.get("glossary") or {}
    if not isinstance(glossary_files, Mapping):
        return {}
    editable: dict[str, Path] = {}
    for rel_path in glossary_files.values():
        rel = str(rel_path or "").strip()
        if not rel:
            continue
        path = (profile_root / rel).resolve()
        try:
            path.relative_to(profile_root)
        except ValueError as exc:
            raise ProfileGlossaryEditError(f"Unsafe glossary path: {rel}") from exc
        editable[path.name] = path
    return editable


def profile_glossary_entry_locations(
    profile_id: str,
    *,
    profiles_root: str | Path | None = None,
) -> dict[tuple[str, str, str], int]:
    """Map loaded entries to file-local indices for API presentation.

    The loader preserves file order. A small signature including source/status
    is enough for UI actions; duplicate entries still receive increasing
    file-local indices because we consume candidates as they appear.
    """
    profile = load_book_profile(profile_id, profiles_root=profiles_root)
    if profile is None:
        return {}
    editable = editable_profile_glossary_files(profile_id, profiles_root=profiles_root)
    counters: dict[str, int] = {}
    locations: dict[tuple[str, str, str], int] = {}
    for entry in profile.glossary_entries:
        source_file = entry.source_file or ""
        if source_file not in editable:
            continue
        index = counters.get(source_file, 0)
        counters[source_file] = index + 1
        locations[(source_file, entry.source, entry.status)] = index
    return locations


def apply_profile_glossary_action(
    profile_id: str,
    *,
    source_file: str,
    entry_index: int,
    action: str,
    target: str = "",
    rationale: str = "",
    review_rationale: str = "",
    updates: Mapping[str, Any] | None = None,
    expected_source: str = "",
    profiles_root: str | Path | None = None,
) -> ProfileGlossaryEditResult:
    """Apply a controlled glossary action to one profile-owned entry."""
    action_key = str(action or "update").strip().lower()
    mutation: dict[str, Any] = dict(updates or {})
    target_text = str(target or "").strip()
    if rationale:
        mutation["rationale"] = str(rationale).strip()
    if review_rationale:
        mutation["review_rationale"] = str(review_rationale).strip()
    mutation.setdefault("reviewed_by", "profile_glossary_editor")

    if action_key == "approve":
        mutation["status"] = "approved"
        mutation.setdefault("review_status", "approved")
        if target_text:
            mutation["target"] = target_text
    elif action_key == "reject":
        mutation["status"] = "rejected"
        mutation["review_status"] = "rejected"
    elif action_key == "translate":
        if not target_text:
            raise ProfileGlossaryEditError("A target is required to mark a term as translated.")
        mutation.update({
            "status": "approved",
            "target": target_text,
            "review_status": "approved",
            "translation_policy": "translate_consistently",
            "injection_policy": "canonical_translation",
        })
    elif action_key == "preserve":
        mutation.update({
            "status": "approved",
            "review_status": "approved",
            "translation_policy": "preserve_exact",
            "injection_policy": "preserve_exact",
        })
    elif action_key == "update":
        pass
    else:
        raise ProfileGlossaryEditError(f"Unsupported glossary action: {action}")

    return update_profile_glossary_entry(
        profile_id,
        source_file=source_file,
        entry_index=entry_index,
        updates=mutation,
        preserve_target_for_source=(action_key == "preserve"),
        expected_source=expected_source,
        profiles_root=profiles_root,
    )


def update_profile_glossary_entry(
    profile_id: str,
    *,
    source_file: str,
    entry_index: int,
    updates: Mapping[str, Any],
    preserve_target_for_source: bool = False,
    expected_source: str = "",
    profiles_root: str | Path | None = None,
) -> ProfileGlossaryEditResult:
    with _GLOSSARY_EDIT_LOCK:
        path = _resolve_owned_glossary_path(profile_id, source_file, profiles_root=profiles_root)
        data, entries, container_key, root_is_list = _read_entries_payload(path)
        index = _coerce_index(entry_index, len(entries))
        raw = _entry_mapping(entries[index])
        _assert_expected_source(raw, expected_source)
        _apply_updates(raw, updates)
        if preserve_target_for_source:
            # Preserve is an invariant, not a suggestion that an arbitrary
            # `updates.target` field may override.
            raw["target"] = str(raw.get("source") or raw.get("term") or "").strip()
        entries[index] = raw
        _write_entries_payload(path, data, entries, container_key, root_is_list)
        entry = ProfileGlossaryEntry.from_dict(
            raw,
            default_scope=profile_id,
            source_file=Path(path).name,
        )
    return ProfileGlossaryEditResult(profile_id, Path(path).name, index, entry)


def merge_profile_glossary_entries(
    profile_id: str,
    *,
    source_file: str,
    entry_indices: list[int],
    target: str = "",
    rationale: str = "",
    expected_sources: Mapping[int, str] | None = None,
    profiles_root: str | Path | None = None,
) -> ProfileGlossaryEditResult:
    with _GLOSSARY_EDIT_LOCK:
        path = _resolve_owned_glossary_path(profile_id, source_file, profiles_root=profiles_root)
        data, entries, container_key, root_is_list = _read_entries_payload(path)
        indices = sorted({_coerce_index(index, len(entries)) for index in entry_indices})
        if len(indices) < 2:
            raise ProfileGlossaryEditError("At least two entries are required to merge.")

        expected = {int(index): str(source) for index, source in (expected_sources or {}).items()}
        for index in indices:
            _assert_expected_source(_entry_mapping(entries[index]), expected.get(index, ""))

        base_index = indices[0]
        base = _entry_mapping(entries[base_index])
        merged = [_entry_mapping(entries[index]) for index in indices]
        source_values = [str(item.get("source") or item.get("term") or "").strip() for item in merged]
        target_values = [str(item.get("target") or item.get("suggested_target") or "").strip() for item in merged]
        base["source"] = source_values[0]
        selected_target = str(target or "").strip() or next((value for value in target_values if value), "")
        if not selected_target:
            raise ProfileGlossaryEditError("A canonical target is required to merge glossary entries.")
        base["target"] = selected_target
        base["status"] = "approved"
        base["review_status"] = "merged"
        base["translation_policy"] = base.get("translation_policy") or "translate_consistently"
        base["injection_policy"] = base.get("injection_policy") or "canonical_translation"
        base["reviewed_by"] = "profile_glossary_editor"
        base["target_options"] = _unique_strings(
            list(_as_list(base.get("target_options")))
            + [value for value in target_values if value.casefold() != selected_target.casefold()]
        )
        base["occurrences"] = sum(_as_int(item.get("occurrences")) for item in merged)
        merge_note = rationale or f"Merged {len(indices)} profile glossary entries."
        base["review_rationale"] = _join_notes(base.get("review_rationale"), merge_note)

        for index in indices[1:]:
            item = _entry_mapping(entries[index])
            # Keep each source variant active while recording one canonical
            # editorial decision. Superseding aliases would silently remove
            # those variants from future per-chunk matching.
            item["target"] = selected_target
            item["status"] = "approved"
            item["review_status"] = "merged_alias"
            item["superseded_by"] = source_values[0]
            item["translation_policy"] = base["translation_policy"]
            item["injection_policy"] = base["injection_policy"]
            item["reviewed_by"] = "profile_glossary_editor"
            item["review_rationale"] = _join_notes(item.get("review_rationale"), merge_note)
            entries[index] = item
        entries[base_index] = base
        _write_entries_payload(path, data, entries, container_key, root_is_list)
        entry = ProfileGlossaryEntry.from_dict(base, default_scope=profile_id, source_file=Path(path).name)
    return ProfileGlossaryEditResult(profile_id, Path(path).name, base_index, entry)


def _resolve_owned_glossary_path(
    profile_id: str,
    source_file: str,
    *,
    profiles_root: str | Path | None = None,
) -> Path:
    if not profile_id:
        raise ProfileGlossaryEditError("Profile id is required.")
    basename = Path(str(source_file or "")).name
    if not basename:
        raise ProfileGlossaryEditError("Glossary source file is required.")
    try:
        files = editable_profile_glossary_files(profile_id, profiles_root=profiles_root)
    except BookProfileError as exc:
        raise ProfileGlossaryEditError(str(exc)) from exc
    path = files.get(basename)
    if path is None:
        raise ProfileGlossaryEditError(f"Glossary file is not editable for this profile: {basename}")
    root = resolve_profiles_root(profiles_root).resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ProfileGlossaryEditError("Resolved glossary path escaped the profiles root.") from exc
    return path


def _read_entries_payload(path: Path) -> tuple[Any, list[Any], str, bool]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"entries": []}, [], "entries", False
    payload = yaml.safe_load(text) if text.strip() else None
    if isinstance(payload, list):
        return payload, list(payload), "", True
    if isinstance(payload, Mapping):
        data = dict(payload)
        key = "entries" if isinstance(data.get("entries"), list) else "suggestions"
        entries = data.get(key)
        if not isinstance(entries, list):
            key = "entries"
            entries = []
            data[key] = entries
        return data, list(entries), key, False
    return {"entries": []}, [], "entries", False


def _write_entries_payload(
    path: Path,
    data: Any,
    entries: list[Any],
    container_key: str,
    root_is_list: bool,
) -> None:
    payload = entries if root_is_list else dict(data)
    if not root_is_list:
        payload[container_key or "entries"] = entries
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if existing_mode is not None:
            os.chmod(tmp_name, existing_mode)
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _assert_expected_source(raw: Mapping[str, Any], expected_source: str) -> None:
    expected = str(expected_source or "").strip()
    if not expected:
        return
    actual = str(raw.get("source") or raw.get("term") or "").strip()
    if actual != expected:
        raise ProfileGlossaryConflictError(
            "The glossary changed after this row was loaded. Refresh it before editing."
        )


def _coerce_index(value: int, length: int) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ProfileGlossaryEditError("Entry index must be an integer.") from exc
    if index < 0 or index >= length:
        raise ProfileGlossaryEditError("Glossary entry index is out of range.")
    return index


def _entry_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileGlossaryEditError("Glossary entry is not editable.")
    return dict(value)


def _apply_updates(raw: dict[str, Any], updates: Mapping[str, Any]) -> None:
    for key, value in (updates or {}).items():
        if key not in _ALLOWED_UPDATE_KEYS:
            continue
        normalized_key = "type" if key == "entry_type" else key
        if value is None:
            continue
        if isinstance(value, str):
            raw[normalized_key] = value.strip()
        else:
            raw[normalized_key] = value


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _unique_strings(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value or "").strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        out.append(text)
    return out


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _join_notes(first: Any, second: str) -> str:
    left = str(first or "").strip()
    right = str(second or "").strip()
    if left and right:
        return f"{left} {right}"
    return left or right
