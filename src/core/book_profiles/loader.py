"""Load and create book-scoped editorial profiles."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Optional
from urllib.parse import unquote

import yaml

from .artifacts import load_editorial_artifacts, load_editorial_signal_index
from .models import BookProfile, ProfileDetector, ProfileGlossaryEntry


_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class BookProfileError(RuntimeError):
    """Raised when a book profile is missing or invalid."""


_PROFILE_CACHE_FILE_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml"}


def resolve_profiles_root(root: str | Path | None = None) -> Path:
    if root:
        return Path(root).expanduser().resolve()
    env_root = os.getenv("BOOK_PROFILES_DIR")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "profiles"


def profile_exists(profile_id: str, *, profiles_root: str | Path | None = None) -> bool:
    if not _safe_profile_id(profile_id):
        return False
    return (resolve_profiles_root(profiles_root) / profile_id / "profile.yml").exists()


def infer_profile_id_from_metadata(
    metadata: Mapping[str, Any],
    *,
    profiles_root: str | Path | None = None,
) -> Optional[str]:
    """Infer a book profile from profile-owned match metadata.

    The matching terms live in each profile.yml, not in the generic pipeline.
    This lets a known book activate its own editorial profile when the UI sends
    a generic transform request, without hardcoding literary equivalences or
    book-specific rules into the engine.
    """
    root = resolve_profiles_root(profiles_root)
    haystack = " ".join(
        str(value)
        for value in metadata.values()
        if value is not None and str(value).strip()
    ).casefold()
    if not haystack or not root.exists():
        return None

    explicit_profile_id = str(
        metadata.get("profile_id")
        or metadata.get("book_profile_id")
        or ""
    ).strip()
    if (
        explicit_profile_id
        and _safe_profile_id(explicit_profile_id)
        and (root / explicit_profile_id / "profile.yml").exists()
    ):
        return explicit_profile_id

    matches: list[tuple[tuple[int, int, int, int, int, int], str]] = []
    for config_path in sorted(root.glob("*/profile.yml")):
        config = _read_yaml_mapping(config_path)
        profile_id = str(config.get("profile_id") or config_path.parent.name).strip()
        if not _safe_profile_id(profile_id):
            continue
        auto_detect = _mapping(config.get("auto_detect"))
        if not auto_detect or auto_detect.get("enabled") is False:
            if _profile_source_matches_metadata(config, haystack):
                matches.append((_profile_match_score(config_path, config, specificity=1), profile_id))
            continue
        all_terms = _as_string_list(auto_detect.get("all"))
        any_terms = _as_string_list(
            auto_detect.get("any")
            or auto_detect.get("keywords")
            or auto_detect.get("match_any")
        )
        if not all_terms and not any_terms:
            if _profile_source_matches_metadata(config, haystack) and _safe_profile_id(profile_id):
                return profile_id
            continue
        if all_terms and not all(term.casefold() in haystack for term in all_terms):
            continue
        if any_terms and not any(term.casefold() in haystack for term in any_terms):
            continue
        matches.append((_profile_match_score(config_path, config, specificity=2), profile_id))
    if not matches:
        return None
    # Prefer a specifically configured match, then a completed profile with an
    # approved glossary. This prevents abandoned/empty preparation directories
    # from shadowing the usable profile merely because they sort first.
    return max(matches, key=lambda item: (item[0], item[1]))[1]


def profile_matches_source_metadata(
    profile_id: str,
    metadata: Mapping[str, Any],
    *,
    profiles_root: str | Path | None = None,
) -> bool:
    """Return whether a generated profile belongs to the supplied document.

    Built-in profiles without a ``source_name`` are intentionally reusable.
    Generated profiles are book-scoped: sharing an author is not sufficient to
    apply one book's glossary and editorial decisions to another book.
    """
    if not _safe_profile_id(str(profile_id or "").strip()):
        return False
    config_path = resolve_profiles_root(profiles_root) / profile_id / "profile.yml"
    config = _read_yaml_mapping(config_path)
    if not config:
        return False
    if not str(config.get("source_name") or "").strip():
        return True
    haystack = " ".join(
        str(value)
        for value in metadata.values()
        if value is not None and str(value).strip()
    ).casefold()
    return _profile_source_matches_metadata(config, haystack)


def _profile_match_score(
    config_path: Path,
    config: Mapping[str, Any],
    *,
    specificity: int,
) -> tuple[int, int, int, int, int, int]:
    profile_dir = config_path.parent
    glossary_files = _mapping(config.get("glossary_files"))
    terms_path = profile_dir / str(glossary_files.get("terms") or "glossary/terms.yml")
    pending_path = profile_dir / str(
        glossary_files.get("pending_suggestions") or "glossary/pending_suggestions.yml"
    )
    terms = _read_yaml_mapping(terms_path)
    pending = _read_yaml_mapping(pending_path)
    approved_count = sum(
        1
        for item in terms.get("entries") or []
        if isinstance(item, Mapping)
        and str(item.get("status") or "approved").strip().lower() == "approved"
    )
    pending_count = sum(
        1
        for item in (pending.get("suggestions") or pending.get("entries") or [])
        if isinstance(item, Mapping)
    )
    has_editorial_map = int((profile_dir / "editorial" / "editorial_map.yml").exists())
    profile_id = str(config.get("profile_id") or profile_dir.name)
    return (
        int(specificity),
        int(approved_count > 0),
        approved_count,
        has_editorial_map,
        pending_count,
        -len(profile_id),
    )


def load_book_profile(
    profile_id: str | None,
    *,
    profiles_root: str | Path | None = None,
    allow_missing: bool = False,
) -> Optional[BookProfile]:
    profile_id = str(profile_id or "").strip()
    if not profile_id:
        return None
    if not _safe_profile_id(profile_id):
        raise BookProfileError(f"Invalid profile id: {profile_id!r}")

    root = resolve_profiles_root(profiles_root)
    profile_dir = root / profile_id
    config_path = profile_dir / "profile.yml"
    if not config_path.exists():
        if allow_missing:
            return None
        raise BookProfileError(f"Book profile not found: {profile_id}")

    fingerprint = _profile_cache_fingerprint(profile_dir, root=root)
    return _load_book_profile_cached(profile_id, str(root), fingerprint)


@lru_cache(maxsize=64)
def _load_book_profile_cached(
    profile_id: str,
    root_value: str,
    fingerprint: tuple[tuple[str, int, int], ...],
) -> BookProfile:
    """Load one immutable profile snapshot.

    ``fingerprint`` is deliberately part of the cache key. Profile/glossary
    edits therefore invalidate the snapshot automatically without reparsing a
    large YAML glossary on every chunk.
    """
    del fingerprint
    root = Path(root_value)
    profile_dir = root / profile_id
    config_path = profile_dir / "profile.yml"
    config = _read_yaml_mapping(config_path)
    prompt_files = _mapping(config.get("prompts"))
    glossary_files = _mapping(config.get("glossary_files") or config.get("glossary"))

    policy_text = _read_text(profile_dir / "editorial_policy.md")
    prompt_texts = {
        key: _read_text(profile_dir / str(path))
        for key, path in prompt_files.items()
        if str(path).strip()
    }

    loaded_glossaries = _as_string_list(config.get("loaded_glossaries"))
    allow_common = bool(config.get("allow_common_glossary", True))

    entries: list[ProfileGlossaryEntry] = []
    if allow_common and "common" in loaded_glossaries:
        entries.extend(_load_common_glossary_entries(root))
    for key, rel_path in glossary_files.items():
        if not str(rel_path).strip():
            continue
        entries.extend(_load_glossary_entries(
            profile_dir / str(rel_path),
            default_scope=profile_id,
        ))

    detectors = []
    for raw in config.get("detectors", []) or []:
        if isinstance(raw, Mapping):
            detector = ProfileDetector.from_dict(raw)
            if detector is not None:
                detectors.append(detector)

    return BookProfile(
        profile_id=str(config.get("profile_id") or profile_id),
        name=str(config.get("name") or profile_id),
        root=profile_dir,
        target_locale=str(config.get("target_locale") or ""),
        editorial_mode=str(config.get("editorial_mode") or "book_profile"),
        modernization_strength=str(config.get("modernization_strength") or "high"),
        preserve_author_voice=bool(config.get("preserve_author_voice", True)),
        allow_common_glossary=allow_common,
        allow_cross_profile_glossary=bool(config.get("allow_cross_profile_glossary", False)),
        min_dimension_score=float(config.get("min_dimension_score") or 8.5),
        max_repair_rounds=int(config.get("max_repair_rounds") or 2),
        policy_text=policy_text,
        prompt_texts=prompt_texts,
        glossary_entries=tuple(entries),
        detectors=tuple(detectors),
        editorial_artifacts=load_editorial_artifacts(profile_dir),
        editorial_signal_index=load_editorial_signal_index(profile_dir),
        raw_config=config,
    )


def clear_book_profile_cache() -> None:
    """Drop cached profile snapshots after bulk filesystem operations."""
    _load_book_profile_cached.cache_clear()


def _profile_cache_fingerprint(
    profile_dir: Path,
    *,
    root: Path,
) -> tuple[tuple[str, int, int], ...]:
    """Return a cheap signature for all files that can affect a profile."""
    files: list[Path] = []
    for directory in (profile_dir, root / "common"):
        if not directory.exists():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.casefold() in _PROFILE_CACHE_FILE_SUFFIXES
        )

    signature: list[tuple[str, int, int]] = []
    for path in sorted(files):
        try:
            stat_result = path.stat()
        except FileNotFoundError:
            # An atomic glossary replacement may race this read. The next call
            # observes the replacement; this call simply omits the old inode.
            continue
        signature.append((
            str(path.relative_to(root)),
            int(stat_result.st_mtime_ns),
            int(stat_result.st_size),
        ))
    return tuple(signature)


def create_profile(
    profile_id: str,
    *,
    profiles_root: str | Path | None = None,
    force: bool = False,
) -> Path:
    if not _safe_profile_id(profile_id):
        raise BookProfileError(f"Invalid profile id: {profile_id!r}")

    root = resolve_profiles_root(profiles_root)
    template = root / "_template"
    destination = root / profile_id
    if destination.exists() and not force:
        raise BookProfileError(f"Profile already exists: {profile_id}")
    if destination.exists() and force:
        shutil.rmtree(destination)
    if template.exists():
        shutil.copytree(template, destination)
    else:
        _create_minimal_profile(destination, profile_id)

    profile_path = destination / "profile.yml"
    config = _read_yaml_mapping(profile_path) if profile_path.exists() else {}
    config["profile_id"] = profile_id
    config.setdefault("name", profile_id.replace("_", " ").title())
    _write_yaml(profile_path, config)
    editorial_path = destination / "editorial" / "editorial_map.yml"
    if editorial_path.exists():
        editorial = _read_yaml_mapping(editorial_path)
        editorial["profile_id"] = profile_id
        _write_yaml(editorial_path, editorial)
    clear_book_profile_cache()
    return destination


def _load_glossary_entries(path: Path, *, default_scope: str) -> list[ProfileGlossaryEntry]:
    payload = _read_yaml(path)
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        raw_entries = payload.get("entries") or payload.get("suggestions") or []
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raw_entries = []

    entries: list[ProfileGlossaryEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            continue
        entry = ProfileGlossaryEntry.from_dict(
            raw,
            default_scope=default_scope,
            source_file=str(path.name),
        )
        if entry is not None:
            entries.append(entry)
    return entries


def _load_common_glossary_entries(root: Path) -> list[ProfileGlossaryEntry]:
    common_dir = root / "common" / "glossary"
    if not common_dir.exists():
        return []
    entries: list[ProfileGlossaryEntry] = []
    for path in sorted(common_dir.glob("*.yml")):
        entries.extend(_load_glossary_entries(path, default_scope="common"))
    return entries


def _safe_profile_id(value: str) -> bool:
    if not value or value in {".", ".."}:
        return False
    return all(ch.isalnum() or ch in {"_", "-"} for ch in value)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _read_yaml(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    return yaml.load(text, Loader=_YAML_SAFE_LOADER) if text.strip() else None


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    data = _read_yaml(path)
    return dict(data) if isinstance(data, Mapping) else {}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _profile_source_matches_metadata(config: Mapping[str, Any], haystack: str) -> bool:
    source_name = str(config.get("source_name") or "").strip()
    if not source_name:
        return False
    candidates = {
        source_name,
        Path(source_name).name,
        Path(source_name).stem,
    }
    candidates.update(_source_name_segments(source_name))
    normalized_haystack = _normalize_match_text(haystack)
    for candidate in candidates:
        normalized = _normalize_match_text(candidate)
        if len(normalized) >= 12 and normalized in normalized_haystack:
            return True
    return False


def _normalize_match_text(value: str) -> str:
    folded = unquote(str(value or "")).casefold()
    folded = re.sub(r"\.[a-z0-9]{1,8}\b", " ", folded)
    return " ".join(
        token for token in re.sub(r"[_\W]+", " ", folded, flags=re.UNICODE).split()
        if token
    )


def _source_name_segments(source_name: str) -> set[str]:
    """Return title-like segments from generated profile source names.

    Author-only segments caused profiles for different books by the same author
    to contaminate one another. Filename inference may use the title segment,
    while the complete title/author stem remains available to the caller.
    """
    stem = Path(str(source_name or "")).stem
    if not stem:
        return set()
    for separator in (" - ", " – ", " — ", " by ", " por "):
        if separator in stem:
            title = stem.split(separator, 1)[0].strip()
            return {title} if len(_normalize_match_text(title)) >= 12 else set()
    return set()


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _create_minimal_profile(destination: Path, profile_id: str) -> None:
    (destination / "prompts").mkdir(parents=True, exist_ok=True)
    (destination / "glossary").mkdir(parents=True, exist_ok=True)
    (destination / "editorial").mkdir(parents=True, exist_ok=True)
    _write_yaml(destination / "profile.yml", {
        "profile_id": profile_id,
        "name": profile_id.replace("_", " ").title(),
        "editorial_mode": "book_profile",
        "target_locale": "",
        "prompts": {
            "modernize": "prompts/modernize.txt",
            "voice_restoration": "prompts/voice_restoration.txt",
            "audit": "prompts/audit.txt",
            "repair": "prompts/repair.txt",
            "glossary_discovery": "prompts/glossary_discovery.txt",
        },
        "glossary_files": {
            "terms": "glossary/terms.yml",
            "treatments": "glossary/treatments.yml",
            "phrases": "glossary/phrases.yml",
            "character_voices": "glossary/character_voices.yml",
            "pending_suggestions": "glossary/pending_suggestions.yml",
        },
    })
    for name in ("modernize", "voice_restoration", "audit", "repair", "glossary_discovery"):
        (destination / "prompts" / f"{name}.txt").write_text("", encoding="utf-8")
    for name in ("terms", "treatments", "phrases", "character_voices", "pending_suggestions"):
        _write_yaml(destination / "glossary" / f"{name}.yml", {"entries": []})
    (destination / "editorial_policy.md").write_text(
        "# Editorial Policy\n\nDescribe this book profile here.\n",
        encoding="utf-8",
    )
