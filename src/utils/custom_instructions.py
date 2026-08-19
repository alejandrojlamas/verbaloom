"""Custom instructions loader.

Loads presets from the `Custom_Instructions/` folder for use in translation
and refinement prompts. Two formats are supported:

- `.yaml` / `.yml`: structured file with optional `translation` and
  `refinement` top-level keys. Either or both may be present; the missing
  phase is left unset.
- `.txt` (legacy): plain text, applied to both phases identically.

Returned shape is always `{"translation": str | None, "refinement": str | None}`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TypedDict

import yaml


SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+\.(?:txt|ya?ml)$")
SUPPORTED_EXTENSIONS = (".txt", ".yaml", ".yml")


class CustomInstructions(TypedDict, total=False):
    translation: Optional[str]
    refinement: Optional[str]


def is_safe_filename(filename: str) -> bool:
    """Whitelist filenames to alphanumerics + `_-.` with a supported extension."""
    return bool(SAFE_FILENAME_RE.match(filename or ""))


def _resolve_inside(directory: Path, filename: str) -> Optional[Path]:
    """Return the file path if it resolves inside `directory`, else None."""
    candidate = directory / filename
    try:
        candidate.resolve().relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate


def _normalize_phase_value(value) -> Optional[str]:
    """Coerce a YAML scalar into a stripped string, or None if empty."""
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    stripped = value.strip()
    return stripped or None


def load_custom_instructions(
    filename: str, custom_instructions_dir: Path
) -> CustomInstructions:
    """Load a preset and return `{"translation": ..., "refinement": ...}`.

    Raises:
        ValueError: filename is unsafe or escapes the directory.
        FileNotFoundError: file does not exist.
        yaml.YAMLError: YAML file is malformed.
    """
    if not is_safe_filename(filename):
        raise ValueError(
            f"Invalid filename '{filename}'. Allowed: alphanumerics, "
            f"`_`, `-`, `.`; extension must be .txt, .yaml, or .yml."
        )

    file_path = _resolve_inside(custom_instructions_dir, filename)
    if file_path is None:
        raise ValueError(
            f"Filename '{filename}' resolves outside Custom_Instructions directory."
        )

    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Custom instructions file not found: {filename}")

    suffix = file_path.suffix.lower()
    # `utf-8-sig` transparently strips a leading BOM (Windows Notepad / Word
    # exports default to UTF-8-with-BOM). Plain `utf-8` would leave the BOM
    # as a literal `﻿` character at the start of the parsed value, and
    # would also raise UnicodeDecodeError on a Latin-1 file.
    raw = file_path.read_text(encoding="utf-8-sig")

    if suffix == ".txt":
        text = raw.strip()
        if not text:
            return {"translation": None, "refinement": None}
        return {"translation": text, "refinement": text}

    # YAML
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        raise

    if parsed is None:
        return {"translation": None, "refinement": None}

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Custom instructions YAML '{filename}' must be a mapping with "
            f"optional 'translation' and 'refinement' keys."
        )

    return {
        "translation": _normalize_phase_value(parsed.get("translation")),
        "refinement": _normalize_phase_value(parsed.get("refinement")),
    }


def list_custom_instructions(custom_instructions_dir: Path) -> list[dict]:
    """List presets in the directory with phase availability metadata.

    Returns a list of dicts:
        {
            "filename": "noir_detective.yaml",
            "display_name": "noir_detective",
            "format": "yaml" | "txt",
            "has_translation": bool,
            "has_refinement": bool,
        }
    Malformed files are silently skipped.
    """
    if not custom_instructions_dir.exists():
        return []

    entries: list[dict] = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in custom_instructions_dir.glob(f"*{ext}"):
            try:
                file_path.resolve().relative_to(custom_instructions_dir.resolve())
            except ValueError:
                continue

            try:
                content = load_custom_instructions(file_path.name, custom_instructions_dir)
            except (ValueError, yaml.YAMLError, FileNotFoundError, OSError, UnicodeDecodeError):
                continue

            entries.append(
                {
                    "filename": file_path.name,
                    "display_name": file_path.stem,
                    "format": "yaml" if ext in (".yaml", ".yml") else "txt",
                    "has_translation": content.get("translation") is not None,
                    "has_refinement": content.get("refinement") is not None,
                }
            )

    entries.sort(key=lambda e: e["display_name"].lower())
    return entries
