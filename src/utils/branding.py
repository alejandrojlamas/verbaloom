"""Canonical product identity and narrowly scoped legacy compatibility.

All newly generated runtime data must use the VerbaLoom vocabulary.  The
legacy constants in this module exist only so existing installations and
checkpoints can still be read during migration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


DISPLAY_NAME = "VerbaLoom"
TECHNICAL_PREFIX = "verbaloom"
ENV_PREFIX = "VERBALOOM_"
ROUTE_PREFIX = "/verbaloom"
REPOSITORY_OWNER = "alejandrojlamas"
REPOSITORY_NAME = "verbaloom"
REPOSITORY_URL = f"https://github.com/{REPOSITORY_OWNER}/{REPOSITORY_NAME}"
GENERATOR_NAME = DISPLAY_NAME

# These values are deliberately private migration details.  They must never be
# used to generate new identifiers, metadata, classes, paths, or instructions.
_LEGACY_ENV_PREFIX = "TBL_"


def env_value(
    name: str,
    default: str | None = None,
    *,
    legacy_names: Iterable[str] = (),
) -> str | None:
    """Read a canonical environment variable with legacy fallbacks.

    Presence wins over truthiness, so an explicitly empty ``VERBALOOM_*``
    value can intentionally disable a non-empty legacy setting.
    """
    canonical_name = f"{ENV_PREFIX}{name}"
    if canonical_name in os.environ:
        return os.environ[canonical_name]
    candidates = (f"{_LEGACY_ENV_PREFIX}{name}", *legacy_names)
    for candidate in candidates:
        if candidate in os.environ:
            return os.environ[candidate]
    return default


def default_data_dir() -> Path:
    """Return the canonical per-user data directory."""
    return Path.home() / ".local" / "share" / TECHNICAL_PREFIX / "data"
