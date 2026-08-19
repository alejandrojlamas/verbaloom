#!/usr/bin/env python3
"""Demote unsafe generated-profile glossary entries to pending review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.book_profiles.migration import migrate_generated_profile_glossaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--profile-id",
        action="append",
        default=None,
        help="Limit migration to one generated profile id. May be passed more than once.",
    )
    args = parser.parse_args()

    report = migrate_generated_profile_glossaries(
        profiles_root=args.profiles_root,
        dry_run=args.dry_run,
        profile_ids=args.profile_id,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
