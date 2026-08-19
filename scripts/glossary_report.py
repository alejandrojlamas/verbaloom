#!/usr/bin/env python3
"""Print a compact report for a book profile glossary."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.book_profiles import load_book_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Report profile glossary status.")
    parser.add_argument("--profile", required=True, help="Active profile id")
    args = parser.parse_args()

    profile = load_book_profile(args.profile)
    by_status = Counter(entry.status for entry in profile.glossary_entries)
    by_type = Counter(entry.entry_type for entry in profile.glossary_entries)

    print(f"# {profile.name}")
    print(f"profile_id: {profile.profile_id}")
    print(f"target_locale: {profile.target_locale or 'n/a'}")
    print(f"loaded_glossaries: {', '.join(profile.raw_config.get('loaded_glossaries') or []) or profile.profile_id}")
    print("\n## Status")
    for key, count in sorted(by_status.items()):
        print(f"- {key}: {count}")
    print("\n## Types")
    for key, count in sorted(by_type.items()):
        print(f"- {key}: {count}")
    print("\n## Pending Suggestions")
    for entry in profile.pending_entries[:50]:
        print(f"- {entry.source} [{entry.entry_type}; confidence={entry.confidence:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
