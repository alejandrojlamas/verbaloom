#!/usr/bin/env python3
"""Run local glossary discovery and store pending suggestions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.book_profiles.discovery import (  # noqa: E402
    merge_pending_suggestions,
    suggest_glossary_entries,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Suggest pending glossary entries for a profile.")
    parser.add_argument("--profile", required=True, help="Active profile id")
    parser.add_argument("--input", required=True, help="Source text file")
    parser.add_argument("--max-suggestions", type=int, default=80)
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    suggestions = suggest_glossary_entries(
        text,
        profile_id=args.profile,
        max_suggestions=args.max_suggestions,
    )
    path = merge_pending_suggestions(args.profile, suggestions)
    print(f"{len(suggestions)} pending suggestion(s) written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
