#!/usr/bin/env python3
"""Create an empty book editorial profile."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.book_profiles import create_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a book-scoped editorial profile.")
    parser.add_argument("--profile", required=True, help="New profile id, e.g. my_book_mx")
    parser.add_argument("--profiles-root", default="", help="Optional profiles root")
    parser.add_argument("--force", action="store_true", help="Replace an existing profile")
    args = parser.parse_args()

    path = create_profile(
        args.profile,
        profiles_root=args.profiles_root or None,
        force=args.force,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
