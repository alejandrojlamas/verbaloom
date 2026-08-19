#!/usr/bin/env python3
"""Promote pending suggestions into a profile glossary."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.book_profiles import load_book_profile  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, Mapping) else {}


def _write(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(yaml.safe_dump(dict(data), allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Approve profile glossary suggestions.")
    parser.add_argument("--profile", required=True, help="Active profile id")
    parser.add_argument("--all", action="store_true", help="Approve all pending suggestions")
    parser.add_argument("--min-confidence", type=float, default=0.92)
    parser.add_argument("--target-file", default="terms.yml", help="Glossary file under profile/glossary")
    args = parser.parse_args()

    profile = load_book_profile(args.profile)
    pending_path = profile.root / "glossary" / "pending_suggestions.yml"
    target_path = profile.root / "glossary" / args.target_file
    pending_payload = _read(pending_path)
    target_payload = _read(target_path)
    pending = [
        dict(item) for item in pending_payload.get("suggestions", [])
        if isinstance(item, Mapping)
    ]
    remaining = []
    approved = []
    for item in pending:
        confidence = float(item.get("confidence") or 0)
        if args.all or confidence >= args.min_confidence:
            item["status"] = "approved"
            item.setdefault("scope", profile.profile_id)
            approved.append(item)
        else:
            remaining.append(item)

    entries = [
        dict(item) for item in target_payload.get("entries", [])
        if isinstance(item, Mapping)
    ]
    seen = {str(item.get("source") or "").casefold() for item in entries}
    for item in approved:
        source = str(item.get("source") or "").strip()
        if source and source.casefold() not in seen:
            entries.append(item)
            seen.add(source.casefold())

    _write(target_path, {"entries": entries})
    _write(pending_path, {"suggestions": remaining})
    print(f"approved={len(approved)} remaining={len(remaining)} target={target_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
