#!/usr/bin/env python3
"""Apply a data-only exact-match repair plan to an EPUB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.epub.unit_repairs import apply_epub_unit_repairs, load_epub_unit_repair_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epub", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    repairs, expected_hash = load_epub_unit_repair_plan(args.plan)
    report = apply_epub_unit_repairs(
        args.epub,
        repairs,
        output_epub=args.output,
        expected_epub_sha256=expected_hash,
    )
    payload = report.to_dict()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
