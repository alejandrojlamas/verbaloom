#!/usr/bin/env python3
"""Audit every packaged EPUB block against its source with exact unit IDs."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEEPSEEK_MODEL
from src.core.epub.publication_gate import snapshot_epub
from src.core.epub.semantic_audit import audit_epub_units_semantically
from src.core.llm.factory import create_llm_provider


async def _run(args) -> int:
    source = snapshot_epub(args.source, recover=True)
    output = snapshot_epub(args.output, recover=False)
    provider = create_llm_provider("deepseek", model=args.model)

    def progress(done: int, total: int) -> None:
        print(f"semantic-audit {done}/{total} ({done / max(1, total) * 100:.1f}%)", flush=True)

    report = await audit_epub_units_semantically(
        source,
        output,
        provider=provider,
        model=args.model,
        source_language=args.source_language,
        target_language=args.target_language,
        cache_path=args.cache,
        max_batch_units=args.max_batch_units,
        max_batch_chars=args.max_batch_chars,
        max_attempts=args.max_attempts,
        progress_callback=progress,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "complete": report.complete,
        "audited": len(report.results),
        "passed": report.passed,
        "failed": report.failed,
        "requests": report.requests,
        "cache_hits": report.cache_hits,
        "validation_failures": report.validation_failures,
        "recursive_splits": report.recursive_splits,
        "report": str(args.report),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if report.complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-language", default="German")
    parser.add_argument("--target-language", default="Spanish")
    parser.add_argument("--model", default=DEEPSEEK_MODEL or "deepseek-v4-pro")
    parser.add_argument("--cache", type=Path, default=Path(".translation-work/semantic_audit_cache.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/semantic_audit.json"))
    parser.add_argument("--max-batch-units", type=int, default=6)
    parser.add_argument("--max-batch-chars", type=int, default=24_000)
    parser.add_argument("--max-attempts", type=int, default=2)
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
