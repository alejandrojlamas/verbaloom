#!/usr/bin/env python3
"""Adjudicate non-passing EPUB semantic findings without external translations."""

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
from src.core.epub.semantic_audit import SemanticAuditReport, adjudicate_semantic_audit
from src.core.llm.factory import create_llm_provider


async def _run(args) -> int:
    source = snapshot_epub(args.source, recover=True)
    output = snapshot_epub(args.output, recover=False)
    report = SemanticAuditReport.from_dict(json.loads(args.report.read_text(encoding="utf-8")))
    provider = create_llm_provider("deepseek", model=args.model)

    def progress(done: int, total: int) -> None:
        print(f"semantic-adjudication {done}/{total} ({done / max(1, total) * 100:.1f}%)", flush=True)

    updated = await adjudicate_semantic_audit(
        source, output, report, provider=provider, model=args.model,
        source_language=args.source_language, target_language=args.target_language,
        cache_path=args.cache, max_attempts=args.max_attempts,
        recheck_initial_findings=args.recheck_initial_findings,
        progress_callback=progress,
    )
    args.report.write_text(json.dumps(updated.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "complete": updated.complete,
        "passed": updated.passed,
        "failed": updated.failed,
        "requests": updated.requests,
        "validation_failures": updated.validation_failures,
        "recursive_splits": updated.recursive_splits,
        "report": str(args.report),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if updated.complete else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".translation-work/semantic_adjudication_cache.json"))
    parser.add_argument("--model", default=DEEPSEEK_MODEL or "deepseek-v4-pro")
    parser.add_argument("--source-language", default="German")
    parser.add_argument("--target-language", default="Spanish")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--recheck-initial-findings", action="store_true")
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
