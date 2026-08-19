#!/usr/bin/env python3
"""Finalize and validate a repaired EPUB against its source and broken predecessor."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.epub.publication_gate import (
    audit_epub_publication,
    normalize_epub_language_metadata,
    snapshot_epub,
)
from src.core.final_artifact_audit import audit_and_clean_final_artifact
from src.utils.language_detector import LanguageDetector


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[^\W\d_][\w'’.-]*\b", text or "", re.UNICODE))


def _language_words(units) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for unit in units:
        language, _confidence = LanguageDetector.detect_language_from_text(
            unit.text,
            confidence_threshold=0.75,
        )
        counts[str(language or "unknown")] += _word_count(unit.text)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _load_glossary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"entries": [], "source": str(path), "warning": "glossary source not found"}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("entries") if isinstance(payload, dict) else payload
    entries = [dict(item) for item in (entries or []) if isinstance(item, dict)]
    return {
        "schema_version": 1,
        "source": str(path),
        "approved_entries": sum(1 for item in entries if item.get("status") == "approved"),
        "entries": entries,
    }


def _qa_markdown(data: dict[str, Any]) -> str:
    final = data["final"]
    baseline = data["baseline"]
    lines = [
        "# EPUB QA - Los anillos de Saturno",
        "",
        f"- Generated: {data['generated_at']}",
        f"- Source: `{Path(data['inputs']['source']).name}`",
        f"- Broken baseline: `{Path(data['inputs']['broken']).name}`",
        f"- Final: `{Path(data['output']).name}`",
        f"- Result: **{'PASS' if final['publishable'] else 'FAIL'}**",
        "",
        "## Coverage",
        "",
        f"- Spine documents: {final['spine_documents']}",
        f"- Source units: {final['source_units']}",
        f"- Output units: {final['output_units']}",
        f"- Reused units: {final['reused_units']}",
        f"- Retranslated units: {final['retranslated_units']}",
        f"- Reviewed units: {final['reviewed_units']}",
        f"- AUDITED units: {final['audited_units']}",
        f"- Failed/pending units: {final['failed_or_pending_units']}",
        f"- Missing units: {final['missing_units']}",
        f"- Duplicate units: {final['duplicate_units']}",
        "",
        "## Language",
        "",
        f"- Broken source-language units: {baseline['source_language_units']}",
        f"- Broken mixed/residual units: {baseline['mixed_language_units']}",
        f"- Final source-language units: {final['source_language_units']}",
        f"- Final mixed/residual units: {final['mixed_language_units']}",
        f"- Words by detected language before: `{json.dumps(baseline['words_by_language'], ensure_ascii=False)}`",
        f"- Words by detected language after: `{json.dumps(final['words_by_language'], ensure_ascii=False)}`",
        "",
        "## Structure",
        "",
        f"- Images preserved byte-for-byte: {final['preserved_images']}",
        f"- Other resources preserved: {final['preserved_resources']}",
        f"- Broken internal links: {final['broken_links']}",
        f"- Unresolved placeholders/protocol: {final['placeholder_findings']}",
        f"- Obvious sentence-spacing findings: {final['spacing_findings']}",
        f"- `dc:language`: `{final['dc_language']}`",
        f"- ZIP mimetype first/uncompressed: {str(final['mimetype_first_stored']).lower()}",
        "",
        "## EPUBCheck",
        "",
        f"- Errors: {final['epubcheck_errors']}",
        f"- Warnings: {final['epubcheck_warnings']}",
    ]
    if final["warnings"]:
        lines.extend(["", "## Warnings", ""] + [f"- {item}" for item in final["warnings"]])
    if final["errors"]:
        lines.extend(["", "## Blocking errors", ""] + [f"- {item}" for item in final["errors"]])
    lines.extend([
        "",
        "## Verification",
        "",
        f"- Source SHA-256 before/after: `{data['hashes']['source_before']}` / `{data['hashes']['source_after']}`",
        f"- Broken SHA-256 before/after: `{data['hashes']['broken_before']}` / `{data['hashes']['broken_after']}`",
        f"- Final SHA-256: `{data['hashes']['final']}`",
        f"- Test summary: {data.get('test_summary') or 'pending final suite'}",
        "",
        "## Limitations",
        "",
        "- The unit manifest proves one-to-one DOM coverage, language, structural integrity and deterministic source-aware checks.",
        "- Literary nuance remains a human editorial judgment; the platform fidelity reports document the LLM review/repair passes used for the repaired chapters.",
    ])
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--broken", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--test-summary", default="")
    args = parser.parse_args()

    source_before = _sha256(args.source)
    broken_before = _sha256(args.broken)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.candidate, args.output)
    artifact_report = audit_and_clean_final_artifact(
        args.output,
        output_format="epub",
        write_report=False,
        source_epub_path=args.source,
    )
    normalize_epub_language_metadata(args.output, "Spanish")
    # Final deliverables are served by the web process and may be copied by
    # another local user.  Do not inherit a private 0600 mode from a temporary
    # repair artifact.
    args.output.chmod(0o644)

    baseline_gate = audit_epub_publication(
        args.source,
        args.broken,
        source_language="German",
        target_language="Spanish",
        epubcheck_command=["/usr/bin/true"],
    )
    final_gate = audit_epub_publication(
        args.source,
        args.output,
        source_language="German",
        target_language="Spanish",
        broken_path=args.broken,
    )
    source_snapshot = snapshot_epub(args.source, recover=True)
    broken_snapshot = snapshot_epub(args.broken, recover=True)
    final_snapshot = snapshot_epub(
        args.output,
        source_texts=[unit.text for unit in source_snapshot.units],
        recover=False,
    )

    status_counts = Counter(item["status"] for item in final_gate.units)
    reused = sum(1 for item in final_gate.units if item["review"]["status"] == "retained")
    retranslated = len(final_gate.units) - reused
    qa = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {"source": str(args.source), "broken": str(args.broken)},
        "output": str(args.output),
        "hashes": {
            "source_before": source_before,
            "source_after": _sha256(args.source),
            "broken_before": broken_before,
            "broken_after": _sha256(args.broken),
            "final": _sha256(args.output),
        },
        "baseline": {
            "source_units": baseline_gate.source_units,
            "source_language_units": baseline_gate.source_language_units,
            "mixed_language_units": baseline_gate.mixed_language_units,
            "words_by_language": _language_words(broken_snapshot.units),
            "errors": baseline_gate.errors,
        },
        "final": {
            **{key: value for key, value in final_gate.to_dict().items() if key != "units"},
            "spine_documents": len(final_snapshot.spine),
            "reused_units": reused,
            "retranslated_units": retranslated,
            "reviewed_units": len(final_gate.units),
            "failed_or_pending_units": len(final_gate.units) - status_counts.get("AUDITED", 0),
            "words_by_language": _language_words(final_snapshot.units),
            "dc_language": final_snapshot.languages,
            "mimetype_first_stored": final_snapshot.mimetype_first_stored,
            "artifact_cleanup": {
                "changed": artifact_report.changed,
                "text_rewrites": artifact_report.text_rewrites,
                "source_artifacts_removed": artifact_report.source_artifacts_removed,
            },
        },
        "test_summary": args.test_summary,
    }

    (args.reports_dir / "epub_qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.reports_dir / "epub_qa.md").write_text(_qa_markdown(qa), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "pipeline_version": final_gate.to_dict()["pipeline_version"],
        "prompt_version": final_gate.to_dict()["prompt_version"],
        "source_sha256": source_before,
        "output_sha256": _sha256(args.output),
        "total_units": len(final_gate.units),
        "status_counts": dict(status_counts),
        "units": final_gate.units,
    }
    (args.reports_dir / "translation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    glossary = _load_glossary(args.glossary) if args.glossary else {"entries": []}
    (args.reports_dir / "translation_glossary.json").write_text(
        json.dumps(glossary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({
        "publishable": final_gate.publishable,
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "units": f"{final_gate.audited_units}/{final_gate.source_units}",
        "errors": final_gate.errors,
        "warnings": final_gate.warnings,
        "epubcheck_errors": final_gate.epubcheck_errors,
        "epubcheck_warnings": final_gate.epubcheck_warnings,
    }, ensure_ascii=False, indent=2))
    return 0 if final_gate.publishable else 1


if __name__ == "__main__":
    raise SystemExit(main())
