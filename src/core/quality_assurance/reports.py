"""Atomic JSON, JSONL, and HTML reports for book quality runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .gates import QualityGateReport
from .models import BookManifest, atomic_write_text
from .validators import ValidationBundle


@dataclass(frozen=True)
class QualityReportPaths:
    root: Path
    manifest: Path
    report_json: Path
    report_html: Path
    failed_units: Path
    entity_diff: Path
    language_report: Path
    structure_report: Path
    export_report: Path

    @classmethod
    def for_root(cls, root: str | Path) -> "QualityReportPaths":
        directory = Path(root)
        return cls(
            root=directory,
            manifest=directory / "translation_manifest.json",
            report_json=directory / "translation_report.json",
            report_html=directory / "translation_report.html",
            failed_units=directory / "failed_units.jsonl",
            entity_diff=directory / "entity_diff.json",
            language_report=directory / "language_report.json",
            structure_report=directory / "structure_report.json",
            export_report=directory / "export_report.json",
        )


def write_quality_reports(
    root: str | Path,
    *,
    manifest: BookManifest,
    gate_report: QualityGateReport,
    bundle: ValidationBundle,
) -> QualityReportPaths:
    paths = QualityReportPaths.for_root(root)
    paths.root.mkdir(parents=True, exist_ok=True)
    manifest.write_json(paths.manifest)
    report = build_translation_report(manifest, gate_report, bundle)
    _write_json(paths.report_json, report)
    atomic_write_text(paths.report_html, render_translation_report_html(report))
    failed_lines = [
        json.dumps(item, ensure_ascii=False)
        for item in gate_report.repair_units
    ]
    atomic_write_text(
        paths.failed_units,
        ("\n".join(failed_lines) + "\n") if failed_lines else "",
    )
    _write_json(paths.entity_diff, bundle.entity_diff)
    _write_json(paths.language_report, bundle.language_report)
    _write_json(paths.structure_report, bundle.structure_report)
    _write_json(
        paths.export_report,
        {
            "schema_version": 1,
            "run_id": manifest.run_id,
            "status": gate_report.status.value,
            "publishable": gate_report.publishable,
            "output_path": manifest.output_path,
            "output_exists": Path(manifest.output_path).is_file(),
            "output_size_bytes": _file_size(manifest.output_path),
            "output_sha256": _file_sha256(manifest.output_path),
            "source_format": manifest.source_format,
            "output_format": manifest.output_format,
            "quality_gates": [gate.to_dict() for gate in gate_report.gates],
            "manifest_counts": manifest.counts(),
            "repair_units": gate_report.repair_units,
        },
    )
    return paths


def build_translation_report(
    manifest: BookManifest,
    gate_report: QualityGateReport,
    bundle: ValidationBundle,
) -> dict[str, Any]:
    counts = manifest.counts()
    return {
        "schema_version": 1,
        "generated_at": gate_report.generated_at,
        "run_id": manifest.run_id,
        "document_id": manifest.document_id,
        "status": gate_report.status.value,
        "publishable": gate_report.publishable,
        "executive_summary": _executive_summary(manifest, gate_report),
        "source": {
            "path": manifest.source_path,
            "format": manifest.source_format,
            "language": manifest.source_language,
        },
        "output": {
            "path": manifest.output_path,
            "format": manifest.output_format,
            "language": manifest.target_language,
            "locale": manifest.target_locale,
        },
        "models": gate_report.model_usage,
        "cost": {
            "estimated_usd": gate_report.model_usage.get("estimated_cost_usd", 0.0),
            "tokens": gate_report.model_usage.get("total_tokens", 0),
            "prompt_tokens": gate_report.model_usage.get("prompt_tokens", 0),
            "completion_tokens": gate_report.model_usage.get("completion_tokens", 0),
            "cache_hit_tokens": gate_report.model_usage.get("cache_hit_tokens", 0),
        },
        "elapsed_seconds": gate_report.elapsed_seconds,
        "chapters": _chapter_count(manifest),
        "segments": counts,
        "coverage": {
            "translatable": counts["translatable"],
            "translated": counts["translated"],
            "reviewed": counts["reviewed"],
            "audited": counts["audited"],
            "approved": counts["approved"],
            "exported": counts["exported"],
            "percent": round(
                100.0 * counts["approved"] / max(1, counts["translatable"]),
                2,
            ),
        },
        "repaired_segments": sum(1 for unit in manifest.units if unit.retry_count > 0),
        "failed_segments": len(gate_report.repair_units),
        "source_language_ratio": bundle.language_report.get("source_language_ratio", 0.0),
        "entity_discrepancies": bundle.entity_diff.get("issue_count", 0),
        "structural_issues": len(bundle.structure_report.get("issues") or []),
        "epub_validation": (
            bundle.structure_report.get("epub_publication_gate")
            or bundle.structure_report.get("epubcheck")
        ),
        "quality_gates": [gate.to_dict() for gate in gate_report.gates],
        "open_issues": [issue.to_dict() for issue in gate_report.issues],
        "explicit_exclusions": list(manifest.explicit_exclusions),
        "repair_units": gate_report.repair_units,
        "limitations": _limitations(manifest, gate_report),
    }


def render_translation_report_html(report: dict[str, Any]) -> str:
    gates = report.get("quality_gates") or []
    gate_rows = "".join(
        "<tr>"
        f"<td>{escape(str(gate.get('gate') or ''))}</td>"
        f"<td><strong>{escape(str(gate.get('status') or ''))}</strong></td>"
        f"<td>{len(gate.get('issues') or [])}</td>"
        "</tr>"
        for gate in gates
    )
    coverage = report.get("coverage") or {}
    models = report.get("models") or {}
    limitations = report.get("limitations") or []
    limitation_items = "".join(f"<li>{escape(str(item))}</li>" for item in limitations) or "<li>None recorded.</li>"
    issues = report.get("open_issues") or []
    issue_rows = "".join(
        "<tr>"
        f"<td>{escape(str(issue.get('severity') or ''))}</td>"
        f"<td><code>{escape(str(issue.get('code') or ''))}</code></td>"
        f"<td>{escape(str(issue.get('unit_id') or 'book'))}</td>"
        f"<td>{escape(str(issue.get('message') or ''))}</td>"
        "</tr>"
        for issue in issues[:500]
    ) or '<tr><td colspan="4">No open issues.</td></tr>'
    exclusions = report.get("explicit_exclusions") or []
    exclusion_items = "".join(
        f"<li><code>{escape(str(item.get('unit_id') or ''))}</code>: "
        f"{escape(str(item.get('reason') or item.get('content_type') or 'explicit exclusion'))}</li>"
        for item in exclusions[:500]
    ) or "<li>None.</li>"
    epub = report.get("epub_validation") or {}
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Translation quality report</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ max-width: 960px; margin: 0 auto; padding: 32px 20px; line-height: 1.5; }}
    header, section {{ border-bottom: 1px solid #8885; padding-bottom: 20px; margin-bottom: 24px; }}
    .status {{ font-size: 1.4rem; font-weight: 750; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #8885; padding: 12px; border-radius: 6px; }}
    .metric strong {{ display: block; font-size: 1.35rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #8885; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <header>
    <h1>Translation quality report</h1>
    <p class="status">{escape(str(report.get('status') or ''))}</p>
    <p>{escape(str(report.get('executive_summary') or ''))}</p>
  </header>
  <section>
    <h2>Run</h2>
    <p><strong>Run ID:</strong> <code>{escape(str(report.get('run_id') or ''))}</code></p>
    <p><strong>Source:</strong> {escape(str((report.get('source') or {}).get('path') or ''))}</p>
    <p><strong>Output:</strong> {escape(str((report.get('output') or {}).get('path') or ''))}</p>
    <p><strong>Languages:</strong> {escape(str((report.get('source') or {}).get('language') or ''))} to {escape(str((report.get('output') or {}).get('locale') or (report.get('output') or {}).get('language') or ''))}</p>
    <p><strong>Models:</strong> {escape(', '.join(models.get('models') or []) or 'Not recorded')}</p>
  </section>
  <section>
    <h2>Coverage</h2>
    <div class="grid">
      <div class="metric"><span>Translatable</span><strong>{coverage.get('translatable', 0)}</strong></div>
      <div class="metric"><span>Translated</span><strong>{coverage.get('translated', 0)}</strong></div>
      <div class="metric"><span>Reviewed</span><strong>{coverage.get('reviewed', 0)}</strong></div>
      <div class="metric"><span>Audited</span><strong>{coverage.get('audited', 0)}</strong></div>
      <div class="metric"><span>Approved</span><strong>{coverage.get('approved', 0)}</strong></div>
      <div class="metric"><span>Exported</span><strong>{coverage.get('exported', 0)}</strong></div>
      <div class="metric"><span>Source language ratio</span><strong>{float(report.get('source_language_ratio') or 0):.3%}</strong></div>
    </div>
  </section>
  <section>
    <h2>Editorial and structural summary</h2>
    <div class="grid">
      <div class="metric"><span>Chapters or sections</span><strong>{int(report.get('chapters') or 0)}</strong></div>
      <div class="metric"><span>Repaired segments</span><strong>{int(report.get('repaired_segments') or 0)}</strong></div>
      <div class="metric"><span>Failed segments</span><strong>{int(report.get('failed_segments') or 0)}</strong></div>
      <div class="metric"><span>Entity discrepancies</span><strong>{int(report.get('entity_discrepancies') or 0)}</strong></div>
      <div class="metric"><span>Structural issues</span><strong>{int(report.get('structural_issues') or 0)}</strong></div>
      <div class="metric"><span>EPUBCheck errors</span><strong>{_finding_count(epub.get('epubcheck_errors') or epub.get('errors'))}</strong></div>
    </div>
  </section>
  <section>
    <h2>Open issues</h2>
    <table><thead><tr><th>Severity</th><th>Code</th><th>Unit</th><th>Finding</th></tr></thead><tbody>{issue_rows}</tbody></table>
  </section>
  <section><h2>Explicit exclusions</h2><ul>{exclusion_items}</ul></section>
  <section>
    <h2>Quality gates</h2>
    <table><thead><tr><th>Gate</th><th>Status</th><th>Issues</th></tr></thead><tbody>{gate_rows}</tbody></table>
  </section>
  <section>
    <h2>Cost and performance</h2>
    <p><strong>Elapsed:</strong> {float(report.get('elapsed_seconds') or 0):.2f} seconds</p>
    <p><strong>Tokens:</strong> {int((report.get('cost') or {}).get('tokens') or 0):,}</p>
    <p><strong>Estimated cost:</strong> ${float((report.get('cost') or {}).get('estimated_usd') or 0):.6f}</p>
  </section>
  <section><h2>Limitations</h2><ul>{limitation_items}</ul></section>
</body>
</html>
"""


def _finding_count(value: Any) -> int:
    """Normalize numeric counters and serialized finding collections."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def _executive_summary(manifest: BookManifest, report: QualityGateReport) -> str:
    counts = manifest.counts()
    if report.publishable:
        return (
            f"All {counts['translatable']} translatable units passed the mandatory "
            f"coverage, language, entity, semantic, structure, typography, metadata, and final-file gates."
        )
    return (
        f"Publication is blocked. {len(report.repair_units)} unit(s) require selective repair; "
        "the current artifact must not replace a valid final output."
    )


def _chapter_count(manifest: BookManifest) -> int:
    parents = {unit.parent_id for unit in manifest.units if unit.parent_id}
    if parents:
        return len(parents)
    return 1 if int(manifest.metadata.get("source_blocks") or 0) > 0 else 0


def _limitations(manifest: BookManifest, report: QualityGateReport) -> list[str]:
    limitations: list[str] = []
    if manifest.metadata.get("manifest_source") == "artifact_alignment":
        limitations.append(
            "No checkpoint manifest was available; source and output blocks were aligned by structural order."
        )
    if not report.model_usage.get("models"):
        limitations.append("Per-unit model and token metadata were not available for this run.")
    return limitations


def _file_size(path: str | Path) -> int:
    artifact = Path(path)
    return artifact.stat().st_size if artifact.is_file() else 0


def _file_sha256(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        return ""
    digest = hashlib.sha256()
    with artifact.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
