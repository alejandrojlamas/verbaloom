"""Whole-book quality assurance orchestration and selective repair planning."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import QualityAssuranceConfig, config_from_job
from .extractors import build_manifest
from .gates import QualityGateReport, evaluate_quality_gates, mark_manifest_exported
from .models import BookManifest
from .reports import QualityReportPaths, write_quality_reports
from .validators import ValidationBundle, validate_manifest


@dataclass
class QualityAssuranceRun:
    manifest: BookManifest
    validation: ValidationBundle
    report: QualityGateReport
    paths: QualityReportPaths
    repair_checkpoint_indices: list[int]

    @property
    def publishable(self) -> bool:
        return self.report.publishable

    def to_dict(self) -> dict[str, Any]:
        return {
            "publishable": self.publishable,
            "status": self.report.status.value,
            "manifest": self.manifest.to_dict(),
            "report": self.report.to_dict(),
            "report_directory": str(self.paths.root),
            "repair_checkpoint_indices": list(self.repair_checkpoint_indices),
        }

    def update_output_path(self, output_path: str | Path) -> None:
        """Keep all persisted reports aligned after atomic publish or quarantine."""

        self.manifest.output_path = str(output_path)
        self.paths = write_quality_reports(
            self.paths.root,
            manifest=self.manifest,
            gate_report=self.report,
            bundle=self.validation,
        )


def run_quality_assurance(
    *,
    source_path: str | Path,
    output_path: str | Path,
    source_language: str,
    target_language: str,
    run_id: str,
    job_config: Mapping[str, Any] | None = None,
    checkpoint_data: Mapping[str, Any] | None = None,
    report_root: str | Path | None = None,
    config: QualityAssuranceConfig | None = None,
    publication_report: Any = None,
    checkpoint_manager: Any = None,
    mark_failed_for_repair: bool = True,
) -> QualityAssuranceRun:
    started = time.monotonic()
    qa_config = config or config_from_job(job_config)
    safe_run_id = _safe_run_id(run_id)
    root = Path(report_root or qa_config.report_root) / safe_run_id
    target_locale = str(
        ((job_config or {}).get("prompt_options") or {}).get("target_locale")
        or qa_config.translation.target_locale
        or target_language
    )
    manifest = build_manifest(
        source_path=source_path,
        output_path=output_path,
        source_language=source_language,
        target_language=target_language,
        target_locale=target_locale,
        run_id=run_id,
        checkpoint_data=checkpoint_data,
        publication_report=publication_report,
    )
    protected_entities = protected_entities_from_job(job_config)
    validation = validate_manifest(
        manifest,
        qa_config,
        protected_entities=protected_entities,
        publication_report=publication_report,
    )
    report = evaluate_quality_gates(
        manifest,
        validation,
        qa_config,
        elapsed_seconds=time.monotonic() - started,
    )
    if report.publishable:
        mark_manifest_exported(manifest)
        report = evaluate_quality_gates(
            manifest,
            validation,
            qa_config,
            elapsed_seconds=time.monotonic() - started,
        )
    repair_indices = sorted(
        {
            int(item["checkpoint_index"])
            for item in report.repair_units
            if item.get("checkpoint_index") is not None
        }
    )
    if (
        repair_indices
        and not report.publishable
        and checkpoint_manager is not None
        and mark_failed_for_repair
    ):
        issue_map = {
            int(item["checkpoint_index"]): item.get("issues") or []
            for item in report.repair_units
            if item.get("checkpoint_index") is not None
        }
        checkpoint_manager.mark_chunks_for_repair(
            run_id,
            repair_indices,
            issues_by_index=issue_map,
        )
    paths = write_quality_reports(
        root,
        manifest=manifest,
        gate_report=report,
        bundle=validation,
    )
    return QualityAssuranceRun(
        manifest=manifest,
        validation=validation,
        report=report,
        paths=paths,
        repair_checkpoint_indices=repair_indices,
    )


def protected_entities_from_job(
    job_config: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    config = dict(job_config or {})
    options = dict(config.get("prompt_options") or {})
    entities: list[dict[str, Any]] = []
    raw_entities = options.get("protected_entities") or []
    if isinstance(raw_entities, Mapping):
        raw_entities = [
            {"source": source, "allowed_targets": value if isinstance(value, list) else [value]}
            for source, value in raw_entities.items()
        ]
    for item in raw_entities:
        if isinstance(item, Mapping) and item.get("source"):
            entities.append(dict(item))

    glossary = options.get("glossary_terms") or {}
    metadata = options.get("glossary_term_metadata") or {}
    if isinstance(glossary, Mapping):
        for source, target in glossary.items():
            source_value = str(source or "").strip()
            target_value = str(target or "").strip()
            if not source_value or not target_value:
                continue
            item_metadata = metadata.get(source, {}) if isinstance(metadata, Mapping) else {}
            entry_type = str((item_metadata or {}).get("type") or "term")
            locked = bool((item_metadata or {}).get("locked", entry_type in {"proper_noun", "publisher", "person", "scientific_name"}))
            if not locked:
                continue
            entities.append(
                {
                    "source": source_value,
                    "allowed_targets": [target_value],
                    "type": entry_type,
                    "locked": True,
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for entity in entities:
        allowed = tuple(str(value) for value in entity.get("allowed_targets") or [])
        key = (str(entity.get("source") or "").casefold(), allowed)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _safe_run_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "run")).strip("._")
    return normalized[:120] or "run"
