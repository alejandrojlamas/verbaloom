"""Ten mandatory publication gates for complete-book jobs."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

from .config import QualityAssuranceConfig
from .models import (
    BookManifest,
    TranslationUnit,
    UnitStatus,
    ValidationIssue,
    total_model_usage,
)
from .validators import ValidationBundle


class GateName(str, Enum):
    EXTRACTION = "extraction"
    SEGMENTATION = "segmentation"
    TRANSLATION = "translation"
    LANGUAGE = "language"
    ENTITIES = "entities"
    SEMANTICS = "semantics"
    STRUCTURE = "structure"
    TYPOGRAPHY = "typography"
    METADATA = "metadata"
    FINAL = "final"


class GateStatus(str, Enum):
    PASSED = "PASSED"
    PASSED_WITH_WARNINGS = "PASSED_WITH_WARNINGS"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass
class QualityGateResult:
    gate: GateName
    status: GateStatus
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in {GateStatus.PASSED, GateStatus.PASSED_WITH_WARNINGS}

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "status": self.status.value,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


@dataclass
class QualityGateReport:
    run_id: str
    document_id: str
    status: GateStatus
    gates: list[QualityGateResult]
    manifest_counts: dict[str, int]
    repair_units: list[dict[str, Any]]
    model_usage: dict[str, Any]
    elapsed_seconds: float = 0.0
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def publishable(self) -> bool:
        return self.status in {GateStatus.PASSED, GateStatus.PASSED_WITH_WARNINGS}

    @property
    def issues(self) -> list[ValidationIssue]:
        return [issue for gate in self.gates for issue in gate.issues]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "document_id": self.document_id,
            "status": self.status.value,
            "publishable": self.publishable,
            "gates": [gate.to_dict() for gate in self.gates],
            "manifest_counts": dict(self.manifest_counts),
            "repair_units": list(self.repair_units),
            "model_usage": dict(self.model_usage),
            "elapsed_seconds": round(float(self.elapsed_seconds), 3),
            "issue_counts": dict(Counter(issue.severity for issue in self.issues)),
        }


def evaluate_quality_gates(
    manifest: BookManifest,
    bundle: ValidationBundle,
    config: QualityAssuranceConfig,
    *,
    elapsed_seconds: float = 0.0,
) -> QualityGateReport:
    issues_by_gate: dict[str, list[ValidationIssue]] = {
        gate.value: [] for gate in GateName
    }
    for issue in bundle.issues:
        gate = issue.gate if issue.gate in issues_by_gate else _gate_for_code(issue.code)
        issues_by_gate[gate].append(issue)

    counts = manifest.counts()
    extraction_issues = issues_by_gate[GateName.EXTRACTION.value]
    if not manifest.units:
        extraction_issues.append(
            ValidationIssue(
                "no_units_extracted",
                "critical",
                "No readable translation units were extracted from the source.",
                gate=GateName.EXTRACTION.value,
            )
        )

    segmentation_issues = issues_by_gate[GateName.SEGMENTATION.value]
    translation_issues = issues_by_gate[GateName.TRANSLATION.value]
    if config.validation.require_full_coverage:
        translated = sum(
            1
            for unit in manifest.translatable_units
            if unit.stage_statuses.get("translation") == "passed" and unit.best_text.strip()
        )
        if translated != len(manifest.translatable_units):
            translation_issues.append(
                ValidationIssue(
                    "incomplete_translation_coverage",
                    "critical",
                    "Not every translatable unit has a valid candidate.",
                    gate=GateName.TRANSLATION.value,
                    details={
                        "translatable": len(manifest.translatable_units),
                        "translated": translated,
                    },
                )
            )

    approved = sum(
        1 for unit in manifest.translatable_units if unit.stage_statuses.get("approval") == "passed"
    )
    if approved != len(manifest.translatable_units):
        issues_by_gate[GateName.FINAL.value].append(
            ValidationIssue(
                "unapproved_units",
                "critical",
                "The book contains translatable units that did not pass every gate.",
                gate=GateName.FINAL.value,
                details={
                    "translatable": len(manifest.translatable_units),
                    "approved": approved,
                },
            )
        )

    gates: list[QualityGateResult] = []
    for gate in GateName:
        gate_issues = _dedupe(issues_by_gate[gate.value])
        metrics = _gate_metrics(gate, manifest, bundle)
        gates.append(
            QualityGateResult(
                gate=gate,
                status=_status_for_issues(gate_issues, config),
                issues=gate_issues,
                metrics=metrics,
            )
        )

    blocking = [gate for gate in gates if gate.status in {GateStatus.BLOCKED, GateStatus.FAILED}]
    warnings = [gate for gate in gates if gate.status == GateStatus.PASSED_WITH_WARNINGS]
    if blocking:
        final_status = GateStatus.BLOCKED
    elif warnings:
        final_status = GateStatus.PASSED_WITH_WARNINGS
    else:
        final_status = GateStatus.PASSED

    repair_units = _repair_units(manifest, gates, config)
    return QualityGateReport(
        run_id=manifest.run_id,
        document_id=manifest.document_id,
        status=final_status,
        gates=gates,
        manifest_counts=manifest.counts(),
        repair_units=repair_units,
        model_usage=total_model_usage(manifest.units),
        elapsed_seconds=elapsed_seconds,
    )


def mark_manifest_exported(manifest: BookManifest) -> None:
    for unit in manifest.units:
        if unit.translatable and unit.stage_statuses.get("approval") != "passed":
            continue
        unit.stage_statuses["export"] = "passed"
        unit.status = UnitStatus.EXPORTED


def _status_for_issues(
    issues: Iterable[ValidationIssue],
    config: QualityAssuranceConfig,
) -> GateStatus:
    values = list(issues)
    if any(issue.blocking for issue in values):
        return GateStatus.BLOCKED
    if values:
        return GateStatus.PASSED_WITH_WARNINGS if config.validation.allow_warnings else GateStatus.BLOCKED
    return GateStatus.PASSED


def _gate_metrics(
    gate: GateName,
    manifest: BookManifest,
    bundle: ValidationBundle,
) -> dict[str, Any]:
    counts = manifest.counts()
    if gate == GateName.EXTRACTION:
        return {"units": len(manifest.units), "excluded": counts["excluded"]}
    if gate == GateName.SEGMENTATION:
        return {
            "unique_ids": len({unit.unit_id for unit in manifest.units}),
            "ordered_units": len(manifest.units),
        }
    if gate == GateName.TRANSLATION:
        return {
            "translatable": counts["translatable"],
            "sent_to_model": counts["sent_to_model"],
            "responded": counts["responded"],
            "translated": counts["translated"],
        }
    if gate == GateName.LANGUAGE:
        return {
            key: bundle.language_report.get(key)
            for key in (
                "source_language",
                "target_language",
                "source_language_ratio",
                "max_source_ratio_document",
            )
        }
    if gate == GateName.ENTITIES:
        return {
            "units_checked": bundle.entity_diff.get("units_checked", 0),
            "issue_count": bundle.entity_diff.get("issue_count", 0),
        }
    if gate == GateName.SEMANTICS:
        return {"audited": counts["audited"], "approved": counts["approved"]}
    if gate == GateName.STRUCTURE:
        return dict(bundle.structure_report)
    if gate == GateName.TYPOGRAPHY:
        return {"units_checked": len(manifest.units)}
    if gate == GateName.METADATA:
        return {"target_locale": manifest.target_locale, "output_format": manifest.output_format}
    return {
        "approved": counts["approved"],
        "exported": counts["exported"],
        "output_path": manifest.output_path,
    }


def _repair_units(
    manifest: BookManifest,
    gates: Iterable[QualityGateResult],
    config: QualityAssuranceConfig,
) -> list[dict[str, Any]]:
    issues_by_unit: dict[str, list[ValidationIssue]] = {}
    for gate in gates:
        for issue in gate.issues:
            if issue.blocking and issue.unit_id:
                issues_by_unit.setdefault(issue.unit_id, []).append(issue)
    units = {unit.unit_id: unit for unit in manifest.units}
    result: list[dict[str, Any]] = []
    for unit_id, issues in sorted(
        issues_by_unit.items(),
        key=lambda item: units[item[0]].order_index if item[0] in units else 10**12,
    ):
        unit = units.get(unit_id)
        if unit is None:
            continue
        result.append(
            {
                "unit_id": unit_id,
                "order_index": unit.order_index,
                "checkpoint_index": unit.source_reference.get("checkpoint_index"),
                "strategy": _repair_strategy(issues),
                "max_rounds": config.retries.max_repair_rounds,
                "issues": [issue.to_dict() for issue in issues],
            }
        )
    return result


def _repair_strategy(issues: Iterable[ValidationIssue]) -> str:
    codes = {issue.code for issue in issues}
    if any("language" in code or "untranslated" in code for code in codes):
        return "strict_target_language"
    if any("entity" in code or "number" in code or "isbn" in code for code in codes):
        return "entity_preserving_repair"
    if any("spacing" in code for code in codes):
        return "deterministic_structure_repair"
    if any("semantic" in code or "negation" in code for code in codes):
        return "source_aware_semantic_repair"
    return "strict_retranslation"


def _gate_for_code(code: str) -> str:
    lowered = str(code or "").lower()
    mapping = (
        ("metadata", GateName.METADATA.value),
        ("language", GateName.LANGUAGE.value),
        ("script", GateName.LANGUAGE.value),
        ("entity", GateName.ENTITIES.value),
        ("number", GateName.ENTITIES.value),
        ("isbn", GateName.ENTITIES.value),
        ("url", GateName.ENTITIES.value),
        ("semantic", GateName.SEMANTICS.value),
        ("omission", GateName.SEMANTICS.value),
        ("addition", GateName.SEMANTICS.value),
        ("negation", GateName.SEMANTICS.value),
        ("spacing", GateName.TYPOGRAPHY.value),
        ("structure", GateName.STRUCTURE.value),
        ("epub", GateName.STRUCTURE.value),
        ("docx", GateName.STRUCTURE.value),
        ("srt", GateName.STRUCTURE.value),
        ("duplicate", GateName.SEGMENTATION.value),
        ("order", GateName.SEGMENTATION.value),
        ("checksum", GateName.SEGMENTATION.value),
        ("source", GateName.EXTRACTION.value),
        ("translation", GateName.TRANSLATION.value),
        ("protocol", GateName.TRANSLATION.value),
    )
    for fragment, gate in mapping:
        if fragment in lowered:
            return gate
    return GateName.FINAL.value


def _dedupe(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    result: list[ValidationIssue] = []
    seen: set[tuple[str, str, str]] = set()
    for issue in issues:
        key = (issue.code, issue.unit_id, issue.message)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result
