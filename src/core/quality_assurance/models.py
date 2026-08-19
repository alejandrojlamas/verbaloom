"""Stable data contracts for whole-book translation quality assurance."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

QA_SCHEMA_VERSION = 1
QA_PIPELINE_VERSION = "universal-book-qa-v1"


class UnitStatus(str, Enum):
    EXTRACTED = "extracted"
    SEGMENTED = "segmented"
    QUEUED = "queued"
    TRANSLATED = "translated"
    TRANSLATION_FAILED = "translation_failed"
    REVIEWED = "reviewed"
    REVIEW_FAILED = "review_failed"
    AUDITED = "audited"
    AUDIT_FAILED = "audit_failed"
    APPROVED = "approved"
    BLOCKED = "blocked"
    EXPORTED = "exported"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    unit_id: str = ""
    gate: str = ""
    source_span: str = ""
    target_span: str = ""
    suggested_fix: str = ""
    confidence: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity.lower() in {"critical", "high", "error", "reject"}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationResult:
    validator: str
    version: str
    passed: bool
    issues: tuple[ValidationIssue, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "validator": self.validator,
            "version": self.version,
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
            "checked_at": self.checked_at,
        }


@dataclass(frozen=True)
class ModelMetadata:
    provider: str = ""
    model: str = ""
    prompt_version: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cache_hit_tokens: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ModelMetadata":
        data = value or {}
        return cls(
            provider=str(data.get("provider") or data.get("llm_provider") or ""),
            model=str(data.get("model") or data.get("model_name") or ""),
            prompt_version=str(data.get("prompt_version") or ""),
            prompt_tokens=_as_int(data.get("prompt_tokens")),
            completion_tokens=_as_int(data.get("completion_tokens")),
            total_tokens=_as_int(data.get("total_tokens") or data.get("context_used")),
            estimated_cost_usd=_as_float(data.get("estimated_cost_usd") or data.get("cost")),
            cache_hit_tokens=_as_int(data.get("prompt_cache_hit_tokens")),
        )


@dataclass
class TranslationUnit:
    document_id: str
    unit_id: str
    parent_id: str
    order_index: int
    source_text: str
    source_language: str
    target_language: str
    translated_text: str = ""
    reviewed_text: str = ""
    final_text: str = ""
    content_type: str = "narrative"
    policy: str = "translate"
    structural_path: str = ""
    html_tag: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    preceding_whitespace: str = ""
    trailing_whitespace: str = ""
    checksum_source: str = ""
    checksum_structure: str = ""
    status: UnitStatus = UnitStatus.EXTRACTED
    retry_count: int = 0
    stage_statuses: dict[str, str] = field(default_factory=dict)
    validation_results: list[ValidationResult] = field(default_factory=list)
    model_metadata: ModelMetadata = field(default_factory=ModelMetadata)
    exclusion_reason: str = ""
    source_reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        order_index: int,
        source_text: str,
        source_language: str,
        target_language: str,
        translated_text: str = "",
        reviewed_text: str = "",
        final_text: str = "",
        parent_id: str = "",
        content_type: str = "narrative",
        policy: str = "translate",
        structural_path: str = "",
        html_tag: str = "",
        attributes: Mapping[str, Any] | None = None,
        preceding_whitespace: str = "",
        trailing_whitespace: str = "",
        status: UnitStatus = UnitStatus.EXTRACTED,
        retry_count: int = 0,
        model_metadata: ModelMetadata | None = None,
        exclusion_reason: str = "",
        source_reference: Mapping[str, Any] | None = None,
    ) -> "TranslationUnit":
        normalized_attributes = {
            str(key): str(value) for key, value in (attributes or {}).items()
        }
        normalized_content_type = content_type or "narrative"
        normalized_policy = policy or "translate"
        normalized_structural_path = structural_path or f"/units/{order_index}"
        source_checksum = stable_checksum(source_text)
        structure_checksum = stable_checksum(
            json.dumps(
                _structure_payload(
                    parent_id=parent_id,
                    order_index=int(order_index),
                    content_type=normalized_content_type,
                    policy=normalized_policy,
                    structural_path=normalized_structural_path,
                    html_tag=html_tag,
                    attributes=normalized_attributes,
                    preceding_whitespace=preceding_whitespace,
                    trailing_whitespace=trailing_whitespace,
                ),
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        identity_payload = "\x1f".join(
            (
                document_id,
                parent_id,
                str(order_index),
                normalized_structural_path,
                source_checksum,
                structure_checksum,
            )
        )
        unit_hash = stable_checksum(identity_payload)[:24]
        stage_statuses = {
            "extraction": "passed",
            "segmentation": "passed",
        }
        if status in {
            UnitStatus.TRANSLATED,
            UnitStatus.REVIEWED,
            UnitStatus.AUDITED,
            UnitStatus.APPROVED,
            UnitStatus.EXPORTED,
        }:
            stage_statuses["translation"] = "passed"
        elif status == UnitStatus.TRANSLATION_FAILED:
            stage_statuses["translation"] = "failed"
        if status in {UnitStatus.REVIEWED, UnitStatus.AUDITED, UnitStatus.APPROVED, UnitStatus.EXPORTED}:
            stage_statuses["review"] = "passed"
        if status in {UnitStatus.AUDITED, UnitStatus.APPROVED, UnitStatus.EXPORTED}:
            stage_statuses["audit"] = "passed"
        return cls(
            document_id=document_id,
            unit_id=f"unit:{order_index:08d}:{unit_hash}",
            parent_id=parent_id,
            order_index=int(order_index),
            source_text=source_text or "",
            source_language=source_language or "",
            target_language=target_language or "",
            translated_text=translated_text or "",
            reviewed_text=reviewed_text or "",
            final_text=final_text or reviewed_text or translated_text or "",
            content_type=normalized_content_type,
            policy=normalized_policy,
            structural_path=normalized_structural_path,
            html_tag=html_tag or "",
            attributes=normalized_attributes,
            preceding_whitespace=preceding_whitespace,
            trailing_whitespace=trailing_whitespace,
            checksum_source=source_checksum,
            checksum_structure=structure_checksum,
            status=status,
            retry_count=max(0, int(retry_count or 0)),
            stage_statuses=stage_statuses,
            model_metadata=model_metadata or ModelMetadata(),
            exclusion_reason=exclusion_reason,
            source_reference=dict(source_reference or {}),
        )

    @property
    def translatable(self) -> bool:
        return self.policy in {"translate", "reconstruct"} and not self.exclusion_reason

    @property
    def best_text(self) -> str:
        return self.final_text or self.reviewed_text or self.translated_text

    def add_validation(self, result: ValidationResult) -> None:
        self.validation_results.append(result)

    def mark_stage(self, stage: str, status: str) -> None:
        self.stage_statuses[str(stage)] = str(status)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["validation_results"] = [item.to_dict() for item in self.validation_results]
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TranslationUnit":
        data = dict(value)
        data["status"] = UnitStatus(str(data.get("status") or UnitStatus.EXTRACTED.value))
        data["model_metadata"] = ModelMetadata.from_mapping(data.get("model_metadata"))
        data["validation_results"] = [
            _validation_result_from_dict(item)
            for item in data.get("validation_results") or []
        ]
        return cls(**data)


@dataclass
class BookManifest:
    document_id: str
    source_path: str
    output_path: str
    source_language: str
    target_language: str
    target_locale: str = ""
    source_format: str = ""
    output_format: str = ""
    run_id: str = ""
    units: list[TranslationUnit] = field(default_factory=list)
    explicit_exclusions: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = QA_SCHEMA_VERSION
    pipeline_version: str = QA_PIPELINE_VERSION
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def translatable_units(self) -> list[TranslationUnit]:
        return [unit for unit in self.units if unit.translatable]

    def counts(self) -> dict[str, int]:
        by_status = {status.value: 0 for status in UnitStatus}
        for unit in self.units:
            by_status[unit.status.value] += 1
        stages = {
            stage: sum(1 for unit in self.units if unit.stage_statuses.get(stage) == "passed")
            for stage in ("extraction", "segmentation", "translation", "review", "audit", "approval", "export")
        }
        return {
            "extracted": len(self.units),
            "translatable": len(self.translatable_units),
            "excluded": len(self.units) - len(self.translatable_units),
            "sent_to_model": sum(
                1 for unit in self.translatable_units
                if unit.stage_statuses.get("translation") in {"passed", "failed"}
            ),
            "responded": sum(
                1 for unit in self.translatable_units if bool(unit.translated_text)
            ),
            "translated": stages["translation"],
            "reviewed": stages["review"],
            "audited": stages["audit"],
            "approved": stages["approval"],
            "exported": stages["export"],
            **{f"status_{status}": count for status, count in by_status.items()},
        }

    def validate_identity(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        ids = [unit.unit_id for unit in self.units]
        if len(ids) != len(set(ids)):
            issues.append(
                ValidationIssue(
                    "duplicate_unit_id",
                    "critical",
                    "The manifest contains duplicate deterministic unit IDs.",
                    gate="segmentation",
                )
            )
        orders = [unit.order_index for unit in self.units]
        contiguous = (
            not orders
            or (
                len(set(orders)) == len(orders)
                and min(orders) == 0
                and max(orders) == len(orders) - 1
            )
        )
        if not contiguous:
            issues.append(
                ValidationIssue(
                    "non_contiguous_order",
                    "critical",
                    "Unit order indexes are missing, duplicated, or out of range.",
                    gate="segmentation",
                    details={
                        "expected_count": len(self.units),
                        "actual_sample": sorted(orders)[:20],
                    },
                )
            )
        if any(left > right for left, right in zip(orders, orders[1:])):
            issues.append(
                ValidationIssue(
                    "unit_order_changed",
                    "critical",
                    "Manifest units are not stored in source order.",
                    gate="structure",
                )
            )
        for unit in self.units:
            if stable_checksum(unit.source_text) != unit.checksum_source:
                issues.append(
                    ValidationIssue(
                        "source_checksum_mismatch",
                        "critical",
                        "A unit source checksum no longer matches its source text.",
                        unit_id=unit.unit_id,
                        gate="segmentation",
                    )
                )
            expected_structure = stable_checksum(
                json.dumps(
                    _structure_payload(
                        parent_id=unit.parent_id,
                        order_index=unit.order_index,
                        content_type=unit.content_type,
                        policy=unit.policy,
                        structural_path=unit.structural_path,
                        html_tag=unit.html_tag,
                        attributes=unit.attributes,
                        preceding_whitespace=unit.preceding_whitespace,
                        trailing_whitespace=unit.trailing_whitespace,
                    ),
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
            if expected_structure != unit.checksum_structure:
                issues.append(
                    ValidationIssue(
                        "structure_checksum_mismatch",
                        "critical",
                        "A unit structure checksum no longer matches its structural metadata.",
                        unit_id=unit.unit_id,
                        gate="structure",
                    )
                )
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at,
            "document_id": self.document_id,
            "run_id": self.run_id,
            "source_path": self.source_path,
            "output_path": self.output_path,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "target_locale": self.target_locale,
            "source_format": self.source_format,
            "output_format": self.output_format,
            "counts": self.counts(),
            "explicit_exclusions": list(self.explicit_exclusions),
            "metadata": dict(self.metadata),
            "units": [unit.to_dict() for unit in self.units],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BookManifest":
        data = dict(value)
        data.pop("counts", None)
        data["units"] = [TranslationUnit.from_dict(item) for item in data.get("units") or []]
        return cls(**data)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        atomic_write_text(
            destination,
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
        )
        return destination


def stable_checksum(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else (value or "").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def document_id_for_path(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"doc:{digest.hexdigest()[:32]}"


def atomic_write_text(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _structure_payload(
    *,
    parent_id: str,
    order_index: int,
    content_type: str,
    policy: str,
    structural_path: str,
    html_tag: str,
    attributes: Mapping[str, Any],
    preceding_whitespace: str,
    trailing_whitespace: str,
) -> dict[str, Any]:
    return {
        "parent_id": parent_id,
        "order_index": int(order_index),
        "content_type": content_type,
        "policy": policy,
        "structural_path": structural_path,
        "html_tag": html_tag,
        "attributes": {str(key): str(value) for key, value in attributes.items()},
        "preceding_whitespace": preceding_whitespace,
        "trailing_whitespace": trailing_whitespace,
    }


def total_model_usage(units: Iterable[TranslationUnit]) -> dict[str, Any]:
    prompt = completion = total = cache = 0
    cost = 0.0
    models: set[str] = set()
    providers: set[str] = set()
    for unit in units:
        metadata = unit.model_metadata
        prompt += metadata.prompt_tokens
        completion += metadata.completion_tokens
        total += metadata.total_tokens or metadata.prompt_tokens + metadata.completion_tokens
        cache += metadata.cache_hit_tokens
        cost += metadata.estimated_cost_usd
        if metadata.model:
            models.add(metadata.model)
        if metadata.provider:
            providers.add(metadata.provider)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cache_hit_tokens": cache,
        "estimated_cost_usd": round(cost, 8),
        "models": sorted(models),
        "providers": sorted(providers),
    }


def _validation_result_from_dict(value: Mapping[str, Any]) -> ValidationResult:
    issues = tuple(
        ValidationIssue(**dict(item)) for item in value.get("issues") or []
    )
    return ValidationResult(
        validator=str(value.get("validator") or "unknown"),
        version=str(value.get("version") or ""),
        passed=bool(value.get("passed")),
        issues=issues,
        metrics=dict(value.get("metrics") or {}),
        checked_at=str(value.get("checked_at") or datetime.now(timezone.utc).isoformat()),
    )


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
