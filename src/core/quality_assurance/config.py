"""Centralized configuration for universal quality gates."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class TranslationConfig:
    source_language: str = "auto"
    target_language: str = "es"
    target_locale: str = "es-MX"
    preserve_style: bool = True


@dataclass(frozen=True)
class SegmentationConfig:
    max_chars: int = 4000
    context_before_units: int = 1
    context_after_units: int = 1
    preserve_inline_markup: bool = True


@dataclass(frozen=True)
class RetryConfig:
    max_translation_attempts: int = 3
    max_review_attempts: int = 2
    max_audit_attempts: int = 2
    max_repair_rounds: int = 2


@dataclass(frozen=True)
class ValidationConfig:
    require_full_coverage: bool = True
    block_on_missing_units: bool = True
    block_on_duplicate_units: bool = True
    block_on_source_language_blocks: bool = True
    block_on_critical_entity_diff: bool = True
    block_on_epubcheck_error: bool = True
    block_on_protocol_leak: bool = True
    block_on_spacing_corruption: bool = True
    allow_warnings: bool = True


@dataclass(frozen=True)
class LanguageValidationConfig:
    enabled: bool = True
    max_source_ratio_document: float = 0.01
    max_source_ratio_unit: float = 0.15
    min_chars: int = 40
    allow_named_entities: bool = True
    allow_quoted_foreign_text: bool = True


@dataclass(frozen=True)
class EntityValidationConfig:
    preserve_numbers: bool = True
    preserve_dates: bool = True
    preserve_percentages: bool = True
    preserve_currencies: bool = True
    preserve_measurements: bool = True
    preserve_identifiers: bool = True
    preserve_scientific_names: bool = True
    preserve_urls: bool = True
    preserve_isbn: bool = True
    preserve_protected_names: bool = True


@dataclass(frozen=True)
class ExportConfig:
    keep_intermediate_files: bool = True
    create_html_report: bool = True
    overwrite_valid_output: bool = False
    atomic_publish: bool = True


@dataclass(frozen=True)
class ModelRoutingConfig:
    translator: str = ""
    reviewer: str = ""
    auditor: str = ""
    audit_flagged_only: bool = True


@dataclass(frozen=True)
class QualityAssuranceConfig:
    strict: bool = True
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    retries: RetryConfig = field(default_factory=RetryConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    language_validation: LanguageValidationConfig = field(
        default_factory=LanguageValidationConfig
    )
    entity_validation: EntityValidationConfig = field(
        default_factory=EntityValidationConfig
    )
    export: ExportConfig = field(default_factory=ExportConfig)
    models: ModelRoutingConfig = field(default_factory=ModelRoutingConfig)
    report_root: str = "data/quality_runs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "QualityAssuranceConfig":
        data = dict(value or {})
        return cls(
            strict=_as_bool(data.get("strict"), True),
            translation=_section(TranslationConfig, data.get("translation")),
            segmentation=_section(SegmentationConfig, data.get("segmentation")),
            retries=_section(RetryConfig, data.get("retries")),
            validation=_section(ValidationConfig, data.get("validation")),
            language_validation=_section(
                LanguageValidationConfig, data.get("language_validation")
            ),
            entity_validation=_section(
                EntityValidationConfig, data.get("entity_validation")
            ),
            export=_section(ExportConfig, data.get("export")),
            models=_section(ModelRoutingConfig, data.get("models")),
            report_root=str(data.get("report_root") or "data/quality_runs"),
        )


def load_quality_assurance_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> QualityAssuranceConfig:
    configured_path = path or os.getenv("QUALITY_ASSURANCE_CONFIG")
    data: dict[str, Any] = {}
    if configured_path:
        config_path = Path(configured_path)
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError("Quality assurance configuration must be a mapping.")
            data = dict(loaded)
    if overrides:
        data = _deep_merge(data, dict(overrides))
    return QualityAssuranceConfig.from_mapping(data)


def config_from_job(
    job_config: Mapping[str, Any] | None,
    *,
    strict: bool | None = None,
) -> QualityAssuranceConfig:
    job = dict(job_config or {})
    prompt_options = dict(job.get("prompt_options") or {})
    qa_options = dict(job.get("quality_assurance") or {})
    if strict is None:
        strict = prompt_options.get("strict_quality_assurance", True) is not False
    qa_options["strict"] = bool(strict)
    qa_options["translation"] = {
        **dict(qa_options.get("translation") or {}),
        "source_language": job.get("source_language") or "auto",
        "target_language": job.get("target_language") or "",
        "target_locale": prompt_options.get("target_locale") or job.get("target_language") or "",
    }
    return QualityAssuranceConfig.from_mapping(qa_options)


def _section(cls, value: Any):
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        return cls()
    allowed = cls.__dataclass_fields__.keys()
    return cls(**{key: item for key, item in value.items() if key in allowed})


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
