"""Deterministic validators used by universal whole-book quality gates."""

from __future__ import annotations

import posixpath
import re
import shutil
import subprocess
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

from lxml import etree

from src.core.fidelity_supervisor import target_language_gate_issues
from src.core.llm_output_guard import guard_llm_output
from src.core.output_formats import extract_readable_text
from src.utils.language_detector import LANGUAGE_CODE_MAP, LanguageDetector
from src.utils.text_encoding import mojibake_score

from .config import QualityAssuranceConfig
from .models import (
    BookManifest,
    TranslationUnit,
    UnitStatus,
    ValidationIssue,
    ValidationResult,
)

VALIDATOR_VERSION = "universal-deterministic-validators-v1"

_DIGIT_GROUP_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?!\w)"
)
_DATE_RE = re.compile(
    r"(?<!\w)(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})(?!\w)"
)
_PERCENT_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:%|‰)")
_CURRENCY_RE = re.compile(
    r"(?i)(?<!\w)(?:[$€£¥₹₽₩₺₴]|USD|EUR|GBP|JPY|CNY|RMB|MXN|CAD|AUD|CHF|BRL|INR|RUB)"
    r"\s*\d+(?:[.,]\d+)*(?!\w)|(?<!\w)\d+(?:[.,]\d+)*\s*"
    r"(?:[$€£¥₹₽₩₺₴]|USD|EUR|GBP|JPY|CNY|RMB|MXN|CAD|AUD|CHF|BRL|INR|RUB)(?!\w)"
)
_MEASUREMENT_RE = re.compile(
    r"(?<!\w)(\d+(?:[.,]\d+)?)(?:\s*("
    r"meters?|metres?|metros?|metern?|meter|metre|metro|"
    r"km|cm|mm|kg|mg|lb|oz|°C|°F|kHz|MHz|GHz|kW|mL|Hz|Pa|bar|mol"
    r")|\s+(m|g|L|K|W|V|A))(?!\w)",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(
    r"(?i)\b(?:doi\s*:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+|"
    r"ORCID\s*:?\s*\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b"
)
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.IGNORECASE)
_ISBN_RE = re.compile(r"(?i)\bISBN(?:-1[03])?\s*:?[\s-]*([0-9X][0-9X\s-]{8,20})")
_SCIENTIFIC_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\s+([a-z][a-z-]{2,})\b")
_SCIENTIFIC_PREFIX_RE = re.compile(
    r"(?:\b(?:species|genus|binomial|taxon|scientific\s+name|described|"
    r"identified\s+as|classified\s+as|especie|g[eé]nero|tax[oó]n|"
    r"nombre\s+cient[ií]fico|describi[oó]|identificad[oa]\s+como|"
    r"clasificad[oa]\s+como|esp[eè]ce|genre|nom\s+scientifique|Gattung|"
    r"Spezies|wissenschaftlicher\s+Name)\s+(?:as\s+|como\s+)?$)",
    re.IGNORECASE,
)
_NON_SCIENTIFIC_GENUS = {
    "After", "Before", "First", "Second", "That", "The", "These", "This", "Those",
    "When", "Where", "While", "Der", "Die", "Das", "Ein", "Eine", "El", "La",
    "Los", "Las", "Un", "Una", "En", "Al", "Del", "Le", "Les", "Une",
    "Son", "Soy", "Somos", "Eres", "Es", "Fue", "Fueron",
}
_NON_SCIENTIFIC_SPECIES = {
    "a", "an", "the", "un", "una", "uno", "el", "la", "los", "las",
    "une", "des", "der", "die", "das", "ein", "eine", "son", "es",
}
_PROSE_VERB_SUFFIX_RE = re.compile(
    r"(?:ando|iendo|yendo|aba|aban|aron|ando|iendo|ing|ed)$",
    re.IGNORECASE,
)
_MEASUREMENT_UNIT_CANONICAL = {
    "meter": "M", "meters": "M", "metre": "M", "metres": "M",
    "metro": "M", "metros": "M", "metern": "M",
}
_CASE_SENSITIVE_SHORT_MEASUREMENT_UNITS = {
    "m": "M",
    "g": "G",
    "L": "L",
    "K": "K",
    "W": "W",
    "V": "V",
    "A": "A",
}
_TITLECASE_WORD_AFTER_MEASUREMENT_RE = re.compile(
    r"\s+([A-Z][a-z\u00c0-\u024f'’-]{1,})\b"
)
_LOWERCASE_FRAGMENT_AFTER_MEASUREMENT_RE = re.compile(
    r"\s+([a-z\u00df-\u024f'’-]{2,})\b"
)
_PROPER_NAME_RE = re.compile(
    r"\b(?:[A-Z][\w'’-]{2,})(?:\s+(?:de|del|da|dos|van|von|la|le|[A-Z][\w'’-]{1,})){0,4}\b"
)
_SPACING_PATTERNS = (
    ("period_join", re.compile(r"(?<=[a-z\u00c0-\u024f0-9])[.](?=[A-Z\u00c0-\u024f])")),
    ("comma_join", re.compile(r"(?<=[a-z\u00c0-\u024f]),(?=[A-Z\u00c0-\u024f])")),
    ("semicolon_join", re.compile(r"(?<=[a-z\u00c0-\u024f]);(?=[A-Z\u00c0-\u024f])")),
    ("colon_join", re.compile(r"(?<=[a-z\u00c0-\u024f]):(?=[A-Z\u00c0-\u024f])")),
    ("word_join", re.compile(r"(?<=[a-z\u00df-\u024f])(?=[A-Z\u00c0-\u00de][a-z\u00df-\u024f]{2,})")),
)
_BCP47_RE = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_NEGATION_MARKERS = {
    "no", "not", "never", "nicht", "kein", "keine", "jamais", "ne", "pas",
    "non", "nunca", "sin", "sans", "senza", "nao", "nao", "нет", "не",
    "μη", "δεν", "无", "不", "未", "ない", "ません",
}
_QUOTE_RE = re.compile(r"(?:[\"'“”‘’«»][^\"'“”‘’«»]{8,}[\"'“”‘’«»])")
_DENSE_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


@dataclass
class ValidationBundle:
    unit_results: dict[str, list[ValidationResult]] = field(default_factory=dict)
    document_results: list[ValidationResult] = field(default_factory=list)
    entity_diff: dict[str, Any] = field(default_factory=dict)
    language_report: dict[str, Any] = field(default_factory=dict)
    structure_report: dict[str, Any] = field(default_factory=dict)

    @property
    def issues(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for results in self.unit_results.values():
            for result in results:
                issues.extend(result.issues)
        for result in self.document_results:
            issues.extend(result.issues)
        return issues


def validate_manifest(
    manifest: BookManifest,
    config: QualityAssuranceConfig,
    *,
    protected_entities: Iterable[Mapping[str, Any]] = (),
    publication_report: Any = None,
) -> ValidationBundle:
    bundle = ValidationBundle()
    identity_issues = manifest.validate_identity()
    bundle.document_results.append(
        ValidationResult(
            validator="manifest_identity",
            version=VALIDATOR_VERSION,
            passed=not any(issue.blocking for issue in identity_issues),
            issues=tuple(identity_issues),
            metrics={"units": len(manifest.units)},
        )
    )

    source_language_chars = 0
    checked_language_chars = 0
    language_units: list[dict[str, Any]] = []
    section_language: dict[str, dict[str, int]] = {}
    entity_details: list[dict[str, Any]] = []

    for unit in manifest.units:
        results: list[ValidationResult] = []
        translation_result = validate_translation_unit(unit, config)
        results.append(translation_result)
        language_result = validate_unit_language(unit, config)
        results.append(language_result)
        entity_result, entity_detail = validate_unit_entities(
            unit,
            config,
            protected_entities=protected_entities,
        )
        results.append(entity_result)
        entity_details.append(entity_detail)
        semantic_result = validate_unit_semantics(unit, config)
        results.append(semantic_result)
        typography_result = validate_unit_typography(unit, config)
        results.append(typography_result)
        bundle.unit_results[unit.unit_id] = results
        for result in results:
            unit.add_validation(result)

        unit.mark_stage("translation", "passed" if translation_result.passed else "failed")
        unit.mark_stage("review", "passed" if translation_result.passed else "failed")
        unit.mark_stage(
            "audit",
            "passed" if all(result.passed for result in results) else "failed",
        )
        blocking = any(issue.blocking for result in results for issue in result.issues)
        if blocking:
            unit.status = UnitStatus.BLOCKED
            unit.mark_stage("approval", "failed")
        else:
            unit.status = UnitStatus.APPROVED
            unit.mark_stage("approval", "passed")

        if unit.translatable and _enough_language_evidence(
            unit.best_text,
            config.language_validation.min_chars,
        ):
            checked_language_chars += len(unit.best_text)
            detected = str(language_result.metrics.get("detected_language") or "")
            confidence = float(language_result.metrics.get("confidence") or 0.0)
            is_source = confidence >= 0.85 and _same_language(
                detected, unit.source_language
            ) and not _same_language(
                unit.source_language, unit.target_language
            ) and not _legitimate_foreign_quote(unit, config)
            if is_source:
                source_language_chars += len(unit.best_text)
            language_units.append(
                {
                    "unit_id": unit.unit_id,
                    "order_index": unit.order_index,
                    "detected_language": detected,
                    "confidence": confidence,
                    "source_language": is_source,
                    "issues": [issue.to_dict() for issue in language_result.issues],
                }
            )
            section_id = unit.parent_id or _section_key(unit.structural_path)
            section = section_language.setdefault(
                section_id,
                {"checked_characters": 0, "source_language_characters": 0},
            )
            section["checked_characters"] += len(unit.best_text)
            if is_source:
                section["source_language_characters"] += len(unit.best_text)

    duplicate_issues = _duplicate_output_issues(manifest)
    alignment_issues = _artifact_alignment_issues(manifest)
    bundle.document_results.append(
        ValidationResult(
            validator="document_coverage",
            version=VALIDATOR_VERSION,
            passed=not any(issue.blocking for issue in (*duplicate_issues, *alignment_issues)),
            issues=tuple((*duplicate_issues, *alignment_issues)),
            metrics={
                "duplicate_outputs": len(duplicate_issues),
                "source_blocks": manifest.metadata.get("source_blocks"),
                "output_blocks": manifest.metadata.get("output_blocks"),
                "unmapped_output_blocks": manifest.metadata.get("unmapped_output_blocks", 0),
            },
        )
    )

    source_ratio = source_language_chars / max(1, checked_language_chars)
    document_language_issues: list[ValidationIssue] = []
    section_rows: list[dict[str, Any]] = []
    for section_id, section in sorted(section_language.items()):
        ratio = section["source_language_characters"] / max(
            1, section["checked_characters"]
        )
        section_rows.append({"section_id": section_id, **section, "source_language_ratio": round(ratio, 6)})
        if (
            config.language_validation.enabled
            and not _same_language(manifest.source_language, manifest.target_language)
            and ratio > config.language_validation.max_source_ratio_unit
            and section["source_language_characters"] >= config.language_validation.min_chars
        ):
            document_language_issues.append(
                ValidationIssue(
                    "section_source_language_ratio",
                    "critical",
                    "A chapter or section exceeds the configured source-language ratio.",
                    gate="language",
                    details={
                        "section_id": section_id,
                        "ratio": round(ratio, 6),
                        "maximum": config.language_validation.max_source_ratio_unit,
                    },
                )
            )
    if (
        config.language_validation.enabled
        and not _same_language(manifest.source_language, manifest.target_language)
        and source_ratio > config.language_validation.max_source_ratio_document
    ):
        document_language_issues.append(
            ValidationIssue(
                "document_source_language_ratio",
                "critical",
                "The final document exceeds the configured source-language ratio.",
                gate="language",
                confidence=1.0,
                details={
                    "ratio": round(source_ratio, 6),
                    "maximum": config.language_validation.max_source_ratio_document,
                    "source_language_characters": source_language_chars,
                    "checked_characters": checked_language_chars,
                },
            )
        )
    bundle.document_results.append(
        ValidationResult(
            validator="document_language",
            version=VALIDATOR_VERSION,
            passed=not document_language_issues,
            issues=tuple(document_language_issues),
            metrics={
                "source_language_ratio": round(source_ratio, 6),
                "source_language_characters": source_language_chars,
                "checked_characters": checked_language_chars,
            },
        )
    )
    bundle.language_report = {
        "source_language": manifest.source_language,
        "target_language": manifest.target_language,
        "target_locale": manifest.target_locale,
        "max_source_ratio_document": config.language_validation.max_source_ratio_document,
        "max_source_ratio_unit": config.language_validation.max_source_ratio_unit,
        "source_language_ratio": round(source_ratio, 6),
        "units": language_units,
        "sections": section_rows,
        "issues": [issue.to_dict() for issue in document_language_issues],
    }

    bundle.entity_diff = _build_entity_report(entity_details)
    artifact_result, structure_report = validate_output_artifact(
        manifest,
        config,
        publication_report=publication_report,
    )
    bundle.document_results.append(artifact_result)
    bundle.structure_report = structure_report
    return bundle


def validate_translation_unit(
    unit: TranslationUnit,
    config: QualityAssuranceConfig,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if not unit.translatable:
        return ValidationResult(
            validator="translation_coverage",
            version=VALIDATOR_VERSION,
            passed=True,
            metrics={"excluded": True, "reason": unit.exclusion_reason},
        )
    target = unit.best_text.strip()
    source = unit.source_text.strip()
    if not source:
        issues.append(
            ValidationIssue(
                "missing_source_unit",
                "critical",
                "A declared translatable unit has no source text.",
                unit_id=unit.unit_id,
                gate="extraction",
            )
        )
    if not target:
        issues.append(
            ValidationIssue(
                "empty_translation",
                "critical",
                "A translatable unit has no translated output.",
                unit_id=unit.unit_id,
                gate="translation",
            )
        )
    if (
        source
        and target
        and not _same_language(unit.source_language, unit.target_language)
        and _normalized(source) == _normalized(target)
        and _enough_language_evidence(source, config.language_validation.min_chars)
        and not _looks_like_protected_name(source)
        and not _legitimate_foreign_quote(unit, config)
    ):
        issues.append(
            ValidationIssue(
                "untranslated_source",
                "critical",
                "The output is identical to a source unit that should be translated.",
                unit_id=unit.unit_id,
                gate="translation",
                source_span=_snippet(source),
                target_span=_snippet(target),
                suggested_fix="Retry this unit with the strict target-language repair prompt.",
            )
        )
    protocol = guard_llm_output(target, phase="final_quality_gate") if target else None
    for item in getattr(protocol, "issues", ()):
        issue_code = str(getattr(item, "code", "model_protocol_leak"))
        issue_severity = _output_guard_issue_severity(item)
        issues.append(
            ValidationIssue(
                issue_code,
                issue_severity,
                str(getattr(item, "message", ""))
                or "Reader-visible model protocol or an incomplete response was detected.",
                unit_id=unit.unit_id,
                gate="translation",
                target_span=_snippet(target),
                suggested_fix="Regenerate only this unit and validate the structured response.",
                details={
                    "guard_severity": str(getattr(item, "severity", "")),
                    "detail": str(getattr(item, "detail", "")),
                },
            )
        )
    if unit.source_reference.get("checkpoint_missing"):
        issues.append(
            ValidationIssue(
                "missing_checkpoint_unit",
                "critical",
                "The checkpoint manifest has a gap at this order index.",
                unit_id=unit.unit_id,
                gate="coverage",
            )
        )
    return ValidationResult(
        validator="translation_coverage",
        version=VALIDATOR_VERSION,
        passed=not any(issue.blocking for issue in issues),
        issues=tuple(issues),
        metrics={"source_chars": len(source), "target_chars": len(target)},
    )


def validate_unit_language(
    unit: TranslationUnit,
    config: QualityAssuranceConfig,
) -> ValidationResult:
    target = unit.best_text
    if (
        not config.language_validation.enabled
        or not unit.translatable
        or _same_language(unit.source_language, unit.target_language)
        or not _enough_language_evidence(target, config.language_validation.min_chars)
    ):
        return ValidationResult(
            validator="target_language",
            version=VALIDATOR_VERSION,
            passed=True,
            metrics={"skipped": True},
        )
    detected, confidence = LanguageDetector.detect_language_from_text(
        target,
        confidence_threshold=0.0,
    )
    issues: list[ValidationIssue] = []
    # A publication-audited EPUB unit is a whole XHTML document assembled from
    # many already-audited blocks. Phrase-level residual heuristics are not
    # calibrated for that aggregate and mistake names, quotes and front matter
    # for untranslated prose. Whole-unit language detection remains active, so
    # a genuinely untranslated chapter is still blocked.
    publication_audited = bool(unit.source_reference.get("publication_audited"))
    local_issues = []
    if not publication_audited:
        local_issues = target_language_gate_issues(
            unit.source_text,
            target,
            source_language=unit.source_language,
            target_language=unit.target_language,
            phase="final_quality_gate",
            prompt_options={
                "target_language_gate": "on",
                "min_target_language_chars": config.language_validation.min_chars,
            },
        )
    preserve_foreign_quote = _legitimate_foreign_quote(unit, config)
    for item in local_issues:
        if preserve_foreign_quote and str(item.code) in {
            "untranslated_source",
            "target_language_missing",
            "source_language_dominant",
        }:
            continue
        severity = "critical" if str(item.severity).lower() == "reject" else "medium"
        issues.append(
            ValidationIssue(
                code=str(item.code),
                severity=severity,
                message=str(item.message),
                unit_id=unit.unit_id,
                gate="language",
                source_span=_snippet(unit.source_text),
                target_span=_snippet(target),
                suggested_fix="Repair this unit only, then rerun the language gate.",
            )
        )
    if (
        detected
        and confidence >= 0.85
        and _same_language(detected, unit.source_language)
        and not _same_language(detected, unit.target_language)
        and not preserve_foreign_quote
        and not any(issue.code == "target_language_missing" for issue in issues)
    ):
        issues.append(
            ValidationIssue(
                "source_language_unit",
                "critical",
                "The unit is still dominated by the source language.",
                unit_id=unit.unit_id,
                gate="language",
                confidence=float(confidence),
                target_span=_snippet(target),
            )
        )
    return ValidationResult(
        validator="target_language",
        version=VALIDATOR_VERSION,
        passed=not any(issue.blocking for issue in issues),
        issues=tuple(_dedupe_issues(issues)),
        metrics={
            "detected_language": detected or "",
            "confidence": float(confidence or 0.0),
        },
    )


def validate_unit_entities(
    unit: TranslationUnit,
    config: QualityAssuranceConfig,
    *,
    protected_entities: Iterable[Mapping[str, Any]] = (),
) -> tuple[ValidationResult, dict[str, Any]]:
    if not unit.translatable:
        result = ValidationResult(
            validator="entity_preservation",
            version=VALIDATOR_VERSION,
            passed=True,
            metrics={"skipped": True},
        )
        return result, {"unit_id": unit.unit_id, "issues": []}
    source = unit.source_text
    target = unit.best_text
    issues: list[ValidationIssue] = []
    details: dict[str, Any] = {"unit_id": unit.unit_id, "order_index": unit.order_index}

    checks = (
        ("number", _extract_digit_groups, config.entity_validation.preserve_numbers, "critical"),
        ("date", _extract_dates, config.entity_validation.preserve_dates, "critical"),
        ("percentage", _extract_percentages, config.entity_validation.preserve_percentages, "critical"),
        ("currency", _extract_currencies, config.entity_validation.preserve_currencies, "critical"),
        ("measurement", _extract_measurements, config.entity_validation.preserve_measurements, "critical"),
        ("identifier", _extract_identifiers, config.entity_validation.preserve_identifiers, "critical"),
        ("url", _extract_urls, config.entity_validation.preserve_urls, "critical"),
        ("isbn", _extract_isbns, config.entity_validation.preserve_isbn, "critical"),
        (
            "scientific_name",
            _extract_scientific_names,
            config.entity_validation.preserve_scientific_names,
            "critical",
        ),
    )
    for label, extractor, enabled, severity in checks:
        if not enabled:
            continue
        source_values = Counter(extractor(source))
        intentionally_excluded = Counter()
        for excluded_text in unit.source_reference.get(
            "intentional_exclusions", ()
        ):
            intentionally_excluded.update(extractor(str(excluded_text or "")))
        if intentionally_excluded:
            source_values -= intentionally_excluded
        target_values = Counter(extractor(target))
        missing = list((source_values - target_values).elements())
        added = list((target_values - source_values).elements())
        details[label] = {
            "source": dict(source_values),
            "target": dict(target_values),
            "missing": missing,
            "added": added,
            "intentionally_excluded": dict(intentionally_excluded),
        }
        if missing or added:
            effective_severity = _entity_mismatch_severity(
                unit,
                label=label,
                default=severity,
                source_values=source_values,
                target_values=target_values,
                missing=missing,
                added=added,
            )
            issues.append(
                ValidationIssue(
                    f"{label}_mismatch",
                    effective_severity,
                    f"The translation changed protected {label} values.",
                    unit_id=unit.unit_id,
                    gate="entities",
                    source_span=_snippet(source),
                    target_span=_snippet(target),
                    suggested_fix=f"Restore the exact {label} values from the source.",
                    details={"missing": missing[:20], "added": added[:20]},
                )
            )

    if config.entity_validation.preserve_protected_names:
        for entity in protected_entities:
            source_value = str(entity.get("source") or "").strip()
            if not source_value or not _contains_phrase(source, source_value):
                continue
            allowed = [
                str(value).strip()
                for value in entity.get("allowed_targets") or []
                if str(value).strip()
            ]
            if entity.get("target"):
                allowed.append(str(entity.get("target")).strip())
            if not allowed:
                allowed = [source_value]
            if not any(_contains_phrase(target, value) for value in allowed):
                severity = "critical" if entity.get("locked", True) else "high"
                issues.append(
                    ValidationIssue(
                        "protected_entity_mismatch",
                        severity,
                        f"Protected entity '{source_value}' is missing or changed.",
                        unit_id=unit.unit_id,
                        gate="entities",
                        source_span=source_value,
                        target_span=_snippet(target),
                        suggested_fix="Use one of the approved entity targets.",
                        details={"allowed_targets": allowed},
                    )
                )

    # Unlocked proper names are reported for human review, not blocked, because
    # transliteration and conventional localized forms are language-dependent.
    source_names = sorted(set(_PROPER_NAME_RE.findall(source)))
    missing_names = [name for name in source_names if not _contains_phrase(target, name)]
    details["proper_name_candidates"] = source_names
    details["missing_proper_name_candidates"] = missing_names
    return (
        ValidationResult(
            validator="entity_preservation",
            version=VALIDATOR_VERSION,
            passed=not any(issue.blocking for issue in issues),
            issues=tuple(issues),
            metrics={"protected_checks": sum(len(value) for value in details.values() if isinstance(value, dict))},
        ),
        {**details, "issues": [issue.to_dict() for issue in issues]},
    )


def validate_unit_semantics(
    unit: TranslationUnit,
    config: QualityAssuranceConfig,
) -> ValidationResult:
    if not unit.translatable:
        return ValidationResult(
            validator="deterministic_semantics",
            version=VALIDATOR_VERSION,
            passed=True,
            metrics={"skipped": True},
        )
    source = unit.source_text.strip()
    target = unit.best_text.strip()
    issues: list[ValidationIssue] = []
    if source and target:
        length_ratio = len(re.sub(r"\s+", "", target)) / max(
            1, len(re.sub(r"\s+", "", source))
        )
        if len(source) >= 100 and length_ratio < 0.55:
            issues.append(
                ValidationIssue(
                    "probable_semantic_omission",
                    "critical",
                    "The candidate is too short to preserve the complete source unit.",
                    unit_id=unit.unit_id,
                    gate="semantics",
                    source_span=_snippet(source),
                    target_span=_snippet(target),
                    details={"length_ratio": round(length_ratio, 4)},
                )
            )
        elif len(source) >= 100 and length_ratio > 2.2:
            issues.append(
                ValidationIssue(
                    "probable_semantic_addition",
                    "high",
                    "The candidate is abnormally long relative to the source unit.",
                    unit_id=unit.unit_id,
                    gate="semantics",
                    details={"length_ratio": round(length_ratio, 4)},
                )
            )
        source_negations = _negation_count(source)
        target_negations = _negation_count(target)
        if source_negations and not target_negations:
            issues.append(
                ValidationIssue(
                    "possible_negation_loss",
                    "critical",
                    "The source contains a negation but the candidate contains none.",
                    unit_id=unit.unit_id,
                    gate="semantics",
                    source_span=_snippet(source),
                    target_span=_snippet(target),
                )
            )
    for key in ("fidelity_decision", "refinement_fidelity_decision"):
        decision = unit.source_reference.get(key)
        if not isinstance(decision, Mapping):
            continue
        if decision.get("accepted") is False:
            issues.append(
                ValidationIssue(
                    "open_fidelity_rejection",
                    "critical",
                    "A prior source-aware fidelity audit rejected this unit.",
                    unit_id=unit.unit_id,
                    gate="semantics",
                    details={"phase": key, "decision": dict(decision)},
                )
            )
    return ValidationResult(
        validator="deterministic_semantics",
        version=VALIDATOR_VERSION,
        passed=not any(issue.blocking for issue in issues),
        issues=tuple(issues),
        metrics={
            "source_chars": len(source),
            "target_chars": len(target),
            "source_negations": _negation_count(source),
            "target_negations": _negation_count(target),
        },
    )


def validate_unit_typography(
    unit: TranslationUnit,
    config: QualityAssuranceConfig,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    target = unit.best_text
    if unit.translatable and target:
        for code, pattern in _SPACING_PATTERNS:
            matches = [match.group(0) for match in pattern.finditer(target)]
            if matches:
                # CamelCase names and coined compounds are indistinguishable
                # from a missing space without lexical context. Punctuation
                # joins remain strict; word joins are review-only.
                severity = (
                    "medium"
                    if code == "word_join"
                    else "critical"
                    if config.validation.block_on_spacing_corruption
                    else "medium"
                )
                issues.append(
                    ValidationIssue(
                        f"spacing_{code}",
                        severity,
                        "Possible lost whitespace was detected after reconstruction.",
                        unit_id=unit.unit_id,
                        gate="typography",
                        target_span=_snippet(target),
                        details={"count": len(matches), "examples": matches[:10]},
                    )
                )
    return ValidationResult(
        validator="typography_spacing",
        version=VALIDATOR_VERSION,
        passed=not any(issue.blocking for issue in issues),
        issues=tuple(issues),
        metrics={"findings": len(issues)},
    )


def validate_output_artifact(
    manifest: BookManifest,
    config: QualityAssuranceConfig,
    *,
    publication_report: Any = None,
) -> tuple[ValidationResult, dict[str, Any]]:
    output = Path(manifest.output_path)
    source = Path(manifest.source_path)
    issues: list[ValidationIssue] = []
    details: dict[str, Any] = {
        "source_format": manifest.source_format,
        "output_format": manifest.output_format,
        "output_exists": output.exists(),
    }
    if not output.exists() or not output.is_file() or output.stat().st_size == 0:
        issues.append(
            ValidationIssue(
                "missing_output_artifact",
                "critical",
                "The final output file does not exist or is empty.",
                gate="final",
            )
        )
    else:
        try:
            readable = extract_readable_text(output)
        except Exception as exc:
            readable = ""
            issues.append(
                ValidationIssue(
                    "output_read_error",
                    "critical",
                    f"The output could not be read: {exc}",
                    gate="final",
                )
            )
        details["readable_characters"] = len(readable)
        details["mojibake_score"] = mojibake_score(readable)
        if not readable.strip():
            issues.append(
                ValidationIssue(
                    "empty_readable_output",
                    "critical",
                    "The final file contains no readable text.",
                    gate="final",
                )
            )
        if details["mojibake_score"]:
            issues.append(
                ValidationIssue(
                    "mojibake_detected",
                    "critical",
                    "The final file contains encoding corruption.",
                    gate="final",
                    details={"score": details["mojibake_score"]},
                )
            )

    if publication_report is not None:
        details["epub_publication_gate"] = publication_report.to_dict()
        for message in publication_report.errors:
            issues.append(
                ValidationIssue(
                    "epub_publication_error",
                    "critical",
                    str(message),
                    gate="structure",
                )
            )
        if getattr(publication_report, "epubcheck_warnings", 0):
            issues.append(
                ValidationIssue(
                    "epubcheck_warning",
                    "medium",
                    f"EPUBCheck reported {publication_report.epubcheck_warnings} warning(s).",
                    gate="final",
                )
            )
    elif output.exists():
        suffix = output.suffix.lower()
        if suffix == ".epub":
            issues.extend(_validate_epub_structure(output, details, config))
        elif suffix == ".docx":
            issues.extend(_validate_docx_structure(source, output, details))
        elif suffix == ".pdf":
            issues.extend(_validate_pdf_structure(output, details))
        elif suffix == ".srt":
            issues.extend(_validate_srt_structure(source, output, details))

    target_locale = manifest.target_locale or manifest.target_language
    if target_locale and not _valid_language_tag(target_locale):
        issues.append(
            ValidationIssue(
                "invalid_target_language_code",
                "critical",
                f"The target locale '{target_locale}' is not a valid BCP 47 language tag or known language label.",
                gate="metadata",
            )
        )
    return (
        ValidationResult(
            validator="output_artifact",
            version=VALIDATOR_VERSION,
            passed=not any(issue.blocking for issue in issues),
            issues=tuple(issues),
            metrics=details,
        ),
        {**details, "issues": [issue.to_dict() for issue in issues]},
    )


def _validate_epub_structure(
    output: Path,
    details: dict[str, Any],
    config: QualityAssuranceConfig,
) -> list[ValidationIssue]:
    from src.core.epub.publication_gate import snapshot_epub

    issues: list[ValidationIssue] = []
    try:
        snapshot = snapshot_epub(output, recover=False)
        details.update(
            {
                "spine_items": len(snapshot.spine),
                "manifest_items": snapshot.manifest_count,
                "images": len(snapshot.image_files),
                "broken_parse_files": list(snapshot.parse_errors),
                "unsafe_paths": list(snapshot.unsafe_paths),
                "temporary_files": list(snapshot.temporary_files),
                "mimetype_first_stored": snapshot.mimetype_first_stored,
                "mimetype_exact": snapshot.mimetype_exact,
                "languages": list(snapshot.languages),
            }
        )
        if snapshot.parse_errors or not snapshot.spine or snapshot.unsafe_paths:
            issues.append(
                ValidationIssue(
                    "invalid_epub_structure",
                    "critical",
                    "The EPUB package contains parse errors, unsafe paths, or no spine.",
                    gate="structure",
                    details={
                        "parse_errors": snapshot.parse_errors,
                        "unsafe_paths": snapshot.unsafe_paths,
                    },
                )
            )
        if not snapshot.mimetype_first_stored or not snapshot.mimetype_exact:
            issues.append(
                ValidationIssue(
                    "invalid_epub_mimetype",
                    "critical",
                    "The EPUB mimetype entry is not first, exact, and uncompressed.",
                    gate="final",
                )
            )
        issues.extend(_run_epubcheck_validation(output, details, config))
    except Exception as exc:
        issues.append(
            ValidationIssue(
                "invalid_epub",
                "critical",
                f"The EPUB package cannot be inspected: {exc}",
                gate="final",
            )
        )
    return issues


def _run_epubcheck_validation(
    output: Path,
    details: dict[str, Any],
    config: QualityAssuranceConfig,
) -> list[ValidationIssue]:
    executable = shutil.which("epubcheck")
    if not executable:
        details["epubcheck"] = {"available": False, "errors": 0, "warnings": 0}
        return [
            ValidationIssue(
                "epubcheck_unavailable",
                "medium",
                "EPUBCheck is not installed; internal package validation still ran.",
                gate="final",
            )
        ]
    try:
        process = subprocess.run(
            [executable, str(output)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            ValidationIssue(
                "epubcheck_execution_error",
                "critical" if config.validation.block_on_epubcheck_error else "medium",
                f"EPUBCheck could not complete: {exc}",
                gate="final",
            )
        ]
    combined = "\n".join(
        value for value in (process.stdout, process.stderr) if value
    ).strip()
    errors = len(re.findall(r"(?im)^.*\bERROR\b", combined))
    warnings = len(re.findall(r"(?im)^.*\bWARNING\b", combined))
    if process.returncode and not errors:
        errors = 1
    details["epubcheck"] = {
        "available": True,
        "return_code": process.returncode,
        "errors": errors,
        "warnings": warnings,
        "output": combined[-12000:],
    }
    issues: list[ValidationIssue] = []
    if errors:
        issues.append(
            ValidationIssue(
                "epubcheck_error",
                "critical" if config.validation.block_on_epubcheck_error else "medium",
                f"EPUBCheck reported {errors} error(s).",
                gate="final",
                details={"errors": errors},
            )
        )
    if warnings:
        issues.append(
            ValidationIssue(
                "epubcheck_warning",
                "medium",
                f"EPUBCheck reported {warnings} warning(s).",
                gate="final",
                details={"warnings": warnings},
            )
        )
    return issues


def _validate_docx_structure(source: Path, output: Path, details: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        output_counts = _docx_counts(output)
        details["docx_output"] = output_counts
        if source.suffix.lower() == ".docx" and source.exists():
            source_counts = _docx_counts(source)
            details["docx_source"] = source_counts
            for key in ("tables", "images"):
                if output_counts[key] < source_counts[key]:
                    issues.append(
                        ValidationIssue(
                            f"docx_{key}_lost",
                            "critical",
                            f"The DOCX output lost {key} from the source package.",
                            gate="structure",
                            details={"source": source_counts[key], "output": output_counts[key]},
                        )
                    )
    except Exception as exc:
        issues.append(
            ValidationIssue(
                "invalid_docx",
                "critical",
                f"The DOCX package is invalid: {exc}",
                gate="final",
            )
        )
    return issues


def _validate_pdf_structure(output: Path, details: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(output), strict=True)
        details["pdf_pages"] = len(reader.pages)
        if not reader.pages:
            raise ValueError("PDF has no pages")
    except Exception as exc:
        issues.append(
            ValidationIssue(
                "invalid_pdf",
                "critical",
                f"The final PDF cannot be parsed: {exc}",
                gate="final",
            )
        )
    return issues


def _validate_srt_structure(source: Path, output: Path, details: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    output_times = re.findall(r"(?m)^\s*\d\d:\d\d:\d\d[,.]\d{3}\s+-->\s+.+$", output.read_text(encoding="utf-8", errors="replace"))
    source_times: list[str] = []
    if source.suffix.lower() == ".srt" and source.exists():
        source_times = re.findall(r"(?m)^\s*\d\d:\d\d:\d\d[,.]\d{3}\s+-->\s+.+$", source.read_text(encoding="utf-8", errors="replace"))
    details["srt_source_entries"] = len(source_times)
    details["srt_output_entries"] = len(output_times)
    if source_times and len(output_times) != len(source_times):
        issues.append(
            ValidationIssue(
                "srt_entry_count_mismatch",
                "critical",
                "The SRT output does not contain every source subtitle entry.",
                gate="structure",
            )
        )
    return issues


def _docx_counts(path: Path) -> dict[str, int]:
    if not zipfile.is_zipfile(path):
        raise ValueError("not a ZIP package")
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if "word/document.xml" not in names:
            raise ValueError("word/document.xml is missing")
        root = etree.fromstring(archive.read("word/document.xml"))
        return {
            "paragraphs": len(root.xpath("//*[local-name()='p']")),
            "tables": len(root.xpath("//*[local-name()='tbl']")),
            "images": sum(1 for name in names if name.startswith("word/media/") and not name.endswith("/")),
        }


def _build_entity_report(details: list[dict[str, Any]]) -> dict[str, Any]:
    issues = [issue for item in details for issue in item.get("issues") or []]
    counts = Counter(issue.get("code") for issue in issues)
    return {
        "units_checked": len(details),
        "issue_count": len(issues),
        "issue_counts": dict(counts),
        "issues": issues,
        "units": details,
    }


def _duplicate_output_issues(manifest: BookManifest) -> list[ValidationIssue]:
    seen: dict[str, TranslationUnit] = {}
    issues: list[ValidationIssue] = []
    for unit in manifest.translatable_units:
        normalized = _normalized(unit.best_text)
        if len(normalized) < 100:
            continue
        previous = seen.get(normalized)
        if previous is None:
            seen[normalized] = unit
            continue
        if previous.checksum_source == unit.checksum_source:
            continue
        issues.append(
            ValidationIssue(
                "duplicate_unrelated_output",
                "critical",
                "Two different source units produced the same substantial output.",
                unit_id=unit.unit_id,
                gate="translation",
                source_span=_snippet(unit.source_text),
                target_span=_snippet(unit.best_text),
                suggested_fix="Retranslate this unit with its local context and preserve the other approved unit.",
                details={"duplicates_unit_id": previous.unit_id},
            )
        )
    return issues


def _artifact_alignment_issues(manifest: BookManifest) -> list[ValidationIssue]:
    if manifest.metadata.get("manifest_source") != "artifact_alignment":
        return []
    extraction_error = str(manifest.metadata.get("output_extraction_error") or "")
    issues: list[ValidationIssue] = []
    if extraction_error:
        issues.append(
            ValidationIssue(
                "output_block_extraction_failed",
                "critical",
                "The final artifact could not be structurally extracted for source/output alignment.",
                gate="structure",
                details={"error": extraction_error},
            )
        )
    source_blocks = int(manifest.metadata.get("source_blocks") or 0)
    output_blocks = int(manifest.metadata.get("output_blocks") or 0)
    if source_blocks == output_blocks:
        return issues
    issues.append(
        ValidationIssue(
            "artifact_block_count_mismatch",
            "critical",
            "Source and output artifacts do not contain the same number of structural text blocks.",
            gate="structure",
            details={"source_blocks": source_blocks, "output_blocks": output_blocks},
        )
    )
    return issues


def _section_key(structural_path: str) -> str:
    path = str(structural_path or "/")
    parts = [part for part in path.split("/") if part]
    return "/" + "/".join(parts[:2]) if parts else "/"


def _extract_digit_groups(text: str) -> list[str]:
    value = text or ""
    matches = list(_DIGIT_GROUP_RE.finditer(value))
    result = [re.sub(r"[ \u00a0\u202f]", "", match.group(0)) for match in matches]
    covered = [match.span() for match in matches]
    for measurement in _iter_measurements(value):
        number_span = measurement.span(1)
        if any(start <= number_span[0] and number_span[1] <= end for start, end in covered):
            continue
        result.append(re.sub(r"[ \u00a0\u202f]", "", measurement.group(1)))
    return result


def _extract_dates(text: str) -> list[str]:
    return [_normalized_entity(value) for value in _DATE_RE.findall(text or "")]


def _extract_percentages(text: str) -> list[str]:
    return [_normalized_entity(value) for value in _PERCENT_RE.findall(text or "")]


def _extract_currencies(text: str) -> list[str]:
    return [_normalized_entity(value) for value in _CURRENCY_RE.findall(text or "")]


def _extract_measurements(text: str) -> list[str]:
    values: list[str] = []
    for match in _iter_measurements(text or ""):
        number, long_unit, short_unit = match.groups()
        unit = long_unit or short_unit
        normalized_number = re.sub(r"[ \u00a0\u202f]", "", number).replace(",", ".")
        normalized_unit = _MEASUREMENT_UNIT_CANONICAL.get(
            unit.casefold(),
            _CASE_SENSITIVE_SHORT_MEASUREMENT_UNITS.get(unit, unit.upper()),
        )
        values.append(f"{normalized_number}{normalized_unit}")
    return values


def _iter_measurements(text: str) -> Iterable[re.Match[str]]:
    for match in _MEASUREMENT_RE.finditer(text or ""):
        short_unit = match.group(3)
        if short_unit:
            # SI symbols are case-sensitive. Without this guard, ordinary
            # ranges such as "2 a 3" are misread as two amperes.
            if short_unit not in _CASE_SENSITIVE_SHORT_MEASUREMENT_UNITS:
                continue
            # In flattened indexes, an entry such as "2 A Civilizing Mission"
            # otherwise becomes the fictitious measurement "2 A". A genuine
            # amperage may be followed by punctuation, a lowercase descriptor,
            # or an all-caps qualifier such as DC, but not an English
            # title-cased article complement.
            if short_unit == "A" and _TITLECASE_WORD_AFTER_MEASUREMENT_RE.match(
                text,
                match.end(),
            ):
                continue
            # Flattened OCR/DOM boundaries can turn a bibliographic heading
            # such as ``Bacon, 1597 Los estudios`` into ``Bacon, 1597 L os
            # estudios``.  The isolated initial happens to be an SI symbol,
            # but the comma-delimited four-digit publication year and the
            # following lowercase fragment prove that this is prose, not a
            # measurement.  Keep real values such as ``2000 L of water``:
            # they are not introduced as a comma-delimited publication year.
            number = str(match.group(1) or "")
            before = text[:match.start()]
            if (
                len(number) == 4
                and number.isdigit()
                and 1000 <= int(number) <= 2199
                and re.search(r",\s*$", before)
                and _LOWERCASE_FRAGMENT_AFTER_MEASUREMENT_RE.match(
                    text,
                    match.end(),
                )
            ):
                continue
        yield match


def _extract_identifiers(text: str) -> list[str]:
    return [_normalized_entity(value) for value in _IDENTIFIER_RE.findall(text or "")]


def _normalized_entity(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).upper()


def _extract_urls(text: str) -> list[str]:
    return [value.rstrip(".,;:!?)\"]}") for value in _URL_RE.findall(text or "")]


def _extract_isbns(text: str) -> list[str]:
    return [re.sub(r"[^0-9X]", "", value.upper()) for value in _ISBN_RE.findall(text or "")]


def _extract_scientific_names(text: str) -> list[str]:
    result: list[str] = []
    value = text or ""
    for match in _SCIENTIFIC_NAME_RE.finditer(value):
        genus, species = match.groups()
        if (
            genus in _NON_SCIENTIFIC_GENUS
            or species.casefold() in _NON_SCIENTIFIC_SPECIES
            or _PROSE_VERB_SUFFIX_RE.search(species)
        ):
            continue
        sentence_start = max(
            value.rfind(".", 0, match.start()),
            value.rfind("!", 0, match.start()),
            value.rfind("?", 0, match.start()),
            value.rfind("\n", 0, match.start()),
        ) + 1
        prefix = value[max(sentence_start, match.start() - 100) : match.start()]
        # Capitalized word + lowercase word is ordinary prose far more often
        # than a taxonomic binomial ("Sheila stood", "Este cuerpo"). Require
        # scientific context before the candidate. A trailing phrase such as
        # "es una especie de..." is commonly figurative and cannot establish a
        # taxonomic name deterministically. Profiles can lock context-free
        # scientific names explicitly.
        if not _SCIENTIFIC_PREFIX_RE.search(prefix):
            continue
        result.append(f"{genus} {species}")
    return result


def _output_guard_issue_severity(item: Any) -> str:
    code = str(getattr(item, "code", ""))
    severity = str(getattr(item, "severity", "warning")).lower()
    if severity in {"reject", "critical", "high", "error"}:
        return "critical"
    if code in {"llm_protocol_leak_cleaned", "reader_artifact_label_cleaned"}:
        # At final QA the guard is observing the published candidate rather
        # than mutating it, so a reader-visible wrapper still requires repair.
        return "high"
    return "medium"


def _entity_mismatch_severity(
    unit: TranslationUnit,
    *,
    label: str,
    default: str,
    source_values: Counter[str],
    target_values: Counter[str],
    missing: list[str],
    added: list[str],
) -> str:
    """Do not re-block an EPUB chapter already audited at block granularity."""

    if not unit.source_reference.get("publication_audited"):
        return default
    if len(unit.source_text) < 4_000:
        return default
    if label not in {"number", "date", "percentage", "currency", "measurement"}:
        return default
    # ``publication_audited`` means the EPUB gate has already compared the
    # chapter's individual aligned blocks. The universal manifest intentionally
    # aggregates those blocks by XHTML file; reinterpreting numeric formatting
    # in that huge aggregate makes localized digits (40000/40 000), Roman
    # ordinals, and translated units look like critical entity loss. Keep those
    # numeric diffs visible, but retain blocking severity for names, identifiers,
    # URLs, ISBNs, and scientific terms.
    return "medium"


def _negation_count(text: str) -> int:
    tokens = re.findall(r"[\w\u0370-\u03ff\u0400-\u052f\u3040-\u30ff\u3400-\u9fff]+", (text or "").casefold())
    return sum(1 for token in tokens if token in _NEGATION_MARKERS)


def _enough_language_evidence(text: str, configured_minimum: int) -> bool:
    """Use a lower evidence threshold for dense scripts without word spacing."""

    compact = re.sub(r"\s+", "", text or "")
    if _DENSE_SCRIPT_RE.search(compact):
        return len(compact) >= min(configured_minimum, 12)
    return len((text or "").strip()) >= configured_minimum


def _legitimate_foreign_quote(unit: TranslationUnit, config: QualityAssuranceConfig) -> bool:
    if not config.language_validation.allow_quoted_foreign_text:
        return False
    source = unit.source_text.strip()
    target = unit.best_text.strip()
    if source != target or len(source) > 800:
        return False
    return bool(_QUOTE_RE.fullmatch(source)) or unit.content_type in {
        "bibliography",
        "critical_apparatus",
    }


def _looks_like_protected_name(text: str) -> bool:
    words = re.findall(r"[\w'’-]+", text or "")
    return bool(words) and len(words) <= 8 and all(
        word[:1].isupper() or word.casefold() in {"de", "del", "da", "dos", "van", "von", "la", "le"}
        for word in words
    )


def _contains_phrase(text: str, phrase: str) -> bool:
    return _normalized(phrase) in _normalized(text)


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().casefold()


def _snippet(value: str, limit: int = 320) -> str:
    compact = re.sub(r"\s+", " ", value or "").strip()
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "..."


def _same_language(left: str | None, right: str | None) -> bool:
    return _language_code(left) == _language_code(right) and bool(_language_code(left))


def _language_code(value: str | None) -> str:
    normalized = str(value or "").strip().casefold().replace("_", "-")
    if not normalized:
        return ""
    first = normalized.split("-", 1)[0]
    reverse = {name.casefold(): code.split("-", 1)[0] for code, name in LANGUAGE_CODE_MAP.items()}
    aliases = {
        "espanol": "es",
        "español": "es",
        "deutsch": "de",
        "francais": "fr",
        "français": "fr",
        "auto": "auto",
    }
    return aliases.get(normalized) or reverse.get(normalized) or first


def _valid_language_tag(value: str) -> bool:
    normalized = str(value or "").strip()
    if not normalized:
        return False
    if _language_code(normalized) in set(code.split("-", 1)[0] for code in LANGUAGE_CODE_MAP):
        return True
    return bool(_BCP47_RE.fullmatch(normalized)) and normalized.casefold() != "sp"


def _dedupe_issues(issues: Iterable[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str]] = set()
    result: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.unit_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result
