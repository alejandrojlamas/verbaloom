"""Serializable quality contract for EPUB translation units."""

from __future__ import annotations

from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional


EPUB_PIPELINE_VERSION = "epub-strict-units-v2"
EPUB_TRANSLATION_PROMPT_VERSION = "epub-translation-2026-07-11"
EPUB_REVIEW_PROMPT_VERSION = "epub-review-2026-07-12"
EPUB_AUDIT_PROMPT_VERSION = "epub-audit-2026-07-12-v2"
EPUB_PLACEHOLDER_VERSION = "epub-placeholders-v2"
EPUB_SEGMENTATION_VERSION = "epub-sentence-clause-v2"
# Compatibility alias used by existing reports and checkpoints.  New code
# should persist the independent versions returned by ``prompt_versions``.
EPUB_PROMPT_VERSION = EPUB_TRANSLATION_PROMPT_VERSION


class UnitStageStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TranslationUnitStatus(str, Enum):
    PENDING = "PENDING"
    TRANSLATED = "TRANSLATED"
    REVIEWED = "REVIEWED"
    AUDITED = "AUDITED"
    FAILED = "FAILED"


def text_sha256(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _safe_prompt_options(options: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in sorted((options or {}).items()):
        lowered = str(key).casefold()
        if key.startswith("_") or any(token in lowered for token in ("key", "token", "secret", "credential")):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = value
        elif isinstance(value, (list, tuple)):
            safe[str(key)] = [item for item in value if isinstance(item, (str, int, float, bool)) or item is None]
        elif isinstance(value, Mapping):
            safe[str(key)] = {
                str(nested_key): nested_value
                for nested_key, nested_value in sorted(value.items())
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
    return safe


def prompt_versions() -> dict[str, str]:
    return {
        "translation": EPUB_TRANSLATION_PROMPT_VERSION,
        "review": EPUB_REVIEW_PROMPT_VERSION,
        "audit": EPUB_AUDIT_PROMPT_VERSION,
        "placeholder": EPUB_PLACEHOLDER_VERSION,
        "segmentation": EPUB_SEGMENTATION_VERSION,
    }


def _stage_option_groups(options: Optional[Mapping[str, Any]]) -> tuple[dict, dict, dict]:
    """Split safe prompt options so a later-stage change has narrow invalidation."""
    safe = _safe_prompt_options(options)
    review_tokens = ("review", "reviewer", "refine", "editorial", "style", "voice", "repair", "postprocess")
    audit_tokens = ("audit", "fidelity", "supervisor", "judge", "guard")
    translation: dict[str, Any] = {}
    review: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for key, value in safe.items():
        lowered = key.casefold()
        if any(token in lowered for token in audit_tokens):
            audit[key] = value
        elif any(token in lowered for token in review_tokens):
            review[key] = value
        else:
            translation[key] = value
    return translation, review, audit


def stage_fingerprints(
    *,
    source_language: str,
    target_language: str,
    model_name: str,
    max_tokens_per_chunk: int,
    max_retries: int,
    prompt_options: Optional[Mapping[str, Any]],
    versions: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Build independently invalidatable translation/review/audit identities."""
    active_versions = {**prompt_versions(), **dict(versions or {})}
    translation_options, review_options, audit_options = _stage_option_groups(prompt_options)
    common = {
        "pipeline_version": EPUB_PIPELINE_VERSION,
        "source_language": source_language,
        "target_language": target_language,
        "model_name": model_name,
    }
    translation_payload = {
        **common,
        "prompt_version": active_versions["translation"],
        "placeholder_version": active_versions["placeholder"],
        "segmentation_version": active_versions["segmentation"],
        "max_tokens_per_chunk": int(max_tokens_per_chunk),
        "max_retries": int(max_retries),
        "prompt_options": translation_options,
    }
    translation_fp = text_sha256(json.dumps(
        translation_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))
    review_payload = {
        **common,
        "translation_fingerprint": translation_fp,
        "prompt_version": active_versions["review"],
        "prompt_options": review_options,
    }
    review_fp = text_sha256(json.dumps(
        review_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))
    audit_payload = {
        **common,
        "review_fingerprint": review_fp,
        "prompt_version": active_versions["audit"],
        "prompt_options": audit_options,
    }
    audit_fp = text_sha256(json.dumps(
        audit_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))
    return {"translation": translation_fp, "review": review_fp, "audit": audit_fp}


def invalidated_stages(
    previous: Mapping[str, str],
    current: Mapping[str, str],
) -> tuple[str, ...]:
    """Return the minimal suffix of stages invalidated by a config change."""
    if previous.get("translation") != current.get("translation"):
        return ("translation", "review", "audit")
    if previous.get("review") != current.get("review"):
        return ("review", "audit")
    if previous.get("audit") != current.get("audit"):
        return ("audit",)
    return ()


def config_fingerprint(
    *,
    source_language: str,
    target_language: str,
    model_name: str,
    max_tokens_per_chunk: int,
    max_retries: int,
    prompt_options: Optional[Mapping[str, Any]],
) -> str:
    return stage_fingerprints(
        source_language=source_language,
        target_language=target_language,
        model_name=model_name,
        max_tokens_per_chunk=max_tokens_per_chunk,
        max_retries=max_retries,
        prompt_options=prompt_options,
    )["translation"]


def stable_unit_id(
    file_href: str,
    ordinal: int,
    source_hash: str,
    *,
    spine_index: Optional[int] = None,
    dom_path: str = "",
    source_order: Optional[int] = None,
) -> str:
    identity = (
        f"{file_href}\0{int(ordinal)}\0{source_hash}"
        if spine_index is None and not dom_path and source_order is None
        else f"{spine_index}\0{file_href}\0{dom_path}\0{source_order}\0{source_hash}"
    )
    digest = text_sha256(identity)[:24]
    return f"epub:{file_href}:{int(ordinal):06d}:{digest}"


def ensure_unit_records(
    chunks: list[dict[str, Any]],
    file_href: str,
    *,
    source_language: str = "",
    target_language: str = "",
    spine_index: Optional[int] = None,
    fingerprints: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, chunk in enumerate(chunks):
        source_text = str(chunk.get("text") or "")
        source_hash = text_sha256(source_text)
        record = chunk.get("unit")
        expected_id = stable_unit_id(file_href, ordinal, source_hash)
        if not isinstance(record, dict) or record.get("source_hash") != source_hash:
            record = {
                "unit_id": expected_id,
                "file_href": file_href,
                "source_document": file_href,
                "spine_index": spine_index,
                "dom_path": f"chunk:{ordinal}",
                "source_order": ordinal,
                "ordinal": ordinal,
                "source_text": source_text,
                "source_hash": source_hash,
                "source_language": source_language,
                "target_language": target_language,
                "status": TranslationUnitStatus.PENDING.value,
                "translation_status": UnitStageStatus.PENDING.value,
                "review_status": UnitStageStatus.PENDING.value,
                "audit_status": UnitStageStatus.PENDING.value,
                "translation": None,
                "translation_hash": None,
                "review": None,
                "audit": None,
                "attempts": 0,
                "translation_attempts": 0,
                "review_attempts": 0,
                "audit_attempts": 0,
                "failure_reason": None,
                "prompt_versions": prompt_versions(),
                "stage_fingerprints": dict(fingerprints or {}),
                "pipeline_version": EPUB_PIPELINE_VERSION,
                "placeholder_manifest": sorted(str(key) for key in (chunk.get("local_tag_map") or {})),
            }
            chunk["unit"] = record
        else:
            record["unit_id"] = expected_id
            record["file_href"] = file_href
            record["source_document"] = file_href
            record["ordinal"] = ordinal
            record["source_order"] = ordinal
            record.setdefault("dom_path", f"chunk:{ordinal}")
            record.setdefault("spine_index", spine_index)
            record["source_text"] = source_text
            record.setdefault("source_language", source_language)
            record.setdefault("target_language", target_language)
            record.setdefault("translation_status", _legacy_translation_status(record))
            record.setdefault("review_status", _legacy_review_status(record))
            record.setdefault("audit_status", _legacy_audit_status(record))
            record.setdefault("translation_attempts", int(record.get("attempts") or 0))
            record.setdefault("review_attempts", 0)
            record.setdefault("audit_attempts", 0)
            record.setdefault("prompt_versions", prompt_versions())
            record["stage_fingerprints"] = dict(fingerprints or record.get("stage_fingerprints") or {})
            record.setdefault("pipeline_version", EPUB_PIPELINE_VERSION)
            record.setdefault(
                "placeholder_manifest",
                sorted(str(key) for key in (chunk.get("local_tag_map") or {})),
            )
        records.append(record)
    return records


def _legacy_translation_status(record: Mapping[str, Any]) -> str:
    return (
        UnitStageStatus.COMPLETED.value
        if record.get("translation") and record.get("status") in {
            TranslationUnitStatus.TRANSLATED.value,
            TranslationUnitStatus.REVIEWED.value,
            TranslationUnitStatus.AUDITED.value,
        }
        else UnitStageStatus.PENDING.value
    )


def _legacy_review_status(record: Mapping[str, Any]) -> str:
    return (
        UnitStageStatus.COMPLETED.value
        if record.get("status") in {TranslationUnitStatus.REVIEWED.value, TranslationUnitStatus.AUDITED.value}
        else UnitStageStatus.PENDING.value
    )


def _legacy_audit_status(record: Mapping[str, Any]) -> str:
    return (
        UnitStageStatus.COMPLETED.value
        if record.get("status") == TranslationUnitStatus.AUDITED.value
        else UnitStageStatus.PENDING.value
    )


def mark_attempt(record: MutableMapping[str, Any], stage: str = "translation") -> None:
    record["attempts"] = int(record.get("attempts") or 0) + 1
    attempt_key = f"{stage}_attempts"
    record[attempt_key] = int(record.get(attempt_key) or 0) + 1
    status_key = f"{stage}_status"
    if status_key in record:
        record[status_key] = UnitStageStatus.IN_PROGRESS.value
    if record.get("status") == TranslationUnitStatus.FAILED.value:
        record["status"] = TranslationUnitStatus.PENDING.value
        record["failure_reason"] = None


def mark_translated(record: MutableMapping[str, Any], text: str) -> None:
    record["translation"] = text
    record["translation_hash"] = text_sha256(text)
    record["status"] = TranslationUnitStatus.TRANSLATED.value
    record["translation_status"] = UnitStageStatus.COMPLETED.value
    record["review_status"] = UnitStageStatus.PENDING.value
    record["audit_status"] = UnitStageStatus.PENDING.value
    record["review"] = None
    record["audit"] = None
    record["failure_reason"] = None


def mark_reviewed(
    record: MutableMapping[str, Any],
    text: str,
    *,
    review: Optional[Mapping[str, Any]] = None,
) -> None:
    mark_translated(record, text)
    record["review"] = dict(review or {"status": "pass", "scope": "chunk"})
    record["review_status"] = UnitStageStatus.COMPLETED.value
    record["audit"] = None
    record["status"] = TranslationUnitStatus.REVIEWED.value


def invalidate_unit_stages(record: MutableMapping[str, Any], stages: Iterable[str]) -> None:
    """Clear only the cache suffix affected by a stage configuration change."""
    invalid = set(stages)
    if "translation" in invalid:
        record["translation"] = None
        record["translation_hash"] = None
        record["translation_status"] = UnitStageStatus.PENDING.value
        record["status"] = TranslationUnitStatus.PENDING.value
    if "review" in invalid:
        record["review"] = None
        record["review_status"] = UnitStageStatus.PENDING.value
        if record.get("translation_status") == UnitStageStatus.COMPLETED.value:
            record["status"] = TranslationUnitStatus.TRANSLATED.value
    if "audit" in invalid:
        record["audit"] = None
        record["audit_status"] = UnitStageStatus.PENDING.value
        if record.get("review_status") == UnitStageStatus.COMPLETED.value:
            record["status"] = TranslationUnitStatus.REVIEWED.value
    record["failure_reason"] = None


def mark_audited(
    record: MutableMapping[str, Any],
    text: str,
    *,
    review: Optional[Mapping[str, Any]] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> None:
    mark_reviewed(record, text, review=review)
    record["audit"] = dict(audit or {"status": "pass", "scope": "chunk"})
    record["audit_status"] = UnitStageStatus.COMPLETED.value
    record["status"] = TranslationUnitStatus.AUDITED.value


def mark_failed(record: MutableMapping[str, Any], reason: str, *, stage: str = "translation") -> None:
    record["status"] = TranslationUnitStatus.FAILED.value
    status_key = f"{stage}_status"
    if status_key in record:
        record[status_key] = UnitStageStatus.FAILED.value
    for later in {
        "translation": ("review_status", "audit_status"),
        "review": ("audit_status",),
        "audit": (),
    }.get(stage, ()):
        record[later] = UnitStageStatus.PENDING.value
    record["failure_reason"] = reason or "translation_failed"
    if stage == "audit":
        record["audit"] = {"status": "fail", "reason": record["failure_reason"]}


def validate_resume_prefix(
    chunks: list[dict[str, Any]],
    translated_chunks: list[str],
    current_chunk_index: int,
) -> tuple[bool, str]:
    if current_chunk_index != len(translated_chunks):
        return False, "translated prefix length does not match current_chunk_index"
    if current_chunk_index > len(chunks):
        return False, "current_chunk_index exceeds unit count"
    for index in range(current_chunk_index):
        record = chunks[index].get("unit") or {}
        if record.get("translation_status") != UnitStageStatus.COMPLETED.value:
            return False, f"unit {index} translation is not COMPLETED"
        if record.get("source_hash") != text_sha256(str(chunks[index].get("text") or "")):
            return False, f"unit {index} source hash changed"
        if record.get("translation_hash") != text_sha256(translated_chunks[index]):
            return False, f"unit {index} translation hash changed"
    return True, ""


def validate_publishable_units(chunks: Iterable[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        record = chunk.get("unit") if isinstance(chunk, Mapping) else None
        if not isinstance(record, Mapping):
            errors.append(f"unit {index} has no quality record")
            continue
        unit_id = str(record.get("unit_id") or "")
        if not unit_id:
            errors.append(f"unit {index} has no stable ID")
        elif unit_id in seen:
            errors.append(f"duplicate unit ID: {unit_id}")
        seen.add(unit_id)
        for stage in ("translation", "review", "audit"):
            if record.get(f"{stage}_status") != UnitStageStatus.COMPLETED.value:
                errors.append(
                    f"{unit_id or index} {stage} status is {record.get(f'{stage}_status')}"
                )
        if record.get("status") != TranslationUnitStatus.AUDITED.value:
            errors.append(f"{unit_id or index} aggregate status is {record.get('status')}")
        if not record.get("translation"):
            errors.append(f"{unit_id or index} has no translation")
        if record.get("source_hash") != text_sha256(str(chunk.get("text") or "")):
            errors.append(f"{unit_id or index} source hash mismatch")
    return not errors, errors


def write_unit_manifest(path: str | Path, records_by_file: Mapping[str, Iterable[Mapping[str, Any]]]) -> Path:
    output = Path(path)
    units = [dict(record) for file_records in records_by_file.values() for record in file_records]
    payload = {
        "schema_version": 1,
        "pipeline_version": EPUB_PIPELINE_VERSION,
        "prompt_version": EPUB_PROMPT_VERSION,
        "prompt_versions": prompt_versions(),
        "total_units": len(units),
        "status_counts": {
            status.value: sum(1 for unit in units if unit.get("status") == status.value)
            for status in TranslationUnitStatus
        },
        "units": units,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
