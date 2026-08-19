"""Checkpointed, exact-ID semantic audit for packaged EPUB translations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.core.structured_units import (
    StructuredBatchStats,
    StructuredUnit,
    request_structured_units,
)
from src.core.llm.request_deadline import await_llm_call

from .publication_gate import EpubSnapshot
from .unit_contract import text_sha256


SEMANTIC_AUDIT_PROMPT_VERSION = "epub-semantic-audit-2026-07-12-v1"
SEMANTIC_ADJUDICATION_PROMPT_VERSION = "epub-semantic-adjudication-2026-07-12-v2"
_MATERIAL_FIELDS = (
    "missing_from_source",
    "added_not_in_source",
    "changed_facts",
    "censored_or_softened",
    "terminology_errors",
    "structure_issues",
)


@dataclass
class SemanticAuditReport:
    model: str
    source_units: int
    output_units: int
    results: list[dict[str, Any]] = field(default_factory=list)
    requests: int = 0
    validation_failures: int = 0
    recursive_splits: int = 0
    cache_hits: int = 0

    @property
    def passed(self) -> int:
        return sum(1 for item in self.results if item.get("accepted"))

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def complete(self) -> bool:
        return (
            self.source_units == self.output_units == len(self.results)
            and self.failed == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "prompt_version": SEMANTIC_AUDIT_PROMPT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "source_units": self.source_units,
            "output_units": self.output_units,
            "audited_units": len(self.results),
            "passed_units": self.passed,
            "failed_units": self.failed,
            "complete": self.complete,
            "requests": self.requests,
            "validation_failures": self.validation_failures,
            "recursive_splits": self.recursive_splits,
            "cache_hits": self.cache_hits,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SemanticAuditReport":
        return cls(
            model=str(data.get("model") or ""),
            source_units=int(data.get("source_units") or 0),
            output_units=int(data.get("output_units") or 0),
            results=[dict(item) for item in (data.get("results") or []) if isinstance(item, Mapping)],
            requests=int(data.get("requests") or 0),
            validation_failures=int(data.get("validation_failures") or 0),
            recursive_splits=int(data.get("recursive_splits") or 0),
            cache_hits=int(data.get("cache_hits") or 0),
        )


def _cache_key(source_hash: str, candidate_hash: str, model: str) -> str:
    return text_sha256(
        f"{SEMANTIC_AUDIT_PROMPT_VERSION}\0{model}\0{source_hash}\0{candidate_hash}"
    )


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "entries": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "entries": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
        return {"schema_version": 1, "entries": {}}
    return payload


def _write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _batches(units: Sequence[StructuredUnit], *, max_units: int, max_chars: int) -> list[list[StructuredUnit]]:
    batches: list[list[StructuredUnit]] = []
    current: list[StructuredUnit] = []
    current_chars = 0
    for unit in units:
        size = len(json.dumps(dict(unit.payload), ensure_ascii=False))
        if current and (len(current) >= max_units or current_chars + size > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(unit)
        current_chars += size
    if current:
        batches.append(current)
    return batches


def _accepted(result: Mapping[str, Any]) -> bool:
    verdict = str(result.get("verdict") or "").strip().casefold()
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    material = any(bool(result.get(field)) for field in _MATERIAL_FIELDS)
    return verdict == "pass" and confidence >= 0.80 and not material


def _audit_payload(source_unit, output_unit) -> dict[str, Any]:
    return {
        "source_document": source_unit.file_href,
        "spine_index": source_unit.spine_index,
        "dom_path": source_unit.dom_path,
        "source_order": source_unit.source_order,
        "source_text": source_unit.text,
        "candidate_text": output_unit.text,
    }


async def audit_epub_units_semantically(
    source: EpubSnapshot,
    output: EpubSnapshot,
    *,
    provider: Any,
    model: str,
    source_language: str,
    target_language: str,
    cache_path: str | Path,
    max_batch_units: int = 6,
    max_batch_chars: int = 24_000,
    max_attempts: int = 2,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SemanticAuditReport:
    """Audit all aligned block units and checkpoint each accepted response."""
    if len(source.units) != len(output.units):
        raise ValueError(f"source/output unit count mismatch: {len(source.units)} != {len(output.units)}")
    for index, (source_unit, output_unit) in enumerate(zip(source.units, output.units)):
        if (
            source_unit.file_href != output_unit.file_href
            or source_unit.ordinal != output_unit.ordinal
            or source_unit.dom_path != output_unit.dom_path
        ):
            raise ValueError(f"unit alignment mismatch at {index}")

    cache_file = Path(cache_path)
    cache = _load_cache(cache_file)
    entries = cache["entries"]
    report = SemanticAuditReport(model=model, source_units=len(source.units), output_units=len(output.units))
    pending: list[StructuredUnit] = []
    unit_meta: dict[str, tuple[Any, Any, str]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    for source_unit, output_unit in zip(source.units, output.units):
        unit_id = source_unit.unit_id
        key = _cache_key(source_unit.source_hash, text_sha256(output_unit.text), model)
        unit_meta[unit_id] = (source_unit, output_unit, key)
        cached = entries.get(key)
        if isinstance(cached, dict) and cached.get("unit_id") == unit_id:
            results_by_id[unit_id] = dict(cached)
            report.cache_hits += 1
        else:
            pending.append(StructuredUnit(unit_id, _audit_payload(source_unit, output_unit)))

    instructions = f"""You are an independent bilingual semantic auditor for a complete EPUB.
Compare each SOURCE_TEXT in {source_language} against its CANDIDATE_TEXT in {target_language}.
Do not rewrite or improve the candidate. Verify complete meaning, negation, causality, chronology,
names, dates, numbers, quotations, references, terminology, register, and order. Flag omissions,
additions, censorship, softening, truncation, duplicated meaning, source-language prose, or structural
loss. Authentic third-language quotations, titles, bibliography, credits, and proper names may remain
foreign when they faithfully match the source. A title or label written in the declared source language
is not third-language content: it must be translated unless an explicit preservation instruction applies.
A publisher or brand name may remain, but ordinary surrounding words must be translated. A fluent candidate is not enough if meaning changed.
Use verdict pass only when no material issue exists; otherwise use warn or fail."""
    response_fields = [
        "verdict", "confidence", "reason", "missing_from_source", "added_not_in_source",
        "changed_facts", "censored_or_softened", "terminology_errors", "structure_issues",
        "source_language_residual",
    ]

    async def request(system: str, user: str):
        if callable(getattr(provider, "make_request", None)):
            return await await_llm_call(
                provider.make_request,
                user,
                model,
                provider=provider,
                system_prompt=system,
            )
        if callable(getattr(provider, "generate", None)):
            return await await_llm_call(
                provider.generate,
                user,
                provider=provider,
                system_prompt=system,
            )
        raise TypeError("semantic audit provider exposes neither make_request nor generate")

    completed = report.cache_hits
    batch_stats = StructuredBatchStats()
    for batch in _batches(pending, max_units=max_batch_units, max_chars=max_batch_chars):
        parsed = await request_structured_units(
            batch,
            request=request,
            system_instructions=instructions,
            response_fields=response_fields,
            required_fields=["verdict", "confidence", "reason"],
            max_attempts=max_attempts,
            stats=batch_stats,
        )
        for unit in batch:
            source_unit, output_unit, key = unit_meta[unit.unit_id]
            result = dict(parsed[unit.unit_id])
            result.update({
                "unit_id": unit.unit_id,
                "source_document": source_unit.file_href,
                "spine_index": source_unit.spine_index,
                "dom_path": source_unit.dom_path,
                "source_order": source_unit.source_order,
                "source_hash": source_unit.source_hash,
                "candidate_hash": text_sha256(output_unit.text),
                "accepted": _accepted(result),
                "audit_model": model,
                "audit_prompt_version": SEMANTIC_AUDIT_PROMPT_VERSION,
            })
            results_by_id[unit.unit_id] = result
            entries[key] = result
        cache["prompt_version"] = SEMANTIC_AUDIT_PROMPT_VERSION
        cache["model"] = model
        _write_cache(cache_file, cache)
        completed += len(batch)
        if progress_callback:
            progress_callback(completed, len(source.units))

    report.requests = batch_stats.requests
    report.validation_failures = batch_stats.validation_failures
    report.recursive_splits = batch_stats.recursive_splits
    report.results = [results_by_id[unit.unit_id] for unit in source.units]
    return report


async def adjudicate_semantic_audit(
    source: EpubSnapshot,
    output: EpubSnapshot,
    report: SemanticAuditReport,
    *,
    provider: Any,
    model: str,
    source_language: str,
    target_language: str,
    cache_path: str | Path,
    max_attempts: int = 2,
    recheck_initial_findings: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SemanticAuditReport:
    """Recheck rejected findings for objective errors versus reviewer preference."""
    source_by_id = {unit.unit_id: unit for unit in source.units}
    source_position = {unit.unit_id: index for index, unit in enumerate(source.units)}
    output_by_position = {
        (unit.file_href, unit.ordinal): unit for unit in output.units
    }
    rejected = [
        item for item in report.results
        if not item.get("accepted")
        or (recheck_initial_findings and item.get("initial_accepted") is False)
    ]
    if not rejected:
        return report
    cache_file = Path(cache_path)
    cache = _load_cache(cache_file)
    entries = cache["entries"]
    pending: list[StructuredUnit] = []
    results: dict[str, dict[str, Any]] = {}
    for first in rejected:
        first_audit = {
            key: value for key, value in first.items()
            if key not in {"adjudication", "acceptance_basis", "initial_accepted"}
        }
        unit_id = str(first.get("unit_id") or "")
        source_unit = source_by_id[unit_id]
        output_unit = output_by_position[(source_unit.file_href, source_unit.ordinal)]
        key = text_sha256(
            f"{SEMANTIC_ADJUDICATION_PROMPT_VERSION}\0{model}\0"
            f"{source_language}\0{target_language}\0"
            f"{source_unit.source_hash}\0{text_sha256(output_unit.text)}\0"
            f"{text_sha256(json.dumps(first_audit, ensure_ascii=False, sort_keys=True))}"
        )
        cached = entries.get(key)
        if isinstance(cached, dict) and cached.get("unit_id") == unit_id:
            results[unit_id] = dict(cached)
            report.cache_hits += 1
            continue
        position = source_position[unit_id]
        previous_source = source.units[position - 1] if position > 0 else None
        previous_output = output.units[position - 1] if position > 0 else None
        next_source = source.units[position + 1] if position + 1 < len(source.units) else None
        next_output = output.units[position + 1] if position + 1 < len(output.units) else None

        def neighbor_text(unit: Any | None, *, file_href: str) -> str:
            if unit is None or unit.file_href != file_href:
                return ""
            text = str(unit.text or "")
            return text[:1_200] if len(text) <= 1_200 else text[:600] + " ... " + text[-600:]

        pending.append(StructuredUnit(unit_id, {
            "source_language": source_language,
            "target_language": target_language,
            "source_text": source_unit.text,
            "candidate_text": output_unit.text,
            "previous_source_text": neighbor_text(previous_source, file_href=source_unit.file_href),
            "previous_candidate_text": neighbor_text(previous_output, file_href=source_unit.file_href),
            "next_source_text": neighbor_text(next_source, file_href=source_unit.file_href),
            "next_candidate_text": neighbor_text(next_output, file_href=source_unit.file_href),
            "first_audit": first_audit,
            "adjudication_cache_key": key,
        }))

    instructions = f"""You are the adjudicator of a prior bilingual fidelity audit.
Determine whether FIRST_AUDIT identified an objective source-to-candidate error or only a stylistic
preference, valid synonym, regional choice, translation of a foreign quotation, or self-contradictory
complaint. Do not consult, cite, reconstruct, or prefer any published/commercial translation.
Judge SOURCE_TEXT in {source_language} against CANDIDATE_TEXT in {target_language}. Neighboring units
are diagnostic context only. Text migrated into a neighboring DOM unit is not a global omission, but it
is an objective structural-boundary error and must not be accepted as aligned.

Source-language prose copied verbatim into the candidate is untranslated residue even when it is an
archaic or historical quotation, unless the candidate also supplies its complete {target_language}
rendering or an explicit preservation instruction is present. A genuine third-language quotation may
be preserved exactly or translated completely. A partially translated quotation that mixes languages
or changes quoted wording is corrupted and must be confirmed.

Confirm an error only for objective omission, addition, changed fact, negation, causality, chronology,
number, name, corrupted quotation, material terminology error, censorship, truncation, untranslated
source prose, or content shifted across unit boundaries. Use exactly verdict false_positive when the
first audit is unsupported and verdict confirmed when an objective error remains."""
    fields = ["verdict", "confidence", "reason", "objective_issues", "recommended_action"]

    async def request(system: str, user: str):
        if callable(getattr(provider, "make_request", None)):
            return await await_llm_call(
                provider.make_request,
                user,
                model,
                provider=provider,
                system_prompt=system,
            )
        return await await_llm_call(
            provider.generate,
            user,
            provider=provider,
            system_prompt=system,
        )

    stats = StructuredBatchStats()
    done = len(results)
    for batch in _batches(pending, max_units=3, max_chars=18_000):
        parsed = await request_structured_units(
            batch,
            request=request,
            system_instructions=instructions,
            response_fields=fields,
            required_fields=["verdict", "confidence", "reason", "recommended_action"],
            max_attempts=max_attempts,
            stats=stats,
        )
        for unit in batch:
            result = dict(parsed[unit.unit_id])
            result["unit_id"] = unit.unit_id
            result["prompt_version"] = SEMANTIC_ADJUDICATION_PROMPT_VERSION
            result["model"] = model
            key = str(unit.payload["adjudication_cache_key"])
            entries[key] = result
            results[unit.unit_id] = result
        cache["prompt_version"] = SEMANTIC_ADJUDICATION_PROMPT_VERSION
        cache["model"] = model
        _write_cache(cache_file, cache)
        done += len(batch)
        if progress_callback:
            progress_callback(done, len(rejected))

    for first in report.results:
        adjudication = results.get(str(first.get("unit_id") or ""))
        if not adjudication:
            continue
        first.setdefault("initial_accepted", bool(first.get("accepted")))
        first["adjudication"] = adjudication
        verdict = str(adjudication.get("verdict") or "").casefold()
        try:
            confidence = float(adjudication.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if _adjudication_accepts_false_positive(adjudication, confidence=confidence):
            first["accepted"] = True
            first["acceptance_basis"] = "adjudicated_false_positive"
        else:
            first["accepted"] = False
            first["acceptance_basis"] = "confirmed_or_unresolved_error"
    report.requests += stats.requests
    report.validation_failures += stats.validation_failures
    report.recursive_splits += stats.recursive_splits
    return report


def _adjudication_accepts_false_positive(
    adjudication: Mapping[str, Any],
    *,
    confidence: float,
) -> bool:
    """Normalize equivalent reviewer labels without accepting confirmed errors.

    Providers occasionally answer ``pass`` or ``no_error`` despite an enum in
    the prompt. Treat those as the requested ``false_positive`` only when no
    objective issue was returned. Explicit confirmed/fail labels always win.
    """
    if confidence < 0.80:
        return False
    verdict = str(adjudication.get("verdict") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    action = str(adjudication.get("recommended_action") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    objective_issues = adjudication.get("objective_issues")
    has_objective_issues = bool(objective_issues)
    confirmed_labels = {"confirmed", "true_positive", "fail", "failed", "error", "warn", "warning"}
    false_positive_labels = {"false_positive", "pass", "passed", "no_error", "valid", "not_confirmed"}
    if verdict in confirmed_labels or has_objective_issues:
        return False
    if verdict in false_positive_labels:
        return True
    return action in {"false_positive", "accept", "accepted", "no_change", "none"}
