"""Exact-ID contracts for batched LLM unit responses."""

from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import random
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence


RESULT_TAG_IN = "<UNIT_RESULTS_JSON>"
RESULT_TAG_OUT = "</UNIT_RESULTS_JSON>"


class StructuredUnitResponseError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class StructuredUnit:
    unit_id: str
    payload: Mapping[str, Any]


@dataclass
class StructuredBatchStats:
    requests: int = 0
    validation_failures: int = 0
    recursive_splits: int = 0


def build_structured_unit_prompt(
    units: Sequence[StructuredUnit],
    *,
    system_instructions: str,
    response_fields: Sequence[str],
) -> tuple[str, str]:
    """Build a stable exact-ID prompt; variable unit data stays in user content."""
    fields = ["unit_id", *[str(field) for field in response_fields if str(field) != "unit_id"]]
    system = (
        f"{system_instructions.strip()}\n\n"
        "Return only one JSON object wrapped in <UNIT_RESULTS_JSON> and </UNIT_RESULTS_JSON>. "
        "The root must be {\"units\": [...]}. Return every requested unit_id exactly once, "
        "no unknown IDs, no commentary, and no empty required fields. "
        f"Each unit object must contain: {', '.join(fields)}."
    )
    user = json.dumps(
        {"requested_units": [{"unit_id": unit.unit_id, **dict(unit.payload)} for unit in units]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return system, user


def _extract_exact_json(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        raise StructuredUnitResponseError("empty_response", "model returned no content")
    start = raw.find(RESULT_TAG_IN)
    end = raw.find(RESULT_TAG_OUT)
    if start >= 0 or end >= 0:
        if start < 0 or end < 0 or end < start:
            raise StructuredUnitResponseError("truncated_response", "structured response tags are incomplete")
        before = raw[:start].strip()
        after = raw[end + len(RESULT_TAG_OUT):].strip()
        if before or after:
            raise StructuredUnitResponseError("unexpected_text", "content exists outside structured tags")
        return raw[start + len(RESULT_TAG_IN):end].strip()
    return raw


def _placeholder_counts(text: str, placeholders: Iterable[str]) -> dict[str, int]:
    return {placeholder: str(text or "").count(placeholder) for placeholder in placeholders}


def parse_structured_unit_response(
    text: str,
    *,
    requested_ids: Sequence[str],
    required_fields: Sequence[str],
    placeholder_manifest: Mapping[str, Sequence[str]] | None = None,
    response_truncated: bool = False,
) -> dict[str, dict[str, Any]]:
    """Parse and validate complete one-to-one coverage for a model batch."""
    if response_truncated:
        raise StructuredUnitResponseError("truncated_response", "provider marked response as truncated")
    payload_text = _extract_exact_json(text)
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        code = "truncated_json" if exc.pos >= max(0, len(payload_text) - 3) else "invalid_json"
        raise StructuredUnitResponseError(code, str(exc)) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), list):
        raise StructuredUnitResponseError("invalid_schema", "root.units must be an array")

    requested = [str(unit_id) for unit_id in requested_ids]
    requested_set = set(requested)
    if len(requested_set) != len(requested):
        raise StructuredUnitResponseError("duplicate_request_id", "requested IDs are not unique")
    parsed: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(payload["units"]):
        if not isinstance(item, dict):
            raise StructuredUnitResponseError("invalid_unit", f"unit {index} is not an object")
        unit_id = str(item.get("unit_id") or "")
        if not unit_id:
            raise StructuredUnitResponseError("missing_id", f"unit {index} has no unit_id")
        if unit_id not in requested_set:
            raise StructuredUnitResponseError("unknown_id", unit_id)
        if unit_id in parsed:
            raise StructuredUnitResponseError("duplicate_id", unit_id)
        for field in required_fields:
            value = item.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise StructuredUnitResponseError("empty_field", f"{unit_id}.{field}")
        expected_placeholders = list((placeholder_manifest or {}).get(unit_id) or [])
        if expected_placeholders and "text" in item:
            counts = _placeholder_counts(str(item.get("text") or ""), expected_placeholders)
            invalid = {key: value for key, value in counts.items() if value != 1}
            if invalid:
                raise StructuredUnitResponseError("placeholder_mismatch", f"{unit_id}: {invalid}")
        parsed[unit_id] = dict(item)

    missing = [unit_id for unit_id in requested if unit_id not in parsed]
    if missing:
        raise StructuredUnitResponseError("missing_ids", ", ".join(missing[:8]))
    if len(parsed) != len(requested):
        raise StructuredUnitResponseError("coverage_mismatch", f"{len(parsed)} != {len(requested)}")
    return {unit_id: parsed[unit_id] for unit_id in requested}


def bisect_structured_units(units: Sequence[StructuredUnit]) -> tuple[list[StructuredUnit], list[StructuredUnit]]:
    """Deterministically split an invalid batch without changing source order."""
    if len(units) < 2:
        return list(units), []
    midpoint = max(1, len(units) // 2)
    return list(units[:midpoint]), list(units[midpoint:])


async def request_structured_units(
    units: Sequence[StructuredUnit],
    *,
    request: Callable[[str, str], Awaitable[Any]],
    system_instructions: str,
    response_fields: Sequence[str],
    required_fields: Sequence[str],
    placeholder_manifest: Mapping[str, Sequence[str]] | None = None,
    max_attempts: int = 2,
    stats: StructuredBatchStats | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, dict[str, Any]]:
    """Request exact-ID results, recursively reducing invalid batches."""
    ordered_units = list(units)
    if not ordered_units:
        return {}
    stats = stats or StructuredBatchStats()
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        system, user = build_structured_unit_prompt(
            ordered_units,
            system_instructions=system_instructions,
            response_fields=response_fields,
        )
        try:
            stats.requests += 1
            response = await request(system, user)
            content = str(getattr(response, "content", response) or "")
            return parse_structured_unit_response(
                content,
                requested_ids=[unit.unit_id for unit in ordered_units],
                required_fields=required_fields,
                placeholder_manifest=placeholder_manifest,
                response_truncated=bool(getattr(response, "was_truncated", False)),
            )
        except StructuredUnitResponseError as exc:
            last_error = exc
            stats.validation_failures += 1
            if attempt + 1 < max(1, int(max_attempts)):
                delay = min(2.0, 0.2 * (2 ** attempt)) + random.uniform(0.0, 0.05)
                await sleep(delay)
    if len(ordered_units) == 1:
        assert last_error is not None
        raise last_error
    left, right = bisect_structured_units(ordered_units)
    stats.recursive_splits += 1
    left_result = await request_structured_units(
        left,
        request=request,
        system_instructions=system_instructions,
        response_fields=response_fields,
        required_fields=required_fields,
        placeholder_manifest=placeholder_manifest,
        max_attempts=max_attempts,
        stats=stats,
        sleep=sleep,
    )
    right_result = await request_structured_units(
        right,
        request=request,
        system_instructions=system_instructions,
        response_fields=response_fields,
        required_fields=required_fields,
        placeholder_manifest=placeholder_manifest,
        max_attempts=max_attempts,
        stats=stats,
        sleep=sleep,
    )
    return {
        unit.unit_id: (left_result | right_result)[unit.unit_id]
        for unit in ordered_units
    }
