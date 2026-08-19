"""Safe helpers for extracting JSON from LLM responses.

LLM judges often wrap JSON in tags, markdown fences, or short prose. Avoid
regexes such as ``\\{.*\\}`` here: on malformed or very long responses they can
hold the GIL long enough to make the web server look frozen.
"""

from __future__ import annotations

import json
from typing import Any


DEFAULT_JSON_SCAN_LIMIT = 200_000


def extract_tagged_payload(
    text: str,
    start_tag: str,
    end_tag: str,
    *,
    max_chars: int = DEFAULT_JSON_SCAN_LIMIT,
) -> str | None:
    """Return content between tags using bounded string search."""
    if not text or not start_tag or not end_tag:
        return None
    start = text.find(start_tag)
    if start < 0:
        return None
    payload_start = start + len(start_tag)
    end = text.find(end_tag, payload_start)
    if end < 0:
        return None
    if end - payload_start > max_chars:
        return None
    return text[payload_start:end].strip()


def strip_markdown_json_fence(text: str) -> str:
    """Strip a single enclosing markdown fence without regex backtracking."""
    payload = (text or "").strip()
    if not payload.startswith("```"):
        return payload

    first_line_end = payload.find("\n")
    last_fence = payload.rfind("```")
    if first_line_end < 0 or last_fence <= first_line_end:
        return payload

    fence_label = payload[3:first_line_end].strip().lower()
    if fence_label and fence_label != "json":
        return payload
    return payload[first_line_end + 1:last_fence].strip()


def extract_first_json_value_text(
    text: str,
    *,
    max_chars: int = DEFAULT_JSON_SCAN_LIMIT,
) -> str | None:
    """Extract the first balanced JSON object or array from text.

    This is a linear scanner that respects JSON strings and escapes. It returns
    quickly on incomplete payloads instead of asking the regex engine to search
    across arbitrary text.
    """
    payload = strip_markdown_json_fence(text)
    if not payload:
        return None

    scan = payload[:max_chars]
    starts = [idx for idx in (scan.find("{"), scan.find("[")) if idx >= 0]
    if not starts:
        return None
    start = min(starts)
    opener = scan[start]
    closer = "}" if opener == "{" else "]"

    stack = [closer]
    in_string = False
    escaped = False
    for idx in range(start + 1, len(scan)):
        char = scan[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or char != stack[-1]:
                return None
            stack.pop()
            if not stack:
                return scan[start:idx + 1]
    return None


def loads_first_json_value(
    text: str,
    *,
    max_chars: int = DEFAULT_JSON_SCAN_LIMIT,
) -> Any | None:
    """Parse either the whole text or the first JSON value embedded in it."""
    payload = strip_markdown_json_fence(text)
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    extracted = extract_first_json_value_text(payload, max_chars=max_chars)
    if extracted is None:
        return None
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        return None


def loads_first_json_object(
    text: str,
    *,
    max_chars: int = DEFAULT_JSON_SCAN_LIMIT,
) -> dict[str, Any] | None:
    """Parse the first JSON object embedded in text."""
    parsed = loads_first_json_value(text, max_chars=max_chars)
    return parsed if isinstance(parsed, dict) else None
