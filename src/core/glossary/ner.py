"""
NER-assisted glossary extraction (Phase 2 of the glossary plan).

Given a sample of source text, ask the configured LLM to propose recurring
named entities (characters, locations, sects, items) with a suggested target
translation. The user reviews the candidates before adding them to a glossary
— nothing is auto-applied.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm.request_deadline import await_llm_call

logger = logging.getLogger("glossary.ner")

ALLOWED_CATEGORIES = {"character", "location", "organization", "item", "title", "other"}

NER_TAG_IN = "<NER_JSON>"
NER_TAG_OUT = "</NER_JSON>"


def parse_ner_response(raw: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Permissive parser for NER output.

    Tries, in order:
      1. Content between <NER_JSON>...</NER_JSON> tags.
      2. Content inside the first markdown ```json fence.
      3. The longest balanced [...] JSON array.
      4. The longest balanced {...} JSON object (and pull a list out of any value).

    Returns (candidates, warnings). `candidates` is a list of dicts with at
    least `source` and one of `target`/`category` populated. `warnings` is a
    list of human-readable issues that the caller should surface.
    """
    if not raw:
        return [], ["empty LLM response"]

    text = _strip_thinking_blocks(raw).strip()
    warnings: List[str] = []

    payload = _extract_payload(text, warnings)
    if payload is None:
        return [], warnings + ["could not locate any JSON payload in response"]

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        repaired = _try_repair_json(payload)
        if repaired is None:
            return [], warnings + [f"JSON parse error: {e}"]
        try:
            data = json.loads(repaired)
            warnings.append("JSON was repaired before parsing (trailing comma or similar)")
        except json.JSONDecodeError as e2:
            return [], warnings + [f"JSON parse error after repair: {e2}"]

    items = _coerce_to_list_of_dicts(data, warnings)

    candidates: List[Dict[str, str]] = []
    seen_sources: set[str] = set()
    for entry in items:
        source = _str(entry.get("source") or entry.get("source_term"))
        target = _str(entry.get("target") or entry.get("translated_term") or entry.get("translation"))
        category = _str(entry.get("category") or entry.get("type") or "").lower()

        if not source:
            warnings.append("skipped entry without 'source'")
            continue
        if source in seen_sources:
            continue
        seen_sources.add(source)

        if category and category not in ALLOWED_CATEGORIES:
            warnings.append(f"unknown category '{category}' for '{source}' (kept as-is)")

        candidates.append({
            "source": source,
            "target": target,
            "category": category or "other",
        })

    return candidates, warnings


async def suggest_terms(
    text: str,
    source_language: str,
    target_language: str,
    llm_provider,
    max_chars: int = 6000,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Run the NER prompt against `text` (truncated to `max_chars`) using the
    given provider. Returns (candidates, warnings).
    """
    from src.prompts.prompts import generate_ner_extraction_prompt

    sample = text[:max_chars] if max_chars and len(text) > max_chars else text
    prompt = generate_ner_extraction_prompt(sample, source_language, target_language)

    response = await await_llm_call(
        llm_provider.generate,
        prompt.user,
        provider=llm_provider,
        system_prompt=prompt.system,
    )
    if response is None:
        return [], ["LLM returned no response"]

    raw = getattr(response, "content", None) or str(response)
    candidates, warnings = parse_ner_response(raw)

    if not candidates:
        snippet = (raw or '').strip().replace('\n', ' ')[:400]
        logger.info(
            "NER returned 0 candidates (sample_chars=%d, response_chars=%d): %s",
            len(sample), len(raw or ''), snippet,
        )
        if not warnings:
            warnings.append("LLM returned a valid empty list — no recurring entities detected")

    return candidates, warnings


def _strip_thinking_blocks(text: str) -> str:
    """Strip any <think>...</think> blocks emitted by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_payload(text: str, warnings: List[str]) -> Optional[str]:
    tag_match = re.search(
        re.escape(NER_TAG_IN) + r"\s*(.*?)\s*" + re.escape(NER_TAG_OUT),
        text,
        flags=re.DOTALL,
    )
    if tag_match:
        return tag_match.group(1).strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        warnings.append("NER tags missing — extracted from markdown code fence")
        return fence_match.group(1).strip()

    # Pick whichever balanced container appears FIRST. If the response is a
    # bare object that wraps the array (e.g. `{"entities": [...]}`), the
    # object's `{` precedes the array's `[`, so the object wins and the
    # `_coerce_to_list_of_dicts` unwrap path can produce the right warning.
    obj_pos = text.find("{")
    arr_pos = text.find("[")

    if obj_pos != -1 and (arr_pos == -1 or obj_pos < arr_pos):
        obj = _find_balanced(text, "{", "}")
        if obj:
            warnings.append("NER tags missing — extracted balanced JSON object")
            return obj

    if arr_pos != -1:
        array = _find_balanced(text, "[", "]")
        if array:
            warnings.append("NER tags missing — extracted balanced JSON array")
            return array

    if obj_pos != -1:
        obj = _find_balanced(text, "{", "}")
        if obj:
            warnings.append("NER tags missing — extracted balanced JSON object")
            return obj

    return None


def _find_balanced(text: str, opener: str, closer: str) -> Optional[str]:
    start = text.find(opener)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


def _try_repair_json(payload: str) -> Optional[str]:
    """Best-effort repairs: strip trailing commas before } or ]."""
    repaired = re.sub(r",\s*([}\]])", r"\1", payload)
    return repaired if repaired != payload else None


def _coerce_to_list_of_dicts(data: Any, warnings: List[str]) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        for key in ("entities", "terms", "candidates", "items", "results"):
            value = data.get(key)
            if isinstance(value, list):
                warnings.append(f"unwrapped list from '{key}' field")
                return [d for d in value if isinstance(d, dict)]
        if "source" in data:
            return [data]
        warnings.append("response was an object without a recognized list field")
        return []
    warnings.append(f"unexpected JSON root type: {type(data).__name__}")
    return []


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()
