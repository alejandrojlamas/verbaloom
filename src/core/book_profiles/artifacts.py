"""Editorial preparation artifacts for book profiles.

The glossary answers "what terms should stay consistent?". These artifacts
answer broader editorial questions before a long job starts: who/what recurs,
where the book is structured, what voices appear, and what risks the pipeline
should watch. The data is compact and scoped to one profile.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import yaml


EDITORIAL_ARTIFACT_VERSION = "editorial-prep-v3"
EDITORIAL_SIGNAL_INDEX_VERSION = "editorial-signal-index-v1"
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

ARTIFACT_KEYS = (
    "entities",
    "voices",
    "chapters",
    "translatable_terms",
    "preserve_terms",
    "risks",
    "canonical_names",
    "blockers",
    "characters_entities",
    "sections",
    "narrative_voices",
    "relationships",
    "technical_cultural_terms",
    "iconic_phrases",
    "do_not_translate",
    "editorial_risks",
)
PROMPT_ARTIFACT_KEYS = (
    "entities",
    "voices",
    "chapters",
    "translatable_terms",
    "preserve_terms",
    "canonical_names",
    "blockers",
    "risks",
    "relationships",
    "technical_cultural_terms",
    "iconic_phrases",
)

_ENTITY_CATEGORIES = {"character", "location", "organization", "title"}
_TECH_CATEGORIES = {"technical", "concept", "acronym", "title", "item"}
_MAX_ITEMS_PER_BUCKET = {
    "entities": 180,
    "voices": 50,
    "chapters": 180,
    "translatable_terms": 260,
    "preserve_terms": 220,
    "risks": 100,
    "canonical_names": 220,
    "blockers": 140,
    "characters_entities": 160,
    "sections": 180,
    "narrative_voices": 40,
    "relationships": 120,
    "technical_cultural_terms": 220,
    "iconic_phrases": 80,
    "do_not_translate": 180,
    "editorial_risks": 80,
}
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")
_HEADING_MARKER_RE = re.compile(
    r"^(?:"
    r"(?:chapter|capitulo|cap[ií]tulo|section|seccion|secci[oó]n|part|parte|book|libro)"
    r"(?:\s+[\wIVXLCDMivxlcdm.-]+)?"
    r"|(?:prologue|prologo|pr[oó]logo|preface|prefacio|introduction|introducci[oó]n)"
    r"|(?:contents|contenido|indice|[ií]ndice)"
    r")\b",
    re.IGNORECASE,
)
_MOJIBAKE_RE = re.compile(r"Ã.|Â.|�|[■□▮▯]")
_DOT_LEADER_RE = re.compile(r"\.{8,}|(?:\.\s*){8,}")
_FORMULA_RE = re.compile(r"(?:[=<>+\-*/^_{}]|[\u0370-\u03ff]|10\^|\\(?:frac|sum|int|sqrt))")
_FOOTNOTE_RE = re.compile(r"(?:\[\d{1,3}\]|\(\d{1,3}\)|^\s*\d{1,3}\.\s+)", re.MULTILINE)
_QUOTED_PHRASE_RE = re.compile(r"[\"“”'‘’«»]([^\"“”'‘’«»]{8,90})[\"“”'‘’«»]")


def build_local_editorial_map(
    text: str,
    *,
    profile_id: str,
    source_name: str,
    source_language: str = "",
    language: str = "",
    target_locale: str = "",
    transform_mode: str = "modernize",
    local_candidates: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a token-free editorial map from the full available text."""
    source_text = text or ""
    candidates = [dict(item) for item in local_candidates if isinstance(item, Mapping)]
    sections = _extract_sections(source_text)
    entities = _entities_from_candidates(candidates)
    technical_terms = _technical_terms_from_candidates(candidates)
    do_not_translate = _do_not_translate_from_candidates(
        candidates,
        transform_mode=transform_mode,
        target_locale=target_locale,
    )
    iconic = _recurrent_quoted_phrases(source_text)
    relationships = _relationship_map(source_text, entities)
    voices = _local_voice_signals(source_text)
    risks = _local_editorial_risks(source_text, candidates, sections)

    base = {
        "version": EDITORIAL_ARTIFACT_VERSION,
        "profile_id": profile_id,
        "source_name": source_name,
        "source_language": source_language,
        "language": language,
        "target_locale": target_locale,
        "transform_mode": transform_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_stats": {
            "text_chars": len(source_text),
            "approx_words": len(re.findall(r"\w+", source_text, flags=re.UNICODE)),
            "paragraphs": len([p for p in re.split(r"\n{2,}", source_text) if p.strip()]),
            "local_candidates": len(candidates),
        },
        "characters_entities": entities,
        "sections": sections,
        "narrative_voices": voices,
        "relationships": relationships,
        "technical_cultural_terms": technical_terms,
        "iconic_phrases": iconic,
        "do_not_translate": do_not_translate,
        "editorial_risks": risks,
    }
    base.update(_v3_alias_buckets(base))
    return _normalise_map(base)


def enrich_editorial_map_with_reviewed_terms(
    editorial_map: Mapping[str, Any],
    *,
    reviewed_candidates: Iterable[Mapping[str, Any]] = (),
    approved_entries: Iterable[Mapping[str, Any]] = (),
    pending_suggestions: Iterable[Mapping[str, Any]] = (),
    profile_id: str = "",
    target_locale: str = "",
    transform_mode: str = "",
) -> dict[str, Any]:
    """Add profile-review decisions to the broader editorial map.

    The local and LLM discovery stages find signals; the reviewer decides how
    those signals should be used. This enrichment makes those decisions visible
    as a compact editorial map for the UI and for prompt briefs.
    """
    data = _normalise_map(editorial_map)
    reviewed_candidates = [
        dict(item) for item in reviewed_candidates if isinstance(item, Mapping)
    ]
    _remove_unapproved_local_entity_signals(data, reviewed_candidates)
    for key, items in _v3_alias_buckets(data).items():
        for item in items:
            _append_unique(data[key], item, key_fields=_dedupe_fields_for_key(key))

    for raw in approved_entries:
        if not isinstance(raw, Mapping):
            continue
        source = _short(raw.get("source"), 140)
        target = _short(raw.get("target"), 140)
        entry_type = str(raw.get("type") or raw.get("entry_type") or "term").strip().lower()
        policy = str(raw.get("translation_policy") or raw.get("injection_policy") or "").strip().lower()
        item = {
            "source": source,
            "target": target,
            "type": entry_type,
            "policy": policy or "approved",
            "confidence": _float(raw.get("confidence"), 0.0),
            "rationale": _short(raw.get("rationale") or raw.get("review_rationale"), 220),
            "source_kind": "reviewed_glossary",
        }
        if not source:
            continue
        if policy in {"translate_exact", "translate_contextual", "contextual"} and target and target.casefold() != source.casefold():
            _append_unique(data["translatable_terms"], item, key_fields=("source",))
        elif entry_type in {"canonical_proper_noun", "proper_noun", "character", "location", "organization", "title"}:
            canonical = {
                **item,
                "name": source,
                "canonical": target or source,
            }
            _append_unique(data["canonical_names"], canonical, key_fields=("name", "source"))
            _append_unique(data["preserve_terms"], item, key_fields=("source",))
        elif policy in {"preserve_exact", "preserve"}:
            _append_unique(data["preserve_terms"], item, key_fields=("source",))

    for raw in reviewed_candidates:
        source = _short(raw.get("source"), 140)
        if not source:
            continue
        status = str(raw.get("review_status") or "").strip().lower()
        entry_type = str(raw.get("review_entry_type") or raw.get("category") or raw.get("type") or "term").strip().lower()
        target = _short(raw.get("review_target") or raw.get("target"), 140)
        confidence = _float(raw.get("review_confidence") or raw.get("confidence"), 0.0)
        demoted_reason = _short(raw.get("review_demoted_reason"), 260)
        base = {
            "source": source,
            "target": target,
            "type": entry_type,
            "status": status or "pending_review",
            "confidence": confidence,
            "reason": _short(raw.get("review_rationale") or raw.get("rationale"), 260),
            "source_kind": str(raw.get("reviewed_by") or "profile_term_review"),
        }
        if status in {"translate_exact", "translate_contextual", "pending_review"} and entry_type in {"technical_term", "concept", "term", "idiom", "syntax_pattern"}:
            _append_unique(data["translatable_terms"], base, key_fields=("source",))
        if status == "reject_noise" or demoted_reason:
            blocker = {
                **base,
                "code": "source_equals_target_translatable" if demoted_reason else "rejected_noise",
                "severity": "medium" if demoted_reason else "low",
                "reason": demoted_reason or base["reason"] or "Rejected by profile term reviewer.",
            }
            _append_unique(data["blockers"], blocker, key_fields=("source", "code"))

    for raw in pending_suggestions:
        if not isinstance(raw, Mapping):
            continue
        source = _short(raw.get("source"), 140)
        if not source:
            continue
        entry_type = str(raw.get("type") or raw.get("entry_type") or "term").strip().lower()
        policy = str(raw.get("translation_policy") or raw.get("injection_policy") or "").strip().lower()
        target = _short(raw.get("suggested_target") or raw.get("target"), 140)
        item = {
            "source": source,
            "target": target,
            "type": entry_type,
            "status": "pending",
            "policy": policy or "pending_review",
            "confidence": _float(raw.get("confidence"), 0.0),
            "reason": _short(raw.get("rationale") or raw.get("review_rationale"), 240),
            "source_kind": "pending_suggestion",
        }
        if policy in {"contextual", "translate_contextual", "pending_review"} and entry_type in {"technical_term", "concept", "term", "idiom", "syntax_pattern"}:
            _append_unique(data["translatable_terms"], item, key_fields=("source",))
        if raw.get("risks"):
            item["risks"] = _short_list(raw.get("risks"), max_items=4, max_chars=160)
            _append_unique(data["blockers"], item, key_fields=("source",))

    data["profile_id"] = data.get("profile_id") or profile_id
    if target_locale:
        data["target_locale"] = target_locale
    if transform_mode:
        data["transform_mode"] = transform_mode
    return _normalise_map(data)


def merge_llm_editorial_map(
    editorial_map: Mapping[str, Any],
    llm_map: Mapping[str, Any] | None,
    *,
    profile_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    """Merge compact LLM discoveries into an existing editorial map."""
    merged = _normalise_map(editorial_map)
    if not isinstance(llm_map, Mapping):
        return merged

    # Discovery output is advisory until the Pro reviewer approves concrete
    # glossary entries. Flash may propose names, but it must never create a
    # binding preserve/canonical decision directly in the prompt artifacts.
    advisory_only_blocked_keys = {"preserve_terms", "do_not_translate"}
    for key in ARTIFACT_KEYS:
        if key in advisory_only_blocked_keys:
            continue
        incoming = llm_map.get(key) or []
        if not isinstance(incoming, list):
            continue
        for raw in incoming:
            item = _normalise_llm_item(
                raw,
                key=key,
                profile_id=profile_id,
                chunk_index=chunk_index,
            )
            if item:
                _append_unique(merged[key], item, key_fields=_dedupe_fields_for_key(key))
        merged[key] = _rank_and_cap(key, merged[key])
    return merged


def _remove_unapproved_local_entity_signals(
    data: dict[str, Any],
    reviewed_candidates: Iterable[Mapping[str, Any]],
) -> None:
    """Remove local proper-name guesses that the term reviewer did not approve.

    Capitalization-based extraction is useful for candidate discovery but is
    not a safe preserve decision, especially in languages that capitalize
    common nouns. Pending/rejected/translated candidates remain available in
    the glossary review artifacts; they are simply kept out of prompt buckets
    that could instruct the translator to preserve them.
    """
    unapproved: set[str] = set()
    for raw in reviewed_candidates:
        source = str(raw.get("source") or "").strip().casefold()
        status = str(raw.get("review_status") or "pending_review").strip().lower()
        confidence = _float(raw.get("review_confidence") or raw.get("confidence"), 0.0)
        if source and (status not in {"preserve_exact", "canonical_name"} or confidence < 0.88):
            unapproved.add(source)
    if not unapproved:
        return

    label_fields = {
        "entities": ("name", "source"),
        "characters_entities": ("name", "source"),
        "preserve_terms": ("source", "term", "name"),
        "canonical_names": ("name", "source"),
        "do_not_translate": ("source", "term", "name"),
    }
    for key, fields in label_fields.items():
        items = data.get(key) if isinstance(data.get(key), list) else []
        data[key] = [
            item for item in items
            if not any(
                str(item.get(field) or "").strip().casefold() in unapproved
                for field in fields
            )
        ]


def write_editorial_artifacts(profile_dir: Path, editorial_map: Mapping[str, Any]) -> tuple[Path, Path]:
    """Persist machine-readable and human-readable profile artifacts."""
    editorial_dir = Path(profile_dir) / "editorial"
    editorial_dir.mkdir(parents=True, exist_ok=True)
    data = _normalise_map(editorial_map)
    map_path = editorial_dir / "editorial_map.yml"
    brief_path = editorial_dir / "editorial_brief.md"
    map_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    brief_path.write_text(render_editorial_brief(data), encoding="utf-8")
    return map_path, brief_path


def build_editorial_signal_index(
    *,
    text: str,
    profile_id: str,
    source_name: str,
    language: str = "",
    target_locale: str = "",
    transform_mode: str = "",
    local_candidates: Iterable[Mapping[str, Any]] = (),
    reviewed_candidates: Iterable[Mapping[str, Any]] = (),
    approved_entries: Iterable[Mapping[str, Any]] = (),
    pending_suggestions: Iterable[Mapping[str, Any]] = (),
    llm_suggestions: Iterable[Mapping[str, Any]] = (),
    llm_chunks: int = 0,
    coverage_mode: str = "sampled",
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    """Persist profile signal metrics without storing the whole book text."""
    local = [dict(item) for item in local_candidates if isinstance(item, Mapping)]
    reviewed = [dict(item) for item in reviewed_candidates if isinstance(item, Mapping)]
    approved = [dict(item) for item in approved_entries if isinstance(item, Mapping)]
    pending = [dict(item) for item in pending_suggestions if isinstance(item, Mapping)]
    llm = [dict(item) for item in llm_suggestions if isinstance(item, Mapping)]
    source_text = text or ""
    return {
        "version": EDITORIAL_SIGNAL_INDEX_VERSION,
        "profile_id": profile_id,
        "source_name": source_name,
        "language": language,
        "target_locale": target_locale,
        "transform_mode": transform_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_stats": {
            "text_chars": len(source_text),
            "approx_words": len(re.findall(r"\w+", source_text, flags=re.UNICODE)),
            "paragraphs": len([p for p in re.split(r"\n{2,}", source_text) if p.strip()]),
        },
        "local_candidates": _signal_bucket(local, key="category", limit=700),
        "reviewed_candidates": _signal_bucket(reviewed, key="review_status", limit=700),
        "approved_entries": _signal_bucket(approved, key="type", limit=500),
        "pending_suggestions": _signal_bucket(pending, key="type", limit=700),
        "llm_suggestions": _signal_bucket(llm, key="type", limit=500),
        "coverage": {
            "mode": coverage_mode,
            "llm_chunks": int(llm_chunks or 0),
            "warnings": list(dict.fromkeys(str(item) for item in warnings if str(item).strip())),
        },
        "risk_flags": _signal_risk_flags(local, reviewed, approved, pending),
    }


def write_editorial_signal_index(profile_dir: Path, signal_index: Mapping[str, Any]) -> Path:
    editorial_dir = Path(profile_dir) / "editorial"
    editorial_dir.mkdir(parents=True, exist_ok=True)
    path = editorial_dir / "signal_index.yml"
    path.write_text(
        yaml.safe_dump(dict(signal_index or {}), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_editorial_signal_index(profile_dir: Path) -> dict[str, Any]:
    path = Path(profile_dir) / "editorial" / "signal_index.yml"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.load(text, Loader=_YAML_SAFE_LOADER) if text.strip() else {}
    return dict(data) if isinstance(data, Mapping) else {}


def load_editorial_artifacts(profile_dir: Path) -> dict[str, Any]:
    path = Path(profile_dir) / "editorial" / "editorial_map.yml"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.load(text, Loader=_YAML_SAFE_LOADER) if text.strip() else {}
    if not isinstance(data, Mapping):
        return {}
    # Files written by ``write_editorial_artifacts`` are already ranked,
    # deduplicated and capped. Re-normalising those large maps on every profile
    # list/read duplicates all v2 aliases and performs millions of comparisons.
    if _is_serialized_normalized_map(data):
        return dict(data)
    return _normalise_map(data)


def editorial_artifact_counts(editorial_map: Mapping[str, Any]) -> dict[str, int]:
    data = (
        dict(editorial_map)
        if _is_serialized_normalized_map(editorial_map)
        else _normalise_map(editorial_map)
    )
    return {
        key: len([item for item in data.get(key, []) if isinstance(item, Mapping)])
        for key in ARTIFACT_KEYS
    }


def render_editorial_brief(editorial_map: Mapping[str, Any], *, max_items: int = 12) -> str:
    data = _normalise_map(editorial_map)
    lines = [
        f"# Editorial Brief - {data.get('source_name') or data.get('profile_id') or 'book'}",
        "",
        f"- Profile: {data.get('profile_id') or 'unknown'}",
        f"- Artifact version: {data.get('version') or EDITORIAL_ARTIFACT_VERSION}",
        f"- Text chars: {data.get('source_stats', {}).get('text_chars', 0)}",
        "",
    ]
    sections = [
        ("Entities", "entities"),
        ("Voices", "voices"),
        ("Chapters / sections", "chapters"),
        ("Translatable terms", "translatable_terms"),
        ("Preserve terms", "preserve_terms"),
        ("Canonical names", "canonical_names"),
        ("Blockers", "blockers"),
        ("Risks", "risks"),
        ("Relationships", "relationships"),
        ("Technical / cultural terms", "technical_cultural_terms"),
        ("Iconic phrases", "iconic_phrases"),
    ]
    for title, key in sections:
        items = [item for item in data.get(key, []) if isinstance(item, Mapping)]
        if not items:
            continue
        lines.extend([f"## {title}", ""])
        for item in items[:max_items]:
            label = _item_label(item, key)
            detail = _item_detail(item)
            lines.append(f"- {label}{': ' + detail if detail else ''}")
        if len(items) > max_items:
            lines.append(f"- ... {len(items) - max_items} more")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compact_editorial_brief_for_prompt(
    editorial_map: Mapping[str, Any],
    *,
    max_chars: int = 2600,
) -> str:
    """Render a bounded prompt block from saved editorial artifacts."""
    data = _normalise_map(editorial_map)
    if not data or not any(data.get(key) for key in ARTIFACT_KEYS):
        return ""
    lines = [
        "# EDITORIAL PREP BRIEF",
        "This compact profile artifact was prepared before the main job. Use it as scoped guidance, not as global rules.",
    ]
    bucket_limits = {
        "entities": 10,
        "voices": 5,
        "chapters": 8,
        "translatable_terms": 10,
        "preserve_terms": 8,
        "risks": 6,
        "canonical_names": 10,
        "blockers": 6,
        "characters_entities": 10,
        "sections": 8,
        "narrative_voices": 5,
        "relationships": 6,
        "technical_cultural_terms": 10,
        "iconic_phrases": 6,
        "do_not_translate": 8,
        "editorial_risks": 6,
    }
    labels = {
        "entities": "Entities",
        "voices": "Voices",
        "chapters": "Chapters",
        "translatable_terms": "Translatable terms",
        "preserve_terms": "Preserve",
        "risks": "Risks",
        "canonical_names": "Canonical names",
        "blockers": "Blockers",
        "characters_entities": "Entities",
        "sections": "Sections",
        "narrative_voices": "Voices",
        "relationships": "Relationships",
        "technical_cultural_terms": "Technical/cultural",
        "iconic_phrases": "Iconic/recurrent phrases",
        "do_not_translate": "Do not translate/modernize",
        "editorial_risks": "Risks",
    }
    for key in PROMPT_ARTIFACT_KEYS:
        items = [item for item in data.get(key, []) if isinstance(item, Mapping)]
        if not items:
            continue
        rendered = [_item_label(item, key) for item in items[: bucket_limits.get(key, 6)]]
        if rendered:
            lines.append(f"- {labels[key]}: " + "; ".join(rendered))
    block = "\n".join(lines).strip()
    if len(block) <= max_chars:
        return block
    return block[: max_chars - 20].rstrip() + "\n[brief truncated]"


def editorial_artifact_saturation(editorial_map: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return bucket counts plus whether prompt-prep buckets hit storage caps."""
    data = (
        dict(editorial_map)
        if _is_serialized_normalized_map(editorial_map)
        else _normalise_map(editorial_map)
    )
    out: dict[str, dict[str, Any]] = {}
    for key in ARTIFACT_KEYS:
        items = data.get(key) if isinstance(data.get(key), list) else []
        cap = _MAX_ITEMS_PER_BUCKET.get(key, 100)
        count = len(items)
        out[key] = {
            "count": count,
            "cap": cap,
            "saturated": count >= cap,
        }
    return out


def _normalise_map(value: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(value or {})
    data.setdefault("version", EDITORIAL_ARTIFACT_VERSION)
    for key, items in _v3_alias_buckets(data).items():
        existing = data.get(key) if isinstance(data.get(key), list) else []
        data[key] = [*(existing or []), *items]
    for key in ARTIFACT_KEYS:
        items = data.get(key) or []
        if not isinstance(items, list):
            items = []
        data[key] = _rank_and_cap(key, [dict(item) for item in items if isinstance(item, Mapping)])
    stats = data.get("source_stats")
    data["source_stats"] = dict(stats) if isinstance(stats, Mapping) else {}
    return data


def _is_serialized_normalized_map(value: Mapping[str, Any]) -> bool:
    """Return whether a persisted v3 map already satisfies the read contract."""
    return (
        str(value.get("version") or "") == EDITORIAL_ARTIFACT_VERSION
        and all(isinstance(value.get(key), list) for key in ARTIFACT_KEYS)
    )


def _v3_alias_buckets(data: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Project older v2 buckets into the explicit v3 editorial-map names."""
    aliases: dict[str, list[dict[str, Any]]] = {
        "entities": [],
        "voices": [],
        "chapters": [],
        "risks": [],
        "preserve_terms": [],
    }
    for item in data.get("characters_entities") or []:
        if isinstance(item, Mapping):
            aliases["entities"].append(dict(item))
    for item in data.get("narrative_voices") or []:
        if isinstance(item, Mapping):
            aliases["voices"].append(dict(item))
    for item in data.get("sections") or []:
        if isinstance(item, Mapping):
            aliases["chapters"].append(dict(item))
    for item in data.get("editorial_risks") or []:
        if isinstance(item, Mapping):
            aliases["risks"].append(dict(item))
    for item in data.get("do_not_translate") or []:
        if isinstance(item, Mapping):
            aliases["preserve_terms"].append(dict(item))
    return aliases


def _extract_sections(text: str, *, max_sections: int = 120) -> list[dict[str, Any]]:
    lines = [line.strip() for line in (text or "").splitlines()]
    sections: list[dict[str, Any]] = []
    for line in lines:
        title = re.sub(r"\s+", " ", line).strip(" -\t")
        if not _looks_like_section_title(title):
            continue
        _append_unique(
            sections,
            {
                "title": _short(title, 140),
                "kind": _section_kind(title),
                "source": "local",
                "confidence": 0.72 if _HEADING_MARKER_RE.search(title) else 0.58,
                "order": len(sections) + 1,
            },
            key_fields=("title",),
        )
        if len(sections) >= max_sections:
            break
    return sections


def _looks_like_section_title(line: str) -> bool:
    if not line or len(line) < 3 or len(line) > 180:
        return False
    if _DOT_LEADER_RE.search(line):
        return True
    if _HEADING_MARKER_RE.search(line):
        return True
    if line.endswith((".", ",", ";")):
        return False
    words = line.split()
    if len(words) > 14:
        return False
    letters = [ch for ch in line if ch.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(ch.isupper() for ch in letters) / max(1, len(letters))
    titlecase = sum(1 for word in words if word[:1].isupper()) / max(1, len(words))
    return upper_ratio > 0.62 or (len(words) <= 9 and titlecase > 0.55)


def _section_kind(title: str) -> str:
    folded = title.casefold()
    if "indice" in folded or "índice" in folded or "contents" in folded:
        return "toc"
    if "prologo" in folded or "prólogo" in folded or "preface" in folded:
        return "front_matter"
    if "chapter" in folded or "capitulo" in folded or "capítulo" in folded:
        return "chapter"
    if "part" in folded or "parte" in folded or "book" in folded or "libro" in folded:
        return "part"
    return "section"


def _entities_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in candidates:
        category = str(item.get("category") or "other").strip().lower()
        if category not in _ENTITY_CATEGORIES:
            continue
        source = _short(item.get("source"), 120)
        if not source:
            continue
        out.append(_candidate_to_artifact_item(item, label_field="name", label=source, item_type=category))
    return _rank_and_cap("characters_entities", _dedupe_items(out, ("name",)))


def _technical_terms_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in candidates:
        category = str(item.get("category") or "other").strip().lower()
        if category not in _TECH_CATEGORIES:
            continue
        source = _short(item.get("source"), 120)
        if not source:
            continue
        out.append(_candidate_to_artifact_item(item, label_field="term", label=source, item_type=category))
    return _rank_and_cap("technical_cultural_terms", _dedupe_items(out, ("term",)))


def _do_not_translate_from_candidates(
    candidates: list[dict[str, Any]],
    *,
    transform_mode: str = "",
    target_locale: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in candidates:
        if not item.get("keep_source"):
            continue
        source = _short(item.get("source"), 120)
        if not source:
            continue
        category = str(item.get("category") or "term")
        if _needs_canonical_review_before_preserve(
            source,
            category,
            transform_mode=transform_mode,
            target_locale=target_locale,
        ):
            continue
        out.append({
            "source": source,
            "reason": "Local preflight classified this as preserve-as-written.",
            "type": category,
            "occurrences": int(item.get("occurrences") or 0),
            "confidence": _float(item.get("confidence"), 0.0),
            "evidence": _first_context(item),
            "source_kind": "local",
        })
    return _rank_and_cap("do_not_translate", _dedupe_items(out, ("source",)))


def _needs_canonical_review_before_preserve(
    source: str,
    category: str,
    *,
    transform_mode: str = "",
    target_locale: str = "",
) -> bool:
    mode = str(transform_mode or "").strip().lower()
    if mode not in {"modernize", "modernizar", "contemporize", "faithful_current_spanish"}:
        return False
    if str(target_locale or "").strip().lower() not in {"es-mx", "spanish", "es"}:
        return False
    if category not in {"character", "location", "organization", "title"}:
        return False
    words = str(source or "").split()
    if len(words) != 1:
        return False
    if source.isupper():
        return False
    if any(ch in source for ch in "ÁÉÍÓÚÜÑáéíóúüñ"):
        return False
    return bool(re.search(r"[A-Za-z]", source or ""))


def _recurrent_quoted_phrases(text: str) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for match in _QUOTED_PHRASE_RE.finditer(text or ""):
        phrase = re.sub(r"\s+", " ", match.group(1)).strip()
        if 8 <= len(phrase) <= 90 and len(phrase.split()) <= 12:
            counts[phrase] += 1
    return [
        {
            "phrase": phrase,
            "occurrences": count,
            "reason": "Repeated quoted phrase; review whether it should be preserved or handled consistently.",
            "source_kind": "local",
            "confidence": min(0.88, 0.45 + count / 10),
        }
        for phrase, count in counts.most_common(40)
        if count >= 2
    ]


def _relationship_map(text: str, entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [str(item.get("name") or "").strip() for item in entities[:30] if item.get("name")]
    if len(names) < 2:
        return []
    counters: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}
    sentences = _SENTENCE_RE.split(re.sub(r"\s+", " ", text or " "))
    for sentence in sentences[:8000]:
        present = [name for name in names if _contains_term(sentence, name)]
        if len(present) < 2:
            continue
        for left, right in combinations(sorted(set(present), key=str.casefold), 2):
            key = (left, right)
            counters[key] += 1
            examples.setdefault(key, _short(sentence, 220))
    out: list[dict[str, Any]] = []
    for (left, right), count in counters.most_common(80):
        out.append({
            "participants": [left, right],
            "label": f"{left} <-> {right}",
            "evidence": examples.get((left, right), ""),
            "occurrences": count,
            "confidence": min(0.9, 0.35 + count / 8),
            "source_kind": "local",
        })
    return out


def _local_voice_signals(text: str) -> list[dict[str, Any]]:
    value = text or ""
    words = re.findall(r"\w+", value, flags=re.UNICODE)
    first_person = len(re.findall(r"\b(?:yo|nosotros|nosotras|me|mi|m[ií]o|nuestro|nuestra|I|we|my|our)\b", value, re.I))
    dialogue = len(re.findall(r"(?:^|\n)\s*[—-]\s*\S", value))
    citations = len(re.findall(r"[\"“”«»]", value))
    notes = len(_FOOTNOTE_RE.findall(value))
    out: list[dict[str, Any]] = []
    if first_person >= 8:
        out.append({
            "label": "first_person_or_testimonial_voice",
            "evidence": f"{first_person} first-person markers detected.",
            "confidence": min(0.9, 0.45 + first_person / max(30, len(words) / 120)),
            "source_kind": "local",
        })
    if dialogue >= 4:
        out.append({
            "label": "dialogue_or_speaker_shift_voice",
            "evidence": f"{dialogue} dialogue dash markers detected.",
            "confidence": min(0.9, 0.45 + dialogue / 40),
            "source_kind": "local",
        })
    if citations >= 20 or notes >= 8:
        out.append({
            "label": "documentary_or_academic_voice",
            "evidence": f"{citations} quote marks and {notes} note markers detected.",
            "confidence": min(0.9, 0.45 + (citations + notes) / 120),
            "source_kind": "local",
        })
    if not out and len(words) > 1000:
        out.append({
            "label": "continuous_expository_or_narrative_voice",
            "evidence": "Long-form prose with no strong local dialogue/testimonial signal.",
            "confidence": 0.45,
            "source_kind": "local",
        })
    return out


def _local_editorial_risks(
    text: str,
    candidates: list[dict[str, Any]],
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    value = text or ""
    risks: list[dict[str, Any]] = []
    if _MOJIBAKE_RE.search(value):
        risks.append(_risk("encoding_or_ocr_artifacts", "Encoding artifacts or replacement glyphs detected.", "high"))
    if _DOT_LEADER_RE.search(value):
        risks.append(_risk("toc_or_pagination_noise", "Dot leaders or flattened table-of-contents artifacts detected.", "medium"))
    if len(value) > 5000:
        paragraph_count = len([p for p in re.split(r"\n{2,}", value) if p.strip()])
        if paragraph_count <= max(2, len(value) // 18000):
            risks.append(_risk("weak_paragraph_structure", "The source has very few paragraph breaks for its length.", "medium"))
    if _FORMULA_RE.search(value):
        risks.append(_risk("formula_or_symbol_density", "Mathematical, code, or formula-like symbols detected.", "medium"))
    if "\t" in value or re.search(r"\|.+\|", value):
        risks.append(_risk("tables_or_columns", "Tabular or column-like text detected; preserve structure carefully.", "medium"))
    if _FOOTNOTE_RE.search(value):
        risks.append(_risk("footnotes_or_citations", "Footnote/citation markers detected; preserve references and placement.", "medium"))
    if not sections and len(value) > 20000:
        risks.append(_risk("missing_section_map", "No clear section headings detected locally in a long text.", "medium"))
    if len(candidates) > 300:
        risks.append(_risk("large_glossary_surface", "Many recurring candidates detected; glossary review may materially affect consistency.", "low"))
    return _dedupe_items(risks, ("code",))


def _normalise_llm_item(
    raw: Any,
    *,
    key: str,
    profile_id: str,
    chunk_index: int,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    item = dict(raw)
    label = _short(
        item.get("name")
        or item.get("source")
        or item.get("term")
        or item.get("phrase")
        or item.get("title")
        or item.get("label")
        or item.get("code"),
        140,
    )
    if key == "relationships":
        participants = _short_list(item.get("participants"), max_items=4, max_chars=80)
        if not participants and label:
            participants = [part.strip() for part in re.split(r"<->|→|->|/|,", label) if part.strip()][:4]
        if len(participants) < 2:
            return {}
        label = label or " <-> ".join(participants[:2])
        item["participants"] = participants
        item["label"] = label
    elif key == "sections":
        if not label:
            return {}
        item["title"] = label
    elif key == "characters_entities":
        if not label:
            return {}
        item["name"] = label
    elif key == "entities":
        if not label:
            return {}
        item["name"] = label
    elif key == "voices":
        if not label:
            return {}
        item["label"] = label
    elif key == "chapters":
        if not label:
            return {}
        item["title"] = label
    elif key == "translatable_terms":
        if not label:
            return {}
        item["source"] = label
    elif key == "preserve_terms":
        if not label:
            return {}
        item["source"] = label
    elif key == "canonical_names":
        if not label:
            return {}
        item["name"] = label
    elif key == "blockers":
        if not label:
            return {}
        item["source"] = label
    elif key == "technical_cultural_terms":
        if not label:
            return {}
        item["term"] = label
    elif key == "iconic_phrases":
        if not label:
            return {}
        item["phrase"] = label
    elif key == "do_not_translate":
        if not label:
            return {}
        item["source"] = label
    elif key == "narrative_voices":
        if not label:
            return {}
        item["label"] = label
    elif key == "editorial_risks":
        if not label:
            return {}
        item["code"] = _slug(label)
        item["label"] = label

    cleaned: dict[str, Any] = {
        "source_kind": "llm",
        "profile_id": profile_id,
        "chunk_index": chunk_index,
    }
    for field in (
        "name", "title", "label", "term", "phrase", "source", "code", "type",
        "kind", "role", "reason", "evidence", "recommended_handling",
        "modernization_guidance", "risk", "severity", "rationale", "target",
        "canonical", "status", "policy", "translation_policy",
        "injection_policy",
    ):
        value = _short(item.get(field), 320)
        if value:
            cleaned[field] = value
    for field in ("risks", "do_not_apply_if", "target_options"):
        values = _short_list(item.get(field), max_items=5, max_chars=160)
        if values:
            cleaned[field] = values
    if "participants" in item:
        cleaned["participants"] = _short_list(item.get("participants"), max_items=6, max_chars=90)
    cleaned["confidence"] = _float(item.get("confidence"), 0.55)
    return cleaned


def _candidate_to_artifact_item(
    item: Mapping[str, Any],
    *,
    label_field: str,
    label: str,
    item_type: str,
) -> dict[str, Any]:
    return {
        label_field: label,
        "type": item_type,
        "occurrences": int(item.get("occurrences") or 0),
        "confidence": _float(item.get("confidence"), 0.0),
        "evidence": _first_context(item),
        "source_kind": "local",
    }


def _risk(code: str, message: str, severity: str) -> dict[str, Any]:
    return {
        "code": code,
        "label": code.replace("_", " "),
        "severity": severity,
        "reason": message,
        "confidence": {"high": 0.9, "medium": 0.72, "low": 0.55}.get(severity, 0.55),
        "source_kind": "local",
    }


def _item_label(item: Mapping[str, Any], key: str) -> str:
    if key == "relationships":
        participants = item.get("participants") if isinstance(item.get("participants"), list) else []
        if participants:
            return " <-> ".join(str(part) for part in participants[:3])
    for field in ("name", "title", "label", "term", "phrase", "source", "code"):
        value = str(item.get(field) or "").strip()
        if value:
            return value
    return key


def _item_detail(item: Mapping[str, Any]) -> str:
    parts = []
    for field in (
        "target", "canonical", "policy", "translation_policy", "status",
        "type", "kind", "role", "severity", "reason", "recommended_handling",
        "evidence",
    ):
        value = _short(item.get(field), 160)
        if value:
            parts.append(value)
        if len(parts) >= 2:
            break
    return "; ".join(parts)


def _append_unique(
    items: list[dict[str, Any]],
    item: dict[str, Any],
    *,
    key_fields: tuple[str, ...],
) -> None:
    key = _dedupe_key(item, key_fields)
    if not key:
        return
    for index, existing in enumerate(items):
        if _dedupe_key(existing, key_fields) == key:
            if _score_item(item) > _score_item(existing):
                items[index] = {**existing, **item}
            return
    items.append(item)


def _dedupe_items(items: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in items:
        _append_unique(out, item, key_fields=key_fields)
    return out


def _dedupe_fields_for_key(key: str) -> tuple[str, ...]:
    return {
        "characters_entities": ("name",),
        "sections": ("title",),
        "narrative_voices": ("label",),
        "relationships": ("label",),
        "technical_cultural_terms": ("term",),
        "iconic_phrases": ("phrase",),
        "do_not_translate": ("source",),
        "editorial_risks": ("code", "label"),
        "entities": ("name", "source"),
        "voices": ("label",),
        "chapters": ("title",),
        "translatable_terms": ("source", "term"),
        "preserve_terms": ("source", "term"),
        "risks": ("code", "label"),
        "canonical_names": ("name", "source"),
        "blockers": ("source", "code"),
    }.get(key, ("label", "source"))


def _dedupe_key(item: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    for field in fields:
        value = item.get(field)
        if isinstance(value, list):
            value = " ".join(str(part) for part in value)
        text = str(value or "").strip().casefold()
        if text:
            return re.sub(r"\s+", " ", text)
    return ""


def _rank_and_cap(key: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = _dedupe_items(items, _dedupe_fields_for_key(key))
    items.sort(key=_score_item, reverse=True)
    return items[: _MAX_ITEMS_PER_BUCKET.get(key, 100)]


def _score_item(item: Mapping[str, Any]) -> float:
    occurrences = _float(item.get("occurrences"), 0.0)
    confidence = _float(item.get("confidence"), 0.0)
    severity = {"high": 5.0, "medium": 3.0, "low": 1.0}.get(str(item.get("severity") or "").lower(), 0.0)
    chunk_bonus = 0.2 if item.get("source_kind") == "llm" else 0.0
    return occurrences + confidence * 4.0 + severity + chunk_bonus


def _signal_bucket(items: list[dict[str, Any]], *, key: str, limit: int) -> dict[str, Any]:
    by_key = Counter(str(item.get(key) or item.get("type") or "unknown").strip().lower() or "unknown" for item in items)
    return {
        "total": len(items),
        "by_kind": dict(by_key.most_common()),
        "stored_items": min(len(items), int(limit)),
        "items_truncated": max(0, len(items) - int(limit)),
        "top_items": [
            _signal_item_summary(item)
            for item in sorted(items, key=_score_item, reverse=True)[: int(limit)]
        ],
    }


def _signal_item_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in (
        "source", "target", "term", "name", "title", "phrase", "type",
        "category", "status", "review_status", "translation_policy",
        "injection_policy", "confidence", "review_confidence", "occurrences",
    ):
        value = item.get(field)
        if value in (None, "", [], {}, 0, 0.0):
            continue
        if isinstance(value, str):
            out[field] = _short(value, 160)
        elif isinstance(value, (int, float, bool)):
            out[field] = value
        elif isinstance(value, list):
            out[field] = [_short(part, 100) for part in value[:6]]
    return out


def _signal_risk_flags(
    local: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    approved: list[dict[str, Any]],
    pending: list[dict[str, Any]],
) -> list[str]:
    flags: list[str] = []
    if len(local) > len(approved) + len(pending) * 2:
        flags.append("many_unclassified_local_candidates")
    same_target = [
        item for item in approved
        if str(item.get("source") or "").casefold() == str(item.get("target") or "").casefold()
        and str(item.get("source") or "").strip()
    ]
    translated = [
        item for item in approved
        if str(item.get("target") or "").strip()
        and str(item.get("source") or "").casefold() != str(item.get("target") or "").casefold()
    ]
    if len(same_target) > max(8, len(translated) * 2):
        flags.append("preserve_exact_dominates_translation_terms")
    if pending:
        flags.append("pending_suggestions_need_review")
    if any(str(item.get("review_status") or "").lower() == "pending_review" for item in reviewed):
        flags.append("reviewer_left_terms_pending")
    return flags


def _first_context(item: Mapping[str, Any]) -> str:
    contexts = item.get("contexts")
    if isinstance(contexts, list) and contexts:
        return _short(contexts[0], 220)
    return ""


def _contains_term(text: str, term: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text or "", re.I))


def _short(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        return ""
    return text[:max_chars].rstrip()


def _short_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, list):
        value = [value] if value else []
    out: list[str] = []
    for item in value:
        text = _short(item, max_chars)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_").lower()
    return slug[:80] or "editorial_risk"
