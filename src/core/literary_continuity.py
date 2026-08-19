"""Token-safe literary continuity memory for long-form translation.

The module keeps a compact, local memory of a literary work while chunks are
translated. It does not call an LLM. The memory is intentionally small so long
books can preserve continuity without burning the prompt budget on every chunk.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Deque, Dict, Iterable, Mapping, Optional


_PLACEHOLDER_RE = re.compile(r"\[id\d+\]|\[\[\d+\]\]|\[\d+\]", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]{1,120}>")
_PARAGRAPH_RE = re.compile(r"\n{2,}")
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+")
_CHAPTER_RE = re.compile(
    r"^\s*(?:"
    r"(?:chapter|cap[ií]tulo|part|parte|book|libro|volume|volumen|canto)\s+"
    r"(?:[0-9]+|[ivxlcdm]+|[a-z]+)[\w\s,.'’:-]{0,100}|"
    r"(?:[ivxlcdm]{1,12}|[0-9]{1,4})[.)-]?\s+[A-ZÁÉÍÓÚÜÑ][\w\s,.'’:-]{2,100}"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SECTION_RE = _CHAPTER_RE
_DIALOGUE_MARK_RE = re.compile(r"[\"“”«»]|(?:^|\n)\s*[—-]\s*\S")
_DIALOGUE_VERB_RE = re.compile(
    r"\b(?:said|asked|replied|answered|whispered|shouted|murmured|"
    r"dijo|pregunt[oó]|respondi[oó]|susurr[oó]|grit[oó]|murmur[oó]|"
    r"dit|demanda|r[eé]pondit|chuchota|cria)\b",
    re.IGNORECASE,
)
_TECH_RE = re.compile(
    r"(?:```|`[^`\n]+`|https?://|\b(?:def|class|function|import|return)\b|"
    r"\b(?:BLEU|GPU|API|HTML|XML|JSON|CPU|HTTP)\b|"
    r"(?:[=<>+\-*/^_{}]|10\^|\[[0-9,\s;]{1,20}\]))"
)
_TITLE_WORDS = (
    "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Sir", "Lady", "Lord",
    "Madame", "Monsieur", "Señor", "Señora", "Don", "Doña", "Captain",
    "Capitan", "Capitán", "General", "King", "Queen", "Prince", "Princess",
    "Father", "Mother", "Brother", "Sister", "Master", "Mistress",
)
_TITLE_PATTERN = "|".join(re.escape(word) for word in sorted(_TITLE_WORDS, key=len, reverse=True))
_TITLE_TOKEN_RE = re.compile(
    rf"\b(?:{_TITLE_PATTERN})\.?\s+[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ'’-]+",
)
_NAME_RE = re.compile(
    r"\b[A-ZÁÉÍÓÚÜÑ][a-zÁÉÍÓÚÜÑáéíóúüñ'’-]{2,}"
    r"(?:\s+(?:de|del|da|do|dos|das|di|du|van|von|la|le|el|y|and|of|the|The)\s+"
    r"[A-ZÁÉÍÓÚÜÑ][a-zÁÉÍÓÚÜÑáéíóúüñ'’-]{2,}|"
    r"\s+[A-ZÁÉÍÓÚÜÑ][a-zÁÉÍÓÚÜÑáéíóúüñ'’-]{2,}){0,4}\b"
)
_FORMULA_RE = re.compile(r"(?:[A-Za-z]\w*(?:_[A-Za-z0-9{}]+)?\s*=|10\^-?\d+|[βαγελμσ]\w*)")

_NAME_STOPWORDS = {
    "abstract", "appendix", "chapter", "conclusion", "contents", "copyright",
    "bible", "data", "discussion", "english", "example", "figure", "figures",
    "french", "german", "introduction", "model", "models", "paper",
    "references", "results", "section", "spanish", "table", "tables",
    "training", "translator", "translation", "epub", "pdf", "docx",
    "capitulo", "capítulo", "parte", "libro", "documento", "volume",
    "volumen", "project", "gutenberg", "ebook", "license",
    # Frequent capitalized sentence starters that polluted the first audit.
    "after", "also", "although", "and", "because", "before", "besides",
    "but", "chapter", "for", "from", "god", "great god", "hark", "halloa", "hello", "however",
    "if", "in", "listen", "look", "mast-head", "meanwhile", "meantime",
    "nevertheless", "now", "once", "only", "or", "since", "so", "stand",
    "still", "than", "that", "the", "then", "there", "these", "this",
    "though", "thus", "toward", "towards", "until", "upon", "what", "what's", "when", "where",
    "while", "with", "yet",
}
_LEADING_ARTICLES = {"a", "an", "the", "el", "la", "los", "las", "le", "les"}
_LITERARY_CATEGORIES = {"character", "location", "organization", "item", "title", "other"}
_CATEGORY_WEIGHTS = {
    "character": 18.0,
    "location": 12.0,
    "organization": 11.0,
    "item": 9.0,
    "title": 7.0,
    "other": 4.0,
}
_TOKEN_COUNTER = None
_NAME_STOPWORDS.update(word.casefold() for word in _TITLE_WORDS)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _clean_text(text: str) -> str:
    text = _PLACEHOLDER_RE.sub(" ", text or "")
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _snippet(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", _clean_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text or "") if p.strip()]


def _count_tokens(text: str) -> int:
    """Count tokens with the project tokenizer, falling back to a rough estimate."""
    global _TOKEN_COUNTER
    if not text:
        return 0
    try:
        if _TOKEN_COUNTER is None:
            from src.core.chunking.token_chunker import TokenChunker
            _TOKEN_COUNTER = TokenChunker(max_tokens=1000)
        return _TOKEN_COUNTER.count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


def _first_interesting_sentence(text: str, names: Iterable[str] = ()) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    candidates = []
    for paragraph in _paragraphs(cleaned)[:4] or [cleaned]:
        candidates.extend(_SENTENCE_RE.split(paragraph))
    name_list = [n for n in names if n]
    for sentence in candidates:
        if any(name in sentence for name in name_list):
            return _snippet(sentence, 180)
    for sentence in candidates:
        if len(sentence.strip()) >= 60:
            return _snippet(sentence, 180)
    return _snippet(candidates[0] if candidates else cleaned, 180)


def _extract_section_title(text: str, fallback: str = "Documento") -> str:
    cleaned = _clean_text(text)
    for line in cleaned.splitlines()[:24]:
        line = line.strip()
        if line and _SECTION_RE.match(line):
            return _snippet(line, 100)
    return fallback or "Documento"


def _normalize_apostrophes(value: str) -> str:
    return (value or "").replace("’", "'").replace("‘", "'").replace("`", "'")


def _normalize_name(name: str) -> str:
    name = _normalize_apostrophes(name)
    name = re.sub(r"\s+", " ", name.strip(" \t\r\n.,;:!?()[]{}<>\"'"))
    return name


def _strip_possessive(value: str) -> str:
    value = _normalize_apostrophes(value)
    value = re.sub(r"(?i)'s\b", "", value)
    value = re.sub(r"(?i)s'\b", "s", value)
    return value


def _strip_title_prefix(value: str) -> str:
    return re.sub(rf"^(?:{_TITLE_PATTERN})\.?\s+", "", value, flags=re.IGNORECASE).strip()


def _display_entity(value: str) -> str:
    value = _normalize_name(_strip_possessive(value))
    words = value.split()
    while len(words) > 1 and words[0].casefold() in _LEADING_ARTICLES:
        words.pop(0)
    return " ".join(words)


def _entity_key(value: str) -> str:
    value = _display_entity(value)
    value = _strip_title_prefix(value)
    value = re.sub(r"[-–—]", " ", value)
    value = re.sub(r"[^\wÁÉÍÓÚÜÑáéíóúüñ\s']", "", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value.casefold()


def _is_probable_name(name: str) -> bool:
    if not name:
        return False
    display = _display_entity(name)
    if display.endswith("-"):
        return False
    folded = _entity_key(name)
    if not folded or folded in _NAME_STOPWORDS:
        return False
    words = display.split()
    if not words or len(words) > 6:
        return False
    for idx, word in enumerate(words):
        folded_word = word.casefold().rstrip(".")
        if folded_word in _NAME_STOPWORDS:
            is_title_prefix = idx == 0 and len(words) > 1 and folded_word in {
                title.casefold() for title in _TITLE_WORDS
            }
            if not is_title_prefix:
                return False
    if len(words) == 1:
        # Single capitalized words are noisy. Keep likely names only when they
        # are not common section/document words and are long enough.
        return len(words[0]) >= 5 and folded not in _NAME_STOPWORDS
    return True


def _display_priority(value: str) -> tuple[int, int]:
    words = value.split()
    has_title = 1 if words and re.match(rf"^(?:{_TITLE_PATTERN})\.?$", words[0], re.IGNORECASE) else 0
    return (has_title, len(value))


def _glossary_entries(prompt_options: Optional[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    if not prompt_options:
        return {}
    terms = prompt_options.get("glossary_terms") or {}
    metadata = prompt_options.get("glossary_term_metadata") or {}
    entries: dict[str, dict[str, str]] = {}
    for source, target in terms.items():
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            continue
        meta = metadata.get(source) or {}
        category = str(meta.get("category") or "other").strip().lower()
        if category and category not in _LITERARY_CATEGORIES:
            # Technical terms remain handled by the glossary prompt itself.
            continue
        for alias in [part.strip() for part in source.split("|") if part.strip()] or [source]:
            key = _entity_key(alias)
            if key and key not in _NAME_STOPWORDS:
                entries[key] = {
                    "source": alias,
                    "target": target,
                    "category": category or "other",
                }
    return entries


@dataclass(frozen=True)
class TextProfile:
    kind: str
    confidence: float
    signals: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_literature(self) -> bool:
        return self.kind == "literature"


@dataclass
class ContinuityEntity:
    canonical_source: str
    category: str = "character"
    target: str = ""
    aliases: set[str] = field(default_factory=set)
    occurrences: int = 0
    first_chunk: int = 0
    last_chunk: int = 0
    sections: Counter[str] = field(default_factory=Counter)
    source_forms: Counter[str] = field(default_factory=Counter)
    confidence: float = 0.35
    salience: float = 0.0
    glossary_confirmed: bool = False
    false_positive: bool = False

    @property
    def name(self) -> str:
        return self.canonical_source

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_source": self.canonical_source,
            "category": self.category,
            "target": self.target,
            "aliases": sorted(self.aliases),
            "occurrences": self.occurrences,
            "first_chunk": self.first_chunk,
            "last_chunk": self.last_chunk,
            "sections": dict(self.sections),
            "source_forms": dict(self.source_forms),
            "confidence": self.confidence,
            "salience": self.salience,
            "glossary_confirmed": self.glossary_confirmed,
            "false_positive": self.false_positive,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContinuityEntity":
        entity = cls(
            canonical_source=str(data.get("canonical_source") or data.get("name") or ""),
            category=str(data.get("category") or "character"),
            target=str(data.get("target") or ""),
            aliases=set(data.get("aliases") or []),
            occurrences=int(data.get("occurrences") or 0),
            first_chunk=int(data.get("first_chunk") or 0),
            last_chunk=int(data.get("last_chunk") or 0),
            confidence=float(data.get("confidence") or 0.35),
            salience=float(data.get("salience") or 0.0),
            glossary_confirmed=bool(data.get("glossary_confirmed")),
            false_positive=bool(data.get("false_positive")),
        )
        entity.sections = Counter(data.get("sections") or {})
        entity.source_forms = Counter(data.get("source_forms") or {})
        if entity.canonical_source:
            entity.aliases.add(entity.canonical_source)
        return entity


@dataclass
class ContinuityWarning:
    chunk_index: int
    section: str
    code: str
    message: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_index": self.chunk_index,
            "section": self.section,
            "code": self.code,
            "message": self.message,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ContinuityWarning":
        return cls(
            chunk_index=int(data.get("chunk_index") or 0),
            section=str(data.get("section") or ""),
            code=str(data.get("code") or ""),
            message=str(data.get("message") or ""),
            detail=str(data.get("detail") or ""),
        )


class LiteraryContinuityState:
    """Small rolling memory for a literary translation job."""

    SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        source_language: str = "",
        target_language: str = "",
        text_profile: Optional[TextProfile] = None,
        max_prompt_chars: int = 1800,
        max_prompt_tokens: int = 220,
        max_entities: int = 600,
    ):
        self.source_language = source_language
        self.target_language = target_language
        self.text_profile = text_profile or TextProfile("literature", 1.0, {})
        self.max_prompt_chars = max(600, int(max_prompt_chars or 1800))
        self.max_prompt_tokens = max(120, min(int(max_prompt_tokens or 220), 800))
        self.max_entities = max(120, min(int(max_entities or 600), 2000))
        self.chunk_index = 0
        self.current_section = "Documento"
        self.entities: dict[str, ContinuityEntity] = {}
        self.recent_events: Deque[tuple[int, str, str]] = deque(maxlen=10)
        self.section_events: dict[str, list[str]] = defaultdict(list)
        self.warnings: list[ContinuityWarning] = []
        self.formula_chunks: Counter[str] = Counter()
        self.reported_sections: Counter[str] = Counter()
        self.last_render_stats: dict[str, int] = {
            "tokens": 0,
            "omitted_lines": 0,
            "entity_lines": 0,
            "event_lines": 0,
        }

    @property
    def characters(self) -> dict[str, ContinuityEntity]:
        """Compatibility view for older reports/audits."""
        return {
            entity.canonical_source: entity
            for entity in self.entities.values()
            if not entity.false_positive
        }

    def ingest_glossary(self, prompt_options: Optional[Mapping[str, Any]]) -> None:
        for key, entry in _glossary_entries(prompt_options).items():
            display = _display_entity(entry["source"])
            if not display:
                continue
            entity = self.entities.get(key)
            if entity is None:
                entity = ContinuityEntity(
                    canonical_source=display,
                    category=entry.get("category") or "other",
                    target=entry.get("target") or "",
                    first_chunk=max(1, self.chunk_index),
                    last_chunk=max(1, self.chunk_index),
                    confidence=0.82,
                    salience=18.0,
                    glossary_confirmed=True,
                )
                entity.aliases.add(display)
                self.entities[key] = entity
            else:
                entity.target = entry.get("target") or entity.target
                entity.category = entry.get("category") or entity.category
                entity.glossary_confirmed = True
                entity.confidence = max(entity.confidence, 0.82)
                entity.salience += 6.0
                entity.aliases.add(display)
        self._prune_entities()

    def observe(
        self,
        source_text: str,
        translated_text: str = "",
        *,
        section: str = "",
        phase: str = "translation",
    ) -> None:
        """Update memory after a chunk has been translated or refined."""
        self.chunk_index += 1
        section = section or _extract_section_title(source_text, self.current_section)
        if section and section != "Documento":
            self.current_section = section
        else:
            section = self.current_section
        self.reported_sections[section] += 1

        names = extract_literary_names(source_text)
        entity_names_for_event: list[str] = []
        for name in names:
            entity = self._observe_entity(name, source_text, section)
            if entity is not None:
                entity_names_for_event.append(entity.canonical_source)

        if translated_text:
            self._check_name_presence(entity_names_for_event, translated_text, section)

        event_text = _first_interesting_sentence(translated_text or source_text, entity_names_for_event)
        if event_text:
            self.recent_events.append((self.chunk_index, section, event_text))
            if len(self.section_events[section]) < 8:
                self.section_events[section].append(event_text)

        if _FORMULA_RE.search(source_text):
            self.formula_chunks[section] += 1

        self._prune_entities()

    def _observe_entity(self, raw_name: str, source_text: str, section: str) -> Optional[ContinuityEntity]:
        key = _entity_key(raw_name)
        display = _display_entity(raw_name)
        if not key or not display or not _is_probable_name(display):
            return None

        occurrence_count = max(1, source_text.count(raw_name), source_text.count(display))
        entity = self.entities.get(key)
        if entity is None:
            category = "character"
            entity = ContinuityEntity(
                canonical_source=display,
                category=category,
                occurrences=0,
                first_chunk=self.chunk_index,
                last_chunk=self.chunk_index,
                confidence=0.42,
                salience=_CATEGORY_WEIGHTS.get(category, 4.0),
            )
            self.entities[key] = entity

        if _display_priority(display) > _display_priority(entity.canonical_source):
            entity.aliases.add(entity.canonical_source)
            entity.canonical_source = display

        entity.aliases.add(display)
        stripped = _strip_title_prefix(display)
        if stripped and stripped != display:
            entity.aliases.add(stripped)
        entity.source_forms[raw_name] += occurrence_count
        entity.occurrences += occurrence_count
        entity.last_chunk = self.chunk_index
        entity.sections[section] += occurrence_count
        entity.confidence = _clamp(entity.confidence + min(0.08, occurrence_count * 0.015), 0.2, 0.98)
        recency_boost = 8.0
        entity.salience += occurrence_count * 1.7 + recency_boost
        return entity

    def _prune_entities(self) -> None:
        if len(self.entities) <= self.max_entities:
            return
        ranked = sorted(
            self.entities.items(),
            key=lambda item: (
                item[1].glossary_confirmed,
                self._entity_rank(item[1], set(), self.current_section),
                item[1].last_chunk,
            ),
            reverse=True,
        )
        self.entities = dict(ranked[: self.max_entities])

    def build_prompt_block(self, current_text: str, *, section: str = "") -> str:
        """Render a compact dynamic block for the next LLM prompt."""
        names_in_chunk = {_entity_key(name) for name in extract_literary_names(current_text)}
        names_in_chunk.discard("")
        section = section or _extract_section_title(current_text, self.current_section)
        if section and section != "Documento":
            self.current_section = section
        else:
            section = self.current_section

        base_lines = [
            "# LITERARY CONTINUITY MEMORY",
            f"- Current chapter/section: {section}",
            "- Use only for continuity; do not add facts absent from the current source chunk.",
        ]
        entity_lines = self._ranked_entity_lines(names_in_chunk, section)
        event_lines = self._recent_event_lines(section)

        candidate_groups: list[tuple[str, list[str]]] = []
        if entity_lines:
            candidate_groups.append(("Entities to keep consistent:", entity_lines))
        if event_lines:
            candidate_groups.append(("Recent continuity:", event_lines))
        candidate_groups.append((
            "Rules:",
            [
                "- Preserve names, relationships, pronouns, titles, recurring objects, narrator register, and ambiguity.",
                "- Do not summarize, skip, add backstory, or resolve ambiguity that the source leaves open.",
            ],
        ))

        lines = list(base_lines)
        omitted = 0
        accepted_entity_lines = 0
        accepted_event_lines = 0
        for heading, group_lines in candidate_groups:
            group_started = False
            for line in group_lines:
                proposed = lines + (["", heading] if not group_started else []) + [line]
                if _count_tokens("\n".join(proposed)) <= self.max_prompt_tokens:
                    if not group_started:
                        lines.extend(["", heading])
                        group_started = True
                    lines.append(line)
                    if heading.startswith("Entities"):
                        accepted_entity_lines += 1
                    elif heading.startswith("Recent"):
                        accepted_event_lines += 1
                else:
                    omitted += 1

        rendered = "\n".join(lines).strip() + "\n"
        # Compatibility with old char budget: if someone configured a very small
        # char budget, rebuild with fewer optional lines instead of hard-cutting.
        while len(rendered) > self.max_prompt_chars and len(lines) > len(base_lines):
            removed = lines.pop(-1)
            if removed.startswith("- "):
                omitted += 1
            while lines and lines[-1] in {"Entities to keep consistent:", "Recent continuity:", "Rules:", ""}:
                lines.pop(-1)
            rendered = "\n".join(lines).strip() + "\n"

        self.last_render_stats = {
            "tokens": _count_tokens(rendered),
            "omitted_lines": omitted,
            "entity_lines": accepted_entity_lines,
            "event_lines": accepted_event_lines,
        }
        return rendered

    def _ranked_entity_lines(self, names_in_chunk: set[str], section: str) -> list[str]:
        ranked = sorted(
            (e for e in self.entities.values() if not e.false_positive),
            key=lambda entity: self._entity_rank(entity, names_in_chunk, section),
            reverse=True,
        )
        lines: list[str] = []
        for entity in ranked[:28]:
            if entity.occurrences <= 0 and not entity.glossary_confirmed:
                continue
            aliases = sorted(a for a in entity.aliases if a and a != entity.canonical_source)
            alias_text = f"; aliases: {', '.join(aliases[:3])}" if aliases else ""
            target_text = f"; glossary: {entity.target}" if entity.target else ""
            where = "current chunk" if _entity_key(entity.canonical_source) in names_in_chunk else f"last seen chunk {entity.last_chunk}"
            category = entity.category or "entity"
            lines.append(
                f"- {entity.canonical_source} [{category}]: keep consistent ({where}{alias_text}{target_text})"
            )
        return lines

    def _entity_rank(self, entity: ContinuityEntity, names_in_chunk: set[str], section: str) -> float:
        key = _entity_key(entity.canonical_source)
        in_current = key in names_in_chunk or any(_entity_key(alias) in names_in_chunk for alias in entity.aliases)
        recency_distance = max(0, self.chunk_index - entity.last_chunk)
        score = 0.0
        score += 90.0 if in_current else 0.0
        score += _CATEGORY_WEIGHTS.get(entity.category, 4.0)
        score += min(entity.occurrences, 40) * 0.75
        score += max(0.0, 18.0 - recency_distance * 1.8)
        score += entity.confidence * 12.0
        score += 25.0 if entity.glossary_confirmed else 0.0
        score += 8.0 if section and entity.sections.get(section) else 0.0
        score += min(entity.salience, 120.0) * 0.08
        return score

    def _recent_event_lines(self, section: str) -> list[str]:
        recent = list(self.recent_events)[-7:]
        lines: list[str] = []
        for chunk_idx, event_section, event in recent:
            prefix = f"chunk {chunk_idx}"
            if event_section and event_section != section:
                prefix += f", {event_section}"
            lines.append(f"- {prefix}: {_snippet(event, 160)}")
        return lines

    def _check_name_presence(self, names: list[str], translated_text: str, section: str) -> None:
        if not names:
            return
        translated_fold = translated_text.casefold()
        for name in names[:12]:
            parts = [p for p in re.split(r"\s+", _strip_title_prefix(name)) if len(p) >= 3]
            if not parts:
                continue
            # This is intentionally soft: it records a doubt for the report,
            # not a rejection. Some languages legitimately adapt names.
            if not any(part.casefold() in translated_fold for part in parts):
                self.warnings.append(ContinuityWarning(
                    chunk_index=self.chunk_index,
                    section=section,
                    code="possible_name_drift",
                    message="A source name/entity may have changed or disappeared in the translation",
                    detail=name,
                ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "text_profile": {
                "kind": self.text_profile.kind,
                "confidence": self.text_profile.confidence,
                "signals": dict(self.text_profile.signals),
            },
            "max_prompt_chars": self.max_prompt_chars,
            "max_prompt_tokens": self.max_prompt_tokens,
            "max_entities": self.max_entities,
            "chunk_index": self.chunk_index,
            "current_section": self.current_section,
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "recent_events": [list(item) for item in self.recent_events],
            "section_events": dict(self.section_events),
            "warnings": [warning.to_dict() for warning in self.warnings],
            "formula_chunks": dict(self.formula_chunks),
            "reported_sections": dict(self.reported_sections),
            "last_render_stats": dict(self.last_render_stats),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LiteraryContinuityState":
        profile_data = data.get("text_profile") or {}
        state = cls(
            source_language=str(data.get("source_language") or ""),
            target_language=str(data.get("target_language") or ""),
            text_profile=TextProfile(
                str(profile_data.get("kind") or "literature"),
                float(profile_data.get("confidence") or 1.0),
                profile_data.get("signals") or {},
            ),
            max_prompt_chars=int(data.get("max_prompt_chars") or 1800),
            max_prompt_tokens=int(data.get("max_prompt_tokens") or 220),
            max_entities=int(data.get("max_entities") or 600),
        )
        state.chunk_index = int(data.get("chunk_index") or 0)
        state.current_section = str(data.get("current_section") or "Documento")
        state.entities = {}
        for entity_data in data.get("entities") or []:
            entity = ContinuityEntity.from_dict(entity_data)
            key = _entity_key(entity.canonical_source)
            if key:
                state.entities[key] = entity
        state.recent_events = deque(
            (
                (int(item[0]), str(item[1]), str(item[2]))
                for item in (data.get("recent_events") or [])
                if isinstance(item, (list, tuple)) and len(item) >= 3
            ),
            maxlen=10,
        )
        state.section_events = defaultdict(list, {
            str(key): list(value)
            for key, value in (data.get("section_events") or {}).items()
        })
        state.warnings = [
            ContinuityWarning.from_dict(item)
            for item in (data.get("warnings") or [])
        ]
        state.formula_chunks = Counter(data.get("formula_chunks") or {})
        state.reported_sections = Counter(data.get("reported_sections") or {})
        state.last_render_stats = dict(data.get("last_render_stats") or state.last_render_stats)
        return state

    def to_markdown(self) -> str:
        active_entities = [e for e in self.entities.values() if not e.false_positive]
        lines = [
            "# Reporte de continuidad literaria",
            "",
            f"- Idioma origen: {self.source_language or 'N/D'}",
            f"- Idioma destino: {self.target_language or 'N/D'}",
            f"- Tipo detectado: {self.text_profile.kind} ({self.text_profile.confidence:.2f})",
            f"- Fragmentos observados: {self.chunk_index}",
            f"- Entidades/nombres rastreados: {len(active_entities)}",
            f"- Dudas de continuidad: {len(self.warnings)}",
            f"- Presupuesto memoria: {self.max_prompt_tokens} tokens",
            "",
            "Este reporte se genera con heuristicas locales; no usa llamadas extra al LLM.",
            "",
        ]

        if active_entities:
            lines.extend(["## Nombres y entidades principales", ""])
            for entity in sorted(active_entities, key=lambda e: (-self._entity_rank(e, set(), self.current_section), e.canonical_source))[:80]:
                sections = ", ".join(section for section, _ in entity.sections.most_common(3))
                aliases = sorted(a for a in entity.aliases if a and a != entity.canonical_source)
                alias_text = f"; alias: {', '.join(aliases[:4])}" if aliases else ""
                target_text = f"; glosario: {entity.target}" if entity.target else ""
                lines.append(
                    f"- {entity.canonical_source}: {entity.occurrences} apariciones; "
                    f"chunks {entity.first_chunk}-{entity.last_chunk}; {sections or 'Documento'}"
                    f"{alias_text}{target_text}"
                )
            lines.append("")

        if self.section_events:
            lines.extend(["## Memoria por capitulo/seccion", ""])
            for section, events in self.section_events.items():
                lines.extend([f"### {section}", ""])
                for event in events[:8]:
                    lines.append(f"- {event}")
                if self.formula_chunks.get(section):
                    lines.append(f"- Nota: {self.formula_chunks[section]} fragmento(s) con formulas o notacion tecnica.")
                lines.append("")

        if self.warnings:
            lines.extend(["## Dudas e inconsistencias posibles", ""])
            for warning in self.warnings[:120]:
                detail = f": {warning.detail}" if warning.detail else ""
                lines.append(
                    f"- Chunk {warning.chunk_index} ({warning.section}) [{warning.code}] "
                    f"{warning.message}{detail}"
                )
            lines.append("")

        if not active_entities and not self.section_events and not self.warnings:
            lines.extend(["Sin memoria suficiente todavia.", ""])

        return "\n".join(lines).rstrip() + "\n"


def detect_text_profile(text: str) -> TextProfile:
    """Classify a sample locally. The first supported specialty is literature."""
    sample = _clean_text(text)[:20000]
    if not sample:
        return TextProfile("general", 0.0, {})

    paragraphs = _paragraphs(sample)
    dialogue_marks = len(_DIALOGUE_MARK_RE.findall(sample))
    dialogue_verbs = len(_DIALOGUE_VERB_RE.findall(sample))
    chapters = len(_CHAPTER_RE.findall(sample))
    named = len(extract_literary_names(sample, max_names=80))
    prose_paragraphs = sum(1 for p in paragraphs if len(p) >= 120 and len(_TECH_RE.findall(p)) <= 2)
    technical = len(_TECH_RE.findall(sample))

    signals = {
        "dialogue_marks": dialogue_marks,
        "dialogue_verbs": dialogue_verbs,
        "chapter_markers": chapters,
        "named_entities": named,
        "prose_paragraphs": prose_paragraphs,
        "technical_markers": technical,
    }

    score = 0.0
    score += min(dialogue_marks / 12.0, 0.25)
    score += min(dialogue_verbs / 8.0, 0.20)
    score += min(chapters / 2.0, 0.18)
    score += min(named / 12.0, 0.20)
    score += min(prose_paragraphs / 8.0, 0.22)
    score -= min(technical / 40.0, 0.35)
    confidence = _clamp(score)

    if confidence >= 0.45:
        kind = "literature"
    elif technical >= max(8, prose_paragraphs * 3):
        kind = "technical"
    else:
        kind = "general"
    return TextProfile(kind, confidence, signals)


def extract_literary_names(text: str, *, max_names: int = 24) -> list[str]:
    """Extract likely literary names/entities with local, conservative regexes."""
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    counts: Counter[str] = Counter()
    for match in _TITLE_TOKEN_RE.finditer(cleaned):
        name = _display_entity(match.group(0))
        if _is_probable_name(name):
            counts[name] += 3
    for match in _NAME_RE.finditer(cleaned):
        name = _display_entity(match.group(0))
        if not _is_probable_name(name):
            continue
        counts[name] += 1

    # Return display forms, not canonical keys. Observation handles alias merge.
    return [
        name for name, _count in counts.most_common(max_names)
        if name and _entity_key(name) not in _NAME_STOPWORDS
    ]


def should_enable_literary_continuity(prompt_options: Optional[Mapping[str, Any]], sample_text: str) -> bool:
    if not prompt_options or not prompt_options.get("literary_continuity"):
        return False
    text_type = str(prompt_options.get("text_type") or "auto").lower()
    if text_type == "general":
        return False
    if text_type == "literature":
        return True
    profile = detect_text_profile(sample_text)
    return profile.kind == "literature" and profile.confidence >= 0.45


def get_literary_continuity_state(
    *,
    prompt_options: Optional[Mapping[str, Any]],
    runtime_state: Optional[dict],
    sample_text: str,
    source_language: str = "",
    target_language: str = "",
    log_callback=None,
) -> Optional[LiteraryContinuityState]:
    """Get or initialize per-job literary continuity state."""
    if runtime_state is None:
        return None
    if not prompt_options or not prompt_options.get("literary_continuity"):
        return None

    text_type = str(prompt_options.get("text_type") or "auto").lower()
    profile = detect_text_profile(sample_text)
    enabled = text_type == "literature" or (
        text_type == "auto" and profile.kind == "literature" and profile.confidence >= 0.45
    )
    if text_type == "general":
        enabled = False
    if not enabled:
        if log_callback and not runtime_state.get("literary_continuity_skip_logged"):
            runtime_state["literary_continuity_skip_logged"] = True
            log_callback(
                "literary_continuity_skipped",
                f"📚 Literary continuity skipped: detected {profile.kind} ({profile.confidence:.2f}).",
            )
        return None

    state = runtime_state.get("literary_continuity_state")
    if isinstance(state, Mapping):
        state = LiteraryContinuityState.from_dict(state)
        runtime_state["literary_continuity_state"] = state

    if state is None:
        max_chars = prompt_options.get("continuity_max_prompt_chars", 1800)
        max_tokens = prompt_options.get("continuity_max_prompt_tokens", 220)
        max_entities = prompt_options.get("continuity_max_entities", 600)
        state = LiteraryContinuityState(
            source_language=source_language,
            target_language=target_language,
            text_profile=TextProfile("literature", max(profile.confidence, 0.75), profile.signals),
            max_prompt_chars=max_chars,
            max_prompt_tokens=max_tokens,
            max_entities=max_entities,
        )
        runtime_state["literary_continuity_state"] = state
        if log_callback:
            forced = "forced" if text_type == "literature" else "auto"
            log_callback(
                "literary_continuity_enabled",
                f"📚 Literary continuity enabled ({forced}, confidence {state.text_profile.confidence:.2f}, budget {state.max_prompt_tokens} tokens).",
            )

    if isinstance(state, LiteraryContinuityState):
        state.ingest_glossary(prompt_options)
        return state
    return None


def build_literary_continuity_block(
    *,
    prompt_options: Optional[Mapping[str, Any]],
    runtime_state: Optional[dict],
    current_text: str,
    source_language: str = "",
    target_language: str = "",
    section: str = "",
    log_callback=None,
) -> str:
    state = get_literary_continuity_state(
        prompt_options=prompt_options,
        runtime_state=runtime_state,
        sample_text=current_text,
        source_language=source_language,
        target_language=target_language,
        log_callback=log_callback,
    )
    if state is None:
        return ""
    return state.build_prompt_block(current_text, section=section)


def observe_literary_continuity(
    *,
    runtime_state: Optional[dict],
    source_text: str,
    translated_text: str = "",
    section: str = "",
    phase: str = "translation",
) -> None:
    if not runtime_state:
        return
    state = runtime_state.get("literary_continuity_state")
    if isinstance(state, Mapping):
        state = LiteraryContinuityState.from_dict(state)
        runtime_state["literary_continuity_state"] = state
    if isinstance(state, LiteraryContinuityState):
        state.observe(source_text, translated_text, section=section, phase=phase)


def export_literary_continuity_state(runtime_state: Optional[dict]) -> Optional[dict[str, Any]]:
    if not runtime_state:
        return None
    state = runtime_state.get("literary_continuity_state")
    if isinstance(state, LiteraryContinuityState):
        return state.to_dict()
    if isinstance(state, Mapping):
        return dict(state)
    return None


def import_literary_continuity_state(runtime_state: Optional[dict], state_data: Optional[Mapping[str, Any]]) -> None:
    if runtime_state is None or not state_data:
        return
    try:
        runtime_state["literary_continuity_state"] = LiteraryContinuityState.from_dict(state_data)
    except Exception:
        # Resume should not fail because an optional continuity cache is stale.
        runtime_state.pop("literary_continuity_state", None)


def literary_continuity_report_path(output_filepath: str | Path) -> Path:
    path = Path(output_filepath)
    return path.with_name(f"{path.stem} - reporte continuidad literaria.md")


def write_literary_continuity_report(
    output_filepath: str | Path,
    runtime_state: Optional[dict],
) -> Optional[Path]:
    if not runtime_state:
        return None
    state = runtime_state.get("literary_continuity_state")
    if isinstance(state, Mapping):
        state = LiteraryContinuityState.from_dict(state)
        runtime_state["literary_continuity_state"] = state
    if not isinstance(state, LiteraryContinuityState) or state.chunk_index == 0:
        return None
    report_path = literary_continuity_report_path(output_filepath)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(state.to_markdown(), encoding="utf-8")
    return report_path
