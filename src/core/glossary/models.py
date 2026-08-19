"""
Dataclasses for glossary entities.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class GlossaryTerm:
    """A single source -> target term entry."""
    source_term: str
    translated_term: str
    category: Optional[str] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "source": self.source_term,
            "target": self.translated_term,
            "category": self.category or "",
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GlossaryTerm":
        return cls(
            source_term=data.get("source") or data.get("source_term") or "",
            translated_term=data.get("target") or data.get("translated_term") or "",
            category=data.get("category") or None,
            id=data.get("id"),
        )


@dataclass
class Glossary:
    """A named collection of terms for a specific source/target language pair."""
    name: str
    source_language: str = ""
    target_language: str = ""
    terms: List[GlossaryTerm] = field(default_factory=list)
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def terms_dict(self) -> Dict[str, str]:
        """Returns complete {source_term: translated_term} mappings for prompts.

        Draft glossary rows may intentionally have a blank target while the user
        reviews auto-extracted technical terms. Those rows stay visible in the
        editor, but must not become ``source ->`` rules in LLM prompts.
        """
        return {
            t.source_term: t.translated_term
            for t in self.terms
            if (t.source_term or "").strip() and (t.translated_term or "").strip()
        }

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_lang": self.source_language,
            "target_lang": self.target_language,
            "terms": [t.to_dict() for t in self.terms],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Glossary":
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            source_language=data.get("source_lang") or data.get("source_language") or "",
            target_language=data.get("target_lang") or data.get("target_language") or "",
            terms=[GlossaryTerm.from_dict(t) for t in (data.get("terms") or [])],
        )


@dataclass
class GlossaryConfig:
    """Behavior knobs for the per-chunk glossary filter."""
    max_entries: int = 50
    case_sensitive: bool = True
    accent_insensitive: bool = False
    warn_on_cap: bool = True


@dataclass
class BulkReplaceResult:
    """Outcome of GlossaryStore.bulk_replace_terms.

    Reports how many rows were inserted and how many were skipped, so callers
    (especially the import endpoint) can explain to the user why N may be
    smaller than the number of rows in the source file.
    """
    inserted: int = 0
    skipped_empty: int = 0
    skipped_duplicate: int = 0
    total_input: int = 0

    def to_dict(self) -> Dict:
        return {
            "inserted": self.inserted,
            "skipped_empty": self.skipped_empty,
            "skipped_duplicate": self.skipped_duplicate,
            "total_input": self.total_input,
        }
