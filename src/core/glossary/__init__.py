"""
Glossary module for consistent translation of recurring terms.

Provides:
- models: Glossary, GlossaryTerm, GlossaryConfig dataclasses
- store: SQLite CRUD operations
- filter: chunk-aware glossary filtering (Latin word-boundary, CJK substring)
- injector: build the glossary block injected into the system prompt
- auto: token-free full-document glossary candidate extraction
- apply: propagate a reviewed glossary correction to editable output files
"""
from src.core.glossary.apply import (
    apply_term_correction_to_file,
    apply_term_correction_to_files,
    apply_term_correction_to_text,
)
from src.core.glossary.auto import AutoGlossaryCandidate, extract_glossary_candidates
from src.core.glossary.models import (
    BulkReplaceResult,
    Glossary,
    GlossaryConfig,
    GlossaryTerm,
)
from src.core.glossary.filter import filter_glossary, filter_glossary_for_purpose
from src.core.glossary.injector import build_glossary_block
from src.core.glossary.store import GlossaryStore
from src.core.glossary.ner import parse_ner_response, suggest_terms

__all__ = [
    "BulkReplaceResult",
    "Glossary",
    "GlossaryTerm",
    "GlossaryConfig",
    "GlossaryStore",
    "AutoGlossaryCandidate",
    "filter_glossary",
    "filter_glossary_for_purpose",
    "build_glossary_block",
    "parse_ner_response",
    "suggest_terms",
    "extract_glossary_candidates",
    "apply_term_correction_to_text",
    "apply_term_correction_to_file",
    "apply_term_correction_to_files",
]
