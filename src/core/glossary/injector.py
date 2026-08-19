"""
Build the glossary block to inject into the system prompt.

The block style mirrors the existing prompt voice in src/prompts/prompts.py
(numbered priorities, MANDATORY phrasing) so the LLM treats glossary entries
with the same weight as the rest of the instructions.
"""
from typing import Dict, Optional


def build_glossary_block(
    filtered_terms: Dict[str, str],
    target_language: str = "",
    term_metadata: Optional[Dict[str, Dict[str, str]]] = None,
    purpose: str = "translation",
) -> str:
    """
    Render the glossary block. Empty string if no terms.

    Args:
        filtered_terms: {source: target} of terms that match the current chunk.
        target_language: present for symmetry with the rest of the prompt API.
        term_metadata: optional {source: {category}} mapping. When a term
            has a category, it is rendered as a bracketed hint after the
            arrow so the LLM can disambiguate homonyms (e.g. a name vs a
            place sharing the same spelling).
        purpose: prompt context for the block. ``translation`` keeps the
            classic source-to-target wording, ``refinement`` asks the model to
            preserve established renderings, and ``transformation`` treats the
            right-hand side as a canonical same-language form.

    The block lives between the optional sections and the placeholder section
    in the system prompt — close enough to the input text that the model will
    not forget it, but not after the FINAL REMINDER so the output-language
    reminder stays last.
    """
    usable_terms = {
        source: target
        for source, target in (filtered_terms or {}).items()
        if (source or "").strip() and (target or "").strip()
    }

    if not usable_terms:
        return ""

    metadata = term_metadata or {}

    normalized_purpose = (purpose or "translation").strip().lower()
    if normalized_purpose in {"transform", "transformation", "modernize", "same-language"}:
        lines = [
            "# GLOSSARY - REQUIRED TERM TREATMENT",
            "",
            "MANDATORY: apply these term rules whenever the source form appears in this same-language transformation.",
            "Use the right-hand form as the canonical rendered form; it may be a modernization, spelling policy, name treatment, or fixed wording, not necessarily a cross-language translation.",
            "Do NOT paraphrase, spell differently, or invent alternatives for listed terms.",
            "For ordinary translated words, adapt gender, number, articles, and prepositions only when target-language grammar requires it; keep proper names and explicit spelling policies unchanged.",
            "Apply each rule consistently every time the term occurs.",
            "When several source forms are listed before the arrow (comma-separated), they are variants of the same entity or expression — align any of them with the canonical form on the right.",
            "Bracketed hints after the arrow (e.g. [character]) describe the entity type — use them only to disambiguate, not as part of the output.",
            "",
        ]
    elif normalized_purpose == "refinement":
        lines = [
            "# GLOSSARY - REQUIRED TERM STABILITY",
            "",
            "MANDATORY: keep these established renderings stable during refinement.",
            "If the draft already uses the right-hand form, preserve it unless grammar absolutely requires a minimal adjustment.",
            "Do NOT paraphrase, transliterate differently, or invent alternatives for listed terms.",
            "For ordinary translated words, adapt gender, number, articles, and prepositions only when target-language grammar requires it; keep proper names and explicit spelling policies unchanged.",
            "Apply each rule consistently every time the term occurs.",
            "When several source forms are listed before the arrow (comma-separated), they are variants of the same entity or expression — keep any of them aligned with the canonical form on the right.",
            "Bracketed hints after the arrow (e.g. [character]) describe the entity type — use them only to disambiguate, not as part of the output.",
            "",
        ]
    else:
        lines = [
            "# GLOSSARY - REQUIRED TRANSLATIONS",
            "",
            "MANDATORY: use these EXACT translations whenever the source term appears.",
            "Do NOT paraphrase, transliterate differently, or invent alternatives.",
            "For ordinary translated words, adapt gender, number, articles, and prepositions only when target-language grammar requires it; keep proper names and explicit spelling policies unchanged.",
            "Apply each rule consistently every time the term occurs.",
            "When several source forms are listed before the arrow (comma-separated), they are inflected variants of the same entity — translate any of them with the canonical target on the right.",
            "Bracketed hints after the arrow (e.g. [character]) describe the entity type — use them only to disambiguate, not as part of the translation.",
            "",
        ]

    for source, target in usable_terms.items():
        meta = metadata.get(source) or {}
        category = (meta.get("category") or "").strip()
        # Render alternatives (declined forms separated by '|' in storage) as
        # a comma-separated list so the LLM reads them as a natural set.
        display_source = ", ".join(
            a.strip() for a in source.split("|") if a.strip()
        ) or source
        if category:
            lines.append(f"  - {display_source} -> {target}  [{category}]")
        else:
            lines.append(f"  - {display_source} -> {target}")

    return "\n".join(lines) + "\n"
