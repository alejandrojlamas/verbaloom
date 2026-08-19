"""Prompt rendering helpers for book profiles."""

from __future__ import annotations

from functools import lru_cache
import re
from typing import Optional

from src.core.document_structure import DocumentBlockClassifier
from src.core.editorial_knowledge_compiler import (
    compile_profile_prompt_context,
    match_profile_entry_in_text,
)
from src.core.glossary.filter import filter_glossary, filter_glossary_for_purpose
from src.core.glossary.models import GlossaryConfig
from src.utils.proper_names import restore_symbol_bearing_names

from .artifacts import compact_editorial_brief_for_prompt, editorial_artifact_counts
from .detectors import fold_profile_match_text
from .knowledge import build_profile_knowledge_base
from .loader import BookProfileError, load_book_profile
from .models import BookProfile, ProfileGlossaryEntry
from .term_review import (
    suspicious_preserve_entry,
    weakly_supported_generated_preserve_entry,
)

_CANONICAL_CORRECTION_TYPES = {
    "canonical_proper_noun",
    "proper_noun",
    "character",
    "location",
    "place",
    "organization",
    "orthographic_variant",
}
_PROTECTED_PHRASE_TYPES = _CANONICAL_CORRECTION_TYPES | {
    "title",
    "work_title",
}
_EPUB_PLACEHOLDER_RE = re.compile(r"\[id\d+\]", re.IGNORECASE)
_PUBLISHED_TITLE_CONNECTORS = {
    "a", "about", "an", "and", "as", "at", "between", "but", "by", "da", "das", "de", "del",
    "der", "des", "di", "do", "dos", "du", "e", "el", "en", "et",
    "for", "from", "in", "into", "la", "las", "le", "les", "los", "of", "on",
    "or", "over", "para", "por", "the", "through", "to", "un", "una", "under",
    "une", "und", "versus", "via", "von", "with", "without", "y",
}
_GENERIC_PROFILE_NOTES = {
    "looks like a recurring named entity for this book profile.",
    "looks like a recurring named entity for this profile.",
}
_PRESERVE_FRAGMENT_START_WORDS = {
    "ah",
    "but",
    "god",
    "no",
    "now",
    "oh",
    "so",
    "well",
    "why",
    "yes",
}
_PRESERVE_FRAGMENT_ANY_WORDS = {
    "and",
    "but",
    "or",
}
_PRESERVE_FRAGMENT_TRAILING_WORDS = {
    "and",
    "but",
    "or",
    "so",
}
_PROFILE_ALIAS_STOP_WORDS = {
    "a",
    "after",
    "an",
    "and",
    "but",
    "city",
    "america",
    "american",
    "academy",
    "agency",
    "association",
    "bank",
    "books",
    "center",
    "centre",
    "college",
    "company",
    "corporation",
    "de",
    "del",
    "department",
    "dr",
    "el",
    "england",
    "european",
    "french",
    "german",
    "god",
    "group",
    "he",
    "hospital",
    "i",
    "india",
    "institute",
    "institution",
    "lab",
    "laboratory",
    "it",
    "italian",
    "la",
    "las",
    "le",
    "les",
    "los",
    "m",
    "mexican",
    "mexico",
    "mr",
    "mrs",
    "museum",
    "network",
    "new",
    "no",
    "now",
    "of",
    "oh",
    "or",
    "press",
    "project",
    "publishing",
    "research",
    "review",
    "school",
    "señor",
    "senor",
    "she",
    "so",
    "spain",
    "spanish",
    "sr",
    "that",
    "the",
    "this",
    "times",
    "team",
    "technologies",
    "technology",
    "university",
    "we",
    "why",
    "yes",
    "you",
}
_PROFILE_ALIAS_ENTRY_TYPES = {
    "character",
    "location",
    "organization",
    "proper_noun",
    "title",
}
_PROFILE_GLOSSARY_MAX_RENDERED_ENTRIES = 48
_PROFILE_WORD_RE = re.compile(
    r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:['’][A-Za-z]+)?"
)
_QUOTED_PUBLISHED_TITLE_RE = re.compile(
    r"[“\"«](?P<title>[^”\"»\n]{3,320})[”\"»]"
)
_UNQUOTED_CITATION_SEGMENT_RE = re.compile(
    r"(?:^|[,;])\s*(?P<title>[^,;\n]{3,320}?)(?=\s*[,;])",
    re.MULTILINE,
)
_BIBLIOGRAPHIC_TITLE_TAIL_RE = re.compile(
    r"""(?ix)
    ^\s*[,.;:)]*\s*
    (?:
        preprint|arxiv|doi|blog|journal|review|press|proceedings|
        vol(?:ume)?\.?|no\.?|pp?\.?|ed(?:ition)?\.?|
        (?:january|february|march|april|may|june|july|august|
           september|october|november|december)\b|
        (?:19|20)\d{2}\b
    )
    """
)
_SOCIAL_MEDIA_CITATION_TAIL_RE = re.compile(
    r"""(?ix)
    ^\s*[,.;:)]*\s*
    (?:twitter|x|facebook|instagram|threads|bluesky|tiktok|linkedin|
       reddit|mastodon|weibo|telegram)\b
    """
)
_SOCIAL_MEDIA_STATUS_URL_RE = re.compile(
    r"""(?ix)
    \b(?:
        (?:x|twitter)\.com/[^/\s<>\[\]]+/status(?:es)?/\d+
        |facebook\.com/[^\s<>\[\]]+/(?:posts|permalink)/[^\s<>\[\]]+
        |instagram\.com/(?:p|reel)/[^\s<>\[\]]+
        |threads\.net/@[^/\s<>\[\]]+/post/[^\s<>\[\]]+
        |bsky\.app/profile/[^/\s<>\[\]]+/post/[^\s<>\[\]]+
    )
    """
)
_IMMUTABLE_LITERAL_PATTERNS = (
    re.compile(r"(?i)(?:https?|ftp)://[^\s<>\"“”«»]+"),
    re.compile(r"(?i)\bwww\.[^\s<>\"“”«»]+"),
    re.compile(r"(?i)\bdoi(?:\.org/|:)\s*10\.\d{4,9}/[^\s<>\"“”«»]+"),
    re.compile(r"(?i)\b10\.\d{4,9}/[^\s<>\"“”«»]+"),
    re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b"),
    re.compile(
        r"(?i)(?<![\w@])"
        r"(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"
        r"[a-z]{2,24}(?:/[^\s<>\"“”«»\[\]]*)?"
    ),
    re.compile(r"(?<!\w)(?:\.\.?/|/)[A-Za-z0-9._~!$&'()*+,;=:@%/\-]+"),
    re.compile(r"(?i)\b[A-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"),
)
_PLACEHOLDER_LITERAL_RE = re.compile(
    r"(?=(?P<open>\[id\d+\])"
    r"(?P<literal>(?:(?!\[id\d+\]).){1,2048}?)"
    r"(?P<close>\[id\d+\]))",
    re.IGNORECASE | re.DOTALL,
)
_PROPER_NAME_GAP_RE = re.compile(r"^[\s\-–—]*$")
_PROPER_NAME_CONNECTORS = _PUBLISHED_TITLE_CONNECTORS - {
    "and",
    "but",
    "e",
    "et",
    "or",
    "und",
    "y",
}


def profile_enabled(prompt_options: Optional[dict]) -> bool:
    options = prompt_options or {}
    return (
        str(options.get("editorial_mode") or "").strip().lower() == "book_profile"
        and bool(str(options.get("profile_id") or "").strip())
    )


def active_profile(prompt_options: Optional[dict]) -> Optional[BookProfile]:
    if not profile_enabled(prompt_options):
        return None
    try:
        return load_book_profile(str((prompt_options or {}).get("profile_id") or ""))
    except BookProfileError:
        if (prompt_options or {}).get("profile_required") is not True:
            return None
        raise


def build_profile_instruction_block(
    prompt_options: Optional[dict],
    *,
    phase: str,
    target_language: str = "",
) -> str:
    profile = active_profile(prompt_options)
    if profile is None:
        return ""

    normalized_phase = str(phase or "").strip().lower()
    prompt_text = (profile.prompt_texts.get(normalized_phase) or "").strip()
    if not prompt_text and normalized_phase in {"voice_restoration", "modernize"}:
        prompt_text = (profile.prompt_texts.get("modernize") or "").strip()

    phase_guidance = _phase_guidance(
        normalized_phase,
        source_language=str((prompt_options or {}).get("_source_language") or ""),
        target_language=target_language,
    )

    lines = [
        "# BOOK EDITORIAL PROFILE",
        "",
        f"Profile id: {profile.profile_id}",
        f"Profile name: {profile.name}",
    ]
    if profile.target_locale:
        lines.append(f"Target locale: {profile.target_locale}")
    if target_language:
        lines.append(f"Output language: {target_language}")
    lines.extend([
        "",
        "Use only the active profile and its approved glossary. Do not apply rules from another book profile.",
        "If a recurring editorial equivalence is needed and not in the active glossary, make the best contextual choice and let the glossary discovery/reporting layer suggest it; do not treat it as a global rule.",
        "When this profile conflicts with generic light-editing or conservative-modernization instructions, the active profile wins.",
        "A high-modernization profile may require substantive sentence-level restructuring while preserving every unit of meaning.",
        "Preserve authorial voice through contemporary literary equivalents; do not preserve obsolete grammar, old conjunctions, enclitic syntax, or archaic/Peninsular address forms merely to create period flavor.",
    ])
    if phase_guidance:
        lines.extend(["", phase_guidance])
    audiobook_guidance = _audiobook_guidance(profile, normalized_phase)
    if audiobook_guidance:
        lines.extend(["", audiobook_guidance])
    editorial_brief = compact_editorial_brief_for_prompt(profile.editorial_artifacts)
    if editorial_brief:
        lines.extend(["", editorial_brief])
    knowledge_snapshot = _profile_knowledge_prompt_snapshot(profile)
    if knowledge_snapshot:
        lines.extend(["", knowledge_snapshot])
    if prompt_text:
        lines.extend(["", prompt_text])
    return "\n".join(lines).strip()


def build_profile_glossary_block(
    chunk_text: str,
    prompt_options: Optional[dict],
    *,
    include_pending: bool = False,
    purpose: str = "translation",
) -> str:
    """Render only the prompt block for backwards-compatible callers."""
    block, _summary = build_profile_glossary_context(
        chunk_text,
        prompt_options,
        include_pending=include_pending,
        purpose=purpose,
    )
    return block


def build_profile_glossary_context(
    chunk_text: str,
    prompt_options: Optional[dict],
    *,
    include_pending: bool = False,
    purpose: str = "translation",
) -> tuple[str, dict | None]:
    """Build a profile glossary block and its diagnostics in one matching pass.

    Translation previously asked for the block and summary separately, which
    loaded the profile, matched every entry, and compiled the prompt context
    twice for every chunk. Keeping both products behind one function makes the
    visibility data describe the exact block that was actually injected.
    """
    profile = active_profile(prompt_options)
    if profile is None:
        return "", None

    entries = _prompt_glossary_entries(profile, include_pending=include_pending)
    excluded_artifact_labels = {
        fold_profile_match_text(entry.source)
        for entry in profile.glossary_entries
        if (
            entry.source
            and entry.approved
            and (
                not _entry_allowed_in_prompt(entry)
                or _entry_conflicts_with_protected_phrase(
                    entry,
                    profile.glossary_entries,
                )
            )
        )
    }
    summary = {
        "profile_id": profile.profile_id,
        "purpose": str(purpose or "translation").strip().lower() or "translation",
        "total_terms": len(entries),
        "matched_terms": 0,
        "rendered_terms": 0,
        "capped": False,
    }
    if not entries:
        return "", summary

    text = chunk_text or ""
    matched = _match_entries(
        text,
        entries,
        match_rendered=_profile_glossary_matches_rendered_side(purpose),
    )
    if not matched:
        return "", summary

    compiled = compile_profile_prompt_context(
        profile,
        text,
        matched,
        purpose=purpose,
        max_entries=_PROFILE_GLOSSARY_MAX_RENDERED_ENTRIES,
        excluded_artifact_labels=excluded_artifact_labels,
    )
    summary.update({
        "matched_terms": compiled.matched_total,
        "rendered_terms": len(compiled.entries),
        "capped": compiled.capped,
    })

    matched_total = compiled.matched_total
    matched = list(compiled.entries)
    capped = compiled.capped
    proper_name_spans = _larger_proper_name_spans(text)
    contextual_title_sources = {
        str(entry.source or "").strip()
        for entry in matched
        if (
            not entry.mechanical_safe
            and _entry_occurs_inside_larger_published_title(
                text,
                entry.source,
            )
        )
    }
    contextual_name_sources = {
        str(entry.source or "").strip()
        for entry in matched
        if (
            not entry.mechanical_safe
            and _is_mechanical_exact_translation_entry(entry)
            and _entry_occurs_inside_larger_proper_name(
                text,
                entry.source,
                name_spans=proper_name_spans,
            )
        )
    }
    artifact_hints = [
        hint
        for hint in compiled.artifact_hints
        if not any(
            re.match(
                rf"^\s*-\s*{re.escape(source)}\s*->",
                str(hint or ""),
                flags=re.IGNORECASE,
            )
            for source in contextual_title_sources
        )
    ]

    lines = [
        "# ACTIVE BOOK GLOSSARY",
        "",
        f"Profile: {profile.profile_id}",
        "Only the entries below matched this passage. Apply them only for this book profile.",
        "The list is ordered by prompt value: exact translations first, then the most specific recurring names and canonical forms.",
        "Preserve exact-spelling entries exactly as shown; do not translate, normalize, or paraphrase those names.",
        "For translated entries, use the right-hand rendering as the canonical lemma; adapt gender, number, articles, and prepositions only when target-language grammar requires it.",
        "",
    ]
    if capped:
        lines.extend([
            f"Showing {len(matched)} highest-signal matches out of {matched_total} matched entries.",
            "",
        ])
    if artifact_hints:
        lines.extend([
            "# ACTIVE EDITORIAL FOCUS FOR THIS PASSAGE",
            "These are compact book-level signals matched to this passage; use them as context, not as permission to add facts.",
            *artifact_hints,
            "",
        ])
    if compiled.warnings:
        lines.extend([
            "# PROFILE PROMPT BUDGET NOTES",
            *[f"Note: {warning}" for warning in compiled.warnings],
            "",
        ])
    for entry in matched:
        lines.extend(
            _render_entry(
                entry,
                source_context=text,
                contextual_name_sources=contextual_name_sources,
            )
        )
    return "\n".join(lines).rstrip() + "\n", summary


def profile_glossary_match_summary(
    chunk_text: str,
    prompt_options: Optional[dict],
    *,
    include_pending: bool = False,
    purpose: str = "translation",
) -> dict | None:
    """Return compact matching diagnostics for the active book profile glossary."""
    _block, summary = build_profile_glossary_context(
        chunk_text,
        prompt_options,
        include_pending=include_pending,
        purpose=purpose,
    )
    return summary


def build_profile_report_summary(prompt_options: Optional[dict]) -> dict:
    profile = active_profile(prompt_options)
    if profile is None:
        return {}
    return {
        "profile_id": profile.profile_id,
        "profile_name": profile.name,
        "target_locale": profile.target_locale,
        "approved_glossary_entries": profile.approved_count,
        "pending_glossary_suggestions": profile.pending_count,
        "allow_common_glossary": profile.allow_common_glossary,
        "allow_cross_profile_glossary": profile.allow_cross_profile_glossary,
        "editorial_artifact_counts": editorial_artifact_counts(profile.editorial_artifacts),
    }


def _profile_knowledge_prompt_snapshot(profile: BookProfile) -> str:
    """Render a small prompt-safe profile health snapshot.

    This is intentionally not a term list. Per-chunk glossary matching remains
    responsible for injecting only relevant entries. The snapshot prevents a
    generated profile from being interpreted as "preserve every source token"
    when most of its reviewed entries are still pending or source==target.
    """
    knowledge = build_profile_knowledge_base(profile).to_dict()
    glossary = knowledge.get("glossary") or {}
    readiness = knowledge.get("prompt_readiness") or {}
    signal_index = knowledge.get("signal_index") or {}
    effective_approved = [
        entry for entry in profile.approved_entries if _entry_allowed_in_prompt(entry)
    ]
    effective_translated = sum(
        1
        for entry in effective_approved
        if entry.target and entry.target.casefold().strip() != entry.source.casefold().strip()
    )
    effective_preserve = sum(
        1
        for entry in effective_approved
        if entry.target and entry.target.casefold().strip() == entry.source.casefold().strip()
    )
    suppressed = max(0, profile.approved_count - len(effective_approved))
    lines = [
        "# PROFILE KNOWLEDGE SNAPSHOT",
        (
            "Prompt-safe approved glossary entries: "
            f"{len(effective_approved)} "
            f"({effective_translated} translate, {effective_preserve} preserve)."
        ),
    ]
    if suppressed:
        lines.append(
            f"Suppressed unsafe generated preserve rules: {suppressed}. Do not infer preserve decisions from suppressed profile artifacts."
        )
    source_equals = effective_preserve
    if source_equals:
        lines.append(
            f"Source-equals-target approved entries: {source_equals}. These are prompt-safe explicit preserve decisions; only the chunk-scoped entries rendered below are binding. Do not infer that ordinary technical or cultural terms should remain untranslated."
        )
    pending = int(glossary.get("pending") or 0)
    if pending:
        lines.append(
            f"Pending glossary suggestions: {pending}. Do not apply pending suggestions as binding rules."
        )
    warnings = list(readiness.get("warnings") or [])
    if warnings:
        lines.append("Profile readiness warnings: " + ", ".join(str(item) for item in warnings[:8]) + ".")
    if signal_index.get("available"):
        coverage = signal_index.get("coverage") or {}
        mode = str(coverage.get("mode") or "").strip()
        risk_flags = list(signal_index.get("risk_flags") or [])
        if mode:
            lines.append(f"Preparation coverage mode: {mode}.")
        if risk_flags:
            lines.append("Editorial signal risks: " + ", ".join(str(item) for item in risk_flags[:8]) + ".")
    return "\n".join(lines).strip()


def apply_profile_glossary_corrections(
    text: str,
    prompt_options: Optional[dict],
    *,
    source_text: str = "",
) -> str:
    """Apply source-safe and approved profile corrections to model output.

    Internal symbols dropped from an otherwise identical source name can be
    restored deterministically. All other book-specific spelling changes still
    require an approved, mechanically safe entry in the active profile.
    """
    if not text:
        return text
    result = restore_symbol_bearing_names(source_text, text)
    result = _restore_placeholder_wrapped_source_literals(source_text, result)
    result = restore_bibliographic_titles(source_text, result)
    profile = active_profile(prompt_options)
    if profile is None:
        return result
    protected_terms = _profile_protected_phrases(profile)

    trigger_text = f"{source_text or ''}\n{text or ''}"
    source_value = str(source_text or "")
    corrections: list[ProfileGlossaryEntry] = []
    for entry in profile.approved_entries:
        if _entry_conflicts_with_protected_phrase(entry, profile.glossary_entries):
            continue
        if (
            not entry.mechanical_safe
            and _entry_occurs_inside_larger_published_title(
                source_value,
                entry.source,
            )
        ):
            continue
        if _is_mechanical_canonical_entry(entry):
            if _contains_term(trigger_text, entry.source):
                corrections.append(entry)
            continue
        if (
            source_value
            and _is_mechanical_exact_translation_entry(entry)
            and _contains_term(source_value, entry.source)
            and _contains_term(result, entry.source)
        ):
            corrections.append(entry)

    # Longer entries win when a title or technical phrase contains another
    # approved source term. Target shielding in _replace_profile_term prevents
    # a replacement from expanding an already-correct target.
    for entry in sorted(
        corrections,
        key=lambda item: (-len(item.source), item.source.casefold()),
    ):
        contextual_protected_terms = (
            protected_terms
            + _proper_name_segments_containing_term(result, entry.source)
        )
        result = _replace_profile_term(
            result,
            entry.source,
            entry.target,
            protected_terms=contextual_protected_terms,
        )
    return result


def apply_profile_exact_translation_corrections(
    text: str,
    prompt_options: Optional[dict],
    *,
    exact_pairs: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str, int]:
    """Apply only approved profile entries explicitly marked ``translate_exact``.

    This is a final consistency net, not a translation pass. Contextual terms,
    pending suggestions, preserve rules, and entries from other profiles are
    deliberately excluded.
    """
    if not text:
        return text, 0
    pairs = exact_pairs
    if pairs is None:
        pairs = profile_exact_translation_pairs(prompt_options)
    if not pairs:
        return text, 0

    profile = active_profile(prompt_options)
    protected_terms = _profile_protected_phrases(profile) if profile is not None else ()
    result = text
    result_lookup = _profile_prefilter_text(result)
    replacements = 0
    for source, target in pairs:
        if _profile_prefilter_text(source) not in result_lookup:
            continue
        contextual_protected_terms = (
            protected_terms
            + _proper_name_segments_containing_term(result, source)
        )
        result, count = _replace_profile_term_counted(
            result,
            source,
            target,
            protected_terms=contextual_protected_terms,
        )
        replacements += count
        if count:
            result_lookup = _profile_prefilter_text(result)
    return result, replacements


def profile_exact_translation_pairs(
    prompt_options: Optional[dict],
) -> tuple[tuple[str, str], ...]:
    """Return approved exact source-target pairs for the active profile."""
    profile = active_profile(prompt_options)
    if profile is None:
        return ()
    return tuple(
        (entry.source, entry.target)
        for entry in profile.approved_entries
        if _is_mechanical_exact_translation_entry(entry)
        and not _entry_conflicts_with_protected_phrase(
            entry,
            profile.glossary_entries,
        )
    )


def _phase_guidance(
    phase: str,
    *,
    source_language: str = "",
    target_language: str = "",
) -> str:
    if phase == "translation":
        return f"""
Translation use of this profile:
- Use the profile to keep names, concepts, titles, acronyms, relationships, voices, and recurring terms consistent across the whole book.
- Apply approved source-to-target glossary entries when the source term appears. When a glossary entry is preserve-as-written, preserve it exactly.
- Use the editorial map as context for register, character/entity continuity, chapter structure, and domain vocabulary.
- Do not modernize, simplify, summarize, censor, or editorialize beyond what is needed for a faithful translation from {source_language or 'the source language'} to {target_language or 'the target language'}.
- If profile guidance conflicts with source fidelity, source fidelity wins; preserve every fact, number, quotation boundary, argument, and scene/action order.
""".strip()

    if phase in {"translation_refinement", "refinement"}:
        return """
Translation refinement use of this profile:
- Improve fluency, consistency, and target-language editorial quality while preserving the translation's source-faithful content.
- Enforce approved glossary choices and consistent recurring terms; do not introduce a new synonym for an already established profile term.
- Keep names, numbers, citations, headings, table values, dialogue roles, and paragraph order intact.
- Do not turn refinement into rewriting, modernization, simplification, explanation, or censorship unless the active task explicitly asks for that.
""".strip()

    if phase in {"audit", "repair"}:
        return """
Audit/repair use of this profile:
- Treat fidelity errors as critical.
- Treat inconsistent glossary use, drifting terminology, wrong register, or profile contamination as repairable editorial issues unless they also alter content.
- Prefer the smallest repair that restores fidelity and consistency.
""".strip()

    return ""


def _audiobook_guidance(profile: BookProfile, phase: str) -> str:
    config = profile.raw_config.get("audiobook") or profile.raw_config.get("audio_sanitization") or {}
    enabled = bool(config.get("enabled") or config.get("generate_companion")) if isinstance(config, dict) else bool(config)
    if not enabled:
        return ""

    if phase == "translation":
        return """
Audiobook companion policy:
- Keep the translation faithful; audiobook cleanup must never summarize, censor, or change source facts.
- Translate informative captions accurately so they can be integrated into the listening text later.
- Do not expand credit-only captions, raw links, page markers, or download-site watermarks into prose.
- Notes and references may be translated, but they should not interrupt the main sentence flow.
""".strip()

    if phase in {"translation_refinement", "refinement", "audit", "repair"}:
        return """
Audiobook companion policy:
- Improve listenability only where it does not alter the source-faithful translation.
- Flag or repair raw URLs, EPUB note anchors, page markers, watermarks, and disruptive inline note calls.
- Informative image captions should remain accurate and concise; visual credits and bibliographic apparatus belong outside the main listening flow.
""".strip()

    return """
Audiobook companion policy:
- Preserve content fidelity while making the final companion text suitable for listening.
""".strip()


def profile_terms_dict(
    prompt_options: Optional[dict],
    *,
    source_text: str = "",
) -> dict[str, str]:
    profile = active_profile(prompt_options)
    if profile is None:
        return {}
    entries = list(profile.approved_entries)
    if source_text:
        entries = _match_entries(source_text, entries)
    terms: dict[str, str] = {}
    for entry in entries:
        if (
            entry.target
            and _entry_allowed_in_prompt(entry)
            and not _entry_conflicts_with_protected_phrase(
                entry,
                profile.glossary_entries,
            )
        ):
            terms[entry.source] = entry.target
    return terms


def _profile_glossary_matches_rendered_side(purpose: str) -> bool:
    normalized = str(purpose or "translation").strip().lower()
    return normalized in {
        "refine",
        "refinement",
        "transform",
        "transformation",
        "modernize",
        "same-language",
    }


def _prompt_glossary_entries(
    profile: BookProfile,
    *,
    include_pending: bool = False,
) -> list[ProfileGlossaryEntry]:
    base_entries = [
        entry for entry in profile.glossary_entries
        if (
            (entry.approved or (include_pending and entry.pending))
            and _entry_allowed_in_prompt(entry)
            and not _entry_conflicts_with_protected_phrase(
                entry,
                profile.glossary_entries,
            )
        )
    ]
    if include_pending:
        return base_entries

    existing_sources = {fold_profile_match_text(entry.source) for entry in base_entries if entry.source}
    alias_seed_entries = [
        entry
        for entry in profile.approved_entries
        if not weakly_supported_generated_preserve_entry(entry.to_dict())
    ]
    aliases = _protected_entity_alias_entries(
        profile.approved_entries,
        existing_sources,
    )
    aliases.extend(_derived_prompt_alias_entries(alias_seed_entries, existing_sources))
    aliases.extend(_editorial_artifact_prompt_alias_entries(profile, existing_sources))
    return base_entries + aliases


def _protected_entity_alias_entries(
    entries: tuple[ProfileGlossaryEntry, ...],
    existing_sources: set[str],
) -> list[ProfileGlossaryEntry]:
    """Expose surnames as contextual aliases without creating rewrite rules."""
    allowed_types = {
        "canonical_proper_noun",
        "character",
        "person",
        "proper_noun",
    }
    aliases: list[ProfileGlossaryEntry] = []
    for entry in entries:
        if (
            not _is_protected_phrase_entry(entry)
            or not _entry_allowed_in_prompt(entry)
            or (entry.entry_type or "").strip().lower() not in allowed_types
        ):
            continue
        words = _PROFILE_WORD_RE.findall(entry.source or "")
        if len(words) < 2:
            continue
        alias = words[-1].strip()
        folded = fold_profile_match_text(alias)
        if (
            not folded
            or folded in existing_sources
            or folded in _PROFILE_ALIAS_STOP_WORDS
            or len(alias) < 3
        ):
            continue
        aliases.append(
            ProfileGlossaryEntry(
                source=alias,
                target=alias,
                entry_type="proper_noun",
                status="approved",
                confidence=max(float(entry.confidence or 0.0), 0.9),
                occurrences=max(int(entry.occurrences or 0), 1),
                injection_policy="contextual",
                translation_policy="translate_contextual",
                decision_rule=(
                    f"Preserve {alias} only when it refers to the established "
                    f"entity {entry.source}; translate ordinary-language uses by context."
                ),
                rationale="Contextual surname alias derived from a protected full name.",
                reviewed_by="derived_protected_entity_alias",
                source_file="derived_prompt_alias",
            )
        )
        existing_sources.add(folded)
    return aliases


def _editorial_artifact_prompt_alias_entries(
    profile: BookProfile,
    existing_sources: set[str],
) -> list[ProfileGlossaryEntry]:
    """Expose high-confidence editorial-map entities as contextual prompt aliases.

    The editorial map is built from the whole book before translation. It often
    knows central entities even when the reviewed glossary contains only noisy
    surrounding phrases. These aliases are deliberately contextual rather than
    hard preserve rules, because the map is broad and can include places or
    common words that should not override ordinary translation decisions.
    """
    artifacts = dict(profile.editorial_artifacts or {})
    seen: set[str] = set()
    aliases: list[ProfileGlossaryEntry] = []
    for section in ("characters_entities", "entities", "do_not_translate"):
        items = artifacts.get(section) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            source = str(item.get("name") or item.get("source") or "").strip()
            if not _artifact_alias_source_allowed(source, item):
                continue
            folded = fold_profile_match_text(source)
            if not folded or folded in existing_sources or folded in seen:
                continue
            seen.add(folded)
            aliases.append(
                ProfileGlossaryEntry(
                    source=source,
                    target=source,
                    entry_type=str(item.get("type") or "proper_noun").strip() or "proper_noun",
                    status="approved",
                    confidence=_artifact_float(item.get("confidence"), default=0.8),
                    occurrences=max(_artifact_int(item.get("occurrences"), default=1), 1),
                    injection_policy="contextual",
                    translation_policy="translate_contextual",
                    decision_rule=(
                        "Use this as a book-specific entity/name spelling when the passage refers "
                        "to the active profile entity; do not override ordinary language meanings."
                    ),
                    rationale="Derived from the profile editorial map.",
                    reviewed_by="derived_editorial_map_alias",
                    source_file="editorial_map",
                )
            )
            existing_sources.add(folded)
    return aliases


def _artifact_alias_source_allowed(source: str, item: dict) -> bool:
    words = _PROFILE_WORD_RE.findall(source or "")
    if not words or len(words) > 4:
        return False
    if any(word.casefold().endswith("'s") for word in words):
        return False
    if any(fold_profile_match_text(word) in _PROFILE_ALIAS_STOP_WORDS for word in words):
        return False
    if not any(word and (word[0].isupper() or word.isupper()) for word in words):
        return False
    if _artifact_int(item.get("occurrences"), default=0) < 3:
        return False
    if _artifact_float(item.get("confidence"), default=0.0) < 0.75:
        return False
    entry_type = str(item.get("type") or "").strip().lower()
    if entry_type and entry_type not in _PROFILE_ALIAS_ENTRY_TYPES:
        return False
    return True


def _artifact_int(value, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _artifact_float(value, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _derived_prompt_alias_entries(
    entries: list[ProfileGlossaryEntry],
    existing_sources: set[str],
) -> list[ProfileGlossaryEntry]:
    """Recover useful proper-name aliases from noisy generated profile phrases.

    Generated book profiles can contain entries such as "Yvonne and Hugh" or
    "Oh Hugh". The full phrases are filtered out before prompt injection, but
    the capitalized names inside them are still useful continuity anchors. These
    aliases are prompt-only; they do not rewrite stored glossary files.
    """
    token_stats: dict[str, dict[str, object]] = {}
    for entry in entries:
        if not _entry_can_seed_prompt_alias(entry):
            continue
        for token in _alias_tokens_from_entry(entry):
            folded = fold_profile_match_text(token)
            if not folded or folded in existing_sources:
                continue
            stats = token_stats.setdefault(
                folded,
                {
                    "source": token,
                    "count": 0,
                    "occurrences": 0,
                    "confidence": 0.0,
                },
            )
            stats["count"] = int(stats["count"]) + 1
            stats["occurrences"] = int(stats["occurrences"]) + max(int(entry.occurrences or 0), 1)
            stats["confidence"] = max(float(stats["confidence"]), float(entry.confidence or 0.0), float(entry.review_confidence or 0.0))

    aliases: list[ProfileGlossaryEntry] = []
    for folded, stats in sorted(token_stats.items(), key=lambda item: (-int(item[1]["occurrences"]), item[0])):
        if int(stats["count"]) < 2:
            continue
        source = str(stats["source"])
        aliases.append(
            ProfileGlossaryEntry(
                source=source,
                target=source,
                entry_type="proper_noun",
                status="approved",
                confidence=float(stats["confidence"]),
                occurrences=int(stats["occurrences"]),
                injection_policy="preserve",
                translation_policy="preserve_exact",
                reviewed_by="derived_profile_prompt_alias",
                source_file="derived_prompt_alias",
            )
        )
        existing_sources.add(folded)
    return aliases


def _entry_can_seed_prompt_alias(entry: ProfileGlossaryEntry) -> bool:
    if not entry.approved or not entry.source or not entry.target:
        return False
    if entry.source.casefold().strip() != entry.target.casefold().strip():
        return False
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    if policy not in {"preserve", "preserve_exact"}:
        return False
    if (entry.entry_type or "").strip().lower() not in _PROFILE_ALIAS_ENTRY_TYPES:
        return False
    words = _PROFILE_WORD_RE.findall(entry.source)
    return len(words) >= 2 or any(word.casefold().endswith("'s") for word in words)


def _alias_tokens_from_entry(entry: ProfileGlossaryEntry) -> list[str]:
    tokens: list[str] = []
    for raw in _PROFILE_WORD_RE.findall(entry.source):
        token = re.sub(r"'s$", "", raw, flags=re.I).strip()
        folded = fold_profile_match_text(token)
        if not folded or folded in _PROFILE_ALIAS_STOP_WORDS:
            continue
        if len(token) < 3:
            continue
        if not (token[0].isupper() or token.isupper()):
            continue
        tokens.append(token)
    return tokens


def _match_entries(
    text: str,
    entries: list[ProfileGlossaryEntry],
    *,
    match_rendered: bool = False,
) -> list[ProfileGlossaryEntry]:
    simple_terms = {
        entry.source: entry.render_target or entry.source
        for entry in entries
        if (
            entry.source
            and not entry.decision_rule
            and entry.render_target
            and not _entry_requires_case_sensitive_match(entry)
        )
    }
    matched_sources: set[str] = set()
    if simple_terms:
        config = GlossaryConfig(
            max_entries=80,
            case_sensitive=False,
            accent_insensitive=True,
            warn_on_cap=False,
        )
        filter_fn = filter_glossary_for_purpose if match_rendered else filter_glossary
        if match_rendered:
            filtered, _capped = filter_fn(
                text,
                simple_terms,
                config,
                "refinement",
            )
        else:
            filtered, _capped = filter_fn(
                text,
                simple_terms,
                config,
            )
        matched_sources.update(filtered.keys())

    folded = fold_profile_match_text(text)
    for entry in entries:
        if entry.source in matched_sources:
            continue
        if _entry_requires_case_sensitive_match(entry):
            if _contains_case_sensitive_term(text, entry.source):
                matched_sources.add(entry.source)
            continue
        purpose = "refinement" if match_rendered else "translation"
        if match_profile_entry_in_text(entry, text, purpose=purpose):
            matched_sources.add(entry.source)
            continue
        source = fold_profile_match_text(entry.source)
        if source and re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", folded):
            matched_sources.add(entry.source)
        elif entry.forbidden_default and fold_profile_match_text(entry.forbidden_default) in folded:
            matched_sources.add(entry.source)

    matched = [entry for entry in entries if entry.source in matched_sources]
    return [
        entry
        for entry in matched
        if not _entry_is_shadowed_by_protected_phrase(text, entry, matched)
    ]


def _is_protected_phrase_entry(entry: ProfileGlossaryEntry) -> bool:
    source = str(entry.source or "").strip()
    target = str(entry.target or "").strip() or source
    if (
        not entry.approved
        or len(_PROFILE_WORD_RE.findall(source)) < 2
        or fold_profile_match_text(source) != fold_profile_match_text(target)
    ):
        return False
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    entry_type = (entry.entry_type or "").strip().lower()
    return policy in {"preserve", "preserve_exact"} or entry_type in _PROTECTED_PHRASE_TYPES


def _profile_protected_phrases(profile: BookProfile) -> tuple[str, ...]:
    return tuple(
        entry.target or entry.source
        for entry in profile.approved_entries
        if _is_protected_phrase_entry(entry)
    )


def _entry_conflicts_with_protected_phrase(
    entry: ProfileGlossaryEntry,
    entries: tuple[ProfileGlossaryEntry, ...] | list[ProfileGlossaryEntry],
) -> bool:
    """Keep ambiguous exact rules out of prompts and mechanical rewrites.

    Generated profiles can contain both a protected full name and a translated
    common word that is part of that name. A short exact rule cannot determine
    whether a later standalone occurrence is a surname/title reference or the
    common word, so that choice must remain contextual.
    """
    if not _is_mechanical_exact_translation_entry(entry):
        return False
    source_words = tuple(
        fold_profile_match_text(word)
        for word in _PROFILE_WORD_RE.findall(entry.source or "")
        if fold_profile_match_text(word)
    )
    if not source_words:
        return False

    source_length = len(source_words)
    for protected in entries:
        if protected is entry or not _is_protected_phrase_entry(protected):
            continue
        protected_words = tuple(
            fold_profile_match_text(word)
            for word in _PROFILE_WORD_RE.findall(protected.source or "")
            if fold_profile_match_text(word)
        )
        if len(protected_words) <= source_length:
            continue
        if any(
            protected_words[index:index + source_length] == source_words
            for index in range(len(protected_words) - source_length + 1)
        ):
            return True
    return False


def _entry_is_shadowed_by_protected_phrase(
    text: str,
    entry: ProfileGlossaryEntry,
    matched_entries: list[ProfileGlossaryEntry],
) -> bool:
    """Suppress a conflicting short rule only inside a protected longer name.

    If the short term also occurs independently in the chunk, it remains
    available. For example, ``Southern -> sur`` is withheld from
    ``Terry Southern`` but still applies to ``Southern audience`` elsewhere.
    """
    source = str(entry.source or "").strip()
    target = str(entry.target or "").strip()
    if (
        not source
        or not target
        or fold_profile_match_text(source) == fold_profile_match_text(target)
    ):
        return False

    source_spans = tuple(
        (match.start(), match.end())
        for match in _profile_term_pattern(source).finditer(text)
    )
    if not source_spans:
        return False

    protected_spans: list[tuple[int, int]] = []
    for protected in matched_entries:
        protected_source = str(protected.source or "").strip()
        if (
            protected is entry
            or len(protected_source) <= len(source)
            or not _is_protected_phrase_entry(protected)
            or not _profile_term_pattern(source).search(protected_source)
        ):
            continue
        protected_spans.extend(
            (match.start(), match.end())
            for match in _profile_term_pattern(protected_source).finditer(text)
        )
    if not protected_spans:
        return False
    return all(
        any(start <= source_start and source_end <= end for start, end in protected_spans)
        for source_start, source_end in source_spans
    )


def _prioritize_matched_entries(entries: list[ProfileGlossaryEntry]) -> list[ProfileGlossaryEntry]:
    return sorted(entries, key=_entry_prompt_priority, reverse=True)


def _entry_prompt_priority(entry: ProfileGlossaryEntry) -> tuple:
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    source = str(entry.source or "").strip()
    target = str(entry.render_target or "").strip()
    words = _PROFILE_WORD_RE.findall(source)
    word_count = len(words)
    confidence = max(float(entry.confidence or 0.0), float(entry.review_confidence or 0.0))
    translated = bool(target and target.casefold() != source.casefold())
    has_decision = bool(entry.decision_rule or entry.forbidden_default or entry.target_options)

    if translated and policy in {"translate_exact", "exact"}:
        category = 5
    elif translated:
        category = 4
    elif has_decision:
        category = 3
    elif policy in {"preserve", "preserve_exact"} and word_count >= 2:
        category = 2
    elif policy in {"preserve", "preserve_exact"}:
        category = 1
    else:
        category = 0

    useful_note = 1 if (
        _profile_note_is_useful(entry.rationale)
        or _profile_note_is_useful(entry.review_rationale)
        or bool(entry.examples)
    ) else 0

    return (
        category,
        useful_note,
        min(max(int(entry.occurrences or 0), 0), 9999),
        min(word_count, 12),
        confidence,
        min(len(source), 120),
        source.casefold(),
    )


def _is_mechanical_canonical_entry(entry: ProfileGlossaryEntry) -> bool:
    if not entry.approved or not entry.source or not entry.target:
        return False
    if entry.source.casefold().strip() == entry.target.casefold().strip():
        return False
    if entry.entry_type not in _CANONICAL_CORRECTION_TYPES:
        return False
    return bool(entry.mechanical_safe or entry.entry_type == "canonical_proper_noun")


def _is_mechanical_exact_translation_entry(entry: ProfileGlossaryEntry) -> bool:
    if not entry.approved or not entry.source or not entry.target:
        return False
    if entry.source.casefold().strip() == entry.target.casefold().strip():
        return False
    if (
        (entry.entry_type or "").strip().lower() in {"title", "work_title"}
        and len(_PROFILE_WORD_RE.findall(entry.source)) == 1
        and not entry.mechanical_safe
    ):
        # A one-word title is also ordinary vocabulary and can appear inside
        # a longer published title (for example, a film title inside a book
        # title). Only explicit mechanical_safe metadata can make that global
        # replacement deterministic; otherwise the model decides by context.
        return False
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    return policy in {"translate_exact", "exact"}


def _entry_requires_case_sensitive_match(entry: ProfileGlossaryEntry) -> bool:
    source = str(entry.source or "").strip()
    return (
        (entry.entry_type or "").strip().lower() == "acronym"
        and source.isupper()
        and len(source) > 1
    )


def _contains_term(text: str, term: str) -> bool:
    return bool(
        re.search(r"(?<!\w)" + re.escape(term or "") + r"(?!\w)", text or "", re.I)
    )


def _contains_case_sensitive_term(text: str, term: str) -> bool:
    return bool(
        re.search(r"(?<!\w)" + re.escape(term or "") + r"(?!\w)", text or "")
    )


def _replace_profile_term(
    text: str,
    source: str,
    target: str,
    *,
    protected_terms: tuple[str, ...] = (),
) -> str:
    result, _ = _replace_profile_term_counted(
        text,
        source,
        target,
        protected_terms=protected_terms,
    )
    return result


def _profile_prefilter_text(value: str) -> str:
    """Cheaply normalize text before invoking exact glossary regexes."""
    return str(value or "").casefold().replace("’", "'")


@lru_cache(maxsize=4096)
def _profile_term_pattern(term: str) -> re.Pattern[str]:
    body = "".join(
        r"['’]" if character in {"'", "’"} else re.escape(character)
        for character in str(term or "")
    )
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.I)


def _replace_profile_term_counted(
    text: str,
    source: str,
    target: str,
    *,
    protected_terms: tuple[str, ...] = (),
) -> tuple[str, int]:
    """Replace a profile term without expanding a target that is already valid.

    Generated canonical entries can legitimately map a shorter source alias to
    a longer target, for example ``Atalante -> L'Atalante``.  A naive substring
    replacement then turns an already-correct target into ``L'L'Atalante`` on
    every retry.  Target spans are shielded while standalone aliases continue
    to be corrected.
    """
    if not text or not source or not target:
        return text, 0

    pattern = _profile_term_pattern(source)
    target_ranges = tuple(
        (match.start(), match.end())
        for match in _profile_term_pattern(target).finditer(text)
    )
    protected_ranges = tuple(
        (match.start(), match.end())
        for term in protected_terms
        if len(str(term or "").strip()) > len(str(source or "").strip())
        for match in _profile_term_pattern(term).finditer(text)
    )
    immutable_ranges = _immutable_literal_ranges(text)
    published_title_ranges = _published_title_ranges(text)
    replacements = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal replacements
        if any(
            start <= match.start() and match.end() <= end
            for start, end in target_ranges + protected_ranges + immutable_ranges
        ):
            return match.group(0)
        if any(
            start <= match.start()
            and match.end() <= end
            and (end - start) > (match.end() - match.start())
            for start, end in published_title_ranges
        ):
            return match.group(0)
        replacements += 1
        found = match.group(0)
        if found.isupper():
            return target.upper()
        if found.islower():
            return target
        return target

    return pattern.sub(replacement, text), replacements


def _render_entry(
    entry: ProfileGlossaryEntry,
    *,
    source_context: str = "",
    contextual_name_sources: set[str] | None = None,
) -> list[str]:
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    entry_type = (entry.entry_type or "profile_entry").replace("_", " ")
    source = _prompt_term_literal(entry.source)
    target = _prompt_term_literal(entry.target)
    nested_published_title = _entry_occurs_inside_larger_published_title(
        source_context,
        entry.source,
    )
    nested_proper_name = (
        str(entry.source or "").strip() in (contextual_name_sources or set())
    )
    head = f"- {source}"
    if policy in {"preserve", "preserve_exact"}:
        head += " -> keep exact spelling"
        if entry.target and entry.target.casefold() != entry.source.casefold():
            head += f" as {target}"
    elif (
        policy in {"translate_exact", "exact"}
        and entry.target
        and (
            (
                (entry.entry_type or "").strip().lower() in {"title", "work_title"}
                and len(_PROFILE_WORD_RE.findall(entry.source)) == 1
            )
            or nested_published_title
        )
    ):
        head += (
            f" -> prefer {target} in ordinary prose or when this is the complete "
            "standalone work title; preserve the source spelling when this term "
            "appears inside a longer published title"
        )
    elif (
        policy in {"translate_exact", "exact"}
        and entry.target
        and nested_proper_name
        and not entry.mechanical_safe
    ):
        head += (
            f" -> prefer {target} in ordinary prose; when this term is part of "
            "a longer proper name, render the complete name contextually and "
            "never replace this word in isolation"
        )
    elif policy in {"translate_exact", "exact"} and entry.target:
        head += f" -> {target}"
    elif policy in {"contextual", "translate_contextual"}:
        if entry.target:
            head += f" -> prefer {target} when context fits"
        elif entry.target_options:
            head += " -> choose contextually from: " + " | ".join(entry.target_options)
        else:
            head += " -> translate by context; do not preserve source wording by default"
    elif entry.target:
        head += f" -> {entry.target}"
    elif entry.target_options:
        head += " -> choose contextually from: " + " | ".join(entry.target_options)
    else:
        head += " -> decide by context"
    head += f" [{entry_type}]"
    lines = [head]
    if entry.decision_rule:
        lines.append(f"  Decision rule: {entry.decision_rule}")
    if entry.forbidden_default:
        lines.append(f"  Forbidden default: {entry.forbidden_default}")
    if _profile_note_is_useful(entry.rationale):
        lines.append(f"  Rationale: {entry.rationale}")
    if (
        entry.review_rationale
        and entry.review_rationale != entry.rationale
        and _profile_note_is_useful(entry.review_rationale)
    ):
        lines.append(f"  Review: {entry.review_rationale}")
    if entry.do_not_apply_if:
        lines.append("  Do not apply if: " + "; ".join(entry.do_not_apply_if[:3]))
    if entry.examples:
        example = entry.examples[0]
        src = example.source_excerpt
        tgt = example.target_excerpt or example.recommended_modernization
        if src or tgt:
            lines.append(f"  Example: {src} => {tgt}")
    return lines


def _entry_occurs_inside_larger_published_title(text: str, term: str) -> bool:
    """Return True when a glossary term is nested in an XHTML title span."""
    value = str(text or "")
    source = str(term or "").strip()
    if not value or not source:
        return False
    source_words = _PROFILE_WORD_RE.findall(source)
    if not source_words:
        return False

    for start, end in _published_title_ranges(value):
        segment = value[start:end].strip()
        if (
            len(_PROFILE_WORD_RE.findall(segment)) > len(source_words)
            and _contains_term(segment, source)
        ):
            return True
    return False


def _entry_occurs_inside_larger_proper_name(
    text: str,
    term: str,
    *,
    name_spans: tuple[tuple[int, int], ...] | None = None,
) -> bool:
    """Return True when a glossary term is nested in a compound proper name.

    This is a structural, profile-agnostic safeguard. Generated profiles can
    legitimately contain a translation for an ordinary word that also occurs
    inside an institution, work, product, or person's name. The short rule
    must not be applied mechanically to that larger name.
    """
    value = str(text or "")
    source = str(term or "").strip()
    if not value or not source:
        return False

    term_spans = tuple(
        (match.start(), match.end())
        for match in _profile_term_pattern(source).finditer(value)
    )
    if not term_spans:
        return False

    resolved_name_spans = (
        name_spans
        if name_spans is not None
        else _larger_proper_name_spans(value)
    )
    return any(
        name_start <= term_start
        and term_end <= name_end
        and (name_end - name_start) > (term_end - term_start)
        for term_start, term_end in term_spans
        for name_start, name_end in resolved_name_spans
    )


def _proper_name_segments_containing_term(
    text: str,
    term: str,
) -> tuple[str, ...]:
    """Return compound-name substrings that must be shielded from replacement."""
    value = str(text or "")
    source = str(term or "").strip()
    if not value or not source:
        return ()
    term_spans = tuple(
        (match.start(), match.end())
        for match in _profile_term_pattern(source).finditer(value)
    )
    protected: list[str] = []
    for name_start, name_end in _larger_proper_name_spans(value):
        if any(
            name_start <= term_start and term_end <= name_end
            for term_start, term_end in term_spans
        ):
            protected.append(value[name_start:name_end])
    return tuple(dict.fromkeys(protected))


def _larger_proper_name_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Find conservative multiword proper-name spans in running source text."""
    value = str(text or "")
    words = list(_PROFILE_WORD_RE.finditer(value))
    spans: set[tuple[int, int]] = set()
    for start_index, first in enumerate(words):
        if not _proper_name_anchor(first.group(0)):
            continue
        anchor_count = 1
        last_anchor_index = start_index
        cursor = start_index
        while cursor + 1 < len(words):
            current = words[cursor]
            following = words[cursor + 1]
            if not _PROPER_NAME_GAP_RE.fullmatch(
                value[current.end():following.start()]
            ):
                break
            following_word = following.group(0)
            if _proper_name_anchor(following_word):
                anchor_count += 1
                last_anchor_index = cursor + 1
                cursor += 1
                continue
            if fold_profile_match_text(following_word) in _PROPER_NAME_CONNECTORS:
                if cursor + 2 >= len(words):
                    break
                after_connector = words[cursor + 2]
                if (
                    not _PROPER_NAME_GAP_RE.fullmatch(
                        value[following.end():after_connector.start()]
                    )
                    or not _proper_name_anchor(after_connector.group(0))
                ):
                    break
                cursor += 1
                continue
            break
        if anchor_count >= 2:
            spans.add((first.start(), words[last_anchor_index].end()))
    return tuple(sorted(spans))


def _proper_name_anchor(word: str) -> bool:
    value = str(word or "").strip()
    return bool(
        value
        and fold_profile_match_text(value) not in _PUBLISHED_TITLE_CONNECTORS
        and (value[:1].isupper() or (len(value) > 1 and value.isupper()))
    )


def profile_term_nested_in_published_title(text: str, term: str) -> bool:
    """Public source-context query shared by glossary and language gates."""
    return _entry_occurs_inside_larger_published_title(text, term)


def profile_phrase_matches_published_title(text: str, phrase: str) -> bool:
    """Return whether a copied phrase is the full lexical content of a title.

    Language-residual alignment omits punctuation, symbols and numeric
    fragments. Comparing normalized word sequences therefore recognizes
    sentence-case media titles such as talks and lectures. Unlike the broader
    glossary title-range detector, this gate-facing query also requires a
    bibliographic citation tail so ordinary title-cased dialogue is not
    mistaken for an immutable work title.
    """
    value = str(text or "")
    phrase_words = tuple(
        fold_profile_match_text(word)
        for word in _PROFILE_WORD_RE.findall(str(phrase or ""))
        if fold_profile_match_text(word)
    )
    if len(phrase_words) < 3:
        return False

    for match in _QUOTED_PUBLISHED_TITLE_RE.finditer(value):
        if not _quoted_match_has_bibliographic_target(
            value,
            match,
            # The tail checks below are stricter than the coarse block
            # classifier and remain useful for short, one-line endnotes that
            # the classifier reasonably labels as narrative.
            critical_context=True,
        ):
            continue
        title_words = tuple(
            fold_profile_match_text(word)
            for word in _PROFILE_WORD_RE.findall(match.group("title"))
            if fold_profile_match_text(word)
        )
        if title_words == phrase_words:
            return True
    return False


def _looks_like_published_title_segment(text: str) -> bool:
    words = _PROFILE_WORD_RE.findall(text or "")
    if not 3 <= len(words) <= 32:
        return False
    anchors = 0
    lowercase_content_words = 0
    for word in words:
        folded = fold_profile_match_text(word)
        if folded in _PUBLISHED_TITLE_CONNECTORS:
            continue
        if word[:1].isupper() or (len(word) >= 2 and word.isupper()):
            anchors += 1
            continue
        lowercase_content_words += 1
    return (
        anchors >= 2
        and lowercase_content_words <= max(1, anchors // 3)
    )


@lru_cache(maxsize=256)
def _is_critical_apparatus_context(text: str) -> bool:
    """Recognize bibliography/note context without adding profile-specific rules."""
    value = str(text or "").strip()
    if not value:
        return False
    try:
        blocks = DocumentBlockClassifier(source_type="text").classify_text(value)
    except Exception:
        return False
    return any(
        block.type == "critical_apparatus" and block.confidence >= 0.70
        for block in blocks
    )


def _published_title_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return larger cited-title spans that short glossary terms must not rewrite."""
    value = str(text or "")
    if not value:
        return ()
    critical_context = _is_critical_apparatus_context(value)
    ranges: set[tuple[int, int]] = set()

    for match in _QUOTED_PUBLISHED_TITLE_RE.finditer(value):
        title = match.group("title").strip()
        word_count = len(_PROFILE_WORD_RE.findall(title))
        if (
            3 <= word_count <= 48
            and (
                _looks_like_published_title_segment(title)
                or _quoted_match_has_bibliographic_target(
                    value,
                    match,
                    critical_context=critical_context,
                )
            )
        ):
            ranges.add((match.start("title"), match.end("title")))

    placeholders = list(_EPUB_PLACEHOLDER_RE.finditer(value))
    for left, right in zip(placeholders, placeholders[1:]):
        if left.end() >= right.start():
            continue
        segment = value[left.end():right.start()].strip()
        if _looks_like_published_title_segment(segment):
            ranges.add((left.end(), right.start()))

    # Final semantic auditing removes XHTML placeholders before comparing a
    # candidate with its source. In an analytical index this can turn
    # ``[id5]Machine Learning for Health[id6]`` into an unquoted,
    # comma-delimited segment. Recover only title-shaped proper-name spans in a
    # dense index row; ordinary prose and copied source clauses retain sentence
    # punctuation and therefore stay outside this exception.
    dense_index_context = bool(
        value.count(",") >= 4
        and not re.search(r"[.!?]", value)
    )
    if dense_index_context:
        for start, end in _larger_proper_name_spans(value):
            segment = value[start:end].strip()
            if _looks_like_published_title_segment(segment):
                ranges.add((start, end))

    # Semantic XHTML auditing intentionally removes inline placeholders. In
    # notes and bibliographies this leaves unquoted titles comma-delimited,
    # for example ``Author, Recent Trends in Large Language Models, Publisher,
    # 2023``. Detect those identity-bearing segments only inside independently
    # verified critical apparatus and only when the remaining citation contains
    # a year or immutable locator. This keeps ordinary narrative prose outside
    # the exemption while preventing a short glossary rule from corrupting a
    # complete published title.
    if critical_context:
        for match in _UNQUOTED_CITATION_SEGMENT_RE.finditer(value):
            segment = match.group("title").strip()
            if not _looks_like_published_title_segment(segment):
                continue
            tail = value[match.end("title"):match.end("title") + 640]
            if not (
                re.search(r"\b(?:19|20)\d{2}\b", tail)
                or _immutable_literal_ranges(tail)
            ):
                continue
            leading = len(match.group("title")) - len(match.group("title").lstrip())
            trailing = len(match.group("title")) - len(match.group("title").rstrip())
            ranges.add(
                (
                    match.start("title") + leading,
                    match.end("title") - trailing,
                )
            )
    return tuple(sorted(ranges))


def _immutable_literal_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """Return URL/DOI/email/path spans that editorial rules may never mutate."""
    value = str(text or "")
    ranges = {
        (match.start(), match.end())
        for pattern in _IMMUTABLE_LITERAL_PATTERNS
        for match in pattern.finditer(value)
    }
    return tuple(sorted(ranges))


def _is_immutable_literal(value: str) -> bool:
    literal = str(value or "").strip()
    return bool(
        literal
        and any(pattern.fullmatch(literal) for pattern in _IMMUTABLE_LITERAL_PATTERNS)
    )


def _quoted_match_has_bibliographic_target(
    text: str,
    match: re.Match[str],
    *,
    critical_context: bool,
) -> bool:
    """Distinguish cited titles from ordinary quotations in note apparatus."""
    if not critical_context:
        return False
    tail = str(text or "")[match.end():match.end() + 640]
    next_quotation = re.search(r"[“\"«]", tail)
    if next_quotation:
        tail = tail[:next_quotation.start()]
    # Quoted posts, messages and status updates are source prose, even when
    # their citation tail looks exactly like a conventional publication entry.
    # Restoring them as immutable work titles would undo a correct translation.
    if (
        _SOCIAL_MEDIA_CITATION_TAIL_RE.search(tail)
        or _SOCIAL_MEDIA_STATUS_URL_RE.search(tail)
    ):
        return False
    return bool(
        _BIBLIOGRAPHIC_TITLE_TAIL_RE.search(tail)
        or (
            _looks_like_published_title_segment(match.group("title"))
            and
            _immutable_literal_ranges(tail)
            and re.search(r"(?:19|20)\d{2}|[,;]\s*[A-ZÁÉÍÓÚÜÑ]", tail)
        )
    )


def restore_bibliographic_titles(
    source_text: str,
    candidate_text: str,
) -> str:
    """Restore complete cited-work titles while preserving candidate quote style.

    Published titles are identity-bearing bibliography data. Translating one
    word through a glossary while leaving the rest in the source language
    creates a hybrid title that is neither a valid citation nor a translation.
    This source-aware repair is limited to text classified as critical
    apparatus; ordinary dialogue and narrative quotations remain untouched.
    """
    source = str(source_text or "")
    candidate = str(candidate_text or "")
    if not source or not candidate:
        return candidate

    source_matches = list(_QUOTED_PUBLISHED_TITLE_RE.finditer(source))
    candidate_matches = list(_QUOTED_PUBLISHED_TITLE_RE.finditer(candidate))
    if not source_matches or len(source_matches) != len(candidate_matches):
        return candidate

    source_title_ranges = set(_published_title_ranges(source))
    replacements: list[tuple[int, int, str]] = []
    for source_match, candidate_match in zip(source_matches, candidate_matches):
        source_range = (source_match.start("title"), source_match.end("title"))
        if (
            source_range not in source_title_ranges
            or not _quoted_match_has_bibliographic_target(
                source,
                source_match,
                critical_context=True,
            )
        ):
            continue
        source_title = source_match.group("title")
        candidate_title = candidate_match.group("title")
        source_core = source_title.rstrip(" \t,.;:")
        candidate_core = candidate_title.rstrip(" \t,.;:")
        candidate_suffix = candidate_title[len(candidate_core):]
        repaired_title = source_core + candidate_suffix
        if repaired_title != candidate_title:
            replacements.append(
                (
                    candidate_match.start("title"),
                    candidate_match.end("title"),
                    repaired_title,
                )
            )

    result = candidate
    for start, end, replacement in reversed(replacements):
        result = result[:start] + replacement + result[end:]
    return result


def _restore_placeholder_wrapped_source_literals(
    source_text: str,
    candidate_text: str,
) -> str:
    """Restore source URLs/DOIs/paths enclosed by the same structural markers.

    EPUB anchors are represented as adjacent ``[idN]`` placeholders while the
    model works on text. Restoring their visible technical literal here repairs
    accidental model edits and prevents glossary rules from changing them again.
    """
    source = str(source_text or "")
    result = str(candidate_text or "")
    if not source or not result:
        return result

    for match in _PLACEHOLDER_LITERAL_RE.finditer(source):
        literal = match.group("literal").strip()
        if not _is_immutable_literal(literal):
            continue
        opening = match.group("open")
        closing = match.group("close")
        candidate_span = re.compile(
            re.escape(opening)
            + r"(?:(?!\[id\d+\]).){0,2048}?"
            + re.escape(closing),
            re.IGNORECASE | re.DOTALL,
        )
        replacement = f"{opening}{literal}{closing}"
        result = candidate_span.sub(lambda _match: replacement, result, count=1)
    return result


def _prompt_term_literal(value: str) -> str:
    """Protect syntax-bearing terms from Markdown-like prompt parsing."""
    text = str(value or "")
    if re.search(r"[*\\`~]", text):
        return "`" + text.replace("`", "\\`") + "`"
    return text


def _profile_note_is_useful(note: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(note or "").strip()).casefold()
    return bool(normalized and normalized not in _GENERIC_PROFILE_NOTES)


def _preserve_entry_looks_like_fragment(entry: ProfileGlossaryEntry) -> bool:
    source = str(entry.source or "").strip()
    target = str(entry.target or "").strip()
    if not source or not target or source.casefold() != target.casefold():
        return False
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    if policy not in {"preserve", "preserve_exact"}:
        return False
    if (entry.entry_type or "").strip().lower() not in {
        "proper_noun",
        "character",
        "location",
        "organization",
        "title",
    }:
        return False

    raw_words = _PROFILE_WORD_RE.findall(source)
    words = [word.casefold() for word in raw_words]
    if not words:
        return False
    if any(word.endswith("'s") for word in words):
        return True
    if len(words) < 2:
        return False
    if words[0] in _PRESERVE_FRAGMENT_START_WORDS:
        return True
    if words[-1] in _PRESERVE_FRAGMENT_TRAILING_WORDS:
        return True
    if any(word in _PRESERVE_FRAGMENT_ANY_WORDS for word in words):
        return True
    if len(words) >= 3 and "the" in words[1:-1]:
        return True
    return False


def _entry_allowed_in_prompt(entry: ProfileGlossaryEntry) -> bool:
    if not entry.source:
        return False
    policy = (entry.injection_policy or entry.translation_policy or "").strip().lower()
    if policy == "do_not_inject":
        return False
    source_target_same = bool(
        entry.target and entry.target.casefold().strip() == entry.source.casefold().strip()
    )
    if source_target_same and policy in {"preserve", "preserve_exact"}:
        if _preserve_entry_looks_like_fragment(entry):
            return False
        if suspicious_preserve_entry(entry.to_dict()):
            return False
    if not source_target_same:
        return True
    if policy in {"preserve", "preserve_exact"}:
        return True
    if entry.entry_type in {"technical", "technical_term", "concept", "term"}:
        return False
    return True
