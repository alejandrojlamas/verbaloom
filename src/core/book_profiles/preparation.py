"""Pre-translation preparation for book-scoped editorial profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import html
import io
import json
import math
import posixpath
import re
import tempfile
from urllib.parse import unquote
import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import yaml

from src.core.glossary import extract_glossary_candidates
from src.core.llm.request_deadline import await_llm_call
from src.core.output_formats import extract_readable_text
from src.utils.file_detector import detect_file_type_by_content
from src.utils.archive_safety import validate_zip_archive, validate_zip_bytes

from .artifacts import (
    build_editorial_signal_index,
    build_local_editorial_map,
    enrich_editorial_map_with_reviewed_terms,
    editorial_artifact_counts,
    merge_llm_editorial_map,
    write_editorial_artifacts,
    write_editorial_signal_index,
)
from .discovery import (
    build_glossary_discovery_prompt,
    merge_pending_suggestions,
    parse_glossary_discovery_payload,
)
from .loader import create_profile, load_book_profile, profile_exists, resolve_profiles_root
from .profile_goals import ProfileGoalRules, resolve_profile_goal
from .term_review import (
    review_profile_terms,
    reviewed_candidate_to_approved_entry,
    review_to_pending_suggestion,
)


SUPPORTED_PREP_EXTENSIONS = {
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".csv",
    ".tsv",
    ".json",
    ".xml",
    ".html",
    ".htm",
    ".yaml",
    ".yml",
    ".srt",
    ".epub",
    ".docx",
    ".pdf",
}

_FULL_TEXT_CAP = 5_000_000
_MIN_CHUNK_CHARS = 900
_MAX_FULL_DISCOVERY_CHUNKS = 600
_MAX_LLM_TERM_CHARS = 120
_MAX_LLM_TERM_WORDS = 10
_MAX_EXAMPLE_CHARS = 260
_MAX_AUTO_APPROVED_LOCAL_ENTRIES = 120
_MAX_LOCAL_PENDING_SUGGESTIONS = 260
_MIN_AUTO_APPROVE_CONFIDENCE = 0.78
_MIN_PENDING_LOCAL_CONFIDENCE = 0.42
_MIN_MULTIWORD_AUTO_APPROVE_OCCURRENCES = 2
_MIN_SINGLE_WORD_AUTO_APPROVE_OCCURRENCES = 4
_USEFUL_LLM_TYPES = {
    "lexical_archaism",
    "orthographic_variant",
    "address_form",
    "idiom",
    "character_voice",
    "proper_noun",
    "false_proper_noun",
    "iconic_phrase",
    "sensitive_term",
    "syntax_pattern",
    "technical_term",
    "concept",
    "title",
    "term",
    "acronym",
}
_ROMAN_NUMERAL_RE = re.compile(r"(?i)^[mdclxvi]{1,8}$")
_STRUCTURAL_ID_RE = re.compile(
    r"(?i)^(?:cap[ií]tulo|chapter|section|subcap|page|pagina|p[aá]gina|fig|figure|tabla|table|siglo)[_-]?\d*[a-z0-9_-]*$"
)
_SAFE_SINGLE_WORD_STOPWORDS = {
    "acaso",
    "ademas",
    "además",
    "algo",
    "algunos",
    "alguna",
    "algunas",
    "ante",
    "ahora",
    "alli",
    "allí",
    "aquel",
    "aquella",
    "aquellas",
    "aquellos",
    "aunque",
    "cada",
    "casi",
    "como",
    "cómo",
    "con",
    "cuando",
    "cuándo",
    "cual",
    "cuál",
    "cuales",
    "cuáles",
    "de",
    "decian",
    "decían",
    "del",
    "dijo",
    "dijeron",
    "dijimos",
    "dicen",
    "despues",
    "después",
    "donde",
    "dónde",
    "durante",
    "el",
    "ellas",
    "ellos",
    "en",
    "entre",
    "entonces",
    "era",
    "eran",
    "eres",
    "esta",
    "está",
    "estaba",
    "estaban",
    "estas",
    "estás",
    "este",
    "esto",
    "estos",
    "existe",
    "finalmente",
    "fueron",
    "gran",
    "hacia",
    "hacía",
    "hasta",
    "hemos",
    "inmediatamente",
    "la",
    "las",
    "lo",
    "los",
    "luego",
    "mas",
    "más",
    "mayor",
    "mientras",
    "muchas",
    "mucho",
    "nadie",
    "nosotros",
    "nunca",
    "otras",
    "otros",
    "para",
    "pero",
    "porque",
    "precisamente",
    "puede",
    "pues",
    "quien",
    "quién",
    "respondieron",
    "segun",
    "según",
    "sobre",
    "solo",
    "sólo",
    "tambien",
    "también",
    "todos",
    "todas",
    "tras",
    "venid",
    "vienen",
    "vinieron",
    "véase",
}
_GENERIC_CAPITALIZED_WORDS = {
    "anales",
    "apendice",
    "apéndice",
    "biblioteca",
    "canal",
    "capitulo",
    "capítulo",
    "caracteres",
    "cartas",
    "conquista",
    "cronica",
    "crónica",
    "dador",
    "estado",
    "favor",
    "firme",
    "general",
    "golfo",
    "hecho",
    "historia",
    "historicas",
    "históricas",
    "igualmente",
    "indias",
    "introduccion",
    "introducción",
    "mexicanos",
    "mexicanas",
    "mundo",
    "nacional",
    "relacion",
    "relación",
    "resurreccion",
    "resurrección",
    "señor",
    "señora",
    "señores",
    "señoras",
    "triste",
    "universitario",
    "vision",
    "visión",
}
_ACRONYM_NOISE = {
    "A",
    "AL",
    "DE",
    "DEL",
    "EL",
    "EN",
    "LA",
    "LAS",
    "LO",
    "LOS",
    "Y",
}
_TREATMENT_WORDS = {
    "don",
    "doña",
    "fray",
    "licenciado",
    "majestad",
    "señor",
    "señora",
    "señores",
    "señoras",
}


@dataclass(frozen=True)
class PreparedProfileResult:
    profile_id: str
    profile_name: str
    profile_path: Path
    text_chars: int
    approved_entries: int
    pending_suggestions: int
    local_candidates: int
    llm_suggestions: int
    llm_chunks: int
    coverage_mode: str = "sampled"
    max_local_terms: int = 0
    llm_chunk_chars: int = 0
    provider: str = ""
    model: str = ""
    review_provider: str = ""
    review_model: str = ""
    editorial_artifact_counts: Mapping[str, int] = field(default_factory=dict)
    term_review: Mapping[str, int] = field(default_factory=dict)
    profile_goal: str = ""
    profile_goal_label: str = ""
    business_limits: Mapping[str, int] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_name": self.profile_name,
            "profile_path": str(self.profile_path),
            "text_chars": self.text_chars,
            "approved_entries": self.approved_entries,
            "pending_suggestions": self.pending_suggestions,
            "local_candidates": self.local_candidates,
            "llm_suggestions": self.llm_suggestions,
            "llm_chunks": self.llm_chunks,
            "coverage_mode": self.coverage_mode,
            "max_local_terms": self.max_local_terms,
            "llm_chunk_chars": self.llm_chunk_chars,
            "provider": self.provider,
            "model": self.model,
            "review_provider": self.review_provider,
            "review_model": self.review_model,
            "editorial_artifact_counts": dict(self.editorial_artifact_counts),
            "term_review": dict(self.term_review),
            "profile_goal": self.profile_goal,
            "profile_goal_label": self.profile_goal_label,
            "business_limits": dict(self.business_limits),
            "warnings": list(self.warnings),
        }


def extract_profile_prep_text_from_bytes(
    file_data: bytes,
    filename: str,
    *,
    hard_cap: int = _FULL_TEXT_CAP,
) -> str:
    """Extract readable text from upload bytes for profile preparation."""
    suffix = Path(filename or "").suffix.lower()

    if suffix in {
        ".txt",
        ".text",
        ".md",
        ".markdown",
        ".rst",
        ".log",
        ".csv",
        ".tsv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".yaml",
        ".yml",
        ".srt",
    }:
        try:
            return file_data.decode("utf-8-sig", errors="replace")[:hard_cap]
        except Exception:
            return file_data.decode("utf-8", errors="replace")[:hard_cap]

    temp_path: Optional[Path] = None
    try:
        if suffix in {".epub", ".docx"}:
            validate_zip_bytes(file_data)
        with tempfile.NamedTemporaryFile(
            prefix="verbaloom_profile_prep_",
            suffix=suffix if suffix else ".bin",
            delete=False,
        ) as temp:
            temp.write(file_data)
            temp_path = Path(temp.name)
        detected_type = None
        if suffix not in SUPPORTED_PREP_EXTENSIONS:
            detected_type = detect_file_type_by_content(str(temp_path))
            if detected_type is None:
                raise ValueError(f"Unsupported file type for profile preparation: {suffix or '?'}")
            if detected_type in {"epub", "docx"}:
                validate_zip_bytes(file_data)
        try:
            return extract_readable_text(temp_path)[:hard_cap]
        except RecursionError:
            if suffix == ".epub" or detected_type == "epub":
                fallback = _extract_epub_profile_text_fallback(file_data, hard_cap=hard_cap)
                if fallback.strip():
                    return fallback[:hard_cap]
            raise
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _extract_epub_profile_text_fallback(file_data: bytes, *, hard_cap: int) -> str:
    """Non-recursive EPUB text fallback used only for profile preparation."""
    parts: list[str] = []
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(file_data), "r") as zf:
            validate_zip_archive(zf)
            for name in _epub_profile_text_names(zf):
                try:
                    text = _html_bytes_to_plain_text(zf.read(name))
                except Exception:
                    continue
                if not text:
                    continue
                remaining = hard_cap - total
                if remaining <= 0:
                    break
                text = text[:remaining]
                parts.append(text)
                total += len(text)
    except Exception:
        return ""
    return "\n\n".join(parts)[:hard_cap]


def _epub_profile_text_names(zf: zipfile.ZipFile) -> list[str]:
    available = set(zf.namelist())
    opf_path = _epub_profile_opf_path(zf)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        normalized = posixpath.normpath(unquote(name or "")).lstrip("/")
        if not normalized or normalized.startswith("../"):
            return
        if normalized in available and normalized not in seen:
            ordered.append(normalized)
            seen.add(normalized)

    if opf_path:
        opf_dir = posixpath.dirname(opf_path)
        try:
            opf_text = zf.read(opf_path).decode("utf-8", errors="replace")
        except Exception:
            opf_text = ""
        manifest: dict[str, tuple[str, str]] = {}
        for item in re.finditer(r"<item\b[^>]*>", opf_text, flags=re.IGNORECASE):
            tag = item.group(0)
            item_id = _xml_attr(tag, "id")
            href = _xml_attr(tag, "href")
            media_type = (_xml_attr(tag, "media-type") or "").lower()
            if item_id and href:
                manifest[item_id] = (href, media_type)
        for itemref in re.finditer(r"<itemref\b[^>]*>", opf_text, flags=re.IGNORECASE):
            idref = _xml_attr(itemref.group(0), "idref")
            href, media_type = manifest.get(idref or "", ("", ""))
            if media_type in {"application/xhtml+xml", "text/html"}:
                add(posixpath.join(opf_dir, href))

    if not ordered:
        for name in zf.namelist():
            lower = name.lower()
            if not lower.endswith((".xhtml", ".html", ".htm")):
                continue
            base = posixpath.basename(lower)
            if base in {"nav.xhtml", "nav.html", "toc.xhtml", "toc.html", "contents.xhtml", "contents.html"}:
                continue
            add(name)
    return ordered


def _epub_profile_opf_path(zf: zipfile.ZipFile) -> str | None:
    try:
        container = zf.read("META-INF/container.xml").decode("utf-8", errors="replace")
    except Exception:
        container = ""
    if container:
        match = re.search(r"full-path\s*=\s*['\"]([^'\"]+)['\"]", container, flags=re.IGNORECASE)
        if match:
            candidate = posixpath.normpath(unquote(match.group(1))).lstrip("/")
            if candidate in zf.namelist():
                return candidate
    for name in zf.namelist():
        if name.lower().endswith(".opf"):
            return name
    return None


def _xml_attr(tag: str, name: str) -> str:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(['\"])(.*?)\1",
        tag or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return html.unescape(match.group(2)) if match else ""


def _html_bytes_to_plain_text(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="replace")
    text = re.sub(r"(?is)<(script|style|svg)\b.*?</\1>", " ", text)
    body_match = re.search(r"(?is)<body\b[^>]*>(.*?)</body>", text)
    if body_match:
        text = body_match.group(1)
    text = re.sub(r"(?i)<\s*br\b[^>]*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|section|article|li|tr|h[1-6]|blockquote)\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


async def prepare_book_profile_from_text(
    text: str,
    *,
    source_name: str,
    profile_id: str = "",
    profile_name: str = "",
    source_language: str = "",
    language: str = "",
    target_locale: str = "",
    transform_mode: str = "modernize",
    profiles_root: str | Path | None = None,
    llm_provider: Any = None,
    term_review_provider: Any = None,
    provider_name: str = "",
    model: str = "",
    term_review_provider_name: str = "",
    term_review_model: str = "",
    max_local_terms: int | None = None,
    llm_chunk_chars: int = 3500,
    max_llm_chunks: int = 8,
    llm_full_coverage: bool = False,
    profile_goal: str = "",
    auto_approve_safe_terms: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PreparedProfileResult:
    """Create a scoped profile and fill it with preflight glossary findings.

    Local extraction scans the full document and can approve only mechanically
    safe preserve-as-written entries. LLM discovery is stored as pending by
    default, so generated book-specific decisions never become global rules.
    """
    source_text = (text or "").strip()
    if len(source_text) < 200:
        raise ValueError("Not enough readable text to prepare an editorial profile.")
    goal_rules = resolve_profile_goal(
        profile_goal,
        transform_mode=transform_mode,
        source_name=source_name,
    )

    _emit_progress(
        progress_callback,
        "profile_setup",
        f"Creando perfil editorial de obra para {goal_rules.label.lower()}...",
        5,
        profile_goal=goal_rules.key,
    )
    root = resolve_profiles_root(profiles_root)
    existing_profile = _existing_generated_profile_for_source(
        root,
        source_name=source_name,
        requested_profile_id=profile_id,
    )
    if existing_profile is not None and existing_profile.approved_count > 0:
        warning = (
            "Reused existing generated profile with an approved glossary for this source; "
            "skipped regeneration to avoid replacing useful glossary entries."
        )
        _emit_progress(
            progress_callback,
            "profile_reused",
            "Perfil existente reutilizado; se conserva el glosario aprobado.",
            100,
            profile_id=existing_profile.profile_id,
            approved_entries=existing_profile.approved_count,
            pending_suggestions=existing_profile.pending_count,
        )
        return PreparedProfileResult(
            profile_id=existing_profile.profile_id,
            profile_name=existing_profile.name,
            profile_path=existing_profile.root,
            text_chars=len(source_text),
            approved_entries=existing_profile.approved_count,
            pending_suggestions=existing_profile.pending_count,
            local_candidates=0,
            llm_suggestions=0,
            llm_chunks=0,
            coverage_mode="reused",
            max_local_terms=0,
            llm_chunk_chars=0,
            provider=provider_name,
            model=model,
            review_provider=term_review_provider_name,
            review_model=term_review_model,
            editorial_artifact_counts=editorial_artifact_counts(existing_profile.editorial_artifacts),
            term_review={},
            profile_goal=str(existing_profile.raw_config.get("profile_goal") or goal_rules.key),
            profile_goal_label=goal_rules.label,
            business_limits={},
            warnings=(warning,),
        )

    resolved_id = _unique_profile_id(
        profile_id or _profile_id_from_source(source_name),
        profiles_root=root,
    )
    resolved_name = profile_name.strip() or _profile_name_from_source(source_name, resolved_id)

    profile_dir = create_profile(resolved_id, profiles_root=root)
    _write_generated_profile_config(
        profile_dir,
        profile_id=resolved_id,
        profile_name=resolved_name,
        source_name=source_name,
        source_language=source_language,
        language=language,
        target_locale=target_locale,
        transform_mode=transform_mode,
        goal_rules=goal_rules,
        provider_name=provider_name,
        model=model,
        term_review_provider_name=term_review_provider_name,
        term_review_model=term_review_model,
    )
    _write_generated_policy(
        profile_dir,
        profile_name=resolved_name,
        source_name=source_name,
        source_language=source_language,
        language=language,
        target_locale=target_locale,
        transform_mode=transform_mode,
        goal_rules=goal_rules,
    )
    _emit_progress(
        progress_callback,
        "profile_setup",
        f"Perfil base guardado para {goal_rules.label.lower()}; iniciando escaneo local.",
        15,
        profile_id=resolved_id,
        profile_goal=goal_rules.key,
    )

    warnings: list[str] = []
    max_local_terms = goal_rules.local_terms_limit(
        len(source_text),
        requested=int(max_local_terms) if max_local_terms else None,
    )
    llm_chunk_chars = max(_MIN_CHUNK_CHARS, int(llm_chunk_chars or 3500))
    max_llm_chunks = max(0, int(max_llm_chunks or 0))
    local_candidates, local_warnings = extract_glossary_candidates(
        source_text,
        max_terms=max_local_terms,
        min_occurrences=2,
        existing_sources=set(),
        source_language=source_language,
    )
    warnings.extend(local_warnings)
    _emit_progress(
        progress_callback,
        "local_scan",
        f"Escaneo local listo para {goal_rules.label.lower()}: {len(local_candidates)} candidatos.",
        30,
        local_candidates=len(local_candidates),
        profile_goal=goal_rules.key,
    )

    editorial_map = build_local_editorial_map(
        source_text,
        profile_id=resolved_id,
        source_name=source_name,
        source_language=source_language,
        language=language,
        target_locale=target_locale,
        transform_mode=transform_mode,
        local_candidates=local_candidates,
    )
    write_editorial_artifacts(profile_dir, editorial_map)
    _emit_progress(
        progress_callback,
        "editorial_map",
        "Mapa editorial local guardado: entidades, estructura, voces y riesgos iniciales.",
        34,
        editorial_artifacts=editorial_artifact_counts(editorial_map),
    )

    _emit_progress(
        progress_callback,
        "term_review",
        (
            f"Revisando y reclasificando terminos para {goal_rules.label.lower()} "
            f"con {term_review_model or model or 'reviewer'} antes de aprobar el glosario."
        ),
        36,
        local_candidates=len(local_candidates),
        review_model=term_review_model or model,
        profile_goal=goal_rules.key,
    )
    reviewed_candidates, review_summary = await review_profile_terms(
        local_candidates,
        profile_id=resolved_id,
        source_name=source_name,
        source_language=source_language,
        language=language,
        target_locale=target_locale,
        transform_mode=transform_mode,
        profile_goal=goal_rules.key,
        goal_rules=goal_rules,
        text_chars=len(source_text),
        llm_provider=term_review_provider or llm_provider,
        model=term_review_model or model,
        progress_callback=progress_callback,
    )
    term_review = review_summary.to_dict()
    _emit_progress(
        progress_callback,
        "term_review",
        (
            "Revision de terminos lista: "
            f"{term_review.get('auto_approved_preserve', 0)} preservados, "
            f"{term_review.get('auto_approved_translations', 0)} traducciones aprobadas, "
            f"{term_review.get('pending_review', 0)} pendientes."
        ),
        39,
        review_model=term_review_model or model,
        **term_review,
    )

    approved_entries = _safe_approved_entries(
        reviewed_candidates,
        profile_id=resolved_id,
        auto_approve=auto_approve_safe_terms,
        transform_mode=transform_mode,
        target_locale=target_locale,
        goal_rules=goal_rules,
        max_entries=goal_rules.approved_limit(
            text_chars=len(source_text),
            candidate_count=len(reviewed_candidates),
        ),
    )
    if approved_entries:
        _write_glossary_entries(profile_dir / "glossary" / "terms.yml", approved_entries)
    _emit_progress(
        progress_callback,
        "local_glossary",
        f"Glosario inicial guardado: {len(approved_entries)} entradas revisadas y seguras.",
        40,
        approved_entries=len(approved_entries),
        **term_review,
    )

    pending: list[dict[str, Any]] = []
    pending.extend(
        _pending_from_local_candidates(
            reviewed_candidates,
            profile_id=resolved_id,
            approved_sources={entry["source"].casefold() for entry in approved_entries},
            goal_rules=goal_rules,
            max_entries=goal_rules.pending_limit(
                text_chars=len(source_text),
                candidate_count=len(reviewed_candidates),
            ),
        )
    )
    # Local pending candidates may be enriched by Flash with a contextual
    # target or risk. Only approved terms and already accepted LLM suggestions
    # block another discovery result; pending/local duplicates are merged below.
    seen_sources = {
        str(entry.get("source") or "").casefold()
        for entry in approved_entries
        if str(entry.get("source") or "").strip()
    }

    llm_suggestions: list[dict[str, Any]] = []
    llm_chunks = 0
    coverage_mode = "full" if llm_full_coverage else "sampled"
    if llm_provider is not None and max_llm_chunks > 0:
        profile = load_book_profile(resolved_id, profiles_root=root)
        chunks = (
            full_coverage_discovery_chunks(
                source_text,
                chunk_chars=llm_chunk_chars,
                max_chunks=max_llm_chunks,
            )
            if llm_full_coverage
            else distributed_discovery_chunks(
                source_text,
                chunk_chars=llm_chunk_chars,
                max_chunks=max_llm_chunks,
            )
        )
        chunk_total = len(chunks)
        coverage_label = "todo el libro" if llm_full_coverage else "muestras del libro"
        _emit_progress(
            progress_callback,
            "llm_discovery",
            f"Analizando {coverage_label}: {chunk_total} fragmento(s) con {model or provider_name or 'LLM'}.",
            42,
            chunk_total=chunk_total,
            model=model,
            coverage_mode=coverage_mode,
        )
        for chunk_index, chunk in enumerate(chunks, start=1):
            if len(chunk.strip()) < _MIN_CHUNK_CHARS and len(source_text) > _MIN_CHUNK_CHARS:
                continue
            start_progress = 42 + int(((chunk_index - 1) / max(1, chunk_total)) * 45)
            _emit_progress(
                progress_callback,
                "llm_discovery",
                f"DeepSeek Flash lee fragmento {chunk_index}/{chunk_total}.",
                start_progress,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                model=model,
                coverage_mode=coverage_mode,
            )
            system, user = build_glossary_discovery_prompt(
                chunk,
                profile=profile,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                coverage_mode=coverage_mode,
                goal_rules=goal_rules,
            )
            try:
                response = await await_llm_call(
                    llm_provider.generate,
                    user,
                    provider=llm_provider,
                    system_prompt=system,
                )
                raw = getattr(response, "content", "") if response is not None else ""
                payload = parse_glossary_discovery_payload(raw)
            except Exception as exc:
                warning = (
                    f"Discovery chunk {chunk_index}/{chunk_total} failed; "
                    f"profile preparation continued with available signals: {exc}"
                )
                warnings.append(warning)
                _emit_progress(
                    progress_callback,
                    "llm_discovery",
                    (
                        f"Fragmento {chunk_index}/{chunk_total} fallo durante lectura LLM; "
                        "se conserva advertencia y se continua."
                    ),
                    start_progress + 1,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    warning=True,
                    coverage_mode=coverage_mode,
                )
                continue
            llm_chunks += 1
            if not payload.get("valid_schema"):
                try:
                    payload, fully_recovered = await _retry_discovery_chunk_in_halves(
                        llm_provider,
                        chunk,
                        profile=profile,
                        chunk_index=chunk_index,
                        chunk_total=chunk_total,
                        coverage_mode=coverage_mode,
                        goal_rules=goal_rules,
                    )
                    if fully_recovered:
                        _emit_progress(
                            progress_callback,
                            "llm_discovery",
                            f"Fragmento {chunk_index}/{chunk_total}: respuesta recuperada en dos mitades acotadas.",
                            start_progress + 1,
                            chunk_index=chunk_index,
                            chunk_total=chunk_total,
                            json_repaired=True,
                            split_retry=True,
                            coverage_mode=coverage_mode,
                        )
                    else:
                        warnings.append(
                            f"Discovery split retry was incomplete for chunk {chunk_index}/{chunk_total}."
                        )
                except Exception as exc:
                    warnings.append(
                        f"Discovery split retry failed for chunk {chunk_index}/{chunk_total}: {exc}"
                    )
            parsed = payload["suggestions"]
            editorial_map = merge_llm_editorial_map(
                editorial_map,
                payload.get("editorial_map"),
                profile_id=resolved_id,
                chunk_index=chunk_index,
            )
            if not payload.get("valid_schema"):
                warnings.append(f"No valid glossary JSON returned for discovery chunk {llm_chunks}.")
                _emit_progress(
                    progress_callback,
                    "llm_discovery",
                    f"Fragmento {chunk_index}/{chunk_total} no devolvio JSON valido; se conserva advertencia.",
                    start_progress + 1,
                    chunk_index=chunk_index,
                    chunk_total=chunk_total,
                    warning=True,
                )
                continue
            rejected = 0
            for item in parsed:
                data = _normalise_llm_suggestion(
                    item,
                    profile_id=resolved_id,
                    chunk_text=chunk,
                    seen_sources=seen_sources,
                    source_language=source_language,
                    goal_rules=goal_rules,
                )
                if data is None:
                    rejected += 1
                    continue
                llm_suggestions.append(data)
                seen_sources.add(data["source"].casefold())
            _emit_progress(
                progress_callback,
                "llm_discovery",
                f"Fragmento {chunk_index}/{chunk_total} listo; {len(llm_suggestions)} sugerencias utiles acumuladas.",
                42 + int((chunk_index / max(1, chunk_total)) * 45),
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                llm_suggestions=len(llm_suggestions),
                rejected_suggestions=rejected,
                coverage_mode=coverage_mode,
            )
        pending = _dedupe_entries([*pending, *llm_suggestions])
    else:
        _emit_progress(
            progress_callback,
            "llm_discovery_skipped",
            "Sin descubrimiento LLM; se usaran solo candidatos locales.",
            86,
        )

    editorial_map = enrich_editorial_map_with_reviewed_terms(
        editorial_map,
        reviewed_candidates=reviewed_candidates,
        approved_entries=approved_entries,
        pending_suggestions=pending,
        profile_id=resolved_id,
        target_locale=target_locale,
        transform_mode=transform_mode,
    )
    write_editorial_artifacts(profile_dir, editorial_map)
    signal_index = build_editorial_signal_index(
        text=source_text,
        profile_id=resolved_id,
        source_name=source_name,
        language=language,
        target_locale=target_locale,
        transform_mode=transform_mode,
        local_candidates=local_candidates,
        reviewed_candidates=reviewed_candidates,
        approved_entries=approved_entries,
        pending_suggestions=pending,
        llm_suggestions=llm_suggestions,
        llm_chunks=llm_chunks,
        coverage_mode=coverage_mode,
        warnings=warnings,
    )
    write_editorial_signal_index(profile_dir, signal_index)
    artifact_counts = editorial_artifact_counts(editorial_map)
    _emit_progress(
        progress_callback,
        "editorial_artifacts_saved",
        "Editorial map and signal index saved for the active profile.",
        92,
        editorial_artifacts=artifact_counts,
        signal_index={
            "local_candidates": signal_index.get("local_candidates", {}).get("total", 0),
            "reviewed_candidates": signal_index.get("reviewed_candidates", {}).get("total", 0),
            "risk_flags": signal_index.get("risk_flags", []),
        },
    )

    pending_path = merge_pending_suggestions(resolved_id, pending)
    pending_count = _count_entries(pending_path)
    _emit_progress(
        progress_callback,
        "profile_saved",
        f"Perfil guardado automaticamente con {pending_count} sugerencias pendientes.",
        96,
        pending_suggestions=pending_count,
    )

    result = PreparedProfileResult(
        profile_id=resolved_id,
        profile_name=resolved_name,
        profile_path=profile_dir,
        text_chars=len(source_text),
        approved_entries=len(approved_entries),
        pending_suggestions=pending_count,
        local_candidates=len(local_candidates),
        llm_suggestions=len(llm_suggestions),
        llm_chunks=llm_chunks,
        coverage_mode=coverage_mode,
        max_local_terms=max_local_terms,
        llm_chunk_chars=llm_chunk_chars,
        provider=provider_name,
        model=model,
        review_provider=term_review_provider_name,
        review_model=term_review_model,
        editorial_artifact_counts=artifact_counts,
        term_review=term_review,
        profile_goal=goal_rules.key,
        profile_goal_label=goal_rules.label,
        business_limits={
            "max_local_terms": max_local_terms,
            "review_terms_limit": goal_rules.review_terms_limit(
                text_chars=len(source_text),
                candidate_count=len(reviewed_candidates),
            ),
            "approved_entries_limit": goal_rules.approved_limit(
                text_chars=len(source_text),
                candidate_count=len(reviewed_candidates),
            ),
            "pending_suggestions_limit": goal_rules.pending_limit(
                text_chars=len(source_text),
                candidate_count=len(reviewed_candidates),
            ),
        },
        warnings=tuple(_dedupe(warnings)),
    )
    _emit_progress(
        progress_callback,
        "completed",
        "Perfil editorial preparado, guardado y listo para seleccionarse.",
        100,
        profile_id=resolved_id,
        profile_goal=goal_rules.key,
        approved_entries=result.approved_entries,
        pending_suggestions=result.pending_suggestions,
        **term_review,
    )
    return result


async def _retry_discovery_chunk_in_halves(
    llm_provider: Any,
    chunk: str,
    *,
    profile: Any,
    chunk_index: int,
    chunk_total: int,
    coverage_mode: str,
    goal_rules: ProfileGoalRules,
) -> tuple[dict[str, Any], bool]:
    """Retry one malformed discovery window as two bounded source halves."""
    halves = _split_discovery_retry_text(chunk)
    combined: dict[str, Any] = {
        "suggestions": [],
        "editorial_map": {},
        "valid_schema": False,
    }
    valid_halves = 0
    for half_index, half in enumerate(halves, start=1):
        system, user = build_glossary_discovery_prompt(
            half,
            profile=profile,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            coverage_mode=f"{coverage_mode}:retry-{half_index}/{len(halves)}",
            goal_rules=goal_rules,
        )
        response = await await_llm_call(
            llm_provider.generate,
            user,
            provider=llm_provider,
            system_prompt=system,
        )
        raw = getattr(response, "content", "") if response is not None else ""
        payload = parse_glossary_discovery_payload(raw)
        if not payload.get("valid_schema"):
            continue
        valid_halves += 1
        combined["suggestions"].extend(payload["suggestions"])
        for key, items in (payload.get("editorial_map") or {}).items():
            if not isinstance(items, list):
                continue
            combined["editorial_map"].setdefault(key, []).extend(
                dict(item) for item in items if isinstance(item, Mapping)
            )
    combined["valid_schema"] = valid_halves == len(halves)
    return combined, bool(combined["valid_schema"])


def _split_discovery_retry_text(text: str) -> tuple[str, str]:
    value = str(text or "").strip()
    midpoint = len(value) // 2
    lower = max(1, int(len(value) * 0.35))
    upper = min(len(value) - 1, int(len(value) * 0.65))
    boundaries: list[int] = []
    for marker in ("\n\n", ". ", "! ", "? ", "; "):
        start = lower
        while True:
            position = value.find(marker, start, upper)
            if position < 0:
                break
            boundaries.append(position + len(marker))
            start = position + len(marker)
    cut = min(boundaries, key=lambda position: abs(position - midpoint)) if boundaries else midpoint
    return value[:cut].strip(), value[cut:].strip()


def distributed_discovery_chunks(
    text: str,
    *,
    chunk_chars: int = 3500,
    max_chunks: int = 8,
) -> list[str]:
    """Return evenly distributed chunks for cheap preflight discovery."""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []
    chunk_chars = max(_MIN_CHUNK_CHARS, int(chunk_chars or 3500))
    max_chunks = max(1, min(int(max_chunks or 8), 50))
    if len(clean) <= chunk_chars:
        return [clean]

    count = min(max_chunks, max(1, len(clean) // chunk_chars))
    if count == 1:
        return [clean[:chunk_chars]]
    stride = (len(clean) - chunk_chars) / (count - 1)
    chunks: list[str] = []
    for index in range(count):
        start = int(round(index * stride))
        end = min(len(clean), start + chunk_chars)
        if start > 0:
            ws = clean.rfind(" ", max(0, start - 100), start + 1)
            if ws != -1:
                start = ws + 1
        if end < len(clean):
            ws = clean.find(" ", end, min(len(clean), end + 100))
            if ws != -1:
                end = ws
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
    return chunks or [clean[:chunk_chars]]


def full_coverage_discovery_chunks(
    text: str,
    *,
    chunk_chars: int = 12000,
    max_chunks: int = _MAX_FULL_DISCOVERY_CHUNKS,
) -> list[str]:
    """Split the whole readable text into ordered LLM discovery windows."""
    clean = re.sub(r"\s+", " ", text or "").strip()
    if not clean:
        return []

    max_chunks = max(1, min(int(max_chunks or _MAX_FULL_DISCOVERY_CHUNKS), _MAX_FULL_DISCOVERY_CHUNKS))
    chunk_chars = max(_MIN_CHUNK_CHARS, int(chunk_chars or 12000))
    if len(clean) <= chunk_chars:
        return [clean]

    chunk_chars = max(chunk_chars, math.ceil(len(clean) / max_chunks))
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        target_end = min(len(clean), start + chunk_chars)
        end = target_end
        if target_end < len(clean):
            min_end = min(len(clean), start + _MIN_CHUNK_CHARS)
            boundary = clean.rfind(". ", min_end, target_end)
            if boundary == -1:
                boundary = clean.rfind("; ", min_end, target_end)
            if boundary != -1:
                end = boundary + 1
            else:
                ws = clean.rfind(" ", min_end, target_end)
                if ws != -1:
                    end = ws
        if end <= start:
            end = target_end
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
        while start < len(clean) and clean[start].isspace():
            start += 1
    return chunks


def _profile_id_from_source(source_name: str) -> str:
    stem = Path(source_name or "book").stem
    base = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    if not base:
        base = "book"
    return f"auto_{base[:46]}"


def _profile_name_from_source(source_name: str, profile_id: str) -> str:
    stem = Path(source_name or "").stem.strip()
    if stem:
        return f"Perfil automatico - {stem[:80]}"
    return profile_id.replace("_", " ").title()


def _existing_generated_profile_for_source(
    profiles_root: Path,
    *,
    source_name: str,
    requested_profile_id: str = "",
) -> Any | None:
    source_key = _source_match_key(source_name)
    requested = str(requested_profile_id or "").strip()
    for profile_path in sorted(profiles_root.glob("*/profile.yml")):
        profile_id = profile_path.parent.name
        if profile_id.startswith("_") or profile_id == "common":
            continue
        if requested and profile_id != requested:
            continue
        try:
            profile = load_book_profile(profile_id, profiles_root=profiles_root, allow_missing=True)
        except Exception:
            continue
        if profile is None:
            continue
        if not (profile.raw_config.get("generated_profile") is True or profile.profile_id.startswith("auto_")):
            continue
        existing_key = _source_match_key(str(profile.raw_config.get("source_name") or ""))
        if requested or (source_key and existing_key == source_key):
            return profile
    return None


def _source_match_key(value: str) -> str:
    decoded = unquote(str(value or ""))
    stem = Path(decoded).stem or decoded
    return re.sub(r"[^a-z0-9]+", " ", stem.casefold()).strip()


def _unique_profile_id(base: str, *, profiles_root: Path) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", (base or "auto_book")).strip("_-").lower()
    safe = safe[:60] or "auto_book"
    if not profile_exists(safe, profiles_root=profiles_root):
        return safe
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    candidate = f"{safe[:46]}_{stamp}"
    if not profile_exists(candidate, profiles_root=profiles_root):
        return candidate
    suffix = 2
    while profile_exists(f"{candidate}_{suffix}", profiles_root=profiles_root):
        suffix += 1
    return f"{candidate}_{suffix}"


def _write_generated_profile_config(
    profile_dir: Path,
    *,
    profile_id: str,
    profile_name: str,
    source_name: str,
    source_language: str,
    language: str,
    target_locale: str,
    transform_mode: str,
    goal_rules: ProfileGoalRules,
    provider_name: str,
    model: str,
    term_review_provider_name: str = "",
    term_review_model: str = "",
) -> None:
    path = profile_dir / "profile.yml"
    data = _read_yaml_mapping(path)
    is_audiobook = str(transform_mode or "").strip().lower() == "audiobook"
    data.update({
        "profile_id": profile_id,
        "name": profile_name,
        "editorial_mode": "book_profile",
        "target_locale": target_locale or _locale_for_language(language),
        "modernization_strength": "high" if transform_mode == "modernize" else "medium",
        "preserve_author_voice": True,
        "allow_common_glossary": True,
        "allow_cross_profile_glossary": False,
        "loaded_glossaries": ["common"],
        "use_profile_glossary": True,
        "glossary_suggestions_enabled": True,
        "auto_approve_glossary_suggestions": False,
        "generated_profile": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name,
        "source_language": source_language or "Auto",
        "target_language": language,
        "profile_goal": goal_rules.key,
        "business_rules": goal_rules.to_config(),
        "preflight_provider": provider_name,
        "preflight_model": model,
        "term_review_provider": term_review_provider_name,
        "term_review_model": term_review_model,
    })
    if is_audiobook:
        data.update({
            "audit_dimensions": "translation_audiobook_full",
            "max_repair_rounds": 1,
            "audiobook": {
                "enabled": True,
                "generate_companion": True,
                "companion_formats": ["txt", "epub_when_main_epub"],
                "notes_policy": "appendix",
                "captions_policy": "integrate_informative",
                "preserve_source_fidelity": True,
            },
        })
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _write_generated_policy(
    profile_dir: Path,
    *,
    profile_name: str,
    source_name: str,
    source_language: str,
    language: str,
    target_locale: str,
    transform_mode: str,
    goal_rules: ProfileGoalRules,
) -> None:
    target = target_locale or _locale_for_language(language) or language or "target locale"
    mode = transform_mode or "transform"
    audiobook_section = ""
    if str(transform_mode or "").strip().lower() == "audiobook":
        audiobook_section = """

## Politica de audiolibro

- Conservar traduccion fiel como prioridad absoluta.
- Crear companion limpio para escucha al final del proceso.
- Mover notas, referencias, URLs y creditos visuales a un apendice cuando interrumpen la narracion.
- Integrar pies de imagen solo cuando aportan informacion sustantiva al texto.
- No inventar descripciones visuales ni convertir creditos de imagen en prosa editorial.
"""
    policy = f"""# {profile_name}

Perfil editorial generado antes del proceso principal a partir de `{source_name}`.

## Objetivo

- Usar este perfil solo para esta obra.
- Objetivo de negocio: {goal_rules.label} (`{goal_rules.key}`).
- Meta del glosario: {goal_rules.glossary_goal}
- Mantener nombres, relaciones, escenas, datos y continuidad interna.
- Aplicar el glosario aprobado de esta obra antes de iniciar el proceso principal.
- Conservar como sugerencias pendientes las decisiones que requieren revision editorial.
- No convertir sugerencias de esta obra en reglas globales.

## Reglas de negocio

{chr(10).join(f"- {rule}" for rule in goal_rules.reviewer_rules)}

## Configuracion

- Idioma fuente: {source_language or "auto"}
- Idioma destino: {language or "auto"}
- Locale objetivo: {target}
- Proceso previsto: {mode}
- Modelo recomendado para descubrimiento previo: deepseek-v4-flash
- Modelo recomendado para revision/clasificacion del glosario: deepseek-v4-pro
- Modelo recomendado para el proceso principal: deepseek-v4-pro
{audiobook_section}
"""
    (profile_dir / "editorial_policy.md").write_text(policy, encoding="utf-8")


def _safe_approved_entries(
    candidates: Iterable[Mapping[str, Any]],
    *,
    profile_id: str,
    auto_approve: bool,
    transform_mode: str = "",
    target_locale: str = "",
    goal_rules: ProfileGoalRules | None = None,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    if not auto_approve:
        return []
    rules = goal_rules or resolve_profile_goal(transform_mode, transform_mode=transform_mode)
    limit = max(1, int(max_entries or _MAX_AUTO_APPROVED_LOCAL_ENTRIES))
    approved: list[dict[str, Any]] = []
    for item in candidates:
        reviewed_entry = reviewed_candidate_to_approved_entry(item, profile_id=profile_id)
        if reviewed_entry is not None:
            approved.append(reviewed_entry)
            continue

        source = str(item.get("source") or "").strip()
        if not source:
            continue
        category = str(item.get("category") or "other").strip().lower()
        keep_source = bool(item.get("keep_source")) or not str(item.get("target") or "").strip()
        needs_translation = bool(item.get("needs_translation"))
        review_status = str(item.get("review_status") or "").strip().lower()
        if review_status and review_status != "preserve_exact":
            continue
        if needs_translation:
            continue
        confidence = _as_float(item.get("confidence"), default=0.0)
        occurrences = _as_int(item.get("occurrences"), default=0)
        if not _safe_to_preserve_without_review(
            source,
            category,
            confidence=confidence,
            occurrences=occurrences,
        ):
            continue
        if category not in set(rules.preserve_categories):
            continue
        if _needs_canonical_review_before_preserve(
            source,
            category,
            transform_mode=transform_mode,
            target_locale=target_locale,
        ):
            continue
        if not keep_source or confidence < _MIN_AUTO_APPROVE_CONFIDENCE:
            continue
        approved.append({
            "source": source,
            "target": source,
            "type": "proper_noun" if category != "acronym" else "acronym",
            "scope": profile_id,
            "status": "approved",
            "confidence": round(confidence, 3),
            "occurrences": occurrences,
            "mechanical_safe": True,
            "translation_policy": "preserve_exact",
            "injection_policy": "preserve",
            "review_status": "preserve_exact",
            "review_confidence": round(confidence, 3),
            "reviewed_by": "legacy_safe_preserve_gate",
            "rationale": "Preflight scan identified this as a recurring named item to preserve consistently for this profile.",
            "examples": _examples_from_candidate(item),
        })
    approved = _dedupe_entries(approved)
    approved.sort(key=_local_entry_priority, reverse=True)
    return approved[:limit]


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


def _pending_from_local_candidates(
    candidates: Iterable[Mapping[str, Any]],
    *,
    profile_id: str,
    approved_sources: set[str],
    goal_rules: ProfileGoalRules | None = None,
    max_entries: int | None = None,
) -> list[dict[str, Any]]:
    rules = goal_rules or resolve_profile_goal("")
    limit = max(1, int(max_entries or _MAX_LOCAL_PENDING_SUGGESTIONS))
    pending: list[dict[str, Any]] = []
    for item in candidates:
        source = str(item.get("source") or "").strip()
        if not source or source.casefold() in approved_sources:
            continue
        category = str(item.get("category") or "other").strip().lower()
        if not _glossary_source_quality_ok(source, category):
            continue
        if str(item.get("review_status") or "").strip().lower() == "reject_noise":
            continue
        reviewed_pending = review_to_pending_suggestion(item, profile_id=profile_id)
        if reviewed_pending is not None:
            pending.append(reviewed_pending)
            continue
        confidence = _as_float(item.get("confidence"), default=0.0)
        occurrences = _as_int(item.get("occurrences"), default=0)
        if not _useful_local_pending_candidate(
            source,
            category,
            confidence=confidence,
            occurrences=occurrences,
        ):
            continue
        target = str(item.get("target") or "").strip()
        pending.append({
            "source": source,
            "suggested_target": target if target != source else "",
            "type": _suggestion_type_for_category(category),
            "scope": profile_id,
            "status": "pending",
            "profile_goal": rules.key,
            "confidence": round(confidence, 3),
            "occurrences": occurrences,
            "rationale": "Detected by the full-document local preflight scan; review before approving for this profile.",
            "examples": _examples_from_candidate(item),
            "risks": ["May require context-specific handling."],
        })
    pending = _dedupe_entries(pending)
    pending.sort(key=_local_entry_priority, reverse=True)
    return pending[:limit]


def _safe_to_preserve_without_review(
    source: str,
    category: str,
    *,
    confidence: float = 0.0,
    occurrences: int = 0,
) -> bool:
    if not _glossary_source_quality_ok(source, category):
        return False
    if confidence < _MIN_AUTO_APPROVE_CONFIDENCE:
        return False
    if category == "acronym":
        return occurrences >= 2
    words = source.split()
    if len(words) >= 2:
        if occurrences < _MIN_MULTIWORD_AUTO_APPROVE_OCCURRENCES:
            return False
        if words[0].casefold() in _SAFE_SINGLE_WORD_STOPWORDS:
            return False
        return True
    if not words:
        return False
    folded = source.casefold()
    if folded in _TREATMENT_WORDS:
        return False
    if occurrences < _MIN_SINGLE_WORD_AUTO_APPROVE_OCCURRENCES:
        return False
    if source.isupper():
        return True
    # Diacritics or unusual casing are stronger evidence for names in Spanish
    # and multilingual books; plain sentence starters stay pending.
    if any(ch in source for ch in "ÁÉÍÓÚÜÑáéíóúüñ"):
        return True
    if any(ch.isupper() for ch in source[1:]):
        return True
    if category in {"character", "location", "organization"} and len(source) >= 8 and occurrences >= 6:
        return True
    return False


def _useful_local_pending_candidate(
    source: str,
    category: str,
    *,
    confidence: float,
    occurrences: int,
) -> bool:
    if not _glossary_source_quality_ok(source, category):
        return False
    words = source.split()
    if confidence < _MIN_PENDING_LOCAL_CONFIDENCE and not _strong_named_or_term_signal(source, category):
        return False
    if len(words) == 1:
        folded = source.casefold()
        if folded in _TREATMENT_WORDS:
            return False
        if category in {"character", "location", "organization", "title"}:
            return occurrences >= 2 and _strong_named_or_term_signal(source, category)
        if category in {"technical", "concept"}:
            return occurrences >= 2 and (confidence >= 0.55 or _strong_named_or_term_signal(source, category))
    return True


def _glossary_source_quality_ok(source: str, category: str = "") -> bool:
    source = re.sub(r"\s+", " ", (source or "").strip())
    if len(source) < 2 or len(source) > _MAX_LLM_TERM_CHARS:
        return False
    words = source.split()
    if not words or len(words) > _MAX_LLM_TERM_WORDS:
        return False
    folded = source.casefold()
    if category == "address_form" and folded in _TREATMENT_WORDS:
        return True
    if folded in _SAFE_SINGLE_WORD_STOPWORDS or folded in _GENERIC_CAPITALIZED_WORDS:
        return False
    if category == "acronym" or source.isupper():
        upper = source.upper()
        if upper in _ACRONYM_NOISE or _ROMAN_NUMERAL_RE.fullmatch(upper):
            return False
    if len(words) == 1 and words[0].casefold() in _TREATMENT_WORDS:
        return False
    if words[0].casefold() in _SAFE_SINGLE_WORD_STOPWORDS:
        return False
    if _STRUCTURAL_ID_RE.fullmatch(source):
        return False
    if re.search(r"[\uFFFD�□■▪●◆]", source):
        return False
    if source.count(".") + source.count("?") + source.count("!") > 1:
        return False
    if len(words) > 4 and source.endswith((".", "?", "!", ";", ":")):
        return False
    if len(words) == 1 and not re.search(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", source):
        return False
    return True


def _strong_named_or_term_signal(source: str, category: str = "") -> bool:
    words = source.split()
    if not words:
        return False
    if source.isupper() and len(source) >= 3:
        return True
    if len(words) >= 2:
        return any(token[:1].isupper() for token in words)
    if any(ch in source for ch in "ÁÉÍÓÚÜÑáéíóúüñ"):
        return True
    if any(ch.isupper() for ch in source[1:]):
        return True
    if "-" in source or "'" in source or "’" in source or "*" in source or "\\" in source:
        return True
    if category in {"technical", "concept", "acronym"} and any(ch.isdigit() for ch in source):
        return True
    return category in {"character", "location", "organization"} and len(source) >= 8


def _local_entry_priority(entry: Mapping[str, Any]) -> tuple[float, int, int, int]:
    source = str(entry.get("source") or "")
    entry_type = str(entry.get("type") or entry.get("category") or "").lower()
    type_weight = {
        "acronym": 5,
        "proper_noun": 4,
        "character": 4,
        "location": 4,
        "organization": 4,
        "technical_term": 3,
        "technical": 3,
        "concept": 2,
        "title": 1,
    }.get(entry_type, 0)
    return (
        _as_float(entry.get("confidence"), default=0.0),
        _as_int(entry.get("occurrences"), default=0),
        type_weight,
        min(len(source), 80),
    )


def _normalise_llm_suggestion(
    item: Mapping[str, Any],
    *,
    profile_id: str,
    chunk_text: str,
    seen_sources: set[str],
    source_language: str = "",
    goal_rules: ProfileGoalRules | None = None,
) -> Optional[dict[str, Any]]:
    rules = goal_rules or resolve_profile_goal("")
    source = re.sub(r"\s+", " ", str(item.get("source") or "").strip(" \t\r\n\"'“”‘’"))
    entry_type = str(item.get("type") or item.get("entry_type") or "term").strip().lower()
    if entry_type not in _USEFUL_LLM_TYPES:
        entry_type = "term"
    if not _is_useful_glossary_source(source, chunk_text, entry_type=entry_type):
        return None
    key = source.casefold()
    if key in seen_sources:
        return None

    target = _short_field(item.get("suggested_target") or item.get("target"), max_chars=_MAX_LLM_TERM_CHARS)
    options = [
        value for value in (
            _short_field(option, max_chars=_MAX_LLM_TERM_CHARS)
            for option in _as_list(item.get("target_options"))
        )
        if value
    ][:6]
    examples = _clean_examples(item.get("examples"))
    confidence = max(0.0, min(1.0, _as_float(item.get("confidence"), default=0.0)))

    return {
        "source": source,
        "suggested_target": target,
        "target_options": options,
        "type": entry_type,
        "scope": profile_id,
        "status": "pending",
        "profile_goal": rules.key,
        "source_language": source_language,
        "confidence": round(confidence, 3),
        "rationale": _short_field(item.get("rationale"), max_chars=320),
        "examples": examples,
        "risks": _short_list(item.get("risks"), max_items=4, max_chars=180),
        "do_not_apply_if": _short_list(item.get("do_not_apply_if"), max_items=4, max_chars=180),
    }


def _is_useful_glossary_source(source: str, chunk_text: str, *, entry_type: str = "") -> bool:
    if not _glossary_source_quality_ok(source, entry_type):
        return False
    words = source.split()
    if "\n" in source or "\r" in source:
        return False
    if source.casefold() == (chunk_text or "").strip().casefold():
        return False
    if len(words) == 1:
        if entry_type in {"address_form", "treatment_error"}:
            return True
        if not _strong_named_or_term_signal(source, entry_type):
            return False
    return bool(re.search(r"(?<!\w)" + re.escape(source) + r"(?!\w)", chunk_text or "", re.I))


def _short_field(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text or len(text) > max_chars:
        return ""
    return text


def _short_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        text = _short_field(item, max_chars=max_chars)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _clean_examples(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    examples: list[dict[str, str]] = []
    for raw in value[:2]:
        if not isinstance(raw, Mapping):
            continue
        source_excerpt = _short_field(raw.get("source_excerpt"), max_chars=_MAX_EXAMPLE_CHARS)
        target_excerpt = _short_field(raw.get("target_excerpt"), max_chars=_MAX_EXAMPLE_CHARS)
        recommended = _short_field(raw.get("recommended_modernization"), max_chars=_MAX_EXAMPLE_CHARS)
        item: dict[str, str] = {}
        if source_excerpt:
            item["source_excerpt"] = source_excerpt
        if target_excerpt:
            item["target_excerpt"] = target_excerpt
        if recommended:
            item["recommended_modernization"] = recommended
        if item:
            examples.append(item)
    return examples


def _write_glossary_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    existing = _read_yaml_mapping(path)
    raw = [
        item for item in (existing.get("entries") or [])
        if isinstance(item, Mapping)
    ]
    seen = {str(item.get("source") or "").casefold() for item in raw}
    for entry in entries:
        source = str(entry.get("source") or "").strip()
        if source and source.casefold() not in seen:
            raw.append(entry)
            seen.add(source.casefold())
    path.write_text(
        yaml.safe_dump({"entries": raw}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _examples_from_candidate(item: Mapping[str, Any]) -> list[dict[str, str]]:
    contexts = item.get("contexts") or []
    examples: list[dict[str, str]] = []
    if isinstance(contexts, list):
        for context in contexts[:2]:
            text = str(context).strip()
            if text:
                examples.append({"source_excerpt": text})
    return examples


def _suggestion_type_for_category(category: str) -> str:
    return {
        "character": "proper_noun",
        "location": "proper_noun",
        "organization": "proper_noun",
        "title": "title",
        "acronym": "acronym",
        "technical": "technical_term",
        "concept": "concept",
        "item": "item",
    }.get(category, "term")


def _locale_for_language(language: str) -> str:
    folded = (language or "").strip().casefold()
    if folded in {"spanish", "es", "espanol", "español"}:
        return "es-MX"
    return ""


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    data = yaml.safe_load(text) if text.strip() else {}
    return dict(data) if isinstance(data, Mapping) else {}


def _count_entries(path: Path) -> int:
    payload = _read_yaml_mapping(path)
    items = payload.get("suggestions") or payload.get("entries") or []
    return len([item for item in items if isinstance(item, Mapping)])


def _dedupe_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_source: dict[str, dict[str, Any]] = {}
    for entry in entries:
        source = str(entry.get("source") or "").strip()
        if not source:
            continue
        key = source.casefold()
        existing = by_source.get(key)
        if existing is not None:
            _merge_duplicate_entry(existing, entry)
            continue
        copied = dict(entry)
        out.append(copied)
        by_source[key] = copied
    return out


def _merge_duplicate_entry(
    existing: dict[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    """Merge richer discovery evidence without replacing an earlier decision."""
    for key in (
        "suggested_target",
        "target",
        "type",
        "scope",
        "profile_goal",
        "review_status",
        "review_target",
        "review_rationale",
        "reviewed_by",
    ):
        if not str(existing.get(key) or "").strip() and str(incoming.get(key) or "").strip():
            existing[key] = incoming[key]

    for key in ("confidence", "review_confidence", "occurrences"):
        if key not in incoming:
            continue
        current = _as_float(existing.get(key), default=0.0)
        candidate = _as_float(incoming.get(key), default=0.0)
        if candidate > current:
            existing[key] = incoming[key]

    for key in ("target_options", "examples", "risks", "do_not_apply_if"):
        merged: list[Any] = []
        seen: set[str] = set()
        for value in [*(existing.get(key) or []), *(incoming.get(key) or [])]:
            identity = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, Mapping) else str(value)
            if identity in seen:
                continue
            merged.append(value)
            seen.add(identity)
        if merged:
            existing[key] = merged

    current_rationale = str(existing.get("rationale") or "").strip()
    incoming_rationale = str(incoming.get("rationale") or "").strip()
    if incoming_rationale and (
        not current_rationale
        or current_rationale.startswith("Detected by the full-document local preflight")
    ):
        existing["rationale"] = incoming_rationale


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _emit_progress(
    progress_callback: Callable[[dict[str, Any]], None] | None,
    stage: str,
    message: str,
    progress: int | float,
    **extra: Any,
) -> None:
    if progress_callback is None:
        return
    event: dict[str, Any] = {
        "stage": stage,
        "message": message,
        "progress": max(0, min(100, int(progress))),
    }
    event.update(extra)
    try:
        progress_callback(event)
    except Exception:
        pass
