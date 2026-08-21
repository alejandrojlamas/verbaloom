"""Deterministic final hygiene pass for generated artifacts.

This module is deliberately outside the LLM pipeline.  It does not translate,
refine, re-score, or ask a model for anything; it only removes mechanical
source/pagination artifacts that are safe to identify after reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import stat
import tempfile
import unicodedata
import zipfile

from lxml import etree

from src.core.book_profiles.rendering import (
    apply_profile_exact_translation_corrections,
    profile_exact_translation_pairs,
)
from src.core.llm_output_guard import guard_llm_output
from src.utils.text_encoding import UTF8_BOM, clean_text_artifacts, encode_utf8_text_download
from src.utils.text_encoding import mojibake_score
from src.utils.proper_names import extract_symbol_bearing_names, restore_symbol_bearing_names


_TEXT_EXTENSIONS = {".txt", ".text", ".md", ".markdown", ".log", ".srt"}
_HTML_EXTENSIONS = {".html", ".htm", ".xhtml"}
_EPUB_BODY_EXTENSIONS = (".xhtml", ".html", ".htm")
_EPUB_TEXT_EXTENSIONS = (".xhtml", ".html", ".htm", ".opf", ".ncx", ".xml", ".txt")
_SANITIZED_ARTIFACT_CLASS = "verbaloom-sanitized-artifact"

_SOURCE_DOMAIN_RE = re.compile(
    r"(?i)(?<![a-z0-9])(?:oceanofpdf\.com|z-library\.sk|1lib\.sk|z-lib\.sk)(?![a-z0-9])"
)
_SOURCE_ARTIFACT_DOMAINS = {
    "1lib.sk",
    "oceanofpdf.com",
    "z-lib.sk",
    "z-library.sk",
}
_DOMAIN_TOKEN_RE = re.compile(
    r"(?i)^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$"
)
_MARKDOWN_DOMAIN_LINK_RE = re.compile(
    r"^\s*\[([^\]\n]+)\]\((https?://[^)\s]+|www\.[^)\s]+)\)\s*$",
    re.IGNORECASE,
)
_BRACKETED_DOMAIN_RE = re.compile(r"^\s*\[([^\]\n]+)\]\s*$")
_PAREN_URL_RE = re.compile(r"^\s*\((https?://[^)\s]+|www\.[^)\s]+)\)\s*$", re.IGNORECASE)
_URL_RE = re.compile(r"^\s*(https?://[^\s)]+|www\.[^\s)]+)\s*$", re.IGNORECASE)
_NUMERIC_LABEL_RE = re.compile(r"^\s*\d{1,5}\.?\s*$")
_CHAPTER_FILE_RE = re.compile(r"chap-(\d+)\.(?:xhtml|html|htm)$", re.IGNORECASE)
_SOURCE_TITLE_SEP_RE = re.compile(
    r"(?i)(?:^|[_\s-]+)(?:oceanofpdf\.com|z-library\.sk|1lib\.sk|z-lib\.sk)(?:[_\s-]+|$)"
)
_PROTOCOL_ARTIFACT_HINT_RE = re.compile(
    r"(?i)(?:<|&lt;)\s*/?\s*(?:translation(?:ation)*|source|target|input|output|"
    r"fidelity_audit_json|editorial_guard_json)(?:\s+[^>]*?)?\s*(?:>|&gt;)|"
    r"\b(?:INPUT_TAG_IN|INPUT_TAG_OUT|TRANSLATE_TAG_IN|TRANSLATE_TAG_OUT|"
    r"BEGIN_(?:SOURCE|INPUT|TRANSLATION)|END_(?:SOURCE|INPUT|TRANSLATION))\b|"
    r"^\s*(?:system|user|assistant|developer)\s*:|"
    r"^\s*(?:(?:descripci[oó]n\s+de\s+(?:la\s+)?imagen|image\s+description)\s*:\s*)+",
    re.MULTILINE,
)
_IMAGE_DESCRIPTION_LABEL_RE = re.compile(
    r"(?im)^\s*(?:(?:descripci[oó]n\s+de\s+(?:la\s+)?imagen|image\s+description)\s*:\s*)+"
)
_DIALOGUE_DASH_RE = re.compile(r"(?m)^\s*—")
_DIALOGUE_STRAIGHT_QUOTE_RE = re.compile(r'(?m)^\s*"')
_DIALOGUE_CURLY_QUOTE_RE = re.compile(r"(?m)^\s*[“”]")
_DIALOGUE_GUILLEMET_RE = re.compile(r"(?m)^\s*«")
_ORPHAN_DASH_AFTER_ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)\s*[—–-](?=\s|$)")
_DUPLICATE_PERIOD_RE = re.compile(r'\."\."|(?<!\.)\.\s*\.(?!\.)')
_BAD_DASH_ATTACHMENT_RE = re.compile(r"(?:[—–-][,.;:!?]|[,.;:!?][—–-])")
_MID_SENTENCE_PARAGRAPH_SPLIT_RE = re.compile(
    r"[a-záéíóúüñ,;:]\s*\n\s*\n\s+[a-záéíóúüñ¿¡]",
    re.IGNORECASE,
)
_MARKDOWN_LINK_RE = re.compile(r"\[[^\]\n]{1,120}\]\((?:https?://|\.\./)[^)]+\)", re.IGNORECASE)
_FOOTNOTE_LINK_RE = re.compile(
    r"\[\[?\d{1,4}\]?\]\((?:\.\./|https?://)[^)]+\)|"
    r"\(?\.\./[^\s)]*(?:notas?|notes?)\.xhtml#[^\s)]+?\)?",
    re.IGNORECASE,
)
_INLINE_VISIBLE_URL_RE = re.compile(r"(?i)\bhttps?://[^\s<>)]+")
_DOT_LEADER_TOC_RE = re.compile(r"\.{4,}(?:\s*\d{1,5}\b)?")
_STANDALONE_PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d{1,5}\.?\s*$")
_MISSING_SENTENCE_SPACE_RE = re.compile(
    r"(?<=[a-záéíóúüñ0-9])([.!?])(?=[A-ZÁÉÍÓÚÜÑ¿¡])",
)
_LEADING_PUNCTUATION_SPACE_RE = re.compile(
    r"^([.!?])(?=[A-Za-zÁÉÍÓÚÜÑáéíóúüñ¿¡])"
)
_ROMAN_SENTENCE_SPACE_RE = re.compile(r"\b([IVXLCDM]{2,})\.(?=[a-záéíóúüñ])")
_CHAPTER_HEADING_RE = re.compile(
    r"(?im)^\s*(?:cap[ií]tulo|chapter|secci[oó]n|parte|book)\b.{0,120}$"
)
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-ZÁÉÍÓÚÜÑ][A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’.-]{4,}\b")
_NAME_VARIANT_STOP_WORDS = {
    "aun",
    "como",
    "cuando",
    "donde",
    "esta",
    "hacia",
    "porque",
    "quien",
    "solo",
}
_READER_NAVIGATION_LABELS = {
    "spanish": {
        "go to note reference in text": "IR A LA NOTA EN EL TEXTO",
        "go to note reference": "IR A LA NOTA EN EL TEXTO",
        "return to note reference": "VOLVER A LA NOTA EN EL TEXTO",
        "back to note reference": "VOLVER A LA NOTA EN EL TEXTO",
        "back to text": "VOLVER AL TEXTO",
    },
}
_LANGUAGE_ALIASES = {
    "es": "spanish",
    "español": "spanish",
    "espanol": "spanish",
}


@dataclass
class FinalArtifactAuditReport:
    output_path: str
    output_format: str
    scanned_files: int = 0
    source_artifacts_removed: int = 0
    text_rewrites: int = 0
    reader_artifact_labels_removed: int = 0
    numeric_toc_labels_rewritten: int = 0
    numeric_titles_rewritten: int = 0
    readable_samples_checked: int = 0
    prompt_protocol_findings: int = 0
    mojibake_score: int = 0
    link_artifact_findings: int = 0
    footnote_link_findings: int = 0
    pagination_artifact_findings: int = 0
    toc_artifact_findings: int = 0
    chapter_heading_count: int = 0
    possible_name_variant_groups: int = 0
    paragraph_continuations_reflowed: int = 0
    paragraph_continuations_dehyphenated: int = 0
    dom_boundary_repairs: int = 0
    dom_boundary_findings: int = 0
    symbol_names_restored: int = 0
    exact_glossary_terms_repaired: int = 0
    reading_quality_warnings: list[str] = field(default_factory=list)
    unresolved_findings: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(
            self.source_artifacts_removed
            or self.text_rewrites
            or self.reader_artifact_labels_removed
            or self.numeric_toc_labels_rewritten
            or self.numeric_titles_rewritten
            or self.paragraph_continuations_reflowed
            or self.dom_boundary_repairs
            or self.symbol_names_restored
            or self.exact_glossary_terms_repaired
        )

    @property
    def clean(self) -> bool:
        return not self.unresolved_findings and not self.reading_quality_warnings

    def summary(self) -> str:
        parts = []
        if self.source_artifacts_removed:
            parts.append(f"{self.source_artifacts_removed} artefactos de fuente/paginacion")
        if self.text_rewrites:
            parts.append(f"{self.text_rewrites} textos internos")
        if self.reader_artifact_labels_removed:
            parts.append(f"{self.reader_artifact_labels_removed} etiquetas de artefacto lector")
        if self.prompt_protocol_findings:
            parts.append(f"{self.prompt_protocol_findings} fugas de protocolo")
        if self.link_artifact_findings or self.footnote_link_findings:
            parts.append(f"{self.link_artifact_findings + self.footnote_link_findings} links visibles")
        if self.pagination_artifact_findings or self.toc_artifact_findings:
            parts.append(f"{self.pagination_artifact_findings + self.toc_artifact_findings} artefactos de indice/pagina")
        if self.numeric_toc_labels_rewritten or self.numeric_titles_rewritten:
            parts.append(
                f"{self.numeric_toc_labels_rewritten + self.numeric_titles_rewritten} etiquetas de indice/titulo"
            )
        if self.paragraph_continuations_reflowed:
            parts.append(
                f"{self.paragraph_continuations_reflowed} continuaciones de parrafo"
            )
        if self.dom_boundary_repairs:
            parts.append(f"{self.dom_boundary_repairs} fronteras DOM")
        if self.symbol_names_restored:
            parts.append(f"{self.symbol_names_restored} nombres con simbolos")
        if self.exact_glossary_terms_repaired:
            parts.append(f"{self.exact_glossary_terms_repaired} terminos exactos de perfil")
        if self.reading_quality_warnings:
            parts.append(f"{len(self.reading_quality_warnings)} alertas de lectura")
        if self.unresolved_findings:
            parts.append(f"{len(self.unresolved_findings)} advertencias")
        return ", ".join(parts) if parts else "sin artefactos"

    def to_markdown(self) -> str:
        lines = [
            "# Auditoria final de artefactos",
            "",
            f"- Archivo: {Path(self.output_path).name}",
            f"- Formato: {self.output_format.upper()}",
            f"- Archivos internos revisados: {self.scanned_files}",
            f"- Artefactos de fuente/paginacion eliminados: {self.source_artifacts_removed}",
            f"- Textos internos limpiados: {self.text_rewrites}",
            f"- Etiquetas de artefacto lector eliminadas: {self.reader_artifact_labels_removed}",
            f"- Fugas de protocolo LLM detectadas: {self.prompt_protocol_findings}",
            f"- Links visibles sospechosos: {self.link_artifact_findings}",
            f"- Links de nota/pie visibles sospechosos: {self.footnote_link_findings}",
            f"- Artefactos de paginacion: {self.pagination_artifact_findings}",
            f"- Ruido de indice/TOC: {self.toc_artifact_findings}",
            f"- Etiquetas numericas de indice corregidas: {self.numeric_toc_labels_rewritten}",
            f"- Titulos internos numericos corregidos: {self.numeric_titles_rewritten}",
            f"- Encabezados de capitulo/seccion detectados: {self.chapter_heading_count}",
            f"- Grupos posibles de variantes de nombre: {self.possible_name_variant_groups}",
            f"- Continuaciones de parrafo recompuestas: {self.paragraph_continuations_reflowed}",
            f"- Palabras partidas recompuestas: {self.paragraph_continuations_dehyphenated}",
            f"- Fronteras DOM reparadas desde fuente: {self.dom_boundary_repairs}",
            f"- Fronteras DOM pendientes: {self.dom_boundary_findings}",
            f"- Nombres con simbolos restaurados: {self.symbol_names_restored}",
            f"- Terminos exactos de perfil reparados: {self.exact_glossary_terms_repaired}",
            f"- Muestras de lectura revisadas: {self.readable_samples_checked}",
            f"- Mojibake score: {self.mojibake_score}",
            f"- Estado: {'limpio' if self.clean else 'con advertencias'}",
        ]
        if self.reading_quality_warnings:
            lines.extend(["", "## Alertas de lectura"])
            lines.extend(f"- {item}" for item in self.reading_quality_warnings)
        if self.unresolved_findings:
            lines.extend(["", "## Advertencias"])
            lines.extend(f"- {item}" for item in self.unresolved_findings)
        return "\n".join(lines).strip() + "\n"

    def write(self, path: str | Path | None = None) -> Path:
        report_path = Path(path) if path else final_artifact_report_path(self.output_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


def final_artifact_report_path(output_filepath: str | Path) -> Path:
    output = Path(output_filepath)
    return output.with_name(f"{output.stem} - auditoria final de artefactos.md")


def audit_and_clean_final_artifact(
    output_filepath: str | Path,
    *,
    output_format: str | None = None,
    write_report: bool = True,
    source_epub_path: str | Path | None = None,
    prompt_options: dict | None = None,
    target_language: str = "",
) -> FinalArtifactAuditReport:
    """Scan and safely clean a completed output artifact in place."""
    path = Path(output_filepath)
    fmt = _resolve_format(path, output_format)
    report = FinalArtifactAuditReport(str(path), fmt)
    if not path.exists():
        report.unresolved_findings.append("El archivo final no existe.")
        if write_report:
            report.write()
        return report

    if fmt == "epub":
        report.exact_glossary_terms_repaired += _repair_epub_exact_profile_terms(
            path,
            prompt_options,
        )
        if source_epub_path and Path(source_epub_path).exists():
            from src.core.epub.dom_boundaries import repair_epub_dom_boundaries
            from src.core.epub.page_furniture import sanitize_epub_page_furniture
            from src.core.epub.paragraph_reflow import repair_epub_split_paragraphs

            report.symbol_names_restored += _repair_epub_symbol_bearing_names(
                Path(source_epub_path),
                path,
            )
            reflow_report = repair_epub_split_paragraphs(source_epub_path, path)
            report.paragraph_continuations_reflowed += reflow_report.merged_continuations
            report.paragraph_continuations_dehyphenated += (
                reflow_report.dehyphenated_continuations
            )
            report.unresolved_findings.extend(reflow_report.errors)
            furniture_report = sanitize_epub_page_furniture(source_epub_path, path)
            report.source_artifacts_removed += furniture_report.sanitized_blocks
            report.unresolved_findings.extend(furniture_report.errors)
            boundary_report = repair_epub_dom_boundaries(source_epub_path, path)
            report.dom_boundary_repairs += boundary_report.repaired_boundaries
            report.dom_boundary_findings += len(boundary_report.findings)
            report.unresolved_findings.extend(boundary_report.structural_mismatches)
        _clean_epub(path, report, target_language=target_language)
    elif fmt in {"txt", "srt", "md", "markdown"} or path.suffix.lower() in _TEXT_EXTENSIONS:
        _clean_text_file(path, report)
    elif path.suffix.lower() in _HTML_EXTENSIONS:
        _clean_html_file(path, report)
    elif fmt in {"pdf", "docx"}:
        _audit_readable_binary(path, fmt, report)

    _audit_readability_samples(path, fmt, report)

    if write_report and (report.changed or report.unresolved_findings or report.reading_quality_warnings):
        report.write()
    return report


def _resolve_format(path: Path, output_format: str | None) -> str:
    fmt = (output_format or "").strip().lower().lstrip(".")
    if fmt and fmt != "auto":
        return fmt
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "txt"


def _clean_text_file(path: Path, report: FinalArtifactAuditReport) -> None:
    raw = path.read_bytes()
    had_bom = raw.startswith(UTF8_BOM)
    before = raw.decode("utf-8-sig", errors="replace")
    after = clean_text_artifacts(before)
    guard = guard_llm_output(after, phase="final_artifact")
    after = guard.text
    report.reader_artifact_labels_removed += int(guard.scores.get("reader_artifact_labels_removed", 0.0))
    if guard.issues:
        report.prompt_protocol_findings += len(guard.issues)
    report.scanned_files = 1
    if after != before:
        report.source_artifacts_removed += max(1, len(before.splitlines()) - len(after.splitlines()))
        if had_bom:
            path.write_bytes(encode_utf8_text_download(after))
        else:
            path.write_text(after, encoding="utf-8")


def _clean_html_file(path: Path, report: FinalArtifactAuditReport) -> None:
    data = path.read_bytes()
    cleaned, removed, rewrites, _toc, _titles = _clean_xhtml_payload(data, path.name, False)
    report.scanned_files = 1
    report.source_artifacts_removed += removed
    report.text_rewrites += rewrites
    if cleaned != data:
        path.write_bytes(cleaned)


def _audit_readable_binary(path: Path, fmt: str, report: FinalArtifactAuditReport) -> None:
    report.scanned_files = 1
    try:
        from src.core.output_formats import extract_readable_text

        text = extract_readable_text(path)
    except Exception as exc:  # pragma: no cover - defensive
        report.unresolved_findings.append(f"No se pudo auditar {fmt.upper()}: {exc}")
        return
    if _has_source_artifacts(text):
        report.unresolved_findings.append(
            f"{fmt.upper()} contiene posibles artefactos; requiere regeneracion desde texto limpio."
        )


def _audit_readability_samples(path: Path, fmt: str, report: FinalArtifactAuditReport) -> None:
    """Run a bounded final-text audit over beginning/middle/end samples."""
    try:
        from src.core.output_formats import extract_readable_text

        text = extract_readable_text(path)
    except Exception as exc:  # pragma: no cover - defensive
        report.unresolved_findings.append(f"Could not run final readability audit for {fmt.upper()}: {exc}")
        return

    text = text or ""
    if not text.strip():
        report.unresolved_findings.append(
            f"{fmt.upper()} no contiene texto legible para la auditoria final."
        )
        return
    structure = _reading_structure_findings(text)
    report.link_artifact_findings = structure["link_artifact_findings"]
    report.footnote_link_findings = structure["footnote_link_findings"]
    report.pagination_artifact_findings = structure["pagination_artifact_findings"]
    report.toc_artifact_findings = structure["toc_artifact_findings"]
    report.chapter_heading_count = structure["chapter_heading_count"]
    report.possible_name_variant_groups = structure["possible_name_variant_groups"]
    for warning in structure["warnings"]:
        if warning not in report.reading_quality_warnings:
            report.reading_quality_warnings.append(warning)

    report.mojibake_score = mojibake_score(text)
    if report.mojibake_score:
        report.reading_quality_warnings.append(
            f"Readable text still contains mojibake markers: score={report.mojibake_score}."
        )
    for warning in _final_text_quality_warnings(text):
        if warning not in report.reading_quality_warnings:
            report.reading_quality_warnings.append(warning)

    samples = _readability_samples(text)
    report.readable_samples_checked = len(samples)
    for index, sample in enumerate(samples, start=1):
        guard = guard_llm_output(sample, phase=f"final_sample_{index}")
        if guard.issues:
            report.prompt_protocol_findings += len(guard.issues)
            report.reading_quality_warnings.append(
                f"Sample {index} contains possible reader-visible LLM protocol: "
                + ", ".join(issue.code for issue in guard.issues[:4])
            )
        if _has_source_artifacts(sample):
            report.reading_quality_warnings.append(
                f"Sample {index} contains possible source-link or pagination artifacts."
            )
        repetition = _repetition_warning(sample)
        if repetition:
            report.reading_quality_warnings.append(f"Sample {index}: {repetition}")


def _clean_epub(
    path: Path,
    report: FinalArtifactAuditReport,
    *,
    target_language: str = "",
) -> None:
    numeric_nav_labels = _epub_numeric_nav_labels(path)
    rewrite_numeric_navigation = _looks_like_page_furniture(numeric_nav_labels)
    changed = False
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}.",
        suffix=".epub",
        dir=str(path.parent),
        delete=False,
    )
    tmp_name = tmp_file.name
    tmp_file.close()

    try:
        with zipfile.ZipFile(path, "r") as zin, zipfile.ZipFile(tmp_name, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                lower = info.filename.lower()
                removed = rewrites = toc_rewrites = title_rewrites = 0
                if lower.endswith(_EPUB_BODY_EXTENSIONS):
                    try:
                        data, removed, rewrites, toc_rewrites, title_rewrites = _clean_xhtml_payload(
                            data,
                            info.filename,
                            rewrite_numeric_navigation,
                            target_language=target_language,
                        )
                    except Exception as exc:
                        report.unresolved_findings.append(f"No se pudo limpiar {info.filename}: {exc}")
                elif lower.endswith((".opf", ".ncx", ".xml")):
                    data, rewrites = _clean_xml_metadata(data)
                if lower.endswith(_EPUB_TEXT_EXTENSIONS):
                    report.scanned_files += 1
                report.source_artifacts_removed += removed
                report.text_rewrites += rewrites
                report.numeric_toc_labels_rewritten += toc_rewrites
                report.numeric_titles_rewritten += title_rewrites
                changed = changed or bool(removed or rewrites or toc_rewrites or title_rewrites)
                _write_zip_member(zout, info, data)
        if changed:
            shutil.move(tmp_name, path)
        else:
            Path(tmp_name).unlink(missing_ok=True)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def _repair_epub_exact_profile_terms(
    output_path: Path,
    prompt_options: dict | None,
) -> int:
    """Apply approved exact profile translations to EPUB text nodes atomically."""
    if not prompt_options:
        return 0
    exact_pairs = profile_exact_translation_pairs(prompt_options)
    if not exact_pairs:
        return 0
    original_mode = stat.S_IMODE(output_path.stat().st_mode)
    replacements = 0
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=".epub",
        dir=str(output_path.parent),
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with (
            zipfile.ZipFile(output_path, "r") as source,
            zipfile.ZipFile(tmp_path, "w") as rebuilt,
        ):
            for info in source.infolist():
                payload = source.read(info.filename)
                if info.filename.lower().endswith(_EPUB_BODY_EXTENSIONS):
                    try:
                        root = etree.fromstring(
                            payload,
                            etree.XMLParser(recover=False, remove_blank_text=False),
                        )
                        local_replacements = 0
                        for element in root.iter():
                            if not isinstance(element.tag, str):
                                continue
                            title_context = _is_published_title_markup(element)
                            if element.text and not title_context:
                                element.text, count = apply_profile_exact_translation_corrections(
                                element.text,
                                prompt_options,
                                exact_pairs=exact_pairs,
                                )
                                local_replacements += count
                            if element.tail:
                                element.tail, count = apply_profile_exact_translation_corrections(
                                element.tail,
                                prompt_options,
                                exact_pairs=exact_pairs,
                                )
                                local_replacements += count
                        if local_replacements:
                            replacements += local_replacements
                            payload = etree.tostring(
                                root,
                                encoding="utf-8",
                                xml_declaration=True,
                                pretty_print=False,
                            )
                    except (etree.XMLSyntaxError, ValueError):
                        pass
                _write_zip_member(rebuilt, info, payload)
        if replacements:
            tmp_path.replace(output_path)
            output_path.chmod(original_mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return replacements


def _is_published_title_markup(element) -> bool:
    """Avoid deterministic glossary rewrites inside citation/title markup."""
    node = element
    while node is not None:
        tag = str(getattr(node, "tag", "") or "")
        local_name = tag.rsplit("}", 1)[-1].lower()
        if local_name in {"cite", "em", "i"}:
            return True
        classes = str(getattr(node, "attrib", {}).get("class") or "").casefold()
        if any(marker in classes for marker in ("book-title", "work-title", "citation-title")):
            return True
        node = node.getparent()
    return False


def _repair_epub_symbol_bearing_names(source_path: Path, output_path: Path) -> int:
    """Restore source-proven punctuation inside proper names atomically.

    This is deliberately source-aware and contains no book vocabulary.  It
    only repairs an unambiguous damaged rendering such as ``ChTril`` when the
    corresponding source XHTML establishes the exact name ``Ch*Tril``.
    """
    original_mode = stat.S_IMODE(output_path.stat().st_mode)
    changed_nodes = 0
    tmp = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}.",
        suffix=".epub",
        dir=str(output_path.parent),
        delete=False,
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        with (
            zipfile.ZipFile(source_path, "r") as source,
            zipfile.ZipFile(output_path, "r") as output,
            zipfile.ZipFile(tmp_path, "w") as rebuilt,
        ):
            source_names = set(source.namelist())
            for info in output.infolist():
                payload = output.read(info.filename)
                if (
                    info.filename in source_names
                    and info.filename.lower().endswith(_EPUB_BODY_EXTENSIONS)
                ):
                    try:
                        source_root = etree.fromstring(
                            source.read(info.filename),
                            etree.XMLParser(recover=True, remove_blank_text=False),
                        )
                        output_root = etree.fromstring(
                            payload,
                            etree.XMLParser(recover=True, remove_blank_text=False),
                        )
                        source_text = " ".join(source_root.itertext())
                        local_changes = 0
                        for element in output_root.iter():
                            if not isinstance(element.tag, str):
                                continue
                            if element.text:
                                repaired = restore_symbol_bearing_names(source_text, element.text)
                                if repaired != element.text:
                                    local_changes += _restored_symbol_occurrences(
                                        source_text,
                                        element.text,
                                        repaired,
                                    )
                                    element.text = repaired
                            if element.tail:
                                repaired = restore_symbol_bearing_names(source_text, element.tail)
                                if repaired != element.tail:
                                    local_changes += _restored_symbol_occurrences(
                                        source_text,
                                        element.tail,
                                        repaired,
                                    )
                                    element.tail = repaired
                        if local_changes:
                            payload = etree.tostring(
                                output_root,
                                encoding="utf-8",
                                xml_declaration=True,
                                pretty_print=False,
                            )
                            changed_nodes += local_changes
                    except Exception:
                        # The strict publication parser will surface malformed
                        # XHTML.  This repair must never make finalization fail.
                        pass
                _write_zip_member(rebuilt, info, payload)
        if changed_nodes:
            tmp_path.replace(output_path)
            output_path.chmod(original_mode)
    finally:
        tmp_path.unlink(missing_ok=True)
    return changed_nodes


def _restored_symbol_occurrences(source_text: str, before: str, after: str) -> int:
    restored = 0
    for exact in extract_symbol_bearing_names(source_text):
        pattern = re.compile(re.escape(exact), re.IGNORECASE)
        restored += max(0, len(pattern.findall(after)) - len(pattern.findall(before)))
    return restored


def _write_zip_member(zout: zipfile.ZipFile, info: zipfile.ZipInfo, data: bytes) -> None:
    new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
    new_info.comment = info.comment
    new_info.extra = info.extra
    new_info.internal_attr = info.internal_attr
    new_info.external_attr = info.external_attr
    new_info.create_system = info.create_system
    new_info.compress_type = zipfile.ZIP_STORED if info.filename == "mimetype" else info.compress_type
    zout.writestr(new_info, data)


def _clean_xhtml_payload(
    data: bytes,
    filename: str,
    rewrite_numeric_navigation: bool,
    *,
    target_language: str = "",
) -> tuple[bytes, int, int, int, int]:
    parser = etree.XMLParser(recover=True, remove_blank_text=False)
    root = etree.fromstring(data, parser=parser)
    removed_nodes = _remove_artifact_elements(root)
    text_rewrites = 0
    toc_rewrites = 0
    title_rewrites = 0

    for elem in root.iter():
        if not isinstance(elem.tag, str):
            continue
        local = etree.QName(elem).localname.lower()
        if local == "title" and _is_numeric_text(_node_text(elem)) and rewrite_numeric_navigation:
            elem.text = _section_label_for_filename(filename)
            title_rewrites += 1
        elif local == "a" and _is_numeric_text(_node_text(elem)) and rewrite_numeric_navigation:
            elem.text = _section_label_for_href(elem.get("href") or "")
            toc_rewrites += 1
        elif local in {"title", "h1"} and elem.text and _SOURCE_DOMAIN_RE.search(elem.text):
            elem.text = _clean_title(elem.text, fallback=_section_label_for_filename(filename))
            text_rewrites += 1

        if elem.text:
            new_text = _localize_reader_navigation_label(
                elem.text,
                target_language=target_language,
            )
            new_text = _clean_inline_text(new_text)
            if new_text != elem.text:
                elem.text = new_text
                text_rewrites += 1
        if elem.tail:
            new_tail = _localize_reader_navigation_label(
                elem.tail,
                target_language=target_language,
            )
            new_tail = _clean_inline_text(new_tail)
            if new_tail != elem.tail:
                elem.tail = new_tail
                text_rewrites += 1

    return (
        etree.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=False),
        removed_nodes,
        text_rewrites,
        toc_rewrites,
        title_rewrites,
    )


def _remove_artifact_elements(root: etree._Element) -> int:
    removed = 0

    def visit(parent: etree._Element) -> None:
        nonlocal removed
        previous_removed_source = False
        for child in list(parent):
            if not isinstance(child.tag, str):
                continue
            text = _node_text(child)
            local = etree.QName(child).localname.lower()
            removable = local in {"p", "div", "span", "h1", "h2", "h3", "h4", "h5", "h6", "li"}
            source_artifact = removable and _is_source_artifact_text(text)
            page_after_source = removable and previous_removed_source and _is_numeric_text(text)
            if source_artifact or page_after_source:
                _clear_artifact_element(child)
                removed += 1
                previous_removed_source = source_artifact
                continue
            visit(child)
            if text.strip():
                previous_removed_source = False

    visit(root)
    return removed


def _clear_artifact_element(element: etree._Element) -> None:
    """Remove reader-visible artifact content without changing the DOM shape."""
    classes = {
        value
        for value in str(element.get("class") or "").split()
        if value
    }
    classes.add(_SANITIZED_ARTIFACT_CLASS)
    element.set("class", " ".join(sorted(classes)))
    existing_style = str(element.get("style") or "").strip().rstrip(";")
    element.set(
        "style",
        f"{existing_style}; display: none".lstrip("; "),
    )
    element.text = ""
    for descendant in element.iterdescendants():
        descendant.text = ""
        descendant.tail = ""
        if etree.QName(descendant).localname.lower() == "a":
            descendant.attrib.pop("href", None)


def _clean_xml_metadata(data: bytes) -> tuple[bytes, int]:
    text = data.decode("utf-8", errors="ignore")
    before = text

    def title_repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{_clean_title(match.group(2), fallback='Libro traducido')}{match.group(3)}"

    text = re.sub(r"(?is)(<dc:title[^>]*>)(.*?)(</dc:title>)", title_repl, text)
    text = re.sub(r"(?is)(<title[^>]*>)(.*?)(</title>)", title_repl, text)
    return text.encode("utf-8"), int(text != before)


def _clean_inline_text(value: str) -> str:
    if not value:
        return value
    leading = re.match(r"^\s*", value).group(0)
    trailing = re.search(r"\s*$", value).group(0)
    core = value.strip()
    if not core:
        return value
    # Inline markup often leaves sentence punctuation in a tail node. In that
    # case whitespace before the punctuation belongs to neither word, while a
    # missing space after it glues the next sentence. Normalize only these
    # mechanical boundaries; do not alter words or punctuation marks.
    if core[:1] in ".,!?;:":
        leading = ""
    repaired_core = re.sub(r"\s+([.,!?;:])", r"\1", core)
    repaired_core = _LEADING_PUNCTUATION_SPACE_RE.sub(r"\1 ", repaired_core)
    repaired_core = _MISSING_SENTENCE_SPACE_RE.sub(r"\1 ", repaired_core)
    repaired_core = _ROMAN_SENTENCE_SPACE_RE.sub(r"\1. ", repaired_core)
    if repaired_core == core and not (_has_source_artifacts(core) or _has_protocol_artifacts(core)):
        return value
    core = repaired_core
    if not (_has_source_artifacts(core) or _has_protocol_artifacts(core)):
        return f"{leading}{core}{trailing}" if core else ""
    cleaned = clean_text_artifacts(core)
    cleaned = guard_llm_output(cleaned, phase="final_artifact_inline").text
    cleaned = _SOURCE_DOMAIN_RE.sub("", cleaned)
    cleaned = re.sub(r"(?i)\(?https?://(?:oceanofpdf\.com|z-library\.sk|1lib\.sk|z-lib\.sk)/?\)?", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return f"{leading}{cleaned}{trailing}" if cleaned else ""


def _localize_reader_navigation_label(
    value: str,
    *,
    target_language: str,
) -> str:
    """Localize mechanical EPUB backlink labels without changing their links."""
    if not value or not target_language:
        return value
    language = _LANGUAGE_ALIASES.get(
        str(target_language).strip().casefold(),
        str(target_language).strip().casefold(),
    )
    labels = _READER_NAVIGATION_LABELS.get(language, {})
    if not labels:
        return value
    leading = re.match(r"^\s*", value).group(0)
    trailing = re.search(r"\s*$", value).group(0)
    localized = labels.get(value.strip().casefold())
    if not localized:
        return value
    return f"{leading}{localized}{trailing}"


def _clean_title(value: str, *, fallback: str) -> str:
    text = clean_text_artifacts(value or "")
    text = _SOURCE_TITLE_SEP_RE.sub(" ", text)
    text = _SOURCE_DOMAIN_RE.sub("", text)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return text or fallback


def _epub_numeric_nav_labels(path: Path) -> list[str]:
    labels: list[str] = []
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            lower = name.lower()
            if "nav" not in lower and "toc" not in lower:
                continue
            if not lower.endswith(_EPUB_BODY_EXTENSIONS):
                continue
            try:
                root = etree.fromstring(zf.read(name), etree.XMLParser(recover=True))
            except Exception:
                continue
            for node in root.xpath("//*[local-name()='nav']//*[local-name()='a' or local-name()='span']"):
                text = _normalize_text("".join(node.itertext()))
                if _is_numeric_text(text):
                    labels.append(text.rstrip("."))
    return labels


def _looks_like_page_furniture(labels: list[str]) -> bool:
    if len(labels) < 8:
        return False
    unique = set(labels)
    duplicate_count = len(labels) - len(unique)
    if duplicate_count >= max(3, int(len(labels) * 0.15)):
        return True
    try:
        values = sorted(int(label) for label in unique)
    except ValueError:
        return False
    contiguous = values == list(range(values[0], values[-1] + 1)) if values else False
    return not contiguous and len(labels) >= 20


def _section_label_for_filename(filename: str) -> str:
    return _section_label(Path(filename).name)


def _section_label_for_href(href: str) -> str:
    return _section_label((href or "").split("#", 1)[0])


def _section_label(value: str) -> str:
    match = _CHAPTER_FILE_RE.search(value or "")
    if not match:
        return "Seccion"
    return f"Seccion {int(match.group(1))}"


def _node_text(elem: etree._Element) -> str:
    return _normalize_text("".join(elem.itertext()))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _is_numeric_text(value: str) -> bool:
    return bool(_NUMERIC_LABEL_RE.match(value or ""))


def _is_source_artifact_text(value: str) -> bool:
    text = _normalize_text(value)
    if not text:
        return False
    if _SOURCE_DOMAIN_RE.search(text):
        return True
    markdown = _MARKDOWN_DOMAIN_LINK_RE.match(text)
    if markdown:
        label_domain = _domain_from_text(markdown.group(1))
        url_domain = _domain_from_text(markdown.group(2))
        return bool(
            label_domain
            and label_domain == url_domain
            and _is_source_artifact_domain(url_domain)
        )
    bracket = _BRACKETED_DOMAIN_RE.match(text)
    if bracket:
        return _is_source_artifact_domain(_domain_from_text(bracket.group(1)))
    url = _PAREN_URL_RE.match(text) or _URL_RE.match(text)
    if url:
        return _is_source_artifact_domain(_domain_from_text(url.group(1)))
    return _is_source_artifact_domain(_domain_from_text(text))


def _has_source_artifacts(value: str) -> bool:
    return bool(
        _SOURCE_DOMAIN_RE.search(value or "")
        or "](http" in (value or "")
        or "](../" in (value or "")
    )


def _has_protocol_artifacts(value: str) -> bool:
    text = value or ""
    if not text:
        return False
    return bool(_PROTOCOL_ARTIFACT_HINT_RE.search(text))


def _final_text_quality_warnings(text: str) -> list[str]:
    warnings: list[str] = []
    value = text or ""
    label_count = len(_IMAGE_DESCRIPTION_LABEL_RE.findall(value))
    if label_count:
        warnings.append(
            f"Readable text still contains {label_count} image-description label(s)."
        )

    dialogue_counts = {
        "dash": len(_DIALOGUE_DASH_RE.findall(value)),
        "straight_quote": len(_DIALOGUE_STRAIGHT_QUOTE_RE.findall(value)),
        "curly_quote": len(_DIALOGUE_CURLY_QUOTE_RE.findall(value)),
        "guillemet": len(_DIALOGUE_GUILLEMET_RE.findall(value)),
    }
    active_dialogue_styles = {
        key: count
        for key, count in dialogue_counts.items()
        if count >= 3
    }
    if len(active_dialogue_styles) >= 2:
        details = ", ".join(f"{key}={count}" for key, count in sorted(active_dialogue_styles.items()))
        warnings.append(f"Mixed dialogue marker conventions detected: {details}.")

    punctuation_hits = {
        "orphan_dash_after_ellipsis": len(_ORPHAN_DASH_AFTER_ELLIPSIS_RE.findall(value)),
        "duplicate_period": len(_DUPLICATE_PERIOD_RE.findall(value)),
        "bad_dash_attachment": len(_BAD_DASH_ATTACHMENT_RE.findall(value)),
    }
    punctuation_total = sum(punctuation_hits.values())
    if punctuation_total:
        details = ", ".join(f"{key}={count}" for key, count in punctuation_hits.items() if count)
        warnings.append(f"TTS-hostile punctuation residue detected: {details}.")

    split_count = len(_MID_SENTENCE_PARAGRAPH_SPLIT_RE.findall(value))
    if split_count:
        warnings.append(f"Possible mid-sentence paragraph split(s) detected: {split_count}.")

    return warnings


def _reading_structure_findings(text: str) -> dict[str, object]:
    value = text or ""
    footnote_spans = _match_spans(_FOOTNOTE_LINK_RE, value)
    markdown_spans = [
        span
        for span in _match_spans(_MARKDOWN_LINK_RE, value)
        if not _span_overlaps_any(span, footnote_spans)
    ]
    covered_link_spans = footnote_spans + markdown_spans
    inline_url_spans = [
        span
        for span in _match_spans(_INLINE_VISIBLE_URL_RE, value)
        if not _span_overlaps_any(span, covered_link_spans)
    ]
    visible_links = len(markdown_spans) + len(inline_url_spans)
    footnote_links = len(footnote_spans)
    standalone_pages = len(_STANDALONE_PAGE_NUMBER_RE.findall(value))
    dot_leaders = len(_DOT_LEADER_TOC_RE.findall(value))
    chapter_headings = len(_CHAPTER_HEADING_RE.findall(value))
    name_variant_groups = _name_variant_groups(value)
    warnings: list[str] = []
    if visible_links:
        warnings.append(
            f"Visible markdown/URL links remain in readable text: {visible_links}."
        )
    if footnote_links:
        warnings.append(f"Reader-visible note/footnote links remain in readable text: {footnote_links}.")
    if standalone_pages >= 3:
        warnings.append(f"Possible standalone pagination lines remain: {standalone_pages}.")
    if dot_leaders:
        warnings.append(f"Possible table-of-contents dot leader noise remains: {dot_leaders}.")
    if chapter_headings == 0 and len(value) > 40000:
        warnings.append("No chapter/section headings detected in a long final artifact.")
    if name_variant_groups:
        sample = "; ".join(
            f"{key}: {', '.join(values[:4])}"
            for key, values in list(name_variant_groups.items())[:4]
        )
        warnings.append(f"Possible proper-name spelling/accent variants detected: {sample}.")
    return {
        "link_artifact_findings": visible_links,
        "footnote_link_findings": footnote_links,
        "pagination_artifact_findings": standalone_pages,
        "toc_artifact_findings": dot_leaders,
        "chapter_heading_count": chapter_headings,
        "possible_name_variant_groups": len(name_variant_groups),
        "warnings": warnings,
    }


def _name_variant_groups(text: str) -> dict[str, list[str]]:
    variants: dict[str, dict[str, int]] = {}
    non_initial_variants: dict[str, set[str]] = {}
    value = text or ""
    for match in _CAPITALIZED_TOKEN_RE.finditer(value):
        token = match.group(0)
        normalized = _strip_accents(token).casefold().strip(".")
        if len(normalized) < 5 or normalized in _NAME_VARIANT_STOP_WORDS:
            continue
        bucket = variants.setdefault(normalized, {})
        bucket[token] = bucket.get(token, 0) + 1
        sentence_initial = _is_sentence_initial_position(value, match.start())
        if not sentence_initial:
            non_initial_variants.setdefault(normalized, set()).add(token)
    return {
        key: sorted(values.keys())
        for key, values in variants.items()
        if len(values) >= 2 and non_initial_variants.get(key)
    }


def _is_sentence_initial_position(text: str, position: int) -> bool:
    """Check the immediate left boundary without rescanning the whole book.

    The previous implementation sliced and regex-searched every prefix before
    every capitalized token. That makes the final readability audit quadratic
    on long books. Walking only the whitespace directly adjacent to the token
    preserves the same boundary semantics while keeping the full scan linear.
    """
    value = text or ""
    cursor = min(max(0, int(position)), len(value)) - 1
    if cursor < 0:
        return True

    saw_whitespace = False
    if value[cursor] in {'"', "'", "«", "“", "("}:
        cursor -= 1
    while cursor >= 0 and value[cursor].isspace():
        saw_whitespace = True
        if value[cursor] == "\n":
            return True
        cursor -= 1
    if cursor < 0:
        return True
    if not saw_whitespace:
        return False
    if value[cursor] in {'"', "'", "»", "”", ")"}:
        cursor -= 1
    return cursor >= 0 and value[cursor] in ".!?"


def _match_spans(pattern: re.Pattern[str], text: str) -> list[tuple[int, int]]:
    return [match.span() for match in pattern.finditer(text or "")]


def _span_overlaps_any(span: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in others)


def _strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value or "")
        if not unicodedata.combining(char)
    )


def _readability_samples(text: str, *, max_samples: int = 5, sample_chars: int = 2200) -> list[str]:
    compact = (text or "").strip()
    if not compact:
        return []
    if len(compact) <= sample_chars:
        return [compact]
    anchors = [0, len(compact) // 4, len(compact) // 2, (len(compact) * 3) // 4, max(0, len(compact) - sample_chars)]
    samples: list[str] = []
    seen: set[tuple[int, int]] = set()
    for anchor in anchors[:max_samples]:
        start = max(0, min(anchor, len(compact) - sample_chars))
        end = min(len(compact), start + sample_chars)
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        samples.append(compact[start:end])
    return samples


def _repetition_warning(sample: str) -> str:
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in re.split(r"\n\s*\n+", sample or "")
        if paragraph.strip()
    ]
    if len(paragraphs) < 4:
        return ""
    counts: dict[str, int] = {}
    for paragraph in paragraphs:
        key = paragraph[:240].casefold()
        counts[key] = counts.get(key, 0) + 1
    repeated = max(counts.values(), default=0)
    if repeated >= 3:
        return "Repeated paragraph pattern detected in final assembled text."
    return ""


def _domain_from_text(value: str) -> str:
    raw = re.sub(r"^\s*https?://", "", value or "", flags=re.IGNORECASE)
    raw = re.sub(r"^\s*www\.", "", raw, flags=re.IGNORECASE)
    raw = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip().strip(").,;:_")
    return raw.casefold() if _DOMAIN_TOKEN_RE.match(raw) else ""


def _is_source_artifact_domain(domain: str) -> bool:
    return str(domain or "").casefold() in _SOURCE_ARTIFACT_DOMAINS
