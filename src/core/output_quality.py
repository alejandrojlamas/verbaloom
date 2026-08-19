"""Local quality report for generated downloadable files."""

from __future__ import annotations

from pathlib import Path
import posixpath
import re
from typing import Any
from urllib.parse import unquote
import zipfile

from lxml import etree

from src.core.llm_output_guard import guard_llm_output
from src.core.locale_quality import (
    collect_mexican_spanish_issue_examples,
    count_mexican_spanish_issues,
    mexican_spanish_issue_catalog,
)
from src.core.output_formats import (
    _epub_spine_xhtml_names,
    _find_epub_opf_path,
    _is_navigation_xhtml_name,
    extract_readable_text,
)
from src.utils.text_encoding import mojibake_score


_WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+)?")
_SOURCE_ARTIFACT_RE = re.compile(
    r"(?i)(?:oceanofpdf\.com|z-library\.sk|1lib\.sk|z-lib\.sk|"
    r"\[\s*\d{1,5}\s*\]\s*\((?:\.\./|[^)]*(?:notes?|notas?|footnotes?|endnotes?)[^)]*)\)|"
    r"https?://(?:oceanofpdf\.com|z-library\.sk|1lib\.sk|z-lib\.sk))"
)
_STANDALONE_PAGE_MARKER_RE = re.compile(r"(?m)^\s*\d{1,5}\.?\s*$")


def analyze_output_file(path: str | Path) -> dict[str, Any]:
    """Build a bounded, local-only report for a generated output artifact."""
    file_path = Path(path)
    suffix = file_path.suffix.lower().lstrip(".") or "unknown"
    warnings: list[str] = []
    errors: list[str] = []

    text = _extract_text_for_quality(file_path, warnings, errors)
    counts = count_mexican_spanish_issues(text)
    examples = collect_mexican_spanish_issue_examples(text, max_per_type=5, max_total=24)
    catalog = mexican_spanish_issue_catalog()
    total_regional = sum(counts.values())
    mojibake = mojibake_score(text)

    glossary_suggestions = [
        {
            "code": code,
            "count": counts.get(code, 0),
            "examples": examples.get(code, []),
            "label": catalog.get(code, {}).get("label", code),
            "category": catalog.get(code, {}).get("category", "regional_register"),
            "suggestion": catalog.get(code, {}).get(
                "suggestion",
                "Revisar y adaptar al español editorial mexicano por contexto.",
            ),
        }
        for code in sorted(counts)
        if counts.get(code, 0) > 0
    ]

    epub_report = _inspect_epub(file_path) if suffix == "epub" else None
    if epub_report:
        warnings.extend(epub_report.get("warnings", []))
        errors.extend(epub_report.get("errors", []))

    final_readability = _final_readability_report(text)
    warnings.extend(final_readability.get("warnings", []))
    reader_artifacts = _reader_artifact_report(text, file_path=file_path)
    warnings.extend(reader_artifacts.get("warnings", []))

    if total_regional:
        warnings.append("El texto contiene formas alejadas del español editorial mexicano.")
    if mojibake:
        errors.append("El texto contiene señales de mojibake o codificación corrupta.")
    if not text.strip():
        errors.append("No se pudo extraer texto legible para auditar el archivo.")

    status = "pass"
    if errors or total_regional:
        status = "fail"
    elif warnings:
        status = "warn"

    return {
        "filename": file_path.name,
        "file_type": suffix,
        "status": status,
        "text": {
            "characters": len(text),
            "words": len(_WORD_RE.findall(text)),
            "paragraphs": len(_paragraphs(text)),
        },
        "mexican_spanish": {
            "total": total_regional,
            "counts": counts,
            "examples": examples,
        },
        "glossary_suggestions": glossary_suggestions,
        "mojibake_score": mojibake,
        "final_readability": final_readability,
        "reader_artifacts": reader_artifacts,
        "epub": epub_report,
        "warnings": _dedupe(warnings),
        "errors": _dedupe(errors),
    }


def _extract_text_for_quality(file_path: Path, warnings: list[str], errors: list[str]) -> str:
    try:
        return extract_readable_text(file_path)
    except Exception as exc:
        errors.append(f"No se pudo extraer texto: {exc}")
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as fallback_exc:
            warnings.append(f"Lectura UTF-8 de respaldo falló: {fallback_exc}")
            return ""


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]


def _final_readability_report(text: str) -> dict[str, Any]:
    samples = _readability_samples(text)
    warnings: list[str] = []
    protocol_issues = 0
    for index, sample in enumerate(samples, start=1):
        guard = guard_llm_output(sample, phase=f"quality_sample_{index}")
        if guard.issues:
            protocol_issues += len(guard.issues)
            warnings.append(
                f"Sample {index} contains possible reader-visible LLM protocol: "
                + ", ".join(issue.code for issue in guard.issues[:4])
            )
        repetition = _repetition_warning(sample)
        if repetition:
            warnings.append(f"Sample {index}: {repetition}")
    return {
        "samples_checked": len(samples),
        "protocol_issues": protocol_issues,
        "warnings": _dedupe(warnings),
    }


def _reader_artifact_report(
    text: str,
    *,
    file_path: Path | None = None,
) -> dict[str, Any]:
    """Find reader-visible debris that should not ship in final downloads."""
    warnings: list[str] = []
    source_artifacts = len(_SOURCE_ARTIFACT_RE.findall(text or ""))
    page_markers = _standalone_page_marker_count(text, file_path=file_path)
    if source_artifacts:
        warnings.append(
            f"El texto contiene {source_artifacts} link(s), watermarks o marcadores de nota internos visibles."
        )
    if page_markers >= 3:
        warnings.append(
            f"El texto contiene {page_markers} posibles marcadores de paginación aislados."
        )
    return {
        "source_artifacts": source_artifacts,
        "standalone_page_markers": page_markers,
        "warnings": _dedupe(warnings),
    }


def _standalone_page_marker_count(
    text: str,
    *,
    file_path: Path | None = None,
) -> int:
    if file_path is not None and file_path.suffix.lower() == ".epub":
        structured_count = _epub_standalone_page_marker_count(file_path)
        if structured_count is not None:
            return structured_count
    count = 0
    for match in _STANDALONE_PAGE_MARKER_RE.finditer(text or ""):
        value = match.group(0).strip().rstrip(".")
        if not value:
            continue
        try:
            number = int(value)
        except ValueError:
            continue
        # Four-digit years are often legitimate section headings in history,
        # biography and fiction.  Treating them as page furniture creates
        # noisy publication warnings and can hide real isolated page numbers.
        if len(value) == 4 and 1000 <= number <= 2100:
            continue
        if 1 <= number <= 5000:
            count += 1
    return count


def _epub_standalone_page_marker_count(file_path: Path) -> int | None:
    """Count numeric furniture without mistaking chapters, years or notes for pages."""
    try:
        with zipfile.ZipFile(file_path) as archive:
            opf_path = _find_epub_opf_path(archive)
            if not opf_path:
                return None
            content_names = _epub_spine_xhtml_names(archive, opf_path)
            count = 0
            for name in content_names:
                root = etree.fromstring(archive.read(name))
                for node in root.xpath("//*[local-name()='body']//*[not(*)]"):
                    value = " ".join("".join(node.itertext()).split()).rstrip(".")
                    if not value.isdigit():
                        continue
                    number = int(value)
                    if not 1 <= number <= 5000:
                        continue
                    if len(value) == 4 and 1000 <= number <= 2100:
                        continue
                    lineage = [node, *node.iterancestors()]
                    local_names = {
                        etree.QName(item).localname.lower()
                        for item in lineage
                        if isinstance(item.tag, str)
                    }
                    classes = {
                        class_name
                        for item in lineage
                        for class_name in str(item.get("class") or "").split()
                    }
                    if local_names & {"h1", "h2", "h3", "h4", "h5", "h6", "sup"}:
                        continue
                    if classes & {"tbl-section-marker", "footnote", "endnote", "noteref"}:
                        continue
                    count += 1
            return count
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, etree.XMLSyntaxError):
        return None


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
    if max(counts.values(), default=0) >= 3:
        return "Repeated paragraph pattern detected in final assembled text."
    return ""


def _inspect_epub(file_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "valid_zip": False,
        "opf_path": None,
        "has_nav": False,
        "has_opf": False,
        "title": "",
        "language": "",
        "content_documents": 0,
        "spine_items": 0,
        "warnings": [],
        "errors": [],
    }
    if not zipfile.is_zipfile(file_path):
        report["errors"].append("El EPUB no es un ZIP válido.")
        return report

    report["valid_zip"] = True
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            lower_names = {name.lower(): name for name in names}
            report["has_nav"] = any(
                lower.endswith(("nav.xhtml", "nav.html", "toc.ncx"))
                for lower in lower_names
            )
            report["content_documents"] = sum(
                1
                for name in names
                if name.lower().endswith((".xhtml", ".html", ".htm"))
                and not _is_navigation_xhtml_name(name.lower())
            )

            opf_path = _find_epub_opf_path(zf)
            if opf_path:
                report["opf_path"] = opf_path
                report["has_opf"] = True
                report.update(_inspect_opf(zf, opf_path))
                spine = _epub_spine_xhtml_names(zf, opf_path)
                report["spine_items"] = len(spine)
                if spine and not all(_safe_epub_name(name) in names for name in spine):
                    report["warnings"].append("El spine contiene referencias que no existen en el paquete.")
            else:
                report["errors"].append("El EPUB no contiene paquete OPF localizable.")

            if not report["has_nav"]:
                report["warnings"].append("El EPUB no tiene navegación/TOC reconocible.")
            if not report["content_documents"]:
                report["errors"].append("El EPUB no contiene documentos de lectura.")
            if not report["spine_items"]:
                report["warnings"].append("El EPUB no declara capítulos en spine.")
            if not report["title"]:
                report["warnings"].append("El EPUB no declara título en metadatos.")
            if not report["language"]:
                report["warnings"].append("El EPUB no declara idioma en metadatos.")
    except Exception as exc:
        report["errors"].append(f"No se pudo inspeccionar el EPUB: {exc}")
    return report


def _inspect_opf(zf: zipfile.ZipFile, opf_path: str) -> dict[str, Any]:
    details = {"title": "", "language": ""}
    try:
        root = etree.fromstring(zf.read(opf_path), etree.XMLParser(recover=True, huge_tree=True))
    except Exception:
        return details

    titles = root.xpath("//*[local-name()='metadata']/*[local-name()='title']/text()")
    languages = root.xpath("//*[local-name()='metadata']/*[local-name()='language']/text()")
    if titles:
        details["title"] = _compact_text(titles[0])
    if languages:
        details["language"] = _compact_text(languages[0])
    nav_items = root.xpath("//*[local-name()='manifest']/*[contains(concat(' ', @properties, ' '), ' nav ')]")
    if nav_items:
        details["has_nav"] = True
    return details


def _safe_epub_name(name: str) -> str:
    return posixpath.normpath(unquote(name)).lstrip("/")


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result
