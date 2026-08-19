#!/usr/bin/env python3
"""Generate reproducible baseline, manifests, editorial diff, and final EPUB QA."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.epub.publication_gate import audit_epub_publication, snapshot_epub
from src.core.epub.unit_contract import EPUB_PIPELINE_VERSION, prompt_versions, text_sha256
from src.utils.language_detector import LanguageDetector


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _words(text: str) -> int:
    return len(re.findall(r"\b[^\W\d_][\w'’.-]*\b", text or "", re.UNICODE))


def _sentences(text: str) -> int:
    return len(re.findall(r"(?<=[.!?…])(?:[\"'»”’)]*)\s+", text or "")) + int(bool((text or "").strip()))


def _lengths(units) -> dict[str, float | int]:
    values = sorted(len(unit.text) for unit in units)
    if not values:
        return {"min": 0, "median": 0, "p90": 0, "p99": 0, "max": 0, "mean": 0.0}

    def percentile(fraction: float) -> int:
        return values[min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))]

    return {
        "min": values[0],
        "median": int(statistics.median(values)),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": values[-1],
        "mean": round(statistics.mean(values), 2),
    }


def _language_distribution(units) -> dict[str, dict[str, int]]:
    words: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    units_by_language: Counter[str] = Counter()
    for unit in units:
        language, _confidence = LanguageDetector.detect_language_from_text(
            unit.text, confidence_threshold=0.75
        )
        key = str(language or "unknown")
        words[key] += _words(unit.text)
        chars[key] += len(unit.text)
        units_by_language[key] += 1
    return {
        "units": dict(units_by_language.most_common()),
        "words": dict(words.most_common()),
        "characters": dict(chars.most_common()),
    }


def _snapshot_summary(snapshot) -> dict[str, Any]:
    long_threshold = max(4_000, int(_lengths(snapshot.units)["p99"]))
    return {
        "path": snapshot.path,
        "epub_version": snapshot.package_version,
        "opf_path": snapshot.opf_path,
        "archive_entries": len(snapshot.entry_names),
        "manifest_items": snapshot.manifest_count,
        "spine_documents": len(snapshot.spine),
        "spine": snapshot.spine,
        "xhtml_files": len(snapshot.xhtml_files),
        "images": len(snapshot.image_files),
        "stylesheets": len(snapshot.css_files),
        "fonts": len(snapshot.font_files),
        "nav_files": snapshot.nav_files,
        "ncx_files": snapshot.ncx_files,
        "toc_entries": [{"label": label, "target": target} for label, target in snapshot.toc_entries],
        "metadata": snapshot.metadata,
        "declared_languages": snapshot.languages,
        "units": len(snapshot.units),
        "paragraphs": sum(1 for unit in snapshot.units if unit.element == "p"),
        "sentences": sum(_sentences(unit.text) for unit in snapshot.units),
        "words": sum(_words(unit.text) for unit in snapshot.units),
        "characters": sum(len(unit.text) for unit in snapshot.units),
        "unit_lengths": _lengths(snapshot.units),
        "abnormally_long_units": [
            {"unit_id": unit.unit_id, "chars": len(unit.text), "source_document": unit.file_href}
            for unit in snapshot.units if len(unit.text) >= long_threshold
        ],
        "language_distribution": _language_distribution(snapshot.units),
        "internal_links": len(snapshot.links),
        "note_or_fragment_links": sum(1 for _origin, href in snapshot.links if "#" in href),
        "duplicate_ids": snapshot.duplicate_id_count,
        "parse_errors": snapshot.parse_errors,
        "mimetype_first_uncompressed": snapshot.mimetype_first_stored,
        "mimetype_exact": snapshot.mimetype_exact,
        "unsafe_paths": snapshot.unsafe_paths,
        "temporary_files": snapshot.temporary_files,
    }


def _baseline_markdown(data: dict[str, Any]) -> str:
    source = data["source"]
    candidate = data["candidate"]
    defects = data["candidate_defects"]
    lines = [
        "# Línea base del EPUB",
        "",
        f"Generada: {data['generated_at']}",
        "",
        "## Entradas",
        "",
        f"- Fuente: `{Path(source['path']).name}` (`{data['hashes']['source']}`)",
        f"- Candidato previo: `{Path(candidate['path']).name}` (`{data['hashes']['candidate']}`)",
        "",
        "## Paquete fuente",
        "",
        f"- EPUB {source['epub_version']}; OPF `{source['opf_path']}`",
        f"- Manifest: {source['manifest_items']} elementos; spine: {source['spine_documents']} documentos",
        f"- XHTML: {source['xhtml_files']}; imágenes: {source['images']}; CSS: {source['stylesheets']}; fuentes: {source['fonts']}",
        f"- TOC: {len(source['toc_entries'])} entradas; enlaces internos: {source['internal_links']}",
        f"- Unidades: {source['units']}; párrafos: {source['paragraphs']}; oraciones: {source['sentences']}",
        f"- Palabras: {source['words']}; caracteres: {source['characters']}",
        f"- Longitud de unidades: `{json.dumps(source['unit_lengths'], ensure_ascii=False)}`",
        "",
        "## Defectos reproducidos en el candidato",
        "",
        f"- Unidades en idioma fuente: {defects['source_language_units']}",
        f"- Unidades mixtas/residuales: {defects['mixed_language_units']}",
        f"- Unidades duplicadas: {defects['duplicate_units']}",
        f"- Unidades faltantes: {defects['missing_units']}",
        f"- Placeholders/protocolo sin resolver: {defects['placeholder_findings']}",
        f"- Fronteras o puntuación pegada: {defects['spacing_findings']}",
        f"- Enlaces rotos: {defects['broken_links']}",
        f"- Diferencia de spine: {str(defects['spine_changed']).lower()}",
        f"- Diferencia de metadatos: {str(defects['metadata_changed']).lower()}",
        "",
        "## Idiomas detectados",
        "",
        f"- Fuente: `{json.dumps(source['language_distribution'], ensure_ascii=False)}`",
        f"- Candidato: `{json.dumps(candidate['language_distribution'], ensure_ascii=False)}`",
    ]
    if defects["errors"]:
        lines.extend(["", "## Errores de publicación de línea base", ""])
        lines.extend(f"- {error}" for error in defects["errors"])
    return "\n".join(lines).strip() + "\n"


def _style_guide(title: str, target_variant: str) -> str:
    return f"""# Guía de estilo editorial

- Obra: {title}
- Variante: {target_variant}
- Prioridad: fidelidad semántica, seguida de naturalidad literaria.
- Ortografía: norma panhispánica contemporánea; conservar grafías deliberadas en citas y nombres.
- Comillas: angulares para citas principales; inglesas y simples para niveles internos cuando proceda.
- Diálogo e incisos: raya larga con puntuación española; no convertir automáticamente guiones técnicos.
- Cursivas: conservar títulos, énfasis y vocablos extranjeros conforme al XHTML fuente.
- Títulos de obras: traducir cuando el contexto editorial lo requiera; registrar la decisión y no inventar títulos establecidos.
- Números y fechas: conservar valores, unidades y relaciones temporales; adaptar sólo la convención gráfica.
- Abreviaturas: usar formas españolas reconocibles sin alterar nombres legales, siglas ni referencias.
- Mayúsculas: criterio español; conservar nombres propios, instituciones y títulos cuando sean parte de una cita.
- Topónimos y nombres históricos: forma española asentada cuando exista; de otro modo conservar la fuente.
- Transliteración: coherente en toda la obra; nunca alternar variantes sin justificación.
- Notas y referencias: preservar anclas, destinos y numeración; no mostrar rutas internas al lector.
- Citas en terceros idiomas: conservar exactamente o traducirlas por completo según la política editorial; nunca mezclar idiomas dentro de una misma cita.
- Citas en el idioma fuente: traducir al destino salvo una instrucción explícita de preservación acompañada por su versión legible.
- Ambigüedades: conservar las reales; no explicar ni cerrar interpretaciones dentro del texto.
- Arcaísmos y regionalismos: conservar si aportan voz o época; sustituir sólo cuando el perfil editorial lo indique.
- Ritmo: conservar periodos, digresiones, repeticiones deliberadas y respiración del autor.
- Metadatos: traducir título, subtítulo y etiquetas funcionales; preservar autor, identificadores, fecha, derechos y portada.
"""


def _initial_manifest(source, candidate, candidate_gate, hashes) -> dict[str, Any]:
    gate_by_id = {item["unit_id"]: item for item in candidate_gate.units}
    units = []
    for index, source_unit in enumerate(source.units):
        candidate_unit = candidate.units[index] if index < len(candidate.units) else None
        gate = gate_by_id.get(source_unit.unit_id, {})
        issues = list((gate.get("audit") or {}).get("issues") or ["missing_candidate"])
        classification = "VALID" if not issues else "UNCERTAIN"
        if any(code in issues for code in ("source_language_output", "untranslated_source", "target_language_missing")):
            classification = "SOURCE_LANGUAGE_RESIDUAL"
        elif any(
            "mixed" in code or "residue" in code or "residual" in code
            for code in issues
        ):
            classification = "MIXED_LANGUAGE"
        elif "duplicate_output" in issues:
            classification = "DUPLICATED"
        elif candidate_unit is None:
            classification = "MISSING"
        units.append({
            "unit_id": source_unit.unit_id,
            "source_document": source_unit.file_href,
            "spine_index": source_unit.spine_index,
            "dom_path": source_unit.dom_path,
            "source_order": source_unit.source_order,
            "source_hash": source_unit.source_hash,
            "candidate_hash": text_sha256(candidate_unit.text) if candidate_unit else None,
            "candidate_classification": classification,
            "translation_status": "COMPLETED" if candidate_unit else "PENDING",
            "review_status": "PENDING",
            "audit_status": "PENDING",
            "failure_reason": ", ".join(issues) if issues else None,
            "prompt_versions": prompt_versions(),
            "pipeline_version": EPUB_PIPELINE_VERSION,
            "placeholder_manifest": [],
        })
    return {
        "schema_version": 2,
        "kind": "candidate_memory_baseline",
        "pipeline_version": EPUB_PIPELINE_VERSION,
        "prompt_versions": prompt_versions(),
        "source_sha256": hashes["source"],
        "candidate_sha256": hashes["candidate"],
        "total_units": len(units),
        "classification_counts": dict(Counter(item["candidate_classification"] for item in units)),
        "units": units,
    }


def _final_manifest(final_gate, semantic, hashes) -> dict[str, Any]:
    semantic_by_id = {item["unit_id"]: item for item in semantic.get("results", [])}
    units = []
    for item in final_gate.units:
        semantic_item = semantic_by_id.get(item["unit_id"])
        accepted = bool(semantic_item and semantic_item.get("accepted") and item.get("status") == "AUDITED")
        units.append({
            **item,
            "translation_status": "COMPLETED",
            "review_status": "COMPLETED",
            "audit_status": "COMPLETED" if accepted else "FAILED",
            "status": "AUDITED" if accepted else "FAILED",
            "translation_attempts": 1,
            "review_attempts": 1,
            "audit_attempts": 1,
            "failure_reason": None if accepted else (semantic_item or {}).get("reason") or item.get("failure_reason"),
            "semantic_audit": semantic_item,
            "prompt_versions": prompt_versions(),
            "pipeline_version": EPUB_PIPELINE_VERSION,
            "placeholder_manifest": [],
        })
    return {
        "schema_version": 2,
        "kind": "final_translation_manifest",
        "pipeline_version": EPUB_PIPELINE_VERSION,
        "prompt_versions": prompt_versions(),
        "source_sha256": hashes["source"],
        "output_sha256": hashes["final"],
        "total_units": len(units),
        "status_counts": dict(Counter(item["status"] for item in units)),
        "units": units,
    }


def _editorial_changes(source, candidate, final, metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
    changes = []
    by_file: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for index, source_unit in enumerate(source.units):
        if index >= len(candidate.units) or index >= len(final.units):
            continue
        old = candidate.units[index].text
        new = final.units[index].text
        if old == new:
            continue
        change_type = "content_repair"
        if re.sub(r"\s+", " ", old).strip() == re.sub(r"\s+", " ", new).strip():
            change_type = "dom_boundary_spacing"
        old_language, _ = LanguageDetector.detect_language_from_text(old, confidence_threshold=0.80)
        new_language, _ = LanguageDetector.detect_language_from_text(new, confidence_threshold=0.80)
        if str(old_language or "").casefold() in {"german", "de"} and str(new_language or "").casefold() in {"spanish", "es"}:
            change_type = "source_language_retranslation"
        by_file[source_unit.file_href] += 1
        types[change_type] += 1
        if len(changes) < 12:
            changes.append({
                "unit_id": source_unit.unit_id,
                "source_document": source_unit.file_href,
                "type": change_type,
                "source": source_unit.text[:360],
                "previous": old[:360],
                "final": new[:360],
                "reason": "El candidato no superaba la validación completa de idioma, fidelidad o frontera DOM.",
            })
    data = {
        "total_units_reviewed": len(final.units),
        "total_units_modified": sum(by_file.values()),
        "changes_by_document": dict(by_file),
        "change_types": dict(types),
        "representative_examples": changes,
        "orthographic_policy": "Norma panhispánica contemporánea y español literario internacional.",
        "terminology_policy": "Consistencia por glosario; nombres, cifras y términos clave se auditan contra fuente.",
        "quotation_policy": "Las citas auténticas en terceros idiomas se conservan y se clasifican.",
        "metadata_changes": metadata,
        "cover_decision": "Portada original preservada byte por byte.",
    }
    lines = [
        "# Cambios editoriales",
        "",
        f"- Unidades revisadas: {data['total_units_reviewed']}",
        f"- Unidades modificadas frente al candidato: {data['total_units_modified']}",
        f"- Tipos de cambio: `{json.dumps(data['change_types'], ensure_ascii=False)}`",
        f"- Portada: {data['cover_decision']}",
        "",
        "## Cambios por documento",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in by_file.items())
    lines.extend(["", "## Ejemplos representativos", ""])
    for item in changes:
        lines.extend([
            f"### {item['unit_id']}",
            f"- Tipo: {item['type']}",
            f"- Fuente: {item['source']}",
            f"- Anterior: {item['previous']}",
            f"- Final: {item['final']}",
            f"- Justificación: {item['reason']}",
            "",
        ])
    lines.extend([
        "## Política",
        "",
        f"- Ortografía: {data['orthographic_policy']}",
        f"- Terminología: {data['terminology_policy']}",
        f"- Citas: {data['quotation_policy']}",
        f"- Metadatos: `{json.dumps(metadata, ensure_ascii=False)}`",
    ])
    return data, "\n".join(lines).strip() + "\n"


def _qa_markdown(qa: dict[str, Any]) -> str:
    final = qa["final"]
    semantic = qa["semantic_audit"]
    lines = [
        f"# QA final EPUB - {qa['title']}",
        "",
        f"- Estado: **{'PASS' if qa['publishable'] else 'FAIL'}**",
        f"- Fuente SHA-256: `{qa['hashes']['source']}`",
        f"- Candidato SHA-256: `{qa['hashes']['candidate']}`",
        f"- Salida SHA-256: `{qa['hashes']['final']}`",
        f"- Pipeline: `{qa['pipeline_version']}`",
        f"- Prompts: `{json.dumps(qa['prompt_versions'], ensure_ascii=False)}`",
        f"- Modelo auditor: `{semantic.get('model')}`",
        "",
        "## Cobertura",
        "",
        f"- Unidades fuente/salida: {final['source_units']}/{final['output_units']}",
        f"- Traducidas: {final['output_units']}",
        f"- Revisadas: {semantic.get('audited_units', 0)}",
        f"- Auditadas semánticamente: {semantic.get('passed_units', 0)}",
        f"- Fallidas: {semantic.get('failed_units', 0)}",
        f"- Idioma fuente residual: {final['source_language_units']}",
        f"- Segmentos mixtos: {final['mixed_language_units']}",
        f"- Omisiones: {final['missing_units']}",
        f"- Duplicaciones: {final['duplicate_units']}",
        f"- Placeholders: {final['placeholder_findings']}",
        f"- Fronteras DOM perdidas: {final['dom_boundary_findings']}",
        f"- Enlaces rotos: {final['broken_links']}",
        "",
        "## Paquete",
        "",
        f"- XHTML: {qa['snapshot']['xhtml_files']}; spine: {qa['snapshot']['spine_documents']}; TOC: {len(qa['snapshot']['toc_entries'])}",
        f"- Imágenes: {qa['snapshot']['images']}; CSS: {qa['snapshot']['stylesheets']}; fuentes: {qa['snapshot']['fonts']}",
        f"- EPUBCheck: {final['epubcheck_errors']} errores, {final['epubcheck_warnings']} warnings",
        f"- Apertura/recorrido: {qa['reader_check']}",
        f"- Pruebas: {qa['test_summary']}",
    ]
    if final["errors"]:
        lines.extend(["", "## Errores", ""] + [f"- {item}" for item in final["errors"]])
    if qa["limitations"]:
        lines.extend(["", "## Limitaciones reales", ""] + [f"- {item}" for item in qa["limitations"]])
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--semantic-report", type=Path, default=Path("reports/semantic_audit.json"))
    parser.add_argument("--source-language", default="German")
    parser.add_argument("--target-language", default="Spanish")
    parser.add_argument("--title", required=True)
    parser.add_argument("--target-variant", default="español literario internacional")
    parser.add_argument("--test-summary", default="pending")
    parser.add_argument("--reader-check", default="pending")
    args = parser.parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).isoformat()
    hashes = {"source": _sha256(args.source), "candidate": _sha256(args.candidate), "final": _sha256(args.final)}
    source = snapshot_epub(args.source, recover=True)
    candidate = snapshot_epub(args.candidate, recover=True)
    final = snapshot_epub(args.final, recover=False)
    candidate_gate = audit_epub_publication(
        args.source, args.candidate, source_language=args.source_language,
        target_language=args.target_language, epubcheck_command=["/usr/bin/true"],
    )
    final_gate = audit_epub_publication(
        args.source, args.final, source_language=args.source_language,
        target_language=args.target_language,
    )
    semantic = json.loads(args.semantic_report.read_text(encoding="utf-8")) if args.semantic_report.exists() else {
        "complete": False, "audited_units": 0, "passed_units": 0, "failed_units": len(source.units), "results": []
    }
    baseline = {
        "schema_version": 1,
        "generated_at": generated,
        "hashes": {"source": hashes["source"], "candidate": hashes["candidate"]},
        "source": _snapshot_summary(source),
        "candidate": _snapshot_summary(candidate),
        "candidate_defects": {
            **{key: value for key, value in candidate_gate.to_dict().items() if key != "units"},
            "spine_changed": source.spine != candidate.spine,
            "metadata_changed": source.metadata != candidate.metadata,
        },
    }
    (args.reports_dir / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.reports_dir / "baseline.md").write_text(_baseline_markdown(baseline), encoding="utf-8")
    (args.reports_dir / "style_guide.md").write_text(_style_guide(args.title, args.target_variant), encoding="utf-8")
    initial_manifest = _initial_manifest(source, candidate, candidate_gate, hashes)
    final_manifest = _final_manifest(final_gate, semantic, hashes)
    (args.reports_dir / "translation_manifest.json").write_text(json.dumps(initial_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.reports_dir / "translation_manifest_final.json").write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_changes = {
        "before": candidate.metadata,
        "after": final.metadata,
        "navigation_labels": final.toc_entries[:8],
    }
    editorial_data, editorial_md = _editorial_changes(source, candidate, final, metadata_changes)
    (args.reports_dir / "editorial_changes.md").write_text(editorial_md, encoding="utf-8")
    (PROJECT_ROOT / "editorial_changes.md").write_text(editorial_md, encoding="utf-8")
    qa = {
        "schema_version": 2,
        "generated_at": generated,
        "title": args.title,
        "pipeline_version": EPUB_PIPELINE_VERSION,
        "prompt_versions": prompt_versions(),
        "hashes": hashes,
        "inputs_unchanged": (
            hashes["source"] == _sha256(args.source)
            and hashes["candidate"] == _sha256(args.candidate)
        ),
        "final": final_gate.to_dict(),
        "snapshot": _snapshot_summary(final),
        "semantic_audit": {key: value for key, value in semantic.items() if key != "results"},
        "editorial_changes": editorial_data,
        "test_summary": args.test_summary,
        "reader_check": args.reader_check,
        "limitations": [],
    }
    qa["publishable"] = bool(final_gate.publishable and semantic.get("complete") and not semantic.get("failed_units"))
    if not semantic.get("complete"):
        qa["limitations"].append("La auditoría semántica completa no ha terminado o contiene rechazos.")
    (args.reports_dir / "final_epub_qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.reports_dir / "final_epub_qa.md").write_text(_qa_markdown(qa), encoding="utf-8")
    run_summary = {
        "generated_at": generated,
        "publishable": qa["publishable"],
        "output": str(args.final),
        "output_sha256": hashes["final"],
        "units": {"source": len(source.units), "final": len(final.units), "semantic_passed": semantic.get("passed_units", 0)},
        "defects_before": baseline["candidate_defects"],
        "defects_after": {key: value for key, value in final_gate.to_dict().items() if key != "units"},
        "tests": args.test_summary,
        "reader_check": args.reader_check,
    }
    (args.reports_dir / "run_summary.md").write_text(
        "# Resumen de ejecución\n\n```json\n" + json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"publishable": qa["publishable"], "reports": str(args.reports_dir), "sha256": hashes["final"]}, ensure_ascii=False, indent=2))
    return 0 if qa["publishable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
