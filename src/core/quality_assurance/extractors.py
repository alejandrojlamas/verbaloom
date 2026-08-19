"""Build universal manifests from checkpoints or source/output artifacts."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lxml import etree

from src.core.document_structure import DocumentBlockClassifier
from src.core.output_formats import extract_readable_text

from .models import (
    BookManifest,
    ModelMetadata,
    TranslationUnit,
    UnitStatus,
    document_id_for_path,
)


@dataclass(frozen=True)
class ExtractedBlock:
    text: str
    structural_path: str
    parent_id: str = ""
    content_type: str = "narrative"
    policy: str = "translate"
    html_tag: str = ""
    attributes: dict[str, str] | None = None
    preceding_whitespace: str = ""
    trailing_whitespace: str = ""
    exclusion_reason: str = ""


def build_manifest(
    *,
    source_path: str | Path,
    output_path: str | Path,
    source_language: str,
    target_language: str,
    target_locale: str = "",
    run_id: str = "",
    checkpoint_data: Mapping[str, Any] | None = None,
    publication_report: Any = None,
) -> BookManifest:
    source = Path(source_path)
    output = Path(output_path)
    document_id = document_id_for_path(source)
    if publication_report is not None and getattr(publication_report, "units", None):
        units, exclusions = _units_from_epub_publication_report(
            publication_report,
            document_id=document_id,
            source_language=source_language,
            target_language=target_language,
        )
        metadata = {"manifest_source": "epub_publication_gate"}
    elif checkpoint_data and (checkpoint_data.get("chunks") or _checkpoint_total(checkpoint_data)):
        units, exclusions = _units_from_checkpoint(
            checkpoint_data,
            document_id=document_id,
            source_language=source_language,
            target_language=target_language,
        )
        metadata = {"manifest_source": "checkpoint"}
    else:
        units, exclusions, metadata = _units_from_artifacts(
            source,
            output,
            document_id=document_id,
            source_language=source_language,
            target_language=target_language,
        )

    return BookManifest(
        document_id=document_id,
        run_id=run_id,
        source_path=str(source),
        output_path=str(output),
        source_language=source_language,
        target_language=target_language,
        target_locale=target_locale,
        source_format=_format_name(source),
        output_format=_format_name(output),
        units=units,
        explicit_exclusions=exclusions,
        metadata=metadata,
    )


def inspect_source(
    source_path: str | Path,
    *,
    source_language: str = "auto",
    target_language: str = "",
    run_id: str = "",
) -> BookManifest:
    source = Path(source_path)
    document_id = document_id_for_path(source)
    blocks = extract_blocks(source)
    units = [
        _unit_from_block(
            block,
            document_id=document_id,
            order_index=index,
            source_language=source_language,
            target_language=target_language,
            status=UnitStatus.EXTRACTED,
        )
        for index, block in enumerate(blocks)
    ]
    exclusions = [_exclusion_for_unit(unit) for unit in units if not unit.translatable]
    return BookManifest(
        document_id=document_id,
        run_id=run_id,
        source_path=str(source),
        output_path="",
        source_language=source_language,
        target_language=target_language,
        source_format=_format_name(source),
        units=units,
        explicit_exclusions=exclusions,
        metadata={"manifest_source": "source_inspection"},
    )


def extract_blocks(path: str | Path) -> list[ExtractedBlock]:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".epub":
        return _extract_epub_blocks(file_path)
    if suffix == ".docx":
        return _extract_docx_blocks(file_path)
    if suffix == ".srt":
        return _extract_srt_blocks(file_path)
    if suffix == ".pdf":
        return _extract_pdf_blocks(file_path)
    return _extract_text_blocks(file_path)


def _units_from_checkpoint(
    checkpoint_data: Mapping[str, Any],
    *,
    document_id: str,
    source_language: str,
    target_language: str,
) -> tuple[list[TranslationUnit], list[dict[str, Any]]]:
    job = dict(checkpoint_data.get("job") or {})
    config = dict(job.get("config") or {})
    progress = dict(job.get("progress") or {})
    chunks = {
        int(chunk.get("chunk_index")): dict(chunk)
        for chunk in checkpoint_data.get("chunks") or []
        if _is_int(chunk.get("chunk_index"))
    }
    total = max(
        int(progress.get("total_chunks") or 0),
        max(chunks.keys(), default=-1) + 1,
    )
    model = ModelMetadata.from_mapping(config)
    units: list[TranslationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for index in range(total):
        chunk = chunks.get(index)
        if chunk is None:
            unit = TranslationUnit.create(
                document_id=document_id,
                order_index=index,
                source_text="",
                source_language=source_language,
                target_language=target_language,
                structural_path=f"/checkpoint/{index}",
                status=UnitStatus.TRANSLATION_FAILED,
                model_metadata=model,
                source_reference={"checkpoint_missing": True},
            )
            units.append(unit)
            continue

        chunk_data = dict(chunk.get("chunk_data") or {})
        source_text = str(chunk.get("original_text") or "")
        target_text = str(chunk.get("translated_text") or "")
        content_type, policy, exclusion = _classify_source_text(
            source_text,
            source_type=str(job.get("file_type") or "text"),
            explicit_type=str(chunk_data.get("content_type") or ""),
            explicit_policy=str(chunk_data.get("policy") or ""),
        )
        status = (
            UnitStatus.TRANSLATED
            if chunk.get("status") == "completed" and target_text
            else UnitStatus.TRANSLATION_FAILED
        )
        unit_model = ModelMetadata.from_mapping(
            {
                **config,
                **dict(chunk_data.get("model_metadata") or {}),
                **dict(chunk_data.get("token_usage") or {}),
            }
        )
        structural_path = str(
            chunk_data.get("structural_path")
            or chunk_data.get("dom_path")
            or chunk_data.get("file_href")
            or chunk_data.get("source_document")
            or f"/checkpoint/{index}"
        )
        unit = TranslationUnit.create(
            document_id=document_id,
            order_index=index,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            translated_text=target_text,
            reviewed_text=target_text if chunk_data.get("inline_refinement") else "",
            final_text=target_text,
            parent_id=str(chunk_data.get("parent_id") or ""),
            content_type=content_type,
            policy=policy,
            structural_path=structural_path,
            html_tag=str(chunk_data.get("html_tag") or ""),
            attributes=dict(chunk_data.get("attributes") or {}),
            preceding_whitespace=str(chunk_data.get("preceding_whitespace") or ""),
            trailing_whitespace=str(chunk_data.get("trailing_whitespace") or ""),
            status=status,
            retry_count=int(chunk_data.get("retry_count") or chunk_data.get("attempts") or 0),
            model_metadata=unit_model,
            exclusion_reason=exclusion,
            source_reference={
                "checkpoint_index": index,
                "completed_at": chunk.get("completed_at"),
                "fidelity_decision": chunk_data.get("fidelity_decision"),
                "refinement_fidelity_decision": chunk_data.get("refinement_fidelity_decision"),
                "candidate_results": chunk_data.get("candidate_results"),
            },
        )
        units.append(unit)
        if not unit.translatable:
            exclusions.append(_exclusion_for_unit(unit))
    return units, exclusions


def _units_from_epub_publication_report(
    report: Any,
    *,
    document_id: str,
    source_language: str,
    target_language: str,
) -> tuple[list[TranslationUnit], list[dict[str, Any]]]:
    units: list[TranslationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for index, item in enumerate(report.units):
        source_text = str(item.get("source_text") or "")
        target_text = str(item.get("translation") or "")
        content_type, policy, exclusion = _classify_source_text(
            source_text,
            source_type="epub",
            explicit_type=str(item.get("element") or ""),
        )
        report_status = str(item.get("status") or "").upper()
        status = UnitStatus.AUDITED if report_status == "AUDITED" else UnitStatus.AUDIT_FAILED
        unit = TranslationUnit.create(
            document_id=document_id,
            order_index=index,
            source_text=source_text,
            source_language=source_language,
            target_language=target_language,
            translated_text=target_text,
            reviewed_text=target_text,
            final_text=target_text,
            parent_id=str(item.get("source_document") or ""),
            content_type=content_type,
            policy=policy,
            structural_path=str(item.get("dom_path") or f"/epub/{index}"),
            html_tag=str(item.get("element") or ""),
            status=status,
            retry_count=int(item.get("attempts") or 0),
            exclusion_reason=exclusion,
            source_reference={
                "source_document": item.get("source_document"),
                "spine_index": item.get("spine_index"),
                "source_order": item.get("source_order"),
                "epub_unit_id": item.get("unit_id"),
                # Publication units aggregate all reader-visible text in one
                # XHTML file. The EPUB gate has already checked its individual
                # blocks, so downstream QA must not reinterpret the aggregate
                # as one ordinary translation chunk.
                "publication_status": report_status,
                "publication_audited": report_status == "AUDITED",
                "intentional_exclusions": list(
                    item.get("intentional_exclusions") or []
                ),
            },
        )
        units.append(unit)
        if not unit.translatable:
            exclusions.append(_exclusion_for_unit(unit))
    return units, exclusions


def _units_from_artifacts(
    source: Path,
    output: Path,
    *,
    document_id: str,
    source_language: str,
    target_language: str,
) -> tuple[list[TranslationUnit], list[dict[str, Any]], dict[str, Any]]:
    source_blocks = extract_blocks(source)
    output_extraction_error = ""
    if output.exists():
        try:
            output_blocks = extract_blocks(output)
        except Exception as exc:
            output_blocks = []
            output_extraction_error = f"{type(exc).__name__}: {exc}"
    else:
        output_blocks = []
    units: list[TranslationUnit] = []
    exclusions: list[dict[str, Any]] = []
    for index, source_block in enumerate(source_blocks):
        output_block = output_blocks[index] if index < len(output_blocks) else None
        status = UnitStatus.TRANSLATED if output_block and output_block.text.strip() else UnitStatus.TRANSLATION_FAILED
        unit = _unit_from_block(
            source_block,
            document_id=document_id,
            order_index=index,
            source_language=source_language,
            target_language=target_language,
            translated_text=output_block.text if output_block else "",
            final_text=output_block.text if output_block else "",
            status=status,
        )
        units.append(unit)
        if not unit.translatable:
            exclusions.append(_exclusion_for_unit(unit))
    metadata = {
        "manifest_source": "artifact_alignment",
        "source_blocks": len(source_blocks),
        "output_blocks": len(output_blocks),
        "unmapped_output_blocks": max(0, len(output_blocks) - len(source_blocks)),
        "output_extraction_error": output_extraction_error,
    }
    return units, exclusions, metadata


def _unit_from_block(
    block: ExtractedBlock,
    *,
    document_id: str,
    order_index: int,
    source_language: str,
    target_language: str,
    translated_text: str = "",
    final_text: str = "",
    status: UnitStatus,
) -> TranslationUnit:
    return TranslationUnit.create(
        document_id=document_id,
        order_index=order_index,
        source_text=block.text,
        source_language=source_language,
        target_language=target_language,
        translated_text=translated_text,
        final_text=final_text,
        parent_id=block.parent_id,
        content_type=block.content_type,
        policy=block.policy,
        structural_path=block.structural_path,
        html_tag=block.html_tag,
        attributes=block.attributes or {},
        preceding_whitespace=block.preceding_whitespace,
        trailing_whitespace=block.trailing_whitespace,
        status=status,
        exclusion_reason=block.exclusion_reason,
    )


def _extract_text_blocks(path: Path) -> list[ExtractedBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return _blocks_from_text(text, source_type=_format_name(path))


def _extract_pdf_blocks(path: Path) -> list[ExtractedBlock]:
    return _blocks_from_text(extract_readable_text(path), source_type="pdf")


def _extract_srt_blocks(path: Path) -> list[ExtractedBlock]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[ExtractedBlock] = []
    for match_index, raw in enumerate(re.split(r"\n\s*\n", text.strip())):
        lines = raw.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        content = "\n".join(lines[2:]).strip()
        blocks.append(
            ExtractedBlock(
                text=content,
                structural_path=f"/subtitles/{match_index}",
                parent_id=str(lines[0]).strip(),
                content_type="narrative",
                policy="translate",
                attributes={"timing": lines[1].strip()},
            )
        )
    return blocks


def _extract_docx_blocks(path: Path) -> list[ExtractedBlock]:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(str(path))
    blocks: list[ExtractedBlock] = []
    body = document.element.body
    index = 0
    for child in body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text
            if not text.strip():
                continue
            content_type, policy, exclusion = _classify_source_text(
                text, source_type="docx"
            )
            blocks.append(
                ExtractedBlock(
                    text=text,
                    structural_path=f"/document/paragraph[{index}]",
                    content_type=content_type,
                    policy=policy,
                    html_tag="p",
                    attributes={"style": paragraph.style.name if paragraph.style else ""},
                    exclusion_reason=exclusion,
                )
            )
            index += 1
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = [" | ".join(cell.text for cell in row.cells) for row in table.rows]
            blocks.append(
                ExtractedBlock(
                    text="\n".join(rows),
                    structural_path=f"/document/table[{index}]",
                    content_type="table",
                    policy="reconstruct",
                    html_tag="table",
                )
            )
            index += 1
    return blocks


def _extract_epub_blocks(path: Path) -> list[ExtractedBlock]:
    from src.core.epub.publication_gate import snapshot_epub

    snapshot = snapshot_epub(path, recover=False)
    blocks: list[ExtractedBlock] = []
    for unit in snapshot.units:
        content_type, policy, exclusion = _classify_source_text(
            unit.text,
            source_type="epub",
            explicit_type=unit.element,
        )
        blocks.append(
            ExtractedBlock(
                text=unit.text,
                structural_path=unit.dom_path or f"/{unit.file_href}/{unit.ordinal}",
                parent_id=unit.file_href,
                content_type=content_type,
                policy=policy,
                html_tag=unit.element,
                attributes={"spine_index": str(unit.spine_index)},
                exclusion_reason=exclusion,
            )
        )
    return blocks


def _blocks_from_text(text: str, *, source_type: str) -> list[ExtractedBlock]:
    classifier = DocumentBlockClassifier(source_type=source_type)
    blocks: list[ExtractedBlock] = []
    for index, match in enumerate(re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text or "", re.DOTALL)):
        value = match.group(0)
        classified = classifier.classify_text(value)
        classification = classified[0] if classified else None
        content_type = classification.type if classification else "narrative"
        policy = classification.policy if classification else "translate"
        exclusion = _policy_exclusion_reason(content_type, policy)
        blocks.append(
            ExtractedBlock(
                text=value.strip(),
                structural_path=f"/blocks/{index}",
                content_type=content_type,
                policy=policy,
                preceding_whitespace=_leading_whitespace(value),
                trailing_whitespace=_trailing_whitespace(value),
                exclusion_reason=exclusion,
            )
        )
    if not blocks and (text or "").strip():
        blocks.append(
            ExtractedBlock(text=text.strip(), structural_path="/blocks/0")
        )
    return blocks


def _classify_source_text(
    text: str,
    *,
    source_type: str,
    explicit_type: str = "",
    explicit_policy: str = "",
) -> tuple[str, str, str]:
    classifier = DocumentBlockClassifier(source_type=source_type)
    classified = classifier.classify_text(text)
    block = classified[0] if classified else None
    content_type = explicit_type or (block.type if block else "narrative")
    if content_type in {"p", "div", "span", "li", "blockquote"}:
        content_type = block.type if block else "narrative"
    policy = explicit_policy or (block.policy if block else "translate")
    exclusion = _policy_exclusion_reason(content_type, policy)
    return content_type, policy, exclusion


def _policy_exclusion_reason(content_type: str, policy: str) -> str:
    if policy == "exclude":
        return f"Block classified as {content_type} and excluded before translation."
    if policy == "preserve":
        return f"Block classified as {content_type} and intentionally preserved."
    return ""


def _exclusion_for_unit(unit: TranslationUnit) -> dict[str, Any]:
    return {
        "unit_id": unit.unit_id,
        "order_index": unit.order_index,
        "content_type": unit.content_type,
        "policy": unit.policy,
        "reason": unit.exclusion_reason or "Explicit non-translation policy.",
    }


def _checkpoint_total(checkpoint_data: Mapping[str, Any]) -> int:
    job = checkpoint_data.get("job") or {}
    return int((job.get("progress") or {}).get("total_chunks") or 0)


def _format_name(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return "txt" if suffix in {"", "text", "md", "markdown"} else suffix


def _leading_whitespace(value: str) -> str:
    match = re.match(r"\s*", value or "")
    return match.group(0) if match else ""


def _trailing_whitespace(value: str) -> str:
    match = re.search(r"\s*$", value or "")
    return match.group(0) if match else ""


def _is_int(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False
