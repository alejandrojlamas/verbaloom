"""
TXT refine-only mode.

Reads an already-translated plain-text file, chunks it the same way as
translation would, then runs refine_chunks() and writes the polished output.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.core.text_processor import split_text_into_chunks
from src.core.translator import refine_chunks
from src.core.post_processor import clean_translated_text
from src.core.editorial_quality import EditorialQualityReport, editorial_report_path
from src.core.fidelity_supervisor import ensure_fidelity_report, write_fidelity_report_from_options
from src.core.postprocess_repair import postprocess_repair_enabled, repair_flagged_chunks
from src.core.text_transform import apply_faithful_modernize_defaults, prompt_bool
from src.config import DEFAULT_MODEL, API_ENDPOINT
from src.utils.ocr_normalizer import normalize_ocr_text
from src.utils.text_reader import read_text_file_with_fallbacks


async def refine_txt_file(
    input_filepath: str,
    output_filepath: str,
    target_language: str,
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
) -> bool:
    """Run a refinement-only pass on an already-translated text file.

    `target_language` names the language the file is already in: refinement
    is monolingual and does not translate.
    """
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input file '{input_filepath}' not found."
        if log_callback:
            log_callback("file_not_found_error", err_msg)
        else:
            print(err_msg)
        return False

    try:
        translated_text, input_encoding = read_text_file_with_fallbacks(input_filepath)
        if log_callback and input_encoding not in ("utf-8", "utf-8-sig"):
            log_callback(
                "text_encoding_fallback",
                f"Read text input using {input_encoding} encoding."
            )
    except Exception as e:
        err_msg = f"ERROR: Reading input file '{input_filepath}': {e}"
        if log_callback:
            log_callback("file_read_error", err_msg)
        else:
            print(err_msg)
        return False

    if not translated_text.strip():
        if log_callback:
            log_callback("txt_empty_input", "TXT file contains no readable text. Nothing to refine.")
        return False

    return await refine_text_content(
        translated_text=translated_text,
        output_filepath=output_filepath,
        target_language=target_language,
        model_name=model_name,
        cli_api_endpoint=cli_api_endpoint,
        log_callback=log_callback,
        stats_callback=stats_callback,
        check_interruption_callback=check_interruption_callback,
        llm_provider=llm_provider,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        context_window=context_window,
        auto_adjust_context=auto_adjust_context,
        max_tokens_per_chunk=max_tokens_per_chunk,
        soft_limit_ratio=soft_limit_ratio,
        prompt_options=prompt_options,
        checkpoint_manager=checkpoint_manager,
        translation_id=translation_id,
        resume_from_index=resume_from_index,
    )


async def refine_text_content(
    translated_text: str,
    output_filepath: str,
    target_language: str,
    model_name: str = DEFAULT_MODEL,
    cli_api_endpoint: str = API_ENDPOINT,
    log_callback: Optional[Callable] = None,
    stats_callback: Optional[Callable] = None,
    check_interruption_callback: Optional[Callable] = None,
    llm_provider: str = "ollama",
    gemini_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    mistral_api_key: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    poe_api_key: Optional[str] = None,
    nim_api_key: Optional[str] = None,
    context_window: int = 2048,
    auto_adjust_context: bool = True,
    max_tokens_per_chunk: Optional[int] = None,
    soft_limit_ratio: Optional[float] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
) -> bool:
    """Run refinement on plain text content and write a text output file."""
    prompt_options = dict(prompt_options or {})
    apply_faithful_modernize_defaults(prompt_options)
    ensure_fidelity_report(
        prompt_options,
        document_name=os.path.basename(output_filepath),
        source_language=prompt_options.get('_source_language', ''),
        target_language=target_language,
        translator_model=model_name,
        translator_provider=llm_provider,
    )
    normalize_scanned_text = prompt_options.get('normalize_scanned_text', True)
    write_quality_report = prompt_options.get('editorial_quality_report', True)
    quality_report = EditorialQualityReport(
        document_name=os.path.basename(output_filepath),
        target_language=target_language,
    )
    prompt_options.setdefault('editorial_quality_guard', True)
    prompt_options['_editorial_quality_report'] = quality_report

    if normalize_scanned_text:
        normalization = normalize_ocr_text(translated_text)
        if normalization.is_likely_scan:
            translated_text = normalization.text
            prompt_options['text_cleanup'] = True
            prompt_options['ocr_normalized'] = True
            if log_callback:
                reason_text = ", ".join(normalization.reasons) or "scan artifacts"
                log_callback(
                    "ocr_normalized",
                    "Detected OCR/scan artifacts; normalized text before refinement "
                    f"({reason_text}, score {normalization.score})."
                )
        elif normalization.changed:
            translated_text = normalization.text
            if log_callback:
                log_callback(
                    "text_normalized",
                    "Normalized whitespace and Unicode before refinement."
                )

    if log_callback:
        log_callback("refine_split_start", "Splitting translated text for refinement...")

    structured_chunks = split_text_into_chunks(
        translated_text,
        max_tokens_per_chunk=max_tokens_per_chunk,
        soft_limit_ratio=soft_limit_ratio,
    )
    total_chunks = len(structured_chunks)

    if total_chunks == 0:
        if log_callback:
            log_callback("txt_no_chunks_warning",
                         "WARNING: No segments generated for non-empty text. Processing as a single block.")
        structured_chunks = [{
            "context_before": "",
            "main_content": translated_text,
            "context_after": "",
        }]
        total_chunks = 1

    if stats_callback:
        stats_callback({'total_chunks': total_chunks, 'completed_chunks': 0, 'failed_chunks': 0})

    source_refs = prompt_options.get('_source_guard_reference_chunks')
    if isinstance(source_refs, list) and len(source_refs) == total_chunks:
        for chunk, source_text in zip(structured_chunks, source_refs):
            if isinstance(source_text, str) and source_text.strip():
                chunk['_source_text'] = source_text
        if log_callback:
            log_callback(
                "source_aware_guard_aligned",
                f"🔎 Source-aware editorial guard aligned with {total_chunks} source chunks."
            )
    elif source_refs:
        prompt_options['_source_guard_reference_chunks_misaligned'] = True
        if log_callback:
            log_callback(
                "source_aware_guard_misaligned",
                "⚠️ Source-aware editorial guard could not align source chunks "
                f"({len(source_refs) if isinstance(source_refs, list) else 'unknown'} source refs, "
                f"{total_chunks} refine chunks). Falling back to local guard."
            )

    if log_callback:
        log_callback("refine_info_chunks",
                     f"Refining {total_chunks} segment(s) in {target_language}.")

    # refine_chunks uses original_chunks only for context_before/after, so in
    # refine-only mode we pass main_content as both draft and original.
    draft_chunks = [c["main_content"] for c in structured_chunks]
    resume_from_index = max(0, min(int(resume_from_index or 0), total_chunks))
    prefilled_parts = []
    if resume_from_index and checkpoint_manager and translation_id:
        checkpoint_data = checkpoint_manager.load_checkpoint(translation_id) or {}
        chunks_by_index = {
            int(chunk.get('chunk_index')): chunk
            for chunk in checkpoint_data.get('chunks', [])
            if chunk.get('chunk_index') is not None
        }
        for idx in range(resume_from_index):
            saved = chunks_by_index.get(idx)
            if not saved or saved.get('status') != 'completed' or saved.get('translated_text') is None:
                break
            prefilled_parts.append(saved.get('translated_text') or draft_chunks[idx])

        if len(prefilled_parts) != resume_from_index:
            if log_callback:
                log_callback(
                    "refine_resume_incomplete_checkpoint",
                    "⚠️ Refinement checkpoint is incomplete; resuming from the "
                    f"first unsaved chunk ({len(prefilled_parts) + 1}/{total_chunks})."
                )
            resume_from_index = len(prefilled_parts)

        if log_callback and resume_from_index:
            log_callback(
                "refine_resume_prefix_restored",
                f"📂 Restored {resume_from_index}/{total_chunks} refined chunks from checkpoint."
            )

    initial_progress = {
        'total_chunks': total_chunks,
        'completed_chunks': resume_from_index,
        'failed_chunks': 0,
    }
    if stats_callback:
        stats_callback(initial_progress)
    if checkpoint_manager and translation_id:
        try:
            checkpoint_manager.update_progress(
                translation_id=translation_id,
                current_chunk_index=resume_from_index - 1,
                total_chunks=total_chunks,
                completed_chunks=resume_from_index,
                failed_chunks=0,
            )
        except Exception as exc:
            if log_callback:
                log_callback(
                    "refine_progress_checkpoint_error",
                    f"⚠️ Could not initialize refinement checkpoint progress: {exc}"
                )

    def _global_stats_callback(local_stats: Dict[str, Any]) -> None:
        if not stats_callback:
            return
        stats_callback({
            **local_stats,
            'total_chunks': total_chunks,
            'completed_chunks': min(
                total_chunks,
                resume_from_index + int(local_stats.get('completed_chunks') or 0),
            ),
            'failed_chunks': int(local_stats.get('failed_chunks') or 0),
        })

    def _checkpoint_callback(chunk_index: int, draft_text: str, refined_text: Optional[str], local_stats: Dict[str, Any]) -> None:
        if not checkpoint_manager or not translation_id:
            return
        checkpoint_manager.save_checkpoint(
            translation_id=translation_id,
            chunk_index=chunk_index,
            original_text=draft_text,
            translated_text=refined_text,
            chunk_data={'file_type': 'txt_refine'},
            total_chunks=total_chunks,
            completed_chunks=min(total_chunks, chunk_index + 1),
            failed_chunks=int(local_stats.get('failed_chunks') or 0),
        )

    remaining_draft_chunks = draft_chunks[resume_from_index:]
    remaining_structured_chunks = structured_chunks[resume_from_index:]

    if not remaining_draft_chunks:
        refined_parts = prefilled_parts
        if log_callback:
            log_callback(
                "refine_resume_already_complete",
                "📂 All refinement chunks were already processed; no LLM calls needed."
            )
    else:
        refined_parts = prefilled_parts + await refine_chunks(
            translated_chunks=remaining_draft_chunks,
            original_chunks=remaining_structured_chunks,
            target_language=target_language,
            model_name=model_name,
            api_endpoint=cli_api_endpoint,
            log_callback=log_callback,
            stats_callback=_global_stats_callback if resume_from_index else stats_callback,
            check_interruption_callback=check_interruption_callback,
            llm_provider=llm_provider,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            context_window=context_window,
            auto_adjust_context=auto_adjust_context,
            prompt_options=prompt_options,
            checkpoint_callback=_checkpoint_callback,
            chunk_index_offset=resume_from_index,
        )

    if (
        postprocess_repair_enabled(prompt_options)
        and not (check_interruption_callback and check_interruption_callback())
    ):
        def _postprocess_checkpoint_callback(chunk_index: int, draft_text: str, repaired_text: str) -> None:
            if not checkpoint_manager or not translation_id:
                return
            checkpoint_manager.save_checkpoint(
                translation_id=translation_id,
                chunk_index=chunk_index,
                original_text=draft_text,
                translated_text=repaired_text,
                chunk_data={'file_type': 'txt_refine', 'postprocess_repair': True},
                total_chunks=total_chunks,
                completed_chunks=total_chunks,
                failed_chunks=0,
            )

        repair_result = await repair_flagged_chunks(
            refined_parts=refined_parts,
            structured_chunks=structured_chunks,
            target_language=target_language,
            model_name=model_name,
            api_endpoint=cli_api_endpoint,
            llm_provider=llm_provider,
            prompt_options=prompt_options,
            log_callback=log_callback,
            check_interruption_callback=check_interruption_callback,
            checkpoint_callback=_postprocess_checkpoint_callback,
            gemini_api_key=gemini_api_key,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            mistral_api_key=mistral_api_key,
            deepseek_api_key=deepseek_api_key,
            poe_api_key=poe_api_key,
            nim_api_key=nim_api_key,
            context_window=context_window,
        )
        refined_parts = repair_result.parts
        if log_callback and repair_result.flagged_indices:
            log_callback(
                "postprocess_repair_summary",
                "🧹 Post-run repair finished: "
                f"{len(repair_result.repaired_indices)} repaired, "
                f"{len(repair_result.kept_indices)} kept, "
                f"{len(repair_result.failed_indices)} with failed attempts."
            )

    from src.config import ATTRIBUTION_ENABLED, GENERATOR_NAME, GENERATOR_SOURCE
    final_text = clean_translated_text("\n".join(refined_parts))
    if ATTRIBUTION_ENABLED and not prompt_bool(prompt_options, "suppress_attribution_footer", False):
        footer = f"\n\n{'=' * 60}\n"
        footer += f"Refined with {GENERATOR_NAME}\n"
        footer += f"{GENERATOR_SOURCE}\n"
        footer += f"{'=' * 60}\n"
        final_text += footer

    try:
        from src.utils.text_encoding import apply_normalization
        final_text = apply_normalization(final_text)
    except Exception:
        pass

    try:
        async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
            await f.write(final_text)
        if write_quality_report:
            try:
                report_path = quality_report.write(editorial_report_path(output_filepath))
                if log_callback:
                    counts = quality_report.summary_counts()
                    log_callback(
                        "editorial_quality_report_saved",
                        "Editorial quality report saved: "
                        f"{report_path.name} "
                        f"({counts['accepted']} accepted, {counts['rejected']} rejected, "
                        f"{counts['warnings']} with warnings)."
                    )
            except Exception as report_error:
                if log_callback:
                    log_callback(
                        "editorial_quality_report_error",
                        f"Could not save editorial quality report: {report_error}"
                    )
        try:
            write_fidelity_report_from_options(
                output_filepath,
                prompt_options,
                log_callback=log_callback,
            )
        except Exception as report_error:
            if log_callback:
                log_callback(
                    "fidelity_report_error",
                    f"Could not save fidelity report: {report_error}"
                )
        if log_callback:
            log_callback("refine_save_success", f"Refined output saved: '{output_filepath}'")
        return True
    except Exception as e:
        err_msg = f"ERROR: Saving output file '{output_filepath}': {e}"
        if log_callback:
            log_callback("refine_save_error", err_msg)
        else:
            print(err_msg)
        return False
