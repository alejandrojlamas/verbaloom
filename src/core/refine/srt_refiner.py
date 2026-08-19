"""SRT refine-only mode.

Runs the existing refine_subtitle_translations helper on each subtitle of
an already-translated SRT and writes a polished file. Timestamps and
subtitle indices are preserved verbatim.
"""

import os
import aiofiles
from typing import Optional, Callable, Dict, Any

from src.config import (
    DEFAULT_MODEL,
    API_ENDPOINT,
    SRT_LINES_PER_BLOCK,
)

# Disable the char cap when grouping: block sizing is purely fixed-count
# (every block holds exactly SRT_LINES_PER_BLOCK subtitles).
_NO_CHAR_CAP = 10 ** 12
from src.core.llm_client import create_llm_client
from src.core.editorial_quality import EditorialQualityReport, editorial_report_path
from src.core.fidelity_supervisor import ensure_fidelity_report, write_fidelity_report_from_options
from src.core.srt_processor import SRTProcessor
from src.core.subtitle_translator import refine_subtitle_translations


async def refine_srt_file(
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
    prompt_options: Optional[Dict[str, Any]] = None,
) -> bool:
    """Run a refinement-only pass on an already-translated SRT file."""
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input SRT file '{input_filepath}' not found."
        if log_callback:
            log_callback("srt_file_not_found", err_msg)
        return False

    try:
        async with aiofiles.open(input_filepath, 'r', encoding='utf-8') as f:
            srt_content = await f.read()
    except Exception as e:
        if log_callback:
            log_callback("srt_read_error",
                         f"ERROR: Reading SRT file '{input_filepath}': {e}")
        return False

    srt_processor = SRTProcessor()
    if not srt_processor.validate_srt(srt_content):
        if log_callback:
            log_callback("srt_invalid_format", "Invalid SRT file format")
        return False

    subtitles = srt_processor.parse_srt(srt_content)
    if not subtitles:
        if log_callback:
            log_callback("srt_no_subtitles", "No subtitles found in file")
        return False

    if log_callback:
        log_callback("srt_refine_start",
                     f"✨ Refining {len(subtitles)} subtitles in {target_language}...")
    prompt_options = dict(prompt_options or {})
    ensure_fidelity_report(
        prompt_options,
        document_name=os.path.basename(output_filepath),
        source_language=prompt_options.get('_source_language', ''),
        target_language=target_language,
        translator_model=model_name,
        translator_provider=llm_provider,
    )
    write_quality_report = prompt_options.get('editorial_quality_report', True)
    quality_report = EditorialQualityReport(
        document_name=os.path.basename(output_filepath),
        target_language=target_language,
    )
    prompt_options.setdefault('editorial_quality_guard', True)
    prompt_options['_editorial_quality_report'] = quality_report

    if stats_callback:
        stats_callback({
            'total_chunks': len(subtitles),
            'completed_chunks': 0,
            'failed_chunks': 0,
        })

    translations: Dict[int, str] = {}
    for sub in subtitles:
        try:
            idx = int(sub['number']) - 1
        except (KeyError, ValueError):
            continue
        translations[idx] = sub.get('text', '')

    llm_client = create_llm_client(
        llm_provider, gemini_api_key, cli_api_endpoint, model_name,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )

    # Fixed-count grouping for refine (no char cap): every block sent to
    # the LLM has the same shape, which keeps [N] marker accounting
    # predictable across the whole file.
    refine_blocks = srt_processor.group_subtitles_for_translation(
        subtitles, SRT_LINES_PER_BLOCK, _NO_CHAR_CAP
    )

    try:
        refined = await refine_subtitle_translations(
            translations=translations,
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            log_callback=log_callback,
            prompt_options=prompt_options,
            post_processing_instructions=(
                prompt_options.get('refinement_instructions', '')
            ),
            stats_callback=stats_callback,
            check_interruption_callback=check_interruption_callback,
            subtitle_blocks=refine_blocks,
        )
    finally:
        if llm_client:
            try:
                await llm_client.close()
            except Exception:
                pass

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("srt_refine_interrupted",
                         "Refinement interrupted before save")
        return False

    refined_subs = srt_processor.update_translated_subtitles(subtitles, refined)
    refined_srt = srt_processor.reconstruct_srt(refined_subs)

    try:
        async with aiofiles.open(output_filepath, 'w', encoding='utf-8') as f:
            await f.write(refined_srt)
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
            log_callback("srt_refine_done",
                         f"✅ Refined SRT saved: {output_filepath}")
        return True
    except Exception as e:
        if log_callback:
            log_callback("srt_save_error",
                         f"ERROR: Saving SRT file '{output_filepath}': {e}")
        return False
