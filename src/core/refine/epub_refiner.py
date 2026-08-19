"""EPUB refine-only mode.

Walks each XHTML content file of an already-translated EPUB, refines its
body in place, and repackages. Interruption stops cleanly between chunks
but does not persist partial state (no resume support in v1).
"""

import os
import tempfile
from typing import Optional, Callable, Dict, Any, List, Tuple
from lxml import etree

from src.config import (
    DEFAULT_MODEL, API_ENDPOINT, MAX_TOKENS_PER_CHUNK, THINKING_MODELS,
    ADAPTIVE_CONTEXT_INITIAL_THINKING,
)
from src.core.epub.translator import (
    _extract_epub, _parse_epub_manifest, _create_llm_client,
    _create_context_manager, _repackage_epub,
)
from src.core.epub.xhtml_translator import (
    _setup_translation, _preserve_tags, _create_chunks,
    _replace_body, _escape_stray_angle_brackets, _refine_epub_chunks,
)
from src.core.epub.container import TranslationContainer
from src.core.context_optimizer import INITIAL_CONTEXT_SIZE
from src.core.editorial_quality import EditorialQualityReport, editorial_report_path
from src.core.fidelity_supervisor import ensure_fidelity_report, write_fidelity_report_from_options
from src.core.text_transform import apply_faithful_modernize_defaults
from src.utils.text_encoding import clean_text_artifacts
from .client_setup import build_refine_client


def _globalize_chunk_text(
    chunk: Dict,
    placeholder_format: Tuple[str, str],
) -> str:
    """Convert a chunk's placeholders from local to global indices.

    `_refine_epub_chunks` expects globally-numbered placeholders because it
    re-localizes internally before sending to the LLM. In translate-mode the
    chunks already carry global indices; in refine-only mode they come fresh
    from HtmlChunker with local indices, so we re-globalize here.
    """
    text = chunk['text']
    global_indices = chunk.get('global_indices', [])
    if not global_indices:
        return text

    prefix, suffix = placeholder_format
    for local_idx, global_idx in enumerate(global_indices):
        local_ph = f"{prefix}{local_idx}{suffix}"
        text = text.replace(local_ph, f"__TEMP_GLOBAL_{global_idx}__")
    for global_idx in global_indices:
        text = text.replace(f"__TEMP_GLOBAL_{global_idx}__",
                            f"{prefix}{global_idx}{suffix}")
    return text


async def _refine_one_xhtml(
    doc_root: etree._Element,
    target_language: str,
    model_name: str,
    llm_client: Any,
    max_tokens_per_chunk: int,
    log_callback: Optional[Callable],
    context_manager: Optional[Any],
    prompt_options: Optional[Dict],
    check_interruption_callback: Optional[Callable],
    container: Optional[TranslationContainer] = None,
    stats_callback: Optional[Callable] = None,
    saved_chunks_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
    chunk_index_offset: int = 0,
    checkpoint_callback: Optional[Callable[[int, str, str, Dict[str, Any]], None]] = None,
) -> bool:
    """Refine a single parsed XHTML document in place."""
    body_html, body_element, tag_preserver = _setup_translation(
        doc_root, log_callback, container
    )
    if not body_html or body_element is None:
        if log_callback:
            log_callback("no_body", "No <body> element found")
        return False

    text_with_placeholders, global_tag_map, placeholder_format = _preserve_tags(
        body_html, tag_preserver, log_callback, protect_technical=True
    )

    chunks = _create_chunks(
        text_with_placeholders, global_tag_map, max_tokens_per_chunk,
        log_callback, container,
    )

    if not chunks:
        if log_callback:
            log_callback("no_chunks", "No translatable chunks in this XHTML, skipping")
        return True

    draft_globalized = [_globalize_chunk_text(c, placeholder_format) for c in chunks]
    restored_chunks: List[str] = []
    start_local_index = 0
    if saved_chunks_by_index:
        while start_local_index < len(draft_globalized):
            saved = saved_chunks_by_index.get(chunk_index_offset + start_local_index)
            if (
                not saved
                or saved.get('status') != 'completed'
                or saved.get('translated_text') is None
            ):
                break
            restored_chunks.append(saved.get('translated_text') or draft_globalized[start_local_index])
            start_local_index += 1

    if start_local_index:
        if log_callback:
            log_callback(
                "epub_refine_resume_prefix_restored",
                f"📂 Restored {start_local_index}/{len(chunks)} refined chunks from checkpoint."
            )
        if stats_callback:
            stats_callback({
                'total_chunks': len(chunks),
                'completed_chunks': start_local_index,
                'failed_chunks': 0,
            })

    if start_local_index < len(chunks):
        def _resume_stats_callback(stats: Dict[str, Any]) -> None:
            if not stats_callback:
                return
            stats_callback({
                **stats,
                'total_chunks': len(chunks),
                'completed_chunks': min(
                    len(chunks),
                    start_local_index + int(stats.get('completed_chunks') or 0),
                ),
            })

        refined_tail = await _refine_epub_chunks(
            translated_chunks=draft_globalized[start_local_index:],
            chunks=chunks[start_local_index:],
            target_language=target_language,
            model_name=model_name,
            llm_client=llm_client,
            context_manager=context_manager,
            placeholder_format=placeholder_format,
            log_callback=log_callback,
            prompt_options=prompt_options,
            stats_callback=_resume_stats_callback if start_local_index else stats_callback,
            check_interruption_callback=check_interruption_callback,
            checkpoint_callback=checkpoint_callback,
            chunk_index_offset=chunk_index_offset + start_local_index,
        )
        refined_chunks = restored_chunks + refined_tail
    else:
        refined_chunks = restored_chunks

    if check_interruption_callback and check_interruption_callback():
        if log_callback:
            log_callback("refine_interrupted",
                         "Refinement interrupted before reconstruction")
        return False

    if len(refined_chunks) < len(chunks):
        if log_callback:
            log_callback(
                "refine_interrupted_partial",
                f"Refinement stopped after {len(refined_chunks)}/{len(chunks)} chunks; "
                "saved checkpoint and skipped reconstruction for this file."
            )
        return False

    full_text = clean_text_artifacts(''.join(refined_chunks))
    full_text = _escape_stray_angle_brackets(full_text)
    final_html = tag_preserver.restore_tags(full_text, global_tag_map)

    return _replace_body(body_element, final_html, log_callback)


def _count_refine_chunks_for_xhtml(
    file_path: str,
    max_tokens_per_chunk: int,
) -> int:
    """Best-effort chunk count for one XHTML file before spending LLM tokens."""
    try:
        parser = etree.XMLParser(recover=True, remove_blank_text=False)
        tree = etree.parse(file_path, parser)
        doc_root = tree.getroot()
        body_html, body_element, tag_preserver = _setup_translation(doc_root, None, None)
        if not body_html or body_element is None:
            return 0
        text_with_placeholders, global_tag_map, _placeholder_format = _preserve_tags(
            body_html,
            tag_preserver,
            None,
            protect_technical=True,
        )
        chunks = _create_chunks(
            text_with_placeholders,
            global_tag_map,
            max_tokens_per_chunk,
            None,
            None,
        )
        return len(chunks)
    except Exception:
        return 0


async def refine_epub_file(
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
    prompt_options: Optional[Dict] = None,
    max_tokens_per_chunk: int = MAX_TOKENS_PER_CHUNK,
    checkpoint_manager: Any = None,
    translation_id: Optional[str] = None,
    resume_from_index: int = 0,
) -> bool:
    """Run a refinement-only pass on an already-translated EPUB."""
    if not os.path.exists(input_filepath):
        err_msg = f"ERROR: Input EPUB file '{input_filepath}' not found."
        if log_callback:
            log_callback("epub_input_file_not_found", err_msg)
        return False

    llm_client, context_manager = build_refine_client(
        model_name=model_name,
        llm_provider=llm_provider,
        cli_api_endpoint=cli_api_endpoint,
        auto_adjust_context=auto_adjust_context,
        context_window=context_window,
        gemini_api_key=gemini_api_key,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        mistral_api_key=mistral_api_key,
        deepseek_api_key=deepseek_api_key,
        poe_api_key=poe_api_key,
        nim_api_key=nim_api_key,
        log_callback=log_callback,
    )
    if llm_client is None:
        return False
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
    write_quality_report = prompt_options.get('editorial_quality_report', True)
    quality_report = EditorialQualityReport(
        document_name=os.path.basename(output_filepath),
        target_language=target_language,
    )
    prompt_options.setdefault('editorial_quality_guard', True)
    prompt_options['_editorial_quality_report'] = quality_report

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            _extract_epub(input_filepath, temp_dir, log_callback)
            manifest_data = _parse_epub_manifest(temp_dir, log_callback)

            content_files: List[str] = manifest_data['content_files']
            opf_dir: str = manifest_data['opf_dir']

            total_files = len(content_files)
            chunks_per_file: List[int] = []
            for href in content_files:
                file_path = os.path.join(opf_dir, href)
                chunks_per_file.append(
                    _count_refine_chunks_for_xhtml(file_path, max_tokens_per_chunk)
                    if os.path.exists(file_path)
                    else 0
                )
            total_refine_chunks = sum(chunks_per_file) or total_files
            saved_chunks_by_index: Dict[int, Dict[str, Any]] = {}
            if checkpoint_manager and translation_id:
                try:
                    checkpoint_data = checkpoint_manager.load_checkpoint(translation_id) or {}
                    for saved_chunk in checkpoint_data.get('chunks', []) or []:
                        if saved_chunk.get('chunk_index') is None:
                            continue
                        chunk_data = saved_chunk.get('chunk_data') or {}
                        if chunk_data.get('file_type') != 'epub_refine':
                            continue
                        saved_chunks_by_index[int(saved_chunk['chunk_index'])] = saved_chunk
                    if saved_chunks_by_index:
                        checkpoint_resume = 0
                        while True:
                            saved = saved_chunks_by_index.get(checkpoint_resume)
                            if (
                                not saved
                                or saved.get('status') != 'completed'
                                or saved.get('translated_text') is None
                            ):
                                break
                            checkpoint_resume += 1
                        resume_from_index = max(
                            int(resume_from_index or 0),
                            checkpoint_resume,
                        )
                    else:
                        # A chained refine-after pass often shares the
                        # translation_id with translation checkpoints. Those
                        # rows may have a high resume index but belong to a
                        # different phase, so refinement must start clean.
                        resume_from_index = 0
                except Exception as exc:
                    if log_callback:
                        log_callback(
                            "epub_refine_checkpoint_load_error",
                            f"⚠️ Could not load EPUB refinement checkpoint: {exc}"
                        )
                resume_from_index = max(0, min(int(resume_from_index or 0), total_refine_chunks))
                try:
                    checkpoint_manager.update_progress(
                        translation_id=translation_id,
                        current_chunk_index=resume_from_index - 1,
                        total_chunks=total_refine_chunks,
                        completed_chunks=resume_from_index,
                        failed_chunks=0,
                    )
                except Exception as exc:
                    if log_callback:
                        log_callback(
                            "epub_refine_progress_checkpoint_error",
                            f"⚠️ Could not initialize EPUB refinement checkpoint progress: {exc}"
                        )
                if resume_from_index and log_callback:
                    log_callback(
                        "epub_refine_resume_prefix_restored",
                        f"📂 Resuming EPUB refinement from chunk {resume_from_index}/{total_refine_chunks}."
                    )

            if log_callback:
                log_callback("epub_refine_start",
                             f"✨ Starting EPUB refine pass over {total_files} content files "
                             f"({total_refine_chunks} chunks)...")

            if stats_callback:
                stats_callback({
                    'total_chunks': total_refine_chunks,
                    'completed_chunks': resume_from_index,
                    'failed_chunks': 0,
                })

            completed, failed, interrupted = 0, 0, False
            completed_refine_chunks, failed_refine_chunks = resume_from_index, 0
            file_chunk_offset = 0
            for idx, href in enumerate(content_files):
                if check_interruption_callback and check_interruption_callback():
                    if log_callback:
                        log_callback("epub_refine_interrupted",
                                     f"Refinement interrupted at file {idx + 1}/{total_files}")
                    interrupted = True
                    break

                file_path = os.path.join(opf_dir, href)
                file_chunk_count = chunks_per_file[idx] if idx < len(chunks_per_file) else 0
                file_start_index = file_chunk_offset
                file_end_index = file_start_index + file_chunk_count
                file_chunk_offset = file_end_index
                if not os.path.exists(file_path):
                    if log_callback:
                        log_callback("epub_refine_missing",
                                     f"⚠️ Content file missing in EPUB: {href}, skipping")
                    failed += 1
                    failed_refine_chunks += file_chunk_count or 1
                    continue

                if log_callback:
                    log_callback("epub_refine_file",
                                 f"📄 Refining file {idx + 1}/{total_files}: {href}")

                try:
                    parser = etree.XMLParser(recover=True, remove_blank_text=False)
                    tree = etree.parse(file_path, parser)
                    doc_root = tree.getroot()
                except Exception as e:
                    if log_callback:
                        log_callback("epub_refine_parse_error",
                                     f"⚠️ Could not parse {href}: {e}")
                    failed += 1
                    failed_refine_chunks += file_chunk_count or 1
                    continue

                file_prompt_options = dict(prompt_options)
                file_prompt_options['_editorial_section_prefix'] = href
                base_failed_chunks = failed_refine_chunks

                def _file_stats_callback(stats: Dict[str, Any]) -> None:
                    if not stats_callback:
                        return
                    local_completed = int(stats.get('completed_chunks') or 0)
                    local_failed = int(stats.get('failed_chunks') or 0)
                    stats_callback({
                        **stats,
                        'total_chunks': total_refine_chunks,
                        'completed_chunks': min(
                            total_refine_chunks,
                            file_start_index + local_completed,
                        ),
                        'failed_chunks': base_failed_chunks + local_failed,
                    })

                def _checkpoint_callback(
                    global_chunk_index: int,
                    draft_text: str,
                    refined_text: str,
                    local_stats: Dict[str, Any],
                ) -> None:
                    if not checkpoint_manager or not translation_id:
                        return
                    checkpoint_manager.save_checkpoint(
                        translation_id=translation_id,
                        chunk_index=global_chunk_index,
                        original_text=draft_text,
                        translated_text=refined_text,
                        chunk_data={
                            'file_type': 'epub_refine',
                            'href': href,
                            'local_chunk_index': max(0, global_chunk_index - file_start_index),
                        },
                        total_chunks=total_refine_chunks,
                        completed_chunks=min(total_refine_chunks, global_chunk_index + 1),
                        failed_chunks=failed_refine_chunks + int(local_stats.get('failed_chunks') or 0),
                    )

                ok = await _refine_one_xhtml(
                    doc_root=doc_root,
                    target_language=target_language,
                    model_name=model_name,
                    llm_client=llm_client,
                    max_tokens_per_chunk=max_tokens_per_chunk,
                    log_callback=log_callback,
                    context_manager=context_manager,
                    prompt_options=file_prompt_options,
                    check_interruption_callback=check_interruption_callback,
                    stats_callback=_file_stats_callback,
                    saved_chunks_by_index=saved_chunks_by_index,
                    chunk_index_offset=file_start_index,
                    checkpoint_callback=_checkpoint_callback,
                )

                if ok:
                    try:
                        tree.write(file_path, xml_declaration=True,
                                   encoding='utf-8', method='xml')
                    except Exception as e:
                        if log_callback:
                            log_callback("epub_refine_write_error",
                                         f"⚠️ Could not write refined XHTML {href}: {e}")
                        failed += 1
                        continue
                    completed += 1
                    completed_refine_chunks = max(completed_refine_chunks, file_end_index)
                else:
                    failed += 1
                    failed_refine_chunks += file_chunk_count or 1
                    if check_interruption_callback and check_interruption_callback():
                        interrupted = True
                        break

                if stats_callback:
                    stats_callback({
                        'total_chunks': total_refine_chunks,
                        'completed_chunks': min(
                            total_refine_chunks,
                            completed_refine_chunks + failed_refine_chunks,
                        ),
                        'failed_chunks': failed_refine_chunks,
                    })

            from src.utils.file_utils import get_partial_output_path
            final_output = (
                get_partial_output_path(output_filepath) if interrupted
                else output_filepath
            )
            _repackage_epub(temp_dir=temp_dir,
                            output_filepath=final_output,
                            log_callback=log_callback)
            if write_quality_report:
                try:
                    report_path = quality_report.write(editorial_report_path(final_output))
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
                    final_output,
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
                log_callback("epub_refine_done",
                             f"✅ EPUB refine complete: {completed} files refined, "
                             f"{failed} failed, output: {final_output}")
            return not interrupted and failed == 0
    finally:
        if llm_client and hasattr(llm_client, 'close'):
            try:
                await llm_client.close()
            except Exception:
                pass
