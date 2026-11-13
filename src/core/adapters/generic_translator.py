"""
Generic translator orchestrator using the adapter pattern.

This module provides a unified translation workflow that works with any file format
through the FormatAdapter interface.
"""

import asyncio
from typing import Callable, Optional, Dict, Any
from pathlib import Path
from contextlib import ExitStack

from .format_adapter import FormatAdapter
from .translation_unit import TranslationUnit
from .stage_journal import JOURNAL_KEY, UnitStageJournal, stage_contract
from src.core.editorial_quality import (
    EditorialQualityReport,
    editorial_report_path,
    infer_section_title,
)
from src.core.literary_continuity import (
    export_literary_continuity_state,
    observe_literary_continuity,
    get_literary_continuity_state,
)
from src.core.fidelity_supervisor import (
    FidelityDecision,
    build_fidelity_retry_prompt_options,
    ensure_fidelity_report,
    fidelity_supervisor_enabled,
    supervise_fidelity,
    target_language_gate_issues,
    write_fidelity_report_from_options,
)
from src.core.llm_output_guard import guard_llm_output
from src.core.llm.exceptions import ProviderActionRequiredError
from src.core.epub.audit_repair import strictly_better
from src.core.review_strategy import (
    bind_runtime_strategy,
    skip_universal_refinement,
    source_audited,
    valid_saved_semantic_approval,
)
from src.core.post_processor import clean_translated_text
from src.core.quality_assurance.models import atomic_write_bytes
from src.core.book_profiles import apply_profile_glossary_corrections
from src.core.translation_memory import (
    TranslationMemoryCache,
    TranslationMemoryRequest,
)
from src.core.usage.context import reset_usage_context, set_usage_context


def _guard_adapter_output(
    text: str,
    *,
    phase: str,
    style_reference: str = "",
    log_callback=None,
) -> str:
    """Apply reader-visible LLM protocol cleanup for adapter outputs."""

    guarded = guard_llm_output(
        text or "",
        phase=phase,
        style_reference=style_reference,
    )
    if log_callback and guarded.issues:
        codes = ", ".join(issue.code for issue in guarded.issues[:4])
        log_callback(
            "llm_output_guard",
            f"⚠️ Output guard flagged {phase}: {codes}",
            data={
                "type": "llm_output_guard",
                "phase": phase,
                "issues": [issue.to_dict() for issue in guarded.issues],
                "scores": guarded.scores,
                "changed": guarded.changed,
            },
        )
    return clean_translated_text(guarded.text)


def _serializable_adapter_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return adapter config safe for checkpoint JSON storage."""
    safe = dict(config or {})
    options = safe.get('prompt_options')
    if isinstance(options, dict):
        safe['prompt_options'] = {
            key: value
            for key, value in options.items()
            if key not in {'_fidelity_report', '_editorial_quality_report'}
        }
    return safe


class GenericTranslator:
    """
    Generic orchestrator for translating files using format adapters.

    This is the single translation engine for TXT and SRT, shared by both the
    web API and the CLI (via src.core.adapters.translate_file). The legacy
    per-format functions it replaced (translate_chunks, the *_with_callbacks
    dispatchers) have been removed.

    It provides a unified workflow:
    1. Prepare file via adapter
    2. Get translation units
    3. Load checkpoint if exists
    4. Translate each unit with LLM
    5. Save each translated unit
    6. Reconstruct output file
    7. Clean up resources
    """

    def __init__(
        self,
        adapter: FormatAdapter,
        checkpoint_manager: Any,  # CheckpointManager
        translation_id: str,
        *,
        memory_path: str | Path | None = None,
    ):
        """
        Initialize the generic translator.

        Args:
            adapter: Format-specific adapter (TxtAdapter, SrtAdapter, etc.)
            checkpoint_manager: Checkpoint manager for resume capability
            translation_id: Unique identifier for this translation job
        """
        self.adapter = adapter
        self.checkpoint_manager = checkpoint_manager
        self.translation_id = translation_id
        self.memory_path = memory_path

    async def _refine_unit_inline(
        self,
        *,
        unit: TranslationUnit,
        unit_index: int,
        total_units: int,
        draft_text: str,
        previous_refined_context: str,
        target_language: str,
        model_name: str,
        llm_provider: str,
        llm_client: Any,
        prompt_options: Dict[str, Any],
        runtime_state: Dict[str, Any],
        log_callback: Optional[Callable],
        stage_journal: UnitStageJournal,
    ) -> tuple[str, Optional[FidelityDecision]]:
        """Run editorial refinement for one already-translated unit.

        This is intentionally conservative: if refinement fails, worsens
        quality, or fails fidelity, it returns the first-pass translation.
        """
        if not draft_text or len(draft_text.strip()) <= 1:
            return draft_text, None

        from src.core.translator import (
            _assess_refinement_with_editorial_guard,
            _make_refinement_request,
            _normalize_guard_text,
        )

        section = infer_section_title(
            draft_text,
            context_before=previous_refined_context,
            fallback=unit.unit_id,
        )
        refined_text = None
        try:
            refined_text, _llm_response = await stage_journal.call(
                "refinement", _make_refinement_request, kind="text_pair",
                draft_translation=draft_text,
                context_before=previous_refined_context,
                context_after="",
                previous_refined_context=previous_refined_context,
                target_language=target_language,
                model=model_name,
                llm_client=llm_client,
                log_callback=log_callback,
                has_placeholders=False,
                prompt_options=prompt_options,
                context_manager=None,
                runtime_state=runtime_state,
                section=section,
                source_text=unit.content,
            )
        except Exception as exc:
            from src.core.llm.exceptions import RateLimitError
            if isinstance(exc, RateLimitError):
                raise
            if log_callback:
                log_callback(
                    "inline_refinement_error",
                    f"Editorial pass errored for unit {unit_index + 1}/{total_units}; keeping translation: {exc}"
                )
            kept = _guard_adapter_output(
                draft_text,
                phase="inline_refinement_fallback",
                style_reference=previous_refined_context,
                log_callback=log_callback,
            )
            kept = apply_profile_glossary_corrections(
                kept,
                prompt_options,
                source_text=unit.content,
            )
            return kept, None
            raise

        if not refined_text:
            if log_callback:
                log_callback(
                    "inline_refinement_failed",
                    f"Editorial pass failed for unit {unit_index + 1}/{total_units}; keeping translation."
                )
            kept = _guard_adapter_output(
                draft_text,
                phase="inline_refinement_fallback",
                style_reference=previous_refined_context,
                log_callback=log_callback,
            )
            kept = apply_profile_glossary_corrections(
                kept,
                prompt_options,
                source_text=unit.content,
            )
            return kept, None

        source_text = unit.content
        editorial_quality_report = prompt_options.get('_editorial_quality_report')
        if prompt_options.get('editorial_quality_guard', True):
            decision, _guard_response = await stage_journal.call(
                "editorial_audit", _assess_refinement_with_editorial_guard, kind="quality",
                draft_text=draft_text,
                refined_text=refined_text,
                chunk_index=unit_index + 1,
                section=section,
                source_text=source_text,
                source_language=prompt_options.get('_source_language', ''),
                target_language=target_language,
                model=model_name,
                client=llm_client,
                log_callback=log_callback,
                prompt_options=prompt_options,
            )
            if editorial_quality_report is not None:
                editorial_quality_report.add(decision)
            if not decision.accepted:
                if log_callback:
                    reason = "; ".join(issue.code for issue in decision.rejections) or "quality_guard"
                    log_callback(
                        "inline_refinement_quality_rejected",
                        f"Editorial pass for unit {unit_index + 1}/{total_units} rejected: {reason}. Keeping translation."
                    )
                kept = _guard_adapter_output(
                    draft_text,
                    phase="inline_refinement_quality_fallback",
                    style_reference=previous_refined_context,
                    log_callback=log_callback,
                )
                kept = apply_profile_glossary_corrections(
                    kept,
                    prompt_options,
                    source_text=source_text,
                )
                return kept, None

        refined_text = _guard_adapter_output(
            refined_text,
            phase="inline_refinement",
            style_reference=previous_refined_context,
            log_callback=log_callback,
        )
        refined_text = apply_profile_glossary_corrections(
            refined_text,
            prompt_options,
            source_text=source_text,
        )
        gate_rejections = [
            issue for issue in target_language_gate_issues(
                source_text,
                refined_text,
                source_language=prompt_options.get('_source_language', ''),
                target_language=target_language,
                phase="refinement",
                prompt_options=prompt_options,
            )
            if issue.severity == "reject"
        ]
        if gate_rejections:
            if log_callback:
                reason = "; ".join(issue.code for issue in gate_rejections)
                log_callback(
                    "inline_refinement_target_language_rejected",
                    f"Editorial pass for unit {unit_index + 1}/{total_units} rejected by target-language gate: {reason}. Keeping translation."
                )
            kept = _guard_adapter_output(
                draft_text,
                phase="inline_refinement_language_fallback",
                style_reference=previous_refined_context,
                log_callback=log_callback,
            )
            kept = apply_profile_glossary_corrections(
                kept,
                prompt_options,
                source_text=source_text,
            )
            return kept, None
        fidelity_decision = None
        if (
            fidelity_supervisor_enabled(prompt_options)
            and source_text
            and _normalize_guard_text(source_text) != _normalize_guard_text(refined_text)
        ):
            fidelity_decision, _fidelity_response = await stage_journal.call(
                "refinement_audit", supervise_fidelity,
                source_text,
                refined_text,
                kind="fidelity",
                chunk_index=unit_index + 1,
                phase="refinement",
                section=section,
                source_language=prompt_options.get('_source_language', ''),
                target_language=target_language,
                primary_model=model_name,
                primary_provider=llm_provider,
                client=llm_client,
                prompt_options=prompt_options,
                log_callback=log_callback,
            )
            if not fidelity_decision.accepted:
                if log_callback:
                    reason = "; ".join(
                        issue.code for issue in fidelity_decision.rejections
                    ) or "fidelity_supervisor"
                    log_callback(
                        "inline_refinement_fidelity_rejected",
                        f"Editorial pass for unit {unit_index + 1}/{total_units} rejected by fidelity supervisor: {reason}. Keeping translation."
                    )
                kept = _guard_adapter_output(
                    draft_text,
                    phase="inline_refinement_fidelity_fallback",
                    style_reference=previous_refined_context,
                    log_callback=log_callback,
                )
                kept = apply_profile_glossary_corrections(
                    kept,
                    prompt_options,
                    source_text=source_text,
                )
                return kept, fidelity_decision

        return refined_text, fidelity_decision

    async def translate(
        self,
        source_language: str,
        target_language: str,
        model_name: str,
        llm_provider: str,
        log_callback: Optional[Callable] = None,
        stats_callback: Optional[Callable] = None,
        check_interruption_callback: Optional[Callable] = None,
        bilingual_output: bool = False,
        **llm_kwargs
    ) -> bool:
        """
        Execute the complete translation workflow.

        Args:
            source_language: Source language name
            target_language: Target language name
            model_name: LLM model identifier
            llm_provider: LLM provider name (ollama, gemini, openai, openrouter)
            log_callback: Optional callback for logging (receives type and message)
            stats_callback: Optional callback for statistics updates (receives dict with total_chunks, completed_chunks, failed_chunks)
            check_interruption_callback: Optional callback to check if translation should be interrupted
            bilingual_output: If True, output will contain both original and translated text
            **llm_kwargs: Additional LLM configuration (endpoint, api_key, etc.)

        Returns:
            True if translation completed successfully, False otherwise
        """
        unit_usage_token = None
        ownership = ExitStack()
        try:
            ownership.enter_context(self.checkpoint_manager.execution_lock(self.translation_id))
            # 1. Prepare file for translation
            if log_callback:
                log_callback("prepare_start", f"Preparing {self.adapter.format_name.upper()} file for translation")

            if not await self.adapter.prepare_for_translation():
                if log_callback:
                    reason = getattr(
                        self.adapter,
                        "last_error",
                        "Failed to prepare file for translation",
                    )
                    log_callback("prepare_failed", reason)
                return False
            input_encoding = getattr(self.adapter, "input_encoding", None)
            if log_callback and input_encoding and input_encoding not in ("utf-8", "utf-8-sig"):
                log_callback(
                    "text_encoding_fallback",
                    f"Read text input using {input_encoding} encoding."
                )

            # 2. Get translation units
            units = self.adapter.get_translation_units()
            total_units = len(units)

            if total_units == 0:
                if log_callback:
                    reason = getattr(
                        self.adapter,
                        "last_error",
                        "No translation units found in file",
                    )
                    log_callback("no_units", reason)
                return False

            if log_callback:
                log_callback("units_found", f"Found {total_units} translation units")

            # Send initial stats with total_chunks
            if stats_callback:
                stats_callback({
                    'total_chunks': total_units,
                    'completed_chunks': 0,
                    'failed_chunks': 0
                })

            # 3. Check for checkpoint and resume
            resume_from = 0
            last_context = ""
            runtime_state: dict = {}
            prompt_options = bind_runtime_strategy(llm_kwargs.get('prompt_options'), file_type=self.adapter.format_name)
            llm_kwargs['prompt_options'] = prompt_options
            memory_enabled = prompt_options.get('translation_memory_enabled', True) is not False
            translation_memory = (
                (TranslationMemoryCache(self.memory_path) if self.memory_path is not None else TranslationMemoryCache())
                if memory_enabled else None
            )
            memory_hits = 0
            structure_summary = prompt_options.get('structured_layout_summary')
            if log_callback and isinstance(structure_summary, dict):
                tables = int(structure_summary.get('tables_detected') or 0)
                figures = int(structure_summary.get('figure_text_blocks_detected') or 0)
                repairs = int(structure_summary.get('repairs_applied') or 0)
                excluded = int(structure_summary.get('excluded_blocks') or 0)
                reconstructed = int(structure_summary.get('reconstructed_blocks') or 0)
                preserved = int(structure_summary.get('preserved_blocks') or 0)
                if tables or figures or repairs or excluded or reconstructed or preserved:
                    log_callback(
                        "structured_layout_detected",
                        "📊 Structure map active: "
                        f"{tables} table(s), {figures} figure-text block(s), "
                        f"{repairs} numeric/formula repair(s), "
                        f"{excluded} excluded, {reconstructed} reconstructed, {preserved} preserved block(s)."
                    )
            if log_callback and translation_memory is not None:
                log_callback(
                    "translation_memory_enabled",
                    "🧠 Translation memory enabled for exact matching units."
                )
            fidelity_report = ensure_fidelity_report(
                prompt_options,
                document_name=self.adapter.output_file_path.name,
                source_language=source_language,
                target_language=target_language,
                translator_model=model_name,
                translator_provider=llm_provider,
            )
            checkpoint_data = self.checkpoint_manager.load_checkpoint(self.translation_id)

            if checkpoint_data:
                for saved_chunk in checkpoint_data.get('chunks', []):
                    index = saved_chunk.get('chunk_index')
                    if (type(index) is not int or not 0 <= index < total_units
                            or saved_chunk.get('original_text') != units[index].content
                            or (saved_chunk.get('chunk_data') or {}).get('chunk_index', index) != index):
                        raise ProviderActionRequiredError(
                            "The current source/segmentation does not match retained units; review the checkpoint before resuming.",
                            reason="checkpoint_source_changed", provider="checkpoint",
                        )
                if source_audited(prompt_options):
                    for saved_chunk in checkpoint_data.get('chunks', []):
                        candidate = saved_chunk.get('translated_text')
                        index = saved_chunk.get('chunk_index')
                        if candidate and (
                            type(index) is not int or not 0 <= index < len(units)
                            or saved_chunk.get('original_text') != units[index].content
                            or (saved_chunk.get('chunk_data') or {}).get('chunk_index', index) != index
                        ):
                            raise ProviderActionRequiredError(
                                "The current source units do not match the retained checkpoint. "
                                "Restore the original input before resuming.",
                                reason="checkpoint_source_changed", provider=llm_provider,
                            )
                        if candidate and not valid_saved_semantic_approval(
                            saved_chunk.get('original_text') or '', candidate,
                            (saved_chunk.get('chunk_data') or {}).get('fidelity_decision'),
                        ):
                            raise ProviderActionRequiredError(
                                "Saved units lack the full semantic approval required by this strategy. "
                                "Resume with the original strategy or review the retained candidates explicitly.",
                                reason="review_strategy_changed", provider=llm_provider,
                            )
                resume_from = await self.adapter.resume_from_checkpoint(checkpoint_data)
                restored_translated_chunks = getattr(self.adapter, "translated_chunks", None)
                if fidelity_report:
                    seen_fidelity_chunks = set()
                    for saved_chunk in checkpoint_data.get('chunks', []):
                        chunk_data = saved_chunk.get('chunk_data') or {}
                        fidelity_data = chunk_data.get('fidelity_decision')
                        if not fidelity_data:
                            continue
                        try:
                            decision = FidelityDecision.from_dict(fidelity_data)
                        except Exception:
                            continue
                        key = (decision.phase, decision.chunk_index, decision.candidate_hash)
                        if key in seen_fidelity_chunks:
                            continue
                        seen_fidelity_chunks.add(key)
                        fidelity_report.add(decision)
                if log_callback:
                    log_callback("checkpoint_resumed",
                        f"Resuming from unit {resume_from}/{total_units}")
                # Update stats with resumed progress
                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': checkpoint_data.get('actual_completed_chunks', resume_from),
                        'failed_chunks': 0
                    })
            else:
                restored_translated_chunks = None
                # 4. Create new translation job
                started = self.checkpoint_manager.start_job(
                    translation_id=self.translation_id,
                    file_type=self.adapter.format_name,
                    config={
                        'input_file_path': str(self.adapter.input_file_path),
                        'output_file_path': str(self.adapter.output_file_path),
                        'source_language': source_language,
                        'target_language': target_language,
                        'model_name': model_name,
                        'llm_provider': llm_provider,
                        **_serializable_adapter_config(self.adapter.config)
                    },
                    input_file_path=str(self.adapter.input_file_path)
                )
                if not started:
                    raise ProviderActionRequiredError("Could not create the job checkpoint.", reason="checkpoint_write_failed", provider="checkpoint")

            # 5. Create LLM client
            from src.core.llm_client import LLMClient
            from src.core.translator import generate_translation_request

            llm_client = LLMClient(
                provider_type=llm_provider,
                model=model_name,
                **llm_kwargs
            )
            refinement_client = None

            # 6. Translate each unit
            failed_count = 0
            inline_refinement = (
                bool(prompt_options.get('inline_refinement'))
                and self.adapter.format_name in {'txt', 'pdf'}
                and not skip_universal_refinement(prompt_options)
            )
            pending_refinement = None
            pending_refinement_payload = None
            if inline_refinement:
                prompt_options.setdefault('editorial_quality_guard', True)
                if prompt_options.get('editorial_quality_report', True):
                    quality_report = EditorialQualityReport(
                        document_name=self.adapter.output_file_path.name,
                        target_language=target_language,
                    )
                    prompt_options['_editorial_quality_report'] = quality_report
                refinement_client = LLMClient(
                    provider_type=llm_provider,
                    model=model_name,
                    **llm_kwargs
                )
                if log_callback:
                    log_callback(
                        "inline_refinement_start",
                        "✨ Full-flow pipeline enabled: translating and editorially refining by chunk."
                    )

            contract = stage_contract(
                units, source_language=source_language, target_language=target_language,
                model=model_name, provider=llm_provider, options=prompt_options,
                parameters={k: v for k, v in llm_kwargs.items() if k != 'prompt_options'},
            )
            saved_chunks = {c['chunk_index']: c for c in (checkpoint_data or {}).get('chunks', [])}
            journals = {}
            completed_indices = {
                i for i, chunk in saved_chunks.items()
                if chunk.get('status') == 'completed' and chunk.get('translated_text')
            }
            for index, chunk in saved_chunks.items():
                data = chunk.get('chunk_data') or {}
                if JOURNAL_KEY in data:
                    if type(index) is not int or not 0 <= index < total_units:
                        raise ProviderActionRequiredError("Saved unit index is invalid.", reason="checkpoint_corrupt", provider="checkpoint")
                    journals[index] = UnitStageJournal(
                        self.checkpoint_manager, self.translation_id, index, units[index], contract,
                        saved=data[JOURNAL_KEY], options=prompt_options, log=log_callback,
                    )
            if not self.checkpoint_manager.update_progress(self.translation_id, total_chunks=total_units):
                raise ProviderActionRequiredError("Could not persist the unit count.", reason="checkpoint_write_failed", provider="checkpoint")

            def save_checkpoint(**kwargs):
                journal = journals.get(kwargs['chunk_index'])
                if journal is not None:
                    kwargs['chunk_data'] = {**(kwargs.get('chunk_data') or {}), JOURNAL_KEY: journal.data}
                next_completed = set(completed_indices)
                if kwargs.get('translated_text'):
                    next_completed.add(kwargs['chunk_index'])
                else:
                    next_completed.discard(kwargs['chunk_index'])
                kwargs['completed_chunks'] = len(next_completed)
                if not self.checkpoint_manager.save_checkpoint(**kwargs):
                    raise ProviderActionRequiredError("Could not persist the completed unit.", reason="checkpoint_write_failed", provider="checkpoint")
                completed_indices.clear()
                completed_indices.update(next_completed)

            def ensure_continuity(unit):
                if 'literary_continuity_state' not in runtime_state:
                    get_literary_continuity_state(
                        prompt_options=prompt_options, runtime_state=runtime_state,
                        sample_text=unit.content, source_language=source_language,
                        target_language=target_language, log_callback=log_callback,
                    )

            ensure_continuity(units[0])

            def accept_context(unit, text, phase):
                nonlocal last_context
                ensure_continuity(unit)
                last_context = text[-200:]
                observe_literary_continuity(
                    runtime_state=runtime_state, source_text=unit.content,
                    translated_text=text,
                    section=infer_section_title(unit.content, context_before=unit.context_before),
                    phase=phase,
                )

            # A sparse checkpoint may contain accepted units AFTER the first
            # hole. Rebuild only the prefix, not a stale future-facing snapshot.
            for index in range(resume_from):
                if index not in saved_chunks:
                    raise ProviderActionRequiredError("The checkpoint prefix is missing a retained unit.", reason="checkpoint_corrupt", provider="checkpoint")
                saved = saved_chunks[index]
                accept_context(units[index], saved['translated_text'],
                               'refinement' if (saved.get('chunk_data') or {}).get('inline_refinement') else 'translation')

            def build_translation_context() -> Dict[str, Any]:
                context = {'previous_translation_context': last_context}
                continuity_state = export_literary_continuity_state(runtime_state)
                if continuity_state:
                    context['literary_continuity_state'] = continuity_state
                return context

            def build_memory_request(
                unit: TranslationUnit,
                previous_translation_context: str,
            ) -> TranslationMemoryRequest:
                return TranslationMemoryRequest(
                    source_text=unit.content,
                    context_before=unit.context_before,
                    context_after=unit.context_after,
                    previous_translation_context=previous_translation_context,
                    source_language=source_language,
                    target_language=target_language,
                    provider=llm_provider,
                    model=model_name,
                    prompt_options=prompt_options,
                    format_name=self.adapter.format_name,
                    unit_id=unit.unit_id,
                    phase="inline_refined" if inline_refinement else "translation",
                )

            async def finalize_pending_refinement() -> None:
                nonlocal pending_refinement, pending_refinement_payload, last_context, failed_count
                if pending_refinement is None or pending_refinement_payload is None:
                    return

                payload = pending_refinement_payload
                refined_content, refinement_fidelity_decision = await pending_refinement
                unit = payload['unit']
                unit_index = payload['unit_index']
                translation_fidelity_decision = payload.get('translation_fidelity_decision')
                memory_request = payload.get('memory_request')

                save_success = await self.adapter.save_unit_translation(
                    unit.unit_id,
                    refined_content
                )
                if not save_success:
                    if log_callback:
                        log_callback("save_failed",
                            f"Failed to save translation for unit {unit.unit_id}")
                    failed_count += 1
                    pending_refinement = None
                    pending_refinement_payload = None
                    return

                accept_context(unit, refined_content, 'refinement')

                chunk_data = dict(unit.metadata)
                if translation_fidelity_decision is not None:
                    chunk_data['fidelity_decision'] = translation_fidelity_decision.to_dict()
                if refinement_fidelity_decision is not None:
                    chunk_data['refinement_fidelity_decision'] = refinement_fidelity_decision.to_dict()
                chunk_data['inline_refinement'] = True

                if translation_memory is not None and memory_request is not None:
                    cache_key = translation_memory.put(
                        memory_request,
                        refined_content,
                        metadata={
                            **unit.metadata,
                            'inline_refinement': True,
                            'format_name': self.adapter.format_name,
                        },
                    )
                    if cache_key:
                        chunk_data['translation_memory_key'] = cache_key[:16]

                save_checkpoint(
                    translation_id=self.translation_id,
                    chunk_index=unit_index,
                    original_text=unit.content,
                    translated_text=refined_content,
                    chunk_data=chunk_data,
                    translation_context=build_translation_context(),
                    total_chunks=total_units,
                    completed_chunks=unit_index + 1,
                    failed_chunks=failed_count
                )

                if stats_callback:
                    stats_callback({
                        'total_chunks': total_units,
                        'completed_chunks': len(completed_indices),
                        'failed_chunks': failed_count
                    })

                if log_callback:
                    log_callback("unit_complete",
                        f"Unit {unit_index+1}/{total_units} translated and refined successfully")

                pending_refinement = None
                pending_refinement_payload = None

            for i, unit in enumerate(units):
                if unit_usage_token is not None:
                    reset_usage_context(unit_usage_token)
                unit_usage_token = set_usage_context(unit_id=unit.unit_id)
                if i < resume_from:
                    continue
                # Adjacent units depend on the accepted final context. Starting
                # the next translation against a provisional draft changes its
                # inputs on resume and can repeat an already received result.
                if inline_refinement:
                    await finalize_pending_refinement()
                if (
                    isinstance(restored_translated_chunks, list)
                    and i < len(restored_translated_chunks)
                    and restored_translated_chunks[i]
                ):
                    restored_text = restored_translated_chunks[i]
                    accept_context(unit, restored_text,
                                   'refinement' if (saved_chunks[i].get('chunk_data') or {}).get('inline_refinement') else 'translation')
                    if log_callback:
                        log_callback(
                            "checkpoint_unit_skipped",
                            f"Skipping already completed unit {i+1}/{total_units} from checkpoint."
                        )
                    continue

                # Check for interruption at the start of each unit
                if check_interruption_callback and check_interruption_callback():
                    if log_callback:
                        log_callback("translation_interrupted",
                            f"Translation interrupted at unit {i+1}/{total_units}")
                    if inline_refinement and pending_refinement is not None:
                        if log_callback:
                            log_callback(
                                "inline_refinement_drain",
                                "Waiting for in-flight editorial pass before pausing."
                            )
                        await finalize_pending_refinement()

                    # The checkpoint retains partial work. Do not overwrite a
                    # published file or fill its missing units with the source.
                    if log_callback:
                        log_callback("checkpoint_retained", "Partial work retained in the checkpoint; no incomplete output was published.")

                    # Mark as paused/interrupted
                    self.checkpoint_manager.mark_paused(self.translation_id)
                    return False

                if log_callback:
                    log_callback("unit_start",
                        f"Translating unit {i+1}/{total_units} ({unit.unit_id})")

                ensure_continuity(unit)
                previous_context_for_unit = last_context
                memory_request = (
                    build_memory_request(unit, previous_context_for_unit)
                    if translation_memory is not None
                    else None
                )
                if translation_memory is not None and memory_request is not None:
                    cached_entry = translation_memory.get(memory_request)
                    if cached_entry is not None:
                        if inline_refinement:
                            await finalize_pending_refinement()
                        cached_text = _guard_adapter_output(
                            cached_entry.translated_text,
                            phase="translation_memory",
                            style_reference=last_context,
                            log_callback=log_callback,
                        )
                        cached_text = apply_profile_glossary_corrections(
                            cached_text,
                            prompt_options,
                            source_text=unit.content,
                        )
                        cached_approval = (cached_entry.metadata.get('metadata') or {}).get('fidelity_decision')
                        if source_audited(prompt_options) and not valid_saved_semantic_approval(
                            unit.content, cached_text, cached_approval,
                        ):
                            # Cleanup/glossary changes invalidate the candidate, not its translation.
                            decision, _ = await supervise_fidelity(
                                unit.content, cached_text, chunk_index=i + 1,
                                phase="translation_memory", section=unit.unit_id,
                                source_language=source_language, target_language=target_language,
                                primary_model=model_name, primary_provider=llm_provider,
                                client=llm_client, prompt_options=prompt_options, log_callback=log_callback,
                            )
                            if not decision.accepted:
                                raise ProviderActionRequiredError(
                                    "The retained memory candidate requires editorial review.",
                                    reason="cached_candidate_rejected", provider=llm_provider,
                                )
                            cached_approval = decision.to_dict()
                            translation_memory.put(memory_request, cached_text, metadata={
                                **unit.metadata, 'format_name': self.adapter.format_name,
                                'fidelity_decision': cached_approval,
                            })
                        gate_rejections = [
                            issue for issue in target_language_gate_issues(
                                unit.content,
                                cached_text,
                                source_language=source_language,
                                target_language=target_language,
                                phase="translation_memory",
                                prompt_options=prompt_options,
                            )
                            if issue.severity == "reject"
                        ]
                        if gate_rejections:
                            translation_memory.delete(memory_request)
                            if log_callback:
                                reason = "; ".join(issue.code for issue in gate_rejections)
                                log_callback(
                                    "translation_memory_target_language_rejected",
                                    f"🧠 Cached translation for unit {i+1}/{total_units} rejected by target-language gate: {reason}. Regenerating."
                                )
                        else:
                            save_success = await self.adapter.save_unit_translation(
                                unit.unit_id,
                                cached_text
                            )
                            if not save_success:
                                if log_callback:
                                    log_callback("save_failed",
                                        f"Failed to save cached translation for unit {unit.unit_id}")
                                failed_count += 1
                                continue

                            memory_hits += 1
                            accept_context(unit, cached_text, 'translation_memory')
                            chunk_data = {
                                **unit.metadata,
                                'translation_memory_hit': True,
                                'translation_memory_key': cached_entry.key[:16],
                            }
                            if source_audited(prompt_options):
                                chunk_data['fidelity_decision'] = cached_approval
                            if inline_refinement:
                                chunk_data['inline_refinement'] = True
                            save_checkpoint(
                                translation_id=self.translation_id,
                                chunk_index=i,
                                original_text=unit.content,
                                translated_text=cached_text,
                                chunk_data=chunk_data,
                                translation_context=build_translation_context(),
                                total_chunks=total_units,
                                completed_chunks=i + 1,
                                failed_chunks=failed_count
                            )
                            if stats_callback:
                                stats_callback({
                                    'total_chunks': total_units,
                                    'completed_chunks': len(completed_indices),
                                    'failed_chunks': failed_count
                                })
                            if log_callback:
                                log_callback(
                                    "translation_memory_hit",
                                    f"🧠 Reused cached translation for unit {i+1}/{total_units}"
                                )
                            continue

                # Translate unit
                saved = saved_chunks.get(i) or {}
                journal = journals.get(i) or UnitStageJournal(
                    self.checkpoint_manager, self.translation_id, i, unit, contract,
                    saved=(saved.get('chunk_data') or {}).get(JOURNAL_KEY),
                    options=prompt_options, log=log_callback,
                )
                journals[i] = journal
                try:
                    translated_content = await journal.call(
                        "translation", generate_translation_request,
                        main_content=unit.content,
                        context_before=unit.context_before,
                        context_after=unit.context_after,
                        previous_translation_context=previous_context_for_unit,
                        source_language=source_language,
                        target_language=target_language,
                        model=model_name,
                        llm_client=llm_client,
                        log_callback=log_callback,
                        prompt_options=prompt_options,
                        runtime_state=runtime_state,
                    )

                    if translated_content:
                        translated_content = _guard_adapter_output(
                            translated_content,
                            phase="translation",
                            style_reference=previous_context_for_unit,
                            log_callback=log_callback,
                        )
                        translated_content = apply_profile_glossary_corrections(
                            translated_content,
                            prompt_options,
                            source_text=unit.content,
                        )
                        fidelity_decision = None
                        if fidelity_supervisor_enabled(prompt_options):
                            fidelity_decision, _ = await journal.call(
                                "translation_audit", supervise_fidelity,
                                unit.content,
                                translated_content,
                                kind="fidelity",
                                chunk_index=i + 1,
                                phase="translation",
                                section=unit.unit_id,
                                source_language=source_language,
                                target_language=target_language,
                                primary_model=model_name,
                                primary_provider=llm_provider,
                                client=llm_client,
                                prompt_options=prompt_options,
                                log_callback=log_callback,
                            )
                            if (
                                not fidelity_decision.accepted
                                and prompt_options.get('fidelity_supervisor_retry', True) is not False
                            ):
                                retry_prompt_options = build_fidelity_retry_prompt_options(
                                    prompt_options,
                                    fidelity_decision,
                                )
                                if log_callback:
                                    log_callback(
                                        "fidelity_retry",
                                        f"Retrying unit {i+1}/{total_units} after fidelity rejection"
                                    )
                                retry_content = await journal.call(
                                    "translation_retry", generate_translation_request,
                                    main_content=unit.content,
                                    context_before=unit.context_before,
                                    context_after=unit.context_after,
                                    previous_translation_context=last_context,
                                    source_language=source_language,
                                    target_language=target_language,
                                    model=model_name,
                                    llm_client=llm_client,
                                    log_callback=log_callback,
                                    prompt_options=retry_prompt_options,
                                    runtime_state=runtime_state,
                                )
                                if retry_content:
                                    retry_content = _guard_adapter_output(
                                        retry_content,
                                        phase="translation_retry",
                                        style_reference=previous_context_for_unit,
                                        log_callback=log_callback,
                                    )
                                    retry_content = apply_profile_glossary_corrections(
                                        retry_content,
                                        prompt_options,
                                        source_text=unit.content,
                                    )
                                    retry_decision, _ = await journal.call(
                                        "translation_retry_audit", supervise_fidelity,
                                        unit.content,
                                        retry_content,
                                        kind="fidelity",
                                        chunk_index=i + 1,
                                        phase="translation_retry",
                                        section=unit.unit_id,
                                        source_language=source_language,
                                        target_language=target_language,
                                        primary_model=model_name,
                                        primary_provider=llm_provider,
                                        client=llm_client,
                                        prompt_options=prompt_options,
                                        log_callback=log_callback,
                                    )
                                    if strictly_better(fidelity_decision, retry_decision):
                                        translated_content = retry_content
                                        fidelity_decision = retry_decision

                            if fidelity_decision and not fidelity_decision.accepted:
                                if inline_refinement:
                                    await finalize_pending_refinement()
                                if log_callback:
                                    reason = "; ".join(
                                        issue.code for issue in fidelity_decision.rejections
                                    ) or "fidelity_supervisor"
                                    log_callback(
                                        "translation_fidelity_rejected",
                                        f"Unit {i+1}/{total_units} rejected by fidelity supervisor: {reason}"
                                    )
                                failed_count += 1
                                chunk_data = {
                                    **unit.metadata,
                                    'fidelity_decision': fidelity_decision.to_dict(),
                                }
                                save_checkpoint(
                                    translation_id=self.translation_id,
                                    chunk_index=i,
                                    original_text=unit.content,
                                    translated_text=None,
                                    chunk_data=chunk_data,
                                    translation_context=build_translation_context(),
                                    total_chunks=total_units,
                                    failed_chunks=failed_count
                                )
                                if stats_callback:
                                    stats_callback({
                                        'total_chunks': total_units,
                                        'completed_chunks': len(completed_indices),
                                        'failed_chunks': failed_count
                                    })
                                continue

                        if inline_refinement:
                            pending_refinement_payload = {
                                'unit': unit,
                                'unit_index': i,
                                'translation_fidelity_decision': fidelity_decision,
                                'memory_request': memory_request,
                            }
                            pending_refinement = asyncio.create_task(
                                self._refine_unit_inline(
                                    unit=unit,
                                    unit_index=i,
                                    total_units=total_units,
                                    draft_text=translated_content,
                                    previous_refined_context=last_context,
                                    target_language=target_language,
                                    model_name=model_name,
                                    llm_provider=llm_provider,
                                    llm_client=refinement_client,
                                    prompt_options=prompt_options,
                                    runtime_state=runtime_state,
                                    log_callback=log_callback,
                                    stage_journal=journal,
                                )
                            )
                            if log_callback:
                                log_callback(
                                    "inline_refinement_queued",
                                    f"Editorial pass queued for unit {i+1}/{total_units}"
                                )
                            continue

                        # Save via adapter
                        save_success = await self.adapter.save_unit_translation(
                            unit.unit_id,
                            translated_content
                        )

                        if not save_success:
                            if log_callback:
                                log_callback("save_failed",
                                    f"Failed to save translation for unit {unit.unit_id}")
                            failed_count += 1
                            continue

                        # Update context for next unit before checkpointing so
                        # resume keeps both immediate flow and long memory.
                        accept_context(unit, translated_content, 'translation')

                        # Save checkpoint
                        chunk_data = dict(unit.metadata)
                        if fidelity_decision is not None:
                            chunk_data['fidelity_decision'] = fidelity_decision.to_dict()
                        if translation_memory is not None and memory_request is not None:
                            cache_key = translation_memory.put(
                                memory_request,
                                translated_content,
                                metadata={
                                    **unit.metadata,
                                    'format_name': self.adapter.format_name,
                                    'fidelity_decision': fidelity_decision.to_dict() if fidelity_decision is not None else None,
                                },
                            )
                            if cache_key:
                                chunk_data['translation_memory_key'] = cache_key[:16]
                        save_checkpoint(
                            translation_id=self.translation_id,
                            chunk_index=i,
                            original_text=unit.content,
                            translated_text=translated_content,
                            chunk_data=chunk_data,
                            translation_context=build_translation_context(),
                            total_chunks=total_units,
                            completed_chunks=i + 1
                        )

                        # Update stats
                        if stats_callback:
                            stats_callback({
                                'total_chunks': total_units,
                                'completed_chunks': len(completed_indices),
                                'failed_chunks': failed_count
                            })

                        if log_callback:
                            log_callback("unit_complete",
                                f"Unit {i+1}/{total_units} translated successfully")

                    else:
                        if inline_refinement:
                            await finalize_pending_refinement()
                        # Translation failed
                        if log_callback:
                            log_callback("unit_failed",
                                f"Failed to translate unit {i+1}/{total_units}")

                        failed_count += 1

                        # Save checkpoint with failure
                        save_checkpoint(
                            translation_id=self.translation_id,
                            chunk_index=i,
                            original_text=unit.content,
                            translated_text=None,
                            chunk_data=unit.metadata,
                            translation_context=build_translation_context(),
                            total_chunks=total_units,
                            failed_chunks=failed_count
                        )

                        # Update stats with failure
                        if stats_callback:
                            stats_callback({
                                'total_chunks': total_units,
                                'completed_chunks': len(completed_indices),
                                'failed_chunks': failed_count
                            })

                except Exception as e:
                    # Re-raise RateLimitError to trigger auto-pause
                    from src.core.llm.exceptions import RateLimitError
                    if isinstance(e, RateLimitError):
                        if inline_refinement and pending_refinement is not None:
                            await finalize_pending_refinement()
                        raise

                    if log_callback:
                        log_callback("unit_error",
                            f"Error translating unit {i+1}/{total_units}: {str(e)}")
                    if inline_refinement:
                        await finalize_pending_refinement()
                    failed_count += 1

                    # Save checkpoint with failure
                    save_checkpoint(
                        translation_id=self.translation_id,
                        chunk_index=i,
                        original_text=unit.content,
                        translated_text=None,
                        chunk_data=unit.metadata,
                        translation_context=build_translation_context(),
                        total_chunks=total_units,
                        failed_chunks=failed_count
                    )

                    # Update stats with failure
                    if stats_callback:
                        stats_callback({
                            'total_chunks': total_units,
                            'completed_chunks': len(completed_indices),
                            'failed_chunks': failed_count
                        })

            if inline_refinement:
                await finalize_pending_refinement()

            completed = {
                chunk['chunk_index'] for chunk in self.checkpoint_manager.db.get_chunks(self.translation_id)
                if chunk['status'] == 'completed' and chunk.get('translated_text')
                and type(chunk['chunk_index']) is int and 0 <= chunk['chunk_index'] < total_units
                and chunk.get('original_text') == units[chunk['chunk_index']].content
            }
            if failed_count or completed != set(range(total_units)):
                self.checkpoint_manager.mark_partial(self.translation_id)
                if log_callback:
                    log_callback("publication_incomplete", f"Publication blocked: {len(completed)}/{total_units} approved units; retained candidates need review.")
                return False

            # 7. Reconstruct output file
            if log_callback:
                log_callback("reconstruct_start", "Reconstructing output file")

            try:
                output_bytes = await self.adapter.reconstruct_output(bilingual=bilingual_output)

                # Save final file
                atomic_write_bytes(self.adapter.output_file_path, output_bytes)

                if log_callback:
                    log_callback("reconstruct_complete",
                        f"Output file written to {self.adapter.output_file_path}")

                try:
                    from src.core.literary_continuity import write_literary_continuity_report
                    report_path = write_literary_continuity_report(
                        self.adapter.output_file_path,
                        runtime_state,
                    )
                    if report_path and log_callback:
                        log_callback(
                            "literary_continuity_report",
                            f"📚 Literary continuity report written: {report_path.name}",
                        )
                except Exception as e:
                    if log_callback:
                        log_callback(
                            "literary_continuity_report_error",
                            f"⚠️ Could not write literary continuity report: {e}",
                        )
                if inline_refinement and prompt_options.get('_editorial_quality_report') is not None:
                    try:
                        report = prompt_options.get('_editorial_quality_report')
                        report_path = report.write(editorial_report_path(self.adapter.output_file_path))
                        if log_callback:
                            counts = report.summary_counts()
                            log_callback(
                                "editorial_quality_report_saved",
                                "Editorial quality report saved: "
                                f"{report_path.name} "
                                f"({counts['accepted']} accepted, {counts['rejected']} rejected, "
                                f"{counts['warnings']} with warnings)."
                            )
                    except Exception as e:
                        if log_callback:
                            log_callback(
                                "editorial_quality_report_error",
                                f"Could not save editorial quality report: {e}"
                            )
                try:
                    write_fidelity_report_from_options(
                        self.adapter.output_file_path,
                        prompt_options,
                        log_callback=log_callback,
                    )
                except Exception as e:
                    if log_callback:
                        log_callback(
                            "fidelity_report_error",
                            f"Could not write fidelity report: {e}",
                        )

            except Exception as e:
                if log_callback:
                    log_callback("reconstruct_failed",
                        f"Failed to reconstruct output: {str(e)}")
                return False

            # 8. Cleanup
            await self.adapter.cleanup()

            # 9. Mark job as completed
            if log_callback and translation_memory is not None:
                log_callback(
                    "translation_memory_summary",
                    f"🧠 Translation memory reused {memory_hits}/{total_units} unit(s)."
                )
            if failed_count == 0:
                self.checkpoint_manager.mark_completed(self.translation_id)
                if log_callback:
                    log_callback("translation_complete",
                        f"Translation completed successfully: {total_units} units")
                return True
            else:
                if log_callback:
                    log_callback("translation_partial",
                        f"Translation completed with {failed_count} failures out of {total_units} units")
                return False

        except Exception as e:
            # Re-raise RateLimitError to trigger auto-pause
            from src.core.llm.exceptions import RateLimitError
            if isinstance(e, RateLimitError):
                raise
            if log_callback:
                log_callback("translation_error", f"Translation error: {str(e)}")
            return False
        finally:
            try:
                pending_task = locals().get('pending_refinement')
                if pending_task is not None:
                    if not pending_task.done():
                        pending_task.cancel()
                    await asyncio.gather(pending_task, return_exceptions=True)
                if unit_usage_token is not None:
                    reset_usage_context(unit_usage_token)
                for client_name in ('llm_client', 'refinement_client'):
                    client = locals().get(client_name)
                    if client is not None and hasattr(client, 'close'):
                        try:
                            await client.close()
                        except Exception:
                            pass
                # Ensure cleanup even on error
                try:
                    await self.adapter.cleanup()
                except Exception:
                    pass
            finally:
                ownership.close()

    def __repr__(self) -> str:
        return (
            f"GenericTranslator("
            f"id={self.translation_id}, "
            f"adapter={self.adapter})"
        )
