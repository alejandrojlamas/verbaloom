"""Runtime configuration helpers for document jobs.

These helpers are intentionally outside Flask handlers so the job orchestration
can resolve profiles, guards, sanitizer defaults, and resume behavior without
being coupled to API routes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict

from src.core.book_profiles import (
    infer_profile_id_from_metadata,
    load_book_profile,
    profile_matches_source_metadata,
)
from src.core.layout_sanitizer import (
    LAYOUT_SANITIZER_VERSION,
    should_use_sanitized_text_pipeline,
)
from src.core.text_transform import apply_faithful_modernize_defaults


def usage_process_type(config: Dict[str, Any]) -> str:
    if config.get('refine_only'):
        return 'editorial_refinement'
    prompt_options = config.get('prompt_options') or {}
    transform_mode = str(prompt_options.get('text_transform_mode') or '').strip().lower()
    if transform_mode:
        return f'transform_{transform_mode}'
    if config.get('refine_after'):
        return 'translation_with_refinement'
    return 'translation'


def configure_editorial_guard_options(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set guard defaults without requiring extra UI controls."""
    prompt_options = config.setdefault('prompt_options', {})
    # The legacy EPUB engine performs review and final source-aware audit per
    # XHTML before assembly. Mark that as inline so the Flask job layer does
    # not launch a second full-book refinement over the already reviewed EPUB.
    if (
        str(config.get('file_type') or '').strip().lower() == 'epub'
        and not config.get('refine_only')
        and prompt_options.get('refine')
    ):
        prompt_options['inline_refinement'] = True
    if config.get('source_language'):
        prompt_options.setdefault('_source_language', config.get('source_language'))

    validate_active_profile_scope(config, prompt_options)
    activate_inferred_book_profile(config, prompt_options)
    apply_faithful_modernize_defaults(prompt_options)

    if (config.get('llm_provider') or '').lower() == 'deepseek':
        uses_profile = (
            (prompt_options.get('editorial_mode') or '').lower() == 'book_profile'
            and bool(str(prompt_options.get('profile_id') or '').strip())
        )
        uses_transform_profile = uses_profile and bool(
            str(prompt_options.get('text_transform_mode') or '').strip()
        )
        if uses_profile:
            apply_active_profile_runtime_defaults(prompt_options)
            strength = profile_strength(prompt_options)
            if (
                not uses_transform_profile
                and not prompt_options.get('translation_profile_llm_audit')
                and strength != 'strict'
            ):
                prompt_options['profile_audit_enabled'] = False
        prompt_options.setdefault('quality_alert_model', 'deepseek-v4-pro')
        prompt_options.setdefault('source_aware_editorial_guard_model', 'deepseek-v4-pro')
        if uses_transform_profile:
            prompt_options['transform_auditor_model'] = 'deepseek-v4-pro'
            prompt_options['source_aware_editorial_guard'] = False
            prompt_options['source_aware_editorial_guard_mode'] = 'off'
            prompt_options['fidelity_supervisor_mode'] = 'always'
            prompt_options['fidelity_supervisor_model'] = 'deepseek-v4-pro'
            prompt_options['profile_audit_model'] = 'deepseek-v4-pro'
            prompt_options['profile_repair_model'] = 'deepseek-v4-pro'
        else:
            prompt_options.setdefault('source_aware_editorial_guard_mode', 'alerted')
            prompt_options.setdefault('fidelity_supervisor_mode', 'alerted')
            prompt_options.setdefault('fidelity_supervisor_model', 'deepseek-v4-pro')
            if uses_profile:
                prompt_options.setdefault('profile_audit_enabled', False)
                prompt_options.setdefault('profile_repair_model', 'deepseek-v4-pro')
        prompt_options.setdefault('fidelity_supervisor_retry', True)
    return prompt_options


def validate_active_profile_scope(
    config: Dict[str, Any],
    prompt_options: Dict[str, Any],
) -> None:
    """Prevent silent reuse of a generated profile from a different book."""
    profile_id = str(prompt_options.get('profile_id') or '').strip()
    if not profile_id:
        prompt_options.pop('_profile_scope_corrected_from', None)
        prompt_options.pop('_profile_scope_correction', None)
        return
    profile = load_book_profile(profile_id, allow_missing=True)
    if profile is None or not bool(profile.raw_config.get('generated_profile')):
        prompt_options.pop('_profile_scope_corrected_from', None)
        prompt_options.pop('_profile_scope_correction', None)
        return

    metadata = {
        'output_filename': config.get('output_filename'),
        'file_path': config.get('file_path'),
        'original_filename': config.get('original_filename'),
        'input_filename': config.get('input_filename'),
    }
    if profile_matches_source_metadata(profile_id, metadata):
        previous_profile_id = str(
            prompt_options.get('_profile_scope_corrected_from') or ''
        ).strip()
        correction = str(prompt_options.get('_profile_scope_correction') or '').strip()
        if correction != 'replaced' or not previous_profile_id or previous_profile_id == profile_id:
            prompt_options.pop('_profile_scope_corrected_from', None)
            prompt_options.pop('_profile_scope_correction', None)
        return

    replacement = infer_profile_id_from_metadata(metadata)
    prompt_options['_profile_scope_corrected_from'] = profile_id
    if replacement and replacement != profile_id:
        prompt_options['profile_id'] = replacement
        prompt_options['editorial_mode'] = 'book_profile'
        prompt_options['use_profile_glossary'] = True
        prompt_options['allow_cross_profile_glossary'] = False
        prompt_options['_profile_scope_correction'] = 'replaced'
        return

    prompt_options.pop('profile_id', None)
    prompt_options.pop('editorial_mode', None)
    prompt_options['use_profile_glossary'] = False
    prompt_options['_profile_scope_correction'] = 'removed'


def apply_active_profile_runtime_defaults(prompt_options: Dict[str, Any]) -> None:
    profile_id = str(prompt_options.get('profile_id') or '').strip()
    if not profile_id:
        return
    profile = load_book_profile(profile_id, allow_missing=True)
    if profile is None:
        return
    prompt_options.setdefault('target_locale', profile.target_locale or 'es-MX')
    prompt_options.setdefault('modernization_strength', profile.modernization_strength or 'high')
    prompt_options.setdefault(
        'profile_strength',
        str(profile.raw_config.get('profile_strength') or 'balanced').strip().lower(),
    )
    prompt_options.setdefault(
        'profile_audit_enabled',
        profile_strength(prompt_options) == 'strict',
    )
    prompt_options.setdefault('repair_until_pass', True)
    prompt_options.setdefault(
        'abort_on_profile_fail',
        bool(profile.raw_config.get('abort_on_profile_fail', True)),
    )
    prompt_options.setdefault('min_dimension_score', profile.min_dimension_score)
    prompt_options.setdefault('max_repair_rounds', profile.max_repair_rounds)
    _apply_profile_strength_defaults(prompt_options)


def profile_strength(prompt_options: Dict[str, Any]) -> str:
    value = str(prompt_options.get('profile_strength') or '').strip().lower()
    if value in {'off', 'none'}:
        return 'off'
    if value in {'light', 'lite', 'prompt'}:
        return 'light'
    if value in {'strict', 'full', 'llm'}:
        return 'strict'
    return 'balanced'


def _apply_profile_strength_defaults(prompt_options: Dict[str, Any]) -> None:
    strength = profile_strength(prompt_options)
    prompt_options['profile_strength'] = strength
    if strength == 'off':
        prompt_options['use_profile_glossary'] = False
        prompt_options['profile_local_precheck_enabled'] = False
        prompt_options['profile_audit_enabled'] = False
        prompt_options['repair_until_pass'] = False
        prompt_options['max_repair_rounds'] = 0
    elif strength == 'light':
        prompt_options['profile_local_precheck_enabled'] = False
        prompt_options['profile_audit_enabled'] = False
        prompt_options['repair_until_pass'] = False
        prompt_options['max_repair_rounds'] = 0
    elif strength == 'strict':
        prompt_options.setdefault('profile_local_precheck_enabled', True)
        prompt_options['profile_audit_enabled'] = True
        prompt_options.setdefault('repair_until_pass', True)
    else:
        prompt_options.setdefault('profile_local_precheck_enabled', True)
        prompt_options.setdefault('profile_audit_enabled', False)
        prompt_options.setdefault('repair_until_pass', True)
        prompt_options.setdefault('max_repair_rounds', 1)


def language_key(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = "".join(
        char
        for char in unicodedata.normalize("NFKD", raw)
        if not unicodedata.combining(char)
    )
    normalized = normalized.replace("_", "-")
    first_token = re.split(r"[\s(/,;]+", normalized, maxsplit=1)[0]
    if (
        first_token in {"auto", "autodetect", "autodetectar", "detect"}
        or normalized.startswith("detectar")
        or normalized.startswith("detect automatically")
    ):
        return "auto"
    if first_token in {"es", "spa", "spanish", "espanol"} or normalized.startswith("es-"):
        return "spanish"
    if first_token in {"en", "eng", "english"} or normalized.startswith("en-"):
        return "english"
    return first_token


def profile_matches_intralingual_request(
    config: Dict[str, Any],
    prompt_options: Dict[str, Any],
    *,
    profile_target_locale: str,
) -> bool:
    source_key = language_key(
        config.get("source_language")
        or prompt_options.get("_source_language")
        or prompt_options.get("source_language")
    )
    target_key = language_key(
        config.get("target_language")
        or prompt_options.get("target_language")
        or prompt_options.get("target_locale")
    )
    profile_target_key = language_key(profile_target_locale)

    if not target_key or not profile_target_key or target_key != profile_target_key:
        return False
    return source_key in {"", "auto", target_key}


def profile_matches_translation_request(
    config: Dict[str, Any],
    prompt_options: Dict[str, Any],
    *,
    profile_target_locale: str,
) -> bool:
    target_key = language_key(
        config.get("target_language")
        or prompt_options.get("target_language")
        or prompt_options.get("target_locale")
    )
    profile_target_key = language_key(profile_target_locale)
    return bool(target_key) and (not profile_target_key or target_key == profile_target_key)


def activate_profile_for_translation(
    prompt_options: Dict[str, Any],
    profile: Any,
) -> None:
    """Use a book profile for translation consistency without changing task type."""
    prompt_options['editorial_mode'] = 'book_profile'
    prompt_options['profile_id'] = profile.profile_id
    prompt_options['target_locale'] = profile.target_locale or prompt_options.get('target_locale') or ''
    prompt_options['preserve_author_voice'] = profile.preserve_author_voice
    prompt_options['use_profile_glossary'] = True
    prompt_options['allow_common_glossary'] = profile.allow_common_glossary
    prompt_options['allow_cross_profile_glossary'] = profile.allow_cross_profile_glossary
    prompt_options['glossary_suggestions_enabled'] = True
    prompt_options['auto_approve_glossary_suggestions'] = False
    prompt_options['min_glossary_suggestion_confidence'] = 0.92
    prompt_options['avoid_hardcoded_editorial_rules'] = True
    prompt_options['audit_dimensions'] = 'translation_editorial_full'
    prompt_options.setdefault('profile_strength', 'balanced')
    prompt_options.setdefault('translation_profile_mode', True)
    # Keep the LLM profile judge opt-in for translations; the deterministic
    # profile precheck still runs and can trigger targeted repair for approved
    # glossary violations. Fidelity remains supervised by the source-aware
    # translation guard.
    prompt_options.setdefault('profile_audit_enabled', False)
    prompt_options.setdefault('repair_until_pass', True)
    prompt_options.setdefault('max_repair_rounds', min(profile.max_repair_rounds, 1))
    prompt_options.setdefault('min_dimension_score', profile.min_dimension_score)
    _apply_profile_strength_defaults(prompt_options)


def activate_inferred_book_profile(
    config: Dict[str, Any],
    prompt_options: Dict[str, Any],
) -> None:
    """Activate a profile declared by metadata when the request omitted it."""
    if str(prompt_options.get('profile_id') or '').strip():
        return

    transform_mode = str(prompt_options.get('text_transform_mode') or '').strip().lower()
    metadata = {
        'output_filename': config.get('output_filename'),
        'file_path': config.get('file_path'),
        'original_filename': config.get('original_filename'),
        'input_filename': config.get('input_filename'),
    }
    profile_id = infer_profile_id_from_metadata(metadata)
    if not profile_id:
        return
    profile = load_book_profile(profile_id, allow_missing=True)
    if profile is None:
        return
    intralingual_match = profile_matches_intralingual_request(
        config,
        prompt_options,
        profile_target_locale=profile.target_locale,
    )
    if transform_mode != 'modernize' and not intralingual_match:
        if profile_matches_translation_request(
            config,
            prompt_options,
            profile_target_locale=profile.target_locale,
        ):
            activate_profile_for_translation(prompt_options, profile)
        return

    prompt_options['text_transform_mode'] = 'modernize'
    prompt_options.setdefault('text_transform_label', 'Modernizar')
    prompt_options.setdefault('text_transform_profile', 'faithful_current_spanish')
    prompt_options['editorial_mode'] = 'book_profile'
    prompt_options['profile_id'] = profile.profile_id
    prompt_options['target_locale'] = profile.target_locale or prompt_options.get('target_locale') or 'es-MX'
    prompt_options['modernization_strength'] = profile.modernization_strength or 'high'
    prompt_options['preserve_author_voice'] = profile.preserve_author_voice
    prompt_options['preserve_archaisms'] = False
    prompt_options['preserve_iconic_formulas'] = True
    prompt_options['use_profile_glossary'] = True
    prompt_options['allow_common_glossary'] = profile.allow_common_glossary
    prompt_options['allow_cross_profile_glossary'] = profile.allow_cross_profile_glossary
    prompt_options['glossary_suggestions_enabled'] = True
    prompt_options['auto_approve_glossary_suggestions'] = False
    prompt_options['min_glossary_suggestion_confidence'] = 0.92
    prompt_options['avoid_hardcoded_editorial_rules'] = True
    prompt_options['audit_dimensions'] = 'editorial_full'
    prompt_options['profile_audit_enabled'] = True
    prompt_options['repair_until_pass'] = True
    prompt_options['abort_on_profile_fail'] = bool(
        profile.raw_config.get('abort_on_profile_fail', True)
    )
    prompt_options['min_dimension_score'] = profile.min_dimension_score
    prompt_options['max_repair_rounds'] = profile.max_repair_rounds


def uses_text_first_pipeline(config: Dict[str, Any]) -> bool:
    """Whether rich input should be translated through the text pipeline."""
    file_type = (config.get('file_type') or '').lower()
    if file_type not in {'epub', 'docx'}:
        return False
    return should_use_sanitized_text_pipeline(
        file_type,
        config.get('prompt_options') or {},
    )


def layout_sanitizer_applies(config: Dict[str, Any]) -> bool:
    file_type = (config.get('file_type') or '').lower()
    if file_type == 'pdf':
        return True
    return should_use_sanitized_text_pipeline(
        file_type,
        config.get('prompt_options') or {},
    )


def ensure_layout_sanitizer_options(config: Dict[str, Any]) -> None:
    if not layout_sanitizer_applies(config):
        return
    prompt_options = config.setdefault('prompt_options', {})
    prompt_options['layout_sanitizer_active'] = True
    prompt_options['layout_sanitizer_version'] = LAYOUT_SANITIZER_VERSION


def resume_requires_layout_sanitizer_restart(config: Dict[str, Any]) -> bool:
    if not layout_sanitizer_applies(config):
        return False
    prompt_options = config.get('prompt_options') or {}
    return prompt_options.get('layout_sanitizer_version') != LAYOUT_SANITIZER_VERSION


def source_refs_from_checkpoint(checkpoint_manager, translation_id: str) -> list[str]:
    """Return source chunk refs from a completed translation checkpoint."""
    try:
        checkpoint = checkpoint_manager.load_checkpoint(translation_id) or {}
    except Exception:
        return []

    refs_by_index: dict[int, str] = {}
    for chunk in checkpoint.get('chunks') or []:
        try:
            index = int(chunk.get('chunk_index'))
        except (TypeError, ValueError):
            continue
        source = chunk.get('original_text') or ''
        if isinstance(source, str) and source.strip():
            refs_by_index[index] = source

    if not refs_by_index:
        return []
    return [refs_by_index[i] for i in sorted(refs_by_index)]


def with_source_guard_refs(
    base_options: Dict[str, Any],
    checkpoint_manager,
    translation_id: str,
    log_callback,
) -> Dict[str, Any]:
    """Attach source references for the chained refine-after phase."""
    options = dict(base_options or {})
    refs = source_refs_from_checkpoint(checkpoint_manager, translation_id)
    if refs:
        options['_source_guard_reference_chunks'] = refs
        if log_callback:
            log_callback(
                "source_aware_guard_refs_loaded",
                f"🔎 Loaded {len(refs)} source chunks for source-aware editorial guard."
            )
    return options
