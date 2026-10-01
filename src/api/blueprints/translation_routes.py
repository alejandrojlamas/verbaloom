"""
Translation job management routes
"""
import os
import time
import copy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from flask import Blueprint, request, jsonify

from src.api.safe_payloads import client_safe_config, client_safe_logs
from src.api.resume_schedule import clear_resume_schedule, resume_schedule_from_config
from src.api.translation_state import translation_lifecycle_guard
from src.core.job_runtime_config import configure_editorial_guard_options
from src.core.deepseek_pricing import (
    get_deepseek_pricing_status,
    is_official_deepseek_endpoint,
)
from src.core.progress import apply_active_timing
from src.persistence.checkpoint_reconcile import checkpoint_progress_snapshot
from src.config import (
    REQUEST_TIMEOUT,
    MAX_TRANSLATION_ATTEMPTS,
    OLLAMA_NUM_CTX,
    AUTO_PAUSE_ON_RATE_LIMIT,
    OLLAMA_API_ENDPOINT,
    OPENAI_API_ENDPOINT,
    MISTRAL_API_ENDPOINT,
    DEEPSEEK_API_ENDPOINT,
    POE_API_ENDPOINT,
    NIM_API_ENDPOINT,
    OUTPUT_DIR,
)
from src.api.services.path_validator import PathValidator
from src.utils.provider_security import (
    EndpointCredentialError,
    endpoint_allows_saved_key,
    endpoint_origin,
    resolve_api_key_for_endpoint,
    sanitize_restored_endpoint_credentials,
)
from src.api.socket_events import EVENT_TRANSLATION_UPDATE
from src.tts.tts_config import TTSConfig


def _resolve_api_key(value, env_var_name):
    """
    Resolve API key value from request or environment.

    Args:
        value: Value from request (can be actual key, '__USE_ENV__', or empty)
        env_var_name: Name of environment variable to fall back to

    Returns:
        Resolved API key string
    """
    if value == '__USE_ENV__' or not value:
        # Use environment variable
        return os.getenv(env_var_name, '')
    return value


# Cloud providers whose key lives in config['<provider>_api_key'] and env var
# '<PROVIDER>_API_KEY'. The mapping is mechanical, so supporting a new provider
# in the resume-override path requires only adding it here (and nowhere else in
# this file).
_KEY_PROVIDERS = ('gemini', 'openai', 'openrouter', 'mistral', 'deepseek', 'poe', 'nim')

# Providers that talk to a user-supplied endpoint; the others use a built-in one.
_ENDPOINT_PROVIDERS = ('ollama', 'openai')
_KEYED_CUSTOM_ENDPOINT_PROVIDERS = ('openai', 'nim')

_PROVIDER_DEFAULT_ENDPOINTS = {
    'ollama': OLLAMA_API_ENDPOINT,
    'openai': OPENAI_API_ENDPOINT,
    'mistral': MISTRAL_API_ENDPOINT,
    'deepseek': DEEPSEEK_API_ENDPOINT,
    'poe': POE_API_ENDPOINT,
    'nim': NIM_API_ENDPOINT,
}

def _client_safe_config(config):
    return client_safe_config(config)


def _client_safe_logs(logs, limit=100):
    return client_safe_logs(logs, limit=limit)


def _json_safe(value):
    """Convert nested runtime objects to JSON-safe primitives for API payloads."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, 'to_dict') and callable(value.to_dict):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _checkpoint_has_resumable_work(checkpoint_data):
    """Allow finalization to resume after all text chunks are already complete."""
    if not checkpoint_data:
        return False
    job = checkpoint_data.get('job') or {}
    status = str(job.get('status') or '').strip().lower()
    if status == 'completed':
        return False
    # A complete text checkpoint can still need assembly, publication gates,
    # metadata localization, or whole-book QA. Those phases are resumable work.
    return True


def _blocks_resume(translation_id, candidate_id, candidate_data):
    """Return whether an in-memory job conflicts with a resume request.

    A scheduled provider wait for the same job is replaceable: the persisted
    price gate is checked before the replacement is published, and lifecycle
    ownership makes the old waiter exit without demoting the new intent.
    """
    status = str((candidate_data or {}).get('status') or '').strip().lower()
    if status not in {'running', 'queued', 'pricing_wait', 'provider_wait'}:
        return False
    if str(candidate_id) != str(translation_id):
        return True
    if status == 'pricing_wait':
        # The live price policy is checked again before any state is changed.
        return False
    if status != 'provider_wait':
        return True
    schedule = resume_schedule_from_config((candidate_data or {}).get('config'))
    return bool(schedule and schedule.resume_at_epoch > time.time())


def _persist_manual_pause_request(state_manager, translation_id):
    """Persist user intent immediately so a server restart cannot revive it."""
    with translation_lifecycle_guard(state_manager):
        job_data = state_manager.get_translation(translation_id) or {}
        config = copy.deepcopy(job_data.get('config') or {})
        config['_manual_pause_requested'] = True
        config['_manual_pause_requested_at'] = time.time()
        state_manager.set_translation_field(translation_id, 'config', config)
        state_manager.set_translation_field(
            translation_id, 'pause_reason', 'manual'
        )
        checkpoint_manager = state_manager.checkpoint_manager
        checkpoint_manager.update_job_config(translation_id, config)
        checkpoint_manager.mark_paused(translation_id)


def _provider_default_endpoint(provider):
    return _PROVIDER_DEFAULT_ENDPOINTS.get((provider or '').lower(), '')


def _provider_key_required(provider, endpoint):
    """Local/custom OpenAI-compatible servers may intentionally be keyless."""
    if (provider or '').lower() != 'openai':
        return provider in _KEY_PROVIDERS
    return endpoint_origin(endpoint) == "https://api.openai.com"


def _deepseek_pricing_block(provider, endpoint):
    """Return a structured start/resume block for the official paid API."""
    if (provider or '').lower() != 'deepseek':
        return None
    if not is_official_deepseek_endpoint(endpoint):
        return None
    pricing = get_deepseek_pricing_status()
    if not pricing.disabled:
        return None
    return {
        "code": "deepseek_peak_pricing",
        "error": "DeepSeek está deshabilitado durante el horario de tarifa alta.",
        "message": (
            "No se enviaron tokens. Podrás usar DeepSeek de nuevo al terminar "
            "la ventana de tarifa alta."
        ),
        "availability": pricing.to_dict(),
    }


def _sanitize_restored_endpoint_credentials(config):
    """Strip ambiguous legacy keys before a custom endpoint is resumed."""
    sanitize_restored_endpoint_credentials(
        config,
        _PROVIDER_DEFAULT_ENDPOINTS,
        keyed_custom_providers=_KEYED_CUSTOM_ENDPOINT_PROVIDERS,
    )


_AUTO_SOURCE_LANGUAGE_VALUES = {
    '',
    'auto',
    'autodetect',
    'auto-detect',
    'autodetectar',
    'detect automatically',
    'detectar automaticamente',
    'detectar automáticamente',
    'detectar automáticamente desde el archivo...',
}


def _should_autodetect_source_language(value):
    normalized = str(value or '').strip().lower()
    return normalized in _AUTO_SOURCE_LANGUAGE_VALUES


def _detect_source_language_from_file(file_path):
    try:
        from src.utils.language_detector import LanguageDetector

        path = Path(file_path)
        with path.open('rb') as fh:
            file_data = fh.read()
        return LanguageDetector.detect_language_from_file(file_data, path.name)
    except Exception:
        return None, 0.0


def _normalize_uploaded_file_path_for_type(file_path, file_type):
    """Preserve known processor suffixes for already-uploaded extensionless files."""
    desired_suffix = {
        'epub': '.epub',
        'docx': '.docx',
        'pdf': '.pdf',
        'srt': '.srt',
    }.get(str(file_type or '').lower())
    if not file_path or not desired_suffix:
        return file_path

    path = Path(file_path)
    if not path.exists() or path.suffix.lower() == desired_suffix:
        return file_path

    try:
        from src.utils.file_detector import detect_file_type_by_content

        detected_type = detect_file_type_by_content(str(path))
    except Exception:
        detected_type = None

    if detected_type != str(file_type or '').lower():
        return file_path

    renamed_path = path.with_suffix(desired_suffix)
    if renamed_path.exists():
        return str(renamed_path)

    path.rename(renamed_path)
    return str(renamed_path)


def _probe_readable_content(file_path, file_type):
    """Return whether an uploaded file has extractable text before queueing."""
    path = Path(file_path)
    label = str(file_type or "file").upper()
    if not path.exists() or not path.is_file():
        return False, 0, "Uploaded file not found on server"
    if path.stat().st_size == 0:
        return False, 0, "Uploaded file is empty"

    try:
        from src.core.output_formats import extract_readable_text

        readable_text = extract_readable_text(path)
    except Exception as exc:
        return (
            False,
            0,
            f"Could not extract readable text from this {label} file: {exc}",
        )

    char_count = len((readable_text or "").strip())
    if char_count <= 0:
        return (
            False,
            0,
            f"This {label} file has no readable text. If it is a scanned PDF or image-only document, run OCR first and upload the OCR text/PDF.",
        )
    return True, char_count, ""


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _apply_resume_overrides(config, overrides):
    """Merge optional model/provider override fields into a resume config in place.

    Lets the resume request switch model/provider for the remaining chunks
    (issue #183). An empty/absent body leaves `config` untouched, so existing
    behavior is preserved. API keys flow through `_resolve_api_key` exactly like
    the start endpoint, and a multi-key string is passed through unchanged so the
    key-rotation pool still works.

    Returns a Flask (response, status) tuple to abort with on validation failure,
    or None on success.
    """
    if not isinstance(overrides, dict) or not overrides:
        return None

    if overrides.get('model'):
        config['model'] = overrides['model']
    if overrides.get('llm_provider'):
        config['llm_provider'] = str(overrides['llm_provider']).lower()
        if not overrides.get('llm_api_endpoint'):
            config['llm_api_endpoint'] = _provider_default_endpoint(
                config['llm_provider']
            )
    if overrides.get('llm_api_endpoint'):
        config['llm_api_endpoint'] = overrides['llm_api_endpoint']
    if overrides.get('context_window') is not None:
        try:
            config['context_window'] = int(overrides['context_window'])
        except (TypeError, ValueError):
            return jsonify({"error": "context_window must be an integer"}), 400
    if isinstance(overrides.get('prompt_options'), dict):
        prompt_options = config.setdefault('prompt_options', {})
        prompt_options.update(overrides['prompt_options'])

    provider = (config.get('llm_provider') or 'ollama').lower()

    # A single generic api_key override maps to the chosen provider's key field,
    # resolved through .env like every other entry point.
    raw_key = overrides.get('api_key')
    endpoint = config.get('llm_api_endpoint') or _provider_default_endpoint(provider)
    provider_or_endpoint_changed = bool(
        overrides.get('llm_provider') or overrides.get('llm_api_endpoint')
    )
    if provider in _KEY_PROVIDERS:
        env_var = f"{provider.upper()}_API_KEY"
        sources = dict(config.get('_credential_sources') or {})
        should_resolve = raw_key not in (None, '') or provider_or_endpoint_changed
        if should_resolve:
            if provider in _KEYED_CUSTOM_ENDPOINT_PROVIDERS:
                try:
                    key, source = resolve_api_key_for_endpoint(
                        raw_key,
                        env_var,
                        endpoint=endpoint,
                        default_endpoint=_provider_default_endpoint(provider),
                    )
                except EndpointCredentialError as exc:
                    return jsonify({"error": "Unsafe API key routing", "message": str(exc)}), 400
            else:
                key = _resolve_api_key(raw_key, env_var)
                source = 'explicit' if raw_key not in (None, '', '__USE_ENV__') else ('environment' if key else 'none')
            config[f"{provider}_api_key"] = key
            sources[provider] = source
            config['_credential_sources'] = sources
        elif provider in _KEYED_CUSTOM_ENDPOINT_PROVIDERS and (
            not endpoint_allows_saved_key(endpoint, _provider_default_endpoint(provider))
            and sources.get(provider) != 'explicit'
        ):
            # Legacy checkpoints did not record whether a key came from .env.
            # Fail closed for a custom origin; a keyless local server still
            # works, while an authenticated gateway must receive a new key.
            config[f"{provider}_api_key"] = ''
            sources[provider] = 'none'
            config['_credential_sources'] = sources

    # A cloud provider needs a key from the override, the restored config, or .env.
    if provider in _KEY_PROVIDERS:
        env_var = f"{provider.upper()}_API_KEY"
        if (
            _provider_key_required(provider, endpoint)
            and not config.get(f"{provider}_api_key")
        ):
            return jsonify({
                "error": "Missing API key for provider",
                "message": (f"Resuming with '{provider}' requires an API key. "
                            f"Set {env_var} in .env or include it in the request."),
            }), 400

    # Endpoint-driven providers need an endpoint to talk to.
    if provider in _ENDPOINT_PROVIDERS and not config.get('llm_api_endpoint'):
        return jsonify({
            "error": "Missing API endpoint for provider",
            "message": f"Resuming with '{provider}' requires an API endpoint.",
        }), 400

    return None


def _reconcile_checkpoint_progress(state_manager, translation_id):
    """Return a stats patch derived from checkpoint rows, not worker memory."""
    job = state_manager.checkpoint_manager.db.get_job(translation_id)
    if not job:
        return None
    snapshot = checkpoint_progress_snapshot(
        state_manager.checkpoint_manager,
        translation_id,
    ) or {}
    progress = dict(job.get('progress') or {})
    total = _safe_int(
        snapshot.get('total_chunks'),
        _safe_int(progress.get('total_chunks'), 0),
    )
    current = _safe_int(progress.get('current_chunk_index'), -1)
    completed = _safe_int(snapshot.get('completed_chunks'), 0)
    failed = _safe_int(snapshot.get('failed_chunks'), 0)

    state_stats = {}
    if state_manager.exists(translation_id):
        state_stats = state_manager.get_translation_field(translation_id, 'stats') or {}
    start_time = state_stats.get('start_time') or progress.get('start_time') or time.time()
    elapsed = time.time() - start_time

    completed_for_progress = max(
        completed,
        _safe_int(state_stats.get('completed_chunks'), 0),
        _safe_int(progress.get('completed_chunks'), 0),
        current + 1,
    )
    if total:
        completed_for_progress = min(completed_for_progress, total)

    stats_patch = {
        'start_time': start_time,
        'elapsed_time': elapsed,
        'total_chunks': total,
        'completed_chunks': completed_for_progress,
        'failed_chunks': failed,
        'checkpoint_completed_chunks': completed,
        'checkpoint_failed_chunks': failed,
        'manual_repair': True,
        'manual_repair_updated_at': time.time(),
    }
    state_manager.checkpoint_manager.db.update_job_progress(
        translation_id,
        current_chunk_index=max(current, completed_for_progress - 1),
        total_chunks=total,
        completed_chunks=completed_for_progress,
        failed_chunks=failed,
    )
    if state_manager.exists(translation_id):
        state_manager.update_stats(translation_id, stats_patch)
    return stats_patch


def create_translation_blueprint(
    state_manager,
    start_translation_job,
    socketio=None,
    output_dir=None,
    cancel_translation_handoff=None,
):
    """
    Create and configure the translation blueprint

    Args:
        state_manager: Translation state manager instance
        start_translation_job: Function to start translation jobs
    """
    bp = Blueprint('translation', __name__)
    managed_uploads = Path(output_dir or OUTPUT_DIR) / 'uploads'

    @bp.route('/api/translate', methods=['POST'])
    def start_translation_request():
        """Start a new translation job"""
        data = request.get_json(silent=True) or {}
        provider = str(data.get('llm_provider') or 'ollama').lower()
        llm_api_endpoint = str(data.get('llm_api_endpoint') or '').strip()
        if not llm_api_endpoint:
            llm_api_endpoint = _provider_default_endpoint(provider)

        pricing_block = _deepseek_pricing_block(provider, llm_api_endpoint)
        if pricing_block:
            return jsonify(pricing_block), 423

        # Validate required fields
        has_file_input = 'file_path' in data
        if has_file_input:
            required_fields = ['file_path', 'target_language',
                             'model', 'output_filename', 'file_type']
        else:
            required_fields = ['text', 'source_language', 'target_language',
                             'model', 'output_filename']

        for field in required_fields:
            if field not in data or (isinstance(data[field], str) and not data[field].strip()) or (not isinstance(data[field], str) and data[field] is None):
                if field == 'text' and data.get('file_type') == 'txt' and data.get('text') == "":
                    pass
                else:
                    return jsonify({"error": f"Missing or empty field: {field}"}), 400

        output_ok, output_error = PathValidator.validate_filename(
            str(data.get('output_filename') or '')
        )
        if not output_ok:
            return jsonify({"error": f"Invalid output_filename: {output_error}"}), 400

        if provider in _ENDPOINT_PROVIDERS and not llm_api_endpoint:
            return jsonify({
                "error": "Missing or empty field: llm_api_endpoint",
                "message": f"Provider '{provider}' requires an API endpoint."
            }), 400

        resolved_keys = {f"{key_provider}_api_key": "" for key_provider in _KEY_PROVIDERS}
        credential_sources = {}
        if provider in _KEY_PROVIDERS:
            env_var = f"{provider.upper()}_API_KEY"
            if provider in _KEYED_CUSTOM_ENDPOINT_PROVIDERS:
                try:
                    resolved_key, key_source = resolve_api_key_for_endpoint(
                        data.get(f"{provider}_api_key"),
                        env_var,
                        endpoint=llm_api_endpoint,
                        default_endpoint=_provider_default_endpoint(provider),
                    )
                except EndpointCredentialError as exc:
                    return jsonify({"error": "Unsafe API key routing", "message": str(exc)}), 400
            else:
                raw_key = data.get(f"{provider}_api_key")
                resolved_key = _resolve_api_key(raw_key, env_var)
                key_source = 'explicit' if raw_key not in (None, '', '__USE_ENV__') else ('environment' if resolved_key else 'none')
            resolved_keys[f"{provider}_api_key"] = resolved_key
            credential_sources[provider] = key_source

        if (
            _provider_key_required(provider, llm_api_endpoint)
            and not resolved_keys.get(f"{provider}_api_key")
        ):
            return jsonify({
                "error": "Missing API key for provider",
                "message": f"Provider '{provider}' requires {provider.upper()}_API_KEY."
            }), 400

        normalized_file_path = None
        if has_file_input:
            try:
                managed_input = PathValidator.resolve_managed_file(
                    data['file_path'],
                    [managed_uploads],
                )
            except ValueError:
                return jsonify({"error": "Input file is outside managed uploads"}), 403
            except FileNotFoundError:
                return jsonify({"error": "Uploaded file not found on server"}), 404
            normalized_file_path = _normalize_uploaded_file_path_for_type(
                str(managed_input),
                data.get('file_type'),
            )
            if not Path(normalized_file_path).exists():
                return jsonify({
                    "error": "Uploaded file not found on server",
                    "message": "The selected file is no longer available on the server. Remove it and select the file again."
                }), 400
            readable_ok, readable_chars, readable_error = _probe_readable_content(
                normalized_file_path,
                data.get('file_type'),
            )
            if not readable_ok:
                return jsonify({
                    "error": "Uploaded file has no readable text",
                    "message": readable_error,
                    "details": {
                        "file_type": data.get('file_type'),
                        "readable_characters": readable_chars,
                    },
                }), 400
            prompt_options = dict(data.get('prompt_options') or {})
            prompt_options['_input_readable_characters'] = readable_chars
            data['prompt_options'] = prompt_options

        source_language = str(data.get('source_language') or '').strip()
        if has_file_input and _should_autodetect_source_language(source_language):
            detected_language, confidence = _detect_source_language_from_file(normalized_file_path)
            prompt_options = dict(data.get('prompt_options') or {})
            if detected_language:
                source_language = detected_language
                prompt_options['_source_language_autodetected'] = True
                prompt_options['_source_language_confidence'] = round(float(confidence or 0.0), 4)
            else:
                # Some valid PDFs begin with front matter, Greek/Latin editions,
                # catalog pages, or OCR noise that makes language detection
                # unreliable. Do not fail job initialization for that; let the
                # translation prompt ask the model to infer the source language
                # from each chunk, while exposing the uncertainty for logs/reports.
                source_language = 'Auto'
                prompt_options['_source_language_autodetected'] = False
                prompt_options['_source_language_confidence'] = 0.0
                prompt_options['_source_language_autodetect_failed'] = True
            data['prompt_options'] = prompt_options

        input_filename = str(
            data.get('input_filename')
            or data.get('original_filename')
            or ''
        ).strip()
        original_filename = str(
            data.get('original_filename')
            or data.get('input_filename')
            or ''
        ).strip()
        if has_file_input and not input_filename:
            input_filename = Path(normalized_file_path or data.get('file_path') or '').name
        if has_file_input and not original_filename:
            original_filename = input_filename

        # Generate unique translation ID
        translation_id = f"trans_{int(time.time() * 1000)}"

        # Build configuration
        config = {
            'source_language': source_language,
            'target_language': data['target_language'],
            'model': data['model'],
            'llm_api_endpoint': llm_api_endpoint,
            'request_timeout': int(data.get('timeout', REQUEST_TIMEOUT)),
            'context_window': int(data.get('context_window', OLLAMA_NUM_CTX)),
            'max_attempts': int(data.get('max_attempts', MAX_TRANSLATION_ATTEMPTS)),
            'retry_delay': int(data.get('retry_delay', 2)),
            'output_filename': data['output_filename'],
            'input_filename': input_filename,
            'original_filename': original_filename,
            'llm_provider': provider,
            'operation': data.get('operation') or (
                'transform'
                if (data.get('prompt_options') or {}).get('text_transform_mode')
                else ('refine' if data.get('refine_only', False) else 'translate')
            ),
            'text_transform_mode': data.get('text_transform_mode') or (data.get('prompt_options') or {}).get('text_transform_mode', ''),
            'text_transform_label': data.get('text_transform_label') or (data.get('prompt_options') or {}).get('text_transform_label', ''),
            **resolved_keys,
            '_credential_sources': credential_sources,
            # Prompt options (optional instructions to include in the system prompt)
            'prompt_options': data.get('prompt_options', {}),
            # Auto-pause on rate limit toggle (request overrides .env default)
            'auto_pause_on_rate_limit': data.get('auto_pause_on_rate_limit', AUTO_PAUSE_ON_RATE_LIMIT),
            # Bilingual output (original + translation interleaved)
            'bilingual_output': data.get('bilingual_output', False),
            # Refine-only mode (skip translation, run only refinement on input)
            'refine_only': data.get('refine_only', False),
            # Chained refinement pass after translation
            'refine_after': data.get('refine_after', False),
            # Final delivery format: auto, txt, docx, pdf, or epub
            'output_format': data.get('output_format', 'auto'),
            # TTS configuration
            'tts_enabled': data.get('tts_enabled', False),
            'tts_config': TTSConfig.from_web_request(data).to_dict() if data.get('tts_enabled') else None
        }

        if config['operation'] == 'translate':
            # Translation is one integrated publishable workflow: draft,
            # editorial review, source-aware audit, then assembly.  API clients
            # may tune models and prompts, but may not silently publish an
            # unreviewed or unaudited translation.
            config['refine_after'] = True
            stage_options = config['prompt_options']
            stage_options.setdefault('refine', True)
            stage_options.setdefault('strict_stage_contract', True)
            stage_options.setdefault('review_entire_book', True)
            stage_options.setdefault('audit_entire_book', True)
            stage_options.setdefault('permit_silent_source_fallback', False)
            stage_options.setdefault('permit_unreviewed_segments', False)
            stage_options.setdefault('permit_unaudited_segments', False)

        # Add file-specific or text-specific configuration
        if 'file_path' in data:
            config['file_path'] = normalized_file_path or data['file_path']
            config['file_type'] = data['file_type']
        else:
            config['text'] = data['text']
            config['file_type'] = data.get('file_type', 'txt')

        configure_editorial_guard_options(config)

        # Create translation in state manager
        state_manager.create_translation(translation_id, config)

        # Start translation job
        start_translation_job(translation_id, config)

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation queued.",
            "config_received": _client_safe_config(config)
        })

    @bp.route('/api/translation/<translation_id>', methods=['GET'])
    def get_translation_job_status(translation_id):
        """Get status of a translation job"""
        job_data = state_manager.get_translation(translation_id)
        if not job_data:
            # Include the status code in the message for already-loaded mobile
            # clients that predate structured API error metadata.
            return jsonify({"error": "404 Translation not found"}), 404

        stats = job_data.get('stats', {
            'start_time': time.time(),
            'total_chunks': 0,
            'completed_chunks': 0,
            'failed_chunks': 0
        })

        status = job_data.get('status')
        stats_payload = _json_safe(apply_active_timing(
            dict(stats),
            status=status,
            advance=status == 'running',
        ))

        return jsonify({
            "translation_id": translation_id,
            "status": job_data.get('status'),
            "pause_reason": job_data.get('pause_reason'),
            "resume_at_utc": job_data.get('resume_at_utc'),
            "resume_at_local": job_data.get('resume_at_local'),
            "progress": _json_safe(job_data.get('progress')),
            "stats": stats_payload,
            "logs": _json_safe(_client_safe_logs(job_data.get('logs', []))),
            "result_preview": "[Preview functionality removed. Download file to view content.]" if job_data.get('status') in ['completed', 'interrupted', 'partial'] else None,
            "error": _json_safe(job_data.get('error')),
            "config": _json_safe(_client_safe_config(job_data.get('config'))),
            "output_filepath": job_data.get('output_filepath')
        })

    @bp.route('/api/translation/<translation_id>/repair-progress', methods=['POST'])
    def repair_translation_progress(translation_id):
        """Reconcile the visible failed count after manual checkpoint repair."""
        stats_patch = _reconcile_checkpoint_progress(state_manager, translation_id)
        if stats_patch is None:
            return jsonify({"error": "Translation not found"}), 404

        payload = request.get_json(silent=True) or {}
        repaired_chunks = payload.get('repaired_chunks') or []
        message = payload.get('message') or "Manual repair checkpoint reconciliation"

        if state_manager.exists(translation_id):
            state_manager.append_log(
                translation_id,
                f"[{time.strftime('%H:%M:%S')}] {message}: "
                f"{stats_patch['checkpoint_failed_chunks']} failed checkpoint(s) remain."
            )

        if socketio:
            socketio.emit(EVENT_TRANSLATION_UPDATE, {
                'translation_id': translation_id,
                'stats': stats_patch,
                'manual_repair': True,
                'repaired_chunks': repaired_chunks,
                'log': (
                    f"🛠️ Reparación aplicada: "
                    f"{stats_patch['checkpoint_failed_chunks']} fragmento(s) fallido(s) persistido(s)."
                ),
            }, namespace='/')

        return jsonify({
            "translation_id": translation_id,
            "stats": stats_patch,
            "repaired_chunks": repaired_chunks,
        })

    @bp.route('/api/translation/<translation_id>/interrupt', methods=['POST'])
    def interrupt_translation_job(translation_id):
        """Interrupt a running translation job"""
        with translation_lifecycle_guard(state_manager):
            if not state_manager.exists(translation_id):
                return jsonify({"error": "Translation not found"}), 404

            job_data = state_manager.get_translation(translation_id)
            status = job_data.get('status')
            recovery_was_scheduled = bool(
                state_manager.get_translation_field(
                    translation_id, 'recovery_scheduled'
                )
            )
            if status in {
                'running', 'queued', 'pricing_wait', 'provider_wait', 'rate_limited'
            }:
                if callable(cancel_translation_handoff):
                    cancel_translation_handoff(translation_id)
                # Invalidate the owning generation before a stale daemon can
                # clear or consume a replacement recovery.
                state_manager.set_translation_field(
                    translation_id, 'recovery_scheduled', False
                )
                state_manager.set_translation_field(
                    translation_id, '_recovery_token', None
                )
            if status in {'pricing_wait', 'provider_wait'}:
                _persist_manual_pause_request(state_manager, translation_id)
                state_manager.set_interrupted(translation_id, True)
                state_manager.set_translation_field(
                    translation_id, 'status', 'interrupted'
                )
                state_manager.set_translation_field(
                    translation_id, 'resume_at_utc', None
                )
                state_manager.set_translation_field(
                    translation_id, 'resume_at_local', None
                )
                state_manager.checkpoint_manager.mark_interrupted(translation_id)
                if socketio:
                    socketio.emit(EVENT_TRANSLATION_UPDATE, {
                        'translation_id': translation_id,
                        'status': 'interrupted',
                        'reason': 'manual',
                        'log': 'Espera programada cancelada; el checkpoint se conservó.',
                    }, namespace='/')
                return jsonify({
                    "message": "Scheduled provider wait cancelled. The checkpoint remains resumable."
                }), 200

            if status in ('running', 'queued'):
                _persist_manual_pause_request(state_manager, translation_id)
                state_manager.set_interrupted(translation_id, True)
                if recovery_was_scheduled:
                    # A deterministic backoff has no worker left to publish the
                    # terminal pause. Make it resumable before returning 200.
                    state_manager.set_translation_field(
                        translation_id, 'status', 'interrupted'
                    )
                    state_manager.checkpoint_manager.mark_interrupted(
                        translation_id
                    )
                return jsonify({
                    "message": "Interruption signal sent. Translation will stop after the current segment."
                }), 200

            if status == 'rate_limited':
                # Cancels any in-flight auto-resume sleep and stops the UI from
                # treating the job as still active.
                _persist_manual_pause_request(state_manager, translation_id)
                state_manager.set_interrupted(translation_id, True)
                state_manager.set_translation_field(
                    translation_id, 'status', 'interrupted'
                )
                state_manager.checkpoint_manager.mark_interrupted(translation_id)
                return jsonify({
                    "message": "Auto-resume cancelled. Translation marked interrupted; you can resume manually later."
                }), 200

        return jsonify({
            "message": "The translation is not in an interruptible state (e.g., already completed or failed)."
        }), 400

    @bp.route('/api/translations', methods=['GET'])
    def list_all_translations():
        """List all translation jobs"""
        summary_list = state_manager.get_translation_summaries()
        return jsonify({"translations": summary_list})

    @bp.route('/api/resumable', methods=['GET'])
    def list_resumable_jobs():
        """List all jobs that can be resumed.

        Each job carries its full `config`, which holds resolved API keys. Strip
        every '*_api_key' before sending it to the browser — the resume endpoint
        reads keys server-side from the checkpoint, so the client never needs them.
        """
        resumable_jobs = state_manager.get_resumable_jobs()
        for job in resumable_jobs:
            cfg = job.get('config')
            if isinstance(cfg, dict):
                job['config'] = _client_safe_config(cfg)
        return jsonify({"resumable_jobs": resumable_jobs})

    @bp.route('/api/resume/<translation_id>', methods=['POST'])
    def resume_translation_job_endpoint(translation_id):
        """Resume a paused or interrupted translation job"""
        # Check if there are any active translations
        all_translations = state_manager.get_all_translations()
        active_translations = []
        for tid, tdata in all_translations.items():
            status = tdata.get('status')
            if _blocks_resume(translation_id, tid, tdata):
                active_translations.append({
                    'id': tid,
                    'status': status,
                    'output_filename': tdata.get('config', {}).get('output_filename', 'unknown')
                })

        if active_translations:
            active_info = ', '.join([f"{t['output_filename']} ({t['status']})" for t in active_translations])
            return jsonify({
                "error": "Cannot resume: active translation in progress",
                "message": f"Please wait for active translation(s) to complete or interrupt them before resuming. Active: {active_info}",
                "active_translations": active_translations
            }), 409  # 409 Conflict status code

        # Check if checkpoint exists
        checkpoint_data = state_manager.checkpoint_manager.load_checkpoint(translation_id)
        if not checkpoint_data:
            return jsonify({"error": "No checkpoint found for this translation"}), 404
        if not _checkpoint_has_resumable_work(checkpoint_data):
            return jsonify({
                "error": "Translation already completed",
                "message": "This checkpoint already passed finalization; there is nothing left to resume."
            }), 409

        overrides = request.get_json(silent=True) or {}
        checkpoint_config = (checkpoint_data.get('job') or {}).get('config') or {}
        resume_provider = str(
            overrides.get('llm_provider')
            or checkpoint_config.get('llm_provider')
            or 'ollama'
        ).lower()
        resume_endpoint = str(
            overrides.get('llm_api_endpoint')
            or (
                _provider_default_endpoint(resume_provider)
                if overrides.get('llm_provider')
                else checkpoint_config.get('llm_api_endpoint')
            )
            or _provider_default_endpoint(resume_provider)
        ).strip()
        pricing_block = _deepseek_pricing_block(resume_provider, resume_endpoint)
        if pricing_block:
            return jsonify(pricing_block), 423

        # Get job config and add resume parameters
        job = checkpoint_data['job']
        config = copy.deepcopy(job['config'])  # Create a deep copy to avoid mutating the stored config
        _sanitize_restored_endpoint_credentials(config)

        # Get preserved input file path if exists
        # Always use preserved_input_path from config (stored during job creation)
        # This ensures consistent file path across multiple resume cycles
        preserved_path = config.get('preserved_input_path')
        resume_input_roots = [managed_uploads]
        checkpoint_uploads = getattr(
            state_manager.checkpoint_manager,
            'uploads_dir',
            None,
        )
        translation_id_ok, _ = PathValidator.validate_filename(translation_id)
        if checkpoint_uploads and translation_id_ok:
            resume_input_roots.append(Path(checkpoint_uploads) / translation_id)
        if preserved_path:
            try:
                managed_preserved = PathValidator.resolve_managed_file(
                    preserved_path,
                    resume_input_roots,
                )
            except (ValueError, FileNotFoundError):
                return jsonify({
                    "error": "Preserved input file not found",
                    "message": "The preserved input file for this job is unavailable.",
                    "suggestion": "This job cannot be resumed. Please delete this checkpoint and start a new translation."
                }), 404
            config['file_path'] = str(managed_preserved)
        else:
            # Fallback: try to get it from checkpoint manager
            preserved_path_fallback = state_manager.checkpoint_manager.get_preserved_input_path(translation_id)
            if preserved_path_fallback:
                try:
                    managed_preserved = PathValidator.resolve_managed_file(
                        preserved_path_fallback,
                        resume_input_roots,
                    )
                except (ValueError, FileNotFoundError):
                    return jsonify({"error": "Preserved input file is unavailable"}), 404
                config['file_path'] = str(managed_preserved)
            else:
                return jsonify({
                    "error": "No preserved input file",
                    "message": "This job has no preserved input file and cannot be resumed.",
                    "suggestion": "Please delete this checkpoint and start a new translation."
                }), 404

        # Add resume parameters to config
        config['resume_from_index'] = checkpoint_data['resume_from_index']
        config['is_resume'] = True
        config.pop('_manual_pause_requested', None)
        config.pop('_manual_pause_requested_at', None)
        # An explicit user action starts a fresh provider attempt. DeepSeek's
        # current peak-price gate above still prevents bypassing the cost policy.
        config = clear_resume_schedule(config)
        # The interrupt fence remains raised until this worker owns execution.
        # An older worker may still be unwinding after a quick Pause/Resume.
        config['_explicit_resume_requested'] = True

        # Optional model/provider overrides for the remaining chunks (issue #183).
        # No body = unchanged behavior.
        override_error = _apply_resume_overrides(config, overrides)
        if override_error is not None:
            return override_error

        with translation_lifecycle_guard(state_manager):
            # Recheck after filesystem and credential validation: an automatic
            # recovery may have become active while this request was preparing.
            active_now = [
                (tid, tdata)
                for tid, tdata in state_manager.get_all_translations().items()
                if _blocks_resume(translation_id, tid, tdata)
            ]
            if active_now:
                return jsonify({
                    "error": "Cannot resume: active translation in progress",
                    "message": "A translation became active while resume was being prepared. Refresh and try again.",
                }), 409

            restored = state_manager.restore_job_from_checkpoint(
                translation_id,
                pending_resume=True,
            )
            if not restored:
                return jsonify({
                    "error": "Failed to restore job from checkpoint"
                }), 500

            state_manager.set_translation_field(
                translation_id, 'recovery_scheduled', False
            )
            state_manager.set_translation_field(
                translation_id, '_recovery_token', None
            )
            state_manager.set_translation_field(
                translation_id, 'config', config
            )
            state_manager.set_translation_field(
                translation_id, 'pause_reason', None
            )
            state_manager.checkpoint_manager.update_job_config(
                translation_id, config
            )
            state_manager.checkpoint_manager.mark_running(translation_id)

            # Publish the replacement while the old timer is fenced by the
            # same lifecycle guard. The worker claims state after this exits.
            start_result = start_translation_job(
                translation_id,
                config,
                allow_handoff=True,
                replace_handoff=True,
            )

        return jsonify({
            "translation_id": translation_id,
            "message": "Translation resumed successfully",
            "resume_from_chunk": checkpoint_data['resume_from_index'],
            "model": config.get('model'),
            "llm_provider": config.get('llm_provider'),
            "worker_state": start_result,
        }), 200

    @bp.route('/api/checkpoint/<translation_id>', methods=['DELETE'])
    def delete_checkpoint_endpoint(translation_id):
        """Delete a checkpoint (manual cleanup by user)"""
        success = state_manager.delete_checkpoint(translation_id)

        if success:
            return jsonify({
                "message": "Checkpoint deleted successfully",
                "translation_id": translation_id
            }), 200
        else:
            return jsonify({"error": "Failed to delete checkpoint or checkpoint not found"}), 404

    return bp
