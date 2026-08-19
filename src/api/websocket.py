"""
WebSocket handlers for real-time communication
"""
from flask import request
from flask_socketio import emit

from src.config import TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT
from src.api.safe_payloads import client_safe_log_entry, redact_for_client, trim_client_preview
from src.core.llm.utils.extraction import TranslationExtractor
from src.utils.text_encoding import clean_text_artifacts

from .socket_events import EVENT_TRANSLATION_UPDATE, LLM_RESPONSE_TYPES

_preview_extractor = TranslationExtractor(TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT)


def _clean_last_translation_preview(response: str) -> str:
    extracted = _preview_extractor.extract(response)
    return clean_text_artifacts(extracted if extracted is not None else response)


def configure_websocket_handlers(socketio, state_manager):
    """Configure WebSocket event handlers"""
    
    @socketio.on('connect')
    def handle_websocket_connect():
        print(f'🔌 WebSocket client connected: {request.sid}')
        emit('connected', {'message': 'Connected to translation server via WebSocket'})

    @socketio.on('disconnect')
    def handle_websocket_disconnect():
        print(f'🔌 WebSocket client disconnected: {request.sid}')


def emit_update(socketio, translation_id, data_to_emit, state_manager):
    """
    Emit WebSocket update for translation progress.

    Stats are NOT auto-attached. Callers that need to send progress stats must
    include them explicitly via `data_to_emit['stats']`. Auto-attaching stats
    on every log/status emit used to create races: a log emit on the main loop
    would read the state snapshot at log time and emit it later, possibly
    overtaking a fresher stats emit on the wire and rolling the progress bar
    backward. Now only the dedicated stats callbacks touch the progress bar.

    Args:
        socketio: SocketIO instance
        translation_id (str): Translation job ID
        data_to_emit (dict): Data to send (must include 'stats' to push progress)
        state_manager: Translation state manager instance
    """
    if not state_manager.exists(translation_id):
        return

    data_to_emit['translation_id'] = translation_id
    try:
        # Store last translation for UI restoration after browser refresh.
        # Both the translate path (`llm_response`) and the refine path
        # (`refinement_response`) produce displayable LLM output — keep the
        # preview in sync for either.
        log_entry = data_to_emit.get('log_entry')
        if (log_entry and log_entry.get('type') in LLM_RESPONSE_TYPES and
            log_entry.get('data', {}).get('response')):
            preview = trim_client_preview(
                _clean_last_translation_preview(log_entry['data']['response'])
            )
            state_manager.set_translation_field(
                translation_id,
                'last_translation',
                preview
            )
            data_to_emit['last_translation'] = preview

        if log_entry:
            data_to_emit['log_entry'] = client_safe_log_entry(log_entry)

        socketio.emit(EVENT_TRANSLATION_UPDATE, redact_for_client(data_to_emit), namespace='/')
    except Exception as e:
        print(f"WebSocket emission error for {translation_id}: {e}")
