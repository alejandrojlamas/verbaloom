"""Canonical names for the translation WebSocket contract (backend side).

These string literals are part of the wire contract with the frontend and used
to be hard-coded at each emit/handler site. Centralizing them here gives one
authoritative definition; the JavaScript client mirrors the same values (see
src/web/static/js/core/websocket-manager.js and translation-tracker.js). Change
both sides together.
"""

# The single channel carrying translation progress/status/log payloads.
EVENT_TRANSLATION_UPDATE = "translation_update"

# log_entry.type values whose payload carries displayable LLM output (used to
# keep the "last translation" preview in sync across translate and refine).
LLM_RESPONSE_TYPES = ("llm_response", "refinement_response")
