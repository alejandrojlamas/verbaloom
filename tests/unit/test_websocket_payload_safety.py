import json

from src.api.socket_events import EVENT_TRANSLATION_UPDATE
from src.api.websocket import emit_update


class FakeSocketIO:
    def __init__(self):
        self.emitted = []

    def emit(self, event, payload, namespace=None):
        self.emitted.append((event, payload, namespace))


class FakeStateManager:
    def __init__(self):
        self.fields = {}

    def exists(self, translation_id):
        return translation_id == "job-1"

    def set_translation_field(self, translation_id, field, value):
        self.fields[(translation_id, field)] = value


def test_emit_update_sends_preview_without_raw_llm_payloads():
    socketio = FakeSocketIO()
    state = FakeStateManager()
    raw_response = "<TRANSLATION>Texto traducido visible</TRANSLATION>"
    raw_prompt = "SOURCE TEXT THAT SHOULD NOT TRAVEL TO CLIENT"

    emit_update(
        socketio,
        "job-1",
        {
            "log": "LLM Response received",
            "log_entry": {
                "type": "llm_response",
                "message": "LLM Response",
                "data": {
                    "response": raw_response,
                    "system_prompt": raw_prompt,
                    "total_tokens": 123,
                },
            },
        },
        state,
    )

    assert socketio.emitted
    event, payload, namespace = socketio.emitted[0]
    payload_json = json.dumps(payload, ensure_ascii=False)

    assert event == EVENT_TRANSLATION_UPDATE
    assert namespace == "/"
    assert payload["last_translation"] == "Texto traducido visible"
    assert state.fields[("job-1", "last_translation")] == "Texto traducido visible"
    assert payload["log_entry"]["data"]["response"] == f"[omitted {len(raw_response)} chars]"
    assert payload["log_entry"]["data"]["system_prompt"] == f"[omitted {len(raw_prompt)} chars]"
    assert payload["log_entry"]["data"]["total_tokens"] == 123
    assert raw_response not in payload_json
    assert raw_prompt not in payload_json
