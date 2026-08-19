"""Unit tests for the webhook notifier."""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.utils import notifier


@pytest.fixture
def cfg():
    """Patch config attributes used by the notifier."""
    with patch.multiple(
        notifier.config,
        NOTIFY_WEBHOOK_URL="https://example.test/hook",
        NOTIFY_WEBHOOK_METHOD="POST",
        NOTIFY_WEBHOOK_HEADERS="",
        NOTIFY_WEBHOOK_PAYLOAD="",
        NOTIFY_ON_SUCCESS=True,
        NOTIFY_ON_FAILURE=True,
        NOTIFY_ON_INTERRUPTION=True,
        NOTIFY_TIMEOUT_SECONDS=2,
    ):
        yield


def _ok_response():
    response = MagicMock()
    response.raise_for_status = MagicMock()
    return response


def test_notify_returns_false_when_url_empty():
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_URL", ""):
        assert notifier.notify(notifier.EVENT_SUCCESS) is False


def test_notify_returns_false_when_event_disabled(cfg):
    with patch.object(notifier.config, "NOTIFY_ON_SUCCESS", False):
        with patch("src.utils.notifier.requests.request") as mock_req:
            assert notifier.notify(notifier.EVENT_SUCCESS) is False
            mock_req.assert_not_called()


def test_notify_default_payload_success(cfg):
    with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
        ok = notifier.notify(notifier.EVENT_SUCCESS, {
            "file": "book.epub",
            "duration_seconds": 12.5,
        })
    assert ok is True
    args, kwargs = mock_req.call_args
    assert args[0] == "POST"
    assert args[1] == "https://example.test/hook"
    assert kwargs["timeout"] == 2
    payload = kwargs["json"]
    assert payload["event"] == "success"
    assert payload["file"] == "book.epub"
    assert payload["title"] == "Translation completed"
    assert "book.epub" in payload["message"]


def test_notify_custom_payload_template(cfg):
    template = json.dumps({"msg": "Done: {file} ({duration_seconds:.1f}s)", "token": "x"})
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_PAYLOAD", template):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_SUCCESS, {"file": "a.txt", "duration_seconds": 3.0})
    payload = mock_req.call_args.kwargs["json"]
    assert payload == {"msg": "Done: a.txt (3.0s)", "token": "x"}


def test_notify_custom_headers(cfg):
    headers = json.dumps({"X-Token": "abc", "X-Source": "{provider}"})
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_HEADERS", headers):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_SUCCESS, {"provider": "ollama"})
    sent_headers = mock_req.call_args.kwargs["headers"]
    assert sent_headers["X-Token"] == "abc"
    assert sent_headers["X-Source"] == "ollama"


def test_notify_url_template(cfg):
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_URL", "https://example.test/hook?event={event}"):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_FAILURE, {"error": "boom"})
    assert mock_req.call_args.args[1] == "https://example.test/hook?event=failure"


def test_notify_get_method(cfg):
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_METHOD", "GET"):
        with patch("src.utils.notifier.requests.get", return_value=_ok_response()) as mock_get:
            ok = notifier.notify(notifier.EVENT_SUCCESS)
    assert ok is True
    mock_get.assert_called_once()


def test_notify_swallows_request_exceptions(cfg):
    with patch("src.utils.notifier.requests.request", side_effect=requests.ConnectionError("nope")):
        assert notifier.notify(notifier.EVENT_SUCCESS) is False


def test_notify_swallows_http_errors(cfg):
    bad = MagicMock()
    bad.raise_for_status.side_effect = requests.HTTPError("500")
    with patch("src.utils.notifier.requests.request", return_value=bad):
        assert notifier.notify(notifier.EVENT_SUCCESS) is False


def test_notify_invalid_headers_json_ignored(cfg):
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_HEADERS", "not-json"):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_SUCCESS)
    assert mock_req.call_args.kwargs["headers"] == {}


def test_notify_invalid_payload_template_falls_back(cfg):
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_PAYLOAD", "{not valid json"):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_SUCCESS, {"file": "x.txt"})
    payload = mock_req.call_args.kwargs["json"]
    assert payload["event"] == "success"
    assert payload["file"] == "x.txt"


def test_safe_format_leaves_unknown_placeholders():
    assert notifier._safe_format("Hello {missing}", {"event": "success"}) == "Hello {missing}"


def test_safe_format_survives_type_mismatch():
    """`{x:d}` against a string must not raise — template returned as-is."""
    assert notifier._safe_format("{x:d}", {"x": "abc"}) == "{x:d}"


def test_notify_rejects_unknown_event(cfg):
    with patch("src.utils.notifier.requests.request") as mock_req:
        assert notifier.notify("not-a-real-event") is False
        mock_req.assert_not_called()


def test_notify_sanitizes_none_in_custom_template(cfg):
    """A None context value must render as empty, not as the literal 'None'."""
    template = json.dumps({"text": "Error: {error}"})
    with patch.object(notifier.config, "NOTIFY_WEBHOOK_PAYLOAD", template):
        with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
            notifier.notify(notifier.EVENT_SUCCESS, {"error": None})
    payload = mock_req.call_args.kwargs["json"]
    assert payload == {"text": "Error: "}


def test_notify_preserves_none_in_default_payload(cfg):
    """Default JSON payload keeps None as explicit null for JSON consumers."""
    with patch("src.utils.notifier.requests.request", return_value=_ok_response()) as mock_req:
        notifier.notify(notifier.EVENT_SUCCESS, {"file": None, "duration_seconds": None})
    payload = mock_req.call_args.kwargs["json"]
    assert payload["file"] is None
    assert payload["duration_seconds"] is None
    assert payload["error"] is None


def test_known_events_contract():
    """known_events() lists exactly the three terminal events."""
    assert set(notifier.known_events()) == {
        notifier.EVENT_SUCCESS,
        notifier.EVENT_FAILURE,
        notifier.EVENT_INTERRUPTION,
    }
