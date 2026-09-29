import json

from flask import Blueprint, Flask

from src.api.blueprints.mobile_access_routes import (
    MobileAccessEventStore,
    external_android_event,
    mobile_request_context,
    register_mobile_access_routes,
)


def test_mobile_event_store_persists_only_allowlisted_metadata(tmp_path):
    store = MobileAccessEventStore(lambda: str(tmp_path))
    store.append(
        "api",
        {
            "request_time": "2026-07-09 12:00:00 CST",
            "request_host": "100.64.0.10",
            "request_path": "/api/mobile-access",
            "remote_addr": "100.64.0.20",
            "forwarded_for": "",
            "is_android": True,
            "is_mobile": True,
            "user_agent": "Android Device " + ("x" * 300),
            "cookie": "session=secret",
            "request_body": "private text",
            "deepseek_api_key": "sk-secret",
        },
    )

    event = json.loads(store.path.read_text(encoding="utf-8"))

    assert event["kind"] == "api"
    assert event["remote_addr"] == "100.64.0.20"
    assert len(event["user_agent"]) == 240
    assert "cookie" not in event
    assert "request_body" not in event
    assert "deepseek_api_key" not in event


def test_mobile_event_store_skips_corrupt_lines_and_reads_newest_first(tmp_path):
    store = MobileAccessEventStore(lambda: str(tmp_path))
    store.append("first", {"request_path": "/first"})
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
    store.append("second", {"request_path": "/second"})

    events = store.read(limit=2)

    assert [event["kind"] for event in events] == ["second", "first"]


def test_mobile_event_store_compacts_unbounded_history(tmp_path):
    store = MobileAccessEventStore(
        lambda: str(tmp_path),
        max_events=3,
        compact_after_bytes=1,
    )

    for index in range(10):
        store.append(f"event-{index}", {"request_path": f"/{index}"})

    persisted = store.path.read_text(encoding="utf-8").splitlines()
    assert len(persisted) == 3
    assert [event["kind"] for event in store.read(limit=10)] == [
        "event-9",
        "event-8",
        "event-7",
    ]


def test_mobile_detection_does_not_treat_desktop_chrome_as_mobile():
    app = Flask(__name__)
    cases = (
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/150.0.0.0 Safari/537.36",
            False,
            False,
        ),
        (
            "Mozilla/5.0 (Linux; Android 15; SM-S938U) "
            "AppleWebKit/537.36 Chrome/126 Mobile Safari/537.36",
            True,
            True,
        ),
        (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1",
            False,
            True,
        ),
    )

    for user_agent, expected_android, expected_mobile in cases:
        with app.test_request_context(headers={"User-Agent": user_agent}):
            context = mobile_request_context()

        assert context["is_android"] is expected_android
        assert context["is_mobile"] is expected_mobile


def test_external_android_event_excludes_probes_from_the_mac():
    local_ip = "100.64.0.10"

    assert not external_android_event(
        {"is_android": True, "remote_addr": "127.0.0.1"},
        local_ip,
    )
    assert not external_android_event(
        {
            "is_android": True,
            "remote_addr": "127.0.0.1",
            "forwarded_for": local_ip,
        },
        local_ip,
    )
    assert external_android_event(
        {
            "is_android": True,
            "remote_addr": "127.0.0.1",
            "forwarded_for": "100.64.0.20",
        },
        local_ip,
    )
    assert not external_android_event(
        {"is_android": False, "remote_addr": "100.64.0.20"},
        local_ip,
    )


def test_mobile_routes_keep_aliases_and_escape_request_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("PORT", "5000")
    app = Flask(__name__)
    blueprint = Blueprint("mobile_test", __name__)
    register_mobile_access_routes(
        blueprint,
        startup_time=1234567890,
        server_version="1.4.6",
        config_path_provider=lambda: str(tmp_path),
        tailnet_ip_provider=lambda: "100.64.0.10",
    )
    app.register_blueprint(blueprint)

    user_agent = "Mozilla/5.0 (Linux; Android) <script>alert(1)</script>"
    with app.test_client() as client:
        for path in (
            "/mobile",
            "/android",
            "/verbaloom/mobile",
            "/verbaloom/android",
        ):
            response = client.get(path, headers={"User-Agent": user_agent})
            page = response.get_data(as_text=True)
            assert response.status_code == 200
            assert '<html lang="en">' in page
            assert "Android detected" in page
            assert "VerbaLoom mobile access" in page
            assert "https://github.com/alejandrojlamas/verbaloom" in page
            assert "<script>alert(1)</script>" not in page
            assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page
            assert response.headers["Cache-Control"].startswith("no-store")

        payload = client.get(
            "/api/mobile-access",
            headers={"User-Agent": user_agent, "X-Forwarded-Proto": "https"},
        ).get_json()

    assert payload["status"] == "ok"
    assert payload["current_url"].startswith("https://")
    assert payload["recommended_url"] == "http://100.64.0.10/verbaloom"
    assert payload["dns_fallback_url"] == "http://100.64.0.10.nip.io/verbaloom"
    assert payload["fallback_url"] == "http://100.64.0.10:5000/verbaloom"
    assert payload["session_id"] == 1234567890
    assert payload["external_android_seen"] is False

    with app.test_client() as client:
        external_payload = client.get(
            "/api/mobile-access",
            headers={
                "User-Agent": user_agent,
                "X-Forwarded-For": "100.64.0.20",
            },
        ).get_json()

    assert external_payload["external_android_seen"] is True
    assert (
        external_payload["latest_external_android_event"]["forwarded_for"]
        == "100.64.0.20"
    )
