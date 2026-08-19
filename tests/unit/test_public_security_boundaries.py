import io
import zipfile

import pytest

from src.api.services.network_security import (
    allowed_browser_origins,
    browser_origin_is_allowed,
    network_bind_is_explicitly_allowed,
)
from src.api.services.path_validator import PathValidator
from src.api.services.file_service import FileService
from src.api.blueprints.cost_routes import _resolve_text_input
from src.api.blueprints.sample_routes import _instantiate_provider
from src.utils.provider_security import (
    EndpointCredentialError,
    resolve_api_key_for_endpoint,
)
from src.utils.archive_safety import (
    UnsafeArchiveError,
    safe_extractall,
    validate_zip_archive,
)
from src.core.llm.factory import create_llm_provider


OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
CUSTOM_ENDPOINT = "https://gateway.example.test/v1/chat/completions"


def test_custom_endpoint_does_not_receive_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    key, source = resolve_api_key_for_endpoint(
        None,
        "OPENAI_API_KEY",
        endpoint=CUSTOM_ENDPOINT,
        default_endpoint=OPENAI_ENDPOINT,
    )

    assert key == ""
    assert source == "none"


def test_custom_endpoint_accepts_only_explicit_or_allowlisted_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    explicit, source = resolve_api_key_for_endpoint(
        "request-secret",
        "OPENAI_API_KEY",
        endpoint=CUSTOM_ENDPOINT,
        default_endpoint=OPENAI_ENDPOINT,
    )
    assert explicit == "request-secret"
    assert source == "explicit"

    with pytest.raises(EndpointCredentialError):
        resolve_api_key_for_endpoint(
            "__USE_ENV__",
            "OPENAI_API_KEY",
            endpoint=CUSTOM_ENDPOINT,
            default_endpoint=OPENAI_ENDPOINT,
        )

    monkeypatch.setenv("VERBALOOM_TRUSTED_KEY_ENDPOINTS", "https://gateway.example.test")
    trusted, source = resolve_api_key_for_endpoint(
        "__USE_ENV__",
        "OPENAI_API_KEY",
        endpoint=CUSTOM_ENDPOINT,
        default_endpoint=OPENAI_ENDPOINT,
    )
    assert trusted == "environment-secret"
    assert source == "environment"


def test_configured_custom_endpoint_is_not_implicitly_trusted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    key, source = resolve_api_key_for_endpoint(
        None,
        "OPENAI_API_KEY",
        endpoint=CUSTOM_ENDPOINT,
        default_endpoint=CUSTOM_ENDPOINT,
    )

    assert key == ""
    assert source == "none"


def test_nim_factory_honors_explicit_no_environment_fallback(monkeypatch):
    monkeypatch.setenv("NIM_API_KEY", "environment-secret")

    with pytest.raises(ValueError, match="NVIDIA NIM provider requires an API key"):
        create_llm_provider(
            "nim",
            api_endpoint=CUSTOM_ENDPOINT,
            nim_api_key="",
        )

    with pytest.raises(ValueError, match="NVIDIA NIM provider requires an API key"):
        create_llm_provider("nim", api_endpoint=CUSTOM_ENDPOINT)

    with pytest.raises(ValueError, match="NVIDIA NIM provider requires an API key"):
        _instantiate_provider({
            "provider": "nim",
            "model": "vendor/model",
            "api_endpoint": CUSTOM_ENDPOINT,
        })


def test_same_origin_and_loopback_are_the_default(monkeypatch):
    assert allowed_browser_origins("") == []
    assert allowed_browser_origins("*, https://ui.example.test/path") == [
        "https://ui.example.test"
    ]
    assert network_bind_is_explicitly_allowed("127.0.0.1") is True

    monkeypatch.delenv("VERBALOOM_ALLOW_NETWORK_BIND", raising=False)
    monkeypatch.delenv("TBL_ALLOW_NETWORK_BIND", raising=False)
    assert network_bind_is_explicitly_allowed("0.0.0.0") is False
    monkeypatch.setenv("VERBALOOM_ALLOW_NETWORK_BIND", "true")
    assert network_bind_is_explicitly_allowed("0.0.0.0") is True
    assert browser_origin_is_allowed(
        "https://lab.example.test",
        "https://lab.example.test",
    )
    assert not browser_origin_is_allowed(
        "https://attacker.example.test",
        "https://lab.example.test",
    )


def test_legacy_security_environment_aliases_remain_readable(monkeypatch):
    monkeypatch.delenv("VERBALOOM_ALLOW_NETWORK_BIND", raising=False)
    monkeypatch.setenv("TBL_ALLOW_NETWORK_BIND", "true")
    assert network_bind_is_explicitly_allowed("0.0.0.0") is True

    monkeypatch.delenv("VERBALOOM_TRUSTED_KEY_ENDPOINTS", raising=False)
    monkeypatch.setenv("TBL_TRUSTED_KEY_ENDPOINTS", "https://gateway.example.test")
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-environment-secret")
    key, source = resolve_api_key_for_endpoint(
        "__USE_ENV__",
        "OPENAI_API_KEY",
        endpoint=CUSTOM_ENDPOINT,
        default_endpoint=OPENAI_ENDPOINT,
    )
    assert (key, source) == ("legacy-environment-secret", "environment")


def test_managed_file_resolution_rejects_sibling_prefix_and_arbitrary_absolute(tmp_path):
    managed = tmp_path / "uploads"
    managed.mkdir()
    source = managed / "book.txt"
    source.write_text("book", encoding="utf-8")
    assert PathValidator.resolve_managed_file(source, [managed]) == source.resolve()
    assert PathValidator.resolve_managed_file("book.txt", [managed]) == source.resolve()

    sibling = tmp_path / "uploads-evil"
    sibling.mkdir()
    outside = sibling / "private.txt"
    outside.write_text("private", encoding="utf-8")
    with pytest.raises(ValueError):
        PathValidator.resolve_managed_file(outside, [managed])
    with pytest.raises(ValueError):
        PathValidator.resolve_managed_file("/etc/hosts", [managed])

    link = managed / "outside-link.txt"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        PathValidator.resolve_managed_file(link, [managed])


@pytest.mark.parametrize("filename", [".", "..", "chapter\n.txt", "chapter\x00.txt"])
def test_output_filename_rejects_dot_segments_and_control_characters(filename):
    valid, _error = PathValidator.validate_filename(filename)
    assert valid is False


def test_open_file_rejects_executable_suffix_before_os_dispatch(tmp_path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "payload.cmd").write_text("not executed", encoding="utf-8")

    success, message, path = FileService(output_dir).open_file("payload.cmd")

    assert success is False
    assert "security" in message.lower()
    assert path is None


def test_tts_voice_prompt_is_confined_to_managed_storage(tmp_path, monkeypatch):
    from flask import Flask

    import src.api.blueprints.tts_routes as tts_routes

    class SocketStub:
        def emit(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(tts_routes, "is_chatterbox_available", lambda: True)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "book.txt").write_text("book", encoding="utf-8")
    app = Flask(__name__)
    app.register_blueprint(tts_routes.create_tts_blueprint(output_dir, SocketStub()))

    with app.test_client() as client:
        response = client.post(
            "/api/tts/generate",
            json={
                "filename": "book.txt",
                "tts_provider": "chatterbox",
                "tts_voice_prompt_path": "/etc/hosts",
            },
        )

    assert response.status_code == 403


def test_tts_voice_prompt_api_uses_relative_identifiers(tmp_path):
    from flask import Flask

    from src.api.blueprints.tts_routes import create_tts_blueprint

    class SocketStub:
        def emit(self, *_args, **_kwargs):
            return None

    app = Flask(__name__)
    app.register_blueprint(create_tts_blueprint(tmp_path / "outputs", SocketStub()))

    with app.test_client() as client:
        uploaded = client.post(
            "/api/tts/voice-prompt/upload",
            data={"file": (io.BytesIO(b"RIFF-safe-fixture"), "voice.wav")},
            content_type="multipart/form-data",
        )
        listing = client.get("/api/tts/voice-prompts")

    assert uploaded.status_code == 200
    upload_path = uploaded.get_json()["path"]
    assert "/" not in upload_path and "\\" not in upload_path
    prompt = listing.get_json()["voice_prompts"][0]
    assert prompt["path"] == prompt["filename"] == upload_path
    assert listing.get_json()["directory"] == "voice_prompts"


def test_cost_estimation_reads_only_managed_uploads(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    managed = uploads / "book.txt"
    managed.write_text("managed content", encoding="utf-8")
    assert _resolve_text_input({"file_path": "book.txt"}, uploads) == "managed content"

    sibling = tmp_path / "uploads-private"
    sibling.mkdir()
    outside = sibling / "private.txt"
    outside.write_text("private content", encoding="utf-8")
    assert _resolve_text_input({"file_path": str(outside)}, uploads) is None


def test_archive_validation_blocks_traversal_and_excessive_expansion(tmp_path):
    traversal_bytes = io.BytesIO()
    with zipfile.ZipFile(traversal_bytes, "w") as archive:
        archive.writestr("../outside.txt", b"no")
    traversal_bytes.seek(0)
    with zipfile.ZipFile(traversal_bytes) as archive:
        with pytest.raises(UnsafeArchiveError):
            safe_extractall(archive, tmp_path / "extract")

    compressed_bytes = io.BytesIO()
    with zipfile.ZipFile(compressed_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("huge.txt", b"A" * 2_000_000)
    compressed_bytes.seek(0)
    with zipfile.ZipFile(compressed_bytes) as archive:
        with pytest.raises(UnsafeArchiveError):
            validate_zip_archive(archive)


def test_safe_archive_extracts_ordinary_document(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("OEBPS/chapter.xhtml", b"<p>Hello</p>")
    payload.seek(0)
    with zipfile.ZipFile(payload) as archive:
        safe_extractall(archive, tmp_path)

    assert (tmp_path / "OEBPS" / "chapter.xhtml").read_bytes() == b"<p>Hello</p>"
