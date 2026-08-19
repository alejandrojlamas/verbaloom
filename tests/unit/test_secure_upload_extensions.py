import io
import zipfile

from flask import Flask
import yaml

from src.api.blueprints.security_routes import create_security_blueprint
from src.core.book_profiles import create_profile
from src.utils.file_detector import detect_file_type
from src.utils.security import SecureFileHandler, rate_limiter
from src.core.output_formats import extract_readable_text


def _minimal_epub_bytes(extra_members: dict[str, bytes | str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", "<container/>")
        zf.writestr("OEBPS/chapter.xhtml", "<html><body><p>Hello.</p></body></html>")
        for name, content in (extra_members or {}).items():
            zf.writestr(name, content)
    return buffer.getvalue()


def _empty_epub_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", "<container/>")
        zf.writestr("OEBPS/chapter.xhtml", "<html><body><p>   </p></body></html>")
    return buffer.getvalue()


def test_upload_preserves_detected_epub_suffix_when_filename_has_dotted_source(tmp_path):
    handler = SecureFileHandler(tmp_path)
    result = handler.validate_and_save_file(
        _minimal_epub_bytes(),
        "Beyond Vibe Coding From Cod_ (z-library.sk, 1lib.sk, z-lib.sk)",
    )

    assert result.is_valid, result.error_message
    assert result.file_path is not None
    assert result.file_path.suffix == ".epub"
    assert detect_file_type(str(result.file_path)) == "epub"


def test_readable_text_extraction_uses_content_detection_for_extensionless_epub(tmp_path):
    path = tmp_path / "4854_Beyond_Vibe_Coding_z-lib.sk_"
    path.write_bytes(_minimal_epub_bytes())

    assert detect_file_type(str(path)) == "epub"
    assert extract_readable_text(path) == "Hello."


def test_upload_removes_plaintext_domain_marker_file_from_epub(tmp_path):
    handler = SecureFileHandler(tmp_path)
    result = handler.validate_and_save_file(
        _minimal_epub_bytes({"oceanofpdf.com": "Downloaded from oceanofpdf.com"}),
        "_OceanofPDF.com_Book.epub",
    )

    assert result.is_valid, result.error_message
    assert any("oceanofpdf.com" in warning for warning in result.warnings)
    with zipfile.ZipFile(result.file_path, "r") as zf:
        assert "oceanofpdf.com" not in zf.namelist()
        assert "OEBPS/chapter.xhtml" in zf.namelist()


def test_upload_rejects_binary_com_file_inside_epub(tmp_path):
    handler = SecureFileHandler(tmp_path)
    result = handler.validate_and_save_file(
        _minimal_epub_bytes({"oceanofpdf.com": b"\x00\x01\x02MZ"}),
        "Book.epub",
    )

    assert not result.is_valid
    assert "EPUB contains suspicious file: oceanofpdf.com" in result.error_message


def test_upload_route_reports_readable_characters_for_epub(tmp_path):
    rate_limiter._requests.clear()
    app = Flask(__name__)
    app.register_blueprint(create_security_blueprint(tmp_path))

    response = app.test_client().post(
        "/api/upload",
        data={"file": (io.BytesIO(_minimal_epub_bytes()), "Book.epub")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["file_type"] == "epub"
    assert body["readable"] is True
    assert body["readable_characters"] == len("Hello.")
    assert body["file_path"] == body["secure_filename"]
    assert "/" not in body["file_path"] and "\\" not in body["file_path"]


def test_upload_route_reports_inferred_book_profile_for_mobile_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path / "profiles"))
    profile_dir = create_profile(
        "auto_under_the_volcano_malcolm_lowry",
        profiles_root=tmp_path / "profiles",
    )
    profile_path = profile_dir / "profile.yml"
    profile_config = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    profile_config.update({
        "source_name": "Under the Volcano - Malcolm Lowry.epub",
        "target_locale": "es-MX",
        "generated_profile": True,
        "auto_detect": {"enabled": False},
    })
    profile_path.write_text(
        yaml.safe_dump(profile_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    rate_limiter._requests.clear()
    app = Flask(__name__)
    app.register_blueprint(create_security_blueprint(tmp_path))

    response = app.test_client().post(
        "/api/upload",
        data={
            "file": (
                io.BytesIO(_minimal_epub_bytes()),
                "Under%20the%20Volcano%20-%20Malcolm%20Lowry.epub",
            )
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert body["book_profile"]["profile_id"] == "auto_under_the_volcano_malcolm_lowry"
    assert body["book_profile"]["target_locale"] == "es-MX"
    assert body["readable_characters"] == len("Hello.")


def test_upload_route_rejects_epub_without_readable_text(tmp_path):
    rate_limiter._requests.clear()
    app = Flask(__name__)
    app.register_blueprint(create_security_blueprint(tmp_path))

    response = app.test_client().post(
        "/api/upload",
        data={"file": (io.BytesIO(_empty_epub_bytes()), "Under the Volcano.epub")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    body = response.get_json()
    assert body["error"] == "Uploaded file has no readable text"
    assert body["details"]["file_type"] == "epub"
    assert body["details"]["readable_characters"] == 0
    assert list((tmp_path / "uploads").iterdir()) == []
