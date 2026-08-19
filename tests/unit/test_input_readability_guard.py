import pytest

from src.api.handlers import _ensure_engine_readable_input


def test_known_readable_character_count_skips_extraction(monkeypatch, tmp_path):
    path = tmp_path / "book.epub"
    path.write_bytes(b"placeholder")

    def fail_extract(_path):
        raise AssertionError("extract_readable_text should not be called")

    monkeypatch.setattr("src.api.handlers.extract_readable_text", fail_extract)
    config = {
        "file_type": "epub",
        "prompt_options": {"_input_readable_characters": 123},
    }

    assert _ensure_engine_readable_input(config, str(path)) == 123


def test_worker_readability_guard_rejects_zero_text(monkeypatch, tmp_path):
    path = tmp_path / "empty.epub"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr("src.api.handlers.extract_readable_text", lambda _path: "   ")

    with pytest.raises(RuntimeError, match="no readable text"):
        _ensure_engine_readable_input({"file_type": "epub", "prompt_options": {}}, str(path))


def test_worker_readability_guard_persists_extracted_count(monkeypatch, tmp_path):
    path = tmp_path / "book.txt"
    path.write_text("Hola mundo", encoding="utf-8")
    monkeypatch.setattr("src.api.handlers.extract_readable_text", lambda _path: "Hola mundo")
    config = {"file_type": "txt", "prompt_options": {}}

    assert _ensure_engine_readable_input(config, str(path)) == len("Hola mundo")
    assert config["prompt_options"]["_input_readable_characters"] == len("Hola mundo")
