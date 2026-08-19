from src.api.services.file_service import FileService


def test_file_listing_hides_atomic_write_temporary_artifacts(tmp_path):
    final = tmp_path / "Book (Spanish).epub"
    staged = tmp_path / "Book (Spanish).heimhm9l.epub"
    finder_metadata = tmp_path / ".DS_Store"
    final.write_bytes(b"final")
    staged.write_bytes(b"staged")
    finder_metadata.write_bytes(b"metadata")

    files = FileService(str(tmp_path)).list_all_files()

    assert [item["filename"] for item in files] == ["Book (Spanish).epub"]


def test_file_listing_keeps_semantic_dotted_names(tmp_path):
    versioned = tmp_path / "Book.v1.epub"
    versioned.write_bytes(b"book")

    files = FileService(str(tmp_path)).list_all_files()

    assert [item["filename"] for item in files] == ["Book.v1.epub"]
