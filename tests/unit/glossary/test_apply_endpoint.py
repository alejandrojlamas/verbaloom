import os
import sys
import tempfile
from pathlib import Path

import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api.blueprints.glossary_routes import create_glossary_blueprint
from src.core.glossary.models import GlossaryTerm
from src.core.glossary.store import GlossaryStore


@pytest.fixture
def store():
    db = os.path.join(
        tempfile.gettempdir(),
        f"glossary_apply_endpoint_{os.getpid()}_{id(object())}.db",
    )
    if os.path.exists(db):
        os.remove(db)
    s = GlossaryStore(db_path=db)
    try:
        yield s
    finally:
        s.close_all()
        try:
            os.remove(db)
        except OSError:
            pass


def test_apply_correction_endpoint_updates_term_and_creates_copy(tmp_path, store):
    output_file = tmp_path / "book.txt"
    output_file.write_text("autoatencion and autoatencion.", encoding="utf-8")
    glossary = store.create_glossary("G", source_language="English", target_language="Spanish")
    term = store.add_term(
        glossary.id,
        GlossaryTerm(source_term="self-attention", translated_term="autoatencion"),
    )

    app = Flask(__name__)
    app.register_blueprint(create_glossary_blueprint(store=store, output_dir=str(tmp_path)))

    with app.test_client() as client:
        response = client.post(
            f"/api/glossaries/{glossary.id}/terms/{term.id}/apply-correction",
            json={
                "old_target": "autoatencion",
                "new_target": "autoatención",
                "filenames": ["book.txt"],
            },
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["total_replacements"] == 2
    assert body["term"]["target"] == "autoatención"
    assert output_file.read_text(encoding="utf-8") == "autoatencion and autoatencion."

    corrected_name = body["files"][0]["output_filename"]
    corrected = tmp_path / corrected_name
    assert corrected_name != "book.txt"
    assert corrected.read_text(encoding="utf-8") == "autoatención and autoatención."
