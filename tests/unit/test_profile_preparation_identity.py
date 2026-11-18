"""Selection and preparation must keep an editorial profile scoped to its job."""

import asyncio

import pytest
import yaml
from flask import Flask

from src.api.blueprints.profile_routes import (
    _profile_prep_request_fingerprint,
    create_profile_blueprint,
)
from src.core.book_profiles import (
    create_profile,
    infer_profile_id_from_metadata,
    load_book_profile,
    prepare_book_profile_from_text,
)


BOOK = "An explorer recorded the height of the lake after each storm. " * 8
TITLE = "The lake in winter.epub"
GOALS = {
    "translation": "faithful_translation",
    "audio": "audiobook",
    "modernize": "modernization",
    "explain": "explanatory_rewrite",
    "literary": "literary_polish",
    "academic": "academic_translation",
}


def approve_one(profile_dir):
    (profile_dir / "glossary/terms.yml").write_text(
        "entries:\n- source: ELL\n  target: ELL\n  status: approved\n", encoding="utf-8",
    )


def prepare(root, text=BOOK, **overrides):
    params = {"source_name": TITLE, "language": "Spanish", "target_locale": "es-MX",
              "source_language": "English", "profile_goal": "translation", "max_llm_chunks": 0}
    return asyncio.run(prepare_book_profile_from_text(text, profiles_root=root, **(params | overrides)))


def test_http_prepare_all_goals_for_one_book_keep_distinct_scopes(tmp_path, monkeypatch):
    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    client = app.test_client()
    ids = {}
    for alias, goal in GOALS.items():
        payload = {
            "text": BOOK, "source_name": TITLE, "source_language": "English",
            "language": "Spanish", "target_locale": "es-MX", "profile_name": "The lake",
            "profile_goal": alias, "transform_mode": "modernize", "provider": "ollama",
            "max_llm_chunks": 0,
        }
        reply = client.post("/api/book-profiles/prepare", json=payload)
        assert reply.status_code == 201, reply.data
        row = reply.get_json()["profile"]
        profile_id = row["profile_id"]
        assert row["profile_goal"] == goal
        assert row["coverage_mode"] != "reused"
        profile = load_book_profile(profile_id, profiles_root=tmp_path)
        assert profile.raw_config["business_rules"]["goal"] == goal
        assert bool(profile.raw_config.get("audiobook", {}).get("generate_companion")) == (goal == "audiobook")
        assert (profile.raw_config["transform_mode"] == "modernize") == (goal == "modernization")
        approve_one(profile.root)
        ids[goal] = profile_id

        again = client.post("/api/book-profiles/prepare", json=payload | {"profile_goal": goal})
        assert again.status_code == 201, again.data
        reused = again.get_json()["profile"]
        assert (reused["profile_id"], reused["coverage_mode"]) == (profile_id, "reused")

    assert len(set(ids.values())) == len(GOALS)
    listing = client.get("/api/book-profiles").get_json()["profiles"]
    assert {row["profile_goal"]: row["profile_id"] for row in listing} == ids
    assert all(row["target_language"] == "Spanish" and row["profile_goal_label"] for row in listing)
    assert infer_profile_id_from_metadata({"input_filename": TITLE}, profiles_root=tmp_path) is None
    for alias, goal in GOALS.items():
        inferred = infer_profile_id_from_metadata(
            {"input_filename": TITLE, "profile_goal": alias, "target_language": "es"}, profiles_root=tmp_path,
        )
        assert inferred == ids[goal]


@pytest.mark.parametrize("change", [
    {"text": BOOK + "New edition."},
    {"source_name": "Another lake.epub"},
    {"source_language": "French"},
    {"language": "French", "target_locale": "fr-FR"},
    {"target_locale": "es-ES"},
    {"profile_goal": "audio"},
])
def test_requested_id_cannot_reuse_an_incompatible_profile(tmp_path, change):
    first = prepare(tmp_path, profile_id="my_book")
    approve_one(first.profile_path)
    original = (first.profile_path / "profile.yml").read_bytes()
    second = prepare(tmp_path, profile_id="my_book", **change)
    assert second.profile_id != first.profile_id
    approve_one(second.profile_path)
    again = prepare(tmp_path, profile_id="my_book", **change)
    assert (again.profile_id, again.coverage_mode) == (second.profile_id, "reused")
    assert (first.profile_path / "profile.yml").read_bytes() == original


def test_infer_locales_and_preserve_explicit_selection_with_short_unicode_title(tmp_path):
    first = prepare(tmp_path, source_name="水辺.epub")
    second = prepare(tmp_path, source_name="水辺.epub", target_locale="es-ES")
    document = {"input_filename": "水辺.epub"}
    assert infer_profile_id_from_metadata(document, profiles_root=tmp_path) is None
    assert infer_profile_id_from_metadata(document | {"target_locale": "ES_es"}, profiles_root=tmp_path) == second.profile_id
    assert infer_profile_id_from_metadata(document | {"profile_id": first.profile_id}, profiles_root=tmp_path) == first.profile_id


def test_legacy_approved_alias_survives_empty_duplicate_and_mismatched_internal_id(tmp_path):
    for profile_id in ("a_empty", "z_complete"):
        path = create_profile(profile_id, profiles_root=tmp_path)
        config_path = path / "profile.yml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config.update({"profile_id": "stale_name", "generated_profile": True, "source_name": TITLE,
                       "business_rules": {"goal": "audio"}, "source_language": "English",
                       "target_language": "Spanish", "target_locale": "es-MX"})
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    approve_one(path)
    original = config_path.read_bytes()
    selected = prepare(tmp_path, profile_goal="audiobook")
    assert (selected.profile_id, selected.profile_goal, selected.coverage_mode) == ("z_complete", "audiobook", "reused")
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("value", [
    {"source_language": "en"}, {"profile_name": "Alternative edition"}, {"review_provider_type": "openai"},
    {"provider_type": "ollama"}, {"api_endpoint": "http://localhost:11434"}, {"max_local_terms": 8},
    {"llm_chunk_chars": 3000}, {"max_llm_chunks": 5}, {"llm_full_coverage": True},
])
def test_inflight_deduplication_distinguishes_preparation_options(value):
    request = {"combined_text": BOOK, "profile_goal": "audiobook"}
    fingerprint = _profile_prep_request_fingerprint(request)
    assert fingerprint == _profile_prep_request_fingerprint(request | {"api_key": "credential"})
    assert fingerprint != _profile_prep_request_fingerprint(request | value)


def test_filename_inference_abstains_for_two_editions_with_the_same_name_and_purpose(tmp_path):
    first = prepare(tmp_path)
    second = prepare(tmp_path, text=BOOK + "Revised.")
    assert first.profile_id != second.profile_id
    assert infer_profile_id_from_metadata({"input_filename": TITLE}, profiles_root=tmp_path) is None


def test_complete_source_title_outranks_a_shared_shortened_title(tmp_path):
    a = prepare(tmp_path, source_name="A season at the observatory - First Author.epub")
    approve_one(a.profile_path)
    b = prepare(tmp_path, source_name="A season at the observatory - Second Author.epub")
    short = {"input_filename": "A season at the observatory.epub"}
    assert infer_profile_id_from_metadata(short, profiles_root=tmp_path) is None
    full = {"input_filename": "A season at the observatory - Second Author.epub"}
    assert infer_profile_id_from_metadata(full, profiles_root=tmp_path) == b.profile_id


def test_prep_queue_deduplicates_repeats_and_keeps_distinct_goals(tmp_path, monkeypatch):
    import src.api.blueprints.profile_routes as routes

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    monkeypatch.setattr(routes, "_PROFILE_PREP_JOBS", {})
    monkeypatch.setattr(routes, "_run_profile_prep_job", lambda *_: None)
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    client = app.test_client()
    payload = {
        "text": BOOK, "source_name": TITLE, "source_language": "English",
        "language": "Spanish", "target_locale": "es-MX", "provider": "ollama", "max_llm_chunks": 0,
    }

    def enqueue(**overrides):
        response = client.post("/api/book-profiles/prepare-jobs", json=payload | overrides)
        assert response.status_code == 202
        return response.get_json()

    audio = enqueue(profile_goal="audio")
    same = enqueue(profile_goal="audiobook")
    literary = enqueue(profile_goal="literary")
    assert same["recovered_existing"] and same["prep_id"] == audio["prep_id"]
    assert literary["prep_id"] != audio["prep_id"]
    jobs = client.get("/api/book-profiles/prepare-jobs").get_json()["jobs"]
    assert {item["profile_goal"] for item in jobs} == {"audiobook", "literary_polish"}
    assert all(item["language"] == "Spanish" and item["target_locale"] == "es-MX" for item in jobs)


def test_incomplete_profile_with_approved_entries_is_not_reused_or_inferred(tmp_path):
    first = prepare(tmp_path)
    approve_one(first.profile_path)
    (first.profile_path / ".preparing").touch()
    assert infer_profile_id_from_metadata({"input_filename": TITLE}, profiles_root=tmp_path) is None
    second = prepare(tmp_path)
    assert second.profile_id != first.profile_id
    assert second.coverage_mode != "reused"


def test_preparing_job_uses_actual_id_and_cannot_be_selected_or_deleted(tmp_path, monkeypatch):
    import src.api.blueprints.profile_routes as routes

    monkeypatch.setenv("BOOK_PROFILES_DIR", str(tmp_path))
    profile = prepare(tmp_path, profile_id="my_book")
    (profile.profile_path / ".preparing").touch()
    monkeypatch.setattr(routes, "_PROFILE_PREP_JOBS", {"abc": {"profile_id": "my_book", "status": "running"}})
    app = Flask(__name__)
    app.register_blueprint(create_profile_blueprint())
    client = app.test_client()
    row = client.get("/api/book-profiles").get_json()["profiles"][0]
    assert row["profile_id"] == "my_book" and row["load_error"] and row["in_use"]
    assert client.delete("/api/book-profiles/my_book").status_code == 409
    assert profile.profile_path.is_dir()
