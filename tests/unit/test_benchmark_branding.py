import json
from pathlib import Path

import pytest

from benchmark.cli import _detect_engine_version, create_parser
from benchmark.config import BenchmarkConfig, OpenRouterConfig
from benchmark.models import Submission, TranslationsFile
from scripts.migrate_to_split_layout import ModelBucket, bucket_to_doc, parse_submissions
from tools.prompt_optimizer.config import OpenRouterConfig as OptimizerOpenRouterConfig


REPO_ROOT = Path(__file__).resolve().parents[2]


def _translations_document(version_key: str) -> dict:
    return {
        "schema_version": "2.0",
        "model": {"provider": "ollama", "id": "example-model"},
        "environment": {version_key: "git-abc123", "prompt_version": "v1"},
        "contributors": [
            {"by": "github:hydropix", "at": "2026-05-10T16:01:28Z"}
        ],
        "translations": [
            {
                "text_id": "sample",
                "source_lang": "en",
                "target_lang": "es",
                "output": "Ejemplo",
                "output_hash": f"sha256:{'0' * 64}",
            }
        ],
    }


def test_translation_reader_accepts_legacy_and_writes_engine_version():
    legacy = _translations_document("tbl_version")
    loaded = TranslationsFile.from_dict(legacy)

    assert loaded.engine_version == "git-abc123"
    assert loaded.tbl_version == "git-abc123"
    rewritten = loaded.to_dict()
    assert rewritten["environment"]["engine_version"] == "git-abc123"
    assert "tbl_version" not in rewritten["environment"]
    assert "verbaloom_version" not in rewritten["environment"]


def test_submission_reader_accepts_transitional_version_and_writes_canonical_key():
    document = {
        "schema_version": "1.0",
        "submission": {
            "submitted_by": "github:hydropix",
            "submitted_at": "2026-05-10T16:01:28Z",
        },
        "environment": {
            "verbaloom_version": "v1.2.4",
            "prompt_version": "v1",
            "judge_id": "judge",
        },
        "model": {"provider": "ollama", "id": "example-model"},
        "results": [],
    }

    loaded = Submission.from_dict(document)
    rewritten = loaded.to_dict()
    assert loaded.engine_version == "v1.2.4"
    assert rewritten["environment"]["engine_version"] == "v1.2.4"
    assert "verbaloom_version" not in rewritten["environment"]


def test_translation_schema_accepts_all_read_keys():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (REPO_ROOT / "benchmark/schemas/translations.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.Draft202012Validator(schema)

    for version_key in ("engine_version", "verbaloom_version", "tbl_version"):
        validator.validate(_translations_document(version_key))


def test_cli_exposes_engine_version_and_keeps_legacy_alias_hidden():
    parser = create_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    add_translations = subparsers.choices["add-translations"]
    help_text = add_translations.format_help()
    assert "--engine-version" in help_text
    assert "--tbl-version" not in help_text

    canonical = parser.parse_args(
        [
            "add-translations",
            "input.json",
            "--by",
            "github:example",
            "--provider",
            "ollama",
            "--engine-version",
            "v2",
        ]
    )
    legacy = parser.parse_args(
        [
            "add-translations",
            "input.json",
            "--by",
            "github:example",
            "--provider",
            "ollama",
            "--tbl-version",
            "v1",
        ]
    )
    assert canonical.engine_version == "v2"
    assert legacy.engine_version == "v1"


def test_engine_version_env_is_canonical_with_legacy_fallbacks(monkeypatch):
    monkeypatch.setenv("ENGINE_VERSION", "engine-v3")
    monkeypatch.setenv("VERBALOOM_VERSION", "verbaloom-v2")
    monkeypatch.setenv("TBL_VERSION", "tbl-v1")
    assert _detect_engine_version() == "engine-v3"

    monkeypatch.delenv("ENGINE_VERSION")
    assert _detect_engine_version() == "verbaloom-v2"

    monkeypatch.delenv("VERBALOOM_VERSION")
    assert _detect_engine_version() == "tbl-v1"


def test_migration_reads_legacy_and_writes_engine_version(tmp_path):
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    document = {
        "model": {"id": "example-model", "provider": "ollama"},
        "environment": {"tbl_version": "v0.2.0", "prompt_version": "v1"},
        "submission": {
            "submitted_by": "github:hydropix",
            "submitted_at": "2026-05-10T16:01:28Z",
        },
        "results": [],
    }
    (submissions / "historical.json").write_text(
        json.dumps(document), encoding="utf-8"
    )

    bucket = parse_submissions(submissions)["example-model"]
    output = bucket_to_doc(bucket, [])
    assert output["environment"]["engine_version"] == "v0.2.0"
    assert "tbl_version" not in output["environment"]
    assert "verbaloom_version" not in output["environment"]


def test_public_benchmark_defaults_point_to_verbaloom():
    benchmark = BenchmarkConfig()
    assert OpenRouterConfig.site_url == "https://github.com/alejandrojlamas/verbaloom"
    assert OpenRouterConfig.site_name == "VerbaLoom Benchmark"
    assert benchmark.paths.wiki_repo_url == (
        "https://github.com/alejandrojlamas/verbaloom.wiki.git"
    )
    assert OptimizerOpenRouterConfig.site_url == (
        "https://github.com/alejandrojlamas/verbaloom"
    )
    assert OptimizerOpenRouterConfig.site_name == "VerbaLoom Prompt Optimizer"


def test_migration_bucket_serializes_canonical_version_key():
    bucket = ModelBucket(
        model_id="example-model",
        provider="ollama",
        engine_version="v2",
        prompt_version="v1",
        observations=[],
        contributors=[],
    )
    assert bucket_to_doc(bucket, [])["environment"] == {
        "engine_version": "v2",
        "prompt_version": "v1",
    }
