from __future__ import annotations

import json

from src.api.sample_state import SampleStateManager


def test_sample_state_persists_redacted_snapshot(tmp_path):
    path = tmp_path / "sample_runs.json"
    manager = SampleStateManager(persist_path=path, ttl_seconds=3600)
    manager.create(
        "sample_1",
        [{"index": 1, "source_text": "Hello", "truncated": False}],
        [{"provider": "deepseek", "model": "deepseek-v4-pro", "api_key": "SECRET"}],
        "translate",
    )
    manager.set_run_context("sample_1", {
        "items": [{"index": 1, "source_text": "Hello", "truncated": False}],
        "columns": [{"provider": "deepseek", "api_key": "SECRET"}],
        "mode": "translate",
    })
    manager.update_cell(
        "sample_1",
        0,
        0,
        "translate",
        status="done",
        output="Hola",
        metrics={"prompt_tokens": 10},
    )

    raw = path.read_text(encoding="utf-8")
    assert "SECRET" not in raw
    payload = json.loads(raw)
    assert payload["samples"]["sample_1"]["columns"][0]["api_key"] is None

    restored = SampleStateManager(persist_path=path, ttl_seconds=3600)
    snapshot = restored.get("sample_1")

    assert snapshot is not None
    assert snapshot["cells"][0]["output"] == "Hola"
    assert snapshot["columns"][0]["api_key"] is None
