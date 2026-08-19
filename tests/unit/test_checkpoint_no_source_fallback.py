from __future__ import annotations

from types import SimpleNamespace

from src.persistence.checkpoint_manager import CheckpointManager


def test_legacy_reconstruction_rejects_unresolved_chunks_instead_of_using_source():
    manager = CheckpointManager.__new__(CheckpointManager)
    manager.db = SimpleNamespace(
        get_chunks=lambda _job_id: [
            {
                "chunk_index": 0,
                "status": "completed",
                "translated_text": "Primer fragmento traducido.",
                "original_text": "Erster übersetzter Abschnitt.",
            },
            {
                "chunk_index": 1,
                "status": "failed",
                "translated_text": None,
                "original_text": "Dieser Text darf nicht veröffentlicht werden.",
            },
        ]
    )

    output, error = manager._build_translated_output_legacy("job", "epub_simple")

    assert output is None
    assert "unresolved translation chunk(s): 1" in error
    assert "Dieser Text" not in error
