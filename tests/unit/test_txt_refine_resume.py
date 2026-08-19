import pytest

from src.core.refine import txt_refiner


class DummyCheckpointManager:
    def __init__(self):
        self.saved = []

    def load_checkpoint(self, translation_id):
        assert translation_id == "job-1"
        return {
            "chunks": [
                {
                    "chunk_index": 0,
                    "status": "completed",
                    "translated_text": "refinado 0",
                },
                {
                    "chunk_index": 1,
                    "status": "completed",
                    "translated_text": "refinado 1",
                },
            ]
        }

    def save_checkpoint(self, **kwargs):
        self.saved.append(kwargs)
        return True


@pytest.mark.asyncio
async def test_txt_refine_resume_uses_checkpoint_prefix_and_remaining_chunks(tmp_path, monkeypatch):
    output_path = tmp_path / "out.txt"
    manager = DummyCheckpointManager()
    seen = {}

    monkeypatch.setattr(
        txt_refiner,
        "split_text_into_chunks",
        lambda *_args, **_kwargs: [
            {"context_before": "", "main_content": "draft 0", "context_after": ""},
            {"context_before": "", "main_content": "draft 1", "context_after": ""},
            {"context_before": "", "main_content": "draft 2", "context_after": ""},
        ],
    )

    async def fake_refine_chunks(**kwargs):
        seen["translated_chunks"] = kwargs["translated_chunks"]
        seen["chunk_index_offset"] = kwargs["chunk_index_offset"]
        kwargs["checkpoint_callback"](
            2,
            "draft 2",
            "refinado 2",
            {"completed_chunks": 1, "failed_chunks": 0},
        )
        kwargs["stats_callback"]({
            "total_chunks": 1,
            "completed_chunks": 1,
            "failed_chunks": 0,
        })
        return ["refinado 2"]

    monkeypatch.setattr(txt_refiner, "refine_chunks", fake_refine_chunks)

    stats = []
    ok = await txt_refiner.refine_text_content(
        translated_text="draft 0\n\ndraft 1\n\ndraft 2",
        output_filepath=str(output_path),
        target_language="Spanish",
        stats_callback=stats.append,
        checkpoint_manager=manager,
        translation_id="job-1",
        resume_from_index=2,
        prompt_options={"editorial_quality_report": False},
    )

    assert ok is True
    assert seen["translated_chunks"] == ["draft 2"]
    assert seen["chunk_index_offset"] == 2
    assert manager.saved[0]["chunk_index"] == 2
    assert manager.saved[0]["completed_chunks"] == 3
    assert stats[-1]["total_chunks"] == 3
    assert stats[-1]["completed_chunks"] == 3
    assert "refinado 0\nrefinado 1\nrefinado 2" in output_path.read_text(encoding="utf-8")
