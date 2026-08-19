"""Checkpoint-derived progress helpers.

Worker memory can drift from persisted checkpoint rows when a job crosses
phases (translation -> refinement) or resumes after a partial failure. These
helpers keep callers from publishing a job as complete while unresolved
checkpoint rows remain.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _chunk_kind(chunk: Dict[str, Any]) -> str:
    data = chunk.get("chunk_data") or {}
    if not isinstance(data, dict):
        return ""
    return str(data.get("file_type") or "").strip()


def _chunk_index(chunk: Dict[str, Any]) -> Optional[int]:
    try:
        return int(chunk.get("chunk_index"))
    except (TypeError, ValueError):
        return None


def _completed_chunk(chunk: Dict[str, Any]) -> bool:
    return chunk.get("status") == "completed" and chunk.get("translated_text") is not None


def _rows_are_logical_chunks(job: Dict[str, Any], chunks: Iterable[Dict[str, Any]]) -> bool:
    """Whether checkpoint rows can be compared 1:1 with progress.total_chunks."""
    config = job.get("config") or {}
    prompt_options = config.get("prompt_options") or {}
    if isinstance(prompt_options, dict) and (
        prompt_options.get("text_first_pipeline")
        or prompt_options.get("text_first_pipeline_active")
    ):
        return True

    file_type = str(job.get("file_type") or "").lower()
    if file_type in {"txt", "srt", "pdf"}:
        return True

    kinds = {_chunk_kind(chunk) for chunk in chunks if _chunk_kind(chunk)}
    if not kinds:
        return False
    # Native EPUB translation checkpoints store one row per XHTML file while
    # progress.total_chunks is the inner text-chunk count. Refinement/text-first
    # checkpoints do store logical chunk rows.
    return "epub_xhtml" not in kinds


def checkpoint_progress_snapshot(checkpoint_manager: Any, translation_id: str) -> Optional[Dict[str, Any]]:
    """Build a progress snapshot from persisted checkpoint state.

    Returns None when no checkpoint exists. The returned dict contains
    checkpoint-safe totals and an `unresolved` flag suitable for finalization
    decisions.
    """
    checkpoint_data = checkpoint_manager.load_checkpoint(translation_id)
    if not checkpoint_data:
        return None

    job = checkpoint_data.get("job") or {}
    progress = dict(job.get("progress") or {})
    chunks = list(checkpoint_data.get("chunks") or [])
    row_truth = _rows_are_logical_chunks(job, chunks)

    total = _as_int(progress.get("total_chunks"), 0)
    if total <= 0:
        total = len(chunks)

    completed_rows = sum(1 for chunk in chunks if _completed_chunk(chunk))
    failed_indices = set()
    for chunk in chunks:
        index = _chunk_index(chunk)
        if index is not None and not _completed_chunk(chunk):
            failed_indices.add(index)

    if row_truth and total > 0:
        rows_by_index = {
            index: chunk
            for chunk in chunks
            if (index := _chunk_index(chunk)) is not None
        }
        for index in range(total):
            chunk = rows_by_index.get(index)
            if chunk is None or not _completed_chunk(chunk):
                failed_indices.add(index)
        failed = len(failed_indices)
        completed = max(0, min(total, total - failed))
    else:
        failed = max(
            _as_int(progress.get("failed_chunks"), 0),
            len(checkpoint_data.get("failed_chunk_indices") or []),
            len(failed_indices),
        )
        completed = min(
            total,
            max(
                _as_int(progress.get("completed_chunks"), 0),
                _as_int(progress.get("current_chunk_index"), -1) + 1,
                completed_rows,
                _as_int(checkpoint_data.get("actual_completed_chunks"), 0),
            ),
        )

    return {
        "total_chunks": total,
        "completed_chunks": completed,
        "failed_chunks": failed,
        "failed_chunk_indices": sorted(failed_indices),
        "checkpoint_complete": bool(checkpoint_data.get("checkpoint_complete")) and failed == 0,
        "rows_are_logical_chunks": row_truth,
        "unresolved": failed > 0,
    }
