"""Token usage and cost ledger routes."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from src.core.usage import default_usage_store


def create_usage_blueprint(output_dir):
    bp = Blueprint("usage", __name__)

    @bp.route("/api/usage/summary", methods=["GET"])
    def usage_summary():
        limit = _int_arg("limit", 80, 1, 500)
        summary = default_usage_store().summary(limit=limit)
        jobs_db = Path(output_dir).parent / "data" / "jobs.db"
        summary["live_jobs"] = _build_live_jobs(jobs_db, summary, limit=limit)
        return jsonify(summary)

    @bp.route("/api/usage/events", methods=["GET"])
    def usage_events():
        limit = _int_arg("limit", 200, 1, 1000)
        translation_id = (request.args.get("translation_id") or "").strip() or None
        return jsonify({"events": default_usage_store().events(limit=limit, translation_id=translation_id)})

    @bp.route("/api/usage/backfill-checkpoints", methods=["POST"])
    def usage_backfill_checkpoints():
        data = request.get_json(silent=True) or {}
        translation_id = str(data.get("translation_id") or "").strip() or None
        jobs_db = Path(output_dir).parent / "data" / "jobs.db"
        result = _backfill_from_checkpoints(jobs_db, translation_id=translation_id)
        return jsonify(result)

    return bp


def _int_arg(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def _estimate_tokens(text: str) -> int:
    return max(0, int(len(text or "") / 3.7))


def _safe_json(raw: str | bytes | None, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except (TypeError, ValueError):
        return default


def _progress_percent(progress: dict[str, Any]) -> float:
    for key in ("percent", "progress_percent", "progress"):
        value = progress.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return max(0.0, min(100.0, float(value)))
    total = _as_int(progress.get("total_chunks"))
    completed = _as_int(progress.get("completed_chunks"))
    failed = _as_int(progress.get("failed_chunks"))
    if total > 0:
        return max(0.0, min(100.0, ((completed + failed) / total) * 100))
    return 0.0


def _load_jobs(jobs_db: Path, usage_ids: set[str], limit: int) -> dict[str, dict[str, Any]]:
    if not jobs_db.exists():
        return {}

    active_statuses = {
        "running", "processing", "queued", "pricing_wait", "paused", "error"
    }
    jobs: dict[str, dict[str, Any]] = {}
    with sqlite3.connect(jobs_db) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT translation_id, status, file_type, config, progress, created_at, updated_at, paused_at, completed_at
            FROM translation_jobs
            WHERE status IN ('running', 'processing', 'queued', 'pricing_wait', 'paused', 'error')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        ids_to_fetch = set(usage_ids)
        for row in rows:
            ids_to_fetch.add(row["translation_id"])

        if ids_to_fetch:
            placeholders = ",".join("?" for _ in ids_to_fetch)
            rows = list(rows) + conn.execute(
                f"""
                SELECT translation_id, status, file_type, config, progress, created_at, updated_at, paused_at, completed_at
                FROM translation_jobs
                WHERE translation_id IN ({placeholders})
                """,
                tuple(ids_to_fetch),
            ).fetchall()

        for row in rows:
            tid = row["translation_id"]
            if not tid or tid in jobs:
                continue
            config = _safe_json(row["config"], {})
            progress = _safe_json(row["progress"], {})
            total_chunks = _as_int(progress.get("total_chunks"))
            completed_chunks = _as_int(progress.get("completed_chunks"))
            failed_chunks = _as_int(progress.get("failed_chunks"))
            jobs[tid] = {
                "translation_id": tid,
                "status": row["status"] or "unknown",
                "is_active": (row["status"] or "").lower() in active_statuses,
                "file_type": row["file_type"] or "",
                "book_name": (
                    config.get("original_filename")
                    or config.get("input_filename")
                    or config.get("output_filename")
                    or tid
                ),
                "input_filename": config.get("input_filename") or config.get("original_filename") or "",
                "output_filename": config.get("output_filename") or "",
                "model": config.get("model") or "",
                "provider": config.get("llm_provider") or "",
                "process_type": _job_process_type(config),
                "current_chunk_index": _as_int(progress.get("current_chunk_index"), -1),
                "total_chunks": total_chunks,
                "completed_chunks": completed_chunks,
                "failed_chunks": failed_chunks,
                "progress_percent": _progress_percent(progress),
                "phase": progress.get("phase") or progress.get("current_phase") or "",
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "paused_at": row["paused_at"],
                "completed_at": row["completed_at"],
            }
    return jobs


def _job_process_type(config: dict[str, Any]) -> str:
    if config.get("operation_mode") == "transform" or config.get("text_process"):
        return f"transform_{config.get('text_process') or 'custom'}"
    if config.get("refine_only"):
        return "editorial_refinement"
    if config.get("enable_refinement"):
        return "translation_with_refinement"
    return "translation"


def _build_live_jobs(jobs_db: Path, summary: dict[str, Any], limit: int = 80) -> list[dict[str, Any]]:
    by_translation = summary.get("by_translation") or []
    usage_by_id = {
        str(row.get("translation_id")): dict(row)
        for row in by_translation
        if row.get("translation_id")
    }
    phase_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in summary.get("phase_by_translation") or []:
        tid = str(row.get("translation_id") or "")
        if not tid:
            continue
        phase_by_id.setdefault(tid, []).append(dict(row))
    recent_by_id = {
        str(row.get("translation_id")): dict(row)
        for row in summary.get("recent_by_translation") or []
        if row.get("translation_id")
    }

    jobs = _load_jobs(jobs_db, set(usage_by_id.keys()), limit=limit)
    recent_cutoff = time.time() - 15 * 60
    live: list[dict[str, Any]] = []

    all_ids = set(jobs.keys()) | set(usage_by_id.keys())
    for tid in all_ids:
        job = jobs.get(tid, {})
        usage = usage_by_id.get(tid, {})
        last_seen = _as_float(usage.get("last_seen"))
        is_recent_usage = bool(last_seen and last_seen >= recent_cutoff)
        is_active = bool(job.get("is_active")) or (not job and is_recent_usage)
        if not is_active:
            continue

        row = {**usage, **job}
        row["translation_id"] = tid
        row["book_name"] = row.get("book_name") or usage.get("book_name") or tid
        row["calls"] = _as_int(usage.get("calls"))
        row["prompt_tokens"] = _as_int(usage.get("prompt_tokens"))
        row["completion_tokens"] = _as_int(usage.get("completion_tokens"))
        row["total_tokens"] = _as_int(usage.get("total_tokens"))
        row["prompt_cache_hit_tokens"] = _as_int(usage.get("prompt_cache_hit_tokens"))
        row["prompt_cache_miss_tokens"] = _as_int(usage.get("prompt_cache_miss_tokens"))
        cache_total = row["prompt_cache_hit_tokens"] + row["prompt_cache_miss_tokens"]
        row["prompt_cache_hit_ratio"] = (
            row["prompt_cache_hit_tokens"] / cache_total if cache_total > 0 else 0.0
        )
        row["input_cost_usd"] = _as_float(usage.get("input_cost_usd"))
        row["output_cost_usd"] = _as_float(usage.get("output_cost_usd"))
        row["total_cost_usd"] = _as_float(usage.get("total_cost_usd"))
        row["estimated_events"] = _as_int(usage.get("estimated_events"))
        row["first_seen"] = usage.get("first_seen")
        row["last_seen"] = usage.get("last_seen")
        row["phase_breakdown"] = _with_phase_shares(phase_by_id.get(tid, []), row["total_tokens"], row["total_cost_usd"])
        row.update(_recent_usage_fields(recent_by_id.get(tid, {}), summary.get("recent_window_seconds") or 300))
        row.update(_projection_fields(row))
        live.append(row)

    live.sort(
        key=lambda row: (
            0 if row.get("is_active") else 1,
            _as_float(row.get("last_seen")),
            str(row.get("updated_at") or ""),
        ),
        reverse=True,
    )
    return live[:limit]


def _with_phase_shares(
    phases: list[dict[str, Any]],
    total_tokens: int,
    total_cost_usd: float,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for phase in phases:
        row = dict(phase)
        tokens = _as_int(row.get("total_tokens"))
        cost = _as_float(row.get("total_cost_usd"))
        cache_hit = _as_int(row.get("prompt_cache_hit_tokens"))
        cache_miss = _as_int(row.get("prompt_cache_miss_tokens"))
        cache_total = cache_hit + cache_miss
        row["token_share_pct"] = (tokens / total_tokens * 100) if total_tokens > 0 else 0.0
        row["cost_share_pct"] = (cost / total_cost_usd * 100) if total_cost_usd > 0 else 0.0
        row["prompt_cache_hit_ratio"] = cache_hit / cache_total if cache_total > 0 else 0.0
        enriched.append(row)
    return enriched


def _recent_usage_fields(recent: dict[str, Any], window_seconds: int) -> dict[str, Any]:
    window_seconds = max(1, _as_int(window_seconds, 300))
    cost = _as_float(recent.get("recent_total_cost_usd"))
    tokens = _as_int(recent.get("recent_total_tokens"))
    calls = _as_int(recent.get("recent_calls"))
    return {
        "recent_window_seconds": window_seconds,
        "recent_calls": calls,
        "recent_total_tokens": tokens,
        "recent_prompt_tokens": _as_int(recent.get("recent_prompt_tokens")),
        "recent_completion_tokens": _as_int(recent.get("recent_completion_tokens")),
        "recent_prompt_cache_hit_tokens": _as_int(recent.get("recent_prompt_cache_hit_tokens")),
        "recent_prompt_cache_miss_tokens": _as_int(recent.get("recent_prompt_cache_miss_tokens")),
        "recent_total_cost_usd": cost,
        "recent_tokens_per_minute": tokens / (window_seconds / 60),
        "recent_cost_per_minute_usd": cost / (window_seconds / 60),
        "recent_first_seen": recent.get("recent_first_seen"),
        "recent_last_seen": recent.get("recent_last_seen"),
    }


def _projection_fields(row: dict[str, Any]) -> dict[str, Any]:
    total_cost = _as_float(row.get("total_cost_usd"))
    total_tokens = _as_int(row.get("total_tokens"))
    progress_percent = _as_float(row.get("progress_percent"))
    total_chunks = _as_int(row.get("total_chunks"))
    processed_chunks = _as_int(row.get("completed_chunks")) + _as_int(row.get("failed_chunks"))
    if processed_chunks <= 0:
        current_index = _as_int(row.get("current_chunk_index"), -1)
        processed_chunks = max(0, current_index + 1)

    cost_per_chunk = (total_cost / processed_chunks) if processed_chunks > 0 and total_cost > 0 else 0.0
    tokens_per_chunk = (total_tokens / processed_chunks) if processed_chunks > 0 and total_tokens > 0 else 0.0
    cost_per_1k = (total_cost / total_tokens * 1000) if total_tokens > 0 and total_cost > 0 else 0.0

    projected_total_cost = 0.0
    projected_total_tokens = 0.0
    projection_basis = ""
    if progress_percent >= 1 and total_cost > 0:
        projected_total_cost = total_cost / (progress_percent / 100)
        projected_total_tokens = total_tokens / (progress_percent / 100) if total_tokens > 0 else 0.0
        projection_basis = "progress_percent"
    elif total_chunks > 0 and processed_chunks > 0 and total_cost > 0:
        ratio = processed_chunks / total_chunks
        projected_total_cost = total_cost / ratio
        projected_total_tokens = total_tokens / ratio if total_tokens > 0 else 0.0
        projection_basis = "chunk_ratio"

    return {
        "processed_chunks_for_cost": processed_chunks,
        "cost_per_processed_chunk_usd": cost_per_chunk,
        "tokens_per_processed_chunk": tokens_per_chunk,
        "cost_per_1k_tokens_usd": cost_per_1k,
        "projected_total_cost_usd": projected_total_cost,
        "projected_remaining_cost_usd": max(0.0, projected_total_cost - total_cost) if projected_total_cost else 0.0,
        "projected_total_tokens": projected_total_tokens,
        "projected_remaining_tokens": max(0.0, projected_total_tokens - total_tokens) if projected_total_tokens else 0.0,
        "projection_basis": projection_basis,
    }


def _backfill_from_checkpoints(jobs_db: Path, translation_id: str | None = None) -> dict:
    if not jobs_db.exists():
        return {"created": 0, "skipped": 0, "error": f"jobs.db not found: {jobs_db}"}
    store = default_usage_store()
    created = 0
    skipped = 0
    with sqlite3.connect(jobs_db) as conn:
        conn.row_factory = sqlite3.Row
        if translation_id:
            jobs = conn.execute(
                "SELECT translation_id, config FROM translation_jobs WHERE translation_id=?",
                (translation_id,),
            ).fetchall()
        else:
            jobs = conn.execute(
                "SELECT translation_id, config FROM translation_jobs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()

        for job in jobs:
            tid = job["translation_id"]
            if store.events(limit=1, translation_id=tid):
                skipped += 1
                continue
            config = json.loads(job["config"] or "{}")
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS chunks,
                    COALESCE(SUM(LENGTH(original_text)), 0) AS original_chars,
                    COALESCE(SUM(LENGTH(translated_text)), 0) AS translated_chars
                FROM checkpoint_chunks
                WHERE translation_id=? AND status='completed'
                """,
                (tid,),
            ).fetchone()
            chunks = int(row["chunks"] or 0)
            if chunks <= 0:
                skipped += 1
                continue
            original_chars = int(row["original_chars"] or 0)
            translated_chars = int(row["translated_chars"] or 0)
            prompt_tokens = _estimate_tokens("x" * original_chars) + chunks * 650
            completion_tokens = _estimate_tokens("x" * translated_chars)
            store.record_call(
                provider=config.get("llm_provider") or "",
                model=config.get("model") or "",
                prompt="historical checkpoint backfill",
                response_content="",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status="estimated_checkpoint_backfill",
                context={
                    "translation_id": tid,
                    "process_id": tid,
                    "process_type": "historical_checkpoint_estimate",
                    "phase": "checkpoint_backfill",
                    "book_name": config.get("original_filename") or config.get("input_filename") or config.get("output_filename"),
                    "input_filename": config.get("input_filename") or config.get("original_filename") or "",
                    "output_filename": config.get("output_filename") or "",
                },
                metadata={
                    "estimated_from": "checkpoint_chunks",
                    "chunks": chunks,
                    "original_chars": original_chars,
                    "translated_chars": translated_chars,
                },
            )
            created += 1
    return {"created": created, "skipped": skipped}
