"""SQLite-backed token/cost ledger for all provider calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from src.core.pricing import get_default_pricing
from src.utils.branding import default_data_dir, env_value


def _default_data_dir() -> Path:
    return Path(str(env_value("DATA_DIR", str(default_data_dir()))))


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Conservative multilingual approximation. Exact accounting comes from the
    # provider when available.
    return max(1, int(len(text) / 3.7))


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[:20]


def _pricing_for(provider: str, model: str) -> tuple[Optional[dict[str, float]], str]:
    provider = (provider or "").lower()
    if provider == "ollama":
        return {"input": 0.0, "output": 0.0}, "local"
    pricing = get_default_pricing(provider, model or "")
    if pricing:
        return pricing, "default_table"
    return None, "unknown"


def _cost_for(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cache_hit_tokens: int = 0,
    prompt_cache_miss_tokens: int = 0,
) -> tuple[float, float, float, str]:
    pricing, source = _pricing_for(provider, model)
    if not pricing:
        return 0.0, 0.0, 0.0, source
    input_rate = float(pricing.get("input_cache_miss", pricing.get("input", 0.0)) or 0.0)
    cache_hit_rate = float(pricing.get("input_cache_hit", input_rate) or 0.0)
    output_rate = float(pricing.get("output", 0.0) or 0.0)
    cache_hit = max(0, int(prompt_cache_hit_tokens or 0))
    cache_miss = max(0, int(prompt_cache_miss_tokens or 0))
    if cache_hit or cache_miss:
        unreported_input = max(0, int(prompt_tokens or 0) - cache_hit - cache_miss)
        input_cost = (cache_hit * cache_hit_rate + (cache_miss + unreported_input) * input_rate) / 1_000_000
    else:
        input_cost = prompt_tokens * input_rate / 1_000_000
    output_cost = completion_tokens * output_rate / 1_000_000
    return input_cost, output_cost, input_cost + output_cost, source


class TokenUsageStore:
    """Append-only usage ledger.

    The ledger stores only metadata, sizes, token counts, hashes and cost
    estimates. It does not persist full prompts or book text.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or (_default_data_dir() / "usage.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL NOT NULL,
                    translation_id TEXT,
                    process_id TEXT,
                    process_type TEXT,
                    phase TEXT,
                    provider TEXT,
                    model TEXT,
                    book_name TEXT,
                    input_filename TEXT,
                    output_filename TEXT,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    prompt_cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                    prompt_cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
                    prompt_chars INTEGER NOT NULL DEFAULT 0,
                    completion_chars INTEGER NOT NULL DEFAULT 0,
                    input_cost_usd REAL NOT NULL DEFAULT 0,
                    output_cost_usd REAL NOT NULL DEFAULT 0,
                    total_cost_usd REAL NOT NULL DEFAULT 0,
                    pricing_source TEXT,
                    estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'ok',
                    prompt_hash TEXT,
                    response_hash TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_created_at ON token_usage_events(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_translation ON token_usage_events(translation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_book ON token_usage_events(book_name)")
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(token_usage_events)").fetchall()
        }
        for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            if name not in existing:
                conn.execute(
                    f"ALTER TABLE token_usage_events ADD COLUMN {name} INTEGER NOT NULL DEFAULT 0"
                )

    def record_call(
        self,
        *,
        provider: str,
        model: str,
        prompt: str,
        system_prompt: str = "",
        response_content: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
        status: str = "ok",
        context: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        context = context or {}
        metadata = metadata or {}
        prompt_text = f"{system_prompt or ''}\n{prompt or ''}".strip()
        estimated = status.startswith("estimated") or bool((metadata or {}).get("estimated_from"))
        if not prompt_tokens:
            prompt_tokens = _estimate_tokens(prompt_text)
            estimated = True
        if not completion_tokens and response_content:
            completion_tokens = _estimate_tokens(response_content)
            estimated = True
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        prompt_cache_hit_tokens = int(prompt_cache_hit_tokens or 0)
        prompt_cache_miss_tokens = int(prompt_cache_miss_tokens or 0)
        total_tokens = prompt_tokens + completion_tokens
        input_cost, output_cost, total_cost, pricing_source = _cost_for(
            provider,
            model,
            prompt_tokens,
            completion_tokens,
            prompt_cache_hit_tokens=prompt_cache_hit_tokens,
            prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        )

        payload = {
            "created_at": time.time(),
            "translation_id": context.get("translation_id"),
            "process_id": context.get("process_id") or context.get("translation_id"),
            "process_type": context.get("process_type") or "unknown",
            "phase": context.get("phase") or "llm_call",
            "provider": (provider or "").lower(),
            "model": model or "",
            "book_name": context.get("book_name") or context.get("source_name") or "",
            "input_filename": context.get("input_filename") or "",
            "output_filename": context.get("output_filename") or "",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_cache_hit_tokens": prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": prompt_cache_miss_tokens,
            "prompt_chars": len(prompt_text),
            "completion_chars": len(response_content or ""),
            "input_cost_usd": input_cost,
            "output_cost_usd": output_cost,
            "total_cost_usd": total_cost,
            "pricing_source": pricing_source,
            "estimated_tokens": 1 if estimated else 0,
            "status": status,
            "prompt_hash": _hash_text(prompt_text),
            "response_hash": _hash_text(response_content or ""),
            "metadata": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        }
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO token_usage_events (
                    created_at, translation_id, process_id, process_type, phase,
                    provider, model, book_name, input_filename, output_filename,
                    prompt_tokens, completion_tokens, total_tokens,
                    prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                    prompt_chars, completion_chars,
                    input_cost_usd, output_cost_usd, total_cost_usd,
                    pricing_source, estimated_tokens, status,
                    prompt_hash, response_hash, metadata
                ) VALUES (
                    :created_at, :translation_id, :process_id, :process_type, :phase,
                    :provider, :model, :book_name, :input_filename, :output_filename,
                    :prompt_tokens, :completion_tokens, :total_tokens,
                    :prompt_cache_hit_tokens, :prompt_cache_miss_tokens,
                    :prompt_chars, :completion_chars,
                    :input_cost_usd, :output_cost_usd, :total_cost_usd,
                    :pricing_source, :estimated_tokens, :status,
                    :prompt_hash, :response_hash, :metadata
                )
                """,
                payload,
            )
            return int(cursor.lastrowid)

    def summary(self, limit: int = 80) -> dict[str, Any]:
        recent_window_seconds = 300
        recent_cutoff = time.time() - recent_window_seconds
        with self._lock, self._connect() as conn:
            totals = self._one(conn, """
                SELECT
                    COUNT(*) AS calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
                    COALESCE(SUM(input_cost_usd), 0) AS input_cost_usd,
                    COALESCE(SUM(output_cost_usd), 0) AS output_cost_usd,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                    COALESCE(SUM(estimated_tokens), 0) AS estimated_events
                FROM token_usage_events
            """)
            by_book = self._all(conn, """
                SELECT
                    COALESCE(NULLIF(book_name, ''), NULLIF(input_filename, ''), NULLIF(output_filename, ''), translation_id, process_id, 'Sin libro') AS book_name,
                    translation_id,
                    process_type,
                    COUNT(*) AS calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                    MAX(created_at) AS last_seen,
                    SUM(CASE WHEN estimated_tokens THEN 1 ELSE 0 END) AS estimated_events
                FROM token_usage_events
                GROUP BY COALESCE(NULLIF(book_name, ''), NULLIF(input_filename, ''), NULLIF(output_filename, ''), translation_id, process_id, 'Sin libro'), translation_id, process_type
                ORDER BY last_seen DESC
                LIMIT ?
            """, (limit,))
            by_model = self._all(conn, """
	                SELECT provider, model, COUNT(*) AS calls,
	                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
	                       COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
	                       COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
	                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM token_usage_events
                GROUP BY provider, model
                ORDER BY total_cost_usd DESC, total_tokens DESC
            """)
            by_translation = self._all(conn, """
                SELECT
                    translation_id,
                    COALESCE(NULLIF(book_name, ''), NULLIF(input_filename, ''), NULLIF(output_filename, ''), translation_id, 'Sin libro') AS book_name,
                    COALESCE(NULLIF(input_filename, ''), '') AS input_filename,
                    COALESCE(NULLIF(output_filename, ''), '') AS output_filename,
                    COALESCE(NULLIF(process_type, ''), 'unknown') AS process_type,
                    COUNT(*) AS calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
                    COALESCE(SUM(input_cost_usd), 0) AS input_cost_usd,
                    COALESCE(SUM(output_cost_usd), 0) AS output_cost_usd,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                    MIN(created_at) AS first_seen,
                    MAX(created_at) AS last_seen,
                    SUM(CASE WHEN estimated_tokens THEN 1 ELSE 0 END) AS estimated_events
                FROM token_usage_events
                WHERE translation_id IS NOT NULL AND translation_id != ''
                GROUP BY translation_id
                ORDER BY last_seen DESC
                LIMIT ?
            """, (limit,))
            by_process = self._all(conn, """
                SELECT process_type, COUNT(*) AS calls,
	                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
	                       COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
	                       COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
	                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM token_usage_events
                GROUP BY process_type
                ORDER BY total_cost_usd DESC, total_tokens DESC
            """)
            by_phase = self._all(conn, """
                SELECT
                    COALESCE(NULLIF(phase, ''), NULLIF(process_type, ''), 'unknown') AS phase,
                    COALESCE(NULLIF(process_type, ''), 'unknown') AS process_type,
                    COUNT(*) AS calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                    SUM(CASE WHEN estimated_tokens THEN 1 ELSE 0 END) AS estimated_events
                FROM token_usage_events
                GROUP BY COALESCE(NULLIF(phase, ''), NULLIF(process_type, ''), 'unknown'), COALESCE(NULLIF(process_type, ''), 'unknown')
                ORDER BY total_cost_usd DESC, total_tokens DESC
            """)
            phase_by_translation = self._all(conn, """
                SELECT
                    translation_id,
                    COALESCE(NULLIF(phase, ''), NULLIF(process_type, ''), 'unknown') AS phase,
                    COALESCE(NULLIF(process_type, ''), 'unknown') AS process_type,
                    COUNT(*) AS calls,
                    COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
                    COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd,
                    SUM(CASE WHEN estimated_tokens THEN 1 ELSE 0 END) AS estimated_events,
                    MAX(created_at) AS last_seen
                FROM token_usage_events
                WHERE translation_id IS NOT NULL AND translation_id != ''
                GROUP BY translation_id, COALESCE(NULLIF(phase, ''), NULLIF(process_type, ''), 'unknown'), COALESCE(NULLIF(process_type, ''), 'unknown')
                ORDER BY translation_id, total_cost_usd DESC, total_tokens DESC
            """)
            recent_by_translation = self._all(conn, """
                SELECT
                    translation_id,
                    COUNT(*) AS recent_calls,
                    COALESCE(SUM(prompt_tokens), 0) AS recent_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS recent_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS recent_total_tokens,
                    COALESCE(SUM(prompt_cache_hit_tokens), 0) AS recent_prompt_cache_hit_tokens,
                    COALESCE(SUM(prompt_cache_miss_tokens), 0) AS recent_prompt_cache_miss_tokens,
                    COALESCE(SUM(total_cost_usd), 0) AS recent_total_cost_usd,
                    MIN(created_at) AS recent_first_seen,
                    MAX(created_at) AS recent_last_seen
                FROM token_usage_events
                WHERE translation_id IS NOT NULL
                  AND translation_id != ''
                  AND created_at >= ?
                GROUP BY translation_id
                ORDER BY recent_last_seen DESC
                LIMIT ?
            """, (recent_cutoff, limit,))
            daily = self._all(conn, """
                SELECT date(created_at, 'unixepoch', 'localtime') AS day,
	                       COUNT(*) AS calls,
	                       COALESCE(SUM(total_tokens), 0) AS total_tokens,
	                       COALESCE(SUM(prompt_cache_hit_tokens), 0) AS prompt_cache_hit_tokens,
	                       COALESCE(SUM(prompt_cache_miss_tokens), 0) AS prompt_cache_miss_tokens,
	                       COALESCE(SUM(total_cost_usd), 0) AS total_cost_usd
                FROM token_usage_events
                GROUP BY day
                ORDER BY day ASC
            """)
            recent = self._all(conn, """
                SELECT *
                FROM token_usage_events
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
        return {
            "totals": totals,
            "by_book": by_book,
            "by_translation": by_translation,
            "by_model": by_model,
            "by_process": by_process,
            "by_phase": by_phase,
            "phase_by_translation": phase_by_translation,
            "recent_by_translation": recent_by_translation,
            "recent_window_seconds": recent_window_seconds,
            "daily": daily,
            "recent_events": recent,
            "db_path": str(self.db_path),
        }

    def events(self, limit: int = 200, translation_id: Optional[str] = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit or 200), 1000))
        with self._lock, self._connect() as conn:
            if translation_id:
                return self._all(conn, """
                    SELECT * FROM token_usage_events
                    WHERE translation_id=?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (translation_id, limit))
            return self._all(conn, """
                SELECT * FROM token_usage_events
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

    @staticmethod
    def _one(conn: sqlite3.Connection, query: str, args: tuple = ()) -> dict[str, Any]:
        row = conn.execute(query, args).fetchone()
        return dict(row) if row else {}

    @staticmethod
    def _all(conn: sqlite3.Connection, query: str, args: tuple = ()) -> list[dict[str, Any]]:
        rows = conn.execute(query, args).fetchall()
        return [dict(row) for row in rows]


_DEFAULT_STORE: Optional[TokenUsageStore] = None
_DEFAULT_LOCK = threading.Lock()


def default_usage_store() -> TokenUsageStore:
    global _DEFAULT_STORE
    with _DEFAULT_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = TokenUsageStore()
        return _DEFAULT_STORE
