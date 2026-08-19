"""Reusable exact-match translation memory.

This is deliberately narrower than a semantic cache.  It only reuses a
translation when the source unit, surrounding context, model/provider, language
pair, and prompt-relevant options match.  That keeps it useful for retries or
repeat runs of the same book without letting one book's editorial decisions
bleed into another.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from typing import Any


TRANSLATION_MEMORY_VERSION = "translation-memory-v1"

_VOLATILE_PROMPT_KEYS = {
    "_editorial_quality_report",
    "_fidelity_report",
    "_source_guard_refs",
    "_source_guard_refs_loaded",
}

_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
)

_HASH_ONLY_KEYS = {
    "custom_instructions",
    "refinement_instructions",
    "glossary_terms",
    "glossary_term_metadata",
    "profile_artifacts",
    "profile_prompt",
}


@dataclass(frozen=True)
class TranslationMemoryRequest:
    source_text: str
    source_language: str
    target_language: str
    provider: str
    model: str
    prompt_options: dict[str, Any]
    format_name: str = ""
    unit_id: str = ""
    context_before: str = ""
    context_after: str = ""
    previous_translation_context: str = ""
    phase: str = "translation"


@dataclass(frozen=True)
class TranslationMemoryEntry:
    key: str
    translated_text: str
    metadata: dict[str, Any]
    hits: int = 0


def normalize_memory_text(value: str) -> str:
    """Normalize text for exact cache identity without rewriting meaning."""
    return re.sub(r"\s+", " ", value or "").strip()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_fingerprint(prompt_options: dict[str, Any] | None) -> str:
    """Return a stable prompt fingerprint with secrets removed."""
    return stable_hash(_sanitize_prompt_options(prompt_options or {}))


def _sanitize_prompt_options(value: Any, key: str = "") -> Any:
    key_lower = key.lower()
    if key in _VOLATILE_PROMPT_KEYS:
        return None
    if any(fragment in key_lower for fragment in _SECRET_KEY_FRAGMENTS):
        return "[redacted]"
    if key in _HASH_ONLY_KEYS:
        return {"sha256": stable_hash(value)}
    if isinstance(value, dict):
        return {
            str(k): _sanitize_prompt_options(v, str(k))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k) not in _VOLATILE_PROMPT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_prompt_options(item, key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class TranslationMemoryCache:
    """Small SQLite-backed exact-match cache for translated units."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or os.getenv("TRANSLATION_MEMORY_DB", "data/translation_memory.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._initialize()

    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.connection = conn
        return conn

    def _initialize(self) -> None:
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translation_memory (
                    cache_key TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    context_hash TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    source_language TEXT NOT NULL,
                    target_language TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    format_name TEXT,
                    phase TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    metadata JSON,
                    hits INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_hit_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_translation_memory_lookup
                ON translation_memory (
                    source_hash, context_hash, prompt_hash,
                    source_language, target_language, provider, model, phase
                )
                """
            )
            conn.commit()

    @staticmethod
    def build_key(request: TranslationMemoryRequest) -> tuple[str, dict[str, str]]:
        source_hash = stable_hash(normalize_memory_text(request.source_text))
        context_hash = stable_hash({
            "before": normalize_memory_text(request.context_before),
            "after": normalize_memory_text(request.context_after),
            "previous": normalize_memory_text(request.previous_translation_context),
        })
        prompt_hash = prompt_fingerprint(request.prompt_options)
        identity = {
            "version": TRANSLATION_MEMORY_VERSION,
            "source_hash": source_hash,
            "context_hash": context_hash,
            "prompt_hash": prompt_hash,
            "source_language": request.source_language or "",
            "target_language": request.target_language or "",
            "provider": request.provider or "",
            "model": request.model or "",
            "format_name": request.format_name or "",
            "phase": request.phase or "translation",
        }
        return stable_hash(identity), {
            "source_hash": source_hash,
            "context_hash": context_hash,
            "prompt_hash": prompt_hash,
        }

    def get(self, request: TranslationMemoryRequest) -> TranslationMemoryEntry | None:
        cache_key, _hashes = self.build_key(request)
        with self._lock:
            conn = self._connection()
            row = conn.execute(
                "SELECT cache_key, translated_text, metadata, hits FROM translation_memory WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute(
                """
                UPDATE translation_memory
                SET hits = hits + 1, last_hit_at = ?, updated_at = ?
                WHERE cache_key = ?
                """,
                (now, now, cache_key),
            )
            conn.commit()
        try:
            metadata = json.loads(row["metadata"] or "{}")
        except Exception:
            metadata = {}
        return TranslationMemoryEntry(
            key=row["cache_key"],
            translated_text=row["translated_text"],
            metadata=metadata,
            hits=int(row["hits"] or 0) + 1,
        )

    def delete(self, request: TranslationMemoryRequest) -> bool:
        """Remove one exact cached translation entry."""
        cache_key, _hashes = self.build_key(request)
        with self._lock:
            conn = self._connection()
            cursor = conn.execute(
                "DELETE FROM translation_memory WHERE cache_key = ?",
                (cache_key,),
            )
            conn.commit()
            return bool(cursor.rowcount)

    def put(
        self,
        request: TranslationMemoryRequest,
        translated_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        if not translated_text or not translated_text.strip():
            return None
        cache_key, hashes = self.build_key(request)
        now = time.time()
        payload = {
            "unit_id": request.unit_id,
            "metadata": metadata or {},
        }
        with self._lock:
            conn = self._connection()
            conn.execute(
                """
                INSERT INTO translation_memory (
                    cache_key, version, source_hash, context_hash, prompt_hash,
                    source_language, target_language, provider, model,
                    format_name, phase, translated_text, metadata,
                    hits, created_at, updated_at, last_hit_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL)
                ON CONFLICT(cache_key) DO UPDATE SET
                    translated_text = excluded.translated_text,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    TRANSLATION_MEMORY_VERSION,
                    hashes["source_hash"],
                    hashes["context_hash"],
                    hashes["prompt_hash"],
                    request.source_language or "",
                    request.target_language or "",
                    request.provider or "",
                    request.model or "",
                    request.format_name or "",
                    request.phase or "translation",
                    translated_text,
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
            conn.commit()
        return cache_key
