"""State manager for Sample & Compare runs.

Holds one entry per ``sample_id`` with the columns, items, and per-cell
results streamed as LLM calls complete.  The manager can be purely in-memory
for tests, or persist JSON snapshots so mobile refreshes and app restarts do
not erase the comparison state.
"""
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional


# Sample entries older than this are pruned on every public access.
SAMPLE_TTL_SECONDS = 3600


class SampleStateManager:
    """Thread-safe registry for Sample & Compare runs."""

    def __init__(
        self,
        persist_path: Optional[str | os.PathLike[str]] = None,
        *,
        ttl_seconds: int = SAMPLE_TTL_SECONDS,
    ) -> None:
        self._samples: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._persist_path = Path(persist_path) if persist_path else None
        self._ttl_seconds = max(60, int(ttl_seconds or SAMPLE_TTL_SECONDS))
        self._load_persisted()

    def create(
        self,
        sample_id: str,
        items: List[Dict[str, Any]],
        columns: List[Dict[str, Any]],
        mode: str,
    ) -> None:
        """Register a fresh sample run with its initial item/column shape."""
        with self._lock:
            self._prune_expired_locked()
            n_rows = len(items)
            n_cols = len(columns)
            self._samples[sample_id] = {
                "sample_id": sample_id,
                "status": "running",
                "mode": mode,
                "items": items,
                "columns": columns,
                "cells": [
                    {
                        "row": r,
                        "col": c,
                        "phase": "translate",
                        "status": "pending",
                        "output": None,
                        "metrics": None,
                        "error": None,
                    }
                    for r in range(n_rows)
                    for c in range(n_cols)
                ],
                "cancelled": False,
                "created_at": time.time(),
            }
            self._persist_locked()

    def get(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Return a deep-ish snapshot of a sample entry, or None if unknown."""
        with self._lock:
            self._prune_expired_locked()
            entry = self._samples.get(sample_id)
            if entry is None:
                return None
            return {
                "sample_id": entry["sample_id"],
                "status": entry["status"],
                "mode": entry["mode"],
                "items": entry["items"],
                "columns": entry["columns"],
                "cells": [dict(cell) for cell in entry["cells"]],
                "run_context": dict(entry.get("run_context") or {}),
            }

    def list_summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight summaries for currently retained sample runs."""
        with self._lock:
            self._prune_expired_locked()
            now = time.time()
            summaries: List[Dict[str, Any]] = []
            for sample_id, entry in self._samples.items():
                cells = [dict(cell) for cell in entry.get("cells", [])]
                done = sum(1 for cell in cells if cell.get("status") == "done")
                error = sum(1 for cell in cells if cell.get("status") == "error")
                pending = sum(1 for cell in cells if cell.get("status") == "pending")
                summaries.append({
                    "sample_id": sample_id,
                    "status": entry.get("status"),
                    "mode": entry.get("mode"),
                    "created_at": entry.get("created_at"),
                    "age_seconds": round(now - float(entry.get("created_at") or now), 1),
                    "items": len(entry.get("items") or []),
                    "columns": len(entry.get("columns") or []),
                    "cells": len(cells),
                    "done": done,
                    "error": error,
                    "pending": pending,
                    "cancelled": bool(entry.get("cancelled")),
                })
            summaries.sort(key=lambda row: row.get("created_at") or 0, reverse=True)
            return summaries

    def exists(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._samples

    def cancel(self, sample_id: str) -> bool:
        """Flag a sample as cancelled. Returns False if the id is unknown."""
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return False
            entry["cancelled"] = True
            self._persist_locked()
            return True

    def is_cancelled(self, sample_id: str) -> bool:
        with self._lock:
            entry = self._samples.get(sample_id)
            return bool(entry and entry["cancelled"])

    def update_cell(
        self,
        sample_id: str,
        row: int,
        col: int,
        phase: str,
        *,
        status: str,
        output: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Upsert the cell record for (row, col, phase). The `cells` list is keyed
        by row/col only for `translate`; `refine` cells live as a separate
        record with phase=='refine' so the front-end can render both stacked.
        """
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return
            for cell in entry["cells"]:
                if cell["row"] == row and cell["col"] == col and cell["phase"] == phase:
                    cell["status"] = status
                    cell["output"] = output
                    cell["metrics"] = metrics
                    cell["error"] = error
                    self._persist_locked()
                    return
            entry["cells"].append({
                "row": row,
                "col": col,
                "phase": phase,
                "status": status,
                "output": output,
                "metrics": metrics,
                "error": error,
            })
            self._persist_locked()

    def set_status(self, sample_id: str, status: str) -> None:
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is not None:
                entry["status"] = status
                self._persist_locked()

    def set_run_context(self, sample_id: str, run_ctx: Dict[str, Any]) -> None:
        """Stash the parameters needed to launch the LLM thread later.

        Used by the deferred-dispatch flow: /api/sample/run prepares items and
        stores the context here; /api/sample/<id>/dispatch reads it back to
        spawn `_run_sample_async`.
        """
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is not None:
                entry["run_context"] = run_ctx
                self._persist_locked()

    def get_run_context(self, sample_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._samples.get(sample_id)
            if entry is None:
                return None
            return entry.get("run_context")

    def _prune_expired_locked(self) -> None:
        now = time.time()
        expired = [
            sid for sid, entry in self._samples.items()
            if now - entry["created_at"] > self._ttl_seconds
        ]
        for sid in expired:
            del self._samples[sid]
        if expired:
            self._persist_locked()

    def _load_persisted(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        now = time.time()
        samples = payload.get("samples")
        if not isinstance(samples, dict):
            return
        for sample_id, entry in samples.items():
            if not isinstance(entry, dict):
                continue
            created_at = float(entry.get("created_at") or now)
            if now - created_at > self._ttl_seconds:
                continue
            safe_entry = self._sanitize_entry(entry)
            safe_entry["sample_id"] = str(sample_id)
            safe_entry.setdefault("status", "prepared")
            safe_entry.setdefault("mode", "translate")
            safe_entry.setdefault("items", [])
            safe_entry.setdefault("columns", [])
            safe_entry.setdefault("cells", [])
            safe_entry.setdefault("cancelled", False)
            safe_entry.setdefault("created_at", created_at)
            self._samples[str(sample_id)] = safe_entry

    def _persist_locked(self) -> None:
        if self._persist_path is None:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "samples": {
                    sample_id: self._sanitize_entry(entry)
                    for sample_id, entry in self._samples.items()
                },
            }
            tmp_path = self._persist_path.with_suffix(self._persist_path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(self._persist_path)
        except Exception:
            # Sample persistence should never break the comparison flow.
            return

    def _sanitize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        clean = dict(entry or {})
        clean["columns"] = [
            _sanitize_column(column)
            for column in clean.get("columns", []) or []
            if isinstance(column, dict)
        ]
        run_context = clean.get("run_context")
        if isinstance(run_context, dict):
            clean_context = dict(run_context)
            clean_context["columns"] = [
                _sanitize_column(column)
                for column in clean_context.get("columns", []) or []
                if isinstance(column, dict)
            ]
            clean["run_context"] = clean_context
        return clean


def _sanitize_column(column: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(column)
    if "api_key" in clean:
        clean["api_key"] = None
    return clean
