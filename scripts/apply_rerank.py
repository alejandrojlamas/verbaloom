"""
Apply rerank decisions back to submission files in place.

Reads a JSON array of:
  [{"text_id": "...", "target_lang": "...",
    "ranking": [{"model": "X", "overall": 8.5}, ...]}]

For each (model, text_id, target_lang) tuple, finds the matching submission
file and patches `overall` in place. Other dimensions
(accuracy/fluency/style/feedback) are not touched. Appends `-reranked` to the
env-level `judge_id` of every touched submission (idempotent).

The original `overall` values are preserved in git history. Use
`git log -p benchmark/data/submissions/` to audit.

Usage:
    python scripts/apply_rerank.py rerank.json [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from benchmark.config import BenchmarkConfig  # noqa: E402


THRESHOLD = 0.3


def _load_rerank(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")
    return data


def _validate_dispersion(triples: list[dict]) -> list[str]:
    errors: list[str] = []
    for t in triples:
        if "ranking" not in t or not isinstance(t["ranking"], list) or not t["ranking"]:
            errors.append(f"{t.get('text_id','?')} → {t.get('target_lang','?')}: missing or empty 'ranking'")
            continue
        overalls = sorted(
            (float(r["overall"]) for r in t["ranking"]),
            reverse=True,
        )
        for i in range(len(overalls) - 1):
            gap = overalls[i] - overalls[i + 1]
            if gap < THRESHOLD - 1e-9:
                errors.append(
                    f"{t['text_id']} → {t['target_lang']}: adjacent gap {gap:.3f} < {THRESHOLD}"
                )
                break
    return errors


def _patch_submission(
    sub_path: Path,
    model_id: str,
    text_id: str,
    target_lang: str,
    new_overall: float,
    dry_run: bool,
) -> tuple[bool, float | None]:
    data = json.loads(sub_path.read_text(encoding="utf-8"))
    if data["model"]["id"] != model_id:
        return False, None

    old_overall: float | None = None
    patched = False
    for r in data["results"]:
        if r["text_id"] == text_id and r["target_lang"] == target_lang:
            old_overall = float(r["scores"]["overall"])
            if old_overall != new_overall:
                if not dry_run:
                    r["scores"]["overall"] = new_overall
                patched = True

    if patched and not dry_run:
        jid = data["environment"]["judge_id"]
        if not jid.endswith("-reranked"):
            data["environment"]["judge_id"] = jid + "-reranked"
        sub_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return patched, old_overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rerank_json", type=Path,
                        help="JSON file with rerank decisions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without writing")
    args = parser.parse_args()

    config = BenchmarkConfig()
    submissions_dir = config.paths.base_dir / "data" / "submissions"

    triples = _load_rerank(args.rerank_json)

    errors = _validate_dispersion(triples)
    if errors:
        print("Dispersion violations in rerank input (would not enforce §5):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    sub_paths = sorted(submissions_dir.glob("*.json"))
    if not sub_paths:
        print(f"No submission files in {submissions_dir}.", file=sys.stderr)
        return 1

    n_results_changed = 0
    n_models_not_found = 0
    touched_paths: set[Path] = set()

    for t in triples:
        for entry in t["ranking"]:
            model = entry["model"]
            new_overall = float(entry["overall"])
            found = False
            for sub_path in sub_paths:
                patched, old = _patch_submission(
                    sub_path, model, t["text_id"], t["target_lang"], new_overall, args.dry_run
                )
                if old is not None:
                    found = True
                    if patched:
                        prefix = "[DRY] " if args.dry_run else ""
                        print(f"{prefix}{sub_path.name}: {model} {t['text_id']}→{t['target_lang']} "
                              f"{old:.2f} → {new_overall:.2f}")
                        touched_paths.add(sub_path)
                        n_results_changed += 1
            if not found:
                print(f"WARN: {model} not found for {t['text_id']}→{t['target_lang']}",
                      file=sys.stderr)
                n_models_not_found += 1

    if args.dry_run:
        print(f"\n[DRY] Would change {n_results_changed} results in "
              f"{len(touched_paths)} submissions (models not found: {n_models_not_found})")
    else:
        print(f"\nChanged {n_results_changed} results in {len(touched_paths)} submissions "
              f"(models not found: {n_models_not_found})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
