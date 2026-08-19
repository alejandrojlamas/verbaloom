"""
Identify (text_id, target_lang) triples in benchmark submissions where the
spread of `overall` scores between adjacent ranks violates rubric §5
(min 0.3 between adjacent ranks).

Produces a Markdown brief listing each affected triple with all model outputs
side-by-side, ready for Claude (or any judge) to score new `overall` values
that enforce dispersion. The other dimensions (accuracy/fluency/style) are
not touched — those are absolute observations.

Usage:
    python scripts/dump_for_rerank.py [--touching MODEL_ID] [--threshold 0.3] [--out brief.md]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
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

from benchmark.aggregator import SubmissionAggregator  # noqa: E402
from benchmark.config import BenchmarkConfig  # noqa: E402
from benchmark.data_loader import load_reference_texts  # noqa: E402


THRESHOLD_DEFAULT = 0.3


def find_triples_needing_rerank(submissions, threshold: float, touching: str | None):
    triples: dict = defaultdict(lambda: defaultdict(list))
    for sub in submissions:
        for r in sub.results:
            triples[(r.text_id, r.target_lang)][sub.model_id].append({
                "overall": float(r.scores.overall),
                "output": r.output,
            })

    affected: list[dict] = []
    for (text_id, target_lang), models_data in triples.items():
        if touching and touching not in models_data:
            continue
        if len(models_data) < 2:
            continue

        ranking = []
        for model, obs in models_data.items():
            ranking.append({
                "model": model,
                "overall": statistics.median(o["overall"] for o in obs),
                "output": obs[0]["output"],
            })
        ranking.sort(key=lambda x: x["overall"], reverse=True)

        gaps = [
            ranking[i]["overall"] - ranking[i + 1]["overall"]
            for i in range(len(ranking) - 1)
        ]
        if min(gaps, default=float("inf")) < threshold:
            affected.append({
                "text_id": text_id,
                "target_lang": target_lang,
                "ranking": ranking,
            })

    affected.sort(key=lambda x: (x["text_id"], x["target_lang"]))
    return affected


def build_brief(triples, ref_texts, threshold: float) -> str:
    lines: list[str] = []
    lines.append("# Rerank brief")
    lines.append("")
    lines.append(f"Triples needing rerank: {len(triples)} (threshold {threshold} per rubric §5)")
    lines.append("")
    lines.append("**Rule:** Reorder the models for each triple and assign new `overall` "
                 "scores. Adjacent overalls (after sorting) must differ by ≥0.3. Don't "
                 "touch the other dimensions — only `overall` is reranked here.")
    lines.append("")
    lines.append("---")
    lines.append("")

    for t in triples:
        ref = ref_texts.get(t["text_id"])
        lines.append(f"## {t['text_id']} → {t['target_lang']}")
        lines.append("")
        if ref is not None:
            lines.append(f"- **Source**: *{ref.title}* ({ref.author}, {ref.year})")
            lines.append(f"- **Source language**: `{ref.source_language}`")
            if ref.challenges:
                lines.append(f"- **Challenges**: {'; '.join(ref.challenges)}")
        lines.append("")
        lines.append("### Source text")
        lines.append("")
        lines.append("```")
        lines.append((ref.content if ref else "<missing reference>").strip())
        lines.append("```")
        lines.append("")
        lines.append("### Outputs (sorted by current overall)")
        lines.append("")
        for r in t["ranking"]:
            lines.append(f"#### `{r['model']}` — current overall: {r['overall']:.2f}")
            lines.append("")
            lines.append("```")
            lines.append(r["output"].strip())
            lines.append("```")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## Reply format")
    lines.append("")
    lines.append("Single JSON array, one object per triple. Inside `ranking`, list each "
                 "model with the *new* `overall` you assign:")
    lines.append("")
    lines.append("```json")
    lines.append("[")
    for i, t in enumerate(triples):
        lines.append(f'  {{"text_id": "{t["text_id"]}", "target_lang": "{t["target_lang"]}", "ranking": [')
        for j, r in enumerate(t["ranking"]):
            comma = "," if j < len(t["ranking"]) - 1 else ""
            lines.append(f'    {{"model": "{r["model"]}", "overall": 0.0}}{comma}')
        comma = "," if i < len(triples) - 1 else ""
        lines.append(f"  ]}}{comma}")
    lines.append("]")
    lines.append("```")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--touching", help="Only triples involving this model_id")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"Min adjacent overall gap (default: {THRESHOLD_DEFAULT})")
    parser.add_argument("--out", type=Path,
                        help="Write brief to this path instead of stdout")
    args = parser.parse_args()

    config = BenchmarkConfig()
    submissions_dir = config.paths.base_dir / "data" / "submissions"
    aggregator = SubmissionAggregator(submissions_dir)
    submissions = aggregator.load()

    if not submissions:
        print(f"No submissions found in {submissions_dir}.")
        return 0

    ref_texts = load_reference_texts(
        base_dir=config.paths.base_dir,
        legacy_file=config.paths.reference_texts_file,
    )

    triples = find_triples_needing_rerank(submissions, args.threshold, args.touching)

    if not triples:
        print("No triples need rerank.")
        return 0

    brief = build_brief(triples, ref_texts, args.threshold)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(brief, encoding="utf-8")
        print(f"Wrote {args.out} ({len(brief)} chars, {len(triples)} triples)")
    else:
        print(brief)

    return 0


if __name__ == "__main__":
    sys.exit(main())
