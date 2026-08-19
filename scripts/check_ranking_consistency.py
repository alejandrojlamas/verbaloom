"""
Check ranking consistency across all benchmark submissions.

Detects (text_id, target_lang) triples where the spread of `overall` scores
between adjacent ranks violates rubric §5 (min 0.3 between adjacent ranks).

Pure diagnostic — modifies nothing. Output is Markdown on stdout, ready to
read in Claude Code.

Usage:
    python scripts/check_ranking_consistency.py
    python scripts/check_ranking_consistency.py --threshold 0.3 --min-observations 2
    python scripts/check_ranking_consistency.py --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
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


THRESHOLD_DEFAULT = 0.3
RUBRIC_VERSION_RE = re.compile(r"-rubric-(v\d+)$")


def _rubric_version(judge_id: str) -> str:
    m = RUBRIC_VERSION_RE.search(judge_id or "")
    return m.group(1) if m else "unknown"


def _bucket_min_gap(gap: float) -> str:
    if gap < 0.1:
        return "<0.1"
    if gap < 0.2:
        return "0.1-0.2"
    if gap < 0.3:
        return "0.2-0.3"
    return ">=0.3"


def _collect_observations(submissions):
    """
    Build:
      triples[(text_id, target_lang)][model] = {
          "overalls": [float, ...],
          "judge_ids": set[str],
          "rubric_versions": set[str],
      }
    """
    triples: dict[tuple[str, str], dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"overalls": [], "judge_ids": set(), "rubric_versions": set()}
    ))

    for sub in submissions:
        rv = _rubric_version(sub.judge_id)
        for r in sub.results:
            key = (r.text_id, r.target_lang)
            entry = triples[key][sub.model_id]
            entry["overalls"].append(float(r.scores.overall))
            entry["judge_ids"].add(sub.judge_id)
            entry["rubric_versions"].add(rv)

    return triples


def _per_judge_spread(submissions) -> dict[str, dict]:
    by_judge: dict[str, list[float]] = defaultdict(list)
    for sub in submissions:
        for r in sub.results:
            by_judge[sub.judge_id].append(float(r.scores.overall))

    out: dict[str, dict] = {}
    for jid, scores in by_judge.items():
        if len(scores) >= 2:
            stddev = statistics.stdev(scores)
        else:
            stddev = 0.0
        out[jid] = {
            "n_scores": len(scores),
            "stddev": stddev,
            "mean": statistics.mean(scores) if scores else 0.0,
        }
    return out


def analyze(submissions, threshold: float, min_observations: int) -> dict:
    triples = _collect_observations(submissions)

    flagged: list[dict] = []
    compliant_count = 0
    skipped_mixed_rubric = 0
    skipped_too_few_models = 0
    gap_distribution: dict[str, int] = {"<0.1": 0, "0.1-0.2": 0, "0.2-0.3": 0, ">=0.3": 0}
    per_model_flagged: dict[str, int] = defaultdict(int)
    per_model_violating: dict[str, int] = defaultdict(int)

    for (text_id, target_lang), models_obs in triples.items():
        all_versions: set[str] = set()
        for m_data in models_obs.values():
            all_versions |= m_data["rubric_versions"]
        if len(all_versions) > 1:
            skipped_mixed_rubric += 1
            continue

        if len(models_obs) < min_observations:
            skipped_too_few_models += 1
            continue

        ranking = []
        for model, m_data in models_obs.items():
            model_overall = statistics.median(m_data["overalls"])
            ranking.append({
                "model": model,
                "overall": model_overall,
                "judge_ids": sorted(m_data["judge_ids"]),
                "n_obs": len(m_data["overalls"]),
            })

        ranking.sort(key=lambda x: x["overall"], reverse=True)

        gaps = [
            ranking[i]["overall"] - ranking[i + 1]["overall"]
            for i in range(len(ranking) - 1)
        ]
        min_gap = min(gaps) if gaps else float("inf")
        gap_distribution[_bucket_min_gap(min_gap)] += 1

        if min_gap < threshold:
            min_gap_idx = gaps.index(min_gap)
            violating_pair = (
                ranking[min_gap_idx]["model"],
                ranking[min_gap_idx + 1]["model"],
            )
            judges_involved = sorted(set().union(*[set(r["judge_ids"]) for r in ranking]))
            flagged.append({
                "text_id": text_id,
                "target_lang": target_lang,
                "n_models": len(ranking),
                "min_gap": round(min_gap, 3),
                "violating_pair": list(violating_pair),
                "judges_involved": judges_involved,
                "ranking": ranking,
            })
            for m in ranking:
                per_model_flagged[m["model"]] += 1
            for m in violating_pair:
                per_model_violating[m] += 1
        else:
            compliant_count += 1

    flagged.sort(key=lambda x: x["min_gap"])

    per_model = {}
    all_models = set(per_model_flagged) | set(per_model_violating)
    for m in all_models:
        per_model[m] = {
            "flagged": per_model_flagged[m],
            "in_violating_pair": per_model_violating[m],
        }

    return {
        "threshold": threshold,
        "n_triples_total": len(triples),
        "n_triples_compared": len(flagged) + compliant_count,
        "n_flagged": len(flagged),
        "n_compliant": compliant_count,
        "n_skipped_mixed_rubric": skipped_mixed_rubric,
        "n_skipped_too_few_models": skipped_too_few_models,
        "gap_distribution": gap_distribution,
        "flagged_triples": flagged,
        "per_model": per_model,
        "per_judge": _per_judge_spread(submissions),
    }


def render_markdown(report: dict, n_submissions: int) -> str:
    lines: list[str] = []
    lines.append("# Ranking consistency report")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    lines.append(f"Threshold: {report['threshold']} (rubric §5)")
    lines.append(
        f"Submissions: {n_submissions} | "
        f"Triples compared: {report['n_triples_compared']} | "
        f"Flagged: {report['n_flagged']}"
    )
    if report["n_skipped_mixed_rubric"] or report["n_skipped_too_few_models"]:
        lines.append("")
        lines.append(
            f"Skipped: {report['n_skipped_mixed_rubric']} mixed-rubric, "
            f"{report['n_skipped_too_few_models']} too-few-models"
        )
    lines.append("")

    if report["n_triples_compared"] == 0:
        lines.append("No comparable triples (need ≥2 distinct models on the same triple).")
        return "\n".join(lines)

    lines.append("## Summary")
    lines.append("")
    lines.append("min_gap distribution:")
    lines.append("")
    lines.append("| Bucket | Count |")
    lines.append("|---|---|")
    for bucket in ("<0.1", "0.1-0.2", "0.2-0.3", ">=0.3"):
        lines.append(f"| {bucket} | {report['gap_distribution'][bucket]} |")
    lines.append("")

    if not report["flagged_triples"]:
        lines.append("All compared triples respect the dispersion rule.")
        lines.append("")
    else:
        lines.append("## Top violations")
        lines.append("")
        lines.append("| # | text_id | target_lang | min_gap | violating pair | judges |")
        lines.append("|---|---|---|---|---|---|")
        for i, t in enumerate(report["flagged_triples"][:10], start=1):
            judges_short = ", ".join(t["judges_involved"]) if len(t["judges_involved"]) <= 2 else f"{len(t['judges_involved'])} judges"
            pair = f"{t['violating_pair'][0]} vs {t['violating_pair'][1]}"
            lines.append(f"| {i} | {t['text_id']} | {t['target_lang']} | {t['min_gap']} | {pair} | {judges_short} |")
        lines.append("")

        lines.append("## Per-model involvement")
        lines.append("")
        lines.append("| Model | Flagged | In violating pair |")
        lines.append("|---|---|---|")
        for model, counts in sorted(report["per_model"].items(), key=lambda x: -x[1]["flagged"]):
            lines.append(f"| {model} | {counts['flagged']} | {counts['in_violating_pair']} |")
        lines.append("")

    lines.append("## Per-judge spread")
    lines.append("")
    lines.append("| judge_id | n_scores | overall mean | overall stddev |")
    lines.append("|---|---|---|---|")
    for jid, stats in sorted(report["per_judge"].items()):
        lines.append(f"| {jid} | {stats['n_scores']} | {stats['mean']:.2f} | {stats['stddev']:.2f} |")
    lines.append("")

    if report["flagged_triples"]:
        lines.append("## Detail by flagged triple")
        lines.append("")
        for t in report["flagged_triples"]:
            lines.append(f"### {t['text_id']} → {t['target_lang']} (min_gap={t['min_gap']})")
            lines.append("")
            lines.append("| Rank | Model | Overall | Judges | n_obs |")
            lines.append("|---|---|---|---|---|")
            for rank, r in enumerate(t["ranking"], start=1):
                judges = ", ".join(r["judge_ids"])
                lines.append(f"| {rank} | {r['model']} | {r['overall']:.2f} | {judges} | {r['n_obs']} |")
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"Min adjacent overall gap (default: {THRESHOLD_DEFAULT}, per rubric §5)")
    parser.add_argument("--min-observations", type=int, default=2,
                        help="Min distinct models per triple to compare (default: 2)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Optional path to also dump the report as JSON")
    args = parser.parse_args()

    config = BenchmarkConfig()
    submissions_dir = config.paths.base_dir / "data" / "submissions"
    aggregator = SubmissionAggregator(submissions_dir)
    submissions = aggregator.load()

    if not submissions:
        print(f"No submissions found in {submissions_dir}.")
        return 0

    report = analyze(submissions, args.threshold, args.min_observations)

    print(render_markdown(report, n_submissions=len(submissions)))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nJSON report: {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
