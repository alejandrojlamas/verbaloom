"""Prepare private editorial comparisons; execution requires explicit paid opt-in.

Usage: python -m scripts.editorial_benchmark --help
"""

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path

from src.core.editorial_benchmark import (
    prepare_corpus,
    render_blind_sample,
    validate_corpus,
)

REPO = Path(__file__).resolve().parents[1]


def private_directory(path):
    path = Path(path).expanduser().resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("Private excerpts must be stored outside the repository")
    if path.exists():
        if not path.is_dir() or path.stat().st_mode & 0o077:
            raise ValueError(
                "Existing experiment directory must be private (mode 0700)"
            )
    else:
        path.mkdir(parents=True, mode=0o700)
    return path


def write_private(path, payload):
    """Atomically publish a new artifact, never overwrite an earlier experiment."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Artifact already exists; use a new experiment directory")
    descriptor, temp_name = tempfile.mkstemp(prefix=".benchmark-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                payload
                if isinstance(payload, str)
                else json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        # An atomic exclusive link avoids replacing another writer's artifact.
        os.link(temp_name, path)
    finally:
        os.unlink(temp_name)


def source_units(path):
    if path.suffix.casefold() == ".txt":
        from src.utils.text_reader import read_text_file_with_fallbacks

        text, _encoding = read_text_file_with_fallbacks(path)
        paragraphs = [
            match.group().strip()
            for match in re.finditer(
                r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)",
                text,
            )
        ]
        return [
            {
                "locator": f"paragraph:{index}",
                "text": text,
                "section": "plain-text-document",
                "position": index,
            }
            for index, text in enumerate(paragraphs)
        ], []
    if path.suffix.casefold() != ".epub":
        raise ValueError("This source selector accepts TXT and EPUB")
    from src.core.epub.publication_gate import snapshot_epub

    snapshot = snapshot_epub(path)
    if snapshot.parse_errors or snapshot.unsafe_paths:
        raise ValueError("Source EPUB needs structural review before corpus extraction")
    return [
        {
            "locator": f"{unit.file_href}:{unit.dom_path}",
            "text": unit.text,
            "section": unit.file_href,
            "position": index,
        }
        for index, unit in enumerate(snapshot.units)
    ], snapshot.languages


def _profile_snapshot(profile_id):
    from src.core.book_profiles.loader import load_book_profile

    profile = load_book_profile(profile_id)
    value = asdict(profile)
    value.pop("root", None)
    return value


def calibrate_local():
    """Synthetic sensitivity evidence, not a semantic or literary verdict."""
    from src.core.fidelity_supervisor import assess_fidelity

    fixture = json.loads(
        (REPO / "scripts/fixtures/editorial_controls.json").read_text(encoding="utf-8")
    )
    records = []
    for control in fixture["controls"]:
        candidate = fixture["legitimate_candidate"]
        if control["old"]:
            if candidate.count(control["old"]) != 1:
                raise ValueError("Control mutation no longer has a unique anchor")
            candidate = candidate.replace(control["old"], control["new"], 1)
        decision = assess_fidelity(
            fixture["source_text"],
            candidate,
            chunk_index=1,
            phase="translation",
            source_language=fixture["source_language"],
            target_language=fixture["target_language"],
            prompt_options={"fidelity_supervisor_mode": "local"},
        )
        records.append(
            {
                "id": control["id"],
                "expected_kind": control["expected_kind"],
                "candidate": candidate,
                "local_accepted": decision.accepted,
                "local_issues": [asdict(issue) for issue in decision.issues],
                "expected_defect_confirmed": None,
            }
        )
    return {
        "evidence": "synthetic_deterministic_controls",
        "paid_requests": 0,
        "source_text": fixture["source_text"],
        "records": records,
        "negative_controls_unflagged": sum(
            r["expected_kind"] is not None and not r["local_issues"] for r in records
        ),
        "legitimate_controls_flagged": sum(
            r["expected_kind"] is None and bool(r["local_issues"]) for r in records
        ),
        "semantic_detection_measured": False,
        "literary_quality_proven": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser(
        "inventory", help="Show source positions and lengths, not private text"
    )
    inventory.add_argument("source", type=Path)
    inventory.add_argument("--start", type=int, default=0)
    inventory.add_argument("--count", type=int, default=40)
    prepare = sub.add_parser(
        "prepare", help="Freeze two adjacent sequences outside the repository"
    )
    prepare.add_argument("source", type=Path)
    profile_group = prepare.add_mutually_exclusive_group(required=True)
    profile_group.add_argument("--profile")
    profile_group.add_argument(
        "--unresolved-profile",
        action="store_true",
        help="Record missing historical profile; not eligible for a paid comparison",
    )
    prepare.add_argument("--source-language", required=True)
    prepare.add_argument("--target-language", required=True)
    prepare.add_argument("--tuning-start", type=int, required=True)
    prepare.add_argument("--heldout-start", type=int, required=True)
    prepare.add_argument("--sequence-length", type=int, default=10)
    prepare.add_argument("--output", type=Path, required=True)
    blind = sub.add_parser(
        "blind", help="Render complete A/B/C/D candidates with a separate answer key"
    )
    blind.add_argument("corpus", type=Path)
    blind.add_argument("candidates", type=Path)
    blind.add_argument("--split", choices=("tuning", "heldout"), required=True)
    blind.add_argument("--seed", required=True)
    blind.add_argument("--output", type=Path, required=True)
    controls = sub.add_parser(
        "calibrate-local",
        help="Run six synthetic negative controls and two legitimate variants",
    )
    controls.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser(
        "pilot-init", help="Create one gross allowance for the entire paid pilot"
    )
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--limit-usd", default="5")
    pilot.add_argument("--max-calls", type=int, default=1000)
    status = sub.add_parser(
        "pilot-status", help="Read total and per-experiment reserved/settled spend"
    )
    status.add_argument("directory", type=Path)
    plan = sub.add_parser("plan", help="Freeze a no-network A/B/C/D TXT experiment")
    plan.add_argument("corpus", type=Path)
    plan.add_argument("--baseline", type=Path, required=True)
    plan.add_argument("--split", choices=("tuning", "heldout"), required=True)
    plan.add_argument("--seed", required=True)
    plan.add_argument("--limit-usd", default="5")
    plan.add_argument("--max-calls", type=int, default=300)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--pilot", type=Path, required=True)
    run = sub.add_parser(
        "run", help="Execute the frozen experiment with a shared gross API budget"
    )
    run.add_argument("directory", type=Path)
    run.add_argument("--allow-paid", action="store_true", required=True)
    run.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.command in {"pilot-init", "pilot-status"}:
        from scripts.editorial_benchmark_pilot import create_pilot, pilot_status

        directory = (
            args.directory
            if args.command == "pilot-status"
            else create_pilot(
                args.output, limit_usd=args.limit_usd, max_calls=args.max_calls
            )
        )
        print(json.dumps(pilot_status(directory)))
        return
    if args.command == "plan":
        from scripts.editorial_benchmark_run import initialize, make_plan

        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        frozen = make_plan(
            corpus,
            baseline,
            split=args.split,
            seed=args.seed,
            limit_usd=args.limit_usd,
            max_calls=args.max_calls,
            pilot=args.pilot,
        )
        directory = initialize(args.output, corpus, frozen)
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "budget": frozen["budget"],
                    "pilot": frozen["pilot"],
                    "plan_sha256": frozen["plan_sha256"],
                    "paid_requests": 0,
                }
            )
        )
        return
    if args.command == "run":
        import asyncio

        from scripts.editorial_benchmark_run import execute
        from src.config import DEEPSEEK_API_KEY, OPENAI_API_KEY
        from src.core.llm.preflight import preflight_openai

        async def run_paid():
            if not DEEPSEEK_API_KEY or not OPENAI_API_KEY:
                raise ValueError(
                    "Both OpenAI and DeepSeek credentials are required; no generation started"
                )
            report = await preflight_openai(api_key=OPENAI_API_KEY)
            if report["status"] != "models_visible":
                raise ValueError(
                    "OpenAI model visibility preflight failed; no generation started"
                )
            return await execute(
                args.directory,
                resume=args.resume,
                credentials={"openai": OPENAI_API_KEY, "deepseek": DEEPSEEK_API_KEY},
            )

        result = asyncio.run(run_paid())
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "budget": result["budget"],
                    "pilot_budget": result["pilot_budget"],
                    "human_review_required": True,
                }
            )
        )
        return
    if args.command == "inventory":
        units, languages = source_units(args.source)
        if args.start < 0 or not 1 <= args.count <= 100:
            parser.error("Inventory range must be non-negative and at most 100 units")
        print(
            json.dumps(
                {
                    "total_units": len(units),
                    "declared_languages": languages,
                    "units": [
                        {k: u[k] for k in ("locator", "section", "position")}
                        | {"characters": len(u["text"])}
                        for u in units[args.start : args.start + args.count]
                    ],
                },
                indent=2,
            )
        )
        return
    directory = private_directory(args.output)
    if args.command == "calibrate-local":
        report = calibrate_local()
        write_private(directory / "local-controls.json", report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in {"records", "source_text"}}
            )
        )
        return
    if args.command == "prepare":
        source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
        units, languages = source_units(args.source)
        if hashlib.sha256(args.source.read_bytes()).hexdigest() != source_hash:
            raise ValueError("Source changed during corpus extraction")
        profile = (
            _profile_snapshot(args.profile)
            if args.profile
            else {"status": "unresolved", "target_locale": ""}
        )
        declared = profile.get("raw_config", {})
        for field, requested in (
            ("source_language", args.source_language),
            ("target_language", args.target_language),
        ):
            if (
                declared.get(field)
                and declared[field].casefold() != requested.casefold()
            ):
                raise ValueError(f"Requested {field} differs from the selected profile")
        corpus = prepare_corpus(
            units,
            book_id=source_hash[:24],
            source_sha256=source_hash,
            source_language=args.source_language,
            target_language=args.target_language,
            target_locale=profile.get("target_locale", ""),
            profile=profile,
            sequences=[
                {
                    "start": args.tuning_start,
                    "count": args.sequence_length,
                    "split": "tuning",
                },
                {
                    "start": args.heldout_start,
                    "count": args.sequence_length,
                    "split": "heldout",
                },
            ],
        )
        write_private(directory / "corpus.json", corpus)
        print(
            json.dumps(
                {
                    "corpus": str(directory / "corpus.json"),
                    "units": len(corpus["units"]),
                    "declared_source_languages": languages,
                    "paid_requests": 0,
                    "coverage_verified_by_human": False,
                }
            )
        )
    else:
        corpus = validate_corpus(json.loads(args.corpus.read_text(encoding="utf-8")))
        records = json.loads(args.candidates.read_text(encoding="utf-8"))
        page, key = render_blind_sample(
            corpus, records, seed=args.seed, split=args.split
        )
        write_private(directory / "blind.html", page)
        write_private(directory / "answer-key.json", key)
        print(
            json.dumps(
                {"blind_sample": str(directory / "blind.html"), "paid_requests": 0}
            )
        )


if __name__ == "__main__":
    main()
