"""Strict command-line workflow for complete-book translation jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

from src.config import (
    API_ENDPOINT,
    DEEPSEEK_API_KEY,
    DEFAULT_MODEL,
    DEFAULT_SOURCE_LANGUAGE,
    DEFAULT_TARGET_LANGUAGE,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    MISTRAL_API_KEY,
    NIM_API_KEY,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    POE_API_KEY,
)
from src.core.adapters import translate_file
from src.persistence.checkpoint_manager import CheckpointManager
from src.utils.file_utils import get_unique_output_path

from .config import QualityAssuranceConfig, config_from_job
from .extractors import inspect_source
from .runner import run_quality_assurance

COMMANDS = {"inspect", "translate", "resume", "validate", "repair", "export", "report"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="translator",
        description="Translate complete books with deterministic publication gates.",
    )
    parser.add_argument(
        "--report-root",
        default="data/quality_runs",
        help="Directory used for manifests, reports, and staged artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect and inventory a source book.")
    inspect_parser.add_argument("input")
    inspect_parser.add_argument("--source-language", default="auto")
    inspect_parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    inspect_parser.add_argument("--run-id", default="")

    translate_parser = subparsers.add_parser("translate", help="Translate, validate, and publish a book.")
    _add_translation_arguments(translate_parser)

    resume_parser = subparsers.add_parser("resume", help="Resume a persisted run without repeating approved units.")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--output", default="")

    validate_parser = subparsers.add_parser("validate", help="Run all quality gates for a run or source/output pair.")
    validate_parser.add_argument("run_id", nargs="?", default="")
    validate_parser.add_argument("--source", default="")
    validate_parser.add_argument("--output", default="")
    validate_parser.add_argument("--source-language", default="")
    validate_parser.add_argument("--target-language", default="")
    validate_parser.add_argument("--target-locale", default="")

    repair_parser = subparsers.add_parser("repair", help="Reprocess only units rejected by quality gates.")
    repair_parser.add_argument("run_id")
    repair_parser.add_argument("--output", default="")

    export_parser = subparsers.add_parser("export", help="Atomically publish an already validated staged artifact.")
    export_parser.add_argument("run_id")
    export_parser.add_argument("--output", default="")

    report_parser = subparsers.add_parser("report", help="Print the final report location and summary.")
    report_parser.add_argument("run_id")
    report_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _add_translation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input")
    parser.add_argument("--output", default="")
    parser.add_argument("--source-language", default=DEFAULT_SOURCE_LANGUAGE)
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--target-locale", default="es-MX")
    parser.add_argument("--provider", default=LLM_PROVIDER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-endpoint", default=API_ENDPOINT)
    parser.add_argument("--translator-model", default="")
    parser.add_argument("--reviewer-model", default="")
    parser.add_argument("--auditor-model", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--glossary", default="")
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--allow-warnings", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default="")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            return _inspect(args)
        if args.command == "translate":
            return asyncio.run(_translate(args))
        if args.command == "resume":
            return asyncio.run(_resume(args, repair=False))
        if args.command == "repair":
            return asyncio.run(_resume(args, repair=True))
        if args.command == "validate":
            return _validate(args)
        if args.command == "export":
            return _export(args)
        if args.command == "report":
            return _report(args)
    except KeyboardInterrupt:
        print("Translation interrupted; checkpoint preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1


def _inspect(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"inspect_{uuid.uuid4().hex[:12]}"
    manifest = inspect_source(
        args.input,
        source_language=args.source_language,
        target_language=args.target_language,
        run_id=run_id,
    )
    root = Path(args.report_root) / _safe_run_id(run_id)
    path = manifest.write_json(root / "translation_manifest.json")
    print(json.dumps({
        "run_id": run_id,
        "manifest": str(path),
        "counts": manifest.counts(),
        "format": manifest.source_format,
    }, ensure_ascii=False, indent=2))
    return 0


async def _translate(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    run_id = args.run_id or f"run_{uuid.uuid4().hex[:12]}"
    run_root = Path(args.report_root).expanduser().resolve() / _safe_run_id(run_id)
    staged = run_root / "staging" / _default_output_name(source, args.target_locale)
    staged.parent.mkdir(parents=True, exist_ok=True)
    desired = Path(args.output).expanduser() if args.output else source.with_name(
        _default_output_name(source, args.target_locale)
    )
    desired = Path(get_unique_output_path(str(desired.resolve())))
    prompt_options: dict[str, Any] = {
        "strict_quality_assurance": True,
        "target_locale": args.target_locale,
        "fidelity_supervisor_mode": "alerted",
        "fidelity_supervisor_retry": True,
    }
    if args.reviewer_model:
        prompt_options["source_aware_editorial_guard_model"] = args.reviewer_model
    if args.auditor_model:
        prompt_options["fidelity_supervisor_model"] = args.auditor_model
    if args.glossary:
        from src.core.glossary.cli_loader import load_glossary_from_file

        terms, metadata = load_glossary_from_file(args.glossary)
        prompt_options["glossary_terms"] = terms
        prompt_options["glossary_term_metadata"] = metadata
    provider = args.provider
    model = args.translator_model or args.model
    job_config = {
        "source_language": args.source_language,
        "target_language": args.target_language,
        "target_locale": args.target_locale,
        "file_type": _file_type(source),
        "file_path": str(source),
        "preserved_input_path": str(source),
        "output_filename": desired.name,
        "output_filepath": str(desired),
        "qa_staging_output": str(staged),
        "llm_provider": provider,
        "model": model,
        "llm_api_endpoint": args.api_endpoint,
        "prompt_options": prompt_options,
        "concurrency": max(1, args.concurrency),
        "max_cost": args.max_cost,
        "quality_assurance": {
            "strict": True,
            "validation": {"allow_warnings": bool(args.allow_warnings)},
            "export": {
                "keep_intermediate_files": True,
                "overwrite_valid_output": False,
                "atomic_publish": True,
            },
        },
    }
    manager = CheckpointManager()
    if not manager.start_job(run_id, _file_type(source), job_config, str(source)):
        raise RuntimeError(f"Run ID already exists: {run_id}")
    if args.dry_run:
        manifest = inspect_source(
            source,
            source_language=args.source_language,
            target_language=args.target_language,
            run_id=run_id,
        )
        manifest.write_json(run_root / "translation_manifest.json")
        print(json.dumps({"run_id": run_id, "dry_run": True, "counts": manifest.counts()}, indent=2))
        return 0
    success = await _invoke_translation(manager, run_id, job_config, source, staged, resume_from=0)
    return _finalize_cli_run(
        manager,
        run_id,
        job_config,
        source,
        staged,
        desired,
        processing_success=success,
        report_root=args.report_root,
    )


async def _resume(args: argparse.Namespace, *, repair: bool) -> int:
    manager = CheckpointManager()
    checkpoint = manager.load_checkpoint(args.run_id)
    if not checkpoint:
        raise RuntimeError(f"No checkpoint found for run {args.run_id}")
    job_config = dict(checkpoint["job"]["config"])
    source = Path(
        job_config.get("preserved_input_path")
        or job_config.get("file_path")
        or ""
    )
    if not source.exists():
        raise FileNotFoundError("The preserved source file is unavailable.")
    staged_value = str(job_config.get("qa_staging_output") or "").strip()
    if staged_value:
        staged = Path(staged_value)
    else:
        staged = Path(args.report_root) / _safe_run_id(args.run_id) / "staging" / _default_output_name(
            source, job_config.get("target_locale") or job_config.get("target_language") or "target"
        )
        job_config["qa_staging_output"] = str(staged)
    desired = Path(args.output or job_config.get("output_filepath") or job_config.get("output_filename") or staged.name)
    if repair:
        options = job_config.setdefault("prompt_options", {})
        options["quality_repair_mode"] = True
        options["fidelity_supervisor_mode"] = "always"
    manager.mark_running(args.run_id)
    manager.update_job_config(args.run_id, job_config)
    success = await _invoke_translation(
        manager,
        args.run_id,
        job_config,
        source,
        staged,
        resume_from=int(checkpoint.get("resume_from_index") or 0),
    )
    return _finalize_cli_run(
        manager,
        args.run_id,
        job_config,
        source,
        staged,
        desired,
        processing_success=success,
        report_root=args.report_root,
    )


async def _invoke_translation(
    manager: CheckpointManager,
    run_id: str,
    config: dict[str, Any],
    source: Path,
    staged: Path,
    *,
    resume_from: int,
) -> bool:
    staged.parent.mkdir(parents=True, exist_ok=True)
    return await translate_file(
        input_filepath=str(source),
        output_filepath=str(staged),
        source_language=config.get("source_language") or "auto",
        target_language=config.get("target_language") or "",
        model_name=config.get("model") or DEFAULT_MODEL,
        llm_provider=config.get("llm_provider") or LLM_PROVIDER,
        checkpoint_manager=manager,
        translation_id=run_id,
        resume_from_index=resume_from,
        llm_api_endpoint=config.get("llm_api_endpoint") or API_ENDPOINT,
        gemini_api_key=GEMINI_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        openrouter_api_key=OPENROUTER_API_KEY,
        mistral_api_key=MISTRAL_API_KEY,
        deepseek_api_key=DEEPSEEK_API_KEY,
        poe_api_key=POE_API_KEY,
        nim_api_key=NIM_API_KEY,
        prompt_options=config.get("prompt_options") or {},
        concurrency=config.get("concurrency", 1),
    )


def _finalize_cli_run(
    manager: CheckpointManager,
    run_id: str,
    job_config: dict[str, Any],
    source: Path,
    staged: Path,
    desired: Path,
    *,
    processing_success: bool,
    report_root: str,
) -> int:
    if not staged.exists():
        manager.mark_partial(run_id)
        raise RuntimeError("Translation produced no staged artifact.")
    checkpoint = manager.load_checkpoint(run_id)
    quality = run_quality_assurance(
        source_path=source,
        output_path=staged,
        source_language=job_config.get("source_language") or "auto",
        target_language=job_config.get("target_language") or "",
        run_id=run_id,
        job_config=job_config,
        checkpoint_data=checkpoint,
        report_root=report_root,
        checkpoint_manager=manager,
    )
    maximum = job_config.get("max_cost")
    estimated_cost = float(quality.report.model_usage.get("estimated_cost_usd") or 0.0)
    cost_exceeded = maximum is not None and estimated_cost > float(maximum)
    if not processing_success or not quality.publishable or cost_exceeded:
        manager.mark_partial(run_id)
        print(json.dumps({
            "run_id": run_id,
            "status": "BLOCKED",
            "processing_success": processing_success,
            "quality_status": quality.report.status.value,
            "repair_units": quality.repair_checkpoint_indices,
            "report": str(quality.paths.report_html),
            "staged_artifact": str(staged),
            "max_cost_exceeded": cost_exceeded,
        }, ensure_ascii=False, indent=2))
        return 2

    published = _atomic_publish(staged, desired)
    quality.update_output_path(published)
    job_config["output_filepath"] = str(published)
    job_config["quality_assurance_result"] = {
        "status": quality.report.status.value,
        "publishable": True,
        "report_dir": str(quality.paths.root),
    }
    manager.update_job_config(run_id, job_config)
    manager.mark_completed(run_id, quality_gate_passed=True)
    print(json.dumps({
        "run_id": run_id,
        "status": quality.report.status.value,
        "output": str(published),
        "report": str(quality.paths.report_html),
        "manifest": str(quality.paths.manifest),
    }, ensure_ascii=False, indent=2))
    return 0


def _validate(args: argparse.Namespace) -> int:
    manager = CheckpointManager()
    checkpoint = manager.load_checkpoint(args.run_id) if args.run_id else None
    job_config = dict((checkpoint or {}).get("job", {}).get("config") or {})
    source = Path(args.source or job_config.get("preserved_input_path") or job_config.get("file_path") or "")
    output = Path(args.output or job_config.get("qa_staging_output") or job_config.get("output_filepath") or "")
    if not source.exists() or not output.exists():
        raise FileNotFoundError("Both source and output artifacts are required for validation.")
    run_id = args.run_id or f"validate_{uuid.uuid4().hex[:12]}"
    if args.target_locale:
        job_config.setdefault("prompt_options", {})["target_locale"] = args.target_locale
    quality = run_quality_assurance(
        source_path=source,
        output_path=output,
        source_language=args.source_language or job_config.get("source_language") or "auto",
        target_language=args.target_language or job_config.get("target_language") or DEFAULT_TARGET_LANGUAGE,
        run_id=run_id,
        job_config=job_config,
        checkpoint_data=checkpoint,
        report_root=args.report_root,
        checkpoint_manager=manager if args.run_id else None,
        mark_failed_for_repair=False,
    )
    print(json.dumps({
        "run_id": run_id,
        "status": quality.report.status.value,
        "publishable": quality.publishable,
        "report": str(quality.paths.report_html),
        "repair_units": quality.repair_checkpoint_indices,
    }, ensure_ascii=False, indent=2))
    return 0 if quality.publishable else 2


def _export(args: argparse.Namespace) -> int:
    root = Path(args.report_root) / _safe_run_id(args.run_id)
    report_path = root / "translation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("publishable"):
        raise RuntimeError("The run is not publishable; repair and validate it first.")
    manager = CheckpointManager()
    job = manager.get_job(args.run_id) or {}
    config = dict(job.get("config") or {})
    staged_value = str(config.get("qa_staging_output") or report.get("output", {}).get("path") or "").strip()
    if not staged_value:
        raise FileNotFoundError("The validated staged artifact is unavailable.")
    staged = Path(staged_value)
    if not staged.exists():
        raise FileNotFoundError("The validated staged artifact is unavailable.")
    desired = Path(args.output or config.get("output_filepath") or staged.name)
    published = _atomic_publish(staged, desired)
    print(str(published))
    return 0


def _report(args: argparse.Namespace) -> int:
    root = Path(args.report_root) / _safe_run_id(args.run_id)
    report_path = root / "translation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Status: {report.get('status')}")
        print(f"Publishable: {report.get('publishable')}")
        print(f"HTML report: {root / 'translation_report.html'}")
        print(f"Manifest: {root / 'translation_manifest.json'}")
    return 0


def _atomic_publish(source: Path, destination: Path) -> Path:
    destination = Path(get_unique_output_path(str(destination.expanduser().resolve()))) if destination.exists() else destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.publishing")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return destination


def _default_output_name(source: Path, target_locale: str) -> str:
    locale = str(target_locale or "target").replace("/", "-")
    return f"{source.stem} ({locale}){source.suffix or '.txt'}"


def _file_type(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return "txt" if suffix in {"", "text", "md", "markdown"} else suffix


def _safe_run_id(value: str) -> str:
    import re

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "run")).strip("._")
    return normalized[:120] or "run"
