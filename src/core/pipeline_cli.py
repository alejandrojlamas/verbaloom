"""CLI adapters for the shared role contract, preflight and execution scope."""

import json
from pathlib import Path

from src.core.llm import preflight
from src.core.llm.exceptions import ProviderActionRequiredError
from src.core.pipeline_config import (
    PRESETS,
    ROLES,
    apply_pipeline,
    pipeline_preset,
    validate_pipeline,
)
from src.core.pipeline_runtime import activate_pipeline, close_pipeline
from src.core.usage.context import reset_usage_context, set_usage_context
from src.utils.provider_security import resolve_pipeline_credentials


def add_pipeline_arguments(parser):
    group = parser.add_argument_group("Editorial pipeline", "Optional; legacy provider behavior is unchanged without these options.")
    choice = group.add_mutually_exclusive_group()
    choice.add_argument("--pipeline-preset", choices=PRESETS)
    choice.add_argument("--pipeline-config", type=Path, help="Complete pipeline JSON snapshot, without credentials.")
    group.add_argument(
        "--role", action="append", nargs=5, default=[],
        metavar=("ROLE", "PROVIDER", "MODEL", "EFFORT", "OUTPUT_TOKENS"),
        help="Override one role explicitly; repeat for different roles. Effort: none or low.",
    )
    group.add_argument("--review-policy", choices=("compatible", "source_audited_v1"))
    group.add_argument("--budget-usd", "--max-cost", dest="budget_usd", help="Gross API USD cap; requires --max-calls.")
    group.add_argument("--max-calls", type=int, help="HTTP attempt cap, including retries; requires --budget-usd.")
    group.add_argument("--preflight-only", action="store_true", help="Check role model visibility without generation or creating a job; balance remains unknown.")


def configure_cli_pipeline(args, config):
    """Validate all overrides before mutating config; never store a credential."""
    selected = getattr(args, "pipeline_preset", None)
    filename = getattr(args, "pipeline_config", None)
    overrides = getattr(args, "role", [])
    review = getattr(args, "review_policy", None)
    budget = getattr(args, "budget_usd", None)
    calls = getattr(args, "max_calls", None)
    if not any((selected, filename, overrides, review, budget is not None, calls is not None)):
        if getattr(args, "preflight_only", False):
            raise ValueError("--preflight-only requires a pipeline preset or configuration")
        return None
    if any(getattr(args, key, "") for key in ("translator_model", "reviewer_model", "auditor_model")):
        raise ValueError("Use --role with pipeline settings, not legacy --translator/reviewer/auditor-model overrides")
    if (budget is None) != (calls is None):
        raise ValueError("--budget-usd/--max-cost and --max-calls must be supplied together")
    if filename:
        try:
            pipeline = validate_pipeline(json.loads(Path(filename).read_text(encoding="utf-8")))
        except (OSError, UnicodeError, ValueError) as exc:
            # Do not echo JSON decode fragments: an invalid file may contain a key.
            raise ValueError("Invalid pipeline JSON file; expected a credential-free pipeline snapshot") from exc
    else:
        pipeline = pipeline_preset(selected or "compatible", config)
    seen = set()
    for name, provider, model, effort, output in overrides:
        if name not in ROLES or name in seen:
            raise ValueError("Each --role must name a distinct supported role")
        seen.add(name)
        try:
            allowance = int(output)
        except ValueError as exc:
            raise ValueError("Role output allowance must be an integer") from exc
        pipeline["roles"][name] = {
            "provider": provider, "model": model, "reasoning_effort": effort,
            "max_output_tokens": allowance,
        }
    if review:
        pipeline["review_policy"] = review
        pipeline["epub_repair_policy"] = "localized_v1" if review == "source_audited_v1" else "compatible"
    if overrides or review:
        pipeline["preset"] = "custom"
    if budget is not None:
        pipeline["budget"] = {"limit_usd": budget, "max_calls": calls}
    return apply_pipeline(config, pipeline)


def cli_credentials(config, supplied):
    """Resolve each provider's own secret into a transient map, never the job."""
    temporary = {"pipeline": config["pipeline"]}
    resolve_pipeline_credentials(temporary, supplied)
    providers = {role["provider"] for role in config["pipeline"]["roles"].values()}
    return {provider: temporary[f"{provider}_api_key"] for provider in providers}


async def check_cli_pipeline(config, supplied):
    pipeline = validate_pipeline(config["pipeline"])
    credentials = cli_credentials(config, supplied)
    reports = []
    for provider in sorted(credentials):
        models = sorted({role["model"] for role in pipeline["roles"].values() if role["provider"] == provider})
        check = preflight.preflight_openai if provider == "openai" else preflight.preflight_deepseek
        reports.append(await check(api_key=credentials[provider], models=models))
    return {
        "status": "models_visible" if all(report["status"] == "models_visible" for report in reports) else "not_ready",
        "pipeline": pipeline, "providers": reports, "generation_tested": False, "balance": None,
    }


async def run_cli_operation(operation, config, run_id, credentials, **kwargs):
    """Use one async scope for every format, including frozen resume and refinement."""
    pipeline = config.get("pipeline")
    keys = {}
    if pipeline is not None:
        report = await check_cli_pipeline(config, credentials)
        if report["status"] != "models_visible":
            raise ProviderActionRequiredError(
                "Pipeline preflight failed; use --preflight-only to inspect model access.",
                reason="pipeline_preflight", provider="pipeline",
            )
        keys = cli_credentials(config, credentials)
    usage_token = set_usage_context(
        translation_id=run_id, process_id=run_id,
        process_type="refine" if config.get("refine_only") else "translate",
        book_name=Path(kwargs["input_filepath"]).name,
        input_filename=Path(kwargs["input_filepath"]).name,
        output_filename=config.get("output_filename", ""),
        provider=config.get("llm_provider"), model=config.get("model"),
    )
    token = None
    try:
        token = activate_pipeline(pipeline, job_id=run_id, credentials=keys)
        return await operation(**kwargs)
    finally:
        try:
            if token is not None:
                await close_pipeline(token)
        finally:
            reset_usage_context(usage_token)
