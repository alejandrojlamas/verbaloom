"""Budgeted benchmark evaluation through the existing role-aware providers."""

import inspect
import json
import time
from dataclasses import asdict

from .editorial_benchmark import (
    JUDGE_VERSION,
    _digest,
    build_judge_prompt,
    validate_assessment,
)
from .llm.request_deadline import await_llm_call
from .pipeline_runtime import active_pipeline
from .usage.budget import SpendBudget
from .usage.context import reset_usage_context, set_usage_context


def _model_identity(provider, model):
    if provider == "deepseek":
        from .llm.providers.deepseek import DeepSeekProvider

        model = DeepSeekProvider.LEGACY_MODEL_ALIASES.get(model, model)
    return provider, model


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate assessment JSON field")
        result[key] = value
    return result


def evaluation_identity(corpus, unit_id, candidate, previous_candidate=""):
    return _digest(
        [
            corpus["content_sha256"],
            unit_id,
            candidate,
            previous_candidate,
            JUDGE_VERSION,
        ]
    )


def restore_evaluation(
    record, corpus, unit_id, candidate, *, previous_candidate="", judge
):
    """Validate a durable raw receipt without paying for the same judgment again."""
    if (
        record.get("input_sha256")
        != evaluation_identity(corpus, unit_id, candidate, previous_candidate)
        or record.get("judge") != judge
        or record.get("unit_id") != unit_id
        or record.get("corpus_sha256") != corpus["content_sha256"]
    ):
        raise ValueError("Retained evaluation does not match its frozen inputs")
    response = record.get("response")
    if (
        not response
        or response.get("was_truncated")
        or response.get("refusal")
        or response.get("status") != "completed"
        or not response.get("content", "").strip()
    ):
        raise ValueError("Benchmark assessment is incomplete or refused")
    unit = next(u for u in corpus["units"] if u["unit_id"] == unit_id)
    value = json.loads(response["content"], object_pairs_hook=_unique_object)
    result = validate_assessment(value, unit["source_text"], candidate["text"])
    return {
        **result,
        "elapsed_seconds": record["elapsed_seconds"],
        "judge": dict(judge),
        "prompt_version": JUDGE_VERSION,
        "unit_id": unit_id,
        "corpus_sha256": corpus["content_sha256"],
        "request_id": response.get("request_id", ""),
        "response_id": response.get("response_id", ""),
    }


async def evaluate_candidate(
    corpus,
    unit_id,
    candidate,
    *,
    experiment_scope,
    response_sink,
    previous_candidate="",
):
    """No unbudgeted or self-judged evaluation, no editorial retry loop.

    The caller freezes the corpus/candidate and persists every raw response in
    its private directory. Provider-level attempts share the experiment ledger.
    A malformed response is recorded but never converted into an approval.
    """
    runtime = active_pipeline.get()
    if (
        runtime is None
        or runtime.usage_store is None
        or experiment_scope not in runtime.budget_scope_ids
        or not str(experiment_scope).startswith("experiment:")
        or not callable(response_sink)
        or inspect.iscoroutinefunction(response_sink)
        or inspect.iscoroutinefunction(type(response_sink).__call__)
    ):
        raise ValueError(
            "Evaluation requires a private ledger, experiment budget and response sink"
        )
    SpendBudget(runtime.usage_store).snapshot(experiment_scope)
    if corpus.get("book", {}).get("profile", {}).get("status") == "unresolved":
        raise ValueError("Recover the historical profile/locale before paid evaluation")
    generators = candidate.get("generators")
    if not isinstance(generators, list) or not generators:
        raise ValueError("Candidate generation provenance is required")
    identities = set()
    for generator in generators:
        if not isinstance(generator, dict) or set(generator) != {"provider", "model"}:
            raise ValueError("Invalid generator provenance")
        if not all(isinstance(v, str) and v for v in generator.values()):
            raise ValueError("Invalid generator identity")
        identities.add(_model_identity(generator["provider"], generator["model"]))
    selected = runtime.config["roles"]["audit"]
    if _model_identity(selected["provider"], selected["model"]) in identities:
        raise ValueError(
            "Benchmark judge must differ from every generating/repairing model"
        )
    system, user = build_judge_prompt(
        corpus,
        unit_id,
        candidate["text"],
        previous_candidate=previous_candidate,
    )
    context = set_usage_context(
        translation_id=f"{experiment_scope}:evaluation:{candidate.get('variant', 'unassigned')}:{unit_id}",
        process_id=experiment_scope,
        process_type="editorial_benchmark",
        phase="audit",
        unit_id=unit_id,
        prompt_version=JUDGE_VERSION,
        profile_hash=corpus["content_sha256"],
    )
    started = time.perf_counter()
    try:
        client = runtime.client_for("audit")
        response = await await_llm_call(
            client.generate,
            user,
            provider=client,
            role="audit",
            system_prompt=system,
        )
        elapsed = time.perf_counter() - started
        record = {
            "input_sha256": evaluation_identity(
                corpus, unit_id, candidate, previous_candidate
            ),
            "unit_id": unit_id,
            "corpus_sha256": corpus["content_sha256"],
            "judge": dict(selected),
            "prompt_version": JUDGE_VERSION,
            "elapsed_seconds": elapsed,
            "response": asdict(response) if response else None,
        }
        response_sink(record)
        return restore_evaluation(
            record,
            corpus,
            unit_id,
            candidate,
            previous_candidate=previous_candidate,
            judge=selected,
        )
    finally:
        reset_usage_context(context)
