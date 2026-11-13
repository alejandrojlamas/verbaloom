"""Private A/B/C/D execution using the production TXT engine and request ledger.

Frozen paragraph boundaries are an experimental input, not a second translator.
This does not substitute for native EPUB/resource publication tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
import time
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path

from scripts.editorial_benchmark import REPO, private_directory, write_private
from src.core.adapters.generic_translator import GenericTranslator
from src.core.adapters.txt_adapter import TxtAdapter
from src.core.benchmark_judge import evaluate_candidate, restore_evaluation
from src.core.book_profiles.loader import frozen_profile_scope
from src.core.book_profiles.models import (
    BookProfile,
    ProfileDetector,
    ProfileGlossaryEntry,
    ProfileGlossaryExample,
)
from src.core.candidate_result import capture_candidates
from src.core.editorial_benchmark import (
    _digest,
    _no_credentials,
    render_blind_sample,
    validate_corpus,
)
from src.core.job_runtime_config import configure_editorial_guard_options
from src.core.pipeline_config import apply_pipeline, pipeline_preset
from src.core.pipeline_runtime import activate_pipeline, close_pipeline
from src.core.usage.budget import SpendBudget
from src.core.usage.context import reset_usage_context, set_usage_context
from src.core.usage.store import TokenUsageStore
from src.persistence.checkpoint_manager import CheckpointManager
from src.persistence.execution_lock import ExecutionLockedError, exclusive_execution

PRESETS = dict(
    zip(
        "ABCD",
        (
            "compatible",
            "openai_equivalent",
            "deepseek_source_audited",
            "openai_source_audited",
        ),
    )
)
SCHEMA = "verbaloom.editorial-execution/v1"


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(
        [
            *REPO.joinpath("src").rglob("*.py"),
            REPO / "scripts/editorial_benchmark.py",
            REPO / "scripts/editorial_benchmark_pilot.py",
            Path(__file__),
        ]
    ):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_contract():
    from src import config

    return {
        name: getattr(config, name)
        for name in (
            "REQUEST_TIMEOUT",
            "LLM_CLOUD_OPERATION_TIMEOUT",
            "LLM_LOCAL_OPERATION_TIMEOUT",
            "MAX_TRANSLATION_ATTEMPTS",
            "TEMPERATURE",
            "INPUT_TAG_IN",
            "INPUT_TAG_OUT",
            "TRANSLATE_TAG_IN",
            "TRANSLATE_TAG_OUT",
        )
    }


def profile_snapshot(corpus, directory):
    payload = deepcopy(corpus["book"]["profile"])
    if payload.get("status") == "unresolved" or not payload.get("profile_id"):
        raise ValueError("Recover the historical profile before execution")
    payload.pop("root", None)
    entries = []
    for entry in payload.get("glossary_entries", []):
        entry["examples"] = tuple(
            ProfileGlossaryExample(**e) for e in entry.get("examples", [])
        )
        for key in ("target_options", "restrictions", "do_not_apply_if"):
            entry[key] = tuple(entry.get(key, ()))
        entries.append(ProfileGlossaryEntry(**entry))
    payload["glossary_entries"] = tuple(entries)
    payload["detectors"] = tuple(
        ProfileDetector(**d) for d in payload.get("detectors", [])
    )
    profile = BookProfile(root=Path(directory), **payload)
    if profile.target_locale != corpus["book"]["target_locale"]:
        raise ValueError("Corpus locale differs from the frozen profile")
    return profile


def make_plan(
    corpus, baseline, *, split, seed, limit_usd="5", max_calls=300, pilot=None
):
    validate_corpus(corpus)
    _no_credentials(baseline)
    if type(max_calls) is not int or not 1 <= max_calls <= 1000:
        raise ValueError("A pilot requires 1 to 1000 maximum HTTP attempts")
    if split not in {"tuning", "heldout"} or not isinstance(seed, str) or not seed:
        raise ValueError("An explicit split and randomization seed are required")
    allowed = {
        "llm_provider",
        "model",
        "source_language",
        "target_language",
        "file_type",
        "prompt_options",
        "refine_after",
        "refine_only",
        "operation",
    }
    if not isinstance(baseline, dict) or set(baseline) - allowed:
        raise ValueError("Baseline accepts only editorial execution configuration")
    if (
        baseline.get("llm_provider") != "deepseek"
        or baseline.get("file_type") != "txt"
        or baseline.get("refine_only")
        or baseline.get("operation", "translate") != "translate"
        or baseline.get("refine_after")
    ):
        raise ValueError(
            "This executor requires the TXT translation baseline; chained transforms are separate experiments"
        )
    book = corpus["book"]
    if any(
        baseline.get(key) != book[key] for key in ("source_language", "target_language")
    ):
        raise ValueError("Baseline languages differ from the frozen corpus")
    profile = profile_snapshot(corpus, Path("frozen-profile"))
    options = baseline.get("prompt_options", {})
    if (
        options.get("profile_id") != profile.profile_id
        or options.get("target_locale", profile.target_locale) != profile.target_locale
        or options.get("text_transform_mode")
    ):
        raise ValueError(
            "Baseline must retain the corpus profile, locale and translation operation"
        )
    variants = {}
    with frozen_profile_scope(profile):
        for label, preset in PRESETS.items():
            config = deepcopy(baseline)
            config["prompt_options"]["target_locale"] = profile.target_locale
            pipeline = pipeline_preset(preset, config)
            pipeline["budget"] = {"limit_usd": str(limit_usd), "max_calls": max_calls}
            apply_pipeline(config, pipeline)
            configure_editorial_guard_options(config)
            # Same rule as the production translate_file TXT entrypoint.
            if config["prompt_options"].get("refine"):
                config["prompt_options"]["inline_refinement"] = True
            variants[label] = config
    sequences = sorted(
        {u["sequence_id"] for u in corpus["units"] if u["split"] == split}
    )
    if any(not re.fullmatch(r"sequence-[0-9]+", s) for s in sequences):
        raise ValueError("Unsafe sequence identifier")
    order = []
    rng = random.Random(seed)
    for sequence in sequences:
        labels = list(PRESETS)
        rng.shuffle(labels)
        order.extend({"variant": label, "sequence": sequence} for label in labels)
    result = {
        "schema": SCHEMA,
        "engine": "production_generic_txt_frozen_units",
        "corpus_sha256": corpus["content_sha256"],
        "code_sha256": code_digest(),
        "runtime_contract": runtime_contract(),
        "split": split,
        "seed": seed,
        "variants": variants,
        "order": order,
        "budget": {
            "limit_usd": variants["A"]["pipeline"]["budget"]["limit_usd"],
            "max_calls": max_calls,
        },
        "profile_preparation": "reuse_frozen_profile_no_new_preparation_calls",
        "native_resource_publication_tested": False,
    }
    if pilot is not None:
        from scripts.editorial_benchmark_pilot import pilot_binding

        result["pilot"] = pilot_binding(pilot)
    result["plan_sha256"] = _digest(result)
    return result


def summarize_usage(events):
    summary = {
        "calls": len(events),
        "observed_gross_api_usd": 0.0,
        "estimated_gross_api_usd": 0.0,
        "unknown_cost_requests": 0,
        "http_elapsed_ms": 0,
        "elapsed_unknown_requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "input_uncached_tokens": 0,
        "input_cache_read_tokens": 0,
        "input_cache_write_tokens": 0,
        "reasoning_tokens_included_in_completion": 0,
        "calls_by_status": {},
        "models": [],
    }
    identities = set()
    for event in events:
        metadata = json.loads(event["metadata"])
        basis = metadata.get("cost_basis")
        if basis == "provider_usage":
            summary["observed_gross_api_usd"] += event["total_cost_usd"]
        elif basis == "estimated":
            summary["estimated_gross_api_usd"] += event["total_cost_usd"]
        else:
            summary["unknown_cost_requests"] += 1
        if metadata.get("elapsed_ms") is None:
            summary["elapsed_unknown_requests"] += 1
        else:
            summary["http_elapsed_ms"] += metadata["elapsed_ms"]
        writes = metadata.get("cache_write_30m_tokens", 0) + metadata.get(
            "cache_write_12h_tokens", 0
        )
        reads = event["prompt_cache_hit_tokens"]
        summary["prompt_tokens"] += event["prompt_tokens"]
        summary["completion_tokens"] += event["completion_tokens"]
        summary["input_uncached_tokens"] += max(
            0, event["prompt_tokens"] - reads - writes
        )
        summary["input_cache_read_tokens"] += reads
        summary["input_cache_write_tokens"] += writes
        summary["reasoning_tokens_included_in_completion"] += metadata.get(
            "reasoning_tokens", 0
        )
        status = event["status"]
        summary["calls_by_status"][status] = (
            summary["calls_by_status"].get(status, 0) + 1
        )
        identities.add(
            (event["provider"], event["model"], metadata.get("response_model", ""))
        )
    summary["models"] = [
        {"provider": p, "requested_model": m, "response_model": r}
        for p, m, r in sorted(identities)
    ]
    return summary


def execution_report(plan, corpus, state, store, scope):
    events = store.events(limit=1000, process_id=scope)
    result = {
        "evidence": state["evidence"],
        "execution_status": state["status"],
        "human_review_required": True,
        "cash_paid_usd": None,
        "credits_applied_usd": None,
        "native_resource_publication_tested": False,
        "variants": {},
        "elapsed_execution_seconds": state.get("elapsed_seconds", 0),
        "budget": state["budget"],
        "pilot_budget": state.get("pilot_budget"),
        "judges_differ_between_provider_groups": True,
        "api_costs_are_simulated": state["evidence"] == "simulated",
        "profile_preparation": plan["profile_preparation"],
        "http_elapsed_is_summed_not_wall_clock": True,
    }
    selected_units = [u for u in corpus["units"] if u["split"] == plan["split"]]
    for variant in PRESETS:
        groups = {}
        for category, matching in (
            (
                "generation",
                [
                    e
                    for e in events
                    if (e["translation_id"] or "").startswith(
                        scope + ":" + variant + "-"
                    )
                ],
            ),
            (
                "evaluation",
                [
                    e
                    for e in events
                    if (e["translation_id"] or "").startswith(
                        scope + ":evaluation:" + variant + ":"
                    )
                ],
            ),
        ):
            stages = {
                stage: summarize_usage([e for e in matching if e["phase"] == stage])
                for stage in sorted({e["phase"] for e in matching})
            }
            by_unit = {
                unit["unit_id"]: summarize_usage(
                    [
                        e
                        for e in matching
                        if json.loads(e["metadata"]).get("unit_id") == unit["unit_id"]
                    ]
                )
                for unit in selected_units
            }
            groups[category] = {
                **summarize_usage(matching),
                "stages": stages,
                "units": by_unit,
                "calls_by_stage": {s: data["calls"] for s, data in stages.items()},
                "calls_per_planned_unit": len(matching) / len(selected_units),
            }
        approved_words = sum(
            len(u["source_text"].split())
            for u in selected_units
            if state["evaluations"]
            .get(variant + "-" + u["sequence_id"] + ":" + u["unit_id"], {})
            .get("accepted")
        )
        # Interrupted/corrupt evidence must not produce a comparable final ratio.
        complete = state["status"] == "completed"
        groups["independently_approved_source_words"] = (
            approved_words if complete else None
        )
        groups["generation_usd_per_1000_independently_approved_words"] = (
            1000 * groups["generation"]["observed_gross_api_usd"] / approved_words
            if complete
            and approved_words
            and not groups["generation"]["unknown_cost_requests"]
            and not groups["generation"]["estimated_gross_api_usd"]
            else None
        )
        for name, phases in (
            ("audited_unit_fraction", {"audit"}),
            ("repaired_unit_fraction", {"repair", "structure_repair"}),
        ):
            visited = {
                json.loads(e["metadata"]).get("unit_id")
                for e in events
                if (e["translation_id"] or "").startswith(scope + ":" + variant + "-")
                and e["phase"] in phases
            }
            groups[name] = len(visited & {u["unit_id"] for u in selected_units}) / len(
                selected_units
            )
        result["variants"][variant] = groups
    return result


def initialize(directory, corpus, plan):
    directory = private_directory(directory)
    if any(directory.iterdir()):
        raise ValueError("Use a new empty experiment directory")
    write_private(directory / "corpus.json", corpus)
    write_private(directory / "plan.json", plan)
    if plan.get("pilot"):
        from scripts.editorial_benchmark_pilot import register_experiment

        register_experiment(directory, plan)
    return directory


def _replace(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def execution_lock(directory):
    try:
        with exclusive_execution(directory / ".runner.lock"):
            yield
    except ExecutionLockedError as exc:
        raise ValueError("An executor already owns this experiment") from exc


class FrozenSequenceAdapter(TxtAdapter):
    def __init__(self, directory, config, units):
        super().__init__(directory / "source.txt", directory / "working.txt", config)
        self.units = units

    async def prepare_for_translation(self):
        self.chunks = [
            {
                "main_content": u["source_text"],
                "context_before": u["context_before"],
                "context_after": u["context_after"],
            }
            for u in self.units
        ]
        self.translated_chunks = [None] * len(self.chunks)
        return True

    def get_translation_units(self):
        return [
            replace(unit, unit_id=frozen["unit_id"])
            for unit, frozen in zip(super().get_translation_units(), self.units)
        ]

    async def save_unit_translation(self, unit_id, translated_content):
        index = next(
            i for i, unit in enumerate(self.units) if unit["unit_id"] == unit_id
        )
        return await super().save_unit_translation(f"chunk_{index}", translated_content)

    async def reconstruct_output(self, bilingual=False):
        if any(not text for text in self.translated_chunks):
            raise ValueError(
                "Incomplete benchmark sequence cannot use original text as fallback"
            )
        return await super().reconstruct_output(bilingual=bilingual)


def _units(corpus, task):
    return [u for u in corpus["units"] if u["sequence_id"] == task["sequence"]]


def _saved_candidates(manager, job_id, units, variant, evidence):
    saved = manager.load_checkpoint(job_id) or {}
    chunks = {c["chunk_index"]: c for c in saved.get("chunks", [])}
    if set(chunks) != set(range(len(units))):
        return None
    result = []
    for index, unit in enumerate(units):
        chunk = chunks[index]
        if (
            chunk["status"] != "completed"
            or not chunk.get("translated_text")
            or chunk["original_text"] != unit["source_text"]
        ):
            return None
        result.append(
            {
                "unit_id": unit["unit_id"],
                "variant": variant,
                "text": chunk["translated_text"],
                "evidence": evidence,
                "phase": "final",
            }
        )
    return result


def _judge_pipeline(plan, variant):
    baseline = plan["variants"]["A"]
    preset = "openai_equivalent" if variant in {"A", "C"} else "compatible"
    result = pipeline_preset(preset, baseline)
    result["roles"]["audit"].update(
        model="gpt-6.1-terra" if variant in {"A", "C"} else "deepseek-v4-pro"
    )
    result["budget"] = deepcopy(plan["budget"])
    return result


async def execute(
    directory, *, credentials, resume=False, evidence="provider_response"
):
    """Run production engines; tests replace HTTP only, never this orchestrator.

    All dispatches, including retries and independent judgments, share the same
    durable experiment scope. A returned engine file alone is not success.
    """
    directory = private_directory(directory)
    with execution_lock(directory):
        return await _execute_locked(
            directory, credentials=credentials, resume=resume, evidence=evidence
        )


async def _execute_locked(directory, *, credentials, resume, evidence):
    execution_started = time.monotonic()
    if evidence not in {"provider_response", "simulated"}:
        raise ValueError("Unknown evidence type")
    corpus = validate_corpus(json.loads((directory / "corpus.json").read_text()))
    plan = json.loads((directory / "plan.json").read_text())
    if (
        plan.get("schema") != SCHEMA
        or plan.get("plan_sha256")
        != _digest({k: v for k, v in plan.items() if k != "plan_sha256"})
        or plan["corpus_sha256"] != corpus["content_sha256"]
        or plan["code_sha256"] != code_digest()
        or plan["runtime_contract"] != runtime_contract()
    ):
        raise ValueError(
            "Frozen code, plan or corpus changed; do not resume this experiment"
        )
    if not all(
        isinstance(credentials.get(p), str)
        and credentials[p].strip()
        and credentials[p] != "__USE_ENV__"
        for p in ("deepseek", "openai")
    ):
        raise ValueError("Both providers need credentials before any A/B/C/D execution")
    if evidence == "provider_response" and not plan.get("pilot"):
        raise ValueError("Paid execution requires a shared parent pilot budget")
    state_path = directory / "state.json"
    if not state_path.exists() and any(
        path.name not in {"corpus.json", "plan.json", ".runner.lock"}
        for path in directory.iterdir()
    ):
        raise ValueError(
            "Execution state is missing from a started experiment; do not reset it"
        )
    if state_path.exists() and not resume:
        raise ValueError("Experiment already started; explicit resume is required")
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"evidence": evidence, "tasks": {}, "evaluations": {}, "status": "pending"}
    )
    if state["evidence"] != evidence:
        raise ValueError("Simulated and observed evidence cannot share an experiment")
    required = ("jobs.db",) if plan.get("pilot") else ("usage.db", "jobs.db")
    if state_path.exists() and any(
        not (directory / name).is_file() for name in required
    ):
        raise ValueError(
            "Ledger or checkpoints are missing; never recreate spent allowances"
        )
    parent_scope = None
    if plan.get("pilot"):
        from scripts.editorial_benchmark_pilot import start_experiment

        store, parent_scope = start_experiment(
            directory, plan, state if state_path.exists() else None, evidence=evidence
        )
    else:
        store = TokenUsageStore(directory / "usage.db")
    scope = "experiment:" + plan["plan_sha256"][:24]
    budget = SpendBudget(store)
    if state_path.exists():
        budget.snapshot(scope)
    budget.configure(scope, **plan["budget"])
    manager = CheckpointManager(
        str(directory / "jobs.db"), uploads_dir=directory / "uploads"
    )
    profile = profile_snapshot(corpus, directory / "profile")
    profile.root.mkdir(mode=0o700, exist_ok=True)
    state["status"] = "running"
    _replace(state_path, state)
    try:
        with frozen_profile_scope(profile):
            for task in plan["order"]:
                label = task["variant"]
                task_id = label + "-" + task["sequence"]
                cell = directory / task_id
                cell.mkdir(mode=0o700, exist_ok=True)
                units = _units(corpus, task)
                if not (cell / "source.txt").exists():
                    write_private(
                        cell / "source.txt",
                        "\n\n".join(u["source_text"] for u in units),
                    )
                job_id = scope + ":" + task_id
                candidates = _saved_candidates(manager, job_id, units, label, evidence)
                if candidates is None:
                    config = deepcopy(plan["variants"][label])
                    state["tasks"][task_id] = {"status": "running"}
                    _replace(state_path, state)
                    token = activate_pipeline(
                        config["pipeline"],
                        job_id=job_id,
                        credentials=credentials,
                        store=store,
                        experiment_scope=scope,
                        parent_scope=parent_scope,
                    )
                    usage = set_usage_context(
                        translation_id=job_id,
                        process_id=scope,
                        process_type="editorial_benchmark",
                        profile_hash=corpus["content_sha256"],
                        prompt_version=plan["code_sha256"],
                    )

                    def capture(record, cell=cell):
                        import uuid

                        from src.core.usage.context import get_usage_context

                        write_private(
                            cell / ("candidate-" + uuid.uuid4().hex + ".json"),
                            {
                                "unit_id": get_usage_context().get("unit_id"),
                                "captured_at": time.time(),
                                "candidate": record,
                            },
                        )

                    started = time.monotonic()
                    try:
                        adapter = FrozenSequenceAdapter(cell, config, units)
                        engine = GenericTranslator(
                            adapter, manager, job_id, memory_path=cell / "memory.db"
                        )
                        with capture_candidates(capture):
                            success = await engine.translate(
                                corpus["book"]["source_language"],
                                corpus["book"]["target_language"],
                                config["model"],
                                config["llm_provider"],
                                prompt_options=config["prompt_options"],
                                api_key=credentials[config["llm_provider"]],
                            )
                    finally:
                        reset_usage_context(usage)
                        await close_pipeline(token)
                    candidates = _saved_candidates(
                        manager, job_id, units, label, evidence
                    )
                    if not success or candidates is None:
                        raise ValueError(
                            "Production engine did not approve the complete sequence"
                        )
                    state["tasks"][task_id] = {
                        "status": "completed",
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    _replace(state_path, state)
                # Conservative provenance prevents self-judging even when a repair
                # role did not end up contributing text to the selected candidate.
                generators = [
                    plan["variants"][label]["pipeline"]["roles"][role]
                    for role in ("translate", "refine", "repair", "structure_repair")
                ]
                for candidate in candidates:
                    candidate["generators"] = [
                        {k: r[k] for k in ("provider", "model")} for r in generators
                    ]
                if not (cell / "accepted.json").exists():
                    write_private(cell / "accepted.json", candidates)
                elif json.loads((cell / "accepted.json").read_text()) != candidates:
                    raise ValueError("Accepted artifact differs from its checkpoint")
                state["tasks"].setdefault(task_id, {})["status"] = "completed"
                _replace(state_path, state)

                previous = ""
                for unit, candidate in zip(units, candidates):
                    evaluation_id = task_id + ":" + unit["unit_id"]
                    raw_path = cell / ("judge-" + unit["unit_id"] + ".json")
                    retained = state["evaluations"].get(evaluation_id)
                    if retained is not None and not raw_path.exists():
                        raise ValueError(
                            "Evaluation receipt is missing; do not regenerate an approved judgment"
                        )
                    if raw_path.exists():
                        raw = json.loads(raw_path.read_text())
                        assessment = restore_evaluation(
                            raw,
                            corpus,
                            unit["unit_id"],
                            candidate,
                            previous_candidate=previous,
                            judge=_judge_pipeline(plan, label)["roles"]["audit"],
                        )
                        if retained is not None and retained != assessment:
                            raise ValueError(
                                "Retained assessment differs from its raw evaluation receipt"
                            )
                    else:
                        token = activate_pipeline(
                            _judge_pipeline(plan, label),
                            job_id=scope + ":evaluation:" + evaluation_id,
                            credentials=credentials,
                            store=store,
                            experiment_scope=scope,
                            parent_scope=parent_scope,
                            roles=("audit",),
                        )
                        try:
                            assessment = await evaluate_candidate(
                                corpus,
                                unit["unit_id"],
                                candidate,
                                previous_candidate=previous,
                                experiment_scope=scope,
                                response_sink=partial(write_private, raw_path),
                            )
                        finally:
                            await close_pipeline(token)
                    state["evaluations"][evaluation_id] = assessment
                    _replace(state_path, state)
                    previous = candidate["text"]
        all_candidates = [
            c
            for task in plan["order"]
            for c in json.loads(
                (
                    directory
                    / (task["variant"] + "-" + task["sequence"])
                    / "accepted.json"
                ).read_text()
            )
        ]
        sample, key = render_blind_sample(
            corpus, all_candidates, split=plan["split"], seed=plan["seed"]
        )
        for name, content in (
            ("candidates.json", all_candidates),
            ("blind.html", sample),
            ("answer-key.json", key),
        ):
            if not (directory / name).exists():
                write_private(directory / name, content)
            else:
                retained = (directory / name).read_text()
                if name.endswith(".json"):
                    retained = json.loads(retained)
                if retained != content:
                    raise ValueError(
                        "Published benchmark artifact differs from its frozen evidence"
                    )
        state["status"] = "completed"
        state.pop("stop_reason", None)
        return state
    except BaseException as exc:
        state["status"] = "needs_review"
        state["stop_reason"] = getattr(exc, "reason", type(exc).__name__)
        raise
    finally:
        state["budget"] = budget.snapshot(scope)
        if parent_scope:
            state["pilot_budget"] = budget.snapshot(parent_scope)
        state["elapsed_seconds"] = (
            state.get("elapsed_seconds", 0) + time.monotonic() - execution_started
        )
        _replace(state_path, state)
        _replace(
            directory / "report.json",
            execution_report(plan, corpus, state, store, scope),
        )
        manager.close()


def baseline_example(corpus):
    """Explicit starter configuration; not inferred evidence of a past job."""
    book = corpus["book"]
    return {
        "file_type": "txt",
        "llm_provider": "deepseek",
        "model": "deepseek-v4-pro",
        "source_language": book["source_language"],
        "target_language": book["target_language"],
        "prompt_options": {
            "editorial_mode": "book_profile",
            "profile_id": book["profile"].get("profile_id"),
            "target_locale": book["target_locale"],
            "refine": True,
        },
    }
