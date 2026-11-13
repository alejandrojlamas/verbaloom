"""Durable stage receipts inside the existing generic unit checkpoint.

Received drafts are not completed chunks or accepted translation-memory entries.
Reservations survive worker loss; an unknown result may have been billed. This
is local idempotence, not an exactly-once guarantee from a model provider.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict

from src.core.editorial_quality import QualityDecision, QualityIssue
from src.core.epub.unit_contract import _dependency_value, with_profile_dependency
from src.core.fidelity_supervisor import FidelityDecision
from src.core.literary_continuity import export_literary_continuity_state
from src.core.llm.exceptions import ProviderActionRequiredError
from src.core.recovery_policy import recovery_limit

JOURNAL_KEY = "generic_stage_journal"
JOURNAL_VERSION = 1
_TRANSIENT_OPTIONS = {
    "_fidelity_report",
    "_editorial_quality_report",
    "_candidate_results",
    "_source_guard_refs",
    "_source_guard_refs_loaded",
    "_last_target_language_gate_rejection",
    "_combined_quality_audit_records",
    "generic_stage_max_attempts",
    "automatic_failure_recovery",
    "automatic_failure_recovery_cycle",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            _dependency_value(value),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def stage_contract(
    units,
    *,
    source_language,
    target_language,
    model,
    provider,
    options,
    parameters=None,
):
    options = with_profile_dependency(options) or {}
    return digest(
        {
            "version": JOURNAL_VERSION,
            "units": [
                (u.unit_id, u.content, u.context_before, u.context_after) for u in units
            ],
            "source_language": source_language,
            "target_language": target_language,
            "model": model,
            "provider": provider,
            "parameters": parameters or {},
            "options": {
                k: v for k, v in options.items() if k not in _TRANSIENT_OPTIONS
            },
        }
    )


def _encode(result, kind):
    if kind == "text":
        if result is not None and not isinstance(result, str):
            raise TypeError("Translation stage did not return text")
        return result
    if kind == "text_pair":
        return _encode(result[0], "text")
    decision = result[0]
    if kind == "fidelity":
        return decision.to_dict()
    if kind == "quality":
        return asdict(decision)
    raise ValueError("Unknown stage result kind")


def _decode(value, kind):
    if kind in {"text", "text_pair"} and (
        not isinstance(value, str) or not value.strip()
    ):
        raise ValueError("Saved text stage is empty")
    if kind == "text":
        return _encode(value, kind)
    if kind == "text_pair":
        return _encode(value, "text"), None
    if not isinstance(value, dict) or type(value.get("accepted")) is not bool:
        raise ValueError("Saved audit lacks an explicit decision")
    if kind == "fidelity":
        return FidelityDecision.from_dict(value), None
    data = dict(value)
    data["issues"] = [QualityIssue(**item) for item in data.get("issues", [])]
    return QualityDecision(**data), None


class UnitStageJournal:
    def __init__(
        self,
        manager,
        job_id,
        index,
        unit,
        contract,
        *,
        saved=None,
        options=None,
        log=None,
    ):
        self.manager, self.job_id, self.index, self.unit = manager, job_id, index, unit
        self.log = log
        self.limit = min(
            10, recovery_limit((options or {}).get("generic_stage_max_attempts"), 3)
        )
        self.data = (
            deepcopy(saved)
            if saved is not None
            else {
                "version": JOURNAL_VERSION,
                "contract": contract,
                "stages": {},
            }
        )
        if (
            not isinstance(self.data, dict)
            or self.data.get("version") != JOURNAL_VERSION
            or self.data.get("contract") != contract
            or not isinstance(self.data.get("stages"), dict)
        ):
            self.stop(
                "checkpoint_dependencies_changed",
                "The retained unit stage contract is incompatible; restore its original configuration.",
            )

    def stop(self, reason, message):
        raise ProviderActionRequiredError(message, reason=reason, provider="checkpoint")

    async def call(self, stage, operation, *args, kind="text", **kwargs):
        inputs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in {
                "llm_client",
                "client",
                "log_callback",
                "runtime_state",
                "prompt_options",
            }
        }
        inputs["args"] = args
        inputs["options"] = {
            k: v
            for k, v in (kwargs.get("prompt_options") or {}).items()
            if k not in _TRANSIENT_OPTIONS
        }
        if "runtime_state" in kwargs:
            inputs["continuity"] = export_literary_continuity_state(
                kwargs["runtime_state"]
            )
            # Providers may observe provisional/rejected candidates internally.
            # Only the engine's accepted result can update shared continuity.
            kwargs["runtime_state"] = deepcopy(kwargs["runtime_state"])
        result = await self.run(
            stage, inputs, lambda: operation(*args, **kwargs), kind=kind
        )
        if kind == "fidelity":
            report = (kwargs.get("prompt_options") or {}).get("_fidelity_report")
            decision = result[0]
            if report is not None and not any(
                (r.phase, r.chunk_index, r.candidate_hash)
                == (decision.phase, decision.chunk_index, decision.candidate_hash)
                for r in report.records
            ):
                report.add(decision)
        return result

    def persist(self):
        if not self.manager.db.save_chunk(
            self.job_id,
            self.index,
            self.unit.content,
            None,
            {**self.unit.metadata, JOURNAL_KEY: self.data},
            status="pending",
        ):
            self.stop(
                "checkpoint_write_failed",
                "Could not persist a unit stage; no further requests will be started.",
            )

    async def run(self, stage, inputs, operation, *, kind="text"):
        key = digest({"inputs": inputs, "kind": kind})
        entries = self.data["stages"].setdefault(stage, [])
        if not isinstance(entries, list) or any(
            not isinstance(e, dict) for e in entries
        ):
            self.stop(
                "checkpoint_corrupt", "The saved stage attempt history is invalid."
            )
        for number, entry in enumerate(entries, 1):
            if (
                type(entry.get("attempt")) is not int
                or entry["attempt"] != number
                or not isinstance(entry.get("key"), str)
                or len(entry["key"]) != 64
            ):
                self.stop(
                    "checkpoint_corrupt", "The saved stage attempt counter is invalid."
                )
            if entry.get("state") not in {"running", "failed", "completed"}:
                self.stop(
                    "checkpoint_corrupt",
                    "The saved stage attempt has an invalid state.",
                )
        for entry in reversed(entries):
            if entry.get("state") == "completed" and entry.get("key") == key:
                try:
                    if entry.get("result_hash") != digest(entry["result"]):
                        raise ValueError("Receipt checksum mismatch")
                    result = _decode(entry["result"], kind)
                except (KeyError, TypeError, ValueError) as exc:
                    self.stop(
                        "checkpoint_corrupt",
                        f"Cannot reuse a damaged {stage} receipt: {type(exc).__name__}.",
                    )
                if self.log:
                    self.log(
                        "checkpoint_stage_reused",
                        f"Reused {stage} for unit {self.index + 1} without a model call.",
                    )
                return result
        if len(entries) >= self.limit:
            self.stop(
                "stage_attempts_exhausted",
                f"Unit {self.index + 1} exhausted its {stage} attempt allowance; retained candidates require review.",
            )
        entry = {"key": key, "state": "running", "attempt": len(entries) + 1}
        entries.append(entry)
        self.persist()
        try:
            result = await operation()
        except BaseException as exc:
            entry.update(state="failed", error_type=type(exc).__name__)
            reason = getattr(exc, "reason", None)
            if isinstance(reason, str):
                entry["reason"] = reason
            self.persist()
            raise
        encoded = _encode(result, kind)
        if kind in {"text", "text_pair"} and (encoded is None or not encoded.strip()):
            entry.update(state="failed", reason="empty_stage_result", result=encoded)
            self.persist()
            if self.log:
                self.log(
                    "stage_empty_result",
                    f"Unit {self.index + 1} returned no text in {stage}; retained as a failed attempt.",
                )
            return (None, None) if kind == "text_pair" else None
        entry.update(state="completed", result=encoded, result_hash=digest(encoded))
        self.persist()
        return result
