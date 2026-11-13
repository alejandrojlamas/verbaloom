# Experimental localized EPUB repair

## Scope

The existing final EPUB audit can now request checksum-bound patches instead of
another complete candidate. This is an opt-in repair serialization experiment,
not a replacement audit, a new pipeline, or the complete optimized C/D strategy.
The compatible default, model roles, accepted-candidate selection and publication
gates are unchanged. TXT/DOCX and other repair entrypoints are not switched by
this EPUB-specific option.

The motivating code observation was `_repair_audited_candidate` asking a model to
return the whole candidate even for a local finding. Its instruction to preserve
unaffected prose was not an enforced edit boundary. The new mode assembles only
validated edits into the current candidate; unaffected spans are kept exactly.

## Configuration

An existing complete `pipeline` snapshot accepts:

```json
{"epub_repair_policy": "localized_v1"}
```

This object is a field update, not a complete job request. For example, prepare a
validated snapshot through the existing configuration API, without making calls:

```python
from src.core.pipeline_config import apply_pipeline, pipeline_preset

config = {
    "file_type": "epub",
    "llm_provider": "deepseek",
    "model": "deepseek-v4-pro",
    "prompt_options": {"refine": True, "audit_entire_book": True},
}
pipeline = pipeline_preset("compatible", config)
pipeline["epub_repair_policy"] = "localized_v1"
apply_pipeline(config, pipeline)
```

Keep the job's actual languages, profile, glossary, model selections and budget;
the example is not a recommendation to change them. The existing job API persists
the validated pipeline. The XHTML entrypoint passes the effective policy into its
existing audit/repair loop. No new paid CLI or frontend toggle is claimed here.

For rollback, set `epub_repair_policy` to `compatible` on a deliberate future job
configuration or an explicitly replaced resume configuration. Do not delete
checkpoints. This affects new proposals, not the validity of already approved
units. A response received under the old selection is still decoded using its
saved contract and audited, rather than charged again. No running service was
restarted to enable this option.

## Edit Contract

`src/core/localized_repair.py` freezes the unit ID, exact source and candidate
SHA-256, findings, approved glossary/context and a contract checksum. The model
copies identifiers; it is not expected to calculate hashes or character offsets.
Source and target text blocks get stable IDs. Paragraph separators, placeholders,
internal block markers and existing HTML delimit the patchable spans.

Each edit names an existing finding, target block, unique exact anchor, operation
and replacement, plus an exact quotation in a named source block. Operations are
replace, insert_before and insert_after; deletion is an empty replacement. The
runtime derives offsets from the unique anchor. It rejects absent or ambiguous
anchors, unknown findings/blocks, wrong checksums, overlapping edits, unsupported
fields, duplicate JSON keys, nonfinite values and structural/control-character
injection. No Unicode or whitespace normalization runs over untouched text.

All edits validate before any are applied. There are at most 32 patches and a
bounded JSON response. Empty patches is valid and reaches the existing repeated-
candidate stop condition; it is not semantic approval. A full rewritten response
is invalid in patch mode and never triggers an automatic full-translation fallback.

Anchoring a quotation establishes provenance, not that a correction is justified.
The existing target-language and placeholder gates still run. The complete
candidate then undergoes the existing source-aware final audit, including its
available document context. A valid JSON patch can still worsen a translation:
the existing best-candidate rule rejects non-improvement and retains prior work.

## Persistence and Recovery

The unit's existing `audit_repair.localized_patches` journal retains the response,
contract, response state/request ID, derived edits, validation status and checksum.
It is written before applying the proposal to the selected translation or asking
for the next audit. This is private checkpoint content, not a public log artifact.

If the worker stops after that save, resume verifies the envelope, recomputes the
contract and processes the saved response without another repair request. This
also applies to a saved malformed response: it consumes the original attempt,
not a replenished allowance. Existing lifetime attempt limits remain in force.
Corrupt/stale envelopes stop visibly; they cannot approve or overwrite the unit.
The final candidate and pending assessment use the existing durable journal.

A crash before receipt/checkpoint persistence remains an uncertain provider
attempt. This feature does not promise exactly-once billing. Existing usage-ledger
and budget reservations continue to record that uncertainty.

## Validation and Limits

```sh
venv/bin/python -m pytest -q tests/unit/test_localized_repair.py tests/unit/epub/test_localized_audit_repair.py
venv/bin/python -m pytest -q
venv/bin/ruff check src/core/localized_repair.py src/core/pipeline_config.py tests/unit/test_localized_repair.py tests/unit/epub/test_localized_audit_repair.py
venv/bin/ruff check --select F821 src/core/epub/xhtml_translator.py src/core/epub/unit_contract.py
git diff --check
```

Tests exercise the actual patch validator, EPUB repair loop and SQLite/file
checkpoint manager, with simulated model responses. They cover accepted and
non-improving patches, invalid/truncated responses, worker loss after receipt,
resume without regeneration, corrupt journal data, stable counters, Unicode,
marker/paragraph integrity, runtime option forwarding and compatible defaults.
They do not prove literary quality, model JSON reliability or lower API cost.
For small corrections, JSON metadata may exceed the size of a whole short block;
measure marginal cost rather than assuming patches are always cheaper.

This version cannot insert/remove paragraphs or edit across inline boundaries in
one patch. Multiple supported spans can be edited atomically; a defect requiring
structural regeneration remains unresolved and blocks publication. A calibrated
larger-unit escalation policy is still required. Cross-unit neighbor/context
invalidation, other formats, complete A/B/C/D execution and paid/human comparison
remain part of the full objective, not completed by these tests.
