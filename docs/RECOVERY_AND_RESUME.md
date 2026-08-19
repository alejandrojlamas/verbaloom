# Recovery, Resume, and Selective Repair

Long-book jobs are expected to survive process restarts, network failures, rate
limits, browser refreshes, and local model errors without repeating approved
work.

## Durable State

The checkpoint database stores:

- immutable run ID and source path
- provider, model, locale, profile, glossary, and quality configuration
- expected chunk count and progress
- original and accepted text per chunk
- candidate/audit metadata and retry count
- failed checkpoint indices and rejected candidate evidence
- staging and intended output paths
- final quality-gate proof and report directory

Strict jobs whose chunks are complete enter `validating`, not `completed`.
`completed` is persisted only after the whole-book gates pass.

## Resume

```bash
./translator.py resume RUN_ID
```

Resume loads the original job configuration and preserved source, restores
approved chunks, starts at the earliest unresolved index, and skips later chunks
already present in the checkpoint. It does not use the earliest failure as a
reason to discard subsequent successful work.

If the application process disappears while a job is genuinely `running` or
`validating`, the next server process automatically resumes the most recent job
from its durable checkpoint. It starts at most one job this way. A manual pause,
insufficient credits, a missing preserved source, or an ordinary historical
error is never revived automatically.

Completing all text chunks does not make a checkpoint non-resumable. If assembly,
publication, or whole-book validation failed, resume starts at the end of the
text checkpoint and repeats only those final phases; it does not translate the
book again.

## Selective Repair

```bash
./translator.py repair RUN_ID
```

When validation rejects units, only their translated values are cleared. The
rejected text and complete issue evidence remain in `chunk_data`. Repair mode
uses a stricter source-aware path and then reruns all whole-book gates.

The default maximum repair rounds is two and is configurable. No repair path may
loop indefinitely.

Worker crashes and finalization failures also have independent two-cycle
recovery budgets. Provider throttling has a separate bounded auto-resume budget.
When a budget is exhausted, the checkpoint remains recoverable and the UI stops
claiming that work is active. Insufficient credits always require user action.

## Validation Without Modification

```bash
./translator.py validate RUN_ID
```

This rebuilds the manifest and reports but does not clear failed units. For an
external source/output pair:

```bash
./translator.py validate \
  --source original.epub \
  --output translated.epub \
  --source-language English \
  --target-language Spanish \
  --target-locale es-MX
```

## Publication and Quarantine

A blocked CLI run remains under:

```text
data/quality_runs/<run_id>/staging/
```

It is never copied over a previous valid output. A blocked web artifact receives
a `[partial]` prefix and remains downloadable for diagnosis, but the job remains
partial and resumable.

Intermediate manifests and reports are kept by default. Set
`export.keep_intermediate_files: false` only when traceability is not required.

## Cancellation

User cancellation is cooperative. The active provider request is allowed to
return or time out, then the current safe checkpoint is preserved. The final
publisher is never invoked for an interrupted run.

Cancellation intent is written to the checkpoint immediately. A server restart
therefore cannot mistake a deliberate pause for a crashed active worker.

## Common Recovery Cases

### Source language remains

Open `language_report.json`, identify the unit/checkpoint indices, run `repair`,
and validate again. Do not manually replace the final file while the run is
blocked.

### Entity mismatch

Review `entity_diff.json`. Correct the approved book glossary when the target
form is intentional; otherwise use entity-preserving repair.

### EPUBCheck failure

Inspect `structure_report.json` and the EPUBCheck output. Package errors require
reconstruction, not prose retranslation.

### Missing preserved source

Resume cannot prove identity without the original source. Restore the exact
source file or start a new run. The tool fails clearly instead of guessing from
the translated artifact.

### Report says artifact alignment

The run had no exact checkpoint/native manifest. Validation aligned source and
output blocks by order and reports this as a limitation. Any count difference is
blocking; high-risk deliveries should be rerun through the normal persisted
pipeline.
