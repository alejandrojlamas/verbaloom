# Private editorial benchmark

## Current boundary

The corpus, candidate capture, blind reader, structured evaluator, synthetic
controls and A/B/C/D TXT executor are implemented. The executor uses the production
`GenericTranslator`, profile renderer, providers, request ledger, checkpoints and
translation memory. Only paragraph boundaries are frozen by a small TXT adapter.
The [C/D source-audited policy](SOURCE_AUDITED_STRATEGY.md) remains experimental.
No paid comparison or human literary evaluation has run.

The existing sample API is not used as a baseline. This executor measures the
production TXT translation/refinement workflow, not the complete native EPUB or
DOCX resource/publication workflow. Reflowed excerpts cannot prove image, note,
navigation or cover preservation. Those acceptance tests remain separate.

## Freeze one pilot and its comparisons

Create **one parent pilot**, initially USD 5 gross, before freezing any child
comparison. Every book and split in this pilot must use that same directory.
The parent's ledger includes generation, independent judges and every HTTP
retry across all children. A child's allowance never increases the parent cap.

Supply an explicit baseline JSON with `file_type: txt`, `llm_provider: deepseek`,
the actual intended model, corpus source/target languages and `prompt_options`
containing its `profile_id`, exact `target_locale` and requested `refine` policy.
Optional existing translation prompt options are retained. Do not call a starter
example evidence of a historical job. Transforms and chained `refine_after` runs
are rejected: they require their own operation-specific experiment.

```sh
venv/bin/python -m scripts.editorial_benchmark pilot-init \
  --output "$PILOT_DIR" --limit-usd 5 --max-calls 1000
venv/bin/python -m scripts.editorial_benchmark plan "$CORPUS_JSON" \
  --baseline "$BASELINE_JSON" --split tuning --seed frozen-seed \
  --pilot "$PILOT_DIR" --limit-usd 5 --max-calls 300 --output "$EXPERIMENT_DIR"
venv/bin/python -m scripts.editorial_benchmark plan "$CORPUS_JSON" \
  --baseline "$BASELINE_JSON" --split heldout --seed reserved-seed \
  --pilot "$PILOT_DIR" --limit-usd 5 --max-calls 300 --output "$HELDOUT_DIR"
venv/bin/python -m scripts.editorial_benchmark pilot-status "$PILOT_DIR"
venv/bin/python -m scripts.preflight_openai --reasoning-effort none
venv/bin/python -m scripts.editorial_benchmark run "$EXPERIMENT_DIR" --allow-paid
venv/bin/python -m scripts.editorial_benchmark run "$EXPERIMENT_DIR" --allow-paid --resume
```

`plan` makes no HTTP requests. It freezes code, runtime settings, corpus, full
profile/glossary, A/B/C/D options and seeded sequence order. A/C have identical
roles; B/D have identical roles. The language pair and regional variant never
change. Each whole sequence executes contiguously, with its own private memory.

`run` is paid opt-in. Both provider keys must be present in the existing ignored
`.env` or environment (`OPENAI_API_KEY` and `DEEPSEEK_API_KEY`), not arguments or
experiment JSON. Official OpenAI model-visibility preflight must succeed first;
it cannot prove generation access or account balance. All generation, editorial
checks, independent evaluation and HTTP retries share the parent ledger.
Reservations enforce job, experiment and parent limits in one SQLite transaction
before dispatch. Unknown in-flight charges retain their full allowance; a timeout
does not make another retry free. Bounds currently use the documented full context
window, not an assumed token estimate. With USD 5, an uncertain Terra request can
prevent another request even when observed spend is small. This stop requires
reconciliation, not an automatic budget reset or a zero-cost assumption.

**Each experiment still contains one corpus and one split.** Additional books,
tuning, heldout and future ablation runs must bind to the same parent using
`--pilot`. Creating a different parent would start a different allowance; it is
not a way to continue this authorized USD 5 pilot. Limits are editable at initial
creation and frozen afterward. Missing/replaced ledgers, changed manifests,
copied child directories and lost started-state files fail before dispatch.
Do not move, rebind, recreate or restore only part of a spent experiment. Keep
its parent and all child evidence together. Synthetic tests cannot use their
allowances for observed paid execution. The unresolved Czech profile and missing
OpenAI credential still block the requested full paid pilot.

An exclusive OS file lock prevents two runners. Resume validates retained code,
configuration, checkpoints, ledger, raw judgments and final artifacts. It reuses
approved translations and checksum-bound judgments; missing/corrupt retained
evidence stops instead of paying for silent regeneration. Generic stage receipts
preserve received drafts, refinements and audit decisions separately from accepted
units; resume replays matching receipts and repeats only unfinished stages within
their durable attempt allowance. A result lost before its receipt is persisted
can still require another billed call. This is not exactly-once billing.

The parent directory contains `pilot.json` and the shared `usage.db`. Each private
experiment directory contains `plan.json`, `corpus.json`, `jobs.db`,
`state.json`, per-sequence `accepted.json`, raw `judge-*.json`, intermediate
`candidate-*.json`, `candidates.json`, `blind.html`, a separate `answer-key.json`
and `report.json`. Reports distinguish observed, estimated and unknown costs;
break down HTTP calls/tokens/timing by variant, stage and unit; and identify cache
read/write and reasoning subsets without adding them twice. Cash/credits remain
unknown. Summed HTTP time is not wall-clock time. Incomplete evidence has no final
cost-per-approved-word ratio. Child reports filter HTTP events by experiment
before applying the event limit; sibling judges have distinct job identities.
`pilot-status` reads parent and child totals from one consistent database snapshot.
Existing frozen profile preparation is reused and
reported as excluded historical cost, not as newly free preparation.

A/C are externally evaluated by Terra; B/D by DeepSeek Pro. Neither judge can be
a generator/repairer of its candidate. Different judges across provider groups
are a comparison confound, explicitly recorded in the report; blind human
evaluation remains required. No provider superiority follows from these scores.

## Prepare private sources

Run from the installed checkout, with its virtual environment:

```sh
venv/bin/python -m scripts.editorial_benchmark inventory "$SOURCE_EPUB" --start 100 --count 30
venv/bin/python -m scripts.editorial_benchmark prepare "$SOURCE_EPUB" \
  --profile "$PROFILE_ID" --source-language English --target-language Spanish \
  --tuning-start 150 --heldout-start 850 --sequence-length 5 \
  --output "$HOME/.local/share/VerbaLoom-benchmarks/new-pilot/book-one"
```

Positions refer to the inventory, not page numbers. Choose actual adjacent ranges
within one section. The example positions are not universal book positions.
The profile supplies its existing target locale; there is no regional default.
The snapshot includes its editorial content and glossary, without a machine root.
Source SHA-256, exact unit locators, languages, locale, profile and context are
bound into a reproducible corpus digest. Changed content needs a new experiment.

TXT sources are also supported. A TXT sequence establishes adjacent paragraphs,
not a verified scene/chapter boundary. Malformed EPUB XML is rejected; recovered
partial extraction must not silently become the evaluation corpus. No original
file is modified. Prepare reads the source hash before and after extraction.

Use `--unresolved-profile` instead of `--profile` only to preserve material whose
historical configuration has not been recovered. The locale remains unknown, not
invented. The evaluator refuses that corpus before HTTP. This is a staging aid,
not a valid paid-comparison configuration.

Each sequence belongs wholly to tuning or heldout. Duplicate source text across
splits, overlapping selections, reordered positions, changed context or a sequence
crossing sections are rejected. Neighbor context stays inside its sequence. The
corpus is limited to 60 units, requires both splits and at least two sequences.
This prevents accidental whole-book evaluation, not a guarantee of sufficient
literary coverage. Human category/coverage annotation remains necessary.

Output directories must be outside the repository and private (0700); files are
0600. Artifact publication uses an exclusive atomic link after fsync. Existing
artifacts are never overwritten. No private extracts are included in these docs.

## Capture and evaluate

`src.core.candidate_result.capture_candidates(sink)` captures complete existing
`CandidateResult` records in the current task. Normal production metadata remains
bounded and text-free. The synchronous sink should write into the private
experiment. It receives original/draft/refinement/repair records only where the
existing pipeline emits them; complete all-format stage coverage remains pending.
The executor binds each record and HTTP event to the corpus unit ID. Separate
async tasks do not share sinks accidentally.

`src.core.benchmark_judge.evaluate_candidate` uses the existing role runtime and
tracked provider. It requires an already configured `experiment:*` scope and
private usage store. The TXT executor puts generators, judges and retries in the
**same child scope and shared parent scope**, initially USD 5 gross in total with
a finite HTTP-attempt cap. Existing atomic request reservations include uncertain
attempts; they are never treated as free. Paid execution without a parent is
rejected; standalone execution remains available only to simulated integration
fixtures, not as a CLI bypass.

The audit role must differ from every model listed in candidate generation/repair
provenance. DeepSeek aliases cannot masquerade as independent judges. There is no
automatic provider fallback. Source, candidate, profile and adjacent context are
data, and model labels/preparation metadata are omitted from the judge prompt.

Evaluation covers fidelity, omissions/additions, voice/rhythm, characters,
naturalness, glossary, continuity and structure separately. Findings require
exact source/target character spans, type, severity, explanation and correction.
A critical omission cannot disappear into an averaged score. Missing dimensions,
unlocated concerns, conflicting verdicts, duplicate fields, incomplete/refused
responses and malformed JSON never count as approval. An unassessable dimension
requires review. Empty findings and a genuine no-change pass are valid.

The raw response is sent to the private persistence callback before validation.
No editorial format-retry loop is added. HTTP attempts remain governed by the
existing provider deadline, ledger and shared budget. A retained raw receipt is
revalidated against corpus, candidate, prior translated context, judge and prompt
version before reuse, even when the summary says completed. A saved invalid
judgment is not repeatedly requested. This does not promise exactly-once billing.

## Blind reader and marginal evidence

Final records use `unit_id`, `variant` (A/B/C/D), `phase: final`, `text` and
`evidence` (`provider_response`, `simulated` or `unverified`). Model and other
identifying fields may exist in private records but are not rendered.

```sh
venv/bin/python -m scripts.editorial_benchmark blind "$CORPUS_JSON" "$CANDIDATES_JSON" \
  --split heldout --seed frozen-seed \
  --output "$HOME/.local/share/VerbaLoom-benchmarks/new-pilot/blind-heldout"
```

The renderer requires every comparison cell. The answer key is a separate file;
candidate order is seeded and rotated between sequences, while labels remain
consistent inside a sequence for continuity review. The HTML escapes text,
blocks scripts/network resources and works without a server. It labels simulated
or unverified inputs explicitly. This is a static reading artifact, not the
production app's reactive seven-locale settings UI or a completed scoring UI.

`pass_delta` compares located findings on the same source and exact candidate
hashes. It reports removed/new/persistent findings, known marginal cost and time.
Text changes without changed findings are **unclassified**, not automatically
cosmetic or improvements. These remain differences in judge reports, not human
proof that an error was corrected. Unknown costs remain null.

## Calibrate without spending

```sh
venv/bin/python -m scripts.editorial_benchmark calibrate-local \
  --output "$HOME/.local/share/VerbaLoom-benchmarks/new-calibration"
venv/bin/python -m pytest -q tests/unit/test_editorial_benchmark.py \
  tests/unit/test_benchmark_judge.py tests/unit/test_benchmark_artifacts.py
venv/bin/python -m pytest -q tests/unit/test_benchmark_execution.py \
  tests/unit/test_benchmark_pilot.py tests/unit/test_spend_budget.py \
  tests/unit/test_inline_refinement_pipeline.py
```

The committed fixture is synthetic, not a private excerpt. Its six mutations are
negation inversion, paragraph omission, changed name, duplication, flattened
voice and damaged markers; two legitimate translations test false positives.
The calibration records actual deterministic `assess_fidelity` results, not mock
semantic verdicts. A warning for some other reason is not confirmed detection of
the intended defect. The separate `negative_control_summary` measures expected
finding kinds only when structured evaluations are supplied, distinguishing
misses, false positives and unexecuted evaluations.

On 2026-11-12, the local detector flagged the omitted paragraph with a length
warning and rejected damaged markers. It left negation, changed name, duplication
and voice flattening unflagged. Both legitimate controls were unflagged. This
small synthetic test argues against equating mechanical acceptance with semantic
quality; it does not estimate real-book sensitivity or provider quality.

Only approved glossary entries become rules in the evaluation prompt. Pending
suggestions and glossary reviewer/model attribution remain in the private frozen
snapshot but are excluded from the judge's editorial context.

## Remaining acceptance work

Recover the Czech job's historical editorial settings, review/expand category
coverage and calibrate C/D without disabling required semantic gates. Extend
stage/dependency recovery verification to all remaining production paths.
Add full prompt-composition and marginal-pass evaluations, negative-control
semantic judgments, heldout human evaluation and structured resource publication
E2E. OpenAI preflight currently lacks credentials. Do not run the full paid pilot
until access, account availability and its complete budget scope are verified.
