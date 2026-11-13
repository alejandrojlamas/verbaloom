# OpenAI integration and verifiable pipeline optimization

## Status and scope

Implementation checkpoint. This document tracks the full user specification of
2026-11-12; provider integration alone is not completion. The user subsequently
requested closure, deployment to the existing Tailscale-connected installation,
and publication to its GitHub project. That authorizes this delivery, not a paid
pilot, default-provider change, or deletion of runtime state. See
[delivery summary](DELIVERY_2026-11-12.md) for verified scope and remaining work.

| Requirement | Evidence needed | Status |
| --- | --- | --- |
| 1. Preserve fidelity and existing DeepSeek configuration | Regression tests, unchanged defaults, editorial evaluation | Pending |
| 2. Inspect/protect current checkout and state | Root/commit/status/runtime inventory | Initial inventory complete |
| 3. Trace every stage and reproduce stalls | Stage map, calls/tokens/time/retries, baseline characterization | In progress |
| 4. Official Luna/Terra integration | Documented parameters, response contracts, account preflight | In progress |
| 5. Independent role and review routing | Reproducible A/B/C/D presets, compatibility rollback | Seven roles, compatible A/B and opt-in source-audited C/D implemented; real/human comparison pending |
| 6. Complete accounting and atomic budgets | Per-attempt ledger, pricing provenance, concurrent reservations | Job/profile, experiment and shared multi-book/split pilot reservations implemented; UI and both CLI budget configuration verified; complete live cost presentation pending |
| 7. Editorial/reliability benchmark | Private 20-unit split corpus, adjacent sequences, blind candidates, negative controls, <= USD 5 | Production TXT A/B/C/D executor with simulated HTTP, frozen inputs and replay checks implemented; full pilot and paid/human evidence pending |
| 8. Evidence-driven passes | Structured findings, sampled/risk audit, versioned local repair, convergence | Malformed judgments fail closed; localized EPUB patches and full-semantic C/D implemented; calibrated risk sampling pending |
| 9. Context, segmentation, cache, memory | Measured prompt composition, reuse isolation, ablations | Accepted-only generic context and ordered adjacent recovery implemented; complete dependency invalidation and measurements pending |
| 10. Persistent bounded recovery | Fault injection, no reset on resume, locks, distinct waiting/failure reasons | Generic stage receipts, finite attempts, ownership and cancellation tested; full lifecycle/profile/plain-route audit pending |
| 11. Publication integrity | Full coverage, resources, EPUBCheck, atomic publish, no source fallback | Resource-rich native EPUB A/B/C/D and resume pass actual EPUBCheck; web worker and both processing CLIs stage through gates and share checksum-bound, non-overwriting publication; standalone export rejects stale approval; complete cross-format CLI coverage remains pending |
| 12. Configuration and UI | Effective role/preset/budget persistence, reactive i18n, truthful progress | Seven-role settings, budget, scoped preflight and frozen resume verified in desktop/mobile browser; shared CLI configuration/runtime added; complete stage/cost display pending |
| 13. Validation | Provider/unit/integration, TXT + structured artifact, desktop/mobile browser | 3,147 regression tests passed; TXT/PDF and native EPUB/resource/resume integration use fake HTTP and actual artifact validators; browser checks cover settings/start/pause/resume/errors and seven locales, with synthetic workers |
| 14. Delivery and decision | Audit, measured A/B/C/D results or explicit access limits, exact run/rollback instructions | Pending |

## Initial authoritative inventory

- Active checkout: `~/.local/share/VerbaLoom`, not the conversation working
  directory or the historical private archive. The older
  `~/.local/share/TranslateBooksWithLLMs` tree also exists and is not being edited.
- Branch: `main`; initial HEAD: `5eddf0018bf0e7cdeb9e8bcb274d2b84192c321f`
  (`Reduce redundant DeepSeek Pro review spend`), one local commit ahead of origin.
- Initial tracked worktree: clean.
- Runtime: PID 820, Python 3.12 virtual environment, loopback port 5050;
  `/api/health` returned OK and `/api/translations` returned an empty list.
  The running process has not been restarted.
- Environment: `LLM_PROVIDER=deepseek`, `DEFAULT_MODEL=deepseek-v4-pro`,
  `DEEPSEEK_MODEL=deepseek-v4-pro`, `MAX_TOKENS_PER_CHUNK=450`,
  `REQUEST_TIMEOUT=900`, `MAX_TRANSLATION_ATTEMPTS=3`,
  `AUTO_PAUSE_ON_RATE_LIMIT=false`, `DEEPSEEK_DISABLE_THINKING=true`.
  `LLM_CLOUD_OPERATION_TIMEOUT` is not set in `.env`; its code default must also
  be included when interpreting timeout behavior. No values were changed.
- DeepSeek credential present; OpenAI credential absent. No secrets recorded.
- Read project guidance and architecture, pipeline, QA, recovery, example env,
  README and characterization-test documentation. No project AGENTS.md found.

## Baseline evidence and early findings

Baseline command (no live provider traffic):

```sh
venv/bin/python -m pytest -q tests/unit/test_openai_compatible_circuit_breaker.py tests/unit/test_llm_provider_retry_manager.py tests/unit/test_provider_model_routes.py tests/unit/test_usage_tracking.py tests/unit/test_pricing_estimator.py tests/unit/test_llm_request_deadline.py
```

Result before implementation: **38 passed in 1.85s**.

Observed code defects, distinct from runtime claims:

- `OpenAICompatibleProvider.generate` ignores `finish_reason` and hardcodes
  `was_truncated=False`. A partial response can therefore bypass that guard.
- The same provider retries permanent HTTP failures, returns empty successful
  content, omits cache/reasoning usage, and does not explicitly configure effort.
- Official-endpoint detection uses substring matching instead of URL origin.
- `UsageTrackingProvider` records one event per `generate`, not each HTTP attempt;
  nested retry consumption and ambiguous timeout costs are not fully observable.
- Circuit breakers are actually per provider instance, despite `.env.example`
  describing one shared across providers.
- The existing USD max-cost publication gate is not a pre-request budget lock.
- Existing provider-refusal splitting and generic exception fallbacks need a
  dedicated audit before asserting compliant review-required behavior.

## Official sources checked on 2026-11-12

- https://developers.openai.com/api/docs/models/gpt-6-luna
- https://developers.openai.com/api/docs/models/gpt-6.1-terra
- https://developers.openai.com/api/docs/guides/latest-model
- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/docs/guides/batch
- https://api-docs.deepseek.com/quick_start/pricing/

Luna and Terra support Chat Completions for text, explicit `none` and `low`
reasoning, and 128,000 maximum output tokens. Keep the existing Chat Completions
client; no tools or agent loop are needed. Omit sampling parameters for non-none
effort. Documentation is not evidence of account access.

Standard short-context USD per million tokens (input / cached / output): Luna
0.10 / 0.01 / 0.50; Terra 1.00 / 0.05 / 5.00. Cache writes are a separate input
category, not an additive charge. Long-context/tier/retention handling is required
before cost claims or a paid pilot. Current DeepSeek aliases and off-peak/peak
prices match the existing table; do not migrate them silently.

## Implemented slice: provider boundary

Implemented in the existing provider/client stack, not a parallel client:

- Exact HTTPS OpenAI origin detection, Luna/Terra discovery entries, explicit
  effort (`none` default), bounded configurable output allowance, and sampling
  only for `none`. Existing DeepSeek defaults are unchanged.
- Normalized completion finish/status/refusal, token/cache/reasoning/raw usage,
  request/response/model identity, and latency. Truncated responses cannot pass
  the provider convenience translation methods.
- Permanent configuration/access/billing errors stop without provider retries.
  Refusal is a separate review-required reason; it does not enter the legacy
  content-filter splitting path. The existing job stop mechanism persists the
  reason and suppresses automatic resume without deleting the checkpoint.
- OpenAI-compatible HTTP attempts now use the existing usage ledger separately,
  with operation identity and attempt parameters. No duplicate aggregate event.
  Billed empty/refused responses retain usage; cancellation/timeouts without
  usage are explicitly uncertain, not proved free. The subsequent accounting
  slice below adds DeepSeek and durable pre-request budget locks.
- Non-billable official model-visibility preflight, using the existing provider
  HTTP client. A visible model is not a successful generation or a known balance.

The new boundary regressions initially produced **14 failed, 1 passed**, including
truncation, permanent retries, missing usage, effort and URL-origin errors. After
the first correction all 15 passed. Additional tests cover SQLite stop reason,
preserved refused/empty usage, cancellation uncertainty, and concurrent contexts.

An intermediate full regression run returned **2434 passed, 2 skipped,
10 deselected in 22.06s**. These are simulated/regression tests, not a paid
editorial benchmark or physical Android validation. Later additions require a
fresh final suite result below.

Final slice regression on 2026-11-12: **2445 passed, 2 skipped, 10 deselected in
21.68s** (`venv/bin/python -m pytest -q`). `compileall` and `git diff --check`
passed. The original PID/process-session health remained unchanged and healthy.

### Historical ledger observation (read-only)

The latest job with ledger activity contains 2,604 recorded operations, 8,302,197
input tokens, 1,926,323 output tokens and USD 4.6236 in recorded calculated costs.
It used both Flash and Pro, not Flash alone:

| Model | Heuristic phase | Operations | Recorded USD |
| --- | --- | ---: | ---: |
| Pro | Translation, successful | 1179 | 2.8945 |
| Flash | Format repair | 905 | 0.8731 |
| Pro | Format repair | 277 | 0.6665 |
| Flash | Translation | 199 | 0.0556 |
| Pro | Fidelity audit | 42 | 0.1339 |
| Pro | Translation, error | 2 | 0.0000 (usage unknown) |

These are historical ledger labels, not independently classified HTTP calls or
invoices. A zero error-row cost does not prove no charge. The current jobs table
is empty, corroborating the empty live job list; historical configuration must
be recovered from retained reports/backups without modifying production state.

### Non-billable account check

From the repository root, use the existing OpenAI settings field or ignored
`.env` entry `OPENAI_API_KEY=YOUR_API_KEY_HERE`. Do not send a key in chat or put it
in shell arguments. Do not change `LLM_PROVIDER` or the DeepSeek settings.

```sh
venv/bin/python -m scripts.preflight_openai --reasoning-effort none
```

Observed local result: exit 2, `missing_credentials`, `balance: null`,
`generation_tested: false`. No paid requests have been sent. A generation probe
will only run under the shared pilot reservation budget once implemented.

### Remaining work (not optional acceptance gates)

Continue the full stage trace and historical job/usage analysis; characterize
and integrate role/preset persistence; complete versioned tier/cache-write costs
and atomic budgets; apply per-attempt tracing to all participating providers;
calibrate selective audit and convergence with negative controls; build the
private A/B/C/D corpus/runner and blind human sample; implement/reactively test
configuration and progress controls; verify TXT and resource-bearing EPUB/DOCX
end to end; audit publication/resume invariants; record measured conclusions.
No claim of savings, literary superiority, or full-scope completion is made.

## Implemented slice: cost certainty and durable request budgets

The existing `usage.db` now owns additive budget scope/reservation/membership
tables. No second usage ledger, queue, service, or state machine was introduced.
The application service and original database were not restarted or migrated.

- `SpendBudget.configure` persists a stable job or experiment cap and call limit.
  Repeating the same configuration on resume does not reset consumption; changing
  it implicitly is rejected.
- A single SQLite `BEGIN IMMEDIATE` reserves the full request allowance in every
  supplied scope (job plus experiment) before HTTP dispatch. Separate connections
  compete against the same durable allowance, not per-worker counters.
- Recording usage and settling the reservation share one database transaction.
  Duplicate settlements conflict rather than inserting another billed event.
  Pending and ambiguous requests keep their allowance after timeout, cancellation,
  worker death, or restart. Reconciliation requires explicit evidence; no TTL
  silently frees possibly consumed budget.
- Observed consumption frees the unused allowance. An actual charge above the
  bound remains recorded and blocks subsequent dispatch for those scopes.
  Failure to persist usage fails closed for budgeted calls; cancellation still
  wins and leaves the pending reservation intact.
- OpenAI and DeepSeek each record their actual HTTP attempts, including retries,
  empty/malformed but billed responses, refusals, and cancellations. Network
  latency no longer includes the provider's retry backoff delay.
- DeepSeek exhausted rate limits now propagate their original retry delay,
  instead of falling into a generic retry handler. Invalid credentials, access,
  model, or parameters are actionable stops. Content refusals are review-required,
  never invitations to evade filtering by splitting the passage. Existing model
  alias behavior is unchanged and both configured/effective identifiers are logged.
- OpenAI text rates include separate noncached/read/write categories, actual
  service tier, and whole-request long-context pricing. Reasoning stays a subset
  of output. Every new usage event stores its rate snapshot and cost basis;
  cash paid and credits applied remain unknown rather than invented.
- Legacy failed rows labeled `not_billed` without usage become `usage_unknown`.
  Numeric aggregate columns remain compatible; `unknown_cost_events` accompanies
  totals. The UI still needs to expose this uncertainty explicitly.

### Bounds and integration limits

The request budget uses a conservative **full model context window**, not a
chars/token estimate, plus the explicit output cap. OpenAI requests pin standard
service and 30-minute cache retention; the bound assumes all input is written to
cache at long-context pricing. DeepSeek reserves peak rates even when a request
starts off-peak, and requires an explicit output cap for budgeted calls. Unbudgeted
legacy DeepSeek requests keep their previous output-limit behavior.

At an 8,192-token output limit, one request reserves USD 0.268644 for Luna,
USD 2.68644 for Terra, or USD 0.3244032 for Flash. These are deliberately loose
**allowances, not measured costs**. A USD 5 pilot may serialize expensive requests;
uncertain requests can exhaust the allowance until reconciled. No budgeted call
uses unknown models, custom gateways, tools, or unsupported request contracts.

Sources rechecked on 2026-11-12:

- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/prompt-caching
- https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
- https://api-docs.deepseek.com/quick_start/pricing/

The hooks activate when `budget_scope_ids` is present in the existing usage
context. Callers must configure those scopes before starting work. **This is not
yet an end-user budget setting**: persisted role/preset/job configuration, UI,
CLI and the experiment runner still need wiring and browser validation. No claim
that all application routes already enforce a user budget is made.

### Validation evidence

Reproducible focused commands, with simulated HTTP only:

```sh
venv/bin/python -m pytest -q tests/unit/test_spend_budget.py tests/unit/test_request_budget.py tests/unit/test_deepseek_attempt_budget.py tests/unit/test_openai_pricing_contract.py tests/unit/test_openai_response_contract.py tests/unit/test_deepseek_transient_failures.py tests/unit/test_usage_tracking.py
venv/bin/python -m pytest -q
venv/bin/python -m scripts.preflight_openai --reasoning-effort none
```

Concurrent separate-connection reservations, multi-scope rollback, durable
unknown costs, idempotent settlement, overruns, malformed responses, retry caps,
cancellation and ledger write failures are covered. Cold-import tests launch new
Python processes; they reproduced and fixed a factory/ledger import cycle that
pytest's preloaded modules initially hid.

A read-only SQLite backup of the actual usage database was migrated in a
temporary directory. All **37,875 events**, ID range and summed token totals were
preserved; zero reservations were created and `PRAGMA integrity_check` returned
`ok`. The original database still has no budget tables. The running service's
process session remains `99f0122bace340b7be6d8525ed4f3229`.

Preflight still exits 2 with `missing_credentials`; no paid OpenAI or DeepSeek
generation occurred. Final regression: **2,495 passed, 2 skipped, 10 deselected
in 22.54s**, saved at `/tmp/verbaloom-budget-regression-20261112.log`. The new
budget modules/tests pass Ruff; `compileall` and `git diff --check` pass.

## Implemented slice: explicit roles and job/profile execution

`src/core/pipeline_config.py` validates a secret-free, versioned snapshot of
seven roles. `pipeline_runtime.py` gives each asynchronous task its own provider
pool and durable job budget. It uses the existing provider factory and tracking
proxy; there is no second HTTP client or usage ledger.

| Role | Explicit call sites | Execution condition |
| --- | --- | --- |
| `profile_prepare` | `book_profiles/preparation.py`, `glossary/ner.py` | Requested discovery/NER, including bounded malformed-JSON recovery |
| `profile_review` | `book_profiles/term_review.py` | Candidate batches require remote classification |
| `translate` | `translator._client_generate`, subtitle translation, convenience translation methods | A source unit needs a candidate |
| `refine` | `translator._client_make_request`, XHTML/subtitle refinement | Existing refinement/monolingual policy requests a pass |
| `audit` | `fidelity_supervisor`, `epub/semantic_audit`, translator audit helper | Existing fidelity/profile/final-audit policy requests a judgment |
| `repair` | Translator alert-repair helper, `postprocess_repair` | A located quality defect requires a new candidate |
| `structure_repair` | XHTML placeholder correction, EPUB missing blocks, conversion heading classifier | Structural recovery or classification invokes a model |

This is a call-site map, not yet the complete measured stage/cost/prompt trace
required by the specification. Existing local checks, memory, assembly and
publication are unchanged. Model substitution alone is not an optimized policy.

- `POST /api/translate` accepts an optional `pipeline` snapshot, validates all
  participating credentials before creating a worker, and persists the snapshot
  in the existing job config. Responses redact both providers' keys.
- Resume uses saved roles, not changed environment defaults. Inconsistent legacy
  model/endpoint overrides and implicit budget changes are rejected. Validation
  occurs on a copy: a missing secondary credential does not partly mutate the
  checkpoint. A generic replacement key belongs only to the primary provider.
- The translation worker activates that configuration through all its stages and
  restores outer usage context in `finally`. Provider cleanup failures neither
  hide the original stop reason nor prevent closing the remaining transports.
- Profile preparation accepts the same JSON configuration (or serialized JSON
  in multipart uploads). Discovery/review may use different providers, and only
  these two roles require credentials for a profile-only operation. Their HTTP
  requests share one `job:<prep_id>` budget. Role changes invalidate request
  deduplication; locale is preserved. Status includes the effective role config
  and separate reviewer provider without credentials.
- The existing synchronous-to-async thread bridge now propagates contextvars,
  preserving book/process attribution when it is called from a running loop.
- Legacy callers without a pipeline remain on their previous provider path.
  No default provider, editorial profile, or locale was changed.

### Reproduced operational failures

Fourteen initial new cases produced **11 failures and 3 passes** before repair:
malformed JSON types escaped validation, resume accepted a contradictory endpoint
and mutated config before detecting a missing key, and profile discovery/review
swallowed rate limits, refusal or exhausted budgets. A glossary batch could split
and retry after a permanent refusal, or the profile could appear successful using
only available local signals after budget exhaustion.

Those typed operational stops now propagate without semantic-repair splitting.
The same correction covers the EPUB phase-two fallback, missing-block repair,
profile audit loop and conversion heading classifier. Profile jobs expose a
typed reason/retryability instead of false completion; translation jobs preserve
their checkpoint and suppress automatic retries for actionable stops. Convenience
translation methods reject incomplete/truncated responses as the provider does.

### Simulated execution evidence

The profile integration tests exercise the real OpenAI/DeepSeek providers and
tracking/reservation stack with simulated HTTP. They verify distinct Authorization
headers and official origins, `low` reasoning on the reviewer, reservation before
dispatch, actual provider/model/phase in the ledger, no secrets in status/usage,
and no extra request when the same preparation ID has exhausted its call cap.
Another case runs the profile bridge from an active event loop and verifies
process attribution survives its worker thread.

Translation route tests cover create/no-create and response redaction. Worker
fault injection verifies saved role activation, durable budget stop, preserved
checkpoint and restored outer context. The EPUB phase-two test forces the real
fallback branch and verifies the original budget exception escapes unchanged;
missing-block refusal leaves output bytes unchanged and sends only one request.

```sh
venv/bin/python -m pytest -q tests/unit/test_pipeline_roles.py tests/unit/test_pipeline_entrypoints.py tests/unit/test_profile_pipeline_runtime.py tests/unit/test_pipeline_stop_propagation.py tests/unit/test_provider_action_required.py tests/unit/epub/test_missing_block_repair.py
venv/bin/python -m pytest -q
```

Final regression: **2,534 passed, 2 skipped, 10 deselected in 22.41s**,
`/tmp/verbaloom-role-regression-final-20261112.log`. The new role modules/tests
pass Ruff; changed execution modules pass undefined-name checks. `compileall`
and `git diff --check` pass. Production health still reports process session
`99f0122bace340b7be6d8525ed4f3229` and no active translations. No service restart,
deployment or paid generation occurred. OpenAI preflight remains
`missing_credentials`, with balance unknown and generation untested.

### Open requirements and next integration boundaries

- A/B now retains an independent legacy guard reference, described below.
  Measured end-to-end equivalence remains unproven. Optimized C/D must remain
  unavailable until the candidate policy and evaluation exist.
- Profile job discovery/status still uses the pre-existing in-memory registry;
  this slice does not claim restart-safe profile resume. A completed profile's
  reuse logic currently depends on source name/approved entries, not the full
  preparation dependencies. Audit draft/completed state and cache keys next.
- Full stage tracing, retry multiplication/convergence, malformed semantic audit
  decisions, context/memory dependencies and refusal handling outside the two
  integrated provider paths still require the complete specification audit.
- CLI controls, reactive UI controls/progress, browser mobile/desktop testing,
  private A/B/C/D corpus/runner/blind sample, negative controls, marginal pass
  costs and TXT/structured-resource end-to-end validation remain unfinished.
- The missing OpenAI key prevents live account access/generation evidence, not
  the remaining implementation and simulated validation. No savings or literary
  superiority is claimed, and the full goal remains active.

## Implemented slice: required judgments, refusal and review independence

### Characterization and causes

The initial 37 semantic-contract cases produced **35 failures and 2 passes**.
Fidelity/profile parsers accepted incomplete objects, non-finite values and
incorrect field types. When a required judgment was absent or malformed, callers
could retain a locally passing result; profile normalization filled omitted
dimensions with perfect scores. Thirteen further cases reproduced the same
fail-open behavior in editorial and combined auditing. These are synthetic fault
injections, not evidence of literary quality or actual provider output.

Eight A/B policy comparisons and one default-role test failed before repair:
switching a DeepSeek job to OpenAI also removed DeepSeek-conditioned review rules,
and a Flash preset incorrectly assigned Flash to a role whose effective legacy
default was Pro. A separate checkpoint test proved that changing a role without
changing the primary model did not invalidate its saved stage.

### Changes

- Fidelity, profile, editorial and combined audit boundaries validate required
  fields/types, finite bounded scores, complete active-profile dimensions and
  structured profile findings. Empty findings remain valid. A missing, refused,
  truncated or malformed required judgment cannot approve the candidate.
- Fidelity retains its existing bounded serialization retries (at most three
  requests). Other audit paths stop after their invalid response, with the typed
  `audit_invalid_response` reason. Missing profile/client reasons are distinct.
  This uses the existing non-retryable worker boundary, not another retry loop.
- Legacy `ContentRiskError` now carries the same permanent `provider_refusal`
  reason as the official providers. Removed refusal-triggered source splitting,
  target-only reconstruction and EPUB local-only approval. Legitimate context
  overflow splitting is unchanged. Tests require exactly one request after
  refusal, including long input.
- EPUB's initial final-audit stop records the reason, preserves the reviewed
  candidate/hash and emits the existing checkpoint callback with
  `review_required`; it cannot become `AUDITED`. Placeholder-only cover units
  still use their structural contract without a meaningless semantic request.
- Presets resolve legacy defaults on a copy and persist `review_reference`
  separately from actual model roles. A/B guard options stay identical across
  Flash/Pro, full-final-audit settings, and re-entry, with Czech source and
  `es-ES` locale unchanged. This is configuration evidence, not a completed
  paid A/B comparison. Effective role identities now feed audit/repair model
  helpers and fidelity report attribution without opening provider transports.
- EPUB checkpoints retain non-secret `_pipeline_roles` dependencies. Changing
  translation/structure roles invalidates all dependent stages; refinement or
  repair changes invalidate review/audit; audit changes invalidate only audit.
  Role parameters include provider/model/reasoning/output allowance. Budget and
  credential fields are excluded. Legacy fingerprints without roles are
  unchanged. Full profile/glossary/context dependency coverage remains pending.

### Validation and remaining scope

```sh
venv/bin/python -m pytest -q tests/unit/test_semantic_audit_contract.py tests/unit/test_pipeline_review_strategy.py tests/unit/test_translator_fallback_context.py tests/unit/epub/test_stage_audit.py tests/unit/epub/test_unit_contract.py
venv/bin/python -m pytest -q tests/characterization/test_progress_baseline.py
venv/bin/python -m pytest -q
```

Full suite: **2,603 passed, 2 skipped, 10 deselected in 22.14s**; captured in
`/tmp/verbaloom-audit-final-20261112.log`. The characterization fake previously
returned translation prose for audit requests. It now returns a complete neutral
synthetic assessment, preserving the local decision. **No golden output or
progress files were changed**, and all ten format/progress characterizations
pass. New modules/tests pass Ruff; changed execution modules pass undefined-name
checks; `git diff --check` passes.

Production still reports session `99f0122bace340b7be6d8525ed4f3229`, no active
translations, and the unchanged read-only ledger count **37,875**. No paid
generation, deployment or restart occurred. Tests do not establish actual account
access, EPUB reader appearance, or literary superiority.

The complete objective remains active. Important next boundaries include:
persisting all candidate/decision state through later adjudication/repair stops;
finite automatic recovery and best-candidate convergence; full cache/profile
reuse and restart-safe profile preparation; measured prompt/context costs;
experimental C/D, private corpus and blind/negative-control evaluation; UI/CLI
and actual browser flows; structured artifact/publication end-to-end checks.
The absent OpenAI credential still blocks live OpenAI evidence, not these
implementation tasks. No cost reduction or quality gain is claimed without
measurement.

## Implemented slice: finite durable automatic recovery

### Reproduction and baseline

The pre-change focused lifecycle suite passed **61 tests in 0.54s**, but encoded
unbounded availability retries as expected behavior. Seven new fault cases all
failed before correction: explicit worker retry limits were ignored for timeouts;
alternating failure classes and advancing checkpoint positions had no cumulative
ceiling; a failed-chunk plan ignored a manual pause; a one-day Retry-After was
shortened to 30 minutes; restart could replenish recovery; and a 429 handoff was
not counted until after its sleep. A separate injected race reproduced an old
429 worker demoting the durable row of a newly queued explicit resume.

The legacy prompt cleanup also discarded saved recovery counters. Removing the
prompt contamination remains necessary, but no longer erases attempt history.
No production job was used as a fault-injection target.

### Policy and persistence

`src/core/recovery_policy.py` accounts for existing chunk, finalization and worker
cycle counters plus rate-limit and startup counters. It is a pure policy over the
existing job configuration, not a second state machine or scheduler. Callers
retain the existing lifecycle lock, SQLite checkpoint and worker supervisor.

- All automatic handoff causes share a finite job allowance. Its default is two
  handoffs per checkpoint source unit, with a minimum of ten; the chosen value
  is saved at the first reservation. `max_automatic_recoveries` can override it.
  Advancing a checkpoint can renew a point-specific allowance, never this total.
- Deterministic worker errors retain two attempts at a stalled point. Transient
  worker errors default to five, using the existing 3/15/60/180/600-second steps.
  An explicit `max_worker_error_recoveries` also constrains transient failures;
  `max_transient_worker_recoveries` can configure that category separately.
- Rate limits default to five attempts at a stalled point and now respect saved
  `max_rate_limit_auto_resumes`. Their 60/180/600/1800-second backoff has positive
  jitter up to 20%, capped at 1800 seconds. A valid Retry-After remains a minimum,
  not a hint to silently shorten. Invalid hints or a hint beyond the configured
  wait allowance stop with an explicit review reason instead of retrying early.
- Default provider wait allowances are 7200 seconds per wait and 14400 seconds
  cumulatively per job (`max_provider_wait_seconds` and
  `max_provider_wait_total_seconds`). Planned waits are reserved before sleep;
  cancelling a wait does not refund its conservative cumulative allowance.
- Startup recovery defaults to three crashes without checkpoint advancement.
  It consumes the same cumulative job allowance. A persisted scheduled wait has
  already reserved its retry, so restarting that waiter does not count twice.
  Planned tariff waits neither consume nor reset technical retry counters.
- Reservations are persisted before waiting/dispatch. Failure to save stops
  handoff with `checkpoint_write_failed`; the existing runtime-config persistence
  helper is reused. A scheduled worker or old waiter cannot erase a manual pause
  or demote a newer queued explicit resume.
- Exhaustion retains source/checkpoints and exposes a specific reason. Reload
  restores that reason; an explicit resume clears the old notice, not attempt or
  cost counters. Corrupt counters fail closed rather than replenishing budget.

These defaults are conservative bounded policy, **not empirically calibrated
provider latency recommendations**. They do not change `REQUEST_TIMEOUT`, chunk
size, the deployed environment, editorial acceptance criteria, or model defaults.
The handoff allowance is distinct from per-HTTP attempts, editorial repair rounds
and monetary reservations. Complete retry multiplication and stage-duration
calibration remain part of the full audit. These configuration fields currently
live in the existing saved job config; full UI/CLI controls remain pending.

### Verification

```sh
venv/bin/python -m pytest -q tests/unit/test_recovery_budget.py tests/unit/test_handler_final_status.py tests/unit/test_startup_recovery.py tests/unit/test_resume_schedule.py tests/unit/test_resume_schedule_races_review.py tests/unit/test_provider_action_required.py
venv/bin/python -m pytest -q
venv/bin/ruff check src/core/recovery_policy.py tests/unit/test_recovery_budget.py src/api/startup_recovery.py
git diff --check
```

Final full regression: **2,646 passed, 2 skipped, 10 deselected in 22.36s**,
captured in `/tmp/verbaloom-recovery-verified-20261112.log`. Tests include real
temporary SQLite checkpoints, reload, rejected checkpoint writes, finite crash
loops, old-worker/resume races, manual cancellation, preserved pricing schedules,
corrupt counters and cumulative budgets across different failure classes. Provider
errors and clocks are simulated; no paid generation or physical-device evidence
is claimed. All existing format/progress characterizations remain passing.

Production health still reports session `99f0122bace340b7be6d8525ed4f3229` and
`/api/translations` is empty. No restart, deployment, production migration or
paid model request occurred. Changes are confined to the working tree and tests.

### Remaining full objective

The goal is **not complete**. Next work must still address durable best-candidate
selection and convergence through later EPUB repair/audit stops; independent-unit
continuation without publishing critical gaps; complete retry-layer/circuit-breaker
and cache/memory dependency audits; restart-safe profile preparation and reuse;
experimental C/D policy and private A/B/C/D benchmark with negative controls and
blind human samples; prompt/cost measurements and marginal pass analysis; UI/CLI
controls plus actual desktop/mobile browser workflows; and resource-rich artifact
publication E2E. Live OpenAI access and literary/cost comparison remain unverified
without the required credential. No quality superiority or savings is inferred
from these deterministic reliability tests.

## Implemented slice: durable EPUB audit candidates and convergence

### Reproduction

The focused pre-change unit/stage suite passed 21 tests. Six new fault injections
all failed before the change: a worsening repair displaced the reviewed text;
generated text was not checkpointed before the next audit; explicit zero repair
rounds became two; one rejected unit prevented independent audits; a stale repair
overwrote a newer candidate; and a failed checkpoint did not prevent the request.

A subsequent integration test exposed the production callback boundary too:
`save_xhtml_partial_state()` returning `False` was only logged, allowing further
requests. Another integration case caught an incorrect failure count: the metrics
method takes a chunk *size*, not a number of failures. Both cases now have direct
runtime regressions, not just helper-level mocks.

### Implemented behavior

- The existing unit dictionary contains an additive `audit_repair` journal with
  candidate texts/hashes, their base candidate hashes, audit/adjudication decisions,
  decision checksums, request counters, selected/pending candidates and stop reasons.
  It is serialized by the existing XHTML checkpoint manager, not a second job
  state machine. Prior contracts are retained in `audit_repair_history`.
- Attempt reservations are checkpointed before requests. A generated candidate is
  saved separately from the selected reviewed translation before semantic auditing.
  Re-entry restores saved judgments and audits the pending candidate without
  regenerating it. Interrupted adjudication resumes adjudication, not the initial
  audit. Approved units remain reusable without another request.
- A candidate replaces the selected text only when accepted under the existing
  semantic/structural policy, or when objective defect sets strictly shrink without
  introducing another reported defect. Confidence, style scores and recency do not
  select the winner. Incomparable or repeated findings preserve the previous text
  and stop with a persisted audit failure reason; they are not treated as improvement.
- Repeated candidates and non-improvement stop immediately. Invalid repair output
  can retry only within the saved repair allowance. Removed the blind fallback to
  the full translation pipeline, which could introduce nested translation audits
  and repairs followed by another outer audit. The existing optional adjudication
  remains confined to the originally reviewed candidate, not every repair.
- Explicit zero/one repair allowances are respected; the fallback default is two
  only when neither repair setting is present. `audit_repair_attempts` is cumulative
  for the unit. Malformed counters exhaust their allowance rather than resetting.
  Audit/adjudication invocations are limited to three per candidate/phase, with a
  cumulative default of `3 * (repair_rounds + 2)` supervisor invocations per unit;
  `epub_audit_max_attempts` can constrain that total.
- These are supervisor invocation limits, **not HTTP-attempt counts**. Existing
  serialization/provider retries still have their separate bounds and per-HTTP
  ledger/budget accounting. Consolidating every retry layer remains unfinished.
- Source/candidate/dependency checks before applying a result reject an in-process
  stale repair without overwriting the newer unit. Journal identity includes unit,
  source, languages, model/provider, tags, prompt/stage versions, role options and
  loaded profile content. Diagnostic collectors are excluded: their mutation is
  not an editorial dependency. Missing/changed decision checksums cannot approve.
- A refusal remains a persisted refusal and is not rewritten, split or retried on
  unchanged resume. Truncated/incomplete repair responses cannot be accepted.
- Editorial failures permit other independent reviewed units in the same XHTML
  to finish auditing. The raised error lists all unresolved unit indices, and the
  runtime/checkpoint failure count reflects them. The source DOM is not replaced
  and publication remains blocked while any required unit is unresolved. Provider,
  budget and checkpoint operational stops still propagate to the job boundary.
- A rejected durable quality save raises `checkpoint_write_failed` before another
  quality request. Low-level calls without a configured checkpoint manager retain
  their existing in-memory behavior; they do not claim crash-safe persistence.

### Verification

```sh
venv/bin/python -m pytest -q tests/unit/epub/test_audit_recovery.py tests/unit/epub/test_stage_audit.py tests/unit/epub/test_unit_contract.py
venv/bin/python -m pytest -q tests/characterization/test_progress_baseline.py tests/test_regression_technical_protection.py
venv/bin/python -m pytest -q
venv/bin/ruff check src/core/epub/audit_repair.py tests/unit/epub/test_audit_recovery.py tests/unit/epub/test_stage_audit.py
venv/bin/ruff check --select F821 src/core/epub/xhtml_translator.py
git diff --check
```

Final full result: **2,682 passed, 2 skipped, 10 deselected in 22.26s**, captured
in `/tmp/verbaloom-epub-convergence-20261112.log`. The new recovery suite has 36
cases, including the actual checkpoint manager with isolated SQLite/files,
restart/reload, manual pause after generation, repeated/worsening repairs,
adjudication interruption, persisted refusal, invalid counters, changed profiles,
corrupt decision checksums, rejected saves and exact unresolved-unit counts.
All ten format/progress characterization cases pass without changing goldens.
The first full run caught diagnostic collectors entering the dependency hash;
that regression was reproduced separately and corrected before the final run.

Production still reports session `99f0122bace340b7be6d8525ed4f3229`; the translations
endpoint is empty. No restart, deployment, production migration or paid generation
occurred. OpenAI preflight still reports `missing_credentials`, unknown balance and
generation untested. Simulated audit verdicts demonstrate execution behavior, not
literary quality, savings or provider superiority.

### Remaining scope and boundaries

The full fourteen-part objective remains active. This repair helper still returns
a complete candidate, although its instruction is issue-local: a minimal textual
patch format with source spans and neighboring-context validation is **not yet
implemented**. Current best-candidate comparison is deliberately conservative;
paraphrased findings are incomparable, not proof that a defect improved.

The journal's own identity excludes operational repair allowances. The outer
legacy stage fingerprints can still invalidate review when those options change;
that broader cache migration must be handled with compatibility evidence, not
silently relabeled as solved by the lower-level resume test. Complete shared
glossary/memory dependencies, cross-worker ownership and abandoned-lock recovery,
stage/inactivity calibration, all-format best-candidate handling, restart-safe
profile jobs and complete retry/breaker isolation still require work.

Experimental C/D, the private A/B/C/D corpus/runner, blind and negative-control
evaluation, prompt/cost and marginal-pass measurements, reactive UI/CLI controls,
real browser desktop/mobile flows, and resource-rich publication/artifact E2E
remain unfinished. Missing OpenAI access blocks live evidence, not these other
implementation tasks. No completion or universal reliability claim is made.

## Implemented slice: semantic cache identity and verified resume migration

### Evidence and reproduction

The prior focused baseline passed 67 tests. Twelve new tests failed before this
change: six operational options invalidated editorial stages, `max_retries`
invalidated translation, full XHTML resumes repeated approved work after a limit
change, a failed migration save did not stop subsequent work, and the entrypoint
accepted an identical-text checkpoint from a different job/document. Three more
tests exposed ignored nested glossary metadata, ignored output-token allowances,
and unchanged identities after editing profile content under the same ID.

The new secret filter initially omitted glossary terms literally named `key`,
`token`, `secret` or `password`; four regressions caught and corrected that
distinction between configuration credentials and book content. A separate fault
injection showed that an already-COMPLETED audit journal skipped its saved decision
checksum. That fast path now applies only to legacy approvals without a journal.
An explicit increase of the audit allowance also required rechecking saved
counters instead of unconditionally retaining the old exhausted-state notice.

### Implemented behavior

- Stage fingerprints now identify `epub-semantic-cache-v1`. Technical retry
  counts and the six explicitly enumerated repair/audit/recovery allowances do
  not alter editorial identity. They still constrain all future requests. Model,
  role reasoning/output settings, prompts, languages, segmentation, profile and
  nested glossary/configuration contents remain dependencies.
- Credential fields are excluded without dropping meaningful `max_output_tokens`
  settings or glossary terms whose spelling resembles a credential field.
  Dataclass glossary options and nested mappings/lists are serialized structurally;
  unknown dependency objects fail explicitly rather than being silently omitted.
- Each XHTML entry captures the loaded profile's policy, prompts, glossary,
  detectors and editorial data in a content hash. The root installation path and
  the profile's repair allowance are not editorial changes. A supplied internal
  hash is replaced by the current loaded content; it is not accepted as authority.
- Old fingerprints are reproduced using the preserved legacy algorithm, including
  its former runtime counters. A fixed pre-change hash fixture independently
  verifies compatibility. Source, saved configuration, per-unit provenance and
  available stage hashes must match before migration; corrupt or unrelated input
  is not relabeled as current. Later-stage changes invalidate only their suffix.
- A changed or previously unrecorded profile content hash retains the translated
  text as an **unapproved candidate**, invalidates review/audit, and records why.
  It cannot enter the low-level deterministic-only publication path with review
  disabled. This is reuse of work, not reuse of its prior approval.
- Migrations preserve texts, source, lifetime request/repair counters, candidates
  and saved decisions. The existing checkpoint must accept the migrated state
  before subsequent work; failure raises `checkpoint_write_failed`.
- Saved audit journals are rebound only when the previous contract hash can be
  reproduced and no semantic stage was invalidated. Thus a larger repair allowance
  resumes the pending repair without regenerating translation/review or repeating
  the saved audit. A larger audit allowance rechecks both cumulative and per-phase
  counters before I/O; unchanged or corrupt counters do not replenish requests.
- New completed audit journals verify their selected decision checksum on reuse.
  A corrupted decision blocks publication without spending another request.
- The XHTML entrypoint checks the checkpoint's job ID and document href before
  reuse. Matching source text alone does not authorize cross-book/document reuse.

### Verification

```sh
venv/bin/python -m pytest -q tests/unit/epub/test_stage_cache_resume.py tests/unit/epub/test_audit_recovery.py tests/test_xhtml_translation_state.py tests/unit/epub/test_unit_contract.py
venv/bin/python -m pytest -q
venv/bin/ruff check tests/unit/epub/test_stage_cache_resume.py tests/unit/epub/test_audit_recovery.py src/core/epub/audit_repair.py
venv/bin/ruff check --select F821 src/core/epub/unit_contract.py src/core/epub/xhtml_translation_state.py src/core/epub/xhtml_translator.py
git diff --check
```

Focused result: **103 passed**. Full result: **2,718 passed, 2 skipped,
10 deselected in 22.27s**, recorded in
`/tmp/verbaloom-cache-final-20261112.log`. The 34 new cache tests and two added
audit regressions include the real XHTML orchestration and temporary SQLite/file
checkpoints: old/new identities, reload, explicit allowance increase, no repeated
approved requests, profile changes, narrow invalidation, corrupt provenance and
decisions, failed writes, and cross-job/document isolation. The actual translator
and reviewer boundaries are instrumented or simulated; the tests prove execution
and cache behavior, not literary quality or provider billing. Existing format and
progress golden characterizations remain unchanged and passing. Selected lint and
diff checks pass; legacy modules still have unrelated style/typing lint findings.

Production health still reports session `99f0122bace340b7be6d8525ed4f3229`, with an
empty translations list. No service restart, deployment, production checkpoint
migration or paid generation occurred. OpenAI preflight still exits 2 with
`missing_credentials`, unknown balance and generation untested.

### Remaining full objective

This supersedes the earlier known gap about outer EPUB fingerprints repeating
review on operational allowance changes. It does **not** complete the fourteen-part
objective. Full accepted-only continuity memory and neighbor dependency handling,
cross-worker ownership/abandoned locks, pinned profile behavior during an active
run, all-format equivalent candidate persistence, stage/inactivity calibration,
retry/breaker isolation and restart-safe profile preparation still need work.

The next independent deliverables remain experimental C/D policy and the private
A/B/C/D benchmark runner/corpus, blind source-aware evaluation and negative
controls, marginal-pass/context/cost measurements, minimal checksum-bound text
patches, reactive UI/CLI configuration and real desktop/mobile browser workflows,
and resource-rich publication/artifact E2E. Missing OpenAI access blocks live
comparison, not those offline tasks. No savings or model-quality superiority is
claimed, and the goal remains active.

## Implemented slice: private benchmark corpus and independent evaluation

The intervening headphone task did not advance this objective. This slice resumed
against the current installed checkout without restarting its service. The sample
API's direct translation call was inspected and rejected as a substitute for the
real baseline: it does not execute the complete production pipeline.

Implemented offline TXT/EPUB corpus extraction, exact source/context/profile
identity, whole-sequence tuning/heldout separation, task-local full candidate
capture, independent budgeted evaluation through existing providers, strict
source-anchored findings, blind static HTML and marginal-finding accounting.
See [benchmark contracts and commands](EDITORIAL_BENCHMARK.md). Normal candidate
metadata remains bounded and does not begin logging full private text.

Private preliminary corpus:
`~/.local/share/VerbaLoom-benchmarks/pilot-20261112/{english,czech}/corpus.json`.
It contains 20 actual paragraph-sized units (812 source words), four adjacent
five-unit sequences, with ten units in each split. The English profile snapshot
preserves its existing es-MX locale. The historical Czech source was found in a
preserved text-first input, but its profile/locale have not been recovered; they
remain explicitly unresolved, and the evaluator blocks that corpus before HTTP.
Neither this preliminary size nor unverified category coverage proves a
representative literary benchmark. No private excerpts entered tracked files.

All corpus files were verified as 0600 under private 0700 directories. A malformed
German EPUB failed source extraction due to undefined XML entities instead of
silently producing a truncated corpus; a valid English EPUB and retained Czech
TXT were used. Source books, profiles and translations were not rewritten.

Actual deterministic calibration used six synthetic negative controls plus two
legitimate variants. Damaged markers were rejected; paragraph omission produced
a length warning but remained locally accepted. Negation inversion, changed name,
duplication and flattened voice were unflagged. The two legitimate controls had
no flags. This is concrete evidence that mechanical acceptance alone is unsafe
as a semantic approval criterion. It is not a model benchmark or proof of voice
quality. The report is in the private `local-calibration/local-controls.json`.

The evaluator requires an existing experiment budget and isolated usage ledger,
uses the existing per-HTTP reservation protocol, and refuses self-judging,
including alias-equivalent models. Malformed/truncated judgments are retained
through the private response sink and fail closed without an editorial retry
loop. HTTP fault injection verified reservation before dispatch, observed usage,
shared call-cap exhaustion before a second dispatch and correct cost attribution
to `editorial_benchmark`. Simulated HTTP is not measured provider performance.

Validation commands:

```sh
venv/bin/python -m pytest -q tests/unit/test_editorial_benchmark.py tests/unit/test_benchmark_judge.py tests/unit/test_benchmark_artifacts.py
venv/bin/python -m pytest -q
venv/bin/ruff check src/core/editorial_benchmark.py src/core/benchmark_judge.py scripts/editorial_benchmark.py tests/unit/test_editorial_benchmark.py tests/unit/test_benchmark_judge.py tests/unit/test_benchmark_artifacts.py
git diff --check
```

The initial new-test collection failed because the module did not exist. Forty-six
new focused cases now pass. Final full-suite result: **2,764 passed, 2 skipped,
10 deselected in 22.35s**, recorded in
`/tmp/verbaloom-benchmark-final-20261112.log`. New modules/tests pass Ruff;
the existing candidate module passes F821/import checks; `git diff --check` passes.
The final review also removed pending glossary suggestions and reviewer-model
attribution from the judge prompt, while preserving the full private snapshot.

Real Playwright browser inspection of a clearly labeled synthetic reader fixture
at 1280x900 and 390x844 showed 10 passages/40 candidates, two desktop columns,
one mobile column, zero scripts and no horizontal overflow. Screenshots were
inspected in `output/playwright/benchmark-{desktop,mobile}.png`. This does not
prove production app role selection, persistence or physical Android behavior.
The temporary loopback-only fixture server and browser session were stopped.

Production remains session `99f0122bace340b7be6d8525ed4f3229`, with no active
translations. OpenAI preflight still reports `missing_credentials`, null balance
and generation untested. No paid calls, deployment or default changes occurred.
The production usage ledger remains 37,875 events with maximum timestamp
1790919396.81183, unchanged from the prior verified inventory.

The complete objective remains active: C/D policy and real A/B/C/D runner, complete
candidate-stage capture and crash-safe benchmark resume, budget wiring across all
benchmark roles, category coverage and Czech profile provenance, real/human
evaluation, localized checksum patches, all-format memory/dependency/ownership
recovery, restart-safe profiles, reactive UI/CLI, and resource-rich publication
E2E remain required. The new reader and isolated judge are not substitutes for
those acceptance gates. Next implementation should connect the real executor
and use the observed mechanical false negatives to calibrate semantic coverage.

## Implemented slice: checksum-bound EPUB repair proposals

The intervening Orbita verification did not advance this goal. The current
checkout was revalidated on `main`, HEAD `5eddf00`, retaining all accumulated
local changes. No production service was restarted or deployed.

Inspection found that final EPUB fidelity repair still requested a whole returned
candidate for local findings. Added opt-in `pipeline.epub_repair_policy` with
`localized_v1`, retaining `compatible` as the default. It uses the same provider
role, request budget, unit journal, semantic audit and best-candidate rules.
See [localized repair contract and configuration](LOCALIZED_REPAIR.md).

The new pure patch validator binds unit/source/candidate/context and findings to
checksums, derives edits from unique anchors in stable text blocks, rejects
ambiguous/stale/overlapping/structural patches and keeps all untouched text exact.
There is no full-rewrite fallback or automatic semantic approval. Existing final
audit and publication checks still reject a structurally valid but non-improving
repair and preserve the prior candidate.

Raw patch receipt is durable before the next semantic call. Controlled worker loss
after receipt and real checkpoint reload reuse that response without another
repair generation or counter reset. Malformed saved responses also retain their
consumed attempt; corrupt envelopes stop visibly. A later policy switch preserves
received work under its original serialization contract. Repair serialization is
an operational dependency, so it does not invalidate existing semantic approvals
or repeat completed translation/review stages.

Initial new-test collection failed because the patch module did not exist. The
first full-suite pass after implementation was **2,812 passed, 2 skipped,
10 deselected in 23.33s**. Additional checkpoint/policy-switch cases were then
added; final verification is recorded below after those changes. The tests use
synthetic provider responses, not paid or human editorial measurements.

OpenAI preflight remains `missing_credentials` (exit 2), with unknown balance and
generation untested. C/D review policy and the real A/B/C/D executor remain the
next larger deliverables. This slice does not complete those, the representative
corpus/Czech provenance work, broader memory/ownership/profile recovery,
reactive UI/CLI or resource-rich publication E2E. The goal remains active.

Final verification after all changes in this slice:

```sh
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-localized-20261112.xml
venv/bin/ruff check src/core/localized_repair.py src/core/pipeline_config.py tests/unit/test_localized_repair.py tests/unit/epub/test_localized_audit_repair.py
venv/bin/ruff check --select F821 src/core/epub/xhtml_translator.py src/core/epub/unit_contract.py
git diff --check
```

Result: **2,816 passed, 2 skipped, 10 deselected in 23.04s**, including 52 new
patch/EPUB recovery cases. JUnit evidence is in
`/tmp/verbaloom-localized-20261112.xml`. New code/tests pass Ruff; selected checks
on the existing large EPUB modules and the diff check pass. There were no UI
changes in this slice and no new browser or physical-device validation.

The service still has no active translations and retains process session
`99f0122bace340b7be6d8525ed4f3229`. Read-only inspection of
`token_usage_events` shows the unchanged 37,875 events and latest `created_at`
1790919396.81183. No paid generation, source-book edits, deployment, restart,
push or production checkpoint migration occurred.

## Implemented slice: source-audited C/D strategy and reuse contracts

The intervening Orbita task was unrelated to this objective. Re-read the full
objective and revalidated this installed checkout on `main`, HEAD `5eddf00`,
preserving all accumulated local changes. The source-audited strategy had been
partially implemented but was not yet connected consistently to all entry paths.

Added/finished explicit C/D presets, separating review policy from role models.
Ordinary translation omits universal rewriting while retaining full source-aware
semantic coverage. Native XHTML uses the existing final unit audit and localized
repair; generic and plain paragraph routes audit their accepted candidates. The
handler uses the same effective plan for chained refinement and progress. Explicit
refinement, transformations and strict profiles retain their required stages.
See [strategy behavior and remaining boundaries](SOURCE_AUDITED_STRATEGY.md).

New real-checkpoint/memory regressions reproduced defects missed by the first
isolated strategy tests: a 16-character FidelityDecision hash was compared with a
full SHA-256, causing approved resume to stop; nested memory metadata was read at
the wrong level; and candidate telemetry changed the memory key during the audit,
preventing cache hits. Fixed those contracts without rewriting stored data.
Source/index changes block reuse; accepted noncritical warnings remain reusable;
changed cached text receives an audit without regenerating its translation.

Initial expanded focused result was 7 failures/16 passes. After fixes, 34 focused
strategy/plain/memory tests passed, followed by 80 strategy/handler/entrypoint/
inline tests. The first full-suite pass was **2,844 passed, 2 skipped,
10 deselected in 23.02s**. An additional runtime-rebinding assertion then covered
explicit transformation preservation; final verification follows below.

No paid calls or production restart occurred. Current health still reports session
`99f0122bace340b7be6d8525ed4f3229` and no active translations. OpenAI preflight exits
2 with `missing_credentials`, unknown balance and generation untested.

The objective remains active. The full benchmark executor, shared all-role budget
through that executor, representative corpus/Czech profile provenance, measured
semantic calibration/heldout evaluation, generic/plain draft recovery and complete
dependency invalidation, remaining lifecycle/profile recovery, UI/CLI and actual
browser flows, and resource-rich publication E2E are still acceptance work. This
slice does not establish savings, literary equivalence or production readiness.

Final checks after runtime-rebinding coverage and import cleanup:

```sh
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-source-audited-final-20261112.xml
venv/bin/ruff check src/core/review_strategy.py src/core/pipeline_config.py tests/unit/test_source_audited_strategy.py
venv/bin/ruff check --select F821 src/core/adapters/generic_translator.py src/core/adapters/translate_file.py src/core/common/plain_text_pipeline.py src/core/job_runtime_config.py src/core/translation_memory.py src/api/handlers.py src/core/fidelity_supervisor.py src/core/epub/xhtml_translator.py
git diff --check
```

Result: **2,844 passed, 2 skipped, 10 deselected in 22.62s**. New code/tests and
selected checks on the existing modules passed Ruff; diff whitespace checks
passed. Production usage remains 37,875 events, latest `created_at`
1790919396.81183. No deployment, restart, paid generation or push occurred.
No browser/physical-device validation was performed for this pipeline slice.

## Implemented slice: production TXT experiment runner and no-change refinement

The previous goal work made progress: it added the first real A/B/C/D executor.
Its completed JUnit file was inspected after the interruption: 2,858 passed,
2 skipped and no failures. The intervening Orbita verification was unrelated.
This continuation re-read the complete objective, checked the current checkout,
and preserved its accumulated changes. It did not restart the live service.

`scripts/editorial_benchmark_run.py` now runs all four frozen configurations
through production TXT orchestration, not the sample API or a fake translator.
Only the HTTP boundary is simulated in tests. Sequences preserve order, source
context, profile/locale and per-sequence memory. A task-scoped profile snapshot
includes approved/pending glossary entries, examples, detectors and editorial
artifacts without consulting mutable production profile files. Private checkpoint
uploads and translation memory are injectable; existing production defaults do
not change. Every request and captured candidate uses the same corpus unit ID.

All generator/auditor/repairer/evaluator attempts inside one experiment share
atomic reservations. Raw external judgments are durable before parsing and bind
the full candidate, source corpus, preceding translated context, judge and prompt
version. Resume revalidates both raw receipts and derived assessments, even after
completion. Missing state, ledger, checkpoints or receipts cannot silently reset
spending or regenerate approved work. Immutable blind/candidate artifacts are
checked against their retained inputs. OS locks prevent concurrent executors;
pending refinement tasks are drained/cancelled before provider cleanup.

New corruption tests reproduced five genuine failures before fixes: generation
used native `chunk_0` identifiers instead of corpus IDs, and four retained-state/
receipt corruption cases were accepted. After fixes they stop without new HTTP.
Additional tests cover gross-money and call caps across resume, cancellation
with uncertain spend, worker loss after a saved judge receipt, active-worker
exclusion, profile roundtrips and pending inline task cancellation. Tests modify
isolated state only; no real account is charged.

### Actual redundant-work defect found by the trace

Inspecting the synthetic per-stage ledger found four unexpected repair calls in
A/B. `_refine_unit_inline` did not pass the original `unit.content` to
`_make_refinement_request`. The latter consequently compared an unchanged Spanish
draft against itself while retaining English as its source language. It labeled
the good no-change result `untranslated_source`, requested a language repair and
then fell back to the same draft. This was not a necessary editorial pass.

The narrow fix supplies the already available original source. A failing
production-engine regression now passes: unchanged good refinements trigger no
repair, while deliberate English source echoes still trigger the language gate
and retain the approved Spanish draft. Original-source glossary matching and
identifier checks also receive the intended source. No gate was removed.

The retained synthetic before/after artifacts show 28 versus 24 HTTP calls for
two tuning units across four variants, including eight independent judgments.
Before: A/B each made two translations, two refinements and two spurious repairs.
After: A/B each make two translations and two refinements; C/D each make two
translations and two semantic audits. These fixed mock responses prove control
flow, **not token savings, literary quality or superiority on a real book**.

Private inspectable artifacts (synthetic only):

- Before: `~/.local/share/VerbaLoom-benchmarks/synthetic-executor-20261112T2145/test_abcd_runs_real_engine_cap0/private/`.
- After: `~/.local/share/VerbaLoom-benchmarks/synthetic-executor-fixed-20261112T2200/test_abcd_runs_real_engine_cap0/private/`.
- Each includes `report.json`, `usage.db`, checkpoints, intermediate candidates,
  raw evaluations and a clearly labeled `blind.html` with separate answer key.

Reports distinguish observed/estimated/unknown costs, cache categories and
reasoning already included in output; calls/tokens/timing by stage and unit;
review/repair coverage; and generation cost per independently approved source
word. They expose different external judges for the two provider groups as a
confound. Partial or corrupt evidence has no final comparable cost ratio.
Historical profile preparation is explicitly reused, not counted as newly free.

### Commands and verification

Exact plan/run/resume and credential instructions are in
[the benchmark guide](EDITORIAL_BENCHMARK.md). A no-network plan against the
retained English corpus preserved its existing es-MX profile and was saved under
`~/.local/share/VerbaLoom-benchmarks/runner-plan-20261112.MLNBoS/english-tuning`.
The adjacent `baseline.example.json` is an explicit starter, not evidence of the
past job's chosen model. The CLI run stopped before generation because the
OpenAI credential is absent. Non-billable preflight again returned exit 2,
`missing_credentials`, unknown balance and `generation_tested: false`.

```sh
venv/bin/python -m pytest -q tests/unit/test_benchmark_execution.py tests/unit/test_target_language_gate_repair.py tests/unit/test_inline_refinement_pipeline.py
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-benchmark-refinement-final-20261112.xml
venv/bin/ruff check scripts/editorial_benchmark_run.py scripts/editorial_benchmark.py src/core/benchmark_judge.py tests/unit/test_benchmark_execution.py
git diff --check
```

Focused result: **39 passed in 2.92s**. Final regression: **2,870 passed,
2 skipped, 10 deselected in 24.84s**. Ruff and whitespace checks passed. No new
production-browser or physical-device validation occurred in this slice.

### Remaining full-objective gates

The runner currently covers one corpus and split per directory. A single parent
budget must span the eventual multi-book/tuning/heldout pilot; separate USD 5
directories must not be launched as if their total were USD 5. Generic/plain
unapproved drafts still need full stage recovery, complete dependency invalidation
and wider ownership/profile lifecycle verification. Required remaining work also
includes Czech profile provenance and category coverage, semantic negative
controls, marginal-pass/prompt/block/effort ablations, real/human heldout evidence,
reactive application UI/CLI settings and desktop/mobile flows, and resource-rich
native EPUB/DOCX publication E2E. The objective is still active, not completed.

Live service health retains session `99f0122bace340b7be6d8525ed4f3229`, with no
active translations. Production usage is unchanged: 37,875 events and latest
`created_at` 1790919396.81183. No paid generation, source-book modification,
production database migration, deployment, restart or push occurred.

## Implemented slice: durable generic stages and consistent resume context

The intervening Orbita verification did not advance this goal. Re-read the full
objective and inspected the authoritative `main` checkout at `5eddf00`, retaining
all existing changes. The service had no active translations and was not restarted.

`UnitStageJournal` stores received translation, retry, refinement and audit
receipts inside existing checkpoint chunk metadata. A received draft is not a
completed unit or an accepted memory entry. Input dependencies and result hashes
are checked before replay. Stage attempts are saved before invocation and remain
bounded across worker loss and resume: `generic_stage_max_attempts` defaults to 3
and is capped at 10. This is a stage-invocation allowance, not a substitute for
the existing per-HTTP request and spend limits. An unknown remote result can
still have been billed; this is not exactly-once provider billing.

The generic engine and benchmark executor reuse a standard-library OS execution
lock. A real child-process kill/reacquire test verifies abandoned ownership on
this Mac. Simultaneous engines cannot spend on the same checkpoint. A separate
regression reproduced a double-cancellation lock leak during client cleanup;
unconditional lock release now passes that test. Windows-specific locking was
not exercised here.

### Reproduced failures and corrections

- Four later-stage worker-loss boundaries originally regenerated a received
  translation. Resume now replays preceding receipts and repeats only the
  interrupted stage, subject to its persisted allowance.
- Adjacent inline work originally used a provisional neighbor during live
  execution but its refined neighbor on resume. Two source units produced three
  generation calls in the injected-failure test, both with and without literary
  memory. Dependent units now wait for their approved final neighbor; both cases
  use two generation calls. **This removes translation/refinement overlap for
  adjacent generic units.** It trades that latency optimization for deterministic
  context and recovery; no real-book throughput improvement is claimed.
- Sparse checkpoints rebuild only the approved prefix, then incorporate retained
  units in source order. Stale future snapshots and rejected/provisional candidates
  do not enter shared continuity. Auto-detection can enable literary memory after
  nonliterary frontmatter. Progress counts accepted units, not the latest index.
- Empty text was stored as a reusable successful stage, leaving resume stuck on
  the same empty value. Empty/whitespace results now remain failed attempts;
  bounded retries can recover or end in `stage_attempts_exhausted` without output.
  Four regression cases failed before this correction and pass afterward.
- TXT/PDF restore uses the canonical checkpoint row index, even if optional
  metadata does not repeat it. Conflicting repeated indices stop before generation.
  An older sparse-resume fixture was also corrected to use the actual adapter's
  normalized segmentation; it had silently relied on incompatible raw chunks.
- TXT and PDF reconstruction reject missing translations instead of silently
  substituting the original. All six new PDF missing/empty/bilingual cases failed
  before the guard was added. Generic publication checks durable complete source
  coverage and uses the existing atomic-write implementation for binary output.
  Pause, partial work and injected rename failures preserve a previous valid file.
- A never-approving auditor and a worsening repair terminate without discarding
  the better candidate or generating endlessly. Resume reuses their decisions;
  it does not treat an editorial rejection as an invitation to start over.

New tests use the production generic engine, actual TXT/PDF adapters, real SQLite
and process locks. Model boundaries are synthetic. The current application
providers/defaults, profile data, source books and production checkpoints were
not changed. No current browser or physical-device validation occurred.

### Verification and remaining gates

```sh
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-stage-recovery-20261112.xml
venv/bin/ruff check src/core/adapters/stage_journal.py src/persistence/execution_lock.py tests/unit/test_generic_stage_recovery.py tests/unit/test_benchmark_execution.py tests/test_adapters/test_pdf_adapter.py scripts/editorial_benchmark_run.py
venv/bin/ruff check --select F821 src/core/adapters/generic_translator.py src/core/adapters/txt_adapter.py src/core/adapters/pdf_adapter.py src/persistence/checkpoint_manager.py src/core/quality_assurance/models.py
git diff --check
```

Full regression: **2,910 passed, 2 skipped, 10 deselected in 25.38s**. A subsequent
test-only lint cleanup removes a redundant `return None`; the final check is
recorded below. The stage recovery file contains 33 cases and PDF coverage adds
7 cases. These checks do not establish literary quality, token savings or human
acceptance. The existing benchmark code digest deliberately rejects old plans
after source changes; create a new frozen plan, not an overwritten experiment.

Legacy checkpoints without journals remain compatible when their actual source
unit map matches. New journal dependency mismatches stop for explicit review;
fine-grained invalidation, nested model-call recovery and all lifecycle/profile
paths still need completion. Parent multi-book/split pilot budget, Czech profile
provenance, negative controls, semantic calibration, marginal-pass/prompt/block/
effort measurements, real/human heldout evidence, reactive UI/app CLI and browser
flows, and resource-rich native publication E2E remain full-objective gates.
The goal is active, not completed.

Final verification after lint cleanup:
`venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-stage-recovery-final-20261112.xml`
returned **2,910 passed, 2 skipped, 10 deselected in 25.30s**. Ruff on the new
modules/tests, selected undefined-name checks on existing modules and
`git diff --check` passed. The live service still reports session
`99f0122bace340b7be6d8525ed4f3229`, no active translations and unchanged usage
(37,875 events; latest `created_at` 1790919396.81183). No paid generation, restart,
deployment or push occurred.

## Implemented slice: one parent budget for the entire pilot

The intervening Orbita verification was no progress toward this objective. This
pass resumed the actual checkout without restarting the live service or changing
its data. The earlier uncompleted full-suite process was polled to completion:
2,926 passed and two newly added tests failed. One test passed ledger context as
unsupported keyword arguments. The other expected an automatic retry despite an
ambiguous Terra timeout retaining enough allowance to exhaust the USD 5 cap.
The test now exercises both the required stop under USD 5 and successful retry
accounting with a larger **synthetic-only** allowance. The paid default was not
raised and unknown charges were not released.

`scripts/editorial_benchmark_pilot.py` freezes a parent identity, evidence kind and
budget in a private directory. Child plans bind to its original path and digest.
All jobs, independent judges, retries, books and splits reserve against the same
existing `SpendBudget` ledger in one transaction. No second cost calculator or
production database was introduced. Child scopes remain available for reports
and tighter limits. Judge job IDs include experiment identity to avoid collisions
when two child plans contain the same units with different limits.

Paid execution now requires this parent. CLI `pilot-init`, required `plan --pilot`
and `pilot-status` expose creation, binding and total usage. Simulation cannot be
upgraded to paid evidence. Missing/replaced ledgers, modified manifests, copied
child directories, missing started-state files and an old ledger contradicting
retained spend stop before dispatch. Resume never resets allowances or retries
accepted work. This is protection against accidental inconsistent recovery, not
a guarantee against an operator replacing all state or creating a different pilot.

A concurrency regression demonstrated that separately read status snapshots could
show a newer parent count than its children. Parent and child snapshots now share
one read transaction. A separate WAL connection commits a reservation mid-read in
the test; the old implementation fails and the corrected implementation returns
a consistent before-or-after view. Reports filter by process before applying the
event limit, so sibling activity cannot displace one experiment's recorded calls.

Verification:

```sh
venv/bin/python -m pytest -q tests/unit/test_benchmark_pilot.py tests/unit/test_benchmark_execution.py tests/unit/test_spend_budget.py tests/unit/test_pipeline_roles.py --junitxml=/tmp/verbaloom-parent-pilot-focused-20261112.xml
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-parent-pilot-final-20261112.xml
```

Focused tests: **79 passed in 5.52s**. Full regression: **2,930 passed, 2 skipped,
10 deselected in 28.76s**. Twenty new parent-budget cases cover two books/splits,
judge isolation, HTTP retry accounting, cancellation uncertainty, simultaneous
workers, 30 threaded reservations through independent SQLite connections, state
loss, stale backups, file permissions, CLI creation/status and snapshot consistency.
HTTP responses are simulated, not provider/editorial benchmark measurements.

An actual no-network CLI freeze created tuning and heldout children for the
retained English corpus under one private parent:
`~/.local/share/VerbaLoom-benchmarks/parent-pilot-20261112.kMIx5l/`.
Both children are unstarted and the parent reports zero calls, zero settled and
reserved cost, and USD 5 available. The real preflight returned
`missing_credentials`; `run --allow-paid` stopped before generation because the
OpenAI key is absent. The starter baseline is explicit configuration, not claimed
as reconstructed historical job configuration. No paid comparison was performed.

Full-window request bounds remain conservative: an uncertain call can reserve
most of a small pilot budget. This is an allowance, not a measured bill. Do not
recreate a pilot to bypass it. Complete remaining profile provenance, category
coverage, semantic controls, marginal/prompt/block/effort measurements, human
heldout evaluation, UI/app CLI and browser flows, native resource E2E, and remaining
lifecycle/dependency coverage before declaring the full objective achieved.

## Implemented slice: native artifacts and gated publication

The native EPUB integration now runs A/B/C/D with navigation both inside and
outside the spine. Only the provider HTTP boundary is synthetic. Source archives
contain a cover, illustration and caption, chapters, inline emphasis, note and
backlink, a table, numbers and a formula. Native extraction, translation,
checkpointing, usage accounting, ZIP assembly, whole-EPUB gates and installed
EPUBCheck 5.3.0 run normally. All eight cases also reopen the checkpoint manager
and reassemble to a second path without another HTTP request or usage event.
Resource bytes, XHTML/CSS and source hashes are checked, not just file existence.

This reproduced two independent publication failures:

- An EPUB 3 contributor carried an EPUB 2 `opf:role` attribute. EPUB 3 now uses
  an ID-refined `role` metadata element with `marc:relators`; EPUB 2 keeps its
  attribute. Existing attribution migrates idempotently, preserving unrelated
  metadata and avoiding ID collisions. See the [EPUB 3.3 metadata
  specification](https://www.w3.org/TR/epub-33/).
- The DOM boundary gate rejected the engine's generated hidden cover landmark
  as added body content. It now projects only the exact generated subtree when
  authorized by the output package's navigation/cover metadata. Visible, nested,
  duplicate, wrong-target and modified landmark trees still fail; original TOC
  content remains checked. Spine element counts subtract only that verified
  addition, not all navigation content.

Native EPUB/DOCX assembly also used direct destination writes. The existing
atomic byte writer now exposes a streaming context, reused by ZIP assembly;
DOCX uses the same byte API. Write/fsync/replace fault injection preserves an
existing destination and removes unfinished temporary files.

Focused native tests: **125 passed in 4.20s**. Resource-rich EPUB integration:
**8 passed in 62.36s**. The completed full native regression returned **2,968
passed, 2 skipped, 10 deselected in 91.10s**. Only the OPF hash changed in each
of two EPUB golden files; other member hashes and progress snapshots were not
rewritten. The initial native full run exposed exactly those two hash changes.

### Final delivery boundary

Atomic ZIP writing alone did not protect a previous delivery: the API worker
assembled into its final filename and ran quality gates afterwards. A failing
gate then renamed that already-replaced file as partial. The QA CLI also used
one shared `.publishing` filename and a check-then-replace sequence, allowing
concurrent publishers to lose a file or overwrite a delivery.

Nine regression cases first failed against that behavior. Web assembly and
conversion now remain in a unique hidden directory, retaining the intended
basename and extension. The staging path is saved with the job configuration.
Publication requires successful processing, no interruption, universal QA and,
where applicable, the strict EPUB gate. Other candidates are labeled partial;
crashed/cancelled runs retain their private candidate and accepted checkpoint.
An explicit engine error is not changed into a successful or generic partial
status. A separate pair of reproductions exposed an ignored `False` from chained
refinement; this now prevents successful publication.

API and QA CLI share `quality_assurance.publication.atomic_publish`. It copies
and fsyncs to a unique same-directory temporary file, then uses an exclusive hard
link as the commit point. Existing paths, including dangling symlinks, are never
replaced. Concurrent contenders obtain separate numbered names. Later changes
to a staged candidate cannot modify its delivered copy. Filesystems without
hard-link support fail closed and retain staging; there is no unsafe overwrite
fallback. This is atomic visibility, not a claim of distributed transactions or
exactly-once completion across process death.

Reports follow the committed filename. Audiobook companions require an approved
main delivery; a permanent error cannot bypass that check merely because it is
not labeled partial. Native conversion intermediates stay
private rather than appearing as unchecked deliveries. Staged files and reports
are retained for recovery/traceability; no production documents are deleted.

The 22 new publication tests cover real worker orchestration with synthetic
translation/conversion/QA boundaries, concurrent publishers, copy/fsync/link
failure, symlinks, independent copies, hidden staging, report naming, success,
rejection, validator exception, manual pause, failed refinement, worker crash,
cancellation, permanent engine error and companion gating. They do not replace the real native
EPUBCheck tests or prove browser behavior. An initial crash fixture had no saved
unit and therefore exercised the intentional unstarted-job cleanup; it was
corrected to persist an accepted unit before injecting the crash.

Focused publication/recovery validation: **103 passed in 0.72s**. The complete
regression after staging/refinement fixes returned **2,990 passed, 2 skipped,
10 deselected in 91.17s**. The final companion condition was then covered by all
22 publication tests before repeating the complete suite. Ruff passed for the
new module/tests, undefined-name checks passed for the modified legacy modules,
and `git diff --check` passed; this is not a claim that legacy Ruff warnings were
all eliminated.

Reproduction commands:

```sh
venv/bin/python -m pytest -q tests/unit/epub/test_native_pipeline_e2e.py
venv/bin/python -m pytest -q tests/unit/test_atomic_native_outputs.py tests/unit/epub/test_metadata_language.py tests/unit/epub/test_navigation_boundaries.py
venv/bin/python -m pytest -q tests/unit/test_publication_staging.py tests/unit/test_quality_cli.py tests/unit/test_handler_final_status.py tests/unit/test_recovery_budget.py
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-staged-publication-final-20261112.xml
```

The post-companion full-suite XML was inspected on continuation, with no live
pytest process remaining: **2,990 passed, 2 skipped, 10 deselected in 90.82s**.
The legacy CLI gap identified during this slice is addressed below. UI/app CLI
configuration, desktop/mobile browser flows, complete profile/context
dependencies, calibrated risk sampling, and paid/human A/B/C/D evaluation remain
required. No paid pilot or literary/cost superiority is claimed.

## Implemented slice: CLI publication and non-mutating checkpoint reads

The old `translate.py` entrypoint ignored adapter return values, wrote directly
to a publicly visible output name, notified success after failures, attempted TTS
on unapproved candidates, and returned shell success even after an exception.
Sixteen controlled translation/refine-only entrypoint cases reproduced these
failures before the correction. They use temporary databases and synthetic
processing boundaries, not paid providers or production books.

Both processing CLIs now share `finalize_cli_run` and the existing whole-book
validator and atomic publisher. The legacy entrypoint creates a durable job
before processing, preserves its input/configuration, stages under a private
`.publication-*` directory next to the intended output, and starts TTS only
after approved delivery. Rejection leaves the candidate/checkpoints available
without overwriting a previous delivery. Exit codes are 0 for approved success,
1 for operational failure, 2 for blocked publication, and 130 for interruption.
The invalid legacy `LogType.INFO` TTS logging constant was also corrected.

Refine-only uses the target language as both source and target for final QA,
without disabling coverage, entity, structure, or output checks. Its operation
flag is saved and respected by the strict CLI's resume dispatcher. Two additional
tests exercise the real TXT adapters and all ten final gates through the legacy
entrypoint; only the LLM provider is synthetic. These are integration tests, not
editorial quality measurements.

Characterization exposed two shared lifecycle defects:

- `load_checkpoint` rewrote a paused/error/interrupted complete checkpoint as
  `completed` or `validating`. Ten regressions failed before the fix. Loading now
  reconciles the returned progress snapshot without writing lifecycle state.
  Complete-but-stopped jobs remain discoverable for finalization. Three legacy
  assertions were updated because they explicitly required the unsafe implicit
  transition; `mark_completed` still requires the configured final quality gate.
- `mark_chunks_for_repair` unconditionally replaced every status with `partial`.
  Three stopped-state regressions failed. Its atomic SQL update now preserves
  explicit stop reasons while recording rejected text and repair indices; active
  or previously completed jobs with rejected content become partial.

The strict CLI also persists worker exceptions/cancellation and closes its job
manager on exit. Duplicate existing run IDs are rejected before source copying
and without modifying that job. A rejected generic-worker lock does not mark
the actual owner's run as errored. This is not a claim of complete cross-process
transactional ownership of every CLI/format path; that audit remains open.

Focused validation: **170 passed in 1.40s**. The first full regression returned
**3,018 passed, 1 failed, 2 skipped, 10 deselected in 91.20s**; the failure was the
legacy assertion that merely reading completed chunks must change status to
validating. It now checks that the read preserves running and that the explicit
worker completion call enters validating until final QA passes.

Reproduction commands, from the installed checkout:

```sh
venv/bin/python -m pytest -q tests/unit/test_legacy_cli_publication.py tests/unit/test_quality_cli.py tests/unit/test_strict_quality_cli_e2e.py tests/unit/test_checkpoint_resume_index.py tests/unit/test_universal_quality_assurance.py tests/unit/test_publication_staging.py tests/unit/test_recovery_budget.py tests/unit/test_startup_recovery.py --junitxml=/tmp/verbaloom-cli-publication-focused-20261112.xml
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-cli-publication-final-20261112.xml
venv/bin/python -m scripts.preflight_openai --reasoning-effort none
```

The preflight again returned `missing_credentials`, no model results, no
generation test and unknown balance. Do not run a paid pilot until access and
the shared budget are verified. Existing frozen benchmark manifests must be
replanned explicitly when these source changes invalidate their code hash;
do not reset an existing ledger or budget to bypass that check.

An adjacent gap found in this slice was standalone `translator.py export`, which
trusted a saved publishable report without checking changed staged bytes. The
following slice addresses it. Full native CLI/refinement
resume coverage, provider/role/budget UI and CLI controls, browser tests, and
real blind/human editorial evidence are still required by the full objective.

Final full-suite result for this slice: **3,028 passed, 2 skipped, 10 deselected
in 91.31s**, recorded in `/tmp/verbaloom-cli-publication-final-20261112.xml`.
The new regression module contains 38 cases. Ruff on that module,
undefined-name checks on the touched legacy modules, `compileall`, and
`git diff --check` passed. `translator.py resume --help` was exercised without
starting a job. The live API still returned no translations and the same healthy
process session `99f0122bace340b7be6d8525ed4f3229`. The read-only production ledger
remained at 37,875 events with last timestamp 1790919396.81183. No production
restart, push/deployment, paid model call or original-book modification occurred.

## Implemented slice: approval bound to delivered bytes

Standalone export trusted an old `publishable` report even after the candidate
or source changed. It also ignored a missing/mismatched export proof and allowed
a paused job to bypass normal finalization. The validator and report writer did
not bind approval to one artifact snapshot: an output changed during validation
could receive the old result with its new hash. Nine controlled regressions
reproduced these boundaries before the initial fix.

The shared QA runner now records source/output checksums before extraction and
checks them after validation. Reports preserve those captured values rather
than recalculating an approval after a path update. Web publication, legacy CLI,
strict CLI and standalone export check the fully copied temporary candidate
against that approved hash before the exclusive commit. A later repair cannot
mutate a delivered copy. Partial diagnostics and report sidecars are not treated
as approved books.

Standalone export requires matching report run IDs, successful statuses,
literal boolean approvals, checksums and validation timestamps. It verifies the
source and requires any saved job to be completed with persisted QA approval.
The manager closes on success and publication failure. Export does not resume a
paused or failed job. Old proofless reports need validation; no old files or
checkpoints are deleted. The shared checkpoint predicate now rejects the string
`"false"`, which previously evaluated as a true approval.

This continuation first reproduced four failures caused by outdated synthetic
QA objects missing the checksum contract; those fixtures now capture the staged
bytes exactly as the real validator does. Further regressions exposed three
actual gaps: a contradictory duplicate hash, mismatched report statuses and a
truthy saved job flag. All are rejected, as are mixed/missing validation times.
Positive tests use the real QA runner for direct pairs and completed saved jobs.
Both API and legacy-CLI integration fixtures alter the candidate after QA and
verify no successful delivery, companion, TTS or overwrite occurs. These are
controlled fault injections, not physical-device or literary-quality claims.

Focused validation: **140 passed in 1.40s**. Commands:

```sh
venv/bin/python -m pytest -q tests/unit/test_publication_approval.py tests/unit/test_publication_staging.py tests/unit/test_legacy_cli_publication.py tests/unit/test_quality_cli.py tests/unit/test_strict_quality_cli_e2e.py tests/unit/test_universal_quality_assurance.py tests/unit/test_checkpoint_resume_index.py
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-publication-approval-final-20261112.xml
venv/bin/python -m ruff check src/core/quality_assurance/publication.py tests/unit/test_publication_approval.py tests/unit/test_publication_staging.py tests/unit/test_legacy_cli_publication.py
venv/bin/python -m scripts.preflight_openai --reasoning-effort none
```

The preflight still reports `missing_credentials`, unknown balance and no tested
generation. The shared USD 5 pilot has not been started. This slice does not
complete the required UI/browser flows, complete lifecycle/ownership audit,
cross-format CLI coverage, calibrated sampling, or paid/blind human A/B/C/D
comparison. No savings or provider superiority are inferred from these tests.

Final complete regression: **3,050 passed, 2 skipped, 10 deselected in 91.51s**,
recorded in `/tmp/verbaloom-publication-approval-final-20261112.xml`. Ruff passed
for the publication helper and new/extended regression modules; undefined-name
checks passed for the touched legacy modules. `compileall` and `git diff --check`
also passed. The live service remained healthy with process session
`99f0122bace340b7be6d8525ed4f3229` and no active translations. Read-only production
usage remained 37,875 events, last timestamp 1790919396.81183. No production
restart, deployment, paid request or original-book mutation occurred.

Next required work remains configuration surfaces and real browser flows, not a
claim that these publication regressions complete the full objective. Keep all
other open items in the requirement matrix, including corpus calibration and
independent paid/human editorial evidence; missing OpenAI credentials block
that paid comparison, not the remaining local implementation.

## Implemented slice: role settings and browser lifecycle

Settings now expose the existing seven-role contract without replacing the
legacy provider controls. A/B keep compatible review; C/D explicitly select
experimental full source audit and localized repair. Full audit is not the
still-pending calibrated risk/random policy. Each role has its own provider,
model, explicit none/low effort where supported and output allowance. Optional
gross-USD and HTTP-attempt limits use the existing core budget contract.

Role settings persist separately from credentials in browser storage. Invalid
numeric fields block submission; a corrupted stored snapshot does not silently
fall back to the legacy models. Choose **Current configuration** to opt out for
new jobs. Already-created jobs retain their frozen configuration and budget.

The settings API delegates validation and presets to `pipeline_config`, and
checks official model metadata using the existing provider clients. Preflight
is non-generative, bounded and reports unknown balance. It does not prove that
generation will succeed. DeepSeek catalog failures cannot become successful
preflight via the discovery catalog's offline fallback. Provider errors retain
distinct authentication, permission, missing-model, rate and connection states.

Browser testing reproduced and corrected these integration defects:

- OpenAI roles previously left their credential field hidden behind the
  unrelated DeepSeek baseline selector. Both required credential fields are
  now reachable; baseline fields are labeled separately.
- A missing baseline-provider connection disabled Start even with valid role
  configuration. Readiness now validates the role snapshot, then preflights
  its providers before creating a job. The header no longer claims a green
  baseline connection represents all role providers.
- Profile preparation checked translation-only providers. Its preflight and
  upload now include only profile preparation/review credentials. Seven failing
  API scope regressions were reproduced before the fix; malformed, empty and
  duplicate role scopes are rejected before network access.
- Preflight inherited the entire translation request. Its payload now contains
  only the pipeline, selected roles and their own credentials, not book text.
- Cancel during preflight could be ineffective or a stale response could start
  another job after restart. A per-submission cancellation fence prevents both,
  preserves the queued file and does not interrupt an unrelated server job.
- Profile model labels showed legacy DeepSeek text and a hardcoded reviewer
  label. They now use the frozen role providers/models, with reactive locale
  markers. Compact option labels preserve the experimental warning on mobile.

### Reproducible isolated browser checks

Start from the repository root:

```sh
venv/bin/python tests/standalone/pipeline_settings_server.py
```

The server binds only `http://127.0.0.1:5099`. It uses temporary checkpoints,
uploads, profiles, usage/memory paths, inert key placeholders, synthetic workers
and blocked provider HTTP. Stop this fixture server with Ctrl+C after the test.
It neither starts the production worker nor demonstrates literary quality.
In another terminal, use an existing Playwright installation and Chrome:

```sh
PLAYWRIGHT_MODULE=/absolute/path/to/playwright/index.mjs \
CHROME_EXECUTABLE=/absolute/path/to/chrome \
node tests/standalone/pipeline_settings_browser.mjs
```

If Playwright is already resolvable by Node and its browser is installed, omit
the two environment overrides. No dependency installation is performed by the
test. `PIPELINE_QA_OUTPUT` optionally changes the default evidence directory
`/tmp/verbaloom-pipeline-browser`. A loopback URL and the isolated-server header
are required; the script rejects an ordinary production service.

Observed: **13 browser checks passed**, using installed Chrome/Playwright because
the Browser plugin was not available. Viewports were 1365x900 and 390x844.
Verified all four presets, hybrid editing, budget validation, persistence,
credential visibility, cancellation/restart, upload/start/pause/resume with an
accepted checkpoint, frozen settings despite later global changes, scoped
profile form, explicit legacy rollback, five access-failure categories and
reactive labels in all seven locales. Profile start is deliberately intercepted
with HTTP 400 so the test cannot start paid generation.

Page identity, meaningful content, absence of framework overlays and uncaught
exceptions, expanded mobile role fields and error messages were checked.
Screenshots were visually inspected. Console warnings from the empty fixture
instruction list and an initial model-selector retry are documented; the latter
must resolve to populated model options. The expected injected profile HTTP 400
is not presented as an error-free real provider run.

### Validation and limits

```sh
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-pipeline-ui-regression-20261112.xml
venv/bin/python -m pytest -q tests/unit/test_pipeline_settings_api.py tests/unit/test_openai_preflight.py tests/unit/test_deepseek_preflight.py tests/test_frontend_i18n.py tests/unit/test_profile_prep_frontend_resilience.py tests/unit/test_frontend_initialization_sequence.py
venv/bin/python -m ruff check src/api/blueprints/pipeline_routes.py src/core/llm/preflight.py tests/unit/test_pipeline_settings_api.py tests/unit/test_deepseek_preflight.py tests/standalone/pipeline_settings_server.py
```

Full regression: **3,091 passed, 2 skipped, 10 deselected in 92.54s**. Final
focused checks: **108 passed in 0.54s**. Ruff, compile checks and diff whitespace
checks passed. Component templates and new locale interpolation tokens are
included in i18n regression coverage.

Read-only production checks still showed process session
`99f0122bace340b7be6d8525ed4f3229`, healthy service, no translations and 37,875 usage
events with last timestamp 1790919396.81183. The real OpenAI preflight still
returns `missing_credentials`, unknown balance and no generation tested. No
paid request, production restart or deployment was performed.

The full goal remains open. Next local requirements include app CLI role/preset
controls, complete truthful stage/cost presentation, cross-entrypoint role and
dependency audit, calibrated risk/random sampling and the remaining ownership
and cross-format coverage. Paid A/B/C/D and independent human editorial
validation remain separate gates; these UI results imply neither cost savings
nor superior translations.

The delayed/lost start-response ownership finding above is addressed by the
following slice. Profile/sample submission ownership and the remaining full
requirements are still separate acceptance gates.

## Implemented slice: durable start receipts and exact cancellation

Characterization reproduced ten failures before the change: repeated starts
created different jobs, conflicting content was accepted under the same client
identity, a restart lost the receipt, and dispatch failure had no durable job.
Twelve concurrent requests across two independent state managers produced nine
different IDs in the baseline. A further regression exposed a queued job missing
from startup recovery. These are synthetic transport/dispatch faults, not
observations of a newly failed paid book.

- `/api/translate` accepts an optional UUID `submission_id`. Its deterministic
  job ID and request fingerprint are saved in the existing checkpoint database
  before worker dispatch. The primary-key insert fences concurrent replay even
  across state-manager instances. No new queue or lifecycle table was added.
- An identical replay returns the existing receipt without updating credentials,
  unpausing, resetting counters, or dispatching again. Different content/config
  under that identity returns 409. Credentials are excluded from the fingerprint
  and remain redacted in responses. Legacy requests get collision-resistant IDs.
- Input preservation happens only after the job row is reserved. A losing
  request cannot overwrite another request's preserved source. New browser job
  records also survive automatic empty-job/error/completion cleanup, preserving
  replay identity. Explicit checkpoint deletion or age-based retention ends that
  guarantee; this is local request idempotency, not exactly-once provider billing.
- Queued work left by a prior server session uses the existing bounded startup
  recovery. TXT jobs with stored inline source can be listed and explicitly
  resumed without inventing a missing source file. Manual pause during worker
  setup is persisted and prevents entry into the translation adapter.
- A per-tab session receipt stores only the ID, minimal file metadata and pause
  intent, never credentials, prompts or source text. The browser retains it
  before dispatch and after ambiguous transport failures, including page reload.
  It never selects an arbitrary active book to cancel. Unknown receipt status
  exposes a localized Check submission action and hides unconfirmed progress.
- Start waits at most 45 seconds; status/pause requests at most 15 seconds.
  A failed lookup is not proof that the original POST was rejected: the receipt
  and pause intent remain, and checking does not send another POST. This is
  deliberately finite. If the request arrives after all observations ended,
  another check or reload is needed to reconcile it; closing a tab is not a
  server-side cancellation acknowledgment.

Reproducible checks (no provider generation):

```sh
venv/bin/python -m pytest -q tests/unit/test_translation_submission.py tests/unit/test_publication_staging.py
venv/bin/python -m pytest -q --tb=short --junitxml=/tmp/verbaloom-submission-final.xml
venv/bin/python tests/standalone/pipeline_settings_server.py
# In a second shell, using an existing Playwright installation and Chrome:
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs CHROME_EXECUTABLE=/path/to/Chrome node tests/standalone/pipeline_settings_browser.mjs
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs CHROME_EXECUTABLE=/path/to/Chrome node tests/standalone/submission_browser.mjs
```

The submission/publication suite passed **61 tests**, including the real worker's staging and
publication paths with reserved submissions and synthetic adapters. Browser
checks passed **13 settings/lifecycle checks plus 4 ownership fault scenarios**
at 1365x900 and 390x844. The new tests verify: cancel while the POST response is
held without stopping an independent job; recover a lost accepted response with
one POST; show unconfirmed status in all seven locales without resubmission; and
retain pause intent through reload before a delayed request arrives. Screenshots
were inspected; the pending message and controls fit mobile. No uncaught browser
exceptions occurred. Console transport failures/404s are intentionally injected;
empty instruction-list warnings are fixture-specific. The QA price window is
now explicitly fixed so this suite does not depend on CDMX wall-clock time.

Final regression: **3,128 passed, 2 skipped, 10 deselected in 91.83s**. The
submission/publication/provider-wait/i18n subset passed **121 tests in 0.91s**.
The old source assertion requiring an arbitrary provider-wait lookup in the
global interrupt handler was replaced with an ownership assertion. Behavioral
tests still cover exact-job cancellation in queued, running, pricing-wait,
provider-wait and rate-limited states. Browser status waits now poll observed
responses explicitly before asserting, rather than relying on an async DOM
predicate. New/affected test and QA modules passed Ruff; syntax-critical checks
passed for the legacy route/handler/persistence files. Full Ruff on those legacy
modules is not clean (existing annotation/import/broad-exception findings), and
is not claimed as passing. Node syntax, Python compilation and `git diff --check`
passed.

Browser plugin was not available; these runs used Playwright with installed
Chrome, not Android hardware. Results are in `/tmp/verbaloom-submission-browser/`
and `/tmp/verbaloom-submission-settings-browser/`. The official OpenAI preflight
still returned `missing_credentials`, unknown balance and no generation tested.
No paid pilot, production restart, push or deployment was performed.
Read-only production checks retained session `99f0122bace340b7be6d8525ed4f3229`,
no active translations and 37,875 usage events (latest 1790919396.81183). The
temporary QA server was stopped after verification; production was left running.

Remaining local audit items include CLI role controls, complete stage/cost
presentation, profile/sample ownership, pricing UI based on effective roles
rather than an unused baseline selector, and completion events that occur before
the browser attaches its current job. Calibration, A/B/C/D paid evidence and
human editorial evaluation remain open; no savings or quality win is claimed.
