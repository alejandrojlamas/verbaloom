# Architecture and Change Guide

This document describes the ownership boundaries of the installed application.
Use it to decide where a change belongs before adding logic to a large
orchestration module.

## Runtime Shape

The three supported root entrypoints are:

- `translation_api.py`: starts the local web application.
- `translate.py`: runs translation workflows from the command line.
- `launcher.py`: desktop-friendly launcher.

The persistent macOS service calls `scripts/run_verbaloom_server.sh`, which
starts `translation_api.py` with the project virtual environment and suppresses
automatic browser opening.

The request path is:

```text
browser
  -> src/web
  -> src/api/blueprints
  -> src/api/handlers.py
  -> src/core
  -> src/persistence and format adapters
```

Dependencies should point down this list. Core modules must not import Flask,
web templates, or API blueprints.

## Before and After the Universal Quality Contract

The earlier flow could complete after a format adapter returned an output file.
Individual chunk guards existed, but non-EPUB formats did not share one final,
deterministic whole-book proof. A complete checkpoint could also transition to
`completed` before final publication validation.

The hardened flow adds a format-neutral contract:

```text
web or strict CLI
  -> JobEngine phases
  -> format adapter and checkpoint
  -> quality_assurance.extractors (stable manifest)
  -> quality_assurance.validators (unit and document evidence)
  -> quality_assurance.gates (ten mandatory decisions)
  -> quality_assurance.reports (atomic audit trail)
  -> atomic publish OR partial quarantine and sparse repair
```

`src/core/quality_assurance/` owns this contract. It does not call Flask and it
does not translate prose. It consumes the exact native EPUB publication report,
the persisted chunk checkpoint, or a clearly labeled artifact-alignment
fallback in that order.

## Universal Quality Modules

| Module | Responsibility |
| --- | --- |
| `models.py` | Stable units, manifests, checksums, validation and model metadata |
| `config.py` | Central nested thresholds, retries, model routing, and export policy |
| `extractors.py` | Format-neutral manifests from native reports, checkpoints, or artifacts |
| `validators.py` | Deterministic coverage, language, entity, semantic, typography, metadata, and package checks |
| `gates.py` | Exactly ten publication gates and selective repair strategies |
| `reports.py` | Atomic JSON, JSONL, and HTML reports |
| `runner.py` | Whole-book orchestration and checkpoint repair marking |
| `cli.py` | Strict inspect, translate, resume, validate, repair, export, and report commands |

The web handler remains a compatibility orchestrator. It delegates reusable
quality policy to this package through `JobEngine` rather than embedding new
heuristics in Flask code.

EPUB reconstruction adds two format-native layers before that universal
contract runs:

| Module | Responsibility |
| --- | --- |
| `epub/structure_safe_fallback.py` | Keeps block placeholders immutable and isolates any inline-tag recovery to one readable block |
| `epub/professionalize.py` | Declares an existing cover and supplies conservative reading semantics without replacing rich publisher CSS |
| `epub/dom_boundaries.py` | Proves block and text-slot identity against the source DOM |
| `epub/publication_gate.py` | Verifies resources, cover, navigation, language, links, complete unit coverage, and EPUBCheck before publication |

## Silent-Failure Boundaries

The following boundaries receive explicit proof:

- extraction to segmentation: expected units, IDs, paths, order, checksums
- model response to candidate: non-empty response, output protocol, target language
- candidate to review: source facts, entities, negations, locale, profile evidence
- review to assembly: only accepted candidate text and source-proven structure
- assembly to publication: full counts, package structure, metadata, readability, standard validators

A failure at any boundary produces a blocked/partial state, report evidence, and
a bounded repair plan. It cannot be converted to a successful fallback by
copying the source text.

## Layer Ownership

| Layer | Owns | Does not own |
| --- | --- | --- |
| `src/web` | UI rendering, browser state, i18n, user interaction | Translation policy or persistence |
| `src/api/blueprints` | HTTP parsing, response shapes, endpoint-specific validation | Long-running translation algorithms |
| `src/api/handlers.py` | Job coordination and bridging API events to core workflows | Reusable text heuristics |
| `src/core` | Translation, refinement, quality policy, profiles, format-neutral processing | Flask request or session state |
| `src/core/adapters` and format packages | File-format extraction and reconstruction | Provider discovery or web behavior |
| `src/persistence` | Checkpoints and durable job state | Editorial decisions |

## High-Change Seams

### Configuration and provider models

`src/api/blueprints/config_routes.py` composes configuration endpoints. Keep
provider-specific model discovery in `provider_model_routes.py` and phone/Tailscale
diagnostics in `mobile_access_routes.py`. Both modules receive their dependencies
from the blueprint registration function so their tests do not need the running
application.

Do not add provider model calls or mobile event storage back to
`config_routes.py`.

### Translation quality guards

`src/core/translator.py` is the compatibility orchestrator for the legacy text
pipeline. Pure alert policy, text normalization, model selection, and repair
instructions live in `src/core/quality_guard.py`.

Some existing callers import private guard helpers from `translator.py`. The
orchestrator reexports those helpers for compatibility. New code should import
them from `quality_guard.py`; network calls and retry orchestration remain in
`translator.py`.

### Frontend lifecycle

`src/web/static/js/index.js` owns the initial WebSocket connection after event
handlers are wired. `lifecycle-manager.js` handles page lifecycle only and must
not start a second connection.

All user-facing strings must follow the reactive i18n rules in `CLAUDE.md` and
exist in every supported locale.

## Local State

The application is local-first. These paths are runtime data, not source code:

| Path | Purpose |
| --- | --- |
| `.env` | Provider credentials and local runtime configuration |
| `data/` | Job, usage, glossary, and translation-memory databases |
| `profiles/` | Book-scoped editorial profiles |
| `translated_files/` | Uploads, reports, and downloadable outputs |
| `output/` | Local audits and generated diagnostics |
| `test-results/`, `.playwright-cli/` | Local QA artifacts |

Generated output and timestamped profile backups are excluded from repository
statistics and Git status. Never move provider credentials into source, tests,
or documentation.

## Validation Matrix

Run the smallest relevant checks during development, then the full suite after a
shared boundary changes.

| Change | Focused validation |
| --- | --- |
| Quality guard or translator | `venv/bin/python -m pytest -q tests/unit/test_quality_guard.py tests/unit/test_quality_alert_repair.py tests/unit/test_editorial_quality.py` |
| Configuration or provider discovery | `venv/bin/python -m pytest -q tests/unit/test_provider_model_routes.py tests/unit/test_state_management.py` |
| Mobile access | `venv/bin/python -m pytest -q tests/unit/test_mobile_access_routes.py` |
| Frontend initialization | `venv/bin/python -m pytest -q tests/unit/test_frontend_initialization_sequence.py` and `node --check` on changed JavaScript |
| Locale changes | `venv/bin/python -m pytest -q tests/test_frontend_i18n.py` plus an in-browser language switch |

The broad regression gate is:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m compileall -q src scripts tests
git diff --check
```

## Runtime Verification

After restarting the service, verify the local application without spending
provider tokens:

```bash
curl --fail http://127.0.0.1:5050/api/health
curl --fail -A 'Mozilla/5.0 (Linux; Android 15)' \
  http://127.0.0.1:5050/api/mobile-access
```

For Tailscale, verify both the declared Serve handler and the final HTTPS path:

```bash
tailscale status --json
tailscale serve status --json
curl --fail https://<tailnet-host>/verbaloom/api/health
```

If MagicDNS is unavailable on the Mac, use `curl --resolve` with the tailnet IP
to validate the same Host header and TLS name. A Mac-side HTTP success does not
prove the phone is connected; the Android peer must be online and the final URL
must be opened on that device before claiming phone verification.

## Refactoring Rules

- Preserve endpoint response shapes and callback contracts before moving code.
- Add characterization tests before extracting behavior from an orchestrator.
- Keep compatibility imports until all known callers have migrated.
- Extract pure policy and parsing before asynchronous orchestration.
- Do not combine refactoring with unrelated output or profile changes.
- Keep `translator.py`, `handlers.py`, and format translators focused on
  coordination; new reusable policy belongs in a dedicated core module.
