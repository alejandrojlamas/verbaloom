<p align="center">
  <img src="src/web/static/verbaloom-logo-512.png" alt="VerbaLoom logo" width="176">
</p>

<h1 align="center">VerbaLoom</h1>

<p align="center">
  <strong>Local-first AI translation with editorial memory and fidelity gates.</strong>
</p>

<p align="center">
  <a href="#recommended-setup"><img alt="Local first" src="https://img.shields.io/badge/local--first-web%20app-111827"></a>
  <a href="#supported-formats"><img alt="Formats" src="https://img.shields.io/badge/formats-EPUB%20%7C%20PDF%20%7C%20DOCX%20%7C%20TXT%20%7C%20SRT-2563eb"></a>
  <a href="#book-profiles"><img alt="Book profiles" src="https://img.shields.io/badge/editorial-book%20profiles-16a34a"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/license-AGPL--3.0-7c3aed"></a>
</p>

---

VerbaLoom is a local, browser-based system for processing entire books with LLMs. It is designed for long literary and academic documents where a simple "translate this file" loop is not enough: the app chunks the book, keeps continuity memory, prepares book-specific editorial profiles, audits output against the source, repairs weak chunks, and rebuilds downloadable files.

## Applied AI Lab

This repository is a personal, experimental applied-AI lab. It explores how
bounded specialist agents, deterministic publication gates, persistent
editorial memory, and human review can make long-document LLM workflows more
traceable and reliable. It is a research and engineering project, not a hosted
translation service or a substitute for professional editorial review.

## Provenance

VerbaLoom is a personal derivative of
[hydropix/TranslateBooksWithLLMs](https://github.com/hydropix/TranslateBooksWithLLMs).
The upstream project established the translation application and provider
ecosystem; this lab extends that foundation with book-scoped editorial
profiles, source-aware audit and repair loops, resumable whole-book quality
gates, and local-first operational hardening. It is not an official upstream
release. Upstream and derivative code remain available under the repository's
[GNU AGPL v3 license](LICENSE).

Historical upstream resources remain available in the
[source repository](https://github.com/hydropix/TranslateBooksWithLLMs),
[wiki](https://github.com/hydropix/TranslateBooksWithLLMs/wiki), and
[issue tracker](https://github.com/hydropix/TranslateBooksWithLLMs/issues).
New VerbaLoom work is tracked in
[alejandrojlamas/verbaloom](https://github.com/alejandrojlamas/verbaloom).

The recommended primary model is **DeepSeek Pro** (`deepseek-v4-pro`). It is the main target for the current long-book workflow and offers a strong balance of cost, context, and quality for translation, audit, and repair. **DeepSeek V4.1 Flash** (`deepseek-flash`) is used for cheaper preflight profile preparation.

## Highlights

- **Long-book ready:** resumable chunk processing for books that take hours.
- **Source-aware quality:** candidates are audited against the original text, not only polished in isolation.
- **Editorial profiles:** names, terms, treatments, voices, and modernization policy live in book-scoped profiles.
- **Faithful modernization:** classic or old prose can be rewritten into contemporary literary language without summarizing or flattening the authorial voice.
- **Repair loops:** suspicious chunks can be reprocessed and replaced without restarting the whole job.
- **Professional EPUB delivery:** preserves source resources, declares an existing cover, keeps navigation targets valid, and forbids prose from crossing paragraph, heading, table, list, or caption boundaries during recovery.
- **Format reconstruction:** supports EPUB, PDF, DOCX, TXT, SRT, and Markdown reports.
- **Android-friendly:** designed to run locally and be used from a phone over Tailscale.

## Table of Contents

- [Applied AI Lab](#applied-ai-lab)
- [Provenance](#provenance)
- [What It Does](#what-it-does)
- [Supported Formats](#supported-formats)
- [Main Workflow](#main-workflow)
- [Universal Publication Gates](#universal-publication-gates)
- [Architecture](#architecture)
- [Multi-Agent Editorial Architecture](#multi-agent-editorial-architecture)
- [Book Profiles](#book-profiles)
- [Quality Reports](#quality-reports)
- [Recommended Setup](#recommended-setup)
- [Android / Tailscale Use](#android--tailscale-use)
- [CLI Examples](#cli-examples)
- [Configuration and Reports](#configuration-and-reports)
- [Recovery and Limitations](#recovery-and-limitations)
- [Development](#development)
- [License](#license)

## What It Does

- Translates long books and documents with LLMs.
- Modernizes same-language texts, such as classic Spanish into contemporary Spanish.
- Runs editorial refinement as part of the same flow instead of a separate disconnected step.
- Audits translated or modernized chunks against the source.
- Repairs failed or weak chunks without restarting a full job.
- Preserves continuity across very long works through compact literary memory.
- Supports resumable jobs after pauses, failures, refreshes, or server restarts.
  Work that was active at a process restart resumes automatically from its
  checkpoint; manual pauses and insufficient-credit states remain paused.
- Produces quality and fidelity reports for review.
- Can be served over Tailscale for use from Android or other devices on the tailnet.

## Supported Formats

Input support includes:

- EPUB
- PDF
- TXT
- DOCX
- SRT
- plain text-like files

Output support depends on the source and workflow, and includes:

- TXT
- EPUB
- DOCX
- SRT
- generated reports in Markdown

For EPUB and DOCX, the app can optionally use a text-first pipeline: extract readable text, process it through the long-book pipeline, and then reconstruct the output format. Rich EPUB/DOCX translation stays the default; text-first is opt-in when you explicitly want the unified long-book/plain-text flow.

In text-first mode, structure is preserved through the LLM round trip where possible: bold, italic, and hyperlinks are encoded as lightweight inline markdown (`**bold**`, `*italic*`, `[text](url)`) and decoded back into native runs/tags at rebuild time. Tables are extracted cell by cell with geometry and rebuilt with the same shape; figures and anchored images are kept. Paragraph alignment is reconciled per chunk so a paragraph merge by the model does not shift styles, images, or table cells across the rest of the document.

PDF and TXT inputs also pass through a generic **Document Structure Map** before chunking. This is separate from book glossaries: glossaries control lexical/editorial consistency, while the structure map detects layout-bearing blocks such as tables, formulas, and figure text. Detected PDF tables are normalized into Markdown-style table blocks so the LLM sees rows and numeric cells as structure instead of collapsed prose; one-token-per-line figure text is compacted into readable figure-text blocks; mechanical numeric artifacts such as `23. 75` or `1. 0 · 1020` are repaired before and after LLM calls. When structure is detected, the prompt receives explicit table-preservation rules, but books without tables are not affected.

For same-language transformations such as Modernizar, quality gates use a best-candidate policy: repairable style or layout warnings keep the best audited candidate and are recorded in the report instead of silently reverting to the untransformed source. Reverting is reserved for hard corruption such as empty output, destroyed block markers, mojibake, weird-symbol injection, or explicit source-aware rejection. The legacy behavior is available with `prompt_options.transform_fallback = "source"`.

## Main Workflow

The current long-book workflow is:

1. **Ingest**
   - Detect file type.
   - Extract text.
   - Normalize encoding, mojibake, OCR-like garbage, wrapper tags, stray markdown emphasis, and obvious scan artifacts.

2. **Chunk**
   - Split into manageable chunks.
   - Preserve source order.
   - Track context before and after each chunk.
   - Keep progress and checkpoints.

3. **Translate or Transform**
   - Translate to a target language, or transform within the same language.
   - For literature, use continuity memory for names, relationships, voices, and recurring entities.

4. **Editorial Pass**
   - Improve flow, rhythm, style, paragraphing, and readability.
   - Preserve content and source order.
   - Avoid summaries, censorship, omissions, or unsupported additions.

5. **Independent Audit**
   - Compare candidate output against the source.
   - Check content fidelity, names, numbers, order, omissions, additions, profile compliance, and style goals.

6. **Repair**
   - If the audit returns `warn` or `fail`, run a targeted repair pass.
   - Repairs receive explicit issue and score guidance.
   - If a later repair is worse, the pipeline keeps the best audited candidate instead of blindly accepting the last result.

7. **Assemble**
   - Rebuild the final file.
   - Save output and reports.
   - Keep logs and checkpoints for debugging and continuation.

8. **Prove and Publish**
   - Build a stable whole-book manifest.
   - Run ten deterministic publication gates.
   - For EPUB, prove cover, navigation, package resources, DOM block identity, target-language metadata, and EPUBCheck validity.
   - Publish atomically only when every translatable unit is approved.
   - Quarantine failed output and preserve only the rejected units for repair.

## Universal Publication Gates

VerbaLoom does not treat "an output file exists" as completion. Every normal web job
and every strict CLI job now runs a format-neutral whole-book quality contract.
The required invariant is:

```text
translatable units = approved units = exported units
```

The ten gates cover extraction, segmentation, translation response hygiene,
target language, entities and verifiable data, semantics, structure, typography
and spacing, metadata, and the final artifact. EPUB deliveries additionally
preserve package resources and run EPUBCheck. A critical finding leaves the job
partial, keeps its checkpoint, writes a selective repair plan, and cannot
overwrite a previous valid output.

Every unit has a deterministic ID and source/structure checksum, so reports can
trace reader-visible output back to the source and resume only rejected work.
See [Universal Translation Quality Assurance](docs/QUALITY_ASSURANCE.md),
[Complete-Book Translation Pipeline](docs/TRANSLATION_PIPELINE.md), and
[Recovery, Resume, and Selective Repair](docs/RECOVERY_AND_RESUME.md).

## Architecture

See [Architecture and Change Guide](docs/ARCHITECTURE.md) for module ownership,
runtime state, refactoring rules, and the validation matrix used for changes to
shared translation, API, and frontend boundaries.

## Multi-Agent Editorial Architecture

The editorial system is implemented as a bounded multi-agent pipeline. It is not an open-ended debate between agents; each step has a narrow role and a clear output contract.

Core agents:

- **Translator / Modernizer**
  - Produces the first candidate chunk.
  - Uses the active provider and model, usually `deepseek-v4-pro`.

- **Continuity Memory Agent**
  - Maintains compact state for long books.
  - Tracks names, entities, character continuity, recent context, and section/chapter context.

- **Editorial Refinement Agent**
  - Improves style and readability while preserving source meaning.
  - Handles literary polish, OCR cleanup, paragraph recovery, and flow.

- **Fidelity Supervisor**
  - Checks source fidelity independently from style.
  - Detects omissions, additions, changed facts, changed numbers, changed names, censorship, and unsupported softening.

- **Profile Auditor**
  - Applies the active book profile.
  - Scores dimensions such as content fidelity, syntactic modernization, target locale, authorial voice, voice differentiation, glossary compliance, and profile isolation.

- **Repair Editor**
  - Receives the source, candidate, audit results, score-based priorities, and approved profile guidance.
  - Repairs only what failed.

- **Glossary Discovery Agent**
  - Suggests recurring terms, treatments, names, formulas, archaic forms, and style decisions for a profile.
  - Suggestions are pending by default, not automatically global.

- **Preflight Profile Preparation**
  - Before the expensive translation/modernization run, the web UI can scan the selected book with **DeepSeek V4.1 Flash** (`deepseek-flash`).
  - The preflight scan creates a book-scoped editorial profile, approves only mechanically safe preserve-as-written entries, and stores all other discoveries as pending suggestions.
  - The preparation runs as a visible background job: the UI shows stage/chunk progress, then auto-saves and selects the generated profile when it finishes.
  - The generated profile appears in the **Editorial profile** selector and is passed into the main run, where **DeepSeek Pro** (`deepseek-v4-pro`) remains the recommended model for translation, audit, and repair.

## Book Profiles

The engine is generic. Book-specific decisions live in profiles.

Generic engine responsibilities:

- chunking
- provider calls
- retry and fallback handling
- checkpoints
- source fidelity checks
- structure validation
- report generation
- output assembly
- encoding and artifact cleanup
- format conversion and reconstruction

Profile responsibilities:

- editorial policy
- book-specific prompts
- approved terms and treatments
- character voice rules
- modernization intensity
- profile-specific detectors
- profile-specific audit rubric
- pending glossary suggestions

Automatic profile preparation:

1. Open **Transform text**.
2. Drop the source book.
3. Choose the profile goal: faithful translation, audiobook, modernization, explanation, or literary polish.
4. Click **Analyze with DeepSeek Flash**.
5. Watch the progress bar while extraction, local scan, reviewer batches, and Flash discovery chunks run.
6. The generated profile is saved automatically and selected in **Editorial profile**.
7. Start the transformation or translation flow with that profile active.

This keeps the engine generic: DeepSeek Flash performs cheap discovery, but
book-specific decisions still live under `profiles/<profile_id>/`.
The selected goal changes the business rules and budgets: faithful translation
looks for canonical names plus terms that must be translated consistently;
audiobook profiles look for captions, references, page furniture, and listening
hygiene; modernization profiles look for archaisms, treatments, syntax, and
voice. These budgets scale with the source length and candidate variety instead
of forcing every profile into the same approved/pending counts.

Profile layout:

```text
profiles/
  common/
    profile.yml
    editorial_policy.md
    prompts/
    glossary/
  audiobook_faithful/
    profile.yml
    editorial_policy.md
    prompts/
      translation.txt
      translation_refinement.txt
      audit.txt
      repair.txt
      glossary_discovery.txt
    glossary/
  book_specific_profile/
    profile.yml
    editorial_policy.md
    prompts/
      modernize.txt
      voice_restoration.txt
      audit.txt
      repair.txt
      glossary_discovery.txt
    glossary/
      terms.yml
      treatments.yml
      phrases.yml
      character_voices.yml
      pending_suggestions.yml
```

Rules specific to one book should not be hardcoded into the engine. If a stable equivalence is needed, it belongs in that book profile. If a rule might be useful across books, it should first be proposed for a common profile and reviewed before use.

Deterministic lexical heuristics are also source-language scoped. The generic
engine loads them from `src/core/glossary/lexical_policies.json`; English term
lists are never applied to an explicitly German, Czech, Greek, or other source.
Unknown languages use only language-neutral structure checks and defer lexical
decisions to the profile reviewer.

Audit dimensions follow the same boundary. Every profile receives fidelity,
structure, glossary, and isolation checks; goal-specific dimensions are added
from its business rules. A profile may add extra dimensions through
`audit_score_fields`, but no book or locale-specific dimension is global.

## Audiobook Profile

The included `audiobook_faithful` profile keeps the translation faithful while preparing companion artifacts for listening.

When this profile is active, the normal translated output is still produced. After the final artifact hygiene audit, the app also creates:

- `(... Audiobook).txt`: clean text for TTS.
- `(... Audiobook).epub`: structured visual companion when the main requested output is EPUB. It is cloned from the translated publication rather than rebuilt from plain text, so the cover, illustrations, captions, navigation, fonts, and image placement remain intact.
- `(... Audiobook report).json`: counts of removed links, moved notes/references, integrated image captions, and structurally preserved visual resources.

The audiobook sanitizer is deterministic and profile-scoped. It removes page furniture, source-download watermarks, raw links, EPUB note anchors, and inline note calls from the TXT listening flow. Notes, references, and credit-only captions move to an appendix. Informative image captions are kept near the text as concise image descriptions, without inventing visual details. The EPUB companion follows a separate preservation contract: every XHTML document and visual resource remains byte-identical, while a blocking local gate proves the cover declaration, image hashes, references, DOM positions, captions, and spine before publication.

For image-heavy books such as film histories, use **Profiles → Prepare editorial profile → Primary use: Audiobook** before translating. The generated book profile keeps its own glossary and editorial map while enabling the same audiobook companion policy.

## Quality Reports

The app can generate:

- editorial quality reports
- source fidelity reports
- profile audit summaries
- chunk-level warnings and failures
- repair decisions
- glossary suggestions
- continuation and progress logs

Reports are meant to make long jobs auditable. A failed chunk is not silently dropped; it is either repaired, kept as the best available candidate, or reported.

## Resumability

Long translations can pause or fail because of:

- rate limits
- network issues
- model errors
- browser refreshes
- app restarts
- machine sleep

The checkpoint system stores chunk progress so jobs can resume from saved state instead of restarting from zero. Refinement and transformation jobs also track chunk-level progress.

## Recommended Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure DeepSeek Pro

Copy `.env.example` to `.env` and set:

```bash
LLM_PROVIDER=deepseek
DEEPSEEK_MODEL=deepseek-v4-pro
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_API_KEY
DEEPSEEK_OFF_PEAK_ONLY=true
```

Do not commit real API keys. The app supports multiple comma-separated keys for providers that use key rotation.

DeepSeek's official high-price periods are blocked by default. A translation
that crosses into one is checkpointed without sending more tokens, remains
visible as a scheduled wait, and resumes automatically when regular pricing
returns. The UI shows the next availability in Mexico City time; see
[Provider configuration](docs/PROVIDERS.md#regular-price-schedule) for the
current schedule and opt-out setting.

Security defaults are intentionally local-first: the server binds to
`127.0.0.1`, REST and WebSocket traffic is same-origin, uploads are bounded,
and JSON file references must stay inside app-managed storage. A custom
OpenAI-compatible endpoint never receives a key loaded from `.env` unless its
origin is explicitly listed in `VERBALOOM_TRUSTED_KEY_ENDPOINTS`; pass a key for that
request instead. Network binding requires the separate
`VERBALOOM_ALLOW_NETWORK_BIND=true` opt-in and should sit behind a trusted proxy.
The application does not implement user authentication: the opt-in changes
reachability only. Do not expose Flask directly to a LAN, tailnet, or the
Internet; use an identity-aware authenticated proxy for any remote access.

Existing installations that still define the legacy `TBL_*` environment names
are read as compatibility inputs during migration. `VERBALOOM_*` is the
canonical prefix and the only prefix used in new configuration examples.

### 3. Run the app

```bash
python translation_api.py
```

Open:

```text
http://127.0.0.1:5000
```

If using the bundled launcher scripts, use:

```bash
./start.sh
```

## Android / Tailscale Use

The app can be exposed to a tailnet with Tailscale Serve. Treat tailnet
membership as network reachability, not application authentication; restrict
the tailnet ACL to the intended user/device or add an identity-aware proxy. A
typical local setup proxies the app to a user-owned MagicDNS host:

```text
https://<device>.<tailnet>.ts.net/
```

The app itself still runs locally, usually on:

```text
http://127.0.0.1:5000
```

Tailscale only exposes it to devices in the same tailnet.

The installed app exposes a token-free diagnostic endpoint that reports the
current Tailscale IP and fallback URLs:

```bash
curl http://127.0.0.1:5000/api/mobile-access
```

From the phone, open `/android` on the recommended URL first. The small diagnostic
page works without the JavaScript application bundle and records only bounded
request metadata. After the phone reaches it, the API returns
`external_android_seen: true` and the latest external Android event. Local
Android-user-agent simulations and requests sent through Tailscale by the Mac
itself do not count as phone verification.

An optional proxied application route can look like:

```text
https://<device>.<tailnet>.ts.net/verbaloom
```

If the phone cannot resolve MagicDNS, use the `recommended_url` returned by
`/api/mobile-access`. The Android device must appear online in `tailscale status --json`
before a successful Mac-side route check can be treated as end-to-end phone
verification.

## CLI Examples

The recommended strict CLI is `translator.py`. It stages output, runs all gates,
and publishes atomically:

```bash
./translator.py inspect book.epub --source-language auto --target-language Spanish

./translator.py translate book.epub \
  --provider deepseek \
  --model deepseek-v4-pro \
  --source-language English \
  --target-language Spanish \
  --target-locale es-MX \
  --output book-es.epub

./translator.py validate RUN_ID
./translator.py repair RUN_ID
./translator.py resume RUN_ID
./translator.py report RUN_ID
```

Use `./translator.py translate ... --dry-run` to inventory the source without a
model call. `translate.py` remains available as the backward-compatible legacy
CLI.

### Legacy CLI

Translate with DeepSeek Pro:

```bash
python translate.py -i book.epub --provider deepseek \
  --deepseek_api_key YOUR_DEEPSEEK_API_KEY \
  -m deepseek-v4-pro \
  -sl English \
  -tl Spanish
```

Modernize a Spanish text with the Quijote profile:

```bash
python translate.py -i quijote.txt --provider deepseek \
  --deepseek_api_key YOUR_DEEPSEEK_API_KEY \
  -m deepseek-v4-pro \
  -sl Spanish \
  -tl Spanish
```

Most profile-driven workflows are easiest from the web UI because the UI passes the profile and audit options together.

## Configuration and Reports

Copy and adapt `config/quality_assurance.example.yml`, then set:

```bash
export QUALITY_ASSURANCE_CONFIG=/absolute/path/to/quality_assurance.yml
```

The file centralizes translation locale, segmentation context, bounded retries,
language residue thresholds, entity categories, warning policy, model routing,
intermediate retention, and atomic publication. Provider credentials remain in
environment variables or `.env`; they are never written into manifests or
reports.

Each run writes the following under `data/quality_runs/<run_id>/`:

- `translation_manifest.json`
- `translation_report.json` and `translation_report.html`
- `failed_units.jsonl`
- `entity_diff.json`
- `language_report.json`
- `structure_report.json`
- `export_report.json`

The HTML report is the human review entrypoint. It includes coverage, chapters,
models, token/cost data when available, repaired and failed units, residual
source language, entity and structural discrepancies, EPUBCheck, exclusions,
all ten gates, and remaining limitations.

## Recovery and Limitations

Strict checkpoints enter `validating` after all chunks finish and become
`completed` only after whole-book validation. Resume skips approved chunks even
when an earlier index failed. Repair clears only rejected candidates and keeps
their evidence.

Known limitations:

- Literary beauty and culturally preferred wording still require human sample
  review; deterministic gates prove integrity, not taste.
- When an old or external artifact has no checkpoint/native manifest, validation
  aligns source and output blocks by structural order and reports that weaker
  provenance explicitly.
- PDF extraction quality depends on the source text layer; image-only scans need
  OCR before reliable semantic translation.
- DOCX structural validation proves package readability and table/image counts,
  but exact run-level styling fidelity remains format-adapter dependent.
- `--max-cost` blocks publication when the recorded budget is exceeded; it does
  not yet cancel an already in-flight provider request.
- Very large EPUB inspection currently builds the native unit snapshot in
  memory. Chunk translation remains checkpointed and resumable, but extraction
  is not yet fully streaming.

## Development

Run focused tests:

```bash
python -m pytest -q tests/unit/test_book_profiles.py \
  tests/unit/test_faithful_modernize.py \
  tests/unit/test_translation_extractor.py \
  tests/unit/test_locale_quality.py \
  tests/unit/test_quality_alert_repair.py
```

Run the universal publication contract tests:

```bash
venv/bin/python -m pytest -q \
  tests/unit/test_universal_quality_assurance.py \
  tests/unit/test_quality_cli.py \
  tests/unit/test_prompt_injection_guard.py \
  tests/unit/epub/test_publication_gate.py \
  tests/unit/epub/test_dom_boundaries.py
```

Run the complete regression gate:

```bash
venv/bin/python -m pytest -q
venv/bin/python -m compileall -q src scripts tests
git diff --check
```

Run frontend i18n checks:

```bash
python -m pytest -q tests/test_frontend_i18n.py
```

Useful scripts:

```bash
python scripts/profile_init.py --profile my_new_book
python scripts/glossary_discover.py --profile my_new_book
python scripts/glossary_report.py --profile my_new_book
python scripts/glossary_approve.py --profile my_new_book
```

## Packaging Notes

When running or adapting the repository:

- remove or sanitize real `.env` values
- do not publish real API keys
- keep generated translations, uploads, reports, checkpoints, and private book profiles out of source control
- keep `.env.example` as the documented template

Runtime artifacts are ignored by default. Review staged files before every
public release because source documents and provider credentials remain the
operator's responsibility.

## Documentation

Additional guides live in `docs/`, including:

- architecture and change ownership
- provider setup
- Docker deployment
- CLI usage
- notifications
- API key rotation
- glossary workflow
- book profiles and glossaries
- troubleshooting
- universal quality assurance and publication gates
- complete translation pipeline
- recovery, resume, and selective repair

## License

[GNU Affero General Public License v3.0](LICENSE). This derivative preserves
the upstream license and attribution.
