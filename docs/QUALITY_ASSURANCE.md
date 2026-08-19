# Universal Translation Quality Assurance

VerbaLoom treats a translated book as publishable only after a deterministic,
whole-book validation run proves coverage, language, entity, semantic,
structural, typographic, metadata, and artifact integrity. An LLM audit may add
evidence, but it never replaces deterministic checks.

The implementation lives in `src/core/quality_assurance/` and is used by both
the web job handler and the strict `translator.py` CLI.

## Publication Rule

The invariant for a successful run is:

```text
translatable units = approved units = exported units
```

Any difference blocks publication. Explicit exclusions are allowed only when
the block classifier records a reason such as formula, URL, code, page
furniture, or intentionally preserved foreign quotation.

## The Ten Gates

| Gate | Validates | Blocking examples |
| --- | --- | --- |
| 1. Extraction | Source inventory and readable units | No units, missing source unit |
| 2. Segmentation | Stable IDs, continuous order, checksums | Duplicate ID, missing index, changed checksum |
| 3. Translation | Candidate coverage and response hygiene | Empty response, source returned unchanged, model protocol leak, duplicate unrelated output |
| 4. Language | Unit, section, and document target-language dominance | Narrative block in source language, excessive section/document residue |
| 5. Entities | Verifiable data and locked glossary entities | Changed number, date, percentage, currency, measurement, DOI, ISBN, URL, scientific name, or locked name |
| 6. Semantics | Source-aware deterministic and prior audit evidence | Probable omission/addition, lost negation, open fidelity rejection |
| 7. Structure | Order, resources, format structure | Extra/missing blocks, broken EPUB links, lost DOCX tables/images, changed EPUB spine |
| 8. Typography | Reconstruction and spacing | Joined sentences or words caused by lost DOM whitespace |
| 9. Metadata | Locale and package metadata | Invalid BCP 47 target code, wrong EPUB language metadata |
| 10. Final artifact | Readability and standard validation | Missing/empty output, mojibake, invalid package, EPUBCheck error |

Critical and high findings are blocking. Medium and low findings are warnings
when `validation.allow_warnings` is true; strict CLI runs can block warnings by
omitting `--allow-warnings`.

## Deterministic Unit Contract

Every `TranslationUnit` records:

- deterministic `unit_id`, independent of translated text
- document and parent identity
- source order and structural path
- source, translated, reviewed, and final text
- content type and processing policy
- inline tag metadata and boundary whitespace
- source and structure SHA-256 checksums
- per-stage status and retry count
- validation results and model/token metadata
- source checkpoint or EPUB DOM reference

The manifest is sufficient to trace a reader-visible unit back to its source.
Changing source text or structural metadata after extraction invalidates its
checksum and blocks the run.

## Language Validation

Language checks are deterministic and seeded for reproducibility. They run at
unit, parent section, and whole-document scope. The target-language gate also
inspects script ratios and mixed-language residue inside a unit.

Short CJK and Korean units use a dense-script evidence threshold rather than a
space-oriented character minimum. Exact foreign quotations and bibliography or
critical-apparatus blocks may remain unchanged when configured. Narrative text,
however, cannot silently fall back to the source.

Default thresholds:

```yaml
language_validation:
  enabled: true
  max_source_ratio_document: 0.01
  max_source_ratio_unit: 0.15
  min_chars: 40
  allow_named_entities: true
  allow_quoted_foreign_text: true
```

## Entity Validation

Deterministic extractors compare exact multisets for numbers, numeric dates,
percentages, currencies, measurements, DOI/ORCID identifiers, URLs, ISBNs, and
scientific names. Locked profile or glossary entities define allowed target
forms. Unlocked proper-name candidates are reported for review instead of being
blindly blocked because conventional localization and transliteration vary by
language.

## Semantic Validation

The final gate combines:

1. deterministic length, negation, entity, and response-contract checks;
2. source-aware fidelity decisions already produced during translation;
3. profile audit evidence recorded in the checkpoint;
4. the strict EPUB source/output publication audit when applicable.

Length is never used as the only coverage control. It is one strong anomaly
signal combined with stable identity, full counts, source-language detection,
entities, prior fidelity evidence, and structural comparison.

## EPUB Validation

EPUB-to-EPUB jobs use the strict source/output publication gate. It verifies the
spine, archive resource set, images, CSS/font resources, XHTML structure, DOM
text slots, internal links, navigation targets, language metadata, duplicate
IDs, spacing, source/output units, and EPUBCheck. If the source contains an
obvious cover image, the output must declare it in the package and expose a
valid cover-page target. A non-empty paragraph, heading, table cell, caption,
or list item may not become empty or absorb a neighboring block.

Placeholder recovery is block-safe. Paragraph, heading, table, list, and
caption boundaries never enter proportional alignment; only inline formatting
inside one readable block can be realigned. A provider response that cannot
preserve that contract remains a failed resumable unit and is not published.

Other workflows that produce an EPUB still run package parsing and EPUBCheck.
An EPUBCheck error blocks publication. If EPUBCheck is unavailable, the report
records a warning and internal package validation still runs.

Audiobook EPUB companions are never rebuilt from the sanitized narration TXT.
They are cloned atomically from the translated EPUB and pass a dedicated visual
preservation gate. The gate requires the same archive resources, image bytes,
image references, DOM positions, nearby captions or credits, cover declaration,
spine order, and XHTML bytes. Only the package title may receive the
`(Audiobook)` suffix. A mismatch aborts companion publication; the clean TXT
remains available for TTS.

## Reports

Every run writes atomically to `data/quality_runs/<run_id>/` by default:

```text
translation_manifest.json
translation_report.json
translation_report.html
failed_units.jsonl
entity_diff.json
language_report.json
structure_report.json
export_report.json
```

`translation_report.html` is the human review entrypoint. It contains the
executive status, languages, models, cost, elapsed time, chapter/section and
unit counts, coverage, repaired and failed units, residual source-language
ratio, entity and structure findings, EPUBCheck result, ten gates, exclusions,
open issues, and known limitations.

`export_report.json` also records the final path, file size, and SHA-256 digest.
When a failed web artifact is renamed with a `[partial]` prefix, all report paths
are rewritten to the quarantine path.

Compact `translation_report.json` example:

```json
{
  "run_id": "run_8d2a7f9c",
  "status": "PASSED",
  "publishable": true,
  "source": {"format": "epub", "language": "English"},
  "output": {"format": "epub", "language": "Spanish", "locale": "es-MX"},
  "coverage": {
    "translatable": 842,
    "translated": 842,
    "reviewed": 842,
    "audited": 842,
    "approved": 842,
    "exported": 842,
    "percent": 100.0
  },
  "failed_segments": 0,
  "source_language_ratio": 0.0,
  "entity_discrepancies": 0,
  "quality_gates": [
    {"gate": "extraction", "status": "PASSED", "issues": []},
    {"gate": "segmentation", "status": "PASSED", "issues": []}
  ],
  "limitations": []
}
```

## Selective Repair

Blocking unit findings generate `failed_units.jsonl` entries with:

- unit and checkpoint index
- issue evidence
- repair strategy
- configured maximum repair rounds

Only rejected checkpoint indices are cleared. Approved chunks remain intact and
are skipped on resume. Strategies include strict target-language translation,
entity-preserving repair, source-aware semantic repair, deterministic structure
repair, and strict retranslation.

## Configuration

Start from `config/quality_assurance.example.yml` or set
`QUALITY_ASSURANCE_CONFIG` to another YAML file. Web jobs may override the same
nested options in their persisted job configuration.

Important controls:

- `validation.allow_warnings`
- language residue thresholds
- protected entity categories
- maximum translation/review/audit/repair attempts
- intermediate-file retention
- atomic publication and overwrite policy
- translator, reviewer, and auditor model routing

## Manual Sample Review

After gates pass:

1. Open `translation_report.html` and review every open warning.
2. Inspect the highest-risk units: long blocks, repaired units, entity-rich
   passages, quotations, tables, formulas, and chapter boundaries.
3. Compare a beginning, middle, and ending sample against the source.
4. Open the final file in at least one real reader.
5. For EPUB, inspect the cover, navigation, images, notes, internal links, and
   metadata in addition to EPUBCheck.

Passing gates proves the configured deterministic contract. It does not replace
human literary judgment about beauty, voice, or culturally preferred wording.

## Commands

```bash
./translator.py inspect input.epub --source-language auto --target-language Spanish

./translator.py translate input.epub \
  --provider deepseek \
  --model deepseek-v4-pro \
  --source-language English \
  --target-language Spanish \
  --target-locale es-MX \
  --output output.epub

./translator.py validate RUN_ID
./translator.py repair RUN_ID
./translator.py resume RUN_ID
./translator.py report RUN_ID
```

Exit code `0` means publishable, `2` means blocked by quality gates, and `1`
means an execution or configuration error.
