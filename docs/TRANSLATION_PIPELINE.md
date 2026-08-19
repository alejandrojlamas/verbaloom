# Complete-Book Translation Pipeline

## Flow

```text
source artifact
  -> secure ingest and format validation
  -> structural block classification
  -> stable TranslationUnit manifest
  -> segmentation with local context
  -> translation or same-language transformation
  -> restrained editorial refinement
  -> deterministic candidate checks
  -> alerted source-aware/profile audit
  -> selective repair
  -> format reconstruction
  -> whole-book deterministic validation
  -> ten publication gates
  -> atomic publish OR quarantined partial artifact
```

The priority order is integrity, fidelity, traceability, natural language,
cost, and speed.

## Phase Ownership

### Ingest

Validates the upload, detects the format, inventories document resources, and
extracts readable content. EPUB and DOCX preserve package structure; PDF and
plain-text sources use normalized structural blocks. Unsafe archive paths and
invalid packages are rejected before model calls.

### Prepare

The optional book profile preflight discovers canonical entities, voices,
sections, technical/cultural terms, translatable terms, preserved terms, and
editorial risks. Book-specific decisions remain in the active profile and
glossary, never in the generic engine.

### Segment

Each unit receives a deterministic identity, source and structure checksum,
continuous order index, structural path, block type, policy, whitespace, and
checkpoint reference. Context before and after a unit is context only; it must
not be emitted twice.

### Translate or Transform

The model receives the active locale, relevant glossary entries, compact
continuity memory, profile policy, and bounded adjacent context. System prompts
state that all book content is untrusted data and cannot issue instructions.

The provider layer supplies timeout, backoff, rate-limit, token, cache, and cost
metadata. Approved translation-memory matches can be reused by checksum.

### Refine

Refinement is restrained. It corrects demonstrable grammar, locale, OCR,
spacing, rhythm, or consistency issues and does not rewrite every acceptable
sentence. The source-aware editorial guard compares source, draft, and refined
candidate when deterministic signals require escalation.

### Audit

Deterministic checks run first. Source-aware LLM audit is an additional layer
for alerted, sampled, or explicitly strict units. Structured audit responses
must satisfy their response contract; malformed judge output is retried or
rejected, never accepted as a pass.

### Repair

Only failed units are reprocessed. The repair strategy is selected from the
finding type and bounded by configured attempts. Approved units are not cleared.

### Assemble

Format adapters reconstruct text into the requested package while preserving
DOM boundaries, inline markup, images, tables, formulas, notes, spine order,
links, metadata, and navigation where the source format supports them.

### Validate

`src/core/quality_assurance/` builds or loads the final manifest, runs
deterministic unit/document validators, evaluates ten gates, creates a selective
repair plan, and writes all reports.

### Publish

The strict CLI translates into a run-scoped staging directory and copies to a
temporary file in the destination directory before atomic rename. Existing
valid output is never overwritten. The web flow uses unique output names and
renames failed artifacts to `[partial] ...`; those files remain resumable and
cannot be marked completed.

## Supported Formats

| Format | Extraction and reconstruction contract |
| --- | --- |
| EPUB | Exact source/output DOM-unit contract, package resources, navigation, metadata, links, images, EPUBCheck |
| DOCX | Ordered paragraphs and tables, package parse, table/image preservation checks |
| PDF | Native text extraction, structure classification, readable final PDF validation when PDF is requested |
| TXT/Markdown | Paragraph/block identity, whitespace and order checks |
| SRT | Indexed subtitle units and exact timing-entry count |

When an audiobook profile creates both TXT and EPUB companions, the outputs
diverge intentionally at the final packaging step. The TXT is sanitized for
narration. The EPUB is cloned from the structured translated publication and
must retain its cover, images, captions, navigation, fonts, and exact image
positions; it is never regenerated from the flattened TXT.

When an exact format-native unit manifest is unavailable, the final validator
aligns source and output blocks by structural order and records that limitation
in the report. A block-count mismatch is blocking.

## Failure Map

| Risk | Control |
| --- | --- |
| Omitted or partial response | Provider truncation flag, block-safe placeholder recovery, stable expected count, empty/missing checkpoint unit, full coverage gate |
| Duplicate output | Deterministic IDs plus unrelated substantial-output duplicate detector |
| Wrong language | Per-unit script/language gate plus section/document ratios |
| Lost spaces | Source-proven DOM boundary restoration and final spacing patterns |
| Changed data | Exact entity multisets and locked profile entities |
| Semantic damage | Negation/length anomalies plus recorded source-aware audit evidence |
| Broken package | Native package parser, resource/link checks, EPUBCheck |
| Silent fallback | Failed unit state, sparse repair plan, blocked final status |
| Interrupted job | Durable checkpoint, preserved source/staging artifact, sparse resume; temporary HTTP 429 responses schedule a fresh worker after `Retry-After` instead of growing the call stack |
| Prompt injection | Stable system instruction that book content is untrusted data |

## Extension Points

To add a provider, implement the provider contract under `src/core/llm/providers`
and ensure usage metadata is recorded. To add a format, implement ordered block
extraction, reconstruction, final readability validation, structural inventory,
and tests for coverage, interruption, and package corruption. The quality core
must not import Flask or provider-specific clients.
