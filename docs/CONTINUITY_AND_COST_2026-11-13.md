# Continuity and cost changes (2026-11-13)

Two goals: a book keeps translating instead of stopping, and it costs less
without lowering quality. Every limit stays finite and every quality gate stays
in place; nothing below publishes text that a gate rejected.

## Why a book stopped

| Cause | Before | Now |
| --- | --- | --- |
| Provider outage (timeouts, connection errors, HTTP 5xx, open circuit breaker, 300 s operation deadline) | `generate()` returned `None`; the unit burned its attempts in seconds, was marked FAILED and stopped the book. An outage during the audit raised `audit_invalid_response` and stopped the job. | `ProviderUnavailableError` (retryable). The job waits 60 s, 180 s, 600 s, then 30 min per wait and resumes from the checkpoint. No unit is marked failed. |
| Three provider waits on one unit's audit | The unit's audit allowance was exhausted and the unit stayed FAILED permanently. | A provider wait is not a judgment; the attempt is returned. |
| One XHTML file with an unresolved unit | The EPUB loop stopped at that file. Two recovery cycles (2 s, 10 s) retried the same unit, then the job stayed `partial` until a manual resume. | The file is deferred. Later files are translated and checkpointed. Recovery and Resume retry only pending files. |
| Provider refusal or unusable audit answer for one passage | Job stop, manual action required. | The unit is blocked for review and recorded; the rest of the book continues. The passage is never re-sent, split or rephrased automatically. |
| Resume pressed during the DeepSeek high-price window | HTTP 423; the user had to come back later. | The resume is queued on a durable schedule and starts by itself when the window ends. Starting a new job during the window is still refused. |
| Provider-wait ceilings | 5 waits per checkpoint position (about 75 minutes) and 4 hours cumulative per job, after which every wait needed a manual resume. | 12 waits per position (about 5 hours) and 24 hours cumulative. Ceilings frozen at the old defaults in existing jobs are upgraded once. |

What still stops a job, by design: no credits, invalid credentials or model,
a manual pause, the optional pipeline budget, an exhausted recovery budget,
and `EPUB_MAX_CONSECUTIVE_FAILED_FILES` failed files in a row.

A unit that no candidate resolves still blocks publication of the book. The
difference is that every other chapter is finished, checkpointed and paid for
only once when that happens.

## Where money was wasted

The pipeline's happy path (translate, refine, audit) was left unchanged. The
waste was in recovery:

1. **A placeholder error re-requested the whole unit.** Up to three full
   attempts, then a block-by-block re-translation. In the last recorded job
   this "format repair" work was about a third of the spend.
   `placeholder_salvage` now keeps every block whose placeholders are intact,
   restores a dropped block boundary only where the source proves its
   position, and re-requests only the damaged blocks.
2. **Up to four translated units were lost on every interruption.** The EPUB
   partial state was saved every fifth unit; a provider wait or the peak-price
   boundary interrupts the loop with an exception. It is now saved after each
   unit.
3. **A paid refinement or audit repair was discarded for a dropped
   placeholder.** When the draft proves every placeholder position, the
   rewrite is repaired and kept. The check also moved before the paid quality
   guards.
4. **A failed file made resume translate later files again** (only relevant
   with deferral): completed files are restored from their checkpoint rows.

Quality safeguards of the salvage path:

- Block boundaries are never guessed. Merged paragraphs are re-translated as
  one region.
- A placeholder the model kept is never moved.
- A lost line break, image or other non-formatting tag is never interpolated;
  the whole-unit retry runs instead.
- Formatting spans (`<i>`, `<b>`, `<span>`, `<a>` ...) are interpolated between
  the surviving tags only after the regional request also failed. The
  structural recovery already placed every inline tag proportionally in that
  situation; its offsets were also biased toward the end of the block, which is
  fixed.
- The result passes the same strict validator and then every existing
  fidelity, language, review and audit gate.

## Settings

All are environment variables (see `.env.example`). They are not prompt
options, so changing them does not invalidate checkpoints.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPUB_DEFER_FAILED_FILES` | `true` | Continue with later files after a failed file. `false` restores stop-at-first-failure. |
| `EPUB_MAX_CONSECUTIVE_FAILED_FILES` | `3` | Failed files in a row that still stop the book. |
| `EPUB_CHECKPOINT_FREQUENCY` | `1` | Translated units between partial-state saves. |
| `EPUB_PLACEHOLDER_SALVAGE` | `first_failure` | `first_failure`, `after_retries` or `off`. |
| `EPUB_PLACEHOLDER_SALVAGE_MAX_DAMAGED_SHARE` | `0.6` | Above this share of damaged blocks the whole unit is retried. |

## Cost levers that remain a product decision

These change what the pipeline does, so they were not touched:

- **Peak hours.** `DEEPSEEK_OFF_PEAK_ONLY=true` halves the price by not sending
  tokens during the high-price window. Turning it off removes those pauses at
  twice the price for the hours involved.
- **Refinement pass.** About 40% of the per-unit cost on the happy path. The
  `deepseek_source_audited` preset skips universal refinement; its effect on
  quality has not been measured on a full book.
- **Model per role.** The Flash model is several times cheaper than Pro per
  token. Moving a role to Flash is a quality trade-off.

## Scope and verification

- The file-level deferral, per-unit checkpoint and placeholder salvage apply to
  the native EPUB path. TXT, PDF, DOCX and plain-text EPUB already continued
  past failed units; they gain the outage handling, the wait ceilings and the
  queued resume.
- Outage detection is exact for the DeepSeek and OpenAI-compatible providers.
  Other providers are covered through the circuit breaker and the operation
  deadline.
- Verified with the unit suite and a native end-to-end test against synthetic
  HTTP (`tests/unit/epub/test_deferred_failure_e2e.py`). Not verified against
  the live provider: real savings and real outage behavior need one book run.
