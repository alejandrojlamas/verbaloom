# Publication and cost patch: reviewed integration

Input: `verbaloom-publicacion-y-costo.patch`, SHA-256
`b1a6789425582d2430d50d5610a68758aff4febc328a5af795fc671deabca292`.
Reviewed against installed commit `f1ae8f7`, which already includes the
OpenAI review-policy integration. The submitted patch targets an older base;
its overlapping review-policy hunks must not replace the newer validation.

## Confirmed publication defect and adopted fix

Successful paragraph alignments changed only the assembled EPUB. A later
publication attempt rebuilt that EPUB from unchanged translated checkpoints,
lost the accepted boundaries and could call the model for the same repair
again. A reproduction using the actual checkpoint manager confirmed this.

The existing alignment pass now saves accepted boundary anchors in the job's
private upload directory. It does not rewrite the translation checkpoint or
reuse an arbitrary previous output. Reuse requires the exact source EPUB,
XHTML member, original region text and markup, cache schema and expected result
hash. The same ordered-anchor, text-conservation and structural checks run
again before accepting the cached alignment. The publication gate remains
authoritative.

Each accepted repair is saved atomically before attempting another region.
Subsequent failure or cancellation can therefore retain earlier accepted work.
Cached alignments do not consume the three-attempt model allowance, allowing
later publication attempts to reach unresolved regions rather than repaying
for the first ones. Failed, stale or corrupt entries are not accepted. A cache
write failure does not undo a successful repair. The existing job cleanup owns
the cache's lifetime.

## Why proportional rebalancing was excluded

Two synthetic counterexamples reproduce defects in the submitted module:

- The source first paragraph says that Mary refused an offer; the translated
  first paragraph is empty, and the second describes only John accepting an
  invitation. The original gate rejects it. Rebalancing puts the opening words
  about John into the empty paragraph and the real publication gate approves
  the file, although Mary's statement is still missing.
- The submitted emphasis fixture originally emphasizes `sala de control`.
  Rebalancing changes the emphasized content to `de control. ` and splits a
  sentence across paragraphs.

Preserving total characters and element counts does not establish source
coverage or preserve emphasis associations. Length-based redistribution can
silence the same heuristic that detected an omission. It is not installed or
enabled, even as an automatic fallback. Regression tests ensure genuine
omissions remain blocked after the cache integration.

## Safe, optional Flex support

The standard tier remains the default. `OPENAI_SERVICE_TIER=flex` explicitly
opts official OpenAI editorial models into Flex. Compatible/local endpoints
and unrelated models keep their existing request shape. The existing request
deadline remains in force; no short Flex timeout or extra timeout-to-standard
retry is introduced.

Only an HTTP 429 with the explicit `resource_unavailable` error code can trigger
standard-tier fallback. It consumes the normal attempt allowance, requires a
new per-attempt budget reservation and respects the job's maximum call count.
Ordinary rate limits, authentication failures, invalid requests and context
errors retain their existing handling. `OPENAI_FLEX_COOLDOWN_SECONDS` defaults
to 600 after an explicit capacity rejection.

The full-window conservative spend bound is unchanged. Each ledger event
records the tier actually requested for that attempt; observed response tiers
continue to determine settled prices. A verified, unbilled Flex capacity
rejection releases its reservation as known zero cost. This exception requires
the official provider, Flex request, matching HTTP status/reason, no usage and
no billing uncertainty. It does not make other tokenless errors free, and
uncertain requests retain their allowance.

Contradictory error responses are checked at the provider boundary, before
their usage evidence could be discarded: an error carrying usage, choices,
output or an unfamiliar envelope is not classified as a free capacity
rejection. It stops after one request with an actionable error and retains its
full uncertain reservation. Regression tests exercise the complete provider,
tracker and budget path, rather than only injecting events into the ledger.

The [official Flex guide](https://developers.openai.com/api/docs/guides/flex-processing)
documents Chat Completions support, variable latency and uncharged capacity
rejections. The installation's `.env` is preserved: this integration does not
activate Flex or silently trade throughput for its lower rates.

## Other decisions

- The cost-aware OpenAI review policy was already integrated in `f1ae8f7`;
  no duplicate changes or rollback were applied.
- Explicit prompt-cache blocks are excluded. The submitted Chat Completions
  shape has no demonstrated live compatibility, and its fallback retries
  unrelated HTTP 400 errors. Existing prompt-caching parameters remain intact.
- The patch correctly preserves the conservative spend bound; that invariant
  is retained in the accepted Flex option.
- The supplied tests were extended with negative cases: ordinary rate limits,
  exhausted call budgets, uncertain costs, changed/corrupt repair evidence,
  missing content and changed emphasis.

Validation uses synthetic books, mocked model responses and real local
checkpoint/publication/budget code. Model traffic is blocked during the test
suite. Results do not establish a new real-book speed, quality or cost
percentage, and no paid translation is needed for this integration.

Final observed validation: **3,403 passed, zero failures**, two skipped and ten
deselected in 195.93 seconds, with outbound network blocked. This adds 63
regression cases to the previously accepted baseline. The focused alignment
suite passed 26 cases; the focused provider/budget suite passed 114 cases.
An independent replay of a contradictory capacity response confirmed one
request, no fallback and a fully retained uncertain reservation. Publication
gate and source/checkpoint translation formats remain unchanged.
