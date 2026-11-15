# Reviewed OpenAI integration — 2026-11-14

The submitted `verbaloom-main-openai.zip` was compared with installed commit
`252a535`. It differs in 109 tracked files, adds two files and omits three.
Most publication, recovery and EPUB differences reproduce older revisions;
replacing the checkout with the archive would remove working fixes.

## Integrated changes

- On the official OpenAI endpoint, legacy DeepSeek reviewer names resolve to
  the corresponding OpenAI model tier. Other endpoints retain their model IDs.
- The existing cost-aware review policy also applies to official OpenAI
  editorial models. Deterministic checks and the required strict final audit
  remain enabled. Explicit pipelines and saved review references retain their
  meaning; a validated OpenAI pipeline uses its official endpoint when choosing
  reviewer defaults.
- Editorial-profile preparation follows OpenAI when selected, using Luna for
  preparation and Terra for review. Explicit pipeline roles and credentials
  take precedence. The UI displays the actual models and updates its new labels
  and running status immediately when the language changes.
- The official model catalog prefers Terra only when the configured model is
  unavailable. Configured models and custom-server defaults are preserved.
- The legacy CLI uses the configured OpenAI endpoint when `--provider openai`
  is selected without an endpoint override. Unlike the archive's suffix-based
  redirect, an explicit `--api_endpoint` is always preserved.
- Two missing publication labels were restored in five locales. These caused
  five failures in the unmodified baseline and are independent of model I/O.

No dependency, saved configuration, provider key, publication gate, checkpoint
format, deployment-default pipeline or release version was replaced.

## Changes deliberately excluded

| Incoming change | Reason |
| --- | --- |
| Older EPUB, recovery, progress and publication files | Would remove fixes from `bfc8c42`, `c6e9a91` and `252a535`, including paragraph preservation and recoverable publication. |
| Global provider/default reset | Existing installation settings and deployment-selected flow B remain authoritative. |
| Explicit prompt-cache blocks enabled by default | The submitted Chat Completions shape is not demonstrated by the supplied live evidence. Its fallback retries almost any non-context HTTP 400, not only an unsupported cache parameter. |
| Automatic Flex fallback | The implementation treats every Flex HTTP 429 as unavailable capacity, including actual rate limits; retrying a timed-out request may also duplicate billed work. |
| Smaller hard-budget reservations | UTF-8 bytes bound the visible text, but the proposed 64-token message and 512-token request allowances are not a verified bound on provider framing. The current conservative spend contract remains intact. |

The official [prompt-caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
also describes eligible implicit prefix lookups, so the archive's categorical
claim that implicit requests can never reuse a shared prefix is too strong.
The [Flex guide](https://developers.openai.com/api/docs/guides/flex-processing)
distinguishes resource-unavailable responses and slower processing. Neither
document substitutes for a live contract test of the submitted implementation.

## Validation scope

Regression tests cover official/custom endpoint boundaries, model aliases,
review policy, saved pipeline references, CLI argument forwarding, catalog
selection and localized model labels. The full baseline and integrated suites
run in separate worktrees without credentials and with outgoing network access
blocked. Existing browser scenarios and new profile scenarios run against a
fixture server with provider requests intercepted.

Observed results:

- Unmodified baseline: 3,283 passed, five existing locale-parity failures,
  12 skipped and 10 deselected (197.11 seconds).
- Final integration: **3,340 passed, zero failures**, two skipped and 10
  deselected (195.56 seconds). This includes 42 added regression cases; ten
  existing characterization cases that were skipped in the baseline environment
  also executed successfully in the integration environment.
- Browser: 15 existing scenarios and five new grouped profile checks passed,
  with no page errors. New checks include all seven UI languages.
- Configuration, EPUB/recovery source files and the conservative request-budget
  implementation were checked for preservation.

These checks validate integration behavior. They do not establish a new
translation-quality, latency or API-cost percentage, and no real book or paid
model request is used for this review.
