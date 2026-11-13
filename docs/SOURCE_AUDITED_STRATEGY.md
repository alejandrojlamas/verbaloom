# Source-audited translation ablation

Status: experimental, opt-in, not a measured quality or savings claim. The existing
compatible preset and production defaults remain unchanged. This is the first
C/D strategy hypothesis, not the completed benchmark or final optimized policy.

## Presets

| Variant | Preset | Models | Review policy |
| --- | --- | --- | --- |
| A | `compatible` | Existing role assignments | Existing behavior |
| B | `openai_equivalent` | Explicit Luna/Terra role mapping | Same behavior as A |
| C | `deepseek_source_audited` | Same assignments as A | Source-audited translation |
| D | `openai_source_audited` | Same assignments as B | Source-audited translation |

C requires a DeepSeek baseline rather than silently retaining OpenAI under a
DeepSeek name. All model roles, reasoning effort, output allowances and review
reference remain serialized through the existing pipeline configuration. Hybrid
roles require explicit custom configuration. No provider fallback is added.

`source_audited_v1` omits universal rewriting for ordinary translation but retains
semantic review of every unit. The preliminary mechanical calibration missed
negation, name and voice errors, so no risk-only or sampled preset is enabled.
Independent benchmark evaluation is still required; the pipeline auditor can
share the translator's model and must not be presented as an independent judge.

## Execution

- Native EPUB/DOCX XHTML keeps deterministic checks during translation, marks the
  deliberate no-rewrite decision, then runs the existing final source-aware unit
  audit. Hard intermediate rejections can still escalate. Final audit and
  publication gates remain mandatory even when the legacy `refine` flag is off.
- TXT/PDF/SRT generic units run their existing fidelity supervisor with full
  semantic coverage. Requested inline refinement is not executed by this policy.
- The plain-text paragraph route used by EPUB/DOCX audits after cleanup and local
  paragraph reconciliation, before accepting the result and neighbor context.
  Malformed judgments propagate the existing typed stop; rejected candidates are
  not returned as successful translations.
- The Flask handler uses the same refinement plan for chained-pass execution and
  progress. Usage metadata does not promise an omitted rewrite phase. Requested
  settings themselves are retained rather than overwritten.
- Explicit refinement-only jobs, transformations and strict/full/LLM profiles
  keep their required stages. Runtime rebinding preserves those exclusions.
- C/D select the existing `localized_v1` EPUB repair contract. A/B retain the
  compatible repair format. This does not add localized repair to every format.

## Reuse and resume

Generic completed units must match the current source text and unit index, and
carry an accepted semantic judgment matching the existing FidelityDecision hash
contract. A valid noncritical warning can be reused. Missing approval stops with
`review_strategy_changed`; changed source stops with `checkpoint_source_changed`.
Neither path silently discards a retained translation or starts a new paid pass.
Switching legacy checkpoints to this policy needs explicit migration/review work;
it is not an automatic approval upgrade.

Exact translation-memory hits reuse accepted judgments without a model request.
When cleanup/glossary application changes a cached candidate, or its semantic
approval is missing, the existing candidate is audited instead of retranslating
the source. A rejected cached candidate requires visible editorial review. The
existing nested cache metadata envelope is retained. Runtime candidate telemetry
is excluded from memory identity; it previously changed the lookup key during a
single audit and prevented later reuse.

This is not a claim that all-format dependency invalidation and crash-safe draft
recovery are complete. Generic/plain routes still need durable intermediate
candidate/audit-attempt recovery, profile/context dependency verification and
bounded localized repair integration. These remain acceptance work for the full
objective before a production recommendation.

## Configuration and verification

Use the existing job configuration, including its profile, glossary, source and
target locale. Do not reconstruct editorial settings from this example:

```python
from src.core.job_runtime_config import configure_editorial_guard_options
from src.core.pipeline_config import apply_pipeline, pipeline_preset

# job_config is an existing, private job configuration, not a new locale default.
apply_pipeline(job_config, pipeline_preset("deepseek_source_audited", job_config))
configure_editorial_guard_options(job_config)
```

The HTTP job boundary already accepts the explicit pipeline snapshot. A reactive
UI preset selector and a complete CLI/benchmark runner are still pending. Restore
the original saved pipeline/settings to return to a previous job's exact policy;
do not treat a newly generated preset as a substitute for a custom snapshot.

```sh
venv/bin/python -m pytest -q tests/unit/test_source_audited_strategy.py \
  tests/unit/test_pipeline_review_strategy.py tests/unit/test_plain_pipeline_realign.py \
  tests/unit/test_translation_memory.py tests/unit/test_inline_refinement_pipeline.py
venv/bin/python -m pytest -q --junitxml=/tmp/verbaloom-source-audited-20261112.xml
```

The new tests execute real generic translators, SQLite checkpoints, translation
memory, XHTML orchestration and plain paragraph processing with synthetic model
responses. They are not paid-provider, literary, full resource-rich publication,
browser or physical-device validation. The complete A/B/C/D executor, independent
heldout judgments, marginal-cost measurements and human review remain pending.
