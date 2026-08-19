"""Characterization tests for the translation/refine progress system.

These tests are a *safety net*, not a specification. They pin the current
observable behavior of the progress machinery (the sequence of stats dicts
emitted per format and per mode, plus the produced output bytes) so that the
progress-system refactoring can proceed step by step without silent
regressions.

The LLM is replaced by a deterministic echo provider (see ``fake_llm``), so
runs are reproducible and require no network or API key. The "translation"
produced is simply the source text echoed back, which preserves placeholders
and subtitle indices exactly — what we assert on is the *progress signal*, not
translation quality.

Golden snapshots live under ``tests/characterization/golden/``. To regenerate
them after an intentional behavior change, run pytest with the environment
variable ``UPDATE_CHARACTERIZATION=1`` set, review the diff, and commit it.
"""
