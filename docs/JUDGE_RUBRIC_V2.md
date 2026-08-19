# TranslateBookWithLLM — Judge Rubric (v2)

**Version:** `v2`
**Identifier to record in submissions:** `<judge-id>-rubric-v2` (e.g.
`claude-opus-4-7-rubric-v2-poe`)

This document defines how a translation is scored in the benchmark v2 series.
It supersedes `JUDGE_RUBRIC.md` (v1) for all submissions made after the v2
re-judging pass. v1 stays frozen for historical reference.

If you change the rubric (new dimensions, new penalty values), bump the
version to `v3` and start a new wiki series. Never silently change the
meaning of `v2`.

---

## What changed vs v1

1. **§5 (comparative dispersion) removed.** v1 forced ≥0.3 between adjacent
   ranks per `(text_id, target_lang)` triple. That was a band-aid for a
   weakly-calibrated judge. v2 trusts the absolute scale and the penalty
   table to produce natural dispersion.
2. **Output format §11 added** — feedback is bounded (≤200 ASCII chars, no
   verbatim source-language quotes). This eliminates output truncation
   failures on CJK/RTL feedback and keeps cost predictable.
3. **Anchor scale tightened** — the same anchors, but the system prompt now
   instructs the judge explicitly to be willing to use 3–5 and to not be
   charitable to LLM output.

---

## 1. Dimensions

Each translation is scored on four dimensions, each in `[1.0, 10.0]` with
decimal precision:

| Dimension | What it measures |
|---|---|
| **accuracy** | Preservation of meaning. No additions, no omissions, no contresens, no hallucinations. |
| **fluency** | Naturalness in the target language. Grammar, syntax, idiomaticity, no untranslated source words, no script mismatches. |
| **style** | Register, tone, period vocabulary, literary voice, rhetorical devices (irony, parallelism, etc.). |
| **overall** | Holistic judgement. **Not the average.** Weighted by what matters most for this passage, with hard ceilings (see §4). |

---

## 2. Anchored scale

Scores are anchored to **professional human translation as the reference**.
This is a hard, non-negotiable framing — every judge calibrates against it.

| Score | Anchor description |
|---|---|
| **10** | Equivalent to a published reference translation by a recognized literary translator (e.g. Lydia Davis, Penguin Classics, Pléiade). **Effectively unreachable** by an LLM in current state of the art. |
| **9** | Excellent professional translation, non-literary tier. Could be published with light editing. Quibbles are stylistic, not factual. |
| **8** | Very good. The reader gets the full meaning and tone, but loses 1–2 nuances or makes a faux-sens mineur. |
| **7** | Good in the main, but 2–3 notable errors of register, idiom, or specialized terminology. A reader misses something they shouldn't. |
| **6** | Comprehensible but clearly clumsy. Multiple problematic word choices. |
| **5** | Usable only with editorial intervention. Real meaning errors or fluency breaks. |
| **4** | Significantly impaired. Frequent contresens or sentence-level breakdowns. |
| **3** | Passages incomprehensible or radically distorted. |
| **2** | Mostly wrong, hallucinations dominant. |
| **1** | Non-translation: wrong target language, source copied verbatim, refusal, or empty. |

**Hard cap at 9.0** when no reference human translation is consulted. Reserve
9.5–10 only when the judge has a published reference in front of them and the
LLM output matches or surpasses it.

---

## 3. Penalty table — start from 10, deduct

Apply these per detected issue. Multiple issues compound (additively) within a
dimension. Cap the result at the anchor scale (no negative scores).

### Accuracy penalties

| Issue | Penalty |
|---|---|
| Contresens / radical meaning reversal on a key sentence | **−2.0** |
| Hallucination of facts not in the source (place name, date, number) | **−2.0** |
| Source word/phrase left untranslated in target (e.g. English "Providence" inside Chinese) | **−1.5** |
| Significant omission (a clause or named entity dropped) | **−1.0** |
| Specialized term wrong (botanical, marine, scientific, legal, etc.) | **−0.5** to **−1.0** |
| Minor semantic drift on a non-load-bearing word | **−0.3** |

### Fluency penalties

| Issue | Penalty |
|---|---|
| Grammar error breaking the sentence | **−1.5** |
| Mixed scripts in target (traditional chars in zh-Hans, Latin word inside CJK, etc.) | **−1.0** |
| Awkward construction reading as machine-translated | **−0.5** to **−1.0** |
| Wrong target-language punctuation (e.g. English quotes in French dialogue) | **−0.3** |
| Pronoun inconsistency or unnecessary subject restatement | **−0.3** to **−0.5** |
| Calque/anglicism that has a native equivalent | **−0.5** |

### Style penalties

| Issue | Penalty |
|---|---|
| Lost a signature rhetorical device (irony, parallelism, archaic register) | **−1.0** to **−1.5** |
| Period-inappropriate vocabulary (modern slang in 19th-c. text) | **−1.0** |
| Register flattened (formal → neutral, gnomic → narrative) | **−0.5** to **−1.0** |
| Weakened idiom into bland equivalent | **−0.3** |

---

## 4. Overall — hard ceilings

`overall` is the judge's holistic call, but constrained:

- If `accuracy < 6.0` → `overall ≤ 6.0`. A translation that distorts meaning is not "good", whatever its prose.
- If `fluency < 5.0` → `overall ≤ 6.0`. An unreadable translation is not usable.
- If any dimension is `≤ 3.0` → `overall ≤ 4.0`.
- `overall` should not exceed the **minimum** of `accuracy, fluency, style` by more than 0.5. (Prevents a high overall masking a single damning weakness.)
- Without a published reference comparison, **`overall` cap is 9.0**.

---

## 5. Comparative dispersion — REMOVED in v2

v1 §5 forced a minimum 0.3 between adjacent ranks per triple. v2 removes
this rule. Judges score each translation **independently in absolute terms**.
Natural dispersion emerges from rigorous application of the penalty table.

---

## 6. Worked example

Same as v1 §6. See `JUDGE_RUBRIC.md` for the detailed Wilde passage walk-through.

---

## 7. Feedback field — bounded format

Each evaluation MUST include a `feedback` string subject to these constraints:

- **≤200 ASCII chars.** Anything beyond gets truncated by output limits and
  is wasted.
- **No verbatim source-language quotes.** Describe issues in English with
  English glosses or generic position labels. CJK/RTL/Greek/Cyrillic quotes
  consume 2-3 tokens per character and have caused JSON truncation failures.
- **Penalty totals only.** Don't enumerate every token. State the major
  deductions with their values, e.g. "Two contresens (−2.0 ×2); register
  flattened (−1.0); minor calques (−0.3)."

Examples:

- ✅ "Two contresens (−2.0 ×2): first Korean phrase mistranslated as 'rumors', streetcar phrase as 'telephone line'. Register flattened (−1.0)."
- ✅ "Faithful Walden rendering; 'marrow' as 精髓 adequate; mild contemplative flatness (−0.5 style)."
- ❌ "Multiple severe contresens: '동소문 안에서 인력거꾼 노릇을 하는' rendered as..." (CJK quote — wasted tokens, truncation risk)
- ❌ "Good translation overall." (no audit trail)

---

## 8. Operational notes

- **Don't average.** `overall` is a holistic call within the ceilings, not
  `(accuracy + fluency + style) / 3`.
- **Don't be charitable to LLM output.** If a human professional wouldn't
  ship it, it isn't a 9.
- **Be willing to use 3, 4, 5.** Most LLM outputs in cross-language pairs
  deserve 6–8. Reserve 9 for excellence.
- **Re-test yourself.** Every 50 evaluations, re-judge 3 of your earliest
  scores. If they drift more than ±0.5 from your original, recalibrate.

---

## 9. Reproducibility

Empirical: Opus 4.7 via Poe at `temperature=0.1, max_tokens=400` produces
score variance of stdev ≤ 0.3 on `overall` across runs of the same
translation. See `plan/calibration_test_multi.py` for the measurement.

If you change the judge model or any prompt parameter, document the change
and bump the judge_id (`<model>-rubric-v2-<provider>`).

---

## 10. Versioning

- This is **v2**. Recorded as `<judge-id>-rubric-v2-<provider>` (e.g.
  `claude-opus-4-7-rubric-v2-poe`).
- A `v3` rubric must be a separate document. Wiki tables surface the rubric
  version next to scores.
- Submissions made under different rubric versions should not be mixed in
  the same wiki ranking series.
