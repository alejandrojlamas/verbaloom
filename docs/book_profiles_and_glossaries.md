# Book Profiles and Scoped Glossaries

This project separates the generic translation/modernization engine from
book-specific editorial decisions.

## Boundaries

The generic engine may handle chunking, LLM calls, reconstruction, checkpointing,
format conversion, source-aware auditing, number/name checks, spacing, encoding,
dialogue dashes, and other mechanical validations.

The generic engine must not contain book-specific modernization equivalences.
Stable literary decisions belong in a profile under `profiles/<profile_id>/`.

## Profile Structure

```text
profiles/<profile_id>/
  editorial_policy.md
  profile.yml
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

`profiles/common/` is reserved for truly general mechanical entries. It should
not contain literary modernization choices. A profile loads `common` only when
its `profile.yml` explicitly lists it.

## Glossary Entry Lifecycle

Glossary entries are scoped to one profile unless they explicitly live in
`profiles/common/`.

Allowed states:

- `pending`: proposed by discovery or human review; not applied as an approved
  rule.
- `approved`: rendered into the prompt when the source chunk contains the entry.
- `rejected`: kept for audit history but not applied.
- `superseded`: replaced by a better entry.

Use `glossary/pending_suggestions.yml` for agent suggestions. Move entries into
`terms.yml`, `treatments.yml`, `phrases.yml`, or `character_voices.yml` only
after approval.

## Commands

Create a new empty profile:

```bash
python scripts/profile_init.py --profile my_book_mx
```

Discover recurring candidates from a source file:

```bash
python scripts/glossary_discover.py --profile my_book_mx --input source.txt
```

Inspect glossary state:

```bash
python scripts/glossary_report.py --profile my_book_mx
```

Approve high-confidence pending suggestions:

```bash
python scripts/glossary_approve.py --profile my_book_mx --min-confidence 0.95
```

## Preflight Profile Preparation

The web UI can create a book profile before the main run:

1. Select files in **Transformar texto**.
2. Choose the main purpose: faithful translation, audiobook, modernization,
   explanation, or literary polish.
3. Click **Analizar con DeepSeek Flash**.
4. The server extracts readable text, scans the full document locally, and runs
   distributed discovery chunks through `deepseek-v4-flash`.
5. A generated profile is written under `profiles/auto_<book>/`.
6. Mechanically safe entries go to `glossary/terms.yml`.
7. Context-sensitive discoveries go to `glossary/pending_suggestions.yml`.
8. The new profile is selected in the **Perfil editorial** dropdown before the
   main process starts.

Profile preparation uses goal-specific business rules. A faithful translation
profile prioritizes canonical names and terms that must be translated
consistently; an audiobook profile prioritizes listening hygiene, note/caption
patterns, and reference cleanup; a modernization profile prioritizes archaisms,
treatments, syntax, and voice. The term-review and pending-suggestion budgets
scale with the book and the selected goal, so different books should not all
produce the same number of approved or review-pending entries.

This flow uses `/api/book-profiles/prepare-jobs`: the browser starts a
background preparation job, polls its status, shows stage/chunk progress, and
then auto-selects the saved profile when the job reaches `completed`.

The preflight model should be cheap and fast. DeepSeek V4 defaults to thinking
mode, so the app explicitly disables thinking for this discovery path to avoid
spending reasoning tokens. The main process should still use `deepseek-v4-pro`
when quality, audit, and repair matter.

## Language-Scoped Lexical Policies

The deterministic extractor and term reviewer load source-language data from
`src/core/glossary/lexical_policies.json`. The engine keeps only algorithms and
structural categories in Python. An English stopword, technical-term hint, or
demonym therefore cannot affect a book whose source language is explicitly
German, Czech, Greek, or another language.

For an unknown or auto-detected language, the local pass uses only neutral
structure checks. DeepSeek Pro remains the contextual reviewer and may leave a
term pending rather than forcing an unsafe preservation or translation.

## Profile-Scoped Audit Dimensions

Every profile is audited on generic fidelity and isolation dimensions. Its
business goal adds only the relevant translation, modernization, academic,
audiobook, explanation, or literary-polish dimensions. Work-specific dimensions
must be listed in that profile's `audit_score_fields`; they are not part of the
generic engine.

## No Hardcoding Rule

Do not add profile-specific source-target equivalences to `src/`.

If a recurring equivalence is needed, add it to the active profile glossary. If
it seems useful for several books, mark it as a candidate for `common`, but do
not activate it globally without review.

The tests include a guard that scans `src/` for known profile-specific
equivalences that must remain inside profile files.
