# Glossary

Force consistent translations of recurring terms (characters, places, organizations, items) across an entire book. The glossary block is injected into the LLM system prompt only for chunks where a term actually appears, so it costs no tokens elsewhere.

> Available in both the Web UI (full CRUD, auto-extract via LLM) and the CLI (`--glossary file.json`).

---

## Table of contents

- [Why use it?](#why-use-it)
- [5-minute quick start](#5-minute-quick-start)
- [End-to-end workflows](#end-to-end-workflows)
- [Handling inflected languages (Russian, German, Polish, ...)](#handling-inflected-languages-russian-german-polish-)
- [Correcting a glossary mid-translation](#correcting-a-glossary-mid-translation)
- [File formats](#file-formats)
- [Categories](#categories)
- [Per-chunk filter](#per-chunk-filter)
- [Auto-extract reference](#auto-extract-reference)
- [Storage and migration](#storage-and-migration)
- [Tuning](#tuning)
- [Troubleshooting](#troubleshooting)
- [REST API](#rest-api)

---

## Why use it?

LLMs translate one chunk at a time and have no memory between chunks. Concretely, here is the same character name across three chapters of a long Chinese novel translated **without** a glossary:

| Chapter | What the LLM produced |
| ------- | --------------------- |
| Ch. 1   | `Li Fanqing`          |
| Ch. 5   | `Li Fan Qing`         |
| Ch. 12  | `Lee Fanqing`         |
| Ch. 20  | `Li Fan-qing`         |

With the glossary entry `李凡青 -> Li Fanqing  [character]`, every chunk that contains `李凡青` gets the rule injected into its prompt, so the output is `Li Fanqing` every single time.

What the LLM actually sees in the system prompt for a matching chunk:

```text
# GLOSSARY - REQUIRED TRANSLATIONS

MANDATORY: use these EXACT translations whenever the source term appears.
Do NOT paraphrase, transliterate differently, or invent alternatives.
Apply each rule consistently every time the term occurs.

  - 李凡青 -> Li Fanqing  [character]
  - 铁宗 -> Iron Sect  [organization]
  - 玄武城 -> Xuanwu City  [location]
```

---

## 5-minute quick start

Goal: build your first glossary, run a translation that uses it, all from the Web UI.

1. **Start the server** and open <http://localhost:5000>.
   ```bash
   python translation_api.py
   ```
2. **Click the "Glossaries" tab** in the header.
3. **Click "New"**, give it a name like `My novel` and pick the source/target languages (e.g. Chinese -> English).
4. **Drag your source file onto the "Auto-extract" button** (or click it and pick a file). Accepted: `.txt`, `.srt`, `.epub`, `.docx`. Up to 100 MB.
5. Leave **Total chars: 6000** and **Samples: 10** (good defaults). Click **Extract**.
6. After 30-90 seconds (depending on the model), you get a table of candidates. Each row has a checkbox, the source term as detected, an editable target translation, and a category.
7. **Review and edit**. Common edits: tweak the target translation, switch a category from `other` to `character`, uncheck rows that are obviously wrong.
8. **Click "Add selected"**. The modal closes, a toast confirms `Added N terms`. The terms appear in the editor table.
9. **Switch to the Translate tab**, drop your file in, then **pick your glossary in the dropdown** above the language selector.
10. **Click "Translate"**. The glossary block is now injected into every chunk that contains a matched term.

Total time: about 5 minutes for a first pass, then 15-30 minutes of manual review for a full novel (depends on how many distinct entities the book has).

---

## End-to-end workflows

### Workflow A: translating a long novel from scratch

You have `wuxia_novel.epub`, 600 pages, Chinese to English. You want consistent character/sect/location names.

```text
1. Web UI -> Glossaries tab -> New -> "Wuxia novel", Chinese -> English
2. Auto-extract: drop the EPUB, click Extract
   -> 10 excerpts of 600 chars each, distributed across the book
   -> ~40-60 candidates (typical for the first chapters of a wuxia)
3. Review the candidates table:
   - Fix mistransliterations: 李凡青 -> "Li Fanqing" (not "Li Fan-qing")
   - Set categories: characters, organizations, locations
   - Uncheck rows where source == target (the LLM gave up)
   - Uncheck noise (common nouns, action verbs)
4. Click "Add selected"
5. Run Auto-extract a SECOND time on the same file
   -> picks up entities the first pass missed (the model sees different excerpts)
   -> rows already in the glossary are tagged "(already in glossary)" and unchecked
6. Add the new ones, repeat until extracts return mostly already-known terms
7. Translate tab -> select the glossary -> drop the EPUB -> Translate
```

The auto-extract is non-destructive, so running it 2-3 times on the same book is the recommended flow. Each run samples different parts of the document and surfaces different recurring entities.

### Workflow B: building the glossary in the UI, running the translation in the CLI

Useful when the translation will take hours and you want to run it in a terminal (e.g. on a remote server, or overnight).

```text
1. Build the glossary in the Web UI (Workflow A, steps 1-6)
2. In the editor toolbar, click "Export" -> JSON
   -> downloads "wuxia_novel.json"
3. Move the file next to your CLI script
4. Run:
```

```bash
python translate.py \
  -i wuxia_novel.epub \
  -o wuxia_novel_en.epub \
  -sl Chinese -tl English \
  --provider openrouter \
  --openrouter_api_key sk-or-... \
  -m anthropic/claude-4.5-haiku \
  --glossary wuxia_novel.json
```

The CLI logs `Glossary loaded: 47 terms from wuxia_novel.json` on startup. From then on, every chunk uses the same per-chunk filter as the Web UI.

### Workflow C: importing an existing translator's glossary

Your translation team already keeps a shared CSV (Google Sheets export, etc.).

```csv
source,target,category
Volodymyr,Vladimir,character
Kyiv,Kiev,location
Sich,Sich,organization
```

```text
1. Web UI -> Glossaries tab -> New -> name + language pair
2. Click "Import" in the editor toolbar
3. Pick the .csv file
4. The whole file is loaded as terms (existing terms in the glossary are REPLACED)
```

If you want to merge instead of replace, add the rows manually via **Add row** (one click per row), or call the bulk-add API endpoint (see [REST API](#rest-api)).

### Workflow D: from a draft translation back into a glossary

You've translated the first 3 chapters by hand or with an earlier tool, and want to harvest the entity translations you settled on. Save them as a CSV and import.

```csv
source,target,category
李凡青,Li Fanqing,character
铁宗,Iron Sect,organization
玄武城,Xuanwu City,location
```

Then run auto-extract on the rest of the book; the rows that match what you already have appear flagged `(already in glossary)`, so you only review the new ones.

### Workflow E: translating subtitles with character names

SRT files often have repeated character names across thousands of lines.

```text
1. Glossaries tab -> New -> "Show name", English -> French
2. Auto-extract: drop the .srt of episode 1
   -> typical output: 5-15 character names + a few recurring locations
3. Add selected
4. Translate tab -> drop the .srt -> select glossary -> Translate
5. For episode 2, REUSE the same glossary (no need to rebuild)
```

You can also auto-extract from a few episodes back-to-back to enrich the glossary before translating any of them.

---

## Handling inflected languages (Russian, German, Polish, ...)

Languages with grammatical case, gender, or strong agglutination produce many surface forms of the same noun. A glossary with only the nominative form would miss most occurrences. The `|` separator declares **alternatives**: any of them in the source chunk triggers the same target.

### Russian: declensions

`Москва` (Moscow) appears as `Москве`, `Москвы`, `Москвой`, `Москву` depending on case. One entry covers all of them:

```json
{
  "source": "Москва|Москве|Москвы|Москвой|Москву",
  "target": "Moscou",
  "category": "location"
}
```

What the LLM sees when any of those forms appears in a chunk:

```text
  - Москва, Москве, Москвы, Москвой, Москву -> Moscou  [location]
```

The block presents the alternatives as a comma-separated list, so the model reads them as one entity with several inflected forms.

### German: declension + gender + case

Articles and adjective endings change but the noun stem stays. `Bürgermeister` (mayor) doesn't change much, but `Schwert` (sword) becomes `Schwertes`/`Schwerter`/`Schwertern` across the genitive/nominative-plural/dative-plural cases.

```json
[
  {
    "source": "Schwert|Schwertes|Schwerter|Schwertern",
    "target": "épée",
    "category": "item"
  },
  {
    "source": "Burg|Burgen",
    "target": "château",
    "category": "location"
  }
]
```

For person names that take genitive `-s`, declare both:

```json
{ "source": "Hans|Hansens", "target": "Jean", "category": "character" }
```

### Polish: heavy declension

Polish nouns inflect through 7 cases x 2 numbers = up to 14 forms. List the common ones (or all of them if you want full coverage):

```json
{
  "source": "Warszawa|Warszawy|Warszawie|Warszawę|Warszawą",
  "target": "Varsovie",
  "category": "location"
}
```

### Tips for inflected languages

| Tip                                                | Why                                                                                  |
| -------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Put the most distinctive form first                | The block displays alternatives in declared order, helps the LLM grasp the entity.   |
| Don't worry about overlap with prefixes            | The filter sorts longest-first, so `Moskovskaya` is checked before `Moskva`.         |
| For verbs / adjectives, use `|` per stem           | `красив-` (beautiful) family: list `красивый|красивая|красивое|красивые` etc.        |
| Latin terms ignore diacritics-insensitivity        | If you want `Cafe` to also match `Café`, list both in the source.                    |
| Up to ~50 alternatives per entry is fine           | Beyond that, regex compilation cost adds up; split into multiple entries.            |

### Auto-extract behavior with inflected languages

The NER prompt asks the model to return the **base form** of each entity. After review, you typically expand the source field manually with `|` alternatives. A useful flow:

```text
1. Auto-extract on a Russian text
   -> rows like {source: "Москва", target: "Moscou", category: "location"}
2. Edit the source cell BEFORE clicking "Add selected":
   change "Москва" to "Москва|Москве|Москвы|Москвой|Москву"
3. Add selected
```

Or add the base form first, then edit it later in the editor table by clicking the source cell.

---

## Correcting a glossary mid-translation

You launched a translation, watched the first few chunks come out, and noticed an entry produces a wrong target. Here is how to fix it without losing progress.

### Case 1: the target translation is wrong

Example: the glossary has `李凡青 -> Li Fan-qing` but you want `Li Fanqing` (no hyphen).

```text
1. Web UI -> Glossaries tab -> open the glossary
2. Find the row in the terms table (use the filter input above the table)
3. Click the target cell, edit "Li Fan-qing" to "Li Fanqing", press Enter
4. Done. The next chunk that contains 李凡青 will use the corrected target.
```

Note that already-translated chunks keep the old target; the fix only applies to chunks that have not been processed yet. If you need to fix already-translated chunks, do a search/replace on the output file after the run completes, or restart the translation from a checkpoint.

### Case 2: the source term is too greedy or not greedy enough

Example: you put `Fan -> Fan` to lock a name, but it now matches inside `Fantasy`.

The Latin word-boundary check should normally prevent this (`Fan` followed by `tasy` is not a word match). If you DO see it happening:

```text
1. Open the glossary editor
2. Edit the source: change "Fan" to "Fan Smith" (the full distinctive name)
3. Save
```

If on the contrary the source is too specific and misses inflected variants, switch to the `|` alternatives form:

```text
old: source = "Москва"
new: source = "Москва|Москве|Москвы|Москвой|Москву"
```

### Case 3: the category is wrong

Click the category dropdown in the row, pick the right one. The change is saved on selection. The injected block updates on the next chunk.

### Case 4: a term should be removed

Click the trash icon at the end of the row. Confirm. Done.

### Case 5: bulk corrections

Select multiple rows via the leftmost checkbox (or the header checkbox to select all visible). The toolbar shows bulk actions: **Delete selected**, **Set category**.

For more complex bulk edits, **Export** to JSON, edit in your text editor, then **Import** to replace.

---

## File formats

### JSON

Full glossary with metadata:

```json
{
  "name": "Wuxia novel",
  "source_lang": "Chinese",
  "target_lang": "English",
  "terms": [
    { "source": "李凡青",   "target": "Li Fanqing",  "category": "character" },
    { "source": "铁宗",     "target": "Iron Sect",   "category": "organization" },
    { "source": "玄武城",   "target": "Xuanwu City", "category": "location" }
  ]
}
```

Minimal (bare list of terms):

```json
[
  { "source": "Москва|Москве|Москвы|Москвой", "target": "Moscou", "category": "location" },
  { "source": "Иван",                         "target": "Ivan",   "category": "character" }
]
```

### CSV

Header row required. `source` and `target` are mandatory; `category` is optional.

```csv
source,target,category
李凡青,Li Fanqing,character
铁宗,Iron Sect,organization
玄武城,Xuanwu City,location
"Москва|Москве|Москвы|Москвой",Moscou,location
```

UTF-8 BOM and quoted values are supported. Use quotes around any source that contains commas (rare with `|` separator, but possible).

### Field aliases

The parser accepts both new and legacy field names:

| Field      | Aliases                       |
| ---------- | ----------------------------- |
| `source`   | `source_term`                 |
| `target`   | `translated_term`, `translation` |
| `category` | `type`                        |

So old exports from other tools should import without renaming.

---

## Categories

Optional metadata that appears as a bracketed hint after the arrow in the injected block:

```text
- Li Fanqing -> Li Fanqing  [character]
- Iron Sect -> Iron Sect   [organization]
```

The LLM is instructed to use the hint **only to disambiguate**, not to include it in the output. Recommended values:

| Category       | Use for                                  | Example                          |
| -------------- | ---------------------------------------- | -------------------------------- |
| `character`    | Person names                             | `Li Fanqing`, `Hans`, `Naruto`   |
| `location`     | Cities, regions, named places            | `Xuanwu City`, `Moscou`, `Konoha` |
| `organization` | Sects, factions, companies, governments  | `Iron Sect`, `Akatsuki`          |
| `item`         | Weapons, artifacts, consumables          | `Excalibur`, `Sharingan`         |
| `title`        | Honorifics, ranks, social titles         | `Sect Leader`, `Hokage`          |
| `other`        | Anything else                            |                                  |

Unknown categories are accepted but the UI may flag them. The auto-extract output is always one of the values above.

---

## Per-chunk filter

The filter scans each translation chunk and only injects the entries that actually appear. This keeps the prompt small and keeps the model focused.

| Behavior                        | Detail                                                                   |
| ------------------------------- | ------------------------------------------------------------------------ |
| Latin terms                     | Word-boundary match. `Fan` does not match `Fantasy`.                     |
| CJK terms                       | Substring match (no word-boundary concept in CJK scripts).               |
| Sort order in the block         | Longest source first, so `Li Fanqing` is checked before `Li Fan`.        |
| Cap                             | 50 entries per chunk by default. Most frequent kept on overflow.         |
| Case sensitivity                | Case-sensitive by default.                                               |
| Inflected forms                 | Sources with `|` are split into alternatives; ANY match triggers the entry. |

### Cap selection on overflow

If a chunk matches more than 50 entries (rare but possible in dense reference works), the filter:

1. Ranks matched entries by occurrence count in the chunk (most frequent first).
2. Tiebreaks by source length (longer wins).
3. Trims to 50.
4. Re-sorts the kept set by length descending for a stable injected block.

This means the most useful entries (frequent, long, specific) survive the cap.

---

## Auto-extract reference

The Web UI's **Auto-extract** button asks the configured LLM to scan a source file and propose recurring entities. Nothing is persisted until you review and click **Add selected**.

### Inputs

| Input         | Default | Cap   | Meaning                                                                  |
| ------------- | ------- | ----- | ------------------------------------------------------------------------ |
| File          |         | 100MB | `.txt`, `.srt`, `.epub`, `.docx`                                         |
| Total chars   | 6000    | 6000  | How much text the LLM sees, in characters.                               |
| Samples       | 10      | 50    | Number of evenly-spaced excerpts. `1` means "first N chars".             |

### How sampling works

For long files, sending the full text would blow past the model's context. Instead:

1. Read up to 5M chars from the file (cap on memory).
2. Divide into `Samples` evenly-spaced excerpts whose total length equals `Total chars`.
3. Snap each excerpt's edges to nearby whitespace to avoid cutting mid-word.
4. Join with a `[...]` separator so the model sees a discontinuity hint.

Each excerpt must be at least 500 chars; if the budget cannot afford that many samples, the count is reduced and a warning is shown above the candidates table.

### How extraction works

The provider/model configured for translation is reused (you don't need a separate API key). It is called with a fixed 8K context window and asked to return a JSON list of `{source, target, category}` candidates inside `<NER_JSON>...</NER_JSON>` tags.

The parser is permissive: if the tags are missing it falls back to markdown-fenced JSON, then balanced arrays, then balanced objects. Reasoning models' `<think>` blocks are stripped before parsing.

### Review flow

- Each candidate appears with an editable target field and a category dropdown.
- Rows whose source equals the proposed target (the LLM gave up translating, often a proper noun kept as-is) are dimmed and unchecked by default. Check them if you want the entry anyway (useful for locking proper nouns).
- Rows whose source already exists in the glossary are tagged `(already in glossary)` and unchecked by default.
- Edit any field before clicking **Add selected**. Common edits: fix transliteration, expand `source` with `|` alternatives, switch category.
- **Add selected** sends the whole batch in one transaction. A toast reports `Added N (M already existed, skipped)`.

### Re-running on the same file

Each run samples different excerpts (the LLM sees different parts of the document each time), so running auto-extract 2-3 times on a long book is the recommended flow. The second run typically surfaces 30-50% new entities. Stop when the table is mostly `(already in glossary)`.

---

## Storage and migration

The Web UI stores glossaries in a dedicated SQLite file:

```text
data/glossaries.db
```

Tables:

- `glossaries` (id, name, source_language, target_language, timestamps)
- `glossary_terms` (id, glossary_id, source_term, translated_term, category) with `UNIQUE(glossary_id, source_term)`

### Migration from older builds

Earlier versions stored glossaries inside `data/jobs.db` alongside translation checkpoints, which caused write-lock contention with running translations. On first launch with the new build, glossary tables are copied from `jobs.db` into `glossaries.db` automatically and idempotently. Look for this line in the server log on first start:

```text
Migrated 4 glossaries (87 terms) from data/jobs.db to data/glossaries.db
```

The legacy rows are not deleted from `jobs.db`, so rollback is safe. The migration runs only when `glossaries.db` is empty.

### Backup

```bash
cp data/glossaries.db data/glossaries.db.bak
```

Or export each glossary to JSON via the UI for human-readable backups.

The CLI does not touch the database at all; it loads JSON/CSV files directly from disk.

---

## Tuning

| Knob                         | Default | Where                                            |
| ---------------------------- | ------- | ------------------------------------------------ |
| Per-chunk cap                | 50      | `GlossaryConfig.max_entries`                     |
| Case sensitivity             | True    | `GlossaryConfig.case_sensitive`                  |
| Auto-extract sample budget   | 6000    | UI field, hard cap also 6000                     |
| Auto-extract sample count    | 10      | UI field (1..50)                                 |
| NER context window           | 8192    | Fixed in the route, independent of OLLAMA_NUM_CTX |

---

## Troubleshooting

| Symptom                                              | Cause / fix                                                                                            |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `database is locked` errors during glossary writes   | Confirm only one `translation_api.py` process is running. WAL mode lets readers and writers coexist within a process, but multiple processes still serialize at the file level. Run `Get-Process python` (Windows) or `pgrep -fa translation_api.py` (macOS/Linux) to check. |
| Auto-extract returns 0 candidates                    | Try increasing **Samples** to 15-20. Some literary excerpts genuinely have no recurring named entities (descriptive passages, dialogue between unnamed characters). |
| Term in glossary but not used in output              | Word boundary issue: `Fan` does not match `Fantasy`. For inflected languages, declare alternatives with `|`. |
| Added 50 terms but only some show up in translations | The per-chunk filter only injects entries that actually appear in that chunk. The cap of 50 is per chunk, not total. A 100-term glossary works fine if only 10-20 match any given chunk. |
| Auto-extract returns identical pairs (source = target) | Some sources are not translated (proper nouns kept as-is). The UI dims them; check or uncheck case-by-case based on whether you want to lock the proper noun. |
| Glossary dropdown empty in the Translate tab         | The dropdown only lists glossaries whose `source_lang`/`target_lang` match the selected language pair. Switch the language pair in the Translate tab or relax the glossary's language fields. |
| CLI says `Glossary file ... contained no usable entries` | JSON has wrong structure: bare list of `{source, target}` or full object with `terms` field. Verify with `python -m json.tool < glossary.json`. |
| Server log: `Migrated N glossaries (M terms)`        | First-run migration from the legacy `jobs.db`. Expected; idempotent on subsequent starts. |

---

## REST API

For programmatic access, the Web UI exposes REST endpoints under `/api/glossaries`:

| Method | Path                                                  | Action                                  |
| ------ | ----------------------------------------------------- | --------------------------------------- |
| GET    | `/api/glossaries`                                     | List glossaries with term counts        |
| POST   | `/api/glossaries`                                     | Create a glossary                       |
| GET    | `/api/glossaries/<gid>`                               | Read a glossary with all its terms      |
| PUT    | `/api/glossaries/<gid>`                               | Patch glossary fields                   |
| DELETE | `/api/glossaries/<gid>`                               | Delete (cascade to terms)               |
| POST   | `/api/glossaries/<gid>/duplicate`                     | Clone a glossary                        |
| POST   | `/api/glossaries/<gid>/terms`                         | Add a single term                       |
| PUT    | `/api/glossaries/<gid>/terms/<tid>`                   | Patch a term                            |
| DELETE | `/api/glossaries/<gid>/terms/<tid>`                   | Delete a term                           |
| POST   | `/api/glossaries/<gid>/terms/bulk`                    | Bulk action: `add`, `delete`, `set_category` |
| POST   | `/api/glossaries/<gid>/import`                        | Replace all terms from JSON or CSV      |
| GET    | `/api/glossaries/<gid>/export?format=json\|csv`       | Download                                |
| POST   | `/api/glossaries/<gid>/preview-block`                 | Render the injected block for a sample chunk |
| POST   | `/api/glossaries/<gid>/suggest-terms`                 | NER auto-extract                        |

### Bulk add example

```bash
curl -X POST http://localhost:5000/api/glossaries/1/terms/bulk \
  -H 'Content-Type: application/json' \
  -d '{
    "action": "add",
    "terms": [
      {"source": "Москва|Москве|Москвы", "target": "Moscou", "category": "location"},
      {"source": "Иван",                  "target": "Ivan",   "category": "character"}
    ]
  }'
```

Response:

```json
{ "added": 2, "conflicts": 0, "skipped_empty": 0, "total_input": 2 }
```

### Preview block example

Useful when authoring a glossary and wanting to see what the LLM will actually receive for a given chunk:

```bash
curl -X POST http://localhost:5000/api/glossaries/1/preview-block \
  -H 'Content-Type: application/json' \
  -d '{"text": "Сегодня в Москве идёт снег."}'
```

Response includes the rendered block, the matched count, the total terms in the glossary, and whether the cap was hit.
