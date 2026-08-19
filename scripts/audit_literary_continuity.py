#!/usr/bin/env python3
"""Dry-run audit for literary continuity prompts.

This script builds translation prompts for a long literary source without
calling an LLM. It advances the continuity memory with a deterministic
source-as-translation stand-in, then writes per-prompt metrics and a summary
report. The goal is to understand prompt overhead, memory drift, entity
selection, and section continuity before spending API tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import re
import statistics
from typing import Iterable

from src.core.chunking.token_chunker import TokenChunker
from src.core.editorial_quality import infer_section_title
from src.core.literary_continuity import (
    build_literary_continuity_block,
    detect_text_profile,
    extract_literary_names,
    observe_literary_continuity,
)
from src.core.text_processor import split_text_into_chunks
from src.prompts.prompts import generate_translation_prompt


START_RE = re.compile(r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG EBOOK.*\*\*\*", re.I)
END_RE = re.compile(r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG EBOOK.*\*\*\*", re.I)
TRANSCRIBER_RE = re.compile(
    r"\n\s*(?:TRANSCRIBER['’]S NOTES?|Transcriber['’]s notes?).*\Z",
    re.I | re.S,
)
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|CAP[IÍ]TULO|Cap[ií]tulo)\s+"
    r"(?:[0-9]+|[IVXLCDM]+|[ivxlcdm]+)\.?\s*[-:.\w ,;'’]{0,100}$",
    re.M,
)


def strip_gutenberg(text: str) -> str:
    start = None
    end = None
    for match in START_RE.finditer(text):
        start = match.end()
        break
    for match in END_RE.finditer(text):
        end = match.start()
        break
    if start is not None:
        text = text[start:]
    if end is not None and (start is None or end > 0):
        text = text[:end if start is None else max(0, end - (start or 0))]
    text = TRANSCRIBER_RE.sub("", text)
    return text.strip()


class ChapterTracker:
    """Carry chapter metadata across chunks instead of inferring from each chunk alone."""

    def __init__(self):
        self.current = "Documento"

    def update(self, main: str, context_before: str = "") -> str:
        combined = "\n".join(part for part in [context_before, main] if part)
        matches = [m.group(0).strip() for m in CHAPTER_HEADING_RE.finditer(combined)]
        if matches:
            self.current = matches[-1]
        return self.current


def sha(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[idx]


def top_records(state: dict, limit: int = 12) -> list[dict]:
    memory = state.get("literary_continuity_state")
    if not memory:
        return []
    records = sorted(
        memory.characters.values(),
        key=lambda r: (-r.occurrences, r.name),
    )[:limit]
    return [
        {
            "name": r.name,
            "category": getattr(r, "category", ""),
            "aliases": sorted(getattr(r, "aliases", []) or [])[:6],
            "target": getattr(r, "target", ""),
            "occurrences": r.occurrences,
            "first_chunk": r.first_chunk,
            "last_chunk": r.last_chunk,
        }
        for r in records
    ]


def count_lines_with_prefix(block: str, prefixes: Iterable[str]) -> int:
    prefixes = tuple(prefixes)
    return sum(1 for line in block.splitlines() if line.strip().startswith(prefixes))


def audit(args: argparse.Namespace) -> dict:
    source_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_text = source_path.read_text(encoding="utf-8", errors="replace")
    clean_text = strip_gutenberg(raw_text)
    clean_path = out_dir / f"{source_path.stem}.clean.txt"
    clean_path.write_text(clean_text, encoding="utf-8")

    chunks = split_text_into_chunks(
        clean_text,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
    )
    token_counter = TokenChunker(max_tokens=args.max_tokens_per_chunk)
    profile = detect_text_profile(clean_text[: args.profile_chars])

    prompt_options = {
        "literary_continuity": True,
        "text_type": args.text_type,
        "continuity_max_prompt_chars": args.continuity_max_prompt_chars,
        "continuity_max_prompt_tokens": args.continuity_max_prompt_tokens,
    }
    runtime_state: dict = {}
    previous_translation_context = ""
    section_tracker = ChapterTracker()
    rows: list[dict] = []

    system_hashes = Counter()
    memory_hashes = Counter()
    section_counts = Counter()

    label = f".{args.label}" if args.label else ""
    jsonl_path = out_dir / f"{source_path.stem}{label}.prompt-audit.jsonl"
    samples_path = out_dir / f"{source_path.stem}{label}.prompt-samples.md"
    sample_indexes = {
        0,
        max(0, len(chunks) // 4),
        max(0, len(chunks) // 2),
        max(0, (len(chunks) * 3) // 4),
        max(0, len(chunks) - 1),
    }
    sample_blocks: list[str] = []

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for idx, chunk in enumerate(chunks):
            main = chunk.get("main_content", "")
            context_before = chunk.get("context_before", "")
            context_after = chunk.get("context_after", "")
            inferred_section = infer_section_title(
                main,
                context_before=context_before,
                fallback=section_tracker.current,
            )
            current_section = section_tracker.update(main, context_before) or inferred_section

            continuity_block = build_literary_continuity_block(
                prompt_options=prompt_options,
                runtime_state=runtime_state,
                current_text=main,
                source_language=args.source_language,
                target_language=args.target_language,
                section=current_section,
            )
            prompt = generate_translation_prompt(
                main_content=main,
                context_before=context_before,
                context_after=context_after,
                previous_translation_context=previous_translation_context,
                source_language=args.source_language,
                target_language=args.target_language,
                has_placeholders=False,
                prompt_options=prompt_options,
                continuity_block=continuity_block,
            )

            main_tokens = token_counter.count_tokens(main)
            continuity_tokens = token_counter.count_tokens(continuity_block)
            system_tokens = token_counter.count_tokens(prompt.system)
            user_tokens = token_counter.count_tokens(prompt.user)
            prompt_tokens = system_tokens + user_tokens
            names = extract_literary_names(main)
            memory = runtime_state.get("literary_continuity_state")
            render_stats = getattr(memory, "last_render_stats", {}) if memory else {}
            system_hash = sha(prompt.system)
            memory_hash = sha(continuity_block)
            system_hashes[system_hash] += 1
            memory_hashes[memory_hash] += 1
            section_counts[current_section] += 1

            row = {
                "chunk_index": idx + 1,
                "section": current_section,
                "main_chars": len(main),
                "main_tokens": main_tokens,
                "context_before_chars": len(context_before),
                "context_after_chars": len(context_after),
                "previous_context_chars": len(previous_translation_context),
                "system_tokens": system_tokens,
                "user_tokens": user_tokens,
                "prompt_tokens_est": prompt_tokens,
                "continuity_chars": len(continuity_block),
                "continuity_tokens": continuity_tokens,
                "continuity_hard_truncated": continuity_block.rstrip().endswith("..."),
                "continuity_omitted_lines": int(render_stats.get("omitted_lines", 0)),
                "continuity_entity_lines": int(render_stats.get("entity_lines", 0))
                or count_lines_with_prefix(continuity_block, ("- ",)),
                "continuity_event_lines": int(render_stats.get("event_lines", 0)),
                "source_names": names[:20],
                "source_name_count": len(names),
                "tracked_character_count_before": len(
                    getattr(runtime_state.get("literary_continuity_state"), "characters", {})
                ),
                "top_tracked_before": top_records(runtime_state, limit=10),
                "system_hash": system_hash,
                "user_hash": sha(prompt.user),
                "memory_hash": memory_hash,
                "memory_block": continuity_block,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

            if idx in sample_indexes:
                sample_blocks.append(
                    "\n".join([
                        f"## Prompt sample chunk {idx + 1}",
                        "",
                        f"- Section: {current_section}",
                        f"- Estimated prompt tokens: {prompt_tokens}",
                        f"- Continuity tokens: {continuity_tokens}",
                        "",
                        "### Continuity block",
                        "",
                        "```text",
                        continuity_block.strip(),
                        "```",
                        "",
                        "### User prompt head",
                        "",
                        "```text",
                        prompt.user[:2500].strip(),
                        "```",
                        "",
                    ])
                )

            fake_translation = main
            observe_literary_continuity(
                runtime_state=runtime_state,
                source_text=main,
                translated_text=fake_translation,
                section=current_section,
                phase="dry_run",
            )
            words = fake_translation.split()
            previous_translation_context = " ".join(words[-25:]) if len(words) > 25 else fake_translation

    samples_path.write_text("\n".join(sample_blocks), encoding="utf-8")

    state = runtime_state.get("literary_continuity_state")
    prompt_tokens = [r["prompt_tokens_est"] for r in rows]
    continuity_tokens = [r["continuity_tokens"] for r in rows]
    main_tokens = [r["main_tokens"] for r in rows]
    continuity_ratios = [
        (r["continuity_tokens"] / max(1, r["prompt_tokens_est"]))
        for r in rows
    ]
    truncations = sum(1 for r in rows if r["continuity_hard_truncated"])
    omitted_lines = [r["continuity_omitted_lines"] for r in rows]
    tracked_counts = [r["tracked_character_count_before"] for r in rows]
    names_per_chunk = [r["source_name_count"] for r in rows]
    repeated_memory_blocks = sum(count for count in memory_hashes.values() if count > 1)

    top_entities = top_records(runtime_state, limit=40)
    warning_count = len(getattr(state, "warnings", [])) if state else 0
    report = {
        "source": str(source_path),
        "clean_text": str(clean_path),
        "jsonl": str(jsonl_path),
        "samples": str(samples_path),
        "book_chars": len(clean_text),
        "chunks": len(chunks),
        "profile": {
            "kind": profile.kind,
            "confidence": profile.confidence,
            "signals": dict(profile.signals),
        },
        "main_tokens": {
            "min": min(main_tokens) if main_tokens else 0,
            "median": int(statistics.median(main_tokens)) if main_tokens else 0,
            "p95": percentile(main_tokens, 95),
            "max": max(main_tokens) if main_tokens else 0,
        },
        "prompt_tokens_est": {
            "median": int(statistics.median(prompt_tokens)) if prompt_tokens else 0,
            "p95": percentile(prompt_tokens, 95),
            "max": max(prompt_tokens) if prompt_tokens else 0,
        },
        "continuity_tokens": {
            "median": int(statistics.median(continuity_tokens)) if continuity_tokens else 0,
            "p95": percentile(continuity_tokens, 95),
            "max": max(continuity_tokens) if continuity_tokens else 0,
        },
        "continuity_prompt_ratio": {
            "median": round(statistics.median(continuity_ratios), 4) if continuity_ratios else 0,
            "p95": round(percentile([int(x * 10000) for x in continuity_ratios], 95) / 10000, 4)
            if continuity_ratios else 0,
        },
        "continuity_truncated_chunks": truncations,
        "continuity_omitted_lines": {
            "total": sum(omitted_lines),
            "p95": percentile(omitted_lines, 95),
            "max": max(omitted_lines) if omitted_lines else 0,
        },
        "tracked_character_count_final": len(getattr(state, "characters", {})) if state else 0,
        "tracked_character_count_p95": percentile(tracked_counts, 95),
        "names_per_chunk_p95": percentile(names_per_chunk, 95),
        "unique_system_prompts": len(system_hashes),
        "unique_memory_blocks": len(memory_hashes),
        "repeated_memory_blocks": repeated_memory_blocks,
        "sections_detected": len(section_counts),
        "top_sections": section_counts.most_common(20),
        "top_entities": top_entities,
        "warning_count": warning_count,
    }

    report_path = out_dir / f"{source_path.stem}{label}.continuity-audit.md"
    report_path.write_text(render_markdown(report, rows), encoding="utf-8")
    json_path = out_dir / f"{source_path.stem}{label}.continuity-audit.summary.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    report["report"] = str(report_path)
    report["summary_json"] = str(json_path)
    return report


def render_markdown(report: dict, rows: list[dict]) -> str:
    high_overhead = [
        r for r in rows
        if r["continuity_tokens"] >= report["continuity_tokens"]["p95"]
    ][:12]
    longest_prompts = sorted(rows, key=lambda r: r["prompt_tokens_est"], reverse=True)[:12]

    lines = [
        "# Literary Continuity Prompt Audit",
        "",
        "Dry run only: no LLM/API calls were made. The continuity memory was advanced with source text as a stand-in translation so prompt construction could be audited without spending tokens.",
        "",
        "## Source",
        "",
        f"- Source file: `{report['source']}`",
        f"- Clean text: `{report['clean_text']}`",
        f"- Per-prompt JSONL: `{report['jsonl']}`",
        f"- Prompt samples: `{report['samples']}`",
        f"- Clean book characters: {report['book_chars']:,}",
        f"- Chunks audited: {report['chunks']:,}",
        "",
        "## Text Profile",
        "",
        f"- Detected kind: {report['profile']['kind']}",
        f"- Confidence: {report['profile']['confidence']:.2f}",
        f"- Signals: {report['profile']['signals']}",
        "",
        "## Prompt Budget",
        "",
        f"- Main chunk tokens: median {report['main_tokens']['median']}, p95 {report['main_tokens']['p95']}, max {report['main_tokens']['max']}",
        f"- Total prompt tokens: median {report['prompt_tokens_est']['median']}, p95 {report['prompt_tokens_est']['p95']}, max {report['prompt_tokens_est']['max']}",
        f"- Continuity tokens: median {report['continuity_tokens']['median']}, p95 {report['continuity_tokens']['p95']}, max {report['continuity_tokens']['max']}",
        f"- Continuity share of prompt: median {report['continuity_prompt_ratio']['median']:.2%}, p95 {report['continuity_prompt_ratio']['p95']:.2%}",
        f"- Hard-truncated continuity blocks: {report['continuity_truncated_chunks']}",
        f"- Omitted continuity lines by token budget: total {report['continuity_omitted_lines']['total']}, p95 {report['continuity_omitted_lines']['p95']}, max {report['continuity_omitted_lines']['max']}",
        "",
        "## Memory Shape",
        "",
        f"- Final tracked names/entities: {report['tracked_character_count_final']}",
        f"- P95 tracked names before a prompt: {report['tracked_character_count_p95']}",
        f"- P95 names detected in a chunk: {report['names_per_chunk_p95']}",
        f"- Unique memory blocks: {report['unique_memory_blocks']} of {report['chunks']}",
        f"- Reused memory blocks: {report['repeated_memory_blocks']}",
        f"- Sections detected: {report['sections_detected']}",
        f"- Continuity warnings: {report['warning_count']}",
        "",
        "## Top Entities",
        "",
    ]
    for entity in report["top_entities"][:30]:
        alias_text = f", aliases {entity['aliases'][:3]}" if entity.get("aliases") else ""
        target_text = f", glossary `{entity['target']}`" if entity.get("target") else ""
        lines.append(
            f"- {entity['name']} [{entity.get('category') or 'entity'}]: {entity['occurrences']} hits, chunks "
            f"{entity['first_chunk']}-{entity['last_chunk']}{alias_text}{target_text}"
        )

    lines.extend(["", "## Longest Prompts", ""])
    for row in longest_prompts:
        lines.append(
            f"- Chunk {row['chunk_index']}: {row['prompt_tokens_est']} tokens "
            f"(main {row['main_tokens']}, memory {row['continuity_tokens']}), section `{row['section']}`"
        )

    lines.extend(["", "## Highest Memory Overhead", ""])
    for row in high_overhead:
        lines.append(
            f"- Chunk {row['chunk_index']}: memory {row['continuity_tokens']} tokens, "
            f"{row['continuity_chars']} chars, omitted lines {row['continuity_omitted_lines']}, "
            f"tracked before {row['tracked_character_count_before']}"
        )

    lines.extend(["", "## Automatic Insights", ""])
    lines.extend(derive_insights(report))
    lines.append("")
    return "\n".join(lines)


def derive_insights(report: dict) -> list[str]:
    insights: list[str] = []
    continuity_p95 = report["continuity_tokens"]["p95"]
    ratio_p95 = report["continuity_prompt_ratio"]["p95"]
    final_entities = report["tracked_character_count_final"]
    sections = report["sections_detected"]

    if ratio_p95 > 0.22:
        insights.append(
            "- Memory overhead is high in the upper tail. Add a token-aware memory budget rather than the current character-only clamp."
        )
    else:
        insights.append(
            "- Memory overhead stayed bounded in the upper tail; the compact block is viable for long books."
        )
    if report["continuity_truncated_chunks"] > 0:
        insights.append(
            "- Some memory blocks were hard-truncated. Prefer ranked omission of low-value lines over cutting the rendered text."
        )
    else:
        insights.append(
            "- No memory block was hard-truncated; low-priority continuity lines were omitted cleanly when needed."
        )
    if final_entities > 1000:
        insights.append(
            "- The entity memory is too inclusive for a novel this long. Add decay or category-specific caps so incidental capitalized phrases do not crowd out characters."
        )
    elif final_entities >= 600:
        insights.append(
            "- Entity memory is bounded at the configured cap. For very long novels, review the top entities and tune `continuity_max_entities` only if important characters are missing."
        )
    if sections < 25:
        insights.append(
            "- Section detection appears weak for this book. Chapter headings should be extracted at chunking time and carried as metadata."
        )
    else:
        insights.append(
            "- Section detection found many boundaries, but it should still be made explicit metadata instead of inferred repeatedly from chunk text."
        )
    if report["warning_count"] > report["chunks"] * 0.2:
        insights.append(
            "- Name-drift warnings are noisy in dry-run mode. In live translation they should compare source-name glossary decisions, not raw target text."
        )
    insights.append(
        "- The audit used source-as-translation to avoid API spend, so it measures prompt/memory shape, not translation quality."
    )
    return insights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out-dir", default="translation_samples/audits")
    parser.add_argument("--source-language", default="English")
    parser.add_argument("--target-language", default="Spanish")
    parser.add_argument("--text-type", default="literature", choices=["auto", "literature", "general"])
    parser.add_argument("--max-tokens-per-chunk", type=int, default=900)
    parser.add_argument("--continuity-max-prompt-chars", type=int, default=1800)
    parser.add_argument("--continuity-max-prompt-tokens", type=int, default=220)
    parser.add_argument("--profile-chars", type=int, default=60000)
    parser.add_argument("--label", default="v2")
    return parser.parse_args()


def main() -> None:
    report = audit(parse_args())
    print(json.dumps({
        "chunks": report["chunks"],
        "report": report["report"],
        "jsonl": report["jsonl"],
        "samples": report["samples"],
        "summary_json": report["summary_json"],
    }, indent=2))


if __name__ == "__main__":
    main()
