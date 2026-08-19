"""Print line-count statistics for the repository's source code.

Usage:
    python scripts/code_stats.py

Reports the total line count across all tracked code files, the top 10
largest files, a per-extension breakdown, and warns about any file
exceeding 1000 lines.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

CODE_EXTENSIONS = {".py", ".js", ".html", ".css", ".json", ".sh", ".bat", ".yml", ".yaml"}

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    "build",
    "dist",
    "translated_files",
    "data",
    ".telemetry",
    ".wiki_repo",
    ".wiki_repo_archive",
    "benchmark_results",
    "output",
    "profiles",
    "test-results",
    ".playwright-cli",
    "plan",
    "experiments",
    ".idea",
    ".claude",
    "DoNotCommit",
    "prompt_optimization_results",
    ".watermark",
}

LARGE_FILE_THRESHOLD = 1000


def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def is_excluded(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(
        part in EXCLUDED_DIRS or part.startswith("profiles_backup_")
        for part in rel_parts
    )


def collect_stats(root: Path) -> list[tuple[Path, int]]:
    results: list[tuple[Path, int]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if is_excluded(path, root):
            continue
        results.append((path, count_lines(path)))
    return results


def format_row(rank: int, rel_path: str, lines: int, warn: bool) -> str:
    marker = "  WARNING >1000 lines" if warn else ""
    return f"{rank:>3}. {lines:>6}  {rel_path}{marker}"


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    stats = collect_stats(root)

    if not stats:
        print("No source files found.")
        return 0

    total_lines = sum(lines for _, lines in stats)
    total_files = len(stats)

    by_ext: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for path, lines in stats:
        files_count, line_count = by_ext[path.suffix.lower()]
        by_ext[path.suffix.lower()] = (files_count + 1, line_count + lines)

    stats.sort(key=lambda item: item[1], reverse=True)
    largest = stats[:10]

    print(f"Repository: {root}")
    print(f"Total files scanned: {total_files}")
    print(f"Total lines:         {total_lines}")
    print()

    print("Lines by extension:")
    for ext, (files_count, line_count) in sorted(
        by_ext.items(), key=lambda item: item[1][1], reverse=True
    ):
        print(f"  {ext:<6} {files_count:>4} files  {line_count:>8} lines")
    print()

    print("Top 10 largest files:")
    warned_any = False
    for rank, (path, lines) in enumerate(largest, start=1):
        rel = path.relative_to(root).as_posix()
        warn = lines > LARGE_FILE_THRESHOLD
        warned_any = warned_any or warn
        print(format_row(rank, rel, lines, warn))

    over_threshold = [(p, n) for p, n in stats if n > LARGE_FILE_THRESHOLD]
    if over_threshold:
        print()
        print(f"Files over {LARGE_FILE_THRESHOLD} lines: {len(over_threshold)}")
        for path, lines in over_threshold:
            rel = path.relative_to(root).as_posix()
            print(f"  {lines:>6}  {rel}")
    elif warned_any:
        pass
    else:
        print()
        print(f"No file exceeds {LARGE_FILE_THRESHOLD} lines.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
