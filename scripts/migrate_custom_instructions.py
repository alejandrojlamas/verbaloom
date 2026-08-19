"""Migrate Custom_Instructions/*.txt presets to the new YAML format.

The new format is a YAML mapping with two optional top-level keys:

    translation: |
      <prose to inject into the translation prompt>
    refinement: |
      <prose to inject into the refinement prompt>

This script handles the mechanical half of the migration:

1. `--init`   — for each `.txt` preset, create a `<name>.yaml` next to it
                with `translation:` populated from the original text and
                `refinement: ""` left empty as a marker for human/LLM follow-up.
                Skips presets that already have a `.yaml`.

2. `--set-refinement <name> <text>` — write a refinement string into an
                existing preset's YAML (used by orchestration to fill in the
                refinement section per preset).

   `--set-refinement-from-file <name> <path>` — same as above, but reads the
                refinement text from a file. Use this to avoid shell quoting
                issues for multi-line content.

3. `--cleanup-txt` — delete `.txt` originals when a matching `.yaml` exists
                AND its refinement section is non-empty. Safe to re-run; will
                not delete anything if the refinement is still empty.

Usage:

    python scripts/migrate_custom_instructions.py --init
    python scripts/migrate_custom_instructions.py --set-refinement noir_detective "..."
    python scripts/migrate_custom_instructions.py --cleanup-txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INSTR_DIR = ROOT / "Custom_Instructions"


def _yaml_dump(payload: dict) -> str:
    """Block-style YAML dump that preserves multi-line strings cleanly."""

    class LiteralStr(str):
        pass

    def literal_str_representer(dumper, data):  # noqa: ANN001
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    yaml.add_representer(LiteralStr, literal_str_representer)

    coerced = {}
    for key in ("translation", "refinement"):
        if key not in payload:
            continue
        value = payload[key]
        if isinstance(value, str) and value and ("\n" in value or len(value) > 80):
            coerced[key] = LiteralStr(value)
        else:
            coerced[key] = value

    return yaml.dump(coerced, allow_unicode=True, sort_keys=False, width=1000)


def cmd_init() -> int:
    if not INSTR_DIR.exists():
        print(f"ERROR: {INSTR_DIR} not found", file=sys.stderr)
        return 1

    created = 0
    skipped = 0
    for txt_path in sorted(INSTR_DIR.glob("*.txt")):
        yaml_path = txt_path.with_suffix(".yaml")
        if yaml_path.exists():
            skipped += 1
            continue

        body = txt_path.read_text(encoding="utf-8").strip()
        if not body:
            print(f"WARN  empty preset, skipping: {txt_path.name}", file=sys.stderr)
            skipped += 1
            continue

        payload = {"translation": body, "refinement": ""}
        yaml_path.write_text(_yaml_dump(payload), encoding="utf-8")
        print(f"INIT  {yaml_path.name}")
        created += 1

    print(f"\n{created} created, {skipped} skipped.")
    return 0


def cmd_set_refinement(name: str, text: str) -> int:
    yaml_path = INSTR_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} does not exist (run --init first)", file=sys.stderr)
        return 1

    text = text.strip()
    if not text:
        print(f"ERROR: refinement text is empty for {name}", file=sys.stderr)
        return 1

    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        print(f"ERROR: {yaml_path} is not a mapping", file=sys.stderr)
        return 1

    payload["refinement"] = text
    yaml_path.write_text(_yaml_dump(payload), encoding="utf-8")
    print(f"SET   {yaml_path.name}")
    return 0


def cmd_cleanup_txt() -> int:
    if not INSTR_DIR.exists():
        print(f"ERROR: {INSTR_DIR} not found", file=sys.stderr)
        return 1

    removed = 0
    kept = 0
    for txt_path in sorted(INSTR_DIR.glob("*.txt")):
        yaml_path = txt_path.with_suffix(".yaml")
        if not yaml_path.exists():
            print(f"KEEP  {txt_path.name} (no matching .yaml)")
            kept += 1
            continue

        try:
            payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            print(f"KEEP  {txt_path.name} (malformed .yaml)", file=sys.stderr)
            kept += 1
            continue

        refinement = (payload.get("refinement") or "").strip() if isinstance(payload, dict) else ""
        if not refinement:
            print(f"KEEP  {txt_path.name} (refinement still empty in .yaml)")
            kept += 1
            continue

        txt_path.unlink()
        print(f"RM    {txt_path.name}")
        removed += 1

    print(f"\n{removed} removed, {kept} kept.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true", help="Bootstrap .yaml stubs from .txt presets.")
    group.add_argument(
        "--set-refinement",
        nargs=2,
        metavar=("NAME", "TEXT"),
        help="Write a refinement section into <NAME>.yaml.",
    )
    group.add_argument(
        "--set-refinement-from-file",
        nargs=2,
        metavar=("NAME", "PATH"),
        help="Same as --set-refinement but reads TEXT from a file.",
    )
    group.add_argument(
        "--cleanup-txt",
        action="store_true",
        help="Delete .txt presets that have a populated .yaml counterpart.",
    )

    args = parser.parse_args()

    if args.init:
        return cmd_init()
    if args.set_refinement:
        return cmd_set_refinement(args.set_refinement[0], args.set_refinement[1])
    if args.set_refinement_from_file:
        name, path = args.set_refinement_from_file
        text = Path(path).read_text(encoding="utf-8")
        return cmd_set_refinement(name, text)
    if args.cleanup_txt:
        return cmd_cleanup_txt()
    return 1


if __name__ == "__main__":
    sys.exit(main())
