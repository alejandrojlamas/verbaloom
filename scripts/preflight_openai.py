"""Run from the repository root: python -m scripts.preflight_openai."""

import argparse
import asyncio
import json

from src import config
from src.core.llm.preflight import preflight_openai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-effort", choices=["none", "low"], default="none")
    args = parser.parse_args()
    report = asyncio.run(preflight_openai(api_key=config.OPENAI_API_KEY,
                                          reasoning_effort=args.reasoning_effort))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "models_visible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
