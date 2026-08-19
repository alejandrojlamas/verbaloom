#!/usr/bin/env python3
"""Strict complete-book CLI entrypoint."""

import os
import sys
from pathlib import Path


def _reexec_in_project_venv() -> None:
    if sys.prefix != sys.base_prefix:
        return
    root = Path(__file__).resolve().parent
    candidates = (root / "venv" / "bin" / "python", root / "venv" / "Scripts" / "python.exe")
    for candidate in candidates:
        if candidate.exists() and candidate.resolve() != Path(sys.executable).resolve():
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]])


_reexec_in_project_venv()

from src.core.quality_assurance.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
