"""
Smoke test for the Poe provider.

Verifies that the project's existing factory wiring can authenticate with Poe
using POE_API_KEY / POE_MODEL from .env and complete a single translation
round-trip. Used as the baseline before working on parallel API requests
(GitHub issue #175): if this test passes today, it must still pass after the
parallelism refactor.

Run from repo root:
    python tests/standalone/manual_poe_smoke.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Force UTF-8 on stdout/stderr so library logs containing emoji (e.g. ❌)
# don't crash on Windows consoles that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Make `src` importable when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# Importing src.config has the side effect of calling load_dotenv() on the
# repo's .env file, so POE_API_KEY / POE_MODEL become visible via os.getenv.
from src import config  # noqa: F401
from src.core.llm.factory import create_llm_provider


SOURCE_TEXT = "The quick brown fox jumps over the lazy dog."
SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text from English "
    "into French. Wrap the translation between <TRANSLATION> and </TRANSLATION> "
    "tags and output nothing else."
)
USER_PROMPT = f"<TRANSLATION>\n{SOURCE_TEXT}\n</TRANSLATION>"


async def run() -> int:
    api_key = os.getenv("POE_API_KEY", "").strip()
    if not api_key:
        print("FAIL: POE_API_KEY is not set in .env")
        return 1

    model = os.getenv("POE_MODEL", "Claude-Sonnet-4").strip() or "Claude-Sonnet-4"
    print(f"Provider: poe")
    print(f"Model:    {model}")
    print(f"Source:   {SOURCE_TEXT}")
    print()

    provider = create_llm_provider("poe", model=model, api_key=api_key)

    started = time.perf_counter()
    try:
        response = await provider.generate(USER_PROMPT, system_prompt=SYSTEM_PROMPT)
    finally:
        await provider.close()
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if response is None:
        print(f"FAIL: provider returned None after {elapsed_ms} ms")
        return 1

    translation = provider.extract_translation(response.content)
    if not translation:
        print(f"FAIL: could not extract translation tags from response after {elapsed_ms} ms")
        print("Raw response:")
        print(response.content)
        return 1

    print(f"OK in {elapsed_ms} ms")
    print(f"Translation: {translation.strip()}")
    if getattr(response, "input_tokens", None) is not None:
        print(f"Tokens:      in={response.input_tokens} out={response.output_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))
