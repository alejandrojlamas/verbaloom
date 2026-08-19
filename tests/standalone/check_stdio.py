"""
Standalone diagnostic: prove that the stdio encoding fix is active.

Run with EXACTLY the same Python invocation the server uses:
    python tests/standalone/check_stdio.py

If the fix is loaded and effective, you should see all four lines below.
If only the first appears and Python errors out, the entry-point fix did
not reach this process and the Poe-print crash will keep happening in
the server.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.stdio_setup import configure_stdio_utf8
configure_stdio_utf8()

print(f"stdout.encoding = {sys.stdout.encoding!r}")
print("ASCII line: hello")
print("💬 Poe-style emoji line")
print("⚠️ ❌ ✅ 📄 mix of emoji")
print("DONE")
