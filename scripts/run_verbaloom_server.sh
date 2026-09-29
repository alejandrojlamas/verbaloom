#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# The launch agent has KeepAlive enabled.  This marker lets the in-process
# watchdog recover a genuinely deadlocked worker by exiting once; launchd then
# starts a clean process and startup recovery resumes the durable checkpoint.
export VERBALOOM_MANAGED_SERVICE=1

exec "$PROJECT_DIR/venv/bin/python" -c 'import webbrowser; webbrowser.open = lambda *args, **kwargs: False; import translation_api; translation_api.start_server()'
