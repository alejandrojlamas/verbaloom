#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

exec "$PROJECT_DIR/venv/bin/python" -c 'import webbrowser; webbrowser.open = lambda *args, **kwargs: False; import translation_api; translation_api.start_server()'
