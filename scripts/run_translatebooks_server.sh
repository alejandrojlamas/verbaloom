#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point for existing launchd and local automation.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/run_verbaloom_server.sh" "$@"
