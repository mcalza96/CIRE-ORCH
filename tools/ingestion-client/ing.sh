#!/bin/bash

# Ingestion Client Wrapper
# This script now delegates all logic to the Python runtime 'app.ing_cli_runtime'.
# It is kept for backward compatibility and portability.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ensure we are in the repo root to find the 'app' package
cd "$REPO_ROOT"

# Use local venv if available
PYTHON_BIN="python3"
if [[ -f "$REPO_ROOT/.venv/bin/python3" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python3"
fi

# Delegate to Python runtime
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
exec "$PYTHON_BIN" -m app.ui.cli.ing_cli_runtime "$@"
