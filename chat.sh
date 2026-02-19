#!/bin/bash

# Chat CLI Wrapper
# Delegates all logic to 'orch_cli.py' (Python runtime).

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use local venv if available
PYTHON_BIN="$BASE_DIR/venv/bin/python3"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"

exec "$PYTHON_BIN" "$BASE_DIR/orch_cli.py" chat "$@"
