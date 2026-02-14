#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_python() {
  if [ -x "$BASE_DIR/.venv/bin/python" ]; then
    printf '%s' "$BASE_DIR/.venv/bin/python"
    return
  fi
  if [ -x "$BASE_DIR/venv/bin/python3" ]; then
    printf '%s' "$BASE_DIR/venv/bin/python3"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    printf '%s' "python3"
    return
  fi
  printf '%s' "python"
}

PY_CMD="$(resolve_python)"

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
exec "$PY_CMD" "$BASE_DIR/orch_cli.py" ingest "$@"
