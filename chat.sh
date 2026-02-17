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

EXTRA_ARGS=()
if [ "${ORCH_CHAT_OBS:-1}" != "0" ]; then
  has_obs=0
  for arg in "$@"; do
    if [ "$arg" = "--obs" ]; then
      has_obs=1
      break
    fi
  done
  if [ "$has_obs" -eq 0 ]; then
    EXTRA_ARGS+=("--obs")
  fi
fi

exec "$PY_CMD" "$BASE_DIR/orch_cli.py" chat "${EXTRA_ARGS[@]}" "$@"
