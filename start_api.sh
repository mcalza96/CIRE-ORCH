#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$BASE_DIR/venv"

if [ ! -f "$VENV_DIR/bin/python" ]; then
  echo "❌ Missing virtualenv in $VENV_DIR"
  echo "💡 Run ./bootstrap.sh first"
  exit 1
fi

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
export RAG_ENGINE_URL="${RAG_ENGINE_URL:-http://localhost:8000}"

echo "🚀 Starting Q/A Orchestrator API on :8001"
"$VENV_DIR/bin/python" -m uvicorn runtime.orchestrator_main:app --host 0.0.0.0 --port 8001 --no-access-log
