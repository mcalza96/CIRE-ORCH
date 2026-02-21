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
export RAG_ENGINE_LOCAL_URL="${RAG_ENGINE_LOCAL_URL:-${RAG_ENGINE_URL:-http://localhost:8000}}"
export RAG_ENGINE_DOCKER_URL="${RAG_ENGINE_DOCKER_URL:-http://localhost:8000}"
export RAG_ENGINE_HEALTH_PATH="${RAG_ENGINE_HEALTH_PATH:-/health}"
export RAG_ENGINE_PROBE_TIMEOUT_MS="${RAG_ENGINE_PROBE_TIMEOUT_MS:-300}"
export RAG_ENGINE_BACKEND_TTL_SECONDS="${RAG_ENGINE_BACKEND_TTL_SECONDS:-20}"
export ORCH_API_HOST="${ORCH_API_HOST:-127.0.0.1}"
export ORCH_API_PORT="${ORCH_API_PORT:-8001}"

echo "🚀 Starting Q/A Orchestrator API on ${ORCH_API_HOST}:${ORCH_API_PORT}"
"$VENV_DIR/bin/python" -m uvicorn app.api.server:app --host "$ORCH_API_HOST" --port "$ORCH_API_PORT" --no-access-log
