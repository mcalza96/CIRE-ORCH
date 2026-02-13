#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_SCRIPT="$BASE_DIR/tools/ingestion-client/ing.sh"

RAG_LOCAL_URL="${RAG_ENGINE_LOCAL_URL:-${RAG_ENGINE_URL:-http://localhost:8000}}"
RAG_DOCKER_URL="${RAG_ENGINE_DOCKER_URL:-http://localhost:8000}"
RAG_HEALTH_PATH="${RAG_ENGINE_HEALTH_PATH:-/health}"
RAG_FORCE_BACKEND="${RAG_ENGINE_FORCE_BACKEND:-}"
RAG_PROBE_TIMEOUT_MS="${RAG_ENGINE_PROBE_TIMEOUT_MS:-300}"

if [ ! -f "$CLIENT_SCRIPT" ]; then
  echo "❌ Missing ingestion client at $CLIENT_SCRIPT"
  exit 1
fi

probe_timeout_seconds() {
  awk -v ms="$RAG_PROBE_TIMEOUT_MS" 'BEGIN { printf "%.3f", (ms + 0) / 1000 }'
}

check_health() {
  local url="$1"
  local timeout_sec
  timeout_sec="$(probe_timeout_seconds)"
  curl -fsS --max-time "$timeout_sec" "${url%/}${RAG_HEALTH_PATH}" >/dev/null 2>&1
}

resolve_rag_url() {
  local forced
  forced="$(printf '%s' "$RAG_FORCE_BACKEND" | tr '[:upper:]' '[:lower:]')"

  if [ "$forced" = "local" ]; then
    printf '%s' "$RAG_LOCAL_URL"
    return
  fi
  if [ "$forced" = "docker" ]; then
    printf '%s' "$RAG_DOCKER_URL"
    return
  fi

  if check_health "$RAG_LOCAL_URL"; then
    printf '%s' "$RAG_LOCAL_URL"
  else
    printf '%s' "$RAG_DOCKER_URL"
  fi
}

if [ -z "${RAG_URL:-}" ]; then
  export RAG_URL="$(resolve_rag_url)"
fi

exec "$CLIENT_SCRIPT" "$@"
