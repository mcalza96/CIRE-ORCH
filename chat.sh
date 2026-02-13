#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_URL="${ORCH_URL:-${ORCHESTRATOR_URL:-http://localhost:8001}}"
RAG_URL="${RAG_URL:-${RAG_ENGINE_URL:-http://localhost:8000}}"
VENV_PYTHON="$BASE_DIR/venv/bin/python3"

check_service_health() {
  local service_name="$1"
  local health_url="$2"
  if ! curl -fsS --max-time 2 "$health_url" >/dev/null 2>&1; then
    echo "❌ $service_name no está disponible en $health_url"
    return 1
  fi
  return 0
}

if ! check_service_health "Orchestrator API" "$ORCH_URL/health"; then
  echo "💡 Ejecuta ./dev.sh o ./stack.sh up"
  exit 1
fi

if ! check_service_health "RAG Engine" "$RAG_URL/health"; then
  echo "⚠️  RAG Engine no está disponible en $RAG_URL/health"
  echo "💡 El orquestador requiere un engine externo accesible por HTTP."
fi

read -r -p "🏢 Tenant ID: " CHAT_TENANT_ID
if [[ -z "${CHAT_TENANT_ID:-}" ]]; then
  echo "❌ tenant_id es obligatorio"
  exit 1
fi

read -r -p "📁 Collection ID (opcional): " CHAT_COLLECTION_ID

if [ ! -f "$VENV_PYTHON" ]; then
  echo "❌ Error: No se encontró el entorno virtual en $BASE_DIR/venv"
  exit 1
fi

CLI_ARGS=(--tenant-id "$CHAT_TENANT_ID" --orchestrator-url "$ORCH_URL")
if [ -n "${CHAT_COLLECTION_ID:-}" ]; then
  CLI_ARGS+=(--collection-id "$CHAT_COLLECTION_ID")
fi

export PYTHONPATH="$BASE_DIR:${PYTHONPATH:-}"
"$VENV_PYTHON" "$BASE_DIR/chat_cli.py" "${CLI_ARGS[@]}" "$@"
