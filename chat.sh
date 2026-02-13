#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH_URL="${ORCH_URL:-${ORCHESTRATOR_URL:-http://localhost:8001}}"
RAG_LOCAL_URL="${RAG_ENGINE_LOCAL_URL:-${RAG_ENGINE_URL:-http://localhost:8000}}"
RAG_DOCKER_URL="${RAG_ENGINE_DOCKER_URL:-http://localhost:8000}"
RAG_HEALTH_PATH="${RAG_ENGINE_HEALTH_PATH:-/health}"
RAG_FORCE_BACKEND="${RAG_ENGINE_FORCE_BACKEND:-}"
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

  if check_service_health "RAG Local" "$RAG_LOCAL_URL$RAG_HEALTH_PATH"; then
    printf '%s' "$RAG_LOCAL_URL"
  else
    printf '%s' "$RAG_DOCKER_URL"
  fi
}

fetch_tenants() {
  local rag_url="$1"
  local py_cmd
  if [ -f "$VENV_PYTHON" ]; then
    py_cmd="$VENV_PYTHON"
  else
    py_cmd="python3"
  fi

  "$py_cmd" - "$rag_url" <<'PY'
import json
import sys
import urllib.request
import urllib.error

base = sys.argv[1].rstrip("/")
paths = [
    "/api/v1/tenants",
    "/api/v1/ingestion/tenants",
    "/api/v1/retrieval/tenants",
    "/tenants",
]
documents_fallback_path = "/api/v1/ingestion/documents?limit=100"

def extract(payload):
    if isinstance(payload, list):
        return [str(x).strip() for x in payload if str(x).strip()]
    if isinstance(payload, dict):
        for key in ("tenants", "items", "data", "results"):
            val = payload.get(key)
            if isinstance(val, list):
                out = []
                for item in val:
                    if isinstance(item, str):
                        if item.strip():
                            out.append(item.strip())
                    elif isinstance(item, dict):
                        candidate = item.get("tenant_id") or item.get("id") or item.get("tenant")
                        if candidate:
                            out.append(str(candidate).strip())
                if out:
                    return out
    return []

def extract_from_documents(payload):
    out = []
    items = payload if isinstance(payload, list) else payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []

    for item in items:
        if not isinstance(item, dict):
            continue
        institution_id = item.get("institution_id")
        if institution_id:
            out.append(str(institution_id).strip())

        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else {}
        tenant_id = nested.get("tenant_id") or metadata.get("tenant_id")
        if tenant_id:
            out.append(str(tenant_id).strip())

    return [x for x in dict.fromkeys(out) if x]

for path in paths:
    req = urllib.request.Request(base + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            if resp.status != 200:
                continue
            data = json.loads(resp.read().decode("utf-8"))
            tenants = extract(data)
            if tenants:
                print("\n".join(dict.fromkeys(tenants)))
                sys.exit(0)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        continue

req = urllib.request.Request(base + documents_fallback_path, method="GET")
try:
    with urllib.request.urlopen(req, timeout=1.8) as resp:
        if resp.status == 200:
            data = json.loads(resp.read().decode("utf-8"))
            tenants = extract_from_documents(data)
            if tenants:
                print("\n".join(tenants))
                sys.exit(0)
except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
    pass

sys.exit(0)
PY
}

if ! check_service_health "Orchestrator API" "$ORCH_URL/health"; then
  echo "💡 Ejecuta ./stack.sh up"
  exit 1
fi

RAG_URL="$(resolve_rag_url)"
TENANTS="$(fetch_tenants "$RAG_URL" || true)"

if [ -n "$TENANTS" ]; then
  echo "📚 Tenants detectados en $RAG_URL:"
  echo "$TENANTS" | nl -w1 -s') '
  echo ""
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
