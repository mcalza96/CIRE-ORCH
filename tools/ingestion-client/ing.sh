#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${INGEST_CLIENT_STATE_DIR:-$SCRIPT_DIR}"
RAG_URL="${RAG_URL:-${RAG_SERVICE_URL:-http://localhost:8000}}"
ORCH_URL="${ORCH_URL:-${ORCHESTRATOR_URL:-http://localhost:8001}}"
REGISTRY_FILE="$STATE_DIR/.rag_tenants.json"
RAG_API_KEY="${RAG_API_KEY:-${API_KEY:-${RAG_SERVICE_SECRET:-}}}"
RAG_SERVICE_SECRET="${RAG_SERVICE_SECRET:-}"
ORCH_ACCESS_TOKEN="${ORCH_ACCESS_TOKEN:-${SUPABASE_ACCESS_TOKEN:-${AUTH_BEARER_TOKEN:-}}}"

TENANT_ID="${TENANT_ID:-}"
TENANT_NAME="${TENANT_NAME:-}"
COLLECTION_ID="${COLLECTION_ID:-}"
COLLECTION_NAME="${COLLECTION_NAME:-}"
COLLECTION_CLEANUP_REQUIRED="false"
EMBEDDING_MODE="${EMBEDDING_MODE:-}"
WAIT_FOR_COMPLETION="true"
BATCH_POLL_INTERVAL="${BATCH_POLL_INTERVAL:-15}"
BATCH_POLL_MAX="${BATCH_POLL_MAX:-120}"
BATCH_SKIP_ESTIMATE="false"
WATCH_MODE="${WATCH_MODE:-stream}"
RESUME_BATCH_ID=""
PROGRESS_JSON="false"
AGENT_PROFILE_ID="${AGENT_PROFILE_ID:-}"
CLEAR_AGENT_PROFILE="false"
BATCH_ID="${BATCH_ID:-}"
VISUAL_RATIO="${VISUAL_ROUTER_MAX_VISUAL_RATIO:-0.5}"
VISUAL_MAX_PAGES="${VISUAL_ROUTER_MAX_VISUAL_PAGES:-100}"
GEMINI_COST_PER_VISUAL_PAGE_USD="${GEMINI_COST_PER_VISUAL_PAGE_USD:-0.00003}"

declare -a FILE_QUEUE=()
declare -a GLOB_PATTERNS=()
declare -a AUTH_CURL_ARGS=()
declare -a ORCH_AUTH_CURL_ARGS=()

STEP_HEALTH="pendiente"
STEP_TENANT="pendiente"
STEP_PROFILE="pendiente"
STEP_COLLECTION="pendiente"
STEP_QUEUE="pendiente"
STEP_PREFLIGHT="pendiente"
STEP_BATCH_CREATE="pendiente"
STEP_BATCH_UPLOAD="pendiente"
STEP_BATCH_SEAL="pendiente"
STEP_WORKER="pendiente"
STEP_WORKER_STATUS="n/a"
STEP_WORKER_DOCS="n/a"
STEP_WORKER_QUEUE="n/a"
STEP_WORKER_ERRORS="n/a"
STEP_WORKER_LOSS="n/a"
STEP_WORKER_PHASE="n/a"
STEP_WORKER_REFS="n/a"

render_checklist() {
  echo ""
  echo "📋 Checklist de ejecución"
  echo "1. Salud API: $STEP_HEALTH"
  echo "2. Tenant: $STEP_TENANT"
  echo "3. Cartucho por tenant: $STEP_PROFILE"
  echo "4. Colección: $STEP_COLLECTION"
  echo "5. Cola de archivos: $STEP_QUEUE"
  echo "6. Pre-estimación visual: $STEP_PREFLIGHT"
  echo "7. Crear batch: $STEP_BATCH_CREATE"
  echo "8. Cargar archivos al batch: $STEP_BATCH_UPLOAD"
  echo "9. Sellar batch: $STEP_BATCH_SEAL"
  echo "10. Procesamiento worker: $STEP_WORKER"
  echo "   10.1 Estado runtime: $STEP_WORKER_STATUS"
  echo "   10.2 Documentos terminales: $STEP_WORKER_DOCS"
  echo "   10.3 Cola activa: $STEP_WORKER_QUEUE"
  echo "   10.4 Fallos acumulados: $STEP_WORKER_ERRORS"
  echo "   10.5 Riesgo pérdida visual: $STEP_WORKER_LOSS"
  echo "   10.6 Fase worker: $STEP_WORKER_PHASE"
  echo "   10.7 Referencias: $STEP_WORKER_REFS"
}

usage() {
  cat <<'EOF'
Uso:
  ./ing.sh [opciones]

Opciones:
  --tenant-id <uuid>           Tenant existente
  --tenant-name <texto>        Alias descriptivo del tenant
  --collection-id <id>         Carpeta/Colección (id lógico)
  --collection-name <texto>    Carpeta/Colección (nombre)
  --agent-profile <id>         Asigna override dev de cartucho para el tenant
  --clear-agent-profile        Limpia override dev de cartucho para el tenant
  --file <ruta>                Archivo a encolar (repetible)
  --glob <patron>              Patrón para encolar (repetible)
  --embedding-mode <modo>      LOCAL o CLOUD (si no, se pregunta)
  --no-wait                    No espera cierre del procesamiento del batch
  --skip-estimate              Omite estimación visual pre-ingesta
  --watch-mode <stream|poll>   Observación worker (default: stream)
  --resume-batch <batch_id>    Reanudar monitoreo de un batch existente
  --progress-json              Imprime snapshots JSON de progreso
  --rag-url <url>              URL base API RAG (default: http://localhost:8000)
  -h, --help                   Mostrar ayuda

Ejemplos:
  ./ing.sh --tenant-id 11111111-1111-1111-1111-111111111111 --collection-name trinorma --file ./docs/iso9001.pdf --file ./docs/iso14001.pdf
  ./ing.sh --tenant-id <uuid> --agent-profile iso_auditor --collection-name normas --file ./docs/manual.pdf
  ./ing.sh --collection-name normas --glob "./docs/*.md"
  ./ing.sh --collection-name iso-v2 --no-wait
  ./ing.sh --tenant-id <uuid> --resume-batch <batch_id> --watch-mode stream
  GEMINI_COST_PER_VISUAL_PAGE_USD=0.00003 ./ing.sh --collection-name iso-v2
EOF
}

validate_watch_mode() {
  case "$WATCH_MODE" in
    stream|poll) ;;
    *)
      echo "❌ --watch-mode inválido: $WATCH_MODE (usa stream o poll)"
      exit 1
      ;;
  esac
}

build_auth_curl_args() {
  AUTH_CURL_ARGS=()
  if [[ -n "${RAG_API_KEY:-}" ]]; then
    AUTH_CURL_ARGS+=(-H "Authorization: Bearer $RAG_API_KEY")
  fi
  if [[ -n "${RAG_SERVICE_SECRET:-}" ]]; then
    AUTH_CURL_ARGS+=(-H "X-Service-Secret: $RAG_SERVICE_SECRET")
  fi
}

build_orch_auth_curl_args() {
  ORCH_AUTH_CURL_ARGS=()
  if [[ -n "${ORCH_ACCESS_TOKEN:-}" ]]; then
    ORCH_AUTH_CURL_ARGS+=(-H "Authorization: Bearer $ORCH_ACCESS_TOKEN")
  fi
}

fetch_api_tenants() {
  local status
  status=$(curl -sS -o /tmp/rag_tenants.json -w "%{http_code}" --max-time 4 \
    ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
    "$RAG_URL/api/v1/management/tenants?limit=200" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

try:
    payload = json.loads(Path("/tmp/rag_tenants.json").read_text(encoding="utf-8"))
except Exception:
    payload = {}

items = payload if isinstance(payload, list) else payload.get("items", [])
for item in items:
    if not isinstance(item, dict):
        continue
    tid = str(item.get("id") or item.get("tenant_id") or "").strip()
    name = str(item.get("name") or item.get("tenant_name") or tid).strip()
    if tid:
        print(f"{tid}|{name or tid}")
PY
}

fetch_orch_tenants() {
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    return 0
  fi

  local status
  status=$(curl -sS -o /tmp/orch_tenants.json -w "%{http_code}" --max-time 4 \
    -H "Authorization: Bearer $ORCH_ACCESS_TOKEN" \
    "$ORCH_URL/api/v1/knowledge/tenants" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

try:
    payload = json.loads(Path("/tmp/orch_tenants.json").read_text(encoding="utf-8"))
except Exception:
    payload = {}

items = payload if isinstance(payload, list) else payload.get("items", [])
for item in items:
    if not isinstance(item, dict):
        continue
    tid = str(item.get("id") or "").strip()
    name = str(item.get("name") or tid).strip()
    if tid:
        print(f"{tid}|{name or tid}")
PY
}

fetch_orch_collections() {
  local tenant_id="$1"
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    return 0
  fi

  local status
  status=$(curl -sS -o /tmp/orch_collections.json -w "%{http_code}" --max-time 4 \
    -H "Authorization: Bearer $ORCH_ACCESS_TOKEN" \
    "$ORCH_URL/api/v1/knowledge/collections?tenant_id=$tenant_id" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

try:
    payload = json.loads(Path("/tmp/orch_collections.json").read_text(encoding="utf-8"))
except Exception:
    payload = {}

items = payload if isinstance(payload, list) else payload.get("items", [])
for item in items:
    if not isinstance(item, dict):
        continue
    cid = str(item.get("id") or "").strip()
    key = str(item.get("collection_key") or "").strip()
    name = str(item.get("name") or key or cid).strip()
    if cid:
        print(f"{cid}|{key}|{name}|open")
PY
}

fetch_orch_agent_profiles() {
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    return 0
  fi

  local status
  status=$(curl -sS -o /tmp/orch_agent_profiles.json -w "%{http_code}" --max-time 4 \
    -H "Authorization: Bearer $ORCH_ACCESS_TOKEN" \
    "$ORCH_URL/api/v1/knowledge/agent-profiles" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

try:
    payload = json.loads(Path("/tmp/orch_agent_profiles.json").read_text(encoding="utf-8"))
except Exception:
    payload = {}

items = payload if isinstance(payload, list) else payload.get("items", [])
for item in items:
    if not isinstance(item, dict):
        continue
    profile_id = str(item.get("id") or "").strip()
    if not profile_id:
        continue
    version = str(item.get("version") or "").strip()
    status = str(item.get("status") or "").strip()
    description = str(item.get("description") or "").strip().replace("|", "/")
    owner = str(item.get("owner") or "").strip().replace("|", "/")
    print(f"{profile_id}|{version}|{status}|{description}|{owner}")
PY
}

fetch_orch_tenant_profile() {
  local tenant_id="$1"
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    return 0
  fi

  local status
  status=$(curl -sS -o /tmp/orch_tenant_profile.json -w "%{http_code}" --max-time 4 \
    -H "Authorization: Bearer $ORCH_ACCESS_TOKEN" \
    -H "X-Tenant-ID: $tenant_id" \
    "$ORCH_URL/api/v1/knowledge/tenant-profile?tenant_id=$tenant_id" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 0
  fi

  python3 - <<'PY'
import json
from pathlib import Path

try:
    payload = json.loads(Path("/tmp/orch_tenant_profile.json").read_text(encoding="utf-8"))
except Exception:
    payload = {}

override = str(payload.get("override_profile_id") or "").strip()
resolution = payload.get("resolution") if isinstance(payload.get("resolution"), dict) else {}
source = str(resolution.get("source") or "").strip()
applied = str(resolution.get("applied_profile_id") or "").strip()
reason = str(resolution.get("decision_reason") or "").strip().replace("|", "/")
print(f"{override}|{source}|{applied}|{reason}")
PY
}

set_or_clear_orch_tenant_profile() {
  local tenant_id="$1"
  local profile_id="$2"
  local clear_flag="$3"
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    return 1
  fi

  local payload
  payload=$(python3 - "$tenant_id" "$profile_id" "$clear_flag" <<'PY'
import json
import sys

tenant_id = sys.argv[1]
profile_id = sys.argv[2].strip()
clear_flag = sys.argv[3].strip().lower() in {"1", "true", "yes", "on"}
body = {"tenant_id": tenant_id, "clear": clear_flag}
if profile_id:
    body["profile_id"] = profile_id
print(json.dumps(body, ensure_ascii=True))
PY
)

  local status
  status=$(curl -sS -o /tmp/orch_tenant_profile_put.json -w "%{http_code}" --max-time 6 \
    -X PUT \
    -H "Authorization: Bearer $ORCH_ACCESS_TOKEN" \
    -H "X-Tenant-ID: $tenant_id" \
    -H "Content-Type: application/json" \
    "$ORCH_URL/api/v1/knowledge/tenant-profile" \
    -d "$payload" || true)

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    return 1
  fi
  return 0
}

uuid_v4() {
  python3 - <<'PY'
import uuid
print(uuid.uuid4())
PY
}

slugify() {
  local input="$1"
  python3 - "$input" <<'PY'
import re
import sys
raw = sys.argv[1].strip().lower()
raw = re.sub(r"[^a-z0-9]+", "-", raw)
raw = re.sub(r"-+", "-", raw).strip("-")
print(raw or "default")
PY
}

ensure_registry() {
  mkdir -p "$(dirname "$REGISTRY_FILE")"
  if [[ ! -f "$REGISTRY_FILE" ]]; then
    printf '{"tenants": []}\n' > "$REGISTRY_FILE"
  fi
}

register_tenant() {
  local tenant_id="$1"
  local tenant_name="$2"
  ensure_registry

  python3 - "$REGISTRY_FILE" "$tenant_id" "$tenant_name" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
tenant_id = sys.argv[2]
tenant_name = sys.argv[3]

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = {"tenants": []}

tenants = data.setdefault("tenants", [])
for t in tenants:
    if str(t.get("id")) == tenant_id:
        if tenant_name:
            t["name"] = tenant_name
        break
else:
    tenants.append({"id": tenant_id, "name": tenant_name or tenant_id})

path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
PY
}

select_or_create_tenant() {
  build_auth_curl_args

  if [[ -n "$TENANT_ID" ]]; then
    [[ -n "$TENANT_NAME" ]] && register_tenant "$TENANT_ID" "$TENANT_NAME"
    return 0
  fi

  ensure_registry

  local registry_tenants
  registry_tenants=$(python3 - "$REGISTRY_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    data = {"tenants": []}

for t in data.get("tenants", []):
    print(f"{t.get('id','')}|{t.get('name','')}")
PY
)
  local api_tenants
  api_tenants=""
  # When user auth is available, tenant discovery must come from ORCH authorization.
  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    api_tenants="$(fetch_api_tenants || true)"
  fi
  local orch_tenants
  orch_tenants="$(fetch_orch_tenants || true)"

  echo ""
  echo "🏢 Tenant setup"

  local tenant_ids=()
  local tenant_names=()
  local seen="|"
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local tid="${line%%|*}"
    local tname="${line#*|}"
    [[ -z "$tid" ]] && continue
    if [[ "$seen" == *"|$tid|"* ]]; then
      continue
    fi
    seen+="$tid|"
    tenant_ids+=("$tid")
    tenant_names+=("${tname:-$tid}")
  done <<< "$orch_tenants"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local tid="${line%%|*}"
    local tname="${line#*|}"
    [[ -z "$tid" ]] && continue
    if [[ "$seen" == *"|$tid|"* ]]; then
      continue
    fi
    seen+="$tid|"
    tenant_ids+=("$tid")
    tenant_names+=("${tname:-$tid}")
  done <<< "$api_tenants"

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local tid="${line%%|*}"
    local tname="${line#*|}"
    [[ -z "$tid" ]] && continue
    if [[ "$seen" == *"|$tid|"* ]]; then
      continue
    fi
    seen+="$tid|"
    tenant_ids+=("$tid")
    tenant_names+=("${tname:-$tid}")
  done <<< "$registry_tenants"

  echo "[1] Crear tenant nuevo"
  local i
  for ((i=0; i<${#tenant_ids[@]}; i++)); do
    printf "[%d] Usar %s (%s)\n" "$((i+2))" "${tenant_names[$i]}" "${tenant_ids[$i]}"
  done

  local opt
  read -r -p "📝 Opción [1]: " opt
  [[ -z "$opt" ]] && opt=1

  if ! [[ "$opt" =~ ^[0-9]+$ ]] || (( opt < 1 || opt > ${#tenant_ids[@]} + 1 )); then
    echo "❌ Opción inválida"
    exit 1
  fi

  if (( opt == 1 )); then
    TENANT_ID="$(uuid_v4)"
    read -r -p "Nombre tenant (opcional): " TENANT_NAME
    [[ -z "$TENANT_NAME" ]] && TENANT_NAME="$TENANT_ID"
    register_tenant "$TENANT_ID" "$TENANT_NAME"
    echo "✅ Tenant creado: $TENANT_ID"
  else
    local idx=$((opt-2))
    TENANT_ID="${tenant_ids[$idx]}"
    TENANT_NAME="${tenant_names[$idx]}"
    echo "✅ Tenant seleccionado: $TENANT_NAME ($TENANT_ID)"
  fi
}

select_or_assign_profile() {
  build_orch_auth_curl_args

  if [[ -z "${ORCH_ACCESS_TOKEN:-}" ]]; then
    if [[ "$CLEAR_AGENT_PROFILE" == "true" || -n "$AGENT_PROFILE_ID" ]]; then
      echo "⚠️  No hay ORCH_ACCESS_TOKEN; no se pudo aplicar override de cartucho."
      STEP_PROFILE="omitido (sin ORCH_ACCESS_TOKEN)"
      return 0
    fi
    STEP_PROFILE="omitido (sin ORCH_ACCESS_TOKEN)"
    return 0
  fi

  if [[ "$CLEAR_AGENT_PROFILE" == "true" ]]; then
    if set_or_clear_orch_tenant_profile "$TENANT_ID" "" "true"; then
      echo "✅ Override de cartucho limpiado para tenant $TENANT_ID"
      STEP_PROFILE="completado (override limpiado)"
    else
      echo "⚠️  No se pudo limpiar override de cartucho (endpoint no disponible)."
      STEP_PROFILE="omitido (endpoint no disponible)"
    fi
    return 0
  fi

  if [[ -n "$AGENT_PROFILE_ID" ]]; then
    if set_or_clear_orch_tenant_profile "$TENANT_ID" "$AGENT_PROFILE_ID" "false"; then
      echo "✅ Cartucho asignado por flag: $AGENT_PROFILE_ID"
      STEP_PROFILE="completado (override=$AGENT_PROFILE_ID)"
    else
      echo "⚠️  No se pudo asignar cartucho '$AGENT_PROFILE_ID' (endpoint no disponible o perfil inválido)."
      STEP_PROFILE="omitido (set fallido)"
    fi
    return 0
  fi

  if [[ ! -t 0 ]]; then
    STEP_PROFILE="omitido (sin TTY)"
    return 0
  fi

  local profiles_rows
  profiles_rows="$(fetch_orch_agent_profiles || true)"
  if [[ -z "$profiles_rows" ]]; then
    STEP_PROFILE="omitido (endpoint no disponible)"
    return 0
  fi

  local current_row
  current_row="$(fetch_orch_tenant_profile "$TENANT_ID" || true)"
  local current_override=""
  local current_source=""
  local current_applied=""
  local current_reason=""
  if [[ -n "$current_row" ]]; then
    IFS='|' read -r current_override current_source current_applied current_reason <<< "$current_row"
  fi

  echo ""
  echo "🧩 Cartucho por tenant"
  if [[ -n "$current_override" ]]; then
    echo "   Override actual: $current_override"
  else
    echo "   Override actual: (sin override)"
  fi
  if [[ -n "$current_source" ]]; then
    echo "   Resolución actual: source=$current_source applied=$current_applied reason=$current_reason"
  fi

  local profile_ids=()
  local profile_labels=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r pid version status description owner <<< "$line"
    [[ -z "$pid" ]] && continue
    profile_ids+=("$pid")
    profile_labels+=("v=${version:-n/a}, status=${status:-n/a}, owner=${owner:-n/a}")
  done <<< "$profiles_rows"

  if (( ${#profile_ids[@]} == 0 )); then
    STEP_PROFILE="omitido (sin perfiles)"
    return 0
  fi

  echo "[0] Mantener resolución actual"
  echo "[C] Limpiar override (usar cascada)"
  local i
  for ((i=0; i<${#profile_ids[@]}; i++)); do
    printf "[%d] %s (%s)\n" "$((i+1))" "${profile_ids[$i]}" "${profile_labels[$i]}"
  done

  local opt
  read -r -p "📝 Cartucho para tenant [0]: " opt
  [[ -z "$opt" ]] && opt="0"

  if [[ "$opt" == "0" ]]; then
    STEP_PROFILE="sin cambios"
    return 0
  fi

  if [[ "$opt" =~ ^[Cc]$ ]]; then
    if set_or_clear_orch_tenant_profile "$TENANT_ID" "" "true"; then
      STEP_PROFILE="completado (override limpiado)"
      echo "✅ Override limpiado."
    else
      STEP_PROFILE="omitido (clear fallido)"
      echo "⚠️  No se pudo limpiar override (endpoint no disponible)."
    fi
    return 0
  fi

  if ! [[ "$opt" =~ ^[0-9]+$ ]] || (( opt < 1 || opt > ${#profile_ids[@]} )); then
    echo "⚠️  Opción inválida. Se mantiene resolución actual."
    STEP_PROFILE="sin cambios"
    return 0
  fi

  local selected_profile="${profile_ids[$((opt-1))]}"
  if set_or_clear_orch_tenant_profile "$TENANT_ID" "$selected_profile" "false"; then
    STEP_PROFILE="completado (override=$selected_profile)"
    echo "✅ Override aplicado: $selected_profile"
  else
    STEP_PROFILE="omitido (set fallido)"
    echo "⚠️  No se pudo aplicar override '$selected_profile' (endpoint no disponible o perfil inválido)."
  fi
}

resolve_collection() {
  COLLECTION_CLEANUP_REQUIRED="false"
  local collections_json="[]"
  local orch_rows
  orch_rows="$(fetch_orch_collections "$TENANT_ID" || true)"
  if [[ -n "$orch_rows" ]]; then
    collections_json="$(python3 - "$orch_rows" <<'PY'
import json
import sys

rows = []
raw = sys.argv[1]
for line in raw.splitlines():
    if not line.strip():
        continue
    parts = line.split("|")
    if len(parts) < 4:
        continue
    cid, key, name, status = parts[:4]
    rows.append({"id": cid, "collection_key": key, "name": name, "status": status})
print(json.dumps(rows, ensure_ascii=True))
PY
)"
  else
    if [[ -n "${ORCH_ACCESS_TOKEN:-}" ]]; then
      collections_json='[]'
    else
      local collections_status
      collections_status=$(curl -sS -o /tmp/rag_collections.json -w "%{http_code}" --max-time 4 \
        ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
        -H "X-Tenant-ID: $TENANT_ID" \
        "$RAG_URL/api/v1/ingestion/collections?tenant_id=$TENANT_ID" || true)
      if [[ "$collections_status" -ge 200 && "$collections_status" -lt 300 ]]; then
        collections_json=$(cat /tmp/rag_collections.json 2>/dev/null || printf '[]')
      else
        collections_json='[]'
        if [[ "$collections_status" == "404" ]]; then
          echo "⚠️  Endpoint /api/v1/ingestion/collections no disponible (HTTP 404)."
          echo "💡 Probablemente la API está desactualizada; reinicia el servicio RAG."
        fi
      fi
    fi
  fi

  local parsed
  parsed=$(COLLECTIONS_JSON="$collections_json" python3 - <<'PY'
import json, os
raw = os.environ.get("COLLECTIONS_JSON", "[]")
try:
    rows = json.loads(raw)
except Exception:
    rows = []
for r in rows:
    cid = str(r.get("id") or "")
    key = str(r.get("collection_key") or "")
    name = str(r.get("name") or key)
    status = str(r.get("status") or "open")
    print(f"{cid}|{key}|{name}|{status}")
PY
)

  local keys=()
  local names=()
  local statuses=()
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS='|' read -r _cid _key _name _status <<< "$line"
    keys+=("$_key")
    names+=("$_name")
    statuses+=("$_status")
  done <<< "$parsed"

  if [[ -n "$COLLECTION_NAME" && -z "$COLLECTION_ID" ]]; then
    COLLECTION_ID="$(slugify "$COLLECTION_NAME")"
  fi
  if [[ -n "$COLLECTION_ID" && -z "$COLLECTION_NAME" ]]; then
    COLLECTION_NAME="$COLLECTION_ID"
  fi

  if [[ -n "$COLLECTION_ID" ]]; then
    local i
    for ((i=0; i<${#keys[@]}; i++)); do
      if [[ "${keys[$i]}" == "$COLLECTION_ID" ]]; then
        COLLECTION_NAME="${names[$i]}"
        COLLECTION_CLEANUP_REQUIRED="true"
        break
      fi
    done
  fi

  if [[ -z "$COLLECTION_ID" && -z "$COLLECTION_NAME" ]]; then
    echo ""
    echo "📁 Colecciones existentes en tenant:"
    if (( ${#keys[@]} == 0 )); then
      echo "   (sin colecciones previas)"
    else
      local i
      for ((i=0; i<${#keys[@]}; i++)); do
        printf "[%d] %s (%s) [%s]\n" "$((i+1))" "${names[$i]}" "${keys[$i]}" "${statuses[$i]}"
      done
    fi
    echo "[N] Crear nueva colección"

    local opt
    read -r -p "📝 Opción [N]: " opt
    [[ -z "$opt" ]] && opt="N"

    if [[ "$opt" =~ ^[0-9]+$ ]] && (( opt >= 1 && opt <= ${#keys[@]} )); then
      local idx=$((opt-1))
      COLLECTION_ID="${keys[$idx]}"
      COLLECTION_NAME="${names[$idx]}"
      COLLECTION_CLEANUP_REQUIRED="true"
    else
      read -r -p "📁 Nombre de carpeta/colección [default]: " COLLECTION_NAME
      [[ -z "$COLLECTION_NAME" ]] && COLLECTION_NAME="default"
      COLLECTION_ID="$(slugify "$COLLECTION_NAME")"
    fi
  fi
}

collect_files() {
  if [[ -z "${GLOB_PATTERNS+x}" ]]; then
    GLOB_PATTERNS=()
  fi
  if [[ -z "${FILE_QUEUE+x}" ]]; then
    FILE_QUEUE=()
  fi

  local path
  for path in "${GLOB_PATTERNS[@]:-}"; do
    [[ -z "$path" ]] && continue
    local expanded=( $path )
    if (( ${#expanded[@]} == 0 )); then
      continue
    fi
    local f
    for f in "${expanded[@]}"; do
      [[ -f "$f" ]] && FILE_QUEUE+=("$f")
    done
  done

  if (( ${#FILE_QUEUE[@]:-0} == 0 )); then
    echo ""
    echo "📎 No se pasaron archivos. Abriendo Finder para selección múltiple..."

    local selected_files=""
    if [[ "$(uname -s)" == "Darwin" ]] && command -v osascript >/dev/null 2>&1; then
      selected_files=$(osascript <<'APPLESCRIPT'
try
  set chosen to choose file with prompt "Selecciona uno o más archivos para ingesta" with multiple selections allowed
  set output to ""
  repeat with f in chosen
    set output to output & (POSIX path of f) & linefeed
  end repeat
  return output
on error
  return ""
end try
APPLESCRIPT
)
      selected_files="${selected_files//$'\r'/}"
    fi

    if [[ -n "$selected_files" ]]; then
      local f
      while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        [[ -f "$f" ]] && FILE_QUEUE+=("$f")
      done <<< "$selected_files"
    fi

    if (( ${#FILE_QUEUE[@]:-0} == 0 )); then
      echo "❌ No seleccionaste archivos válidos en Finder."
      exit 1
    fi
  fi

  if (( ${#FILE_QUEUE[@]:-0} == 0 )); then
    echo "❌ No hay archivos válidos para encolar."
    exit 1
  fi

  # dedupe
  local deduped=()
  local seen=""
  local f
  for f in "${FILE_QUEUE[@]}"; do
    if [[ "$seen" == *"|$f|"* ]]; then
      continue
    fi
    seen+="|$f|"
    deduped+=("$f")
  done
  FILE_QUEUE=("${deduped[@]}")
}

check_health() {
  if ! curl -fsS --max-time 3 "$RAG_URL/health" >/dev/null 2>&1; then
    echo "❌ RAG API no disponible en $RAG_URL"
    echo "💡 Levántala con ./stack.sh up"
    exit 1
  fi
}

choose_embedding_mode() {
  if [[ -n "$EMBEDDING_MODE" ]]; then
    EMBEDDING_MODE="$(echo "$EMBEDDING_MODE" | tr '[:lower:]' '[:upper:]')"
    if [[ "$EMBEDDING_MODE" == "LOCAL" || "$EMBEDDING_MODE" == "CLOUD" ]]; then
      return 0
    fi
    echo "❌ --embedding-mode inválido: $EMBEDDING_MODE (usa LOCAL o CLOUD)"
    exit 1
  fi

  local has_jina_key="false"
  if [[ -n "${JINA_API_KEY:-}" ]]; then
    has_jina_key="true"
  else
    local env_files=(
      "$SCRIPT_DIR/.env.local"
      "$SCRIPT_DIR/.env"
      "./.env.local"
      "./.env"
    )
    local f
    for f in "${env_files[@]}"; do
      [[ -f "$f" ]] || continue
      if python3 - "$f" <<'PY' >/dev/null 2>&1
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding='utf-8', errors='ignore')
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if line.startswith('JINA_API_KEY='):
        value = line.split('=', 1)[1].strip().strip('"').strip("'")
        if value:
            raise SystemExit(0)
raise SystemExit(1)
PY
      then
        has_jina_key="true"
        break
      fi
    done
  fi

  echo ""
  echo "🧠 Embeddings"
  if [[ "$has_jina_key" == "true" ]]; then
    echo "[1] CLOUD (Jina API)"
    echo "[2] LOCAL"
    local opt
    read -r -p "📝 Modo embeddings [1]: " opt
    [[ -z "$opt" ]] && opt="1"
    if [[ "$opt" == "2" ]]; then
      EMBEDDING_MODE="LOCAL"
    else
      EMBEDDING_MODE="CLOUD"
    fi
  else
    echo "[1] LOCAL"
    echo "[2] CLOUD (requiere JINA_API_KEY)"
    local opt
    read -r -p "📝 Modo embeddings [1]: " opt
    [[ -z "$opt" ]] && opt="1"
    if [[ "$opt" == "2" ]]; then
      EMBEDDING_MODE="CLOUD"
      echo "⚠️  Elegiste CLOUD; asegúrate que la API/worker tengan JINA_API_KEY cargada."
    else
      EMBEDDING_MODE="LOCAL"
    fi
  fi

  echo "✅ Embeddings mode: $EMBEDDING_MODE"
}

cleanup_collection_if_needed() {
  if [[ "$COLLECTION_CLEANUP_REQUIRED" != "true" ]]; then
    return 0
  fi

  echo ""
  echo "🧹 Limpiando colección existente antes de la nueva ingesta..."
  local payload
  payload=$(python3 - "$TENANT_ID" "$COLLECTION_ID" <<'PY'
import json
import sys
tenant_id, collection_key = sys.argv[1:]
print(json.dumps({"tenant_id": tenant_id, "collection_key": collection_key}, ensure_ascii=True))
PY
)

  local status
  status=$(curl -sS -o /tmp/rag_collection_cleanup.json -w "%{http_code}" \
    -X POST "$RAG_URL/api/v1/ingestion/collections/cleanup" \
    ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
    -H "X-Tenant-ID: $TENANT_ID" \
    -H "Content-Type: application/json" \
    -d "$payload")

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    echo "❌ Error limpiando colección '$COLLECTION_ID' (HTTP $status)"
    echo "Respuesta: $(cat /tmp/rag_collection_cleanup.json 2>/dev/null || true)"
    return 1
  fi

  python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/rag_collection_cleanup.json').read_text(encoding='utf-8'))
deleted = data.get('deleted', {}) if isinstance(data, dict) else {}
print("✅ Limpieza aplicada")
print(f"   source_documents: {int(deleted.get('source_documents') or 0)}")
print(f"   regulatory_nodes: {int(deleted.get('regulatory_nodes') or 0)}")
print(f"   ingestion_batches: {int(deleted.get('ingestion_batches') or 0)}")
PY
}

create_batch() {
  local payload
  payload=$(python3 - "$TENANT_ID" "$TENANT_NAME" "$COLLECTION_ID" "$COLLECTION_NAME" "${#FILE_QUEUE[@]}" <<'PY'
import json
import sys

tenant_id, tenant_name, collection_key, collection_name, total = sys.argv[1:]
print(json.dumps({
    "tenant_id": tenant_id,
    "collection_key": collection_key,
    "collection_name": collection_name,
    "total_files": int(total),
    "auto_seal": False,
    "metadata": {"tenant_name": tenant_name},
}))
PY
)

  local status
  status=$(curl -sS -o /tmp/rag_batch_create.json -w "%{http_code}" \
    -X POST "$RAG_URL/api/v1/ingestion/batches" \
    ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
    -H "X-Tenant-ID: $TENANT_ID" \
    -H "Content-Type: application/json" \
    -d "$payload")

  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    echo "❌ Error creando batch (HTTP $status)"
    echo "Respuesta: $(cat /tmp/rag_batch_create.json 2>/dev/null || true)"
    return 1
  fi

  BATCH_ID=$(python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/rag_batch_create.json').read_text(encoding='utf-8'))
print(data.get('batch_id',''))
PY
)
  if [[ -z "${BATCH_ID:-}" ]]; then
    echo "❌ No pude resolver batch_id"
    return 1
  fi
  return 0
}

add_file_to_batch() {
  local file_path="$1"
  local metadata
  metadata=$(python3 - "$TENANT_ID" "$COLLECTION_ID" "$COLLECTION_NAME" "$file_path" "$EMBEDDING_MODE" <<'PY'
import json
import sys
tenant_id, collection_key, collection_name, path, embedding_mode = sys.argv[1:]
print(json.dumps({
    "title": path.rsplit('/',1)[-1],
    "institution_id": tenant_id,
    "collection_key": collection_key,
    "collection_name": collection_name,
    "metadata": {
        "tenant_id": tenant_id,
        "collection_key": collection_key,
        "collection_name": collection_name,
        "embedding_mode": embedding_mode,
    }
}))
PY
)

  local status
  status=$(curl -sS -o /tmp/rag_batch_file.json -w "%{http_code}" \
    -X POST "$RAG_URL/api/v1/ingestion/batches/$BATCH_ID/files" \
    ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
    -H "X-Tenant-ID: $TENANT_ID" \
    -F "file=@$file_path" \
    -F "metadata=$metadata")

  if [[ "$status" -ge 200 && "$status" -lt 300 ]]; then
    return 0
  fi

  echo "❌ Error agregando archivo al batch: $file_path (HTTP $status)"
  echo "Respuesta: $(cat /tmp/rag_batch_file.json 2>/dev/null || true)"
  return 1
}

show_batch_status() {
  local status
  status=$(curl -sS -o /tmp/rag_batch_status.json -w "%{http_code}" \
    ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
    -H "X-Tenant-ID: $TENANT_ID" \
    "$RAG_URL/api/v1/ingestion/batches/$BATCH_ID/status")
  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    echo "⚠️  No pude leer estado del batch (HTTP $status)"
    return 1
  fi

  python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/rag_batch_status.json').read_text(encoding='utf-8'))
batch = data.get('batch', {})
docs = data.get('documents', [])
visual = data.get('visual_accounting', {})
worker = data.get('worker_progress', {})
print("\n📦 Estado batch")
print(f"   ID: {batch.get('id')}")
print(f"   Status: {batch.get('status')}")
print(f"   Progress: {batch.get('completed',0)} ok / {batch.get('failed',0)} fail / {batch.get('total_files',0)} total")
print(f"   Docs: {len(docs)}")
if isinstance(visual, dict):
    loss_events = int(visual.get('loss_events') or 0)
    docs_with_loss = int(visual.get('docs_with_loss') or 0)
    docs_with_visual = int(visual.get('docs_with_visual') or 0)
    attempted = int(visual.get('attempted') or 0)
    stitched = int(visual.get('stitched') or 0)
    copyright_blocks = int(visual.get('parse_failed_copyright') or 0)
    print(
        "   Visual accounting: "
        f"loss_events={loss_events}, docs_with_loss={docs_with_loss}/{docs_with_visual}, "
        f"stitched={stitched}/{attempted}, copyright_blocks={copyright_blocks}"
    )
    refs = visual.get('copyright_refs')
    refs_total = int(visual.get('copyright_refs_total') or 0)
    if isinstance(refs, list) and refs:
        print("   Referencias copyright (muestra):")
        shown = 0
        for item in refs[:8]:
            if not isinstance(item, dict):
                continue
            doc = str(item.get('filename') or item.get('doc_id') or 'doc')
            page = int(item.get('page') or 0)
            chunk = str(item.get('parent_chunk_id') or '-')
            image = str(item.get('image') or '-')
            print(f"     - doc={doc}, page={page}, chunk={chunk}, image={image}")
            shown += 1
        if refs_total > shown:
            print(f"     ... y {refs_total - shown} referencia(s) adicionales")
if isinstance(worker, dict):
    stage_counts = worker.get('stage_counts')
    sample_refs = worker.get('sample_refs')
    if isinstance(stage_counts, dict) and stage_counts:
        ordered = ["INGEST", "PERSIST", "VISUAL", "RAPTOR", "GRAPH", "ERROR", "DONE", "OTHER"]
        parts = []
        for key in ordered:
            value = int(stage_counts.get(key) or 0)
            if value > 0:
                parts.append(f"{key}={value}")
        if not parts:
            parts = [f"{k}={int(v or 0)}" for k, v in stage_counts.items()]
        print("   Worker fases: " + ", ".join(parts))
    if isinstance(sample_refs, list) and sample_refs:
        print("   Worker refs (muestra):")
        shown = 0
        for item in sample_refs[:6]:
            if not isinstance(item, dict):
                continue
            name = str(item.get('filename') or item.get('doc_id') or 'doc')
            stage = str(item.get('stage') or 'OTHER')
            print(f"     - {name}: {stage}")
            shown += 1
PY
}

preflight_visual_estimate() {
  if [[ "$BATCH_SKIP_ESTIMATE" == "true" ]]; then
    return 0
  fi

  local py_exec="${INGEST_CLIENT_PYTHON:-}"
  if [[ -n "$py_exec" && ! -x "$py_exec" ]]; then
    py_exec=""
  fi
  if [[ -z "$py_exec" && -x "$SCRIPT_DIR/venv/bin/python" ]]; then
    py_exec="$SCRIPT_DIR/venv/bin/python"
  fi
  if [[ -z "$py_exec" && -x "./venv/bin/python" ]]; then
    py_exec="./venv/bin/python"
  fi
  if [[ -z "$py_exec" ]]; then
    if command -v python3 >/dev/null 2>&1; then
      py_exec="python3"
    else
      py_exec="python"
    fi
  fi

  local out
  out=$(
    "$py_exec" - "$VISUAL_RATIO" "$VISUAL_MAX_PAGES" "$GEMINI_COST_PER_VISUAL_PAGE_USD" "${FILE_QUEUE[@]}" <<'PY'
import math
import os
import sys

ratio = float(sys.argv[1])
max_visual_pages = int(sys.argv[2])
cost_per_visual = sys.argv[3].strip()
files = sys.argv[4:]

try:
    import fitz  # type: ignore
except Exception:
    print("WARN|NO_PYMUPDF")
    raise SystemExit(0)

total_pages = 0
total_image_pages = 0
total_images = 0
total_upper_budget_pages = 0

for idx, path in enumerate(files, start=1):
    if not path.lower().endswith('.pdf'):
        print(f"FILE|{idx}|{path}|NON_PDF|0|0|0|0|")
        continue
    if not os.path.exists(path):
        print(f"FILE|{idx}|{path}|MISSING|0|0|0|0|")
        continue
    try:
        doc = fitz.open(path)
    except Exception:
        print(f"FILE|{idx}|{path}|OPEN_ERROR|0|0|0|0|")
        continue

    pages = len(doc)
    image_pages = 0
    images = 0
    for i in range(pages):
        p = doc[i]
        imgs = p.get_images(full=True)
        if imgs:
            image_pages += 1
            images += len(imgs)
    doc.close()

    ratio_budget = max(1, int(math.ceil(pages * max(0.0, min(ratio, 1.0)))))
    upper_budget_pages = min(ratio_budget, max(1, max_visual_pages))

    total_pages += pages
    total_image_pages += image_pages
    total_images += images
    total_upper_budget_pages += upper_budget_pages

    file_cost = ""
    if cost_per_visual:
        try:
            file_cost = f"{upper_budget_pages * float(cost_per_visual):.6f}"
        except Exception:
            file_cost = ""

    print(f"FILE|{idx}|{path}|OK|{pages}|{image_pages}|{images}|{upper_budget_pages}|{file_cost}")

est_cost = ""
if cost_per_visual:
    try:
        est_cost = f"{total_upper_budget_pages * float(cost_per_visual):.6f}"
    except Exception:
        est_cost = ""

print(f"TOTAL|{total_pages}|{total_image_pages}|{total_images}|{total_upper_budget_pages}|{est_cost}")
PY
  )

  [[ -z "$out" ]] && return 0

  echo ""
  echo "🔎 Pre-estimación visual (antes de Gemini)"
  local line
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == WARN\|NO_PYMUPDF* ]]; then
      echo "⚠️  No se pudo estimar visualmente (PyMuPDF no disponible en este entorno)."
      return 0
    fi
    if [[ "$line" == FILE\|* ]]; then
      IFS='|' read -r _ idx tag_path tag_status pages image_pages images upper_budget file_cost <<< "$line"
      if [[ "$tag_status" == "OK" ]]; then
        if [[ -n "$file_cost" ]]; then
          printf "   [%s] %s\n" "$idx" "$(basename "$tag_path"): páginas=$pages, páginas_con_imagen=$image_pages, imágenes=$images, presupuesto_visual_max=$upper_budget, costo_estimado_usd=$file_cost"
        else
          printf "   [%s] %s\n" "$idx" "$(basename "$tag_path"): páginas=$pages, páginas_con_imagen=$image_pages, imágenes=$images, presupuesto_visual_max=$upper_budget"
        fi
      else
        printf "   [%s] %s\n" "$idx" "$(basename "$tag_path"): $tag_status"
      fi
    fi
    if [[ "$line" == TOTAL\|* ]]; then
      IFS='|' read -r _ t_pages t_img_pages t_images t_budget t_cost <<< "$line"
      echo "   Total páginas: $t_pages"
      echo "   Total páginas con imagen: $t_img_pages"
      echo "   Total imágenes detectadas: $t_images"
      echo "   Cota superior páginas visuales a procesar: $t_budget"
      if [[ -n "$t_cost" ]]; then
        echo "   Costo estimado (USD, aprox): $t_cost"
      elif [[ -n "$GEMINI_COST_PER_VISUAL_PAGE_USD" ]]; then
        echo "   Costo estimado: no disponible (valor inválido en GEMINI_COST_PER_VISUAL_PAGE_USD)"
      else
        echo "   Costo estimado: define GEMINI_COST_PER_VISUAL_PAGE_USD para calcularlo"
      fi
    fi
  done <<< "$out"

  echo ""
  read -r -p "🎛️  ¿Quieres excluir archivos por costo/imágenes antes de ingestar? [y/N]: " prune_opt
  if [[ "${prune_opt:-N}" =~ ^[Yy]$ ]]; then
    read -r -p "📝 Índices a excluir (ej: 2,4): " prune_indexes
    if [[ -n "${prune_indexes// /}" ]]; then
      local filtered=()
      local token
      local keep
      local idx
      for idx in "${!FILE_QUEUE[@]}"; do
        keep=1
        for token in ${prune_indexes//,/ }; do
          if [[ "$token" =~ ^[0-9]+$ ]] && (( token == idx + 1 )); then
            keep=0
            break
          fi
        done
        if (( keep == 1 )); then
          filtered+=("${FILE_QUEUE[$idx]}")
        fi
      done
      FILE_QUEUE=("${filtered[@]}")
      if (( ${#FILE_QUEUE[@]} == 0 )); then
        echo "❌ Excluiste todos los archivos. Nada que ingestar."
        exit 1
      fi
      echo "✅ Archivos restantes para ingesta: ${#FILE_QUEUE[@]}"
    fi
  fi
}

wait_for_batch_completion_rag_poll() {
  local n=0
  local status=""
  local last_line=""

  echo ""
  echo "⏳ Esperando procesamiento del batch en worker..."
  STEP_WORKER_STATUS="processing (0%)"
  STEP_WORKER_DOCS="0/0"
  STEP_WORKER_QUEUE="processing=0, queued=0, uploaded=0"
  STEP_WORKER_ERRORS="0"
  STEP_WORKER_LOSS="0 eventos"
  STEP_WORKER_PHASE="iniciando"
  STEP_WORKER_REFS="n/a"
  render_checklist

  while (( n < BATCH_POLL_MAX )); do
    local http_code
    http_code=$(curl -sS -o /tmp/rag_batch_status.json -w "%{http_code}" \
      ${AUTH_CURL_ARGS[@]+"${AUTH_CURL_ARGS[@]}"} \
      -H "X-Tenant-ID: $TENANT_ID" \
      "$RAG_URL/api/v1/ingestion/batches/$BATCH_ID/status" || true)

    if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
      echo "⚠️  Poll #$((n+1)): no pude leer estado (HTTP $http_code)"
      sleep "$BATCH_POLL_INTERVAL"
      n=$((n+1))
      continue
    fi

    local line
    line=$(python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/rag_batch_status.json').read_text(encoding='utf-8'))
batch = data.get('batch', {})
docs = data.get('documents', [])
visual = data.get('visual_accounting', {})
worker = data.get('worker_progress', {})
status = str(batch.get('status') or 'unknown')
completed = int(batch.get('completed') or 0)
failed = int(batch.get('failed') or 0)
total = int(batch.get('total_files') or 0)
terminal = 0
processing = 0
queued = 0
success_states = {"success", "processed", "completed", "ready"}
failed_states = {"failed", "error", "dead_letter"}
for d in docs:
    st = str(d.get('status') or '').lower()
    if st in success_states or st in failed_states:
        terminal += 1
    elif st == 'processing':
        processing += 1
    elif st == 'queued':
        queued += 1
loss_events = int(visual.get('loss_events') or 0) if isinstance(visual, dict) else 0
docs_with_loss = int(visual.get('docs_with_loss') or 0) if isinstance(visual, dict) else 0
docs_with_visual = int(visual.get('docs_with_visual') or 0) if isinstance(visual, dict) else 0
copyright_blocks = int(visual.get('parse_failed_copyright') or 0) if isinstance(visual, dict) else 0
stage_detail = "n/a"
ref_detail = "n/a"
if isinstance(worker, dict):
    stage_counts = worker.get('stage_counts')
    if isinstance(stage_counts, dict) and stage_counts:
        ordered = ["INGEST", "PERSIST", "VISUAL", "RAPTOR", "GRAPH", "ERROR", "DONE", "OTHER"]
        parts = []
        for key in ordered:
            val = int(stage_counts.get(key) or 0)
            if val > 0:
                parts.append(f"{key}={val}")
        if not parts:
            parts = [f"{k}={int(v or 0)}" for k, v in stage_counts.items()]
        stage_detail = ",".join(parts)
    sample_refs = worker.get('sample_refs')
    if isinstance(sample_refs, list) and sample_refs:
        ref_parts = []
        for item in sample_refs:
            if not isinstance(item, dict):
                continue
            name = str(item.get('filename') or item.get('doc_id') or 'doc')
            stage = str(item.get('stage') or 'OTHER')
            ref_parts.append(f"{name}:{stage}")
            if len(ref_parts) >= 4:
                break
        if ref_parts:
            ref_detail = ",".join(ref_parts)
stage_detail = stage_detail.replace('|', '/').replace('\n', ' ')
ref_detail = ref_detail.replace('|', '/').replace('\n', ' ')
print(f"{status}|{completed}|{failed}|{total}|{len(docs)}|{terminal}|{processing}|{queued}|{loss_events}|{docs_with_loss}|{docs_with_visual}|{copyright_blocks}|{stage_detail}|{ref_detail}")
PY
)

    status="${line%%|*}"
    local rest="${line#*|}"
    local completed="${rest%%|*}"
    rest="${rest#*|}"
    local failed="${rest%%|*}"
    rest="${rest#*|}"
    local total="${rest%%|*}"
    rest="${rest#*|}"
    local docs_count="${rest%%|*}"
    rest="${rest#*|}"
    local terminal_count="${rest%%|*}"
    rest="${rest#*|}"
    local processing_count="${rest%%|*}"
    rest="${rest#*|}"
    local queued_count="${rest%%|*}"
    rest="${rest#*|}"
    local loss_events="${rest%%|*}"
    rest="${rest#*|}"
    local docs_with_loss="${rest%%|*}"
    rest="${rest#*|}"
    local docs_with_visual="${rest%%|*}"
    rest="${rest#*|}"
    local copyright_blocks="${rest%%|*}"
    rest="${rest#*|}"
    local worker_phase="${rest%%|*}"
    local worker_refs="${rest#*|}"

    local percent=0
    if (( total > 0 )); then
      percent=$(( (terminal_count * 100) / total ))
    fi

    local progress_line
    progress_line="📡 status=$status progress=${percent}% terminal=${terminal_count}/${total} processing=${processing_count} queued=${queued_count} uploaded=${docs_count} fase=${worker_phase}"

    STEP_WORKER_STATUS="$status (${percent}%)"
    STEP_WORKER_DOCS="${terminal_count}/${total}"
    STEP_WORKER_QUEUE="processing=${processing_count}, queued=${queued_count}, uploaded=${docs_count}"
    STEP_WORKER_ERRORS="$failed"
    STEP_WORKER_LOSS="${loss_events} eventos (${docs_with_loss}/${docs_with_visual} docs, copyright=${copyright_blocks})"
    STEP_WORKER_PHASE="$worker_phase"
    STEP_WORKER_REFS="$worker_refs"

    if [[ "$progress_line" != "$last_line" ]]; then
      echo "$progress_line"
      last_line="$progress_line"
      render_checklist
    fi

    case "$status" in
      completed|partial|failed)
        echo "✅ Estado terminal alcanzado: $status"
        STEP_WORKER="completado/terminal"
        return 0
        ;;
    esac

    sleep "$BATCH_POLL_INTERVAL"
    n=$((n+1))
  done

  echo "⚠️  Timeout de espera alcanzado. El batch sigue en progreso."
  echo "💡 Puedes consultar luego: $RAG_URL/api/v1/ingestion/batches/$BATCH_ID/status"
  return 0
}

emit_progress_json() {
  local status="$1"
  local percent="$2"
  local terminal_count="$3"
  local total="$4"
  local processing_count="$5"
  local queued_count="$6"
  local docs_count="$7"
  local failed="$8"
  local loss_events="$9"
  local worker_phase="${10}"
  local eta_seconds="${11}"
  local stalled="${12}"
  if [[ "$PROGRESS_JSON" != "true" ]]; then
    return 0
  fi
  python3 - "$status" "$percent" "$terminal_count" "$total" "$processing_count" "$queued_count" "$docs_count" "$failed" "$loss_events" "$worker_phase" "$eta_seconds" "$stalled" <<'PY'
import json
import sys
status, percent, terminal_count, total, processing_count, queued_count, docs_count, failed, loss_events, phase, eta, stalled = sys.argv[1:]
payload = {
    "status": status,
    "progress_percent": float(percent),
    "terminal_docs": int(terminal_count),
    "total_files": int(total),
    "processing_docs": int(processing_count),
    "queued_docs": int(queued_count),
    "uploaded_docs": int(docs_count),
    "failed_docs": int(failed),
    "loss_events": int(loss_events),
    "dominant_stage": phase,
    "eta_seconds": int(eta),
    "stalled": str(stalled).lower() in {"1", "true", "yes"},
}
print(json.dumps(payload, ensure_ascii=True))
PY
}

apply_progress_snapshot() {
  local status="$1"
  local percent="$2"
  local terminal_count="$3"
  local total="$4"
  local processing_count="$5"
  local queued_count="$6"
  local docs_count="$7"
  local failed="$8"
  local loss_events="$9"
  local docs_with_loss="${10}"
  local docs_with_visual="${11}"
  local copyright_blocks="${12}"
  local worker_phase="${13}"
  local worker_refs="${14}"
  local eta_seconds="${15}"
  local stalled="${16}"
  local progress_line
  progress_line="📡 status=$status progress=${percent}% terminal=${terminal_count}/${total} processing=${processing_count} queued=${queued_count} uploaded=${docs_count} fase=${worker_phase} eta=${eta_seconds}s stalled=${stalled}"

  STEP_WORKER_STATUS="$status (${percent}%)"
  STEP_WORKER_DOCS="${terminal_count}/${total}"
  STEP_WORKER_QUEUE="processing=${processing_count}, queued=${queued_count}, uploaded=${docs_count}, eta=${eta_seconds}s, stalled=${stalled}"
  STEP_WORKER_ERRORS="$failed"
  STEP_WORKER_LOSS="${loss_events} eventos (${docs_with_loss}/${docs_with_visual} docs, copyright=${copyright_blocks})"
  STEP_WORKER_PHASE="$worker_phase"
  STEP_WORKER_REFS="$worker_refs"

  if [[ "${LAST_PROGRESS_LINE:-}" != "$progress_line" ]]; then
    echo "$progress_line"
    LAST_PROGRESS_LINE="$progress_line"
    render_checklist
  fi
  emit_progress_json "$status" "$percent" "$terminal_count" "$total" "$processing_count" "$queued_count" "$docs_count" "$failed" "$loss_events" "$worker_phase" "$eta_seconds" "$stalled"
}

wait_for_batch_completion_orch_poll() {
  build_orch_auth_curl_args
  if [[ ${#ORCH_AUTH_CURL_ARGS[@]} -eq 0 ]]; then
    return 1
  fi
  local n=0
  echo ""
  echo "⏳ Esperando procesamiento (ORCH observability poll)..."
  STEP_WORKER_STATUS="processing (0%)"
  STEP_WORKER_DOCS="0/0"
  STEP_WORKER_QUEUE="processing=0, queued=0, uploaded=0"
  STEP_WORKER_ERRORS="0"
  STEP_WORKER_LOSS="0 eventos"
  STEP_WORKER_PHASE="iniciando"
  STEP_WORKER_REFS="n/a"
  render_checklist

  while (( n < BATCH_POLL_MAX )); do
    local http_code
    http_code=$(curl -sS -o /tmp/orch_batch_progress.json -w "%{http_code}" \
      ${ORCH_AUTH_CURL_ARGS[@]+"${ORCH_AUTH_CURL_ARGS[@]}"} \
      -H "X-Tenant-ID: $TENANT_ID" \
      "$ORCH_URL/api/v1/observability/batches/$BATCH_ID/progress?tenant_id=$TENANT_ID" || true)

    if [[ "$http_code" -lt 200 || "$http_code" -ge 300 ]]; then
      echo "⚠️  Poll ORCH #$((n+1)): no pude leer estado (HTTP $http_code)"
      sleep "$BATCH_POLL_INTERVAL"
      n=$((n+1))
      continue
    fi

    local parsed
    parsed=$(python3 - <<'PY'
import json
from pathlib import Path
data = json.loads(Path('/tmp/orch_batch_progress.json').read_text(encoding='utf-8'))
batch = data.get('batch', {})
obs = data.get('observability', {}) if isinstance(data.get('observability'), dict) else {}
visual = data.get('visual_accounting', {}) if isinstance(data.get('visual_accounting'), dict) else {}
worker = data.get('worker_progress', {}) if isinstance(data.get('worker_progress'), dict) else {}
status = str(batch.get('status') or 'unknown')
total = int(obs.get('total_files') or batch.get('total_files') or 0)
terminal = int(obs.get('terminal_docs') or 0)
processing = int(obs.get('processing_docs') or 0)
queued = int(obs.get('queued_docs') or 0)
docs_count = int(data.get('documents_count') or 0)
percent = float(obs.get('progress_percent') or 0.0)
failed = int(batch.get('failed') or 0)
loss_events = int(visual.get('loss_events') or 0)
docs_with_loss = int(visual.get('docs_with_loss') or 0)
docs_with_visual = int(visual.get('docs_with_visual') or 0)
copyright_blocks = int(visual.get('parse_failed_copyright') or 0)
phase = str(obs.get('dominant_stage') or 'OTHER')
eta_seconds = int(obs.get('eta_seconds') or 0)
stalled = bool(obs.get('stalled', False))
refs = worker.get('sample_refs')
ref_detail = "n/a"
if isinstance(refs, list) and refs:
    parts = []
    for item in refs:
        if not isinstance(item, dict):
            continue
        name = str(item.get('filename') or item.get('doc_id') or 'doc')
        stage = str(item.get('stage') or 'OTHER')
        parts.append(f"{name}:{stage}")
        if len(parts) >= 4:
            break
    if parts:
        ref_detail = ",".join(parts)
ref_detail = ref_detail.replace('|','/').replace('\n',' ')
print(f"{status}|{percent}|{terminal}|{total}|{processing}|{queued}|{docs_count}|{failed}|{loss_events}|{docs_with_loss}|{docs_with_visual}|{copyright_blocks}|{phase}|{ref_detail}|{eta_seconds}|{str(stalled).lower()}")
PY
)

    IFS='|' read -r status percent terminal_count total processing_count queued_count docs_count failed loss_events docs_with_loss docs_with_visual copyright_blocks worker_phase worker_refs eta_seconds stalled <<< "$parsed"
    apply_progress_snapshot "$status" "$percent" "$terminal_count" "$total" "$processing_count" "$queued_count" "$docs_count" "$failed" "$loss_events" "$docs_with_loss" "$docs_with_visual" "$copyright_blocks" "$worker_phase" "$worker_refs" "$eta_seconds" "$stalled"

    case "$status" in
      completed|partial|failed)
        if [[ "$worker_phase" == "GRAPH" ]]; then
           echo "🧠 Refinando Grafo de Conocimiento..."
           STEP_WORKER_STATUS="refining_graph (${percent}%)"
        else
           echo "✅ Estado terminal alcanzado: $status"
           STEP_WORKER="completado/terminal"
           return 0
        fi
        ;;
    esac

    sleep "$BATCH_POLL_INTERVAL"
    n=$((n+1))
  done

  echo "⚠️  Timeout de espera alcanzado (ORCH poll)."
  return 1
}

wait_for_batch_completion_orch_stream() {
  build_orch_auth_curl_args
  if [[ ${#ORCH_AUTH_CURL_ARGS[@]} -eq 0 ]]; then
    return 1
  fi

  echo ""
  echo "⏳ Esperando procesamiento (ORCH observability stream)..."
  STEP_WORKER_STATUS="processing (0%)"
  STEP_WORKER_DOCS="0/0"
  STEP_WORKER_QUEUE="processing=0, queued=0, uploaded=0"
  STEP_WORKER_ERRORS="0"
  STEP_WORKER_LOSS="0 eventos"
  STEP_WORKER_PHASE="iniciando"
  STEP_WORKER_REFS="n/a"
  render_checklist

  local stream_failed=0
  local reached_terminal=0
  local stream_pipe
  stream_pipe="$(mktemp -u "/tmp/orch_stream_${BATCH_ID}.XXXX")"
  if ! mkfifo "$stream_pipe"; then
    echo "⚠️  No se pudo crear FIFO para stream ORCH; fallback a poll."
    return 1
  fi

  python3 - "$ORCH_URL" "$TENANT_ID" "$BATCH_ID" "$ORCH_ACCESS_TOKEN" "$BATCH_POLL_MAX" "$BATCH_POLL_INTERVAL" <<'PY' > "$stream_pipe" &
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

orch_url, tenant_id, batch_id, token, poll_max, poll_interval = sys.argv[1:]
interval_ms = max(500, int(float(poll_interval) * 1000))
params = urllib.parse.urlencode({"tenant_id": tenant_id, "interval_ms": interval_ms})
url = f"{orch_url.rstrip('/')}/api/v1/observability/batches/{batch_id}/stream?{params}"
headers = {"Accept": "text/event-stream", "X-Tenant-ID": tenant_id}
if token:
    headers["Authorization"] = f"Bearer {token}"
request = urllib.request.Request(url, headers=headers)

current_event = "message"
timeout_seconds = max(60, int(poll_max) * int(float(poll_interval)) + 30)
socket.setdefaulttimeout(timeout_seconds)

try:
    with urllib.request.urlopen(request, timeout=timeout_seconds) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line.split(":", 1)[1].strip() or "message"
                continue
            if not line.startswith("data:"):
                continue
            payload_raw = line.split(":", 1)[1].strip()
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except Exception:
                continue
            if current_event == "snapshot":
                progress = payload.get("progress") if isinstance(payload, dict) else {}
                batch = progress.get("batch") if isinstance(progress, dict) and isinstance(progress.get("batch"), dict) else {}
                obs = progress.get("observability") if isinstance(progress, dict) and isinstance(progress.get("observability"), dict) else {}
                visual = progress.get("visual_accounting") if isinstance(progress, dict) and isinstance(progress.get("visual_accounting"), dict) else {}
                worker = progress.get("worker_progress") if isinstance(progress, dict) and isinstance(progress.get("worker_progress"), dict) else {}
                refs = worker.get("sample_refs")
                ref_detail = "n/a"
                if isinstance(refs, list) and refs:
                    parts = []
                    for item in refs:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("filename") or item.get("doc_id") or "doc")
                        stage = str(item.get("stage") or "OTHER")
                        parts.append(f"{name}:{stage}")
                        if len(parts) >= 4:
                            break
                    if parts:
                        ref_detail = ",".join(parts)
                ref_detail = ref_detail.replace("|", "/").replace("\n", " ")
                print(
                    "SNAP|"
                    + "|".join(
                        [
                            str(batch.get("status") or "unknown"),
                            str(float(obs.get("progress_percent") or 0.0)),
                            str(int(obs.get("terminal_docs") or 0)),
                            str(int(obs.get("total_files") or batch.get("total_files") or 0)),
                            str(int(obs.get("processing_docs") or 0)),
                            str(int(obs.get("queued_docs") or 0)),
                            str(int(progress.get("documents_count") or 0)),
                            str(int(batch.get("failed") or 0)),
                            str(int(visual.get("loss_events") or 0)),
                            str(int(visual.get("docs_with_loss") or 0)),
                            str(int(visual.get("docs_with_visual") or 0)),
                            str(int(visual.get("parse_failed_copyright") or 0)),
                            str(obs.get("dominant_stage") or "OTHER"),
                            ref_detail,
                            str(int(obs.get("eta_seconds") or 0)),
                            str(bool(obs.get("stalled", False))).lower(),
                        ]
                    )
                )
                sys.stdout.flush()
            elif current_event == "terminal":
                print(f"TERMINAL|{str(payload.get('status') or 'unknown')}")
                sys.stdout.flush()
                break
except urllib.error.HTTPError as exc:
    print(f"ERR|http|{exc.code}")
except Exception as exc:
    print(f"ERR|stream|{str(exc).replace('|','/')}")
PY
  local stream_pid=$!

  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    if [[ "$line" == ERR\|* ]]; then
      echo "⚠️  Stream ORCH falló: $line"
      stream_failed=1
      break
    fi
    if [[ "$line" == TERMINAL\|* ]]; then
      local status="${line#TERMINAL|}"
      if [[ "$status" == "timeout" ]]; then
        echo "⚠️  Stream ORCH expiró por timeout de sesión; cambiando a poll."
        stream_failed=1
        break
      fi
      
      # Check if we are still in GRAPH phase despite terminal status
      # We need the last snapshot's phase.
      if [[ "$worker_phase" == "GRAPH" ]]; then
         echo "🧠 Refinando Grafo de Conocimiento (Stream)..."
         STEP_WORKER_STATUS="refining_graph (${percent}%)"
         # Continue loop to get more updates
         continue
      else
          echo "✅ Estado terminal alcanzado: $status"
          STEP_WORKER="completado/terminal"
          reached_terminal=1
          break
      fi
    fi
    if [[ "$line" == SNAP\|* ]]; then
      local rest="${line#SNAP|}"
      IFS='|' read -r status percent terminal_count total processing_count queued_count docs_count failed loss_events docs_with_loss docs_with_visual copyright_blocks worker_phase worker_refs eta_seconds stalled <<< "$rest"
      apply_progress_snapshot "$status" "$percent" "$terminal_count" "$total" "$processing_count" "$queued_count" "$docs_count" "$failed" "$loss_events" "$docs_with_loss" "$docs_with_visual" "$copyright_blocks" "$worker_phase" "$worker_refs" "$eta_seconds" "$stalled"
    fi
  done < "$stream_pipe"

  if ! wait "$stream_pid"; then
    stream_failed=1
  fi
  rm -f "$stream_pipe"

  if [[ "$stream_failed" == "1" || "$reached_terminal" == "0" ]]; then
    return 1
  fi
  return 0
}

wait_for_batch_completion() {
  local used_orch="false"
  if [[ -n "${ORCH_URL:-}" && -n "${ORCH_ACCESS_TOKEN:-}" ]]; then
    used_orch="true"
    if [[ "$WATCH_MODE" == "stream" ]]; then
      if wait_for_batch_completion_orch_stream; then
        return 0
      fi
      echo "⚠️  Fallback a ORCH poll..."
      if wait_for_batch_completion_orch_poll; then
        return 0
      fi
    else
      if wait_for_batch_completion_orch_poll; then
        return 0
      fi
    fi
    echo "⚠️  Fallback final a RAG poll..."
  fi
  if [[ "$used_orch" == "false" ]]; then
    echo "ℹ️  ORCH observability no disponible, usando RAG poll."
  fi
  wait_for_batch_completion_rag_poll
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tenant-id)
      TENANT_ID="$2"; shift 2 ;;
    --tenant-name)
      TENANT_NAME="$2"; shift 2 ;;
    --collection-id)
      COLLECTION_ID="$2"; shift 2 ;;
    --collection-name)
      COLLECTION_NAME="$2"; shift 2 ;;
    --agent-profile)
      AGENT_PROFILE_ID="$2"; shift 2 ;;
    --clear-agent-profile)
      CLEAR_AGENT_PROFILE="true"; shift ;;
    --file)
      FILE_QUEUE+=("$2"); shift 2 ;;
    --glob)
      GLOB_PATTERNS+=("$2"); shift 2 ;;
    --embedding-mode)
      EMBEDDING_MODE="$2"; shift 2 ;;
    --no-wait)
      WAIT_FOR_COMPLETION="false"; shift ;;
    --skip-estimate)
      BATCH_SKIP_ESTIMATE="true"; shift ;;
    --watch-mode)
      WATCH_MODE="$2"; shift 2 ;;
    --resume-batch)
      RESUME_BATCH_ID="$2"; shift 2 ;;
    --progress-json)
      PROGRESS_JSON="true"; shift ;;
    --rag-url)
      RAG_URL="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "❌ Argumento no reconocido: $1"
      usage
      exit 1 ;;
  esac
done

validate_watch_mode

check_health
STEP_HEALTH="completado"
render_checklist

select_or_create_tenant
STEP_TENANT="completado"
render_checklist

select_or_assign_profile
if [[ "$STEP_PROFILE" == "pendiente" ]]; then
  STEP_PROFILE="completado"
fi
render_checklist

if [[ -n "$RESUME_BATCH_ID" ]]; then
  STEP_COLLECTION="omitido (resume-batch)"
else
  resolve_collection
  STEP_COLLECTION="completado"
fi
render_checklist

ok=0
failed=0

if [[ -n "$RESUME_BATCH_ID" ]]; then
  BATCH_ID="$RESUME_BATCH_ID"
  STEP_QUEUE="omitido (resume-batch)"
  STEP_PREFLIGHT="omitido (resume-batch)"
  STEP_BATCH_CREATE="omitido (resume-batch=$BATCH_ID)"
  STEP_BATCH_UPLOAD="omitido (resume-batch)"
  STEP_BATCH_SEAL="omitido (resume-batch)"
  render_checklist
  echo "♻️ Reanudando monitoreo del batch: $BATCH_ID"
else
  choose_embedding_mode

  collect_files
  STEP_QUEUE="completado (${#FILE_QUEUE[@]} archivos)"
  render_checklist

  preflight_visual_estimate
  if [[ "$BATCH_SKIP_ESTIMATE" == "true" ]]; then
    STEP_PREFLIGHT="omitido"
  else
    STEP_PREFLIGHT="completado"
  fi
  render_checklist

  echo ""
  echo "🧾 Cola de ingesta preparada"
  echo "   Tenant: ${TENANT_NAME:-$TENANT_ID} ($TENANT_ID)"
  echo "   Carpeta: $COLLECTION_NAME ($COLLECTION_ID)"
  echo "   Embeddings: $EMBEDDING_MODE"
  if [[ "$COLLECTION_CLEANUP_REQUIRED" == "true" ]]; then
    echo "   Modo colección existente: se limpiará antes de iniciar"
  fi
  echo "   Archivos: ${#FILE_QUEUE[@]}"

  for i in "${!FILE_QUEUE[@]}"; do
    printf "   [%d] %s\n" "$((i+1))" "${FILE_QUEUE[$i]}"
  done

  read -r -p "\n¿Enviar cola a ingestión? [Y/n]: " confirm
  if [[ "${confirm:-Y}" =~ ^[Nn]$ ]]; then
    echo "❌ Operación cancelada."
    exit 0
  fi

  if ! cleanup_collection_if_needed; then
    exit 1
  fi

  if ! create_batch; then
    STEP_BATCH_CREATE="fallido"
    render_checklist
    exit 1
  fi
  STEP_BATCH_CREATE="completado (batch_id=$BATCH_ID)"
  render_checklist

  echo "🚀 Batch creado: $BATCH_ID"
  for file_path in "${FILE_QUEUE[@]}"; do
    echo "➡️  Subiendo al batch: $file_path"
    if add_file_to_batch "$file_path"; then
      ok=$((ok+1))
      echo "✅ Agregado al batch"
    else
      failed=$((failed+1))
    fi
  done

  if (( failed == 0 )); then
    STEP_BATCH_UPLOAD="completado (${ok}/${#FILE_QUEUE[@]} cargados)"
  else
    STEP_BATCH_UPLOAD="parcial (${ok} ok / ${failed} fail)"
  fi
  render_checklist

  if (( failed == 0 )); then
    echo "🔓 Sellado automático omitido (colecciones reescribibles activadas)."
    STEP_BATCH_SEAL="omitido (overwritable)"
  else
    echo "⚠️  Batch no sellado por errores de carga."
    echo "💡 Corrige los archivos fallidos y reintenta con el mismo batch o crea uno nuevo."
    STEP_BATCH_SEAL="omitido por fallos"
  fi
  render_checklist
fi

show_batch_status || true
if [[ "$WAIT_FOR_COMPLETION" == "true" && "$failed" -eq 0 ]]; then
  if wait_for_batch_completion; then
    STEP_WORKER="completado/terminal"
  else
    STEP_WORKER="pendiente (timeout/fallback)"
  fi
  show_batch_status || true
elif [[ "$WAIT_FOR_COMPLETION" == "false" && "$failed" -eq 0 ]]; then
  STEP_WORKER="pendiente (no-wait)"
  STEP_WORKER_STATUS="processing (no-wait)"
  STEP_WORKER_DOCS="pendiente"
  STEP_WORKER_QUEUE="seguimiento manual"
  STEP_WORKER_ERRORS="0"
  STEP_WORKER_LOSS="0 eventos (no-wait)"
  STEP_WORKER_PHASE="processing (no-wait)"
  STEP_WORKER_REFS="consulta /batches/$BATCH_ID/status"
else
  STEP_WORKER="pendiente (cargas con error)"
  STEP_WORKER_STATUS="no iniciado"
  STEP_WORKER_DOCS="0/${#FILE_QUEUE[@]}"
  STEP_WORKER_QUEUE="cargas fallidas"
  STEP_WORKER_ERRORS="$failed"
  STEP_WORKER_LOSS="n/a (batch no iniciado)"
  STEP_WORKER_PHASE="no iniciado"
  STEP_WORKER_REFS="n/a"
fi
render_checklist

echo ""
echo "📊 Resumen"
if [[ -n "$RESUME_BATCH_ID" ]]; then
  echo "   Total: n/a (resume-batch)"
else
  echo "   Total: ${#FILE_QUEUE[@]}"
fi
echo "   OK:    $ok"
echo "   Fail:  $failed"
echo "   Pérdida visual potencial: $STEP_WORKER_LOSS"

if (( failed > 0 )); then
  exit 1
fi
