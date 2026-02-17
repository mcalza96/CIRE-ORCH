# Getting Started

## Espanol

### Prerrequisitos

- Python 3.11+
- `bash`
- Un RAG engine externo accesible por HTTP

### 1) Bootstrap

```bash
cp .env.example .env.local
./bootstrap.sh
```

### 2) Configurar conexion al engine

Define en `.env.local`:

```bash
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://api:8000
RAG_ENGINE_HEALTH_PATH=/health
RAG_ENGINE_PROBE_TIMEOUT_MS=300
RAG_ENGINE_BACKEND_TTL_SECONDS=20
```

Opcional para forzar backend:

```bash
RAG_ENGINE_FORCE_BACKEND=local
# o
RAG_ENGINE_FORCE_BACKEND=docker
```

### 3) Levantar orquestador

```bash
./stack.sh up
```

Health check:

```bash
curl http://localhost:8001/health
```

### 4) Probar flujo end-to-end

```bash
./chat.sh
```

Opcional: prueba de ingesta HTTP (cliente portable):

```bash
./ing.sh --help
./ing.sh --file ./docs/manual.pdf
./ing.sh --tenant-id <TENANT_ID> --agent-profile iso_auditor --collection-name normas --file ./docs/manual.pdf
./ing.sh --tenant-id <TENANT_ID> --clear-agent-profile --collection-name normas --file ./docs/manual.pdf
```

Nota: `./ing.sh` usa `RAG_URL` si está definido; si no, aplica resolución híbrida con `RAG_ENGINE_*`.
También permite asignar override dev de cartucho por tenant desde la interfaz (`--agent-profile` / `--clear-agent-profile`).

o via HTTP:

```bash
curl -X POST http://localhost:8001/api/v1/knowledge/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Que exige ISO 9001 en 7.5.3?","tenant_id":"tenant-1"}'
```

Si no tienes `tenant_id`, puedes derivar uno desde documentos del backend:

```bash
python3 - <<'PY'
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:8000/api/v1/ingestion/documents?limit=100', timeout=3).read().decode())
items = data if isinstance(data, list) else data.get('items', [])
for item in items:
    if isinstance(item, dict) and item.get('institution_id'):
        print(item['institution_id'])
        break
PY
```

### 5) Apagar servicios locales

```bash
./stack.sh down
```

## English

### Prerequisites

- Python 3.11+
- `bash`
- An external RAG engine reachable over HTTP

### 1) Bootstrap

```bash
cp .env.example .env.local
./bootstrap.sh
```

### 2) Configure engine connectivity

Set in `.env.local`:

```bash
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://api:8000
RAG_ENGINE_HEALTH_PATH=/health
RAG_ENGINE_PROBE_TIMEOUT_MS=300
RAG_ENGINE_BACKEND_TTL_SECONDS=20
```

Optional backend override:

```bash
RAG_ENGINE_FORCE_BACKEND=local
# or
RAG_ENGINE_FORCE_BACKEND=docker
```

### 3) Start the orchestrator

```bash
./stack.sh up
```

Health check:

```bash
curl http://localhost:8001/health
```

### 4) Validate the end-to-end flow

```bash
./chat.sh
```

Optional: HTTP ingestion smoke test (portable client):

```bash
./ing.sh --help
./ing.sh --file ./docs/manual.pdf
./ing.sh --tenant-id <TENANT_ID> --agent-profile iso_auditor --collection-name norms --file ./docs/manual.pdf
./ing.sh --tenant-id <TENANT_ID> --clear-agent-profile --collection-name norms --file ./docs/manual.pdf
```

Note: `./ing.sh` uses `RAG_URL` when defined; otherwise it applies hybrid resolution from `RAG_ENGINE_*`.
It also supports per-tenant dev cartridge overrides from the CLI (`--agent-profile` / `--clear-agent-profile`).

or via HTTP:

```bash
curl -X POST http://localhost:8001/api/v1/knowledge/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What does ISO 9001 require in 7.5.3?","tenant_id":"tenant-1"}'
```

If you do not have a `tenant_id` handy, derive one from backend documents:

```bash
python3 - <<'PY'
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:8000/api/v1/ingestion/documents?limit=100', timeout=3).read().decode())
items = data if isinstance(data, list) else data.get('items', [])
for item in items:
    if isinstance(item, dict) and item.get('institution_id'):
        print(item['institution_id'])
        break
PY
```

### 5) Stop local services

```bash
./stack.sh down
```
