# Q/A Orchestrator (Orchestrator-Only)

## Espanol

Repositorio del servicio **Q/A Orchestrator** desacoplado del runtime interno de RAG engine.

Flujo soportado:

`cliente -> orchestrator -> RAG engine externo (HTTP) -> respuesta orquestada`

### Inicio rapido

```bash
cp .env.example .env.local
./bootstrap.sh
./stack.sh up
./ing.sh --help
```

Health check:

```bash
curl http://localhost:8001/health
```

Chat CLI:

```bash
./chat.sh
```

### Variables clave

- `RAG_ENGINE_LOCAL_URL` (default: `http://localhost:8000`)
- `RAG_ENGINE_DOCKER_URL` (default recomendado en compose: `http://api:8000`)
- `RAG_ENGINE_HEALTH_PATH` (default: `/health`)
- `RAG_ENGINE_PROBE_TIMEOUT_MS` (default: `300`)
- `RAG_ENGINE_BACKEND_TTL_SECONDS` (default: `20`)
- `RAG_ENGINE_FORCE_BACKEND` (opcional: `local|docker`)
- `OPENAI_API_KEY` (opcional para generacion de respuesta)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)

### Estructura

- `app/api/server.py`: API FastAPI del orquestador
- `app/api/v1/routes/knowledge.py`: endpoint principal `/api/v1/knowledge/answer`
- `app/agent/*`: politicas, validacion y caso de uso
- `chat_cli.py`: CLI HTTP para conversacion manual
- `tests/unit/*qa_orchestrator*`: pruebas unitarias del orquestador

### Comandos de desarrollo

```bash
./stack.sh up
./stack.sh logs
./stack.sh down
./ing.sh --help
venv/bin/python -m pytest tests/unit -q
```

### Cliente de ingesta (HTTP)

- Entrada recomendada para equipo: `./ing.sh`
- Ubicacion canonica: `tools/ingestion-client/`
- Si `RAG_URL` no está definido, `./ing.sh` resuelve backend con la misma política híbrida (`RAG_ENGINE_LOCAL_URL` -> fallback `RAG_ENGINE_DOCKER_URL`, con `RAG_ENGINE_FORCE_BACKEND`).

Ejemplos:

```bash
# Interactivo
./ing.sh --file ./docs/manual.pdf

# No interactivo
RAG_URL=http://localhost:8000 \
TENANT_ID=<tenant_uuid> \
COLLECTION_ID=iso-9001 \
./ing.sh --file ./docs/manual.pdf --embedding-mode CLOUD --no-wait
```

### Modo hibrido local -> docker

- Regla: si local responde health, usa local; si no, usa docker.
- Decision cacheada por `RAG_ENGINE_BACKEND_TTL_SECONDS`.
- Si falla conexion al backend elegido, hace un retry unico al alternativo (salvo `RAG_ENGINE_FORCE_BACKEND`).

Ejemplos:

```bash
# Dev local puro
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://localhost:8000
RAG_ENGINE_FORCE_BACKEND=local

# Dev con fallback a docker (auto)
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://api:8000

# Forzar docker
RAG_ENGINE_FORCE_BACKEND=docker
RAG_ENGINE_DOCKER_URL=http://api:8000
```

### Contrato con RAG externo

- Retrieval de chunks: `POST {resolved_backend}/api/v1/retrieval/chunks`
- Retrieval de summaries: `POST {resolved_backend}/api/v1/retrieval/summaries`
- Health recomendado del engine: `GET {resolved_backend}/health`

### Validacion rapida (real)

```bash
# backend + orchestrator
curl -f http://localhost:8000/health
curl -f http://localhost:8001/health

# tenant de ejemplo desde backend (si no tienes uno a mano)
python3 - <<'PY'
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:8000/api/v1/ingestion/documents?limit=100', timeout=3).read().decode())
items = data if isinstance(data, list) else data.get('items', [])
seen = []
for item in items:
    if isinstance(item, dict) and item.get('institution_id'):
        t = str(item['institution_id'])
        if t not in seen:
            seen.append(t)
print(seen[0] if seen else '')
PY

# pregunta e2e al orquestador
curl -X POST http://localhost:8001/api/v1/knowledge/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Que exige ISO 9001 en 7.5.3?","tenant_id":"<TENANT_ID>"}'
```

## English

This repository contains the **Q/A Orchestrator** service, decoupled from any internal RAG engine runtime.

Supported flow:

`client -> orchestrator -> external RAG engine (HTTP) -> orchestrated response`

### Quickstart

```bash
cp .env.example .env.local
./bootstrap.sh
./stack.sh up
./ing.sh --help
```

Health check:

```bash
curl http://localhost:8001/health
```

Chat CLI:

```bash
./chat.sh
```

### Key variables

- `RAG_ENGINE_LOCAL_URL` (default: `http://localhost:8000`)
- `RAG_ENGINE_DOCKER_URL` (recommended compose default: `http://api:8000`)
- `RAG_ENGINE_HEALTH_PATH` (default: `/health`)
- `RAG_ENGINE_PROBE_TIMEOUT_MS` (default: `300`)
- `RAG_ENGINE_BACKEND_TTL_SECONDS` (default: `20`)
- `RAG_ENGINE_FORCE_BACKEND` (optional: `local|docker`)
- `OPENAI_API_KEY` (optional, for answer generation)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)

### Project layout

- `app/api/server.py`: FastAPI server
- `app/api/v1/routes/knowledge.py`: main endpoint `/api/v1/knowledge/answer`
- `app/agent/*`: policies, validation, and use case logic
- `chat_cli.py`: HTTP chat CLI entrypoint
- `tests/unit/*qa_orchestrator*`: orchestrator unit tests

### Development commands

```bash
./stack.sh up
./stack.sh logs
./stack.sh down
./ing.sh --help
venv/bin/python -m pytest tests/unit -q
```

### Ingestion Client (HTTP)

- Team entrypoint: `./ing.sh`
- Canonical location: `tools/ingestion-client/`
- If `RAG_URL` is not defined, `./ing.sh` resolves backend with the same hybrid policy (`RAG_ENGINE_LOCAL_URL` -> fallback `RAG_ENGINE_DOCKER_URL`, with `RAG_ENGINE_FORCE_BACKEND`).

Examples:

```bash
# Interactive
./ing.sh --file ./docs/manual.pdf

# Non-interactive
RAG_URL=http://localhost:8000 \
TENANT_ID=<tenant_uuid> \
COLLECTION_ID=iso-9001 \
./ing.sh --file ./docs/manual.pdf --embedding-mode CLOUD --no-wait
```

### Hybrid mode local -> docker

- Rule: if local health is OK, local is used; otherwise docker is used.
- Selection is cached for `RAG_ENGINE_BACKEND_TTL_SECONDS`.
- On connection failure to the selected backend, a single retry is made against the alternate backend (unless forced).

Examples:

```bash
# Pure local dev
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://localhost:8000
RAG_ENGINE_FORCE_BACKEND=local

# Dev with docker fallback (auto)
RAG_ENGINE_LOCAL_URL=http://localhost:8000
RAG_ENGINE_DOCKER_URL=http://api:8000

# Force docker
RAG_ENGINE_FORCE_BACKEND=docker
RAG_ENGINE_DOCKER_URL=http://api:8000
```

### External RAG contract

- Chunk retrieval: `POST {resolved_backend}/api/v1/retrieval/chunks`
- Summary retrieval: `POST {resolved_backend}/api/v1/retrieval/summaries`
- Recommended engine health check: `GET {resolved_backend}/health`

### Quick real-world verification

```bash
# backend + orchestrator
curl -f http://localhost:8000/health
curl -f http://localhost:8001/health

# sample tenant discovery from backend
python3 - <<'PY'
import json, urllib.request
data = json.loads(urllib.request.urlopen('http://localhost:8000/api/v1/ingestion/documents?limit=100', timeout=3).read().decode())
items = data if isinstance(data, list) else data.get('items', [])
seen = []
for item in items:
    if isinstance(item, dict) and item.get('institution_id'):
        t = str(item['institution_id'])
        if t not in seen:
            seen.append(t)
print(seen[0] if seen else '')
PY

# e2e question to orchestrator
curl -X POST http://localhost:8001/api/v1/knowledge/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"What does ISO 9001 require in 7.5.3?","tenant_id":"<TENANT_ID>"}'
```
