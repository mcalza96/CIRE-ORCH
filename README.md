# Q/A Orchestrator (Orchestrator-Only)

## Espanol

Repositorio del servicio **Q/A Orchestrator** desacoplado del runtime interno de RAG engine.

Flujo soportado:

`cliente -> orchestrator -> RAG engine externo (HTTP) -> respuesta orquestada`

### Documentación E2E
- **Flujo de Orquestación y LangGraph: [docs/e2e.md](docs/e2e.md)**

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
- `RAG_SERVICE_SECRET` (requerido para auth S2S contra RAG en despliegues)
- `SUPABASE_URL` (requerido para login interactivo en `chat.sh`)
- `SUPABASE_ANON_KEY` (requerido para login interactivo en `chat.sh`)
- `SUPABASE_SERVICE_ROLE_KEY` (requerido para resolver membresias tenant en ORCH)
- `SUPABASE_MEMBERSHIPS_TABLE` (default: `tenant_memberships`, ej: `memberships`)
- `GROQ_API_KEY` (opcional para generacion de respuesta)
- `GROQ_MODEL_CHAT` (default: `openai/gpt-oss-20b`)
- `GEMINI_API_KEY` (opcional, reservado para integraciones futuras)

`chat.sh` e `ing.sh` son wrappers livianos. La orquestación CLI vive en Python:

- entrypoint técnico unificado: `orch_cli.py` (`chat` / `ingest`)
- runtime: `app/orch_cli_runtime.py`

- sesión en `~/.cire/orch/session.json`
- refresh automático
- login interactivo como fallback

Flags nuevos del chat:

- `--doctor`: ejecuta diagnóstico de auth/tenant/collection/retrieval y sale
- `--non-interactive`: falla si necesita prompts interactivos

### Estructura

- `app/api/server.py`: API FastAPI del orquestador
- `app/api/v1/routes/knowledge.py`: endpoint principal `/api/v1/knowledge/answer`
- `app/agent/*`: politicas, validacion y caso de uso
- `orch_cli.py`: dispatcher CLI unificado (`chat` / `ingest`)
- `app/chat_cli_runtime.py`: runtime del subcomando `chat`
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
- Si `RAG_URL` no está definido, el subcomando `ingest` resuelve backend con la política híbrida (`RAG_ENGINE_LOCAL_URL` -> fallback `RAG_ENGINE_DOCKER_URL`, con `RAG_ENGINE_FORCE_BACKEND`).
- `./ing.sh` y `./chat.sh` delegan al mismo dispatcher Python (`orch_cli.py`).
- Si no se provee `TENANT_ID`, `ingest` intenta descubrir tenants autorizados en ORCH y preseleccionar uno antes de entrar al flujo interactivo de ingesta.

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

- Retrieval de chunks (debug): `POST {resolved_backend}/api/v1/debug/retrieval/chunks`
- Retrieval de summaries (debug): `POST {resolved_backend}/api/v1/debug/retrieval/summaries`
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
- `RAG_SERVICE_SECRET` (required for S2S auth against RAG in deployed environments)
- `SUPABASE_URL` (required for interactive login in `chat.sh`)
- `SUPABASE_ANON_KEY` (required for interactive login in `chat.sh`)
- `SUPABASE_SERVICE_ROLE_KEY` (required for tenant membership lookup in ORCH)
- `SUPABASE_MEMBERSHIPS_TABLE` (default: `tenant_memberships`, e.g. `memberships`)
- `GROQ_API_KEY` (optional, for answer generation)
- `GROQ_MODEL_CHAT` (default: `openai/gpt-oss-20b`)
- `GEMINI_API_KEY` (optional, reserved for future integrations)

`chat.sh` and `ing.sh` are thin wrappers. CLI orchestration is handled in Python:

- unified technical entrypoint: `orch_cli.py` (`chat` / `ingest`)
- runtime: `app/orch_cli_runtime.py`

- session file at `~/.cire/orch/session.json`
- automatic refresh
- interactive login fallback

New chat flags:

- `--doctor`: runs auth/tenant/collection/retrieval diagnostics and exits
- `--non-interactive`: fails if user prompts are required

### Documentación Central
- **[E2E Flow & LangGraph](docs/e2e.md)**: Detalles sobre el motor de orquestación.
- **[Documentation Hub](docs/README.md)**: Índice de guías técnicas.

### Verificación E2E
Para validar el sistema completo de forma automatizada:
```bash
venv/bin/python -m pytest tests/test_orch_e2e_flow.py
```
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
- If `RAG_URL` is not defined, the `ingest` subcommand resolves backend using the same hybrid policy (`RAG_ENGINE_LOCAL_URL` -> fallback `RAG_ENGINE_DOCKER_URL`, with `RAG_ENGINE_FORCE_BACKEND`).
- `./ing.sh` and `./chat.sh` delegate to the same Python dispatcher (`orch_cli.py`).
- If `TENANT_ID` is not provided, `ingest` tries to discover authorized tenants from ORCH and preselect one before entering interactive ingestion setup.

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

- Chunk retrieval (debug): `POST {resolved_backend}/api/v1/debug/retrieval/chunks`
- Summary retrieval (debug): `POST {resolved_backend}/api/v1/debug/retrieval/summaries`
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
