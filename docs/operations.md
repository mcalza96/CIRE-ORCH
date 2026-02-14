# Operations

## Runtime local

- `./stack.sh up` inicia solo Orchestrator API (`:8001`).
- `./stack.sh logs` muestra logs del orquestador.
- `./stack.sh down` detiene el servicio.
- Arranque directo alternativo: `./start_api.sh`.
- Log local principal: `.logs/orchestrator-api.log`.

## Ingestion client (tooling)

- Entrada de equipo: `./ing.sh`
- Dispatcher técnico unificado: `python orch_cli.py ingest ...`
- Ruta canonica: `tools/ingestion-client/`
- Config principal: `RAG_URL` (si no se define, `ingest` usa `RAG_ENGINE_LOCAL_URL`/`RAG_ENGINE_DOCKER_URL`)
- Si `TENANT_ID` no está definido, `ingest` intenta resolver tenants autorizados desde ORCH y preseleccionar uno.

Comandos utiles:

- `./ing.sh --help`
- `./ing.sh --file ./docs/manual.pdf`
- `TENANT_ID=<uuid> COLLECTION_ID=<id> ./ing.sh --file ./docs/manual.pdf --no-wait`
- `RAG_ENGINE_FORCE_BACKEND=docker ./ing.sh --file ./docs/manual.pdf --no-wait`

## Health checks

- Orchestrator: `GET /health`
- Dependencia externa recomendada: `GET {RAG_ENGINE_LOCAL_URL}/health` y/o `GET {RAG_ENGINE_DOCKER_URL}/health`

## Checklist operativo

- `RAG_ENGINE_LOCAL_URL` y `RAG_ENGINE_DOCKER_URL` configurados.
- `RAG_SERVICE_SECRET` configurado y sincronizado con el servicio RAG.
- API del orquestador activa en `http://localhost:8001/health`.
- Logs sin errores de timeout/5xx contra el engine externo.

## Smoke test E2E

- `curl -f http://localhost:8000/health`
- `curl -f http://localhost:8001/health`
- `curl -X POST http://localhost:8001/api/v1/knowledge/answer -H "Content-Type: application/json" -H "Authorization: Bearer <ACCESS_TOKEN>" -d '{"query":"Que exige ISO 9001 en 7.5.3?","tenant_id":"<TENANT_ID>"}'`
- `./ing.sh --help`

## Modo hibrido (local -> docker)

- Seleccion automatica: local si health=200, en caso contrario docker.
- Cache de backend por `RAG_ENGINE_BACKEND_TTL_SECONDS`.
- Retry unico al backend alternativo ante error de conexion (si no hay `RAG_ENGINE_FORCE_BACKEND`).
- Eventos relevantes en logs: `rag_backend_selected`, `rag_backend_probe_failed`, `rag_backend_fallback_retry`.

## Incidentes comunes

- `502/500` al responder: revisar conectividad a `RAG_ENGINE_LOCAL_URL` y `RAG_ENGINE_DOCKER_URL`.
- Respuestas sin evidencia: validar payload de retrieval devuelto por el engine.
- `ing.sh` falla por API caída: validar `curl {RAG_URL}/health`.
- `ing.sh` con tenant/collection inválidos: revisar `TENANT_ID`, `COLLECTION_ID` y respuesta HTTP del endpoint de ingestión.
