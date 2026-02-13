# Q/A Orchestrator (Orchestrator-Only)

Repositorio del servicio **Q/A Orchestrator** desacoplado del runtime interno de RAG engine.

El flujo soportado es:

`cliente -> orchestrator -> RAG engine externo (HTTP) -> respuesta orquestada`

## Quickstart

```bash
cp .env.example .env.local
./bootstrap.sh
./start_api.sh
```

Health check:

```bash
curl http://localhost:8001/health
```

Chat CLI:

```bash
./chat.sh
```

## Variables clave

- `RAG_ENGINE_URL` (default: `http://localhost:8000`)
- `OPENAI_API_KEY` (opcional para generación de respuesta)
- `OPENAI_MODEL` (default: `gpt-4o-mini`)

## Estructura

- `orchestrator/runtime/orchestrator_main.py`: API FastAPI del orquestador
- `orchestrator/runtime/orchestrator_api/v1/routers/knowledge.py`: endpoint principal `/api/v1/knowledge/answer`
- `orchestrator/runtime/qa_orchestrator/*`: políticas, validación y caso de uso
- `tests/unit/*qa_orchestrator*`: pruebas unitarias del orquestador

## Comandos de desarrollo

```bash
./stack.sh up
./stack.sh logs
./stack.sh down
pytest tests/unit -q
```
