# Operations

## Runtime local

- `./stack.sh up` inicia solo Orchestrator API (`:8001`).
- `./stack.sh logs` muestra logs del orquestador.
- `./stack.sh down` detiene el servicio.

## Health checks

- Orchestrator: `GET /health`
- Dependencia externa recomendada: `GET {RAG_ENGINE_URL}/health`

## Checklist operativo

- `RAG_ENGINE_URL` configurado y alcanzable.
- API del orquestador activa en `http://localhost:8001/health`.
- Logs sin errores de timeout/5xx contra el engine externo.

## Incidentes comunes

- `502/500` al responder: revisar conectividad a `RAG_ENGINE_URL`.
- Respuestas sin evidencia: validar payload de retrieval devuelto por el engine.
