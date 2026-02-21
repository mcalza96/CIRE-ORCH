# Architecture

## Servicio único

Este repositorio contiene solo el **Q/A Orchestrator**.

## Flujo principal

1. Cliente envía pregunta al endpoint del orquestador.
2. El orquestador clasifica intención y aplica políticas de alcance/validación.
3. El orquestador consulta un RAG engine externo vía HTTP (selector híbrido local/docker).
4. El orquestador genera y valida respuesta final antes de responder.

## Componentes

- API: `app/api/server.py`
- Router: `app/api/v1/routes/knowledge.py`
- Caso de uso: `app/agent/application.py`
- Políticas: `app/agent/policies.py`
- Adaptadores HTTP al engine: `app/agent/http_adapters.py`
- Configuración: `app/core/config.py`
- Métricas de alcance: `app/core/scope_metrics.py`
- CLI dispatcher: `orch_cli.py`
- CLI runtime: `app/orch_cli_runtime.py`
- Chat runtime: `app/chat_cli_runtime.py`

## Límites explícitos

- No hay endpoints locales de ingestión.
- No hay workers locales de procesamiento de documentos.
- No hay embeddings, chunking ni retrieval core local en este repo.
