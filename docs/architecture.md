# Architecture

## Servicio único

Este repositorio contiene solo el **Q/A Orchestrator**.

## Flujo principal

1. Cliente envía pregunta al endpoint del orquestador.
2. El orquestador clasifica intención y aplica políticas de alcance/validación.
3. El orquestador consulta un RAG engine externo vía HTTP (`RAG_ENGINE_URL`).
4. El orquestador genera y valida respuesta final antes de responder.

## Componentes

- API: `orchestrator/runtime/orchestrator_main.py`
- Router: `orchestrator/runtime/orchestrator_api/v1/routers/knowledge.py`
- Caso de uso: `orchestrator/runtime/qa_orchestrator/application.py`
- Políticas: `orchestrator/runtime/qa_orchestrator/policies.py`
- Adaptadores HTTP al engine: `orchestrator/runtime/qa_orchestrator/http_adapters.py`

## Límites explícitos

- No hay endpoints locales de ingestión.
- No hay workers locales de procesamiento de documentos.
- No hay embeddings, chunking ni retrieval core local en este repo.
