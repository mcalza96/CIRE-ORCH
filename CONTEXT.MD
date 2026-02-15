# CIRE PLATFORM: CONTEXTO Y ARQUITECTURA MAESTRA

> **ADVERTENCIA PARA AGENTES DE IA:**
> Este no es un proyecto de juguete. Es una plataforma industrial distribuida.
> Lee este documento ANTES de sugerir cambios. Ignorar la separación de responsabilidades (SoC) romperá el sistema.

## 1. Visión Global (The Big Picture)

CIRE no es una aplicación monolítica. Es una **Línea de Ensamblaje de Conocimiento** diseñada para escalar horizontalmente en múltiples verticales de negocio (Legal, Educación, Auditoría).

El sistema se divide estrictamente en dos dominios soberanos:

### A. CIRE-RAG (El Motor de Verdad)
* **Rol:** El "Cerebro Profundo". Es una infraestructura agnóstica de almacenamiento, indexación y recuperación.
* **Filosofía:** "Potencia bruta, Cero Opinión".
* **Responsabilidad:** Ingestar documentos, mantener vectores, ejecutar búsquedas híbridas complejas y garantizar la seguridad de datos (RLS).
* **Lo que NUNCA hace:** No sabe de "mallas curriculares", "multas de tráfico" o "usuarios finales". Solo sabe de `DocumentChunks`, `Vectors` y `Tenants`.
* **Stack:** Python (FastAPI), Supabase (pgvector + RPCs pesados).

### B. CIRE-ORCH (El Cerebro Ejecutivo)
* **Rol:** El "Planificador y Estratega". Es la capa de lógica de negocio que se adapta al cliente.
* **Filosofía:** "Contexto y Acción".
* **Responsabilidad:** Entender la intención del usuario, descomponer tareas complejas, orquestar llamadas al RAG, gestionar la memoria de conversación y formatear la salida final.
* **Lo que NUNCA hace:** No almacena documentos originales. No gestiona índices vectoriales.
* **Stack:** Python (FastAPI), LangGraph/Agents, SDK interno del RAG.

---

## 2. Reglas de Oro para el Desarrollo (The Commandments)

### Regla #1: La Base de Datos es la Ley (Supabase/Postgres)
* En `cire-rag`, la lógica crítica (búsqueda, seguridad, mutaciones atómicas) VIVE en la base de datos (funciones RPC de PostgreSQL).
* **Prohibido:** Traer 10,000 filas a Python para filtrarlas. Úsalo solo como pegamento.
* **Seguridad:** RLS (Row Level Security) es obligatorio. El `tenant_id` debe propagarse desde la API hasta la consulta SQL.

### Regla #2: El RAG es Sagrado e Inmutable
* No modifiques `cire-rag` para satisfacer un capricho de un cliente específico en `cire-orch`.
* Si `cire-orch` necesita filtrar por "año escolar", `cire-rag` debe proveer un filtro genérico de `metadata`, no un campo `school_year` en la tabla principal.

### Regla #3: El Orquestador es un Estratega, no un Router
* Evita lógica simple de `if/else` o Regex para decisiones complejas.
* Usa patrones de **Agentes (ReAct)** o **Grafos de Estado** para planificar la respuesta antes de consultar al RAG.
* El Orquestador es responsable de la "Traducción de Intención": Convierte *"Ayúdame con mi tarea"* (Usuario) en *"Buscar conceptos clave del libro X"* (RAG Query).

### Regla #4: Optimización Industrial
* **RAG:** Prioriza índices HNSW y búsquedas RPC.
* **ORCH:** Prioriza Streaming (TTFT) y no bloquees el hilo principal esperando al RAG.

---

## 3. Protocolo de Comunicación (The Handshake)

La comunicación entre ORCH y RAG es **Service-to-Service (S2S)** pero con contexto de usuario.

1.  **Auth:** ORCH se autentica con RAG usando un `Service Secret`.
2.  **Contexto:** ORCH **DEBE** enviar el header `X-Tenant-ID` en cada petición.
3.  **Trazabilidad:** ORCH genera un `Trace-ID` y lo pasa al RAG. Si hay un error, el log debe mostrar el camino completo.

## 3.1 Contrato HTTP del RAG (inventario)

Base URL: `http(s)://<rag-host>/api/v1`

Headers obligatorios (S2S):
- `X-Tenant-ID: <tenant_id>` en todas las rutas S2S (excepto `/health`, `/docs`, `/openapi.json`, `/api/v1/management/tenants`).
- En entornos desplegados: `Authorization: Bearer <RAG_SERVICE_SECRET>` o `X-Service-Secret: <RAG_SERVICE_SECRET>`.
- Si el payload/query incluye `tenant_id`, debe coincidir con `X-Tenant-ID`.

Endpoints disponibles (RAG):
- Health: `GET /health`
- Chat: `POST /chat/completions`, `POST /chat/feedback`
- Documents: `POST /documents`, `GET /documents`, `GET /documents/{document_id}/status`, `DELETE /documents/{document_id}`
- Management: `GET /management/tenants`, `GET /management/collections`, `GET /management/queue/status`, `GET /management/health`, `GET /management/retrieval/metrics`
- Retrieval (oficial v1): `POST /retrieval/validate-scope`, `POST /retrieval/hybrid`, `POST /retrieval/multi-query`, `POST /retrieval/explain`
- Retrieval (debug/diagnostico): `POST /debug/retrieval/chunks`, `POST /debug/retrieval/summaries`
- Ingestion: `POST /ingestion/embed`, `POST /ingestion/ingest`, `POST /ingestion/institutional`, `GET /ingestion/documents`, `GET /ingestion/collections`, `GET /ingestion/queue/status`, `POST /ingestion/collections/cleanup`, `POST /ingestion/retry/{doc_id}`, `POST /ingestion/batches`, `POST /ingestion/batches/{batch_id}/files`, `POST /ingestion/batches/{batch_id}/seal`, `GET /ingestion/batches/{batch_id}/status`, `GET /ingestion/batches/{batch_id}/progress`, `GET /ingestion/batches/{batch_id}/events`, `GET /ingestion/batches/active`, `GET /ingestion/batches/{batch_id}/stream`

Legacy retirado (no usar):
- `POST /knowledge/retrieve`
- `POST /retrieval/chunks`
- `POST /retrieval/summaries`

## 4. Estado Actual del Proyecto (A febrero 2026)

* **RAG:** Estable y robusto. Migrando lógica compleja a RPCs de Supabase. Enfocado en rendimiento (Matryoshka Embeddings, HNSW).
* **ORCH:** En evolución. Transformándose de un simple router a una "Plataforma Base" (Chasis) sobre la cual se construirán verticales específicas (Mallas, Legal, etc.).

---

> **NOTA FINAL:**
> Si vas a escribir código aquí, pregúntate:
> *¿Esto pertenece al Motor (RAG) o al Conductor (ORCH)?*
> Si dudas, revisa este archivo de nuevo.
