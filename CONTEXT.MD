# CIRE PLATFORM: CONTEXTO Y ARQUITECTURA MAESTRA

> **ADVERTENCIA PARA AGENTES DE IA:**
> Este no es un proyecto de juguete. Es una plataforma industrial distribuida.
> Lee este documento ANTES de sugerir cambios. Ignorar la separacion de responsabilidades (SoC) rompera el sistema.

## 1. Vision Global (The Big Picture)

CIRE no es una aplicacion monolitica. Es una **Linea de Ensamblaje de Conocimiento** para operar como plataforma multi-tenant en multiples verticales.

La filosofia vigente es:

## **Chasis Agentico + Cartuchos de Experto**

Tres capas estrictas y no negociables:

### Capa 1: Cliente (Thin Client)
* **Rol:** Interfaz minima.
* **Entrada:** `query` + `tenant_id` (+ metadatos de sesion).
* **Responsabilidad:** UX/UI y transporte.
* **Prohibido:** Embutir logica de IA, reglas de negocio o retrieval strategy.

### Capa 2: Orquestador Centralizado (Core Universal - Nivel 3)
* **Rol:** El **Chasis** metodologico.
* **Inteligencia Nivel 3:** Sabe **como** investigar (planificar, descomponer, verificar faltantes, reintentar, autocorregir).
* **Switching Yard:** Un ruteador ligero clasifica intencion (`saludo`, `simple`, `investigacion`) y decide si activar loop agentico.
* **Responsabilidad:** Orquestacion, razonamiento agnostico, control de flujo, memoria conversacional, observabilidad, streaming.
* **Prohibido:** Hardcodear dominios (`ISO`, `Leyes`, `Mineria`, etc.).

### Capa 3: Inyeccion Dinamica (Cartucho - Nivel 4)
* **Rol:** El **Driver** experto por tenant.
* **Inteligencia Nivel 4:** Aporta juicio especializado en runtime.
* **Contenido:** system prompts de rol, reglas de validacion/prioridad normativa, politicas y definiciones de herramientas.
* **Carga:** Se resuelve por `tenant_id` desde BD/cache.
* **Prohibido:** Acoplar cartuchos al core como codigo fijo.

---

## 2. Dominios Soberanos

### A. CIRE-RAG (El Motor de Verdad)
* **Rol:** Infraestructura agnostica de ingesta, indexacion y retrieval.
* **Filosofia:** "Potencia bruta, Cero Opinion".
* **Responsabilidad:** Documentos, chunks, embeddings, ranking hibrido, seguridad por tenant.
* **Inmutable ante verticales:** No se adapta a caprichos de negocio; solo expone primitivas genericas.
* **Stack:** Python (FastAPI), Supabase/Postgres (pgvector + RPC).

### B. CIRE-ORCH (El Cerebro Ejecutivo)
* **Rol:** Microservicio Orquestador Universal.
* **Filosofia:** "Metodo estable + juicio inyectable".
* **Responsabilidad:** Traducir intencion, coordinar agentes, consumir RAG y sintetizar respuesta bajo reglas del cartucho.
* **Limite:** No almacena documentos originales ni administra indices vectoriales.
* **Stack:** Python (FastAPI), LangGraph/Agents, SDK cliente RAG.

---

## 3. Flujo Canonico de una Peticion

1. Llega `Request(query, tenant_id)`.
2. **Router Nivel 3 ligero:** clasifica complejidad/intencion.
3. **Loader de cartucho:** obtiene config experta del `tenant_id` (cache + fallback BD).
4. **Reasoning Loop Nivel 3 pesado:** piensa con metodo agnostico y juzga con reglas Nivel 4 cargadas.
5. **Retrieval:** consulta al `RAG Engine` externo e inmutable.
6. **Synthesis:** responde aplicando restricciones y estilo del cartucho.
7. **Observabilidad:** traza completa (`trace_id`) entre cliente, ORCH y RAG.

---

## 4. Reglas de Oro para Desarrollo

### Regla #1: El Core no sabe de negocio
* El orquestador universal no debe contener terminos ni reglas de vertical.
* Toda especializacion vive en cartuchos versionados por tenant.

### Regla #2: El RAG es sagrado e inmutable
* No modificar `cire-rag` para resolver necesidades puntuales de un tenant.
* Si ORCH requiere filtros nuevos, deben ser genericos (metadata/scope), no columnas ad hoc por vertical.

### Regla #3: La BD manda en RAG
* Logica critica de retrieval/seguridad/mutaciones atomicas vive en SQL/RPC.
* Prohibido filtrar masivamente en Python lo que debe resolver Postgres.
* RLS obligatorio y propagacion estricta de `tenant_id`.

### Regla #4: El ORCH es estratega, no router tonto
* Evitar cadenas de `if/else` para decisiones de razonamiento complejo.
* Usar patrones agenticos/estado para planificacion y autocorreccion.

### Regla #5: Optimizacion industrial
* **RAG:** HNSW, RPC, consultas eficientes.
* **ORCH:** streaming temprano (TTFT), no bloquear hilo principal, fallbacks controlados.

---

## 5. Handshake ORCH <-> RAG (S2S)

La comunicacion ORCH-RAG es service-to-service con contexto de usuario:

1. **Auth:** ORCH usa `RAG_SERVICE_SECRET`.
2. **Tenant Context:** `X-Tenant-ID` obligatorio en rutas S2S (salvo excepciones operativas).
3. **Traceabilidad:** `Trace-ID` extremo a extremo.
4. **Consistencia:** Si payload incluye `tenant_id`, debe coincidir con header.

### 5.1 Inventario HTTP RAG (v1)

Base URL: `http(s)://<rag-host>/api/v1`

Endpoints disponibles:
* Health: `GET /health`
* Chat: `POST /chat/completions`, `POST /chat/feedback`
* Documents: `POST /documents`, `GET /documents`, `GET /documents/{document_id}/status`, `DELETE /documents/{document_id}`
* Management: `GET /management/tenants`, `GET /management/collections`, `GET /management/queue/status`, `GET /management/health`, `GET /management/retrieval/metrics`
* Retrieval (v1): `POST /retrieval/validate-scope`, `POST /retrieval/hybrid`, `POST /retrieval/multi-query`, `POST /retrieval/explain`
* Debug retrieval: `POST /debug/retrieval/chunks`, `POST /debug/retrieval/summaries`
* Ingestion: `POST /ingestion/embed`, `POST /ingestion/ingest`, `POST /ingestion/institutional`, `GET /ingestion/documents`, `GET /ingestion/collections`, `GET /ingestion/queue/status`, `POST /ingestion/collections/cleanup`, `POST /ingestion/retry/{doc_id}`, `POST /ingestion/batches`, `POST /ingestion/batches/{batch_id}/files`, `POST /ingestion/batches/{batch_id}/seal`, `GET /ingestion/batches/{batch_id}/status`, `GET /ingestion/batches/{batch_id}/progress`, `GET /ingestion/batches/{batch_id}/events`, `GET /ingestion/batches/active`, `GET /ingestion/batches/{batch_id}/stream`

Legacy retirado (no usar):
* `POST /knowledge/retrieve`
* `POST /retrieval/chunks`
* `POST /retrieval/summaries`

---

## 6. Estado Actual (Febrero 2026)

* **RAG:** Estable, robusto y optimizado hacia RPC + performance.
* **ORCH:** En transicion formal a chasis universal con inyeccion dinamica de cartuchos.
* **Meta de arquitectura:** un solo core para todos los tenants; cero logica vertical hardcodeada.

---

> **NOTA FINAL:**
> Antes de escribir codigo, responde mentalmente:
> *¿Esto pertenece al Motor (RAG), al Chasis (ORCH Core), o al Cartucho del tenant?*
> Si dudas, detente y redefine la responsabilidad.
