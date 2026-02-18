# CIRE PLATFORM: CONTEXTO Y ARQUITECTURA MAESTRA

> ADVERTENCIA PARA AGENTES DE IA:
> Esta plataforma usa separacion estricta de responsabilidades. No mezclar logica de negocio en core.

## 1. Vision Global

CIRE opera bajo el paradigma:

## Chasis Agentico (Nivel 3) + Cartuchos (Nivel 4)

Capas:

1. Cliente thin: envia `query` + `tenant_id`.
2. ORCH core agnostico: clasifica, planifica, reintenta, valida, sintetiza.
3. Cartucho por tenant: inyecta identidad, reglas, retrieval y validacion.
4. RAG engine: retrieval/indexacion agnostica, sin opinion de dominio.

## 2. Estado Arquitectonico Actual (Febrero 2026)

### ORCH

Implementacion vigente en `/Users/mcalzadilla/cire/orch`:

- Cartuchos v2 estrictos (sin legacy), profile-driven:
  - `meta { id, description, owner }`
  - `identity { role, tone, style_guide[] }`
  - `router` (heuristicas y patrones)
  - `query_modes { default_mode, modes{}, intent_rules[] }`
  - `retrieval { by_mode, search_hints[], min_score }`
  - `validation { require_citations, forbidden_concepts[], fallback_message }`
  - `synthesis` (templates de prompts/estilo)
- Tooling agnostico:
  - `ToolName` es dinamico (`str`), no enum cerrado en core.
  - Validacion de herramientas ocurre en runtime (registry/policies), no por hardcode en esquema.
- QueryMode agnostico:
  - Clasificacion de intencion y planificacion guiadas por `query_modes` del cartucho.
  - Core sin ramas `if mode == literal_*`.
- Carga fail-fast al startup:
  - Si un YAML de `app/cartridges/*.yaml` no cumple esquema v2, la API no levanta.
  - `base.yaml` es fallback obligatorio de cartucho.
- Resolucion de perfil auditable por request:
  - `source`: `db | header | dev_map | tenant_map | tenant_yaml | base`
  - `requested_profile_id`
  - `applied_profile_id`
  - `decision_reason`
- Cascada obligatoria de cartucho:
  1) DB (`tenant_configs`)
  2) Header autorizado o tenant->profile map/dev map
  3) `app/cartridges/{tenant_id}.yaml`
  4) `app/cartridges/base.yaml`
- Seguridad de override por header:
  - whitelist por tenant
  - log explicito cuando override es denegado
  - fallback seguro con reason trazable
- Observabilidad S2S activa:
  - `X-Tenant-ID`, `X-Request-ID`, `X-Correlation-ID`, `X-Service-Secret`
  - metadata de resolucion en trace y respuesta API
- Retrieval profile-driven:
  - `search_hints` aplicados a query/subqueries
  - `min_score` aplicado al pipeline
  - decomposition por scope y clausula para consultas densas
  - coverage gate por scope y clausula
  - traza con `missing_scopes`, `missing_clause_refs`
- Control-flow robusto por codigos:
  - Retry/replan usa `error_codes` estructurados de retrieval
  - Evitar parseo por substrings de mensajes de log
- Hard limits de infraestructura:
  - `ReasoningBudget` y `RetrievalModeConfig` con cotas de seguridad
  - clamps runtime para `max_steps` y `max_reflections`

### RAG

Implementacion vigente en `/Users/mcalzadilla/cire/rag`:

- RAG se mantiene agnostico a verticales.
- No se incorporan reglas de negocio de cartuchos en el motor.
- Mantiene contrato de retrieval y aislamiento por tenant.
- Expone trazas y codigos estructurados consumidos por ORCH (`error_codes`, coverage diagnostics).

## 3. Flujo Canonico E2E

1. Llega request con `query` + `tenant_id`.
2. Se autoriza tenant para el usuario.
3. Se resuelve cartucho por cascada (DB -> header/dev_map/map -> tenant yaml -> base).
4. Router/intent rules clasifican intencion desde `query_modes`.
5. Planner arma estrategia retrieval profile-driven.
6. ORCH consulta RAG con headers S2S y contexto de traza.
7. Se aplica validacion/auto-correccion y coverage repair si corresponde.
8. Se emite clarificacion declarativa cuando hay ambiguedad.
9. Se sintetiza con identidad/reglas/formato del cartucho.
10. Respuesta incluye metadata de perfil y observabilidad.

## 4. Reglas de Oro

1. No hardcodear verticales en core (`app/agent`, `app/api`, `app/core`, `app/security`).
2. RAG no absorbe logica de negocio de un tenant.
3. Politicas de dominio viven en cartuchos YAML versionados.
4. Todo fallback debe ser seguro y trazable.
5. Toda decision de tenancy/profile debe quedar auditada.
6. Control-flow operativo debe basarse en codigos estructurados, no en parsing de strings.

## 5. Handshake ORCH <-> RAG

Headers obligatorios en S2S:

- `X-Service-Secret`
- `X-Tenant-ID`
- `X-Request-ID` (cuando disponible)
- `X-Correlation-ID` (cuando disponible)
- `X-User-ID` en rutas que lo requieren

Si payload incluye `tenant_id`, debe ser consistente con `X-Tenant-ID`.

## 6. Contrato de Cartucho Privado en DB

Tabla sugerida: `tenant_configs`

Campos minimos:

- `tenant_id text not null`
- `profile_id text null`
- `profile_version text null`
- `status text null`
- `agent_profile jsonb not null` (schema v2)
- `updated_at timestamptz not null default now()`

Reglas:

- seleccionar fila mas reciente
- validar schema v2 antes de aplicar
- si invalida: fallback seguro, sin romper request
- registrar decision_reason en traza/logs

## 7. Notas Operativas

- Este repositorio se considera greenfield para cartuchos v2: no mantener rutas legacy.
- Antes de proponer cambios, ubicar responsabilidad correcta:
  - Motor (RAG)
  - Chasis (ORCH core)
  - Cartucho (tenant/profile)
