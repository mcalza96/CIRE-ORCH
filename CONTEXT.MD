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

- Cartuchos v2 estrictos (sin legacy):
  - `meta { id, description, owner }`
  - `identity { role, tone, style_guide[] }`
  - `router` (heuristicas)
  - `retrieval { by_mode, search_hints[], min_score }`
  - `validation { require_citations, forbidden_concepts[], fallback_message }`
  - `synthesis`
- Carga fail-fast al startup: si un YAML de `app/cartridges/*.yaml` no cumple esquema v2, la API no levanta.
- Resolucion de perfil auditable por request:
  - `source`: `db | header | tenant_map | tenant_yaml | base`
  - `requested_profile_id`
  - `applied_profile_id`
  - `decision_reason`
- Cascada obligatoria de cartucho:
  1) DB (`tenant_configs`)
  2) Header autorizado o tenant->profile map
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
  - traza de expansiones y filtrado
- Validacion declarativa profile-driven:
  - enforcement de citas
  - deteccion de conceptos prohibidos
  - fallback message por cartucho cuando no hay evidencia

### RAG

Implementacion vigente en `/Users/mcalzadilla/cire/rag`:

- RAG se mantiene agnostico a verticales.
- No se incorporan reglas de negocio de cartuchos en el motor.
- Mantiene contrato de retrieval y aislamiento por tenant.

## 3. Flujo Canonico E2E

1. Llega request con `query` + `tenant_id`.
2. Se autoriza tenant para el usuario.
3. Se resuelve cartucho por cascada (DB -> header/map -> tenant yaml -> base).
4. Router clasifica intencion usando chasis + profile hints.
5. Planner arma estrategia retrieval profile-driven.
6. ORCH consulta RAG con headers S2S y contexto de traza.
7. Se aplica validacion/auto-correccion si corresponde.
8. Se emite clarificacion declarativa cuando hay ambiguedad.
9. Se sintetiza con identidad/reglas/formato del cartucho.
10. Respuesta incluye metadata de perfil y observabilidad.

## 4. Reglas de Oro

1. No hardcodear verticales en core (`app/agent`, `app/api`, `app/core`, `app/security`).
2. RAG no absorbe logica de negocio de un tenant.
3. Politicas de dominio viven en cartuchos YAML versionados.
4. Todo fallback debe ser seguro y trazable.
5. Toda decision de tenancy/profile debe quedar auditada.

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
