CIRE PLATFORM: CONTEXTO Y ARQUITECTURA MAESTRA
ADVERTENCIA PARA AGENTES DE IA:
Esta plataforma usa separación estricta de responsabilidades. No mezclar lógica de negocio en core.

1. Visión Global
CIRE opera bajo el paradigma:

Chasis Agéntico (Nivel 3) + Cartuchos (Nivel 4)
Capas:
Cliente thin: envía query + tenant_id.
ORCH core agnóstico: clasifica, planifica, reintenta, valida, sintetiza.
Cartucho por tenant: inyecta identidad, reglas, retrieval y validación.
RAG engine: retrieval/indexación agnóstica (Vectorial, GraphRAG, RAPTOR), sin opinión de dominio, y operando bajo un modelo de "Retrieval as a Pointer" (Late Grounding).
2. Estado Arquitectónico Actual (Febrero 2026)
ORCH (Orquestador)
Implementación vigente en /Users/mcalzadilla/cire/orch:
Cartuchos v2 estrictos (sin legacy), profile-driven:
meta { id, description, owner }
identity { role, tone, style_guide[] }
router (heurísticas y patrones)
query_modes { default_mode, modes{}, intent_rules[] }
retrieval { by_mode, search_hints[], min_score }
validation { require_citations, forbidden_concepts[], fallback_message }
synthesis (templates de prompts/estilo)
Tooling agnóstico:
ToolName es dinámico (str), no enum cerrado en core.
Validación de herramientas ocurre en runtime (registry/policies), no por hardcode en esquema.
QueryMode agnóstico:
Clasificación de intención y planificación guiadas por query_modes del cartucho.
Core sin ramas if mode == literal_*.
Carga fail-fast al startup:
Si un YAML de app/profiles/*.yaml no cumple esquema v2, la API no levanta.
base.yaml es fallback obligatorio de cartucho.
Resolución de perfil auditable por request:
source: db | header | dev_map | tenant_map | tenant_yaml | base
requested_profile_id
applied_profile_id
decision_reason
Cascada obligatoria de cartucho:
DB (tenant_configs)
Header autorizado o tenant->profile map/dev map
app/profiles/{tenant_id}.yaml
app/profiles/base.yaml
Seguridad de override por header:
whitelist por tenant
log explícito cuando override es denegado
fallback seguro con reason trazable
Observabilidad S2S activa:
X-Tenant-ID, X-Request-ID, X-Correlation-ID, X-Service-Secret
metadata de resolución en trace y respuesta API
Retrieval profile-driven & Evidencia Estricta:
search_hints aplicados a query/subqueries
min_score aplicado al pipeline
decomposition por scope y cláusula para consultas densas
coverage gate por scope y cláusula
traza con missing_scopes, missing_clause_refs
El generador (grounded_answer_service) basa sus respuestas exclusivamente en los fragmentos de texto crudo devueltos por el RAG, asegurando validación de citas exacta sin alucinaciones causadas por metadatos abstractos.
Control-flow robusto por códigos:
Retry/replan usa error_codes estructurados de retrieval
Evitar parseo por substrings de mensajes de log
Hard limits de infraestructura:
ReasoningBudget y RetrievalModeConfig con cotas de seguridad
clamps runtime para max_steps y max_reflections
RAG (Motor de Recuperación)
Implementación vigente en /Users/mcalzadilla/cire/rag:
RAG se mantiene agnóstico a verticales.
No se incorporan reglas de negocio de cartuchos en el motor.
Two-Stage Retrieval (Retrieval as a Pointer / Late Grounding):
GraphRAG actúa como "El Compás" (navega lógica compleja y relaciones normativas).
RAPTOR actúa como "El Mapa" (provee entendimiento del contexto global).
Vector/Híbrido actúa como "El Territorio" (búsqueda semántica directa).
Estas 3 estrategias no devuelven su texto abstracto o nodos combinados a ORCH; resuelven internamente un Set de Chunk IDs.
Reranking Estricto: El RetrievalBroker hace una deduplicación (Unión) de los Chunk IDs, extrae el texto crudo de Supabase (chunks) y somete solo el texto original al Reranker (Cohere/Jina), optimizando drásticamente la exactitud de la similitud semántica.
Expone un payload limpio hacia ORCH compuesto únicamente de chunks (evidencia verificable), referencias gráficas (visual anchors) y metadatos operativos (error_codes, coverage diagnostics).
3. Flujo Canónico E2E
Llega request con query + tenant_id.
Se autoriza tenant para el usuario.
Se resuelve cartucho por cascada (DB -> header/dev_map/map -> tenant yaml -> base).
Router/intent rules clasifican intención desde query_modes.
Planner arma estrategia retrieval profile-driven (descomponiendo queries complejas).
ORCH consulta RAG con headers S2S y contexto de traza.
RAG ejecuta Búsqueda Avanzada (Late Grounding): Navega Grafos/RAPTOR/Vectores internamente, recupera los Chunk IDs correspondientes, re-rankea el texto base y devuelve evidencia limpia.
Se aplica validación/auto-corrección de suficiencia de evidencia y coverage repair si corresponde.
Se emite clarificación declarativa cuando hay ambigüedad o falta de datos.
Se sintetiza con identidad/reglas/formato del cartucho, obligando la citación literal a los chunks entregados.
Respuesta incluye metadata de perfil y observabilidad.
4. Reglas de Oro
No hardcodear verticales en core (app/agent, app/api, app/core, app/security).
RAG no absorbe lógica de negocio de un tenant.
Políticas de dominio viven en cartuchos YAML versionados.
Todo fallback debe ser seguro y trazable.
Toda decisión de tenancy/profile debe quedar auditada.
Control-flow operativo debe basarse en códigos estructurados, no en parsing de strings.
Abstracción de Evidencia (Separation of Search vs. Generation Context): A ORCH solo se le entrega el "territorio" (fragmentos literales / chunks). La complejidad de cómo se encontró la aguja en el pajar (RAPTOR/Grafos) es responsabilidad exclusiva del RetrievalBroker del RAG.
5. Handshake ORCH <-> RAG
Headers obligatorios en S2S:
X-Service-Secret
X-Tenant-ID
X-Request-ID (cuando disponible)
X-Correlation-ID (cuando disponible)
X-User-ID en rutas que lo requieren
Si payload incluye tenant_id, debe ser consistente con X-Tenant-ID. El payload de respuesta obedece al contrato estricto de evidencia cruda (chunks y visual_anchors).

6. Contrato de Cartucho Privado en DB
Tabla sugerida: tenant_configs

Campos mínimos:
tenant_id text not null
profile_id text null
profile_version text null
status text null
agent_profile jsonb not null (schema v2)
updated_at timestamptz not null default now()
Reglas:
seleccionar fila más reciente
validar schema v2 antes de aplicar
si invalida: fallback seguro, sin romper request
registrar decision_reason en traza/logs
7. Notas Operativas
Este repositorio se considera greenfield para cartuchos v2: no mantener rutas legacy.
Antes de proponer cambios, ubicar responsabilidad correcta:
Motor (RAG): Indexación, Late Chunking, Grafo, Recuperación, Fusión (Two-Stage), Reranking algorítmico.
Chasis (ORCH core): Lógica de negocio multi-tenant, orquestación LangGraph, evaluación de suficiencia, validación de citas, generación.
Cartucho (tenant/profile): Perfiles, restricciones, tono, estilo, templates de prompting, configuración de reintentos.