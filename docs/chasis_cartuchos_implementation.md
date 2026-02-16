# Implementacion Chasis/Cartuchos (Ejecucion por fases)

## Fase 0 - Baseline y guardrails

### Hallazgos iniciales (deuda de hardcoding)

- `app/agent/policies.py`: contiene vocabulario y reglas verticales (ISO, clausulas, conflicto denunciante/evidencia).
- `app/agent/retrieval_planner.py`: depende de regex fijas para referencias numericas (`\d+\.\d+`).
- `app/agent/grounded_answer_service.py`: prompt de auditor ISO embebido en codigo.
- `app/agent/http_adapters.py`: contiene heuristicas ISO durante sintesis y trazabilidad.

### Guardrails de arquitectura

- El core (nivel 3) no define negocio por tenant.
- La especializacion (nivel 4) se modela como datos validados (`AgentProfile`).
- Seleccion de perfil por `tenant_id` con fallback seguro (`base`).
- Toda respuesta debe registrar `tenant_id`, `profile_id` y `profile_version`.

## Fases tecnicas

1. Contrato de cartucho (`AgentProfile`) y validaciones.
2. Loader cacheado + resolucion por tenant/header.
3. Migracion inicial de ISO a `app/cartridges/iso_auditor.yaml`.
4. Inyeccion de perfil en sintesis.
5. Refactor progresivo de router/planner para consumir perfil.
6. Observabilidad por cartucho y gates de despliegue.

## Estado de ejecucion (actual)

- Fase 0: completada.
- Fase 1: completada.
- Fase 2: completada.
- Fase 3: completada (inyeccion de perfil en endpoint, use case y sintesis).
- Fase 4: en progreso (router/planner migrando a config inyectable).
- Fase 5: iniciado (trazas con `agent_profile` y respuesta API con metadata de perfil).

## Criterio anti-monolito

Si una regla menciona dominio especifico y no esta en cartucho, esta en lugar incorrecto.
