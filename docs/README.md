# Documentation

Documentación canónica del repositorio **orchestrator-only**.

## Start here

- Setup local: `getting-started.md`
- Arquitectura: `architecture.md`
- Operación: `operations.md`
- Guía de desarrollo: `developer-guide.md`

## Política de source-of-truth

- `docs/*` contiene la documentación vigente.
- No hay runtime local de ingestión/retrieval en este repo.
- Dependencias de RAG se consumen exclusivamente por `RAG_ENGINE_URL`.
