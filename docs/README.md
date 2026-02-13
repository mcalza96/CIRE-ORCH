# Documentation

Documentación canónica del repositorio **orchestrator-only**.

## Start here

- Setup local: `getting-started.md`
- Arquitectura: `architecture.md`
- Operación: `operations.md`
- Guía de desarrollo: `developer-guide.md`

## Operación rápida

- Levantar servicio: `../stack.sh up`
- Ver logs: `../stack.sh logs`
- Apagar servicio: `../stack.sh down`
- Ingesta HTTP: `../ing.sh --help`

## Política de source-of-truth

- `docs/*` contiene la documentación vigente.
- No hay runtime local de ingestión/retrieval en este repo.
- Dependencias de RAG se consumen exclusivamente por `RAG_ENGINE_LOCAL_URL` y `RAG_ENGINE_DOCKER_URL`.
