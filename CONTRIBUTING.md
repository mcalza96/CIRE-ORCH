# Contributing

Gracias por contribuir al repositorio **Q/A Orchestrator**.

## Scope

Este repo mantiene solo el runtime de orquestación Q/A y su contrato HTTP con un RAG engine externo.

## Setup

```bash
cp .env.example .env.local
./bootstrap.sh
```

## Quality gates

```bash
ruff check runtime tests chat_cli.py
python -m compileall runtime tests chat_cli.py
pytest tests/unit -q
```

## Checklist de PR

- PR pequeño y enfocado.
- Tests actualizados para comportamiento modificado.
- Docs actualizadas cuando cambian políticas o contrato HTTP.

Mapa de docs: `docs/README.md`.
