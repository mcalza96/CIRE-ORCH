# Ingestion Client (Portable)

`ing.sh` en esta carpeta es un cliente HTTP para orquestar batches de ingestion contra CIRE-RAG.

Se puede copiar tal cual a otro repositorio (por ejemplo el de orchestrator) con cambios minimos.

## Uso rapido

```bash
./ing.sh --file ./docs/manual.pdf
```

Entrada alternativa estilo "bin":

```bash
./bin/ing.sh --file ./docs/manual.pdf
```

Modo no interactivo:

```bash
RAG_URL=http://localhost:8000 \
TENANT_ID=<tenant_uuid> \
COLLECTION_ID=iso-9001 \
./ing.sh --file ./docs/manual.pdf --embedding-mode CLOUD --no-wait
```

## Variables principales

- `RAG_URL` (default: `http://localhost:8000`)
- `TENANT_ID`
- `COLLECTION_ID`
- `EMBEDDING_MODE` (`LOCAL` o `CLOUD`)
- `BATCH_POLL_INTERVAL` (default `15`)
- `BATCH_POLL_MAX` (default `120`)

## Dependencias

- `bash`
- `curl`
- `python3`

Opcional para estimaciones locales:

- `venv/bin/python` en la raiz del repo (si existe, se usa automaticamente)

## Portabilidad a otro repo

1. Copia la carpeta `tools/ingestion-client/` completa.
2. Ejecuta `chmod +x ing.sh`.
3. Define `RAG_URL` apuntando al engine desplegado.
4. Ejecuta `./ing.sh`.

No requiere cambios de rutas internas: el script es standalone.

Overrides opcionales:

- `INGEST_CLIENT_STATE_DIR=/path/state` (donde guardar `.rag_tenants.json`)
- `INGEST_CLIENT_PYTHON=/path/python` (interprete Python a usar)

## Paquete minimo para copiar/pegar

Archivos recomendados:

- `ing.sh`
- `bin/ing.sh`
- `.env.example`
- `README.md`

## Notas

- El wrapper de raiz `./ing.sh` en este repo solo delega a este script.
- Si no defines `RAG_URL`, el wrapper de raiz puede derivarlo desde `RAG_ENGINE_*` (modo híbrido local/docker).
- Este cliente no requiere acceso a DB ni secretos internos del engine; opera via API.
