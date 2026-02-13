# Getting Started

## Prerrequisitos

- Python 3.11+
- `bash`
- Un RAG engine externo accesible por HTTP

## 1) Bootstrap

```bash
cp .env.example .env.local
./bootstrap.sh
```

## 2) Configurar conexión al engine

Define en `.env.local`:

```bash
RAG_ENGINE_URL=http://<host-del-engine>:8000
```

## 3) Levantar orquestador

```bash
./start_api.sh
```

Health check:

```bash
curl http://localhost:8001/health
```

## 4) Probar flujo end-to-end

```bash
./chat.sh
```

o vía HTTP:

```bash
curl -X POST http://localhost:8001/api/v1/knowledge/answer \
  -H "Content-Type: application/json" \
  -d '{"query":"Que exige ISO 9001 en 7.5.3?","tenant_id":"tenant-1"}'
```
