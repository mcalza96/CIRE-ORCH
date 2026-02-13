# CIRE-RAG Python SDK (base)

Minimal client for product endpoints under `/api/v1`.

## Install

```bash
cd sdk/python
pip install -e .
```

## Usage

```python
from cire_rag_sdk import CireRagClient, CireRagApiError

client = CireRagClient(base_url="http://localhost:8000", api_key=None)

try:
    response = client.create_chat_completion(
        message="Que exige la clausula 8.5 de ISO 9001?",
        tenant_id="tenant-demo",
        max_context_chunks=8,
    )
    print(response["answer"])
    print(response["citations"])
except CireRagApiError as err:
    print(err.status, err.code, err.request_id)
```

## Async usage (orchestrators)

```python
import asyncio
from cire_rag_sdk import AsyncCireRagClient


async def main() -> None:
    async with AsyncCireRagClient(base_url="http://localhost:8000") as client:
        result = await client.create_chat_completion(
            message="Resume ISO 14001 clause 6.1",
            tenant_id="tenant-demo",
        )
        print(result["answer"])


asyncio.run(main())
```

## Covered methods

- `create_document`
- `list_documents`
- `get_document_status`
- `delete_document`
- `create_chat_completion`
- `submit_chat_feedback`
- `list_tenant_collections`
- `get_tenant_queue_status`
- `get_management_health`

Sync and async clients expose the same method names.
