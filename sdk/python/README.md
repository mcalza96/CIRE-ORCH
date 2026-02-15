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

client = CireRagClient(
    base_url="http://localhost:8000",
    api_key=None,
    tenant_storage_path="./tenant-context.json",  # opcional
)
client.set_tenant("tenant-demo")

try:
    response = client.create_chat_completion(
        message="Que exige la clausula 8.5 de ISO 9001?",
        max_context_chunks=8,
    )
    print(response["context_chunks"])
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
        client.set_tenant("tenant-demo")
        result = await client.create_chat_completion(
            message="Resume ISO 14001 clause 6.1",
        )
        print(result["context_chunks"])


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
- `validate_scope`
- `retrieval_hybrid`
- `retrieval_multi_query`
- `retrieval_explain`

Sync and async clients expose the same method names.

## Tenant behavior

- For tenant-scoped endpoints, the SDK automatically injects `X-Tenant-ID`.
- If `tenant_id` is missing, the SDK blocks the request with `TenantSelectionRequiredError`.
- If request tenant and active context tenant conflict, the SDK raises `TenantMismatchLocalError`.
- Backend tenant protocol errors (`TENANT_HEADER_REQUIRED`, `TENANT_MISMATCH`) are raised as `TenantProtocolError` with a UX-safe `user_message`.
