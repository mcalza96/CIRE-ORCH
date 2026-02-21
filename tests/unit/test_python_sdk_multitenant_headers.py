import asyncio
import json

import pytest

from sdk.python.cire_rag_sdk.client import (
    AsyncCireRagClient,
    CireRagClient,
    TENANT_MISMATCH_CODE,
    TenantContext,
    TenantMismatchLocalError,
    TenantSelectionRequiredError,
)


class _SyncResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload, ensure_ascii=True)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _SyncSession:
    def __init__(self, responses: list[_SyncResponse], on_request=None):
        self.responses = responses
        self.requests: list[dict] = []
        self.on_request = on_request

    def request(self, method, url, headers=None, params=None, json=None, data=None, files=None, timeout=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "params": params,
                "json": json,
            }
        )
        if self.on_request:
            self.on_request(len(self.requests))
        return self.responses.pop(0)

    def close(self):
        return None


class _AsyncResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload, ensure_ascii=True)

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _AsyncClient:
    def __init__(self, responses: list[_AsyncResponse]):
        self.responses = responses
        self.requests: list[dict] = []

    async def request(self, method, url, headers=None, params=None, json=None, data=None, files=None):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "params": params,
                "json": json,
            }
        )
        return self.responses.pop(0)

    async def aclose(self):
        return None


def test_sdk_adds_tenant_header_and_syncs_payload():
    session = _SyncSession([_SyncResponse(200, {"answer": "ok"})])
    context = TenantContext(tenant_id="tenant-ctx")
    client = CireRagClient(base_url="http://localhost:8000", session=session, tenant_context=context)

    client.create_chat_completion(message="hola")

    req = session.requests[0]
    assert req["headers"]["X-Tenant-ID"] == "tenant-ctx"
    assert req["json"]["tenant_id"] == "tenant-ctx"


def test_sdk_blocks_request_without_tenant():
    session = _SyncSession([_SyncResponse(200, {"status": "ok"})])
    client = CireRagClient(base_url="http://localhost:8000", session=session)

    with pytest.raises(TenantSelectionRequiredError):
        client.create_chat_completion(message="hola")
    assert session.requests == []


def test_sdk_syncs_tenant_query_params():
    session = _SyncSession([_SyncResponse(200, {"items": []})])
    context = TenantContext(tenant_id="tenant-q")
    client = CireRagClient(base_url="http://localhost:8000", session=session, tenant_context=context)

    client.list_tenant_collections()

    req = session.requests[0]
    assert req["headers"]["X-Tenant-ID"] == "tenant-q"
    assert req["params"]["tenant_id"] == "tenant-q"


def test_sdk_raises_on_local_tenant_conflict():
    session = _SyncSession([_SyncResponse(200, {"items": []})])
    context = TenantContext(tenant_id="tenant-a")
    client = CireRagClient(base_url="http://localhost:8000", session=session, tenant_context=context)

    with pytest.raises(TenantMismatchLocalError):
        client.list_tenant_collections(tenant_id="tenant-b")
    assert session.requests == []


def test_sdk_retries_once_on_backend_tenant_mismatch(tmp_path):
    storage = tmp_path / "tenant-context.json"
    context = TenantContext(tenant_id="tenant-old", storage_path=storage)

    def _on_request(request_count: int) -> None:
        if request_count == 1:
            storage.write_text(json.dumps({"tenant_id": "tenant-new"}), encoding="utf-8")

    session = _SyncSession(
        [
            _SyncResponse(
                400,
                {
                    "error": {
                        "code": TENANT_MISMATCH_CODE,
                        "message": "mismatch",
                        "details": {},
                        "request_id": "req-1",
                    }
                },
            ),
            _SyncResponse(200, {"answer": "ok"}),
        ],
        on_request=_on_request,
    )
    client = CireRagClient(base_url="http://localhost:8000", session=session, tenant_context=context)

    response = client.create_chat_completion(message="hola")

    assert response["answer"] == "ok"
    assert len(session.requests) == 2
    assert session.requests[0]["headers"]["X-Tenant-ID"] == "tenant-old"
    assert session.requests[1]["headers"]["X-Tenant-ID"] == "tenant-new"
    assert session.requests[1]["json"]["tenant_id"] == "tenant-new"


def test_async_sdk_adds_header_and_payload():
    async_client = _AsyncClient([_AsyncResponse(200, {"answer": "ok"})])
    context = TenantContext(tenant_id="tenant-async")
    client = AsyncCireRagClient(base_url="http://localhost:8000", client=async_client, tenant_context=context)

    result = asyncio.run(client.create_chat_completion(message="hola"))

    assert result["answer"] == "ok"
    req = async_client.requests[0]
    assert req["headers"]["X-Tenant-ID"] == "tenant-async"
    assert req["json"]["tenant_id"] == "tenant-async"
