import asyncio
import json

import pytest

from app.ui.chat_query_flow import _post_answer
from sdk.python.cire_rag_sdk.client import (
    TENANT_MISMATCH_CODE,
    TenantContext,
    TenantProtocolError,
    TenantSelectionRequiredError,
)


class _Response:
    def __init__(self, status_code: int, payload: dict, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload, ensure_ascii=True)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _Client:
    def __init__(self, responses: list[_Response], on_post=None):
        self.responses = responses
        self.posts: list[dict] = []
        self.on_post = on_post

    async def post(self, url, json=None, headers=None):
        self.posts.append({"url": url, "json": json or {}, "headers": headers or {}})
        if self.on_post:
            self.on_post(len(self.posts))
        return self.responses.pop(0)


def test_chat_post_sends_tenant_header_and_payload():
    context = TenantContext(tenant_id="tenant-ok")
    client = _Client([_Response(200, {"answer": "ok"})])

    result = asyncio.run(
        _post_answer(
            client=client,
            orchestrator_url="http://localhost:8001",
            tenant_context=context,
            query="hola",
            collection_id=None,
        )
    )

    assert result["answer"] == "ok"
    post = client.posts[0]
    assert post["headers"]["X-Tenant-ID"] == "tenant-ok"
    assert post["json"]["tenant_id"] == "tenant-ok"


def test_chat_post_blocks_when_tenant_missing():
    context = TenantContext()
    client = _Client([_Response(200, {"answer": "ok"})])

    with pytest.raises(TenantSelectionRequiredError):
        asyncio.run(
            _post_answer(
                client=client,
                orchestrator_url="http://localhost:8001",
                tenant_context=context,
                query="hola",
                collection_id=None,
            )
        )
    assert client.posts == []


def test_chat_post_retries_once_on_tenant_mismatch(tmp_path):
    storage = tmp_path / "tenant-context.json"
    context = TenantContext(tenant_id="tenant-old", storage_path=storage)

    def _on_post(post_count: int) -> None:
        if post_count == 1:
            storage.write_text(json.dumps({"tenant_id": "tenant-new"}), encoding="utf-8")

    client = _Client(
        [
            _Response(
                400,
                {
                    "error": {
                        "code": TENANT_MISMATCH_CODE,
                        "message": "mismatch",
                        "details": None,
                        "request_id": "req-1",
                    }
                },
            ),
            _Response(200, {"answer": "ok"}),
        ],
        on_post=_on_post,
    )

    result = asyncio.run(
        _post_answer(
            client=client,
            orchestrator_url="http://localhost:8001",
            tenant_context=context,
            query="hola",
            collection_id=None,
        )
    )

    assert result["answer"] == "ok"
    assert len(client.posts) == 2
    assert client.posts[0]["headers"]["X-Tenant-ID"] == "tenant-old"
    assert client.posts[1]["headers"]["X-Tenant-ID"] == "tenant-new"


def test_chat_post_maps_tenant_header_required_error():
    context = TenantContext(tenant_id="tenant-1")
    client = _Client(
        [
            _Response(
                400,
                {
                    "error": {
                        "code": "TENANT_HEADER_REQUIRED",
                        "message": "missing",
                        "details": None,
                        "request_id": "req-2",
                    }
                },
            ),
        ]
    )

    with pytest.raises(TenantProtocolError) as exc:
        asyncio.run(
            _post_answer(
                client=client,
                orchestrator_url="http://localhost:8001",
                tenant_context=context,
                query="hola",
                collection_id=None,
            )
        )
    assert "Selecciona una organización antes de continuar." in str(exc.value)
