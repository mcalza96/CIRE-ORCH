import asyncio

import httpx
import pytest
from fastapi import HTTPException

from app.api.v1.routers import observability as obs_routes
from app.infrastructure.config import settings


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    captured: dict = {}

    def __init__(self, *args, **kwargs):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, params=None, headers=None):
        _FakeClient.captured = {
            "url": url,
            "params": dict(params or {}),
            "headers": dict(headers or {}),
        }
        return _FakeResponse({"ok": True})


class _FakeSelector:
    async def resolve_base_url(self) -> str:
        return "http://rag:8000"


def test_proxy_json_get_adds_s2s_headers(monkeypatch):
    monkeypatch.setattr(obs_routes, "_selector", lambda: _FakeSelector())
    monkeypatch.setattr(obs_routes.httpx, "AsyncClient", _FakeClient)
    monkeypatch.setattr(settings, "RAG_SERVICE_SECRET", "secret-123")

    payload = asyncio.run(
        obs_routes._proxy_json_get(
            path="/api/v1/ingestion/batches/b1/progress",
            params={"tenant_id": "tenant-a"},
            tenant_id="tenant-a",
            user_id="user-1",
            operation="batch_progress",
        )
    )

    assert payload == {"ok": True}
    assert _FakeClient.captured["url"] == "http://rag:8000/api/v1/ingestion/batches/b1/progress"
    assert _FakeClient.captured["headers"]["X-Service-Secret"] == "secret-123"
    assert _FakeClient.captured["headers"]["X-Tenant-ID"] == "tenant-a"
    assert _FakeClient.captured["headers"]["X-User-ID"] == "user-1"


def test_map_upstream_401_to_502():
    request = httpx.Request("GET", "http://rag:8000/api/v1/ingestion/batches/b1/progress")
    response = httpx.Response(401, request=request)
    exc = httpx.HTTPStatusError("unauthorized", request=request, response=response)

    with pytest.raises(HTTPException) as caught:
        obs_routes._map_upstream_http_error(exc, operation="batch_progress")

    assert caught.value.status_code == 502
    assert caught.value.detail["code"] == "RAG_UPSTREAM_AUTH_FAILED"
    assert caught.value.detail["upstream_status"] == 401


def test_map_upstream_500_to_502():
    request = httpx.Request("GET", "http://rag:8000/api/v1/ingestion/batches/b1/progress")
    response = httpx.Response(500, request=request)
    exc = httpx.HTTPStatusError("server-error", request=request, response=response)

    with pytest.raises(HTTPException) as caught:
        obs_routes._map_upstream_http_error(exc, operation="batch_progress")

    assert caught.value.status_code == 502
    assert caught.value.detail["code"] == "RAG_UPSTREAM_ERROR"
    assert caught.value.detail["upstream_status"] == 500
