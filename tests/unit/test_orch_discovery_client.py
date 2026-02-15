import asyncio

import httpx
import pytest

from app.core import orch_discovery_client as discovery


def test_list_authorized_tenants_parses_items(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"items": [{"id": "t1", "name": "Tenant 1"}, {"id": "t2"}]}

        text = "ok"

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(discovery.httpx, "AsyncClient", _Client)
    tenants = asyncio.run(discovery.list_authorized_tenants("http://localhost:8001", "token"))
    assert [t.id for t in tenants] == ["t1", "t2"]
    assert [t.name for t in tenants] == ["Tenant 1", "t2"]


def test_list_authorized_collections_parses_items(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"items": [{"id": "c1", "name": "ISO", "collection_key": "iso"}]}

        text = "ok"

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(discovery.httpx, "AsyncClient", _Client)
    collections = asyncio.run(discovery.list_authorized_collections("http://localhost:8001", "token", "t1"))
    assert len(collections) == 1
    assert collections[0].id == "c1"
    assert collections[0].collection_key == "iso"


def test_discovery_raises_on_http_error(monkeypatch):
    class _Response:
        status_code = 403
        text = "forbidden"

        @staticmethod
        def json():
            return {"error": "forbidden"}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(discovery.httpx, "AsyncClient", _Client)
    with pytest.raises(discovery.OrchestratorDiscoveryError) as exc:
        asyncio.run(discovery.list_authorized_tenants("http://localhost:8001", "token"))
    assert exc.value.status_code == 403


def test_discovery_raises_on_network_error(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://localhost:8001"))

    monkeypatch.setattr(discovery.httpx, "AsyncClient", _Client)
    with pytest.raises(discovery.OrchestratorDiscoveryError) as exc:
        asyncio.run(discovery.list_authorized_tenants("http://localhost:8001", "token"))
    assert exc.value.status_code is None


def test_list_authorized_collections_dedupes_by_id(monkeypatch):
    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "items": [
                    {"id": "c1", "name": "ISO", "collection_key": "iso"},
                    {"id": "c1", "name": "ISO", "collection_key": "iso"},
                    {"id": "c2", "name": "SM"},
                ]
            }

        text = "ok"

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(discovery.httpx, "AsyncClient", _Client)
    collections = asyncio.run(discovery.list_authorized_collections("http://localhost:8001", "token", "t1"))
    assert [c.id for c in collections] == ["c1", "c2"]
