import asyncio

import httpx

from app.infrastructure.clients import backend_selector as selector_module
from app.infrastructure.clients.backend_selector import RagBackendSelector


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_selector_prefers_local_when_healthy(monkeypatch):
    calls = {"get": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            calls["get"] += 1
            return _Response(200)

    monkeypatch.setattr(selector_module.httpx, "AsyncClient", _Client)

    selector = RagBackendSelector(
        local_url="http://local:8000",
        docker_url="http://docker:8000",
        ttl_seconds=20,
    )
    chosen = asyncio.run(selector.resolve_base_url())

    assert chosen == "http://local:8000"
    assert calls["get"] == 1


def test_selector_falls_back_to_docker_when_local_down(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            req = httpx.Request("GET", url)
            raise httpx.ConnectError("connection failed", request=req)

    monkeypatch.setattr(selector_module.httpx, "AsyncClient", _Client)

    selector = RagBackendSelector(
        local_url="http://local:8000",
        docker_url="http://docker:8000",
        ttl_seconds=20,
    )
    chosen = asyncio.run(selector.resolve_base_url())

    assert chosen == "http://docker:8000"


def test_selector_respects_force_backend(monkeypatch):
    calls = {"get": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            calls["get"] += 1
            return _Response(200)

    monkeypatch.setattr(selector_module.httpx, "AsyncClient", _Client)

    selector = RagBackendSelector(
        local_url="http://local:8000",
        docker_url="http://docker:8000",
        force_backend="docker",
    )
    chosen = asyncio.run(selector.resolve_base_url())

    assert chosen == "http://docker:8000"
    assert calls["get"] == 0


def test_selector_cache_ttl_avoids_repeated_probes(monkeypatch):
    calls = {"get": 0}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url):
            calls["get"] += 1
            return _Response(200)

    monkeypatch.setattr(selector_module.httpx, "AsyncClient", _Client)

    selector = RagBackendSelector(
        local_url="http://local:8000",
        docker_url="http://docker:8000",
        ttl_seconds=20,
    )
    first = asyncio.run(selector.resolve_base_url())
    second = asyncio.run(selector.resolve_base_url())

    assert first == "http://local:8000"
    assert second == "http://local:8000"
    assert calls["get"] == 1
