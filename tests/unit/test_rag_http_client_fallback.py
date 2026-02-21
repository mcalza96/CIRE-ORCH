import asyncio
from typing import cast

import httpx

from app.agent import http_adapters as adapters_module
from app.infrastructure.clients.http_adapters import RagEngineRetrieverAdapter
from app.agent.types.models import RetrievalPlan
from app.clients.backend_selector import RagBackendSelector


class _FakeSelector:
    def __init__(self):
        self.selected = "local"
        self.updated_to: str | None = None

    async def current_backend(self) -> str:
        return self.selected

    async def resolve_base_url(self) -> str:
        return "http://local:8000"

    def is_forced(self) -> bool:
        return False

    def alternate_backend(self, backend: str) -> str:
        return "docker" if backend == "local" else "local"

    def base_url_for(self, backend: str) -> str:
        if backend == "docker":
            return "http://docker:8000"
        return "http://local:8000"

    def set_backend(self, backend: str) -> None:
        self.updated_to = backend


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://example")
            raise httpx.HTTPStatusError("http error", request=req, response=httpx.Response(self.status_code, request=req))

    def json(self):
        return self._payload


def test_retriever_retries_with_alternate_backend_on_connection_error(monkeypatch):
    calls: list[str] = []

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            calls.append(url)
            if url.startswith("http://local:8000"):
                req = httpx.Request("POST", url)
                raise httpx.ConnectError("local unavailable", request=req)
            return _FakeResponse(
                200,
                {
                    "items": [
                        {"source": "C1", "content": "texto recuperado", "score": 0.9, "metadata": {}},
                    ]
                },
            )

    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", _Client)

    fake_selector = _FakeSelector()
    retriever = RagEngineRetrieverAdapter(backend_selector=cast(RagBackendSelector, fake_selector))
    plan = RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2)

    items = asyncio.run(
        retriever.retrieve_chunks(
            query="prueba",
            tenant_id="tenant-1",
            collection_id=None,
            plan=plan,
        )
    )

    assert len(items) == 1
    assert calls[0].startswith("http://local:8000")
    assert calls[1].startswith("http://docker:8000")
    assert fake_selector.updated_to == "docker"


def test_summaries_degrade_to_empty_when_both_backends_fail(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            req = httpx.Request("POST", url)
            raise httpx.ConnectError("backend unavailable", request=req)

    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", _Client)

    retriever = RagEngineRetrieverAdapter(backend_selector=cast(RagBackendSelector, _FakeSelector()))
    plan = RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2)

    items = asyncio.run(
        retriever.retrieve_summaries(
            query="prueba",
            tenant_id="tenant-1",
            collection_id=None,
            plan=plan,
        )
    )

    assert items == []


def test_chunks_raise_when_both_backends_fail(monkeypatch):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            req = httpx.Request("POST", url)
            raise httpx.ConnectError("backend unavailable", request=req)

    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", _Client)

    retriever = RagEngineRetrieverAdapter(backend_selector=cast(RagBackendSelector, _FakeSelector()))
    plan = RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2)

    raised = False
    try:
        asyncio.run(
            retriever.retrieve_chunks(
                query="prueba",
                tenant_id="tenant-1",
                collection_id=None,
                plan=plan,
            )
        )
    except httpx.RequestError:
        raised = True

    assert raised is True
