import asyncio
import pytest
import httpx
from app.agent import http_adapters as adapters_module
from app.infrastructure.clients.http_adapters import RagEngineRetrieverAdapter
from app.agent.types.models import RetrievalPlan
from app.infrastructure.config import settings

def test_retriever_injects_service_and_context_headers(monkeypatch):
    captured_headers = []

    class _Client:
        def __init__(self, *args, **kwargs):
            nonlocal captured_headers
            captured_headers.append(kwargs.get("headers", {}))

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, url, json):
            req = httpx.Request("POST", url)
            return httpx.Response(200, json={"items": []}, request=req)

        async def get(self, url):
            req = httpx.Request("GET", url)
            return httpx.Response(200, json={"status": "ok"}, request=req)

    # Configuramos un secreto para el test
    monkeypatch.setattr(settings, "RAG_SERVICE_SECRET", "test-secret-123")
    monkeypatch.setattr(adapters_module.httpx, "AsyncClient", _Client)

    retriever = RagEngineRetrieverAdapter(base_url="http://rag:8000")
    plan = RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2)

    asyncio.run(
        retriever.retrieve_chunks(
            query="test",
            tenant_id="t1",
            collection_id=None,
            plan=plan,
            user_id="u-1",
        )
    )

    assert len(captured_headers) > 0
    assert any(h.get("X-Service-Secret") == "test-secret-123" for h in captured_headers)
    assert any(h.get("X-Tenant-ID") == "t1" for h in captured_headers)
    assert any(h.get("X-User-ID") == "u-1" for h in captured_headers)

def test_retriever_fails_if_secret_missing(monkeypatch):
    # Aseguramos que el secreto sea None o vacío
    monkeypatch.setattr(settings, "RAG_SERVICE_SECRET", None)
    
    with pytest.raises(RuntimeError, match="RAG_SERVICE_SECRET must be configured"):
        RagEngineRetrieverAdapter()
