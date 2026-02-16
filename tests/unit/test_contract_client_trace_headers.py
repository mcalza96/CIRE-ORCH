import pytest


@pytest.mark.asyncio
async def test_contract_client_sends_trace_headers(monkeypatch):
    import httpx

    from app.core.rag_retrieval_contract_client import RagRetrievalContractClient

    sent_headers = {}

    class _FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"items": []}

    class _FakeClient:
        def __init__(self, timeout, headers):
            sent_headers.update(headers)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    client = RagRetrievalContractClient()
    await client._post_once(
        base_url="http://local",
        path="/api/v1/retrieval/hybrid",
        payload={"tenant_id": "t"},
        tenant_id="t",
        user_id=None,
    )
    assert "X-Trace-ID" in sent_headers
    assert "X-Correlation-ID" in sent_headers
