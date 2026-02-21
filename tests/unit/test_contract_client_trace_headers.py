import pytest
from typing import Any, cast


@pytest.mark.asyncio
async def test_contract_client_sends_trace_headers():
    from app.infrastructure.clients.rag_client import RagRetrievalContractClient

    sent_headers = {}

    class _FakeResponse:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {"items": []}

    class _FakeClient:
        timeout = type(
            "_Timeout",
            (),
            {"connect": 0.5, "read": 1.0, "write": 1.0, "pool": 0.5},
        )()

        async def post(self, url, json, headers):
            del url, json
            sent_headers.update(headers)
            return _FakeResponse()

    client = RagRetrievalContractClient(http_client=cast(Any, _FakeClient()))
    await client._post_once(
        base_url="http://local",
        path="/api/v1/retrieval/hybrid",
        payload={"tenant_id": "t"},
        tenant_id="t",
        user_id=None,
    )
    assert "X-Trace-ID" in sent_headers
    assert "X-Correlation-ID" in sent_headers
