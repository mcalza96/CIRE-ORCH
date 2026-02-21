from unittest.mock import AsyncMock

import pytest

from app.infrastructure.clients.rag_client import RagRetrievalContractClient


@pytest.mark.asyncio
async def test_comprehensive_forwards_payload_to_post_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.config.settings.RAG_SERVICE_SECRET", "secret"
    )
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": [], "trace": {}})

    await client.comprehensive(
        query="consulta",
        tenant_id="tenant-x",
        user_id="user-x",
        request_id="req-1",
        correlation_id="corr-1",
        coverage_requirements={"require_all_scopes": True},
    )

    args, kwargs = client._post_json.call_args
    assert args[0] == "/api/v1/retrieval/comprehensive"
    payload = args[1]
    assert payload["tenant_id"] == "tenant-x"
    assert payload["coverage_requirements"]["require_all_scopes"] is True


@pytest.mark.asyncio
async def test_comprehensive_raises_on_http_404_without_legacy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.infrastructure.config.settings.RAG_SERVICE_SECRET", "secret"
    )
    client = RagRetrievalContractClient()

    async def _raise_404(*args, **kwargs):
        import httpx

        request = httpx.Request("POST", "http://local/api/v1/retrieval/comprehensive")
        response = httpx.Response(404, request=request)
        raise httpx.HTTPStatusError("not found", request=request, response=response)

    client._post_json = AsyncMock(side_effect=_raise_404)

    with pytest.raises(Exception):
        await client.comprehensive(
            query="consulta",
            tenant_id="tenant-x",
            user_id="user-x",
        )
