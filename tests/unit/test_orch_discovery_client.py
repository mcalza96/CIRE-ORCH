import pytest

import app.infrastructure.clients.discovery_client as client


@pytest.mark.asyncio
async def test_list_authorized_tenants_includes_network_details(monkeypatch):
    class _FailingAsyncClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        async def get(self, *args, **kwargs):
            del args, kwargs
            req = client.httpx.Request("GET", "http://localhost:8001/api/v1/knowledge/tenants")
            raise client.httpx.ConnectError("boom", request=req)

    monkeypatch.setattr(client.httpx, "AsyncClient", _FailingAsyncClient)

    with pytest.raises(client.OrchestratorDiscoveryError) as exc:
        await client.list_authorized_tenants("http://localhost:8001", token="")

    text = str(exc.value)
    assert "Discovery request failed (network) GET" in text
    assert "/api/v1/knowledge/tenants" in text
    assert "ConnectError" in text
