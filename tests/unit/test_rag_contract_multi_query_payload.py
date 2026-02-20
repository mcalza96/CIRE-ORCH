import pytest
from typing import Any, cast

from app.core.rag_retrieval_contract_client import RagRetrievalContractClient


class _FakeSelector:
    async def current_backend(self) -> str:
        return "local"

    async def resolve_base_url(self) -> str:
        return "http://local:8000"

    def is_forced(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_comprehensive_payload_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def _post_once(
        *, base_url: str, path: str, payload: dict, tenant_id: str, user_id: str | None, **kwargs
    ):
        captured["base_url"] = base_url
        captured["path"] = path
        captured["payload"] = payload
        return {"items": [], "trace": {}}

    monkeypatch.setattr(
        "app.core.rag_retrieval_contract_client.settings.RAG_SERVICE_SECRET", "secret"
    )
    client = RagRetrievalContractClient(backend_selector=cast(Any, _FakeSelector()))
    monkeypatch.setattr(client, "_post_once", _post_once)

    await client.comprehensive(
        query="compara ISO 9001 vs ISO 14001",
        tenant_id="tenant-1",
        user_id="u1",
        collection_id=None,
        k=12,
        fetch_k=60,
        filters={"source_standards": ["ISO 9001", "ISO 14001"]},
        coverage_requirements={
            "requested_standards": ["ISO 9001", "ISO 14001"],
            "require_all_scopes": True,
            "min_clause_refs": 1,
        },
    )

    assert captured["path"] == "/api/v1/retrieval/comprehensive"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["tenant_id"] == "tenant-1"
    assert payload["query"]
    assert payload["coverage_requirements"]["require_all_scopes"] is True
    assert "collection_id" not in payload
