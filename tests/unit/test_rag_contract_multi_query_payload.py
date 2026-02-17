import asyncio

import pytest

from app.core.rag_retrieval_contract_client import RagRetrievalContractClient


class _FakeSelector:
    async def current_backend(self) -> str:
        return "local"

    async def resolve_base_url(self) -> str:
        return "http://local:8000"

    def is_forced(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_multi_query_sanitizes_payload_and_defaults_merge(monkeypatch):
    captured = {}

    async def _post_once(
        *, base_url: str, path: str, payload: dict, tenant_id: str, user_id: str | None, **kwargs
    ):
        captured["base_url"] = base_url
        captured["path"] = path
        captured["payload"] = payload
        return {"items": [], "subqueries": [], "partial": False, "trace": {}}

    client = RagRetrievalContractClient(backend_selector=_FakeSelector())
    monkeypatch.setattr(client, "_post_once", _post_once)

    await client.multi_query(
        tenant_id="tenant-1",
        user_id=None,
        collection_id=None,
        queries=[
            {
                "id": "q1",
                "query": "iso 9001 clausula 8.5",
                "filters": {"source_standard": "ISO 9001"},
            },
            {"id": "bad", "oops": "nope"},
        ],
        merge={"strategy": "rrf", "rrf_k": 42, "top_k": 9, "extra": "ignored"},
    )

    assert captured["path"] == "/api/v1/retrieval/multi-query"
    payload = captured["payload"]
    assert payload["tenant_id"] == "tenant-1"
    assert "collection_id" not in payload

    assert len(payload["queries"]) == 1
    assert payload["queries"][0]["id"] == "q1"
    assert payload["merge"]["strategy"] == "rrf"
    assert payload["merge"]["rrf_k"] == 42
    assert payload["merge"]["top_k"] == 9
    assert "extra" not in payload["merge"]
