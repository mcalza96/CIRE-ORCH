import pytest
from unittest.mock import AsyncMock, patch
from app.core.rag_retrieval_contract_client import RagRetrievalContractClient
from app.core.rag_contract_schemas import SubQueryRequest, MergeOptions

@pytest.mark.asyncio
async def test_multi_query_validation_drops_invalid_subqueries():
    client = RagRetrievalContractClient()
    # Mock _post_json to capture the payload
    client._post_json = AsyncMock(return_value={"items": []})

    queries = [
        {"id": "q1", "query": "valid query"},
        {"id": "q2"},  # Missing 'query' - invalid
        "not a dict",   # Not a dict - invalid
    ]

    await client.multi_query(
        tenant_id="test-tenant",
        user_id="test-user",
        collection_id="test-coll",
        queries=queries
    )

    args, kwargs = client._post_json.call_args
    payload = args[1]
    
    assert len(payload["queries"]) == 1
    assert payload["queries"][0]["id"] == "q1"
    assert payload["queries"][0]["query"] == "valid query"

@pytest.mark.asyncio
async def test_multi_query_merge_defaults_on_invalid():
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": []})

    # Invalid merge options (e.g. strategy not 'rrf')
    invalid_merge = {"strategy": "invalid", "rrf_k": 1000}

    await client.multi_query(
        tenant_id="test-tenant",
        user_id="test-user",
        collection_id=None,
        queries=[{"id": "q1", "query": "test"}],
        merge=invalid_merge
    )

    args, kwargs = client._post_json.call_args
    payload = args[1]
    
    # Defaults should be preserved
    assert payload["merge"]["strategy"] == "rrf"
    assert payload["merge"]["rrf_k"] == 60

@pytest.mark.asyncio
async def test_multi_query_merge_ignores_extra_keys():
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": []})

    merge_with_extra = {"strategy": "rrf", "extra_key": "ignore me", "top_k": 20}

    await client.multi_query(
        tenant_id="test-tenant",
        user_id="test-user",
        collection_id=None,
        queries=[{"id": "q1", "query": "test"}],
        merge=merge_with_extra
    )

    args, kwargs = client._post_json.call_args
    payload = args[1]
    
    assert "extra_key" not in payload["merge"]
    assert payload["merge"]["top_k"] == 20

@pytest.mark.asyncio
async def test_multi_query_payload_final_structure():
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": []})

    await client.multi_query(
        tenant_id="test-tenant",
        user_id="test-user",
        collection_id="test-coll",
        queries=[{"id": "q1", "query": "test"}],
        merge={"strategy": "rrf", "top_k": 10}
    )

    args, kwargs = client._post_json.call_args
    payload = args[1]
    
    assert payload["tenant_id"] == "test-tenant"
    assert payload["collection_id"] == "test-coll"
    assert "queries" in payload
    assert "merge" in payload

@pytest.mark.asyncio
async def test_filters_serialization_with_alias_and_metadata():
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": []})

    queries = [{
        "id": "q1",
        "query": "test",
        "filters": {
            "metadata": {"doc_type": "policy"},
            "time_range": {
                "field": "updated_at",
                "from": "2023-01-01T00:00:00Z"
            }
        }
    }]

    await client.multi_query(
        tenant_id="test-tenant",
        user_id="test-user",
        collection_id=None,
        queries=queries
    )

    args, kwargs = client._post_json.call_args
    payload = args[1]
    
    q_filters = payload["queries"][0]["filters"]
    assert q_filters["metadata"] == {"doc_type": "policy"}
    # Verify alias 'from' is used instead of 'from_'
    assert q_filters["time_range"]["from"] == "2023-01-01T00:00:00Z"
    assert q_filters["time_range"]["field"] == "updated_at"

@pytest.mark.asyncio
async def test_multi_query_excludes_none_collection_id():
    client = RagRetrievalContractClient()
    client._post_json = AsyncMock(return_value={"items": []})
    
    await client.multi_query(
        tenant_id="t1",
        collection_id=None,
        user_id="u1",
        queries=[{"id": "1", "query": "q1"}]
    )
    
    args, _ = client._post_json.call_args
    payload = args[1]
    assert "collection_id" not in payload
    assert payload["tenant_id"] == "t1"
