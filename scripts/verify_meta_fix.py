
import asyncio
import json
from uuid import UUID
from orch.app.agent.http_adapters import RagEngineRetrieverAdapter
from orch.app.agent.models import RetrievalPlan

async def test_diagnostics():
    adapter = RagEngineRetrieverAdapter()
    plan = RetrievalPlan(
        requested_standards=["ISO 9001"],
        chunk_k=5,
        chunk_fetch_k=20,
        summary_k=2,
        mode="interpretive"
    )
    
    # Query for something we know exists: PHVA (0.3.2)
    query = "¿Qué dice la norma ISO 9001 sobre el ciclo PHVA?"
    tenant_id = UUID("b18a053c-1787-4a43-ac97-60c459f455b8")
    collection_id = UUID("5cdcd14c-c256-41b3-ade0-f93b73d71429")
    
    print(f"Executing retrieval for query: {query}")
    evidence = await adapter.retrieve_chunks(
        query=query,
        tenant_id=str(tenant_id),
        collection_id=str(collection_id),
        user_id=str(UUID("cfc19919-db95-452a-878d-5793f6b2fee6")),
        plan=plan
    )
    
    diag = adapter.last_retrieval_diagnostics
    print("\n--- Diagnostics ---")
    if diag:
        print(f"Strategy: {diag.strategy}")
        # Check layer stats for standards
        trace = diag.trace
        if isinstance(trace, dict):
            print(f"Standards detected: {trace.get('standards')}")
            print(f"Raptor levels: {trace.get('raptor_levels')}")
    
    for i, ev in enumerate(evidence):
        print(f"\nChunk {i+1} Full Metadata: {json.dumps(ev.metadata, indent=2)}")
        row = ev.metadata.get("row", {})
        inner_meta = row.get("metadata", {})
        print(f"  Detected -> Source Standard: {inner_meta.get('source_standard')}")
        print(f"  Detected -> Clause ID: {inner_meta.get('clause_id')}")

if __name__ == "__main__":
    asyncio.run(test_diagnostics())
