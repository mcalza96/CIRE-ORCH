import asyncio
import json
from unittest.mock import MagicMock
from app.graph.universal_flow import UniversalReasoningOrchestrator, UniversalState
from app.agent.application import HandleQuestionCommand

async def test_budget_trace():
    print("🔍 Testing UniversalFlow Reasoning Trace for Budget Fields...")
    
    # Mock dependencies
    retriever = MagicMock()
    generator = MagicMock()
    validator = MagicMock()
    
    orchestrator = UniversalReasoningOrchestrator(
        retriever=retriever,
        answer_generator=generator,
        validator=validator
    )
    
    # Simulate a final state
    mock_state: UniversalState = {
        "user_query": "test query",
        "working_query": "test query",
        "tenant_id": "test_tenant",
        "collection_id": "test_coll",
        "user_id": "test_user",
        "request_id": "test_req",
        "correlation_id": "test_corr",
        "scope_label": "test_scope",
        "agent_profile": None,
        "tool_results": [],
        "tool_cursor": 0,
        "plan_attempts": 1,
        "reflections": 0,
        "reasoning_steps": [],
        "working_memory": {},
        "chunks": [],
        "summaries": [],
        "retrieved_documents": [],
        "stop_reason": "done",
        "validation": MagicMock(accepted=True)
    }
    
    trace = orchestrator._build_reasoning_trace(mock_state)
    
    print("\n📊 Reasoning Trace Output:")
    print(json.dumps(trace, indent=2))
    
    if "stage_budgets_ms" in trace:
        print("\n✅ SUCCESS: 'stage_budgets_ms' found in reasoning trace.")
        budgets = trace["stage_budgets_ms"]
        required_keys = ["planner", "execute_tool", "generator", "validation", "total"]
        missing = [k for k in required_keys if k not in budgets]
        if not missing:
            print(f"✅ SUCCESS: All required budget keys present: {required_keys}")
        else:
            print(f"❌ FAILURE: Missing budget keys: {missing}")
    else:
        print("\n❌ FAILURE: 'stage_budgets_ms' NOT found in reasoning trace.")

if __name__ == "__main__":
    asyncio.run(test_budget_trace())
