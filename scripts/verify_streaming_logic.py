import asyncio
import json
import sys
import os
import time
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request

# Ensure imports work
# Ensure imports work (add 'orch' to path)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.api.v1.routes.knowledge import answer_with_orchestrator_stream, OrchestratorQuestionRequest
from app.api.deps import UserContext
from app.agent.application import HandleQuestionUseCase, HandleQuestionResult
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalDiagnostics, RetrievalPlan, ValidationResult, QueryIntent

# Mocking classes
class MockResult:
    def __init__(self):
        self.answer = AnswerDraft(
            text="This is a streaming answer.",
            mode="synthesis",
            evidence=[
                EvidenceItem(content="Chunk 1", source="s1", score=0.9),
                EvidenceItem(content="Chunk 2", source="s2", score=0.8),
            ]
        )
        self.plan = RetrievalPlan(
            mode="synthesis", 
            chunk_k=10, 
            chunk_fetch_k=40, 
            summary_k=5, 
            requested_standards=()
        )
        self.intent = QueryIntent(mode="synthesis", rationale="test")
        self.engine = "mock_engine"
        self.retrieval = RetrievalDiagnostics(
            contract="advanced", 
            strategy="mock_strategy", 
            partial=False, 
            trace={}
        )
        self.validation = ValidationResult(accepted=True, issues=[])
        self.clarification = None
        self.reasoning_trace = {}

class MockUseCase:
    async def execute(self, command):
        # Simulate processing time
        await asyncio.sleep(1.5)
        return MockResult()

async def verify_stream():
    print("--- Verifying Streaming Logic ---")
    
    # Mocks
    mock_request = MagicMock(spec=Request)
    mock_request.headers.get.return_value = "req-123"
    
    mock_user = UserContext(user_id="user-1", email="test@example.com", is_superuser=False)
    
    mock_use_case = MockUseCase()
    
    payload = OrchestratorQuestionRequest(query="Test stream", tenant_id="tenant-1")
    
    # Patch authorize dependencies (tricky without injection override)
    # Actually, the function calls `authorize_requested_tenant` directly.
    # We might need to mock the module function.
    
    from unittest.mock import patch
    
    with patch("app.api.v1.routes.knowledge.authorize_requested_tenant", new_callable=AsyncMock) as mock_auth, \
         patch("app.api.v1.routes.knowledge.resolve_allowed_tenants", new_callable=AsyncMock) as mock_resolve, \
         patch("app.api.v1.routes.knowledge.resolve_agent_profile", new_callable=AsyncMock) as mock_profile, \
         patch("app.api.v1.routes.knowledge.scope_metrics_store") as mock_metrics:
         
        mock_auth.return_value = "tenant-1"
        mock_profile.return_value.profile.profile_id = "default"
        mock_profile.return_value.profile.version = "1.0"
        mock_profile.return_value.profile.status = "active"
        mock_profile.return_value.resolution.source = "default"
        mock_profile.return_value.resolution.decision_reason = "default"
        mock_profile.return_value.resolution.model_dump.return_value = {}
        
        # Execute
        response = await answer_with_orchestrator_stream(
            http_request=mock_request,
            request=payload,
            current_user=mock_user,
            use_case=mock_use_case
        )
        
        print("Response created. Iterating stream...")
        start_time = time.perf_counter()
        
        async for chunk in response.body_iterator:
            chunk_str = chunk.decode("utf-8")
            print(f"Received chunk: {chunk_str.strip()}")
            
            if "event: status" in chunk_str:
                data = json.loads(chunk_str.split("data: ")[1].strip())
                if data["type"] == "working":
                    print(f"  -> Working pulse: {data.get('pulse')} (Elapsed: {data.get('elapsed_ms')}ms)")
            
            if "event: result" in chunk_str:
                data = json.loads(chunk_str.split("data: ")[1].strip())
                if data["type"] == "final_answer":
                    print("  -> Final Answer received!")
                    print(f"  -> Answer: {data['answer']}")
        
        duration = time.perf_counter() - start_time
        print(f"Stream finished in {duration:.2f}s")
        
        if duration >= 1.5:
             print("✅ SUCCESS: Stream simulated processing delay and events.")
        else:
             print("⚠️ WARNING: Stream was too fast, mocking might be off.")

if __name__ == "__main__":
    asyncio.run(verify_stream())
