import pytest
from app.graph.logic.clarification_llm import rewrite_plan_with_feedback_llm

@pytest.mark.asyncio
async def test_replan_with_feedback_llm(monkeypatch):
    monkeypatch.setenv("ORCH_CLARIFICATION_LLM_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "dummy_key")
    
    # We will mock the AsyncOpenAI response to return a controlled JSON
    class MockMessage:
        content = '{"new_plan": ["semantic_retrieval", "structural_extraction"], "dynamic_inputs": {"structural_extraction": {"schema_definition": "lista_roles"}}}'
    class MockChoice:
        message = MockMessage()
    class MockCompletion:
        choices = [MockChoice()]
        
    class MockCompletions:
        async def create(self, **kwargs):
            return MockCompletion()
            
    class MockChat:
        completions = MockCompletions()
        
    class MockClient:
        chat = MockChat()

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **kw: MockClient())
    import app.graph.universal.clarification_llm as cl_llm
    
    result = await cl_llm.rewrite_plan_with_feedback_llm(
        current_tools=["semantic_retrieval", "logical_comparison"],
        feedback="no quiero comparar, solo dame una tabla con la lista de roles",
        allowed_tools=["semantic_retrieval", "logical_comparison", "structural_extraction"]
    )
    
    assert isinstance(result, dict)
    assert result["new_plan"] == ["semantic_retrieval", "structural_extraction"]
    assert result.get("dynamic_inputs", {}).get("structural_extraction", {}).get("schema_definition") == "lista_roles"
