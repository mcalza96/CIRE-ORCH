import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.agent.semantic_subquery_planner import SemanticSubqueryPlanner
from app.core.config import settings

@pytest.mark.asyncio
async def test_plan_returns_empty_without_api_key():
    with patch("app.agent.semantic_subquery_planner.settings") as mock_settings:
        mock_settings.GROQ_API_KEY = None
        planner = SemanticSubqueryPlanner()
        result = await planner.plan(query="test query")
        assert result == []

@pytest.mark.asyncio
async def test_plan_extracts_json_from_text():
    # Mocking settings and client
    with patch("app.agent.semantic_subquery_planner.AsyncOpenAI") as mock_openai_class:
        mock_settings = MagicMock()
        mock_settings.GROQ_API_KEY = "fake-key"
        mock_settings.ORCH_PLANNER_MODEL = "test-model"
        
        with patch("app.agent.semantic_subquery_planner.settings", mock_settings):
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            
            planner = SemanticSubqueryPlanner()
            
            # Simulate LLM returning text with JSON embedded
            mock_completion = MagicMock()
            mock_completion.choices = [
                MagicMock(message=MagicMock(content='Sure, here is the plan: {"subqueries": [{"id": "q1", "query": "test query"}]} hope it helps!'))
            ]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
            
            result = await planner.plan(query="test query")
            
            assert len(result) == 1
            assert result[0]["id"] == "q1"
            assert result[0]["query"] == "test query"

@pytest.mark.asyncio
async def test_plan_one_repair_attempt_on_invalid_json():
    with patch("app.agent.semantic_subquery_planner.AsyncOpenAI") as mock_openai_class:
        mock_settings = MagicMock()
        mock_settings.GROQ_API_KEY = "fake-key"
        
        with patch("app.agent.semantic_subquery_planner.settings", mock_settings):
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            planner = SemanticSubqueryPlanner()
            
            # First call returns invalid JSON (but with braces to trigger extraction), second (repair) returns valid
            mock_comp1 = MagicMock()
            mock_comp1.choices = [MagicMock(message=MagicMock(content='{ "invalid": json }'))]
            
            mock_comp2 = MagicMock()
            mock_comp2.choices = [MagicMock(message=MagicMock(content='{"subqueries": [{"id": "repaired", "query": "fixed"}]}'))]
            
            mock_client.chat.completions.create = AsyncMock(side_effect=[mock_comp1, mock_comp2])
            
            result = await planner.plan(query="test query")
            
            # SemanticSubqueryPlanner calls self._client.chat.completions.create twice: 
            # once for initial and once for repair.
            assert mock_client.chat.completions.create.call_count == 2
            assert result[0]["id"] == "repaired"

@pytest.mark.asyncio
async def test_plan_drops_invalid_items_and_enforces_limit():
    with patch("app.agent.semantic_subquery_planner.AsyncOpenAI") as mock_openai_class:
        mock_settings = MagicMock()
        mock_settings.GROQ_API_KEY = "fake-key"
        mock_settings.ORCH_PLANNER_MAX_QUERIES = 2
        
        with patch("app.agent.semantic_subquery_planner.settings", mock_settings):
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            planner = SemanticSubqueryPlanner()
            
            # Result with 3 items (exceeds limit 2) and 1 invalid item
            mock_comp = MagicMock()
            mock_comp.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "subqueries": [
                        {"id": "q1", "query": "valid 1"},
                        {"id": "q2"}, # invalid - missing query
                        {"id": "q3", "query": "valid 3"},
                        {"id": "q4", "query": "valid 4"} # exceeds limit
                    ]
                })))
            ]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_comp)
            
            result = await planner.plan(query="test query", max_queries=2)
            
            # Should only match 'valid 1' and 'valid 3' because limit is 2 after dropping invalid
            assert len(result) == 2
            assert result[0]["id"] == "q1"
            assert result[1]["id"] == "q3"

@pytest.mark.asyncio
async def test_plan_subquery_filters_normalization():
    with patch("app.agent.semantic_subquery_planner.AsyncOpenAI") as mock_openai_class:
        mock_settings = MagicMock()
        mock_settings.GROQ_API_KEY = "fake-key"
        
        with patch("app.agent.semantic_subquery_planner.settings", mock_settings):
            mock_client = MagicMock()
            mock_openai_class.return_value = mock_client
            planner = SemanticSubqueryPlanner()
            
            mock_comp = MagicMock()
            mock_comp.choices = [
                MagicMock(message=MagicMock(content=json.dumps({
                    "subqueries": [
                        {
                            "id": "q1", 
                            "query": "test", 
                            "filters": {
                                "source_standard": "ISO 9001",
                                "time_range": {"field": "created_at", "from": "2024-01-01"},
                                "extra_key": "junk"
                            }
                        }
                    ]
                })))
            ]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_comp)
            
            result = await planner.plan(query="test")
            
            assert "extra_key" not in result[0]["filters"]
            assert result[0]["filters"]["source_standard"] == "ISO 9001"
            assert result[0]["filters"]["time_range"]["field"] == "created_at"
