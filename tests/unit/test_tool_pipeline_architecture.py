"""Tests for Phase 3 architectural fixes: Map-Reduce, Context Injection, and Piping."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    RetrievalPlan,
    RetrievalDiagnostics,
    ToolResult,
)
from app.graph.universal.steps import aggregate_subqueries_node, generator_node, execute_tool_node
from app.graph.universal import steps as steps_module
from app.agent.tools.base import AgentTool, ToolRuntimeContext


class MockAnswerGenerator:
    def __init__(self):
        self.generate = AsyncMock(
            return_value=AnswerDraft(text="Mocked Answer", mode="default", evidence=[])
        )


class MockComponents:
    def __init__(self):
        self.answer_generator = MockAnswerGenerator()
        self.retriever = MagicMock()
        self.validator = MagicMock()
        self.tools = {}

    def _runtime_context(self):
        return MagicMock(spec=ToolRuntimeContext)


class MockTool(AgentTool):
    name = "mock_tool"

    async def run(self, payload: dict, state: dict, context: ToolRuntimeContext) -> ToolResult:
        # Echo payload for verification
        return ToolResult(tool="mock_tool", ok=True, output=payload)


@pytest.mark.asyncio
async def test_aggregate_subqueries_uses_llm_summarization(monkeypatch):
    monkeypatch.setattr(
        "app.graph.universal.steps.settings.ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", True
    )

    components = MockComponents()
    state = {
        "subquery_groups": [
            {
                "id": "q1",
                "query": "Who is the CEO?",
                "items": [{"content": "The CEO is Alice.", "source": "Doc1"}],
            }
        ],
        "chunks": [],
        "retrieval": RetrievalDiagnostics(contract="advanced", strategy="multi_query", trace={}),
    }

    result = await aggregate_subqueries_node(state, components)

    partials = result.get("partial_answers", [])
    assert len(partials) == 1
    assert partials[0]["summary"] == "Mocked Answer"

    # Verify AnswerGenerator was called
    components.answer_generator.generate.assert_called_once()
    call_kwargs = components.answer_generator.generate.call_args.kwargs
    assert "Who is the CEO?" in call_kwargs["query"]
    assert len(call_kwargs["chunks"]) == 1
    assert call_kwargs["chunks"][0].content == "The CEO is Alice."


@pytest.mark.asyncio
async def test_aggregate_subqueries_runs_with_asyncio_gather(monkeypatch):
    monkeypatch.setattr(
        "app.graph.universal.steps.settings.ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", True
    )

    components = MockComponents()
    state = {
        "subquery_groups": [
            {
                "id": "q1",
                "query": "Who is the CEO?",
                "items": [{"content": "The CEO is Alice.", "source": "Doc1"}],
            },
            {
                "id": "q2",
                "query": "Where is HQ?",
                "items": [{"content": "HQ is in Bogota.", "source": "Doc2"}],
            },
        ],
        "chunks": [],
        "retrieval": RetrievalDiagnostics(contract="advanced", strategy="multi_query", trace={}),
    }

    real_gather = asyncio.gather
    gather_spy = AsyncMock(
        side_effect=lambda *aws, return_exceptions=False: real_gather(
            *aws, return_exceptions=return_exceptions
        )
    )
    monkeypatch.setattr(steps_module.asyncio, "gather", gather_spy)

    await aggregate_subqueries_node(state, components)

    gather_spy.assert_called_once()
    assert components.answer_generator.generate.await_count == 2


@pytest.mark.asyncio
async def test_generator_node_injects_structured_context(monkeypatch):
    components = MockComponents()
    state = {
        "retrieval_plan": RetrievalPlan(
            mode="default", chunk_k=1, chunk_fetch_k=1, summary_k=1, require_literal_evidence=False
        ),
        "chunks": [],
        "summaries": [],
        "working_memory": {"tool_a": {"key": "value"}},
        "partial_answers": [{"id": "q1", "summary": "Part 1"}],
        "user_query": "Test Query",
    }

    await generator_node(state, components)

    components.answer_generator.generate.assert_called_once()
    call_kwargs = components.answer_generator.generate.call_args.kwargs

    # Verify context injection
    assert call_kwargs["working_memory"] == {"tool_a": {"key": "value"}}
    assert call_kwargs["partial_answers"] == [{"id": "q1", "summary": "Part 1"}]


@pytest.mark.asyncio
async def test_generator_node_does_not_inject_synthetic_partial_summaries(monkeypatch):
    components = MockComponents()
    state = {
        "retrieval_plan": RetrievalPlan(
            mode="default", chunk_k=1, chunk_fetch_k=1, summary_k=1, require_literal_evidence=False
        ),
        "chunks": [],
        "summaries": [EvidenceItem(source="S1", content="base summary", score=0.5)],
        "working_memory": {},
        "partial_answers": [{"id": "q1", "query": "q", "summary": "Part 1"}],
        "user_query": "Test Query",
    }

    await generator_node(state, components)

    call_kwargs = components.answer_generator.generate.call_args.kwargs
    passed_summaries = call_kwargs["summaries"]

    assert len(passed_summaries) == 1
    assert passed_summaries[0].source == "S1"
    assert not any(str(item.source).startswith("RMAP") for item in passed_summaries)


@pytest.mark.asyncio
async def test_execute_tool_node_injects_full_working_memory(monkeypatch):
    components = MockComponents()
    tool = MockTool()
    components.tools = {"mock_tool": tool}

    state = {
        "reasoning_plan": MagicMock(steps=[MagicMock(tool="mock_tool", input={})]),
        "working_memory": {"prev_tool": {"data": 123}},
        "tool_cursor": 0,
        "tool_results": [],
    }

    # Mock get_tool/resolve_allowed_tools infrastructure if needed,
    # but here we rely on components.tools being used.
    # Note: execute_tool_node calls get_tool(components.tools, ...)

    result_updates = await execute_tool_node(state, components)

    tool_results = result_updates.get("tool_results", [])
    assert len(tool_results) == 1
    output = tool_results[0].output

    # Verify working_memory was injected into payload
    assert "working_memory" in output
    assert output["working_memory"] == {"prev_tool": {"data": 123}}
