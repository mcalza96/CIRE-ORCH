from __future__ import annotations

import pytest

from app.agent.models import (
    ReasoningPlan,
    RetrievalDiagnostics,
    ToolCall,
    ToolResult,
)
from app.graph.universal.flow import UniversalReasoningOrchestrator
from app.graph.universal.steps import reflect_node


class _DummyRetriever:
    async def retrieve_chunks(self, *args, **kwargs):
        del args, kwargs
        return []

    async def retrieve_summaries(self, *args, **kwargs):
        del args, kwargs
        return []

    async def validate_scope(self, *args, **kwargs):
        del args, kwargs
        return {}

    def apply_validated_scope(self, *args, **kwargs):
        del args, kwargs


class _DummyAnswer:
    async def generate(self, *args, **kwargs):
        del args, kwargs
        return None


class _DummyValidator:
    def validate(self, *args, **kwargs):
        del args, kwargs
        return None


def _orchestrator() -> UniversalReasoningOrchestrator:
    return UniversalReasoningOrchestrator(
        retriever=_DummyRetriever(),
        answer_generator=_DummyAnswer(),
        validator=_DummyValidator(),
    )


def _state_with_last_tool(last: ToolResult) -> dict[str, object]:
    return {
        "user_query": "q",
        "reasoning_plan": ReasoningPlan(goal="q", steps=[ToolCall(tool=last.tool)]),
        "tool_cursor": 1,
        "reflections": 0,
        "max_reflections": 2,
        "plan_attempts": 1,
        "tool_results": [last],
        "reasoning_steps": [],
    }


@pytest.mark.asyncio
async def test_reflect_does_not_retry_non_retryable_errors() -> None:
    flow = _orchestrator()
    updates = await reflect_node(
        _state_with_last_tool(
            ToolResult(tool="python_calculator", ok=False, error="missing_expression")
        )
    )
    assert updates["next_action"] == "generate"
    assert updates["stop_reason"] == "tool_error_non_retryable"
    assert updates["reflections"] == 0


@pytest.mark.asyncio
async def test_reflect_retries_retryable_errors() -> None:
    flow = _orchestrator()
    updates = await reflect_node(
        _state_with_last_tool(ToolResult(tool="semantic_retrieval", ok=False, error="empty_retrieval"))
    )
    assert updates["next_action"] == "replan"
    assert updates["reflections"] == 1
    assert updates["plan_attempts"] == 2
    assert "[REPLAN_REASON]" in str(updates.get("working_query") or "")


@pytest.mark.asyncio
async def test_reflect_retries_on_strong_retrieval_signal_with_empty_results() -> None:
    flow = _orchestrator()
    state = _state_with_last_tool(
        ToolResult(
            tool="semantic_retrieval",
            ok=True,
            output={"chunk_count": 0, "summary_count": 0},
        )
    )
    state["retrieval"] = RetrievalDiagnostics(contract="advanced", strategy="multi_query_primary")
    updates = await reflect_node(state)
    assert updates["next_action"] == "replan"
    assert updates["reflections"] == 1
    assert updates["plan_attempts"] == 2


@pytest.mark.asyncio
async def test_reflect_retries_on_scope_mismatch_signal() -> None:
    flow = _orchestrator()
    state = _state_with_last_tool(
        ToolResult(
            tool="semantic_retrieval",
            ok=True,
            output={"chunk_count": 3, "summary_count": 1},
        )
    )
    state["retrieval"] = RetrievalDiagnostics(
        contract="advanced",
        strategy="multi_query_primary",
        scope_validation={"valid": False},
    )
    updates = await reflect_node(state)
    assert updates["next_action"] == "replan"
    assert updates["reflections"] == 1
    assert updates["plan_attempts"] == 2
