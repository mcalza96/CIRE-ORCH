from __future__ import annotations

import asyncio
import time

import structlog

from app.agent.models import ReasoningPlan, ReasoningStep, RetrievalDiagnostics, ToolResult
from app.agent.tools import get_tool
from app.cartridges.models import AgentProfile
from app.graph.universal.logic import _infer_expression_from_query
from app.graph.universal.state import (
    DEFAULT_MAX_STEPS,
    HARD_MAX_STEPS,
    UniversalState,
)
from app.graph.universal.utils import (
    _append_tool_timing,
    _effective_execute_tool_timeout_ms,
    _sanitize_payload,
    get_adaptive_timeout_ms,
    state_get_dict,
    state_get_int,
    state_get_list,
    track_node_timing,
)

from .types import OrchestratorComponents

logger = structlog.get_logger(__name__)


@track_node_timing("execute_tool")
async def execute_tool_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
        }

    cursor = state_get_int(state, "tool_cursor", 0)
    if cursor >= len(plan.steps):
        return {
            "next_action": "generate",
        }

    max_steps = min(HARD_MAX_STEPS, int(state.get("max_steps") or DEFAULT_MAX_STEPS))
    tool_results = state_get_list(state, "tool_results")
    if len(tool_results) >= max_steps:
        return {
            "next_action": "generate",
            "stop_reason": "max_steps_reached",
        }

    step_call = plan.steps[cursor]
    tool_name = str(step_call.tool or "").strip()
    tool = get_tool(components.tools or {}, tool_name)
    tool_elapsed_ms = 0.0
    if tool is None:
        result = ToolResult(tool=tool_name, ok=False, error="tool_not_registered")
    else:
        payload = dict(step_call.input or {})
        if cursor > 0 and tool_results:
            prev = tool_results[-1]
            if isinstance(prev, ToolResult) and prev.ok:
                if prev.output:
                    payload.setdefault("previous_tool_output", dict(prev.output))
                if prev.metadata:
                    payload.setdefault("previous_tool_metadata", dict(prev.metadata))

        working_memory = dict(state_get_dict(state, "working_memory"))
        if working_memory:
            payload["working_memory"] = working_memory

        if tool_name == "python_calculator" and not payload.get("expression"):
            inferred = _infer_expression_from_query(str(state.get("working_query") or ""))
            if inferred:
                payload["expression"] = inferred
        t_tool = time.perf_counter()
        tool_timeout_ms = _effective_execute_tool_timeout_ms(tool_name)
        profile = state.get("agent_profile")
        if isinstance(profile, AgentProfile):
            policy = profile.capabilities.tool_policies.get(tool_name)
            if policy is not None:
                tool_timeout_ms = max(20, min(5000, int(policy.timeout_ms)))

        tool_timeout_ms = get_adaptive_timeout_ms(
            state, stage_default_ms=tool_timeout_ms, headroom_ms=2800
        )

        try:
            result = await asyncio.wait_for(
                tool.run(
                    payload,
                    state=dict(state),
                    context=components._runtime_context(),
                ),
                timeout=tool_timeout_ms / 1000.0,
            )
        except TimeoutError:
            result = ToolResult(tool=tool_name, ok=False, error="tool_timeout")
        except Exception as exc:
            logger.error("tool_execution_failed", tool=tool_name, error=str(exc))
            result = ToolResult(tool=tool_name, ok=False, error=f"tool_error: {str(exc)}")
        tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000.0

    if tool_name == "semantic_retrieval":
        diag_chunks = list(result.metadata.get("chunks") or []) if result.metadata else []
        logger.warning(
            "DIAG_semantic_retrieval_result",
            ok=result.ok,
            error=result.error,
            chunk_count=len(diag_chunks),
            output_keys=list((result.output or {}).keys()),
            metadata_keys=list((result.metadata or {}).keys()),
            tool_elapsed_ms=round(tool_elapsed_ms, 2),
        )

    updated_results = [*tool_results, result]
    trace_steps = state_get_list(state, "reasoning_steps")
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="tool",
            tool=tool_name,
            description=step_call.rationale or "tool_execution",
            input=_sanitize_payload(dict(step_call.input or {})),
            output={
                **_sanitize_payload(dict(result.output or {})),
                "duration_ms": round(tool_elapsed_ms, 2),
            },
            ok=bool(result.ok),
            error=result.error,
        )
    )

    updates: dict[str, object] = {
        "tool_results": updated_results,
        "tool_cursor": cursor + 1,
        "reasoning_steps": trace_steps,
    }
    if result.tool == "semantic_retrieval" and result.metadata:
        chunks = list(result.metadata.get("chunks") or [])
        summaries = list(result.metadata.get("summaries") or [])
        subquery_groups = list(result.metadata.get("subquery_groups") or [])
        retrieval = result.metadata.get("retrieval")

        existing_chunks = state_get_list(state, "chunks")
        existing_summaries = state_get_list(state, "summaries")
        existing_docs = state_get_list(state, "retrieved_documents")
        existing_subq = state_get_list(state, "subquery_groups")

        if chunks or summaries:
            updates["chunks"] = [*existing_chunks, *chunks]
            updates["summaries"] = [*existing_summaries, *summaries]
            updates["retrieved_documents"] = [*existing_docs, *chunks, *summaries]
        if subquery_groups:
            valid_groups = [group for group in subquery_groups if isinstance(group, dict)]
            updates["subquery_groups"] = [*existing_subq, *valid_groups]
        if isinstance(retrieval, RetrievalDiagnostics):
            updates["retrieval"] = retrieval
    elif result.ok:
        memory = dict(state_get_dict(state, "working_memory"))
        memory[result.tool] = dict(result.output or {})
        updates["working_memory"] = memory
    if tool_name:
        updates["tool_timings_ms"] = _append_tool_timing(
            state,
            tool=tool_name,
            elapsed_ms=tool_elapsed_ms,
        )
    return updates
