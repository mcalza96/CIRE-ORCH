from __future__ import annotations

import structlog

from app.agent.types.models import ReasoningPlan, ReasoningStep, ToolResult
from app.graph.universal.logic import _extract_retry_signal_from_retrieval, _is_retryable_reason
from app.graph.universal.state import (
    DEFAULT_MAX_REFLECTIONS,
    HARD_MAX_REFLECTIONS,
    MAX_PLAN_ATTEMPTS,
    RETRY_REASON_LIMIT,
    UniversalState,
)
from app.graph.universal.utils import (
    state_get_int,
    state_get_list,
    state_get_str,
    track_node_timing,
)

logger = structlog.get_logger(__name__)


@track_node_timing("reflect")
async def reflect_node(state: UniversalState) -> dict[str, object]:
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
        }

    cursor = state_get_int(state, "tool_cursor", 0)
    reflections = state_get_int(state, "reflections", 0)
    max_reflections = min(
        HARD_MAX_REFLECTIONS,
        int(state.get("max_reflections") or DEFAULT_MAX_REFLECTIONS),
    )
    plan_attempts = int(state.get("plan_attempts") or 1)
    tool_results = state_get_list(state, "tool_results")
    last = tool_results[-1] if tool_results else None
    trace_steps = state_get_list(state, "reasoning_steps")

    next_action = "generate"
    stop_reason = state_get_str(state, "stop_reason", "")
    retry_reason = ""
    retryable = False
    if isinstance(last, ToolResult) and not last.ok:
        retry_reason = str(last.error or "")
        retryable = _is_retryable_reason(retry_reason)
        if retryable and reflections < max_reflections and plan_attempts < MAX_PLAN_ATTEMPTS:
            reflections += 1
            plan_attempts += 1
            next_action = "replan"
        else:
            next_action = "generate"
            if not stop_reason:
                stop_reason = (
                    "tool_error_unrecoverable" if retryable else "tool_error_non_retryable"
                )
    elif isinstance(last, ToolResult) and last.ok and cursor >= len(plan.steps):
        retry_reason = _extract_retry_signal_from_retrieval(state, last)
        retryable = _is_retryable_reason(retry_reason)
        if retryable and reflections < max_reflections and plan_attempts < MAX_PLAN_ATTEMPTS:
            reflections += 1
            plan_attempts += 1
            next_action = "replan"
    elif cursor < len(plan.steps):
        next_action = "execute_tool"
    else:
        next_action = "generate"

    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="reflection",
            description="reflection_decision",
            output={
                "next_action": next_action,
                "plan_attempts": plan_attempts,
                "reflections": reflections,
                "last_tool_ok": bool(last.ok) if isinstance(last, ToolResult) else True,
                "retryable": retryable,
                "retry_reason": (retry_reason[:RETRY_REASON_LIMIT] if retry_reason else ""),
            },
            ok=True,
        )
    )

    logger.warning(
        "DIAG_reflect_decision",
        next_action=next_action,
        retry_reason=retry_reason[:80] if retry_reason else "",
        retryable=retryable,
        reflections=reflections,
        plan_attempts=plan_attempts,
        cursor=cursor,
        plan_steps=len(plan.steps),
        last_tool_ok=bool(last.ok) if isinstance(last, ToolResult) else None,
        last_tool_name=last.tool if isinstance(last, ToolResult) else None,
    )
    updates: dict[str, object] = {
        "next_action": next_action,
        "plan_attempts": plan_attempts,
        "reflections": reflections,
        "reasoning_steps": trace_steps,
    }
    if stop_reason:
        updates["stop_reason"] = stop_reason
    if next_action == "replan":
        reason = retry_reason or (str(last.error) if isinstance(last, ToolResult) else "retry")
        # Store the retry reason in working_memory, NOT in the query string.
        # Injecting metadata into working_query poisons the embedding space.
        memory = dict(state.get("working_memory") or {})
        memory["last_replan_reason"] = reason[:RETRY_REASON_LIMIT]
        updates["working_memory"] = memory
        # Always reset working_query to the clean user query.
        updates["working_query"] = str(state.get("user_query") or "")
    return updates
