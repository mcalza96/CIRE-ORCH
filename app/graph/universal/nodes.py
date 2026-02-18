from __future__ import annotations

import asyncio
import time
from typing import Any, cast, Protocol

import structlog

from app.agent.application import AnswerGeneratorPort, RetrieverPort, ValidatorPort
from app.agent.error_codes import RETRIEVAL_CODE_EMPTY_RETRIEVAL
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    ReasoningPlan,
    ReasoningStep,
    RetrievalDiagnostics,
    RetrievalPlan,
    ToolResult,
    ValidationResult,
)
from app.agent.tools import AgentTool, ToolRuntimeContext, get_tool, resolve_allowed_tools
from app.cartridges.models import AgentProfile
from app.core.config import settings
from app.graph.nodes.universal_planner import build_universal_plan
from app.graph.universal.state import (
    ANSWER_PREVIEW_LIMIT,
    DEFAULT_MAX_REFLECTIONS,
    DEFAULT_MAX_STEPS,
    HARD_MAX_REFLECTIONS,
    HARD_MAX_STEPS,
    MAX_PLAN_ATTEMPTS,
    RETRY_REASON_LIMIT,
    UniversalState,
)
from app.graph.universal.utils import (
    _append_stage_timing,
    _append_tool_timing,
    _clip_text,
    _effective_execute_tool_timeout_ms,
    _sanitize_payload,
    _timeout_ms_for_stage,
    get_adaptive_timeout_ms,
)
from app.graph.universal.logic import (
    _extract_retry_signal_from_retrieval,
    _infer_expression_from_query,
    _is_retryable_reason,
)

logger = structlog.get_logger(__name__)


class OrchestratorComponents(Protocol):
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, AgentTool] | None

    def _runtime_context(self) -> ToolRuntimeContext: ...


async def planner_node(
    state: UniversalState, 
    components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    query = str(state.get("working_query") or state.get("user_query") or "").strip()
    profile = state.get("agent_profile")
    allowed_tools = resolve_allowed_tools(profile, components.tools or {})

    # Use adaptive timeout for planner
    base_planner_ms = int(getattr(settings, "ORCH_TIMEOUT_CLASSIFY_MS", 2000) or 2000) + int(
        getattr(settings, "ORCH_TIMEOUT_PLAN_MS", 3000) or 3000
    )
    planner_timeout_ms = get_adaptive_timeout_ms(
        state, stage_default_ms=base_planner_ms, headroom_ms=3000
    )

    try:
        intent, retrieval_plan, reasoning_plan, trace_steps = await asyncio.wait_for(
            asyncio.to_thread(
                build_universal_plan,
                query=query,
                profile=profile,
                allowed_tools=allowed_tools,
            ),
            timeout=planner_timeout_ms / 1000.0,
        )
    except TimeoutError:
        return {
            "next_action": "generate",
            "stop_reason": "planner_timeout",
            "stage_timings_ms": _append_stage_timing(
                state, stage="planner", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    max_steps = DEFAULT_MAX_STEPS
    max_reflections = DEFAULT_MAX_REFLECTIONS
    if isinstance(profile, AgentProfile):
        max_steps = min(HARD_MAX_STEPS, int(profile.capabilities.reasoning_budget.max_steps))
        max_reflections = min(
            HARD_MAX_REFLECTIONS,
            int(profile.capabilities.reasoning_budget.max_reflections),
        )
    reasoning_plan = ReasoningPlan(
        goal=reasoning_plan.goal,
        steps=list(reasoning_plan.steps[: max(1, max_steps)]),
        complexity=reasoning_plan.complexity,
    )
    existing_steps = list(state.get("reasoning_steps") or [])
    updates: dict[str, object] = {
        "intent": intent,
        "retrieval_plan": retrieval_plan,
        "reasoning_plan": reasoning_plan,
        "allowed_tools": allowed_tools,
        "max_steps": max_steps,
        "max_reflections": max_reflections,
        "tool_cursor": 0,
        "reasoning_steps": [*existing_steps, *trace_steps],
        "next_action": ("execute" if reasoning_plan.steps else "generate"),
    }
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="planner", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    return updates


async def execute_tool_node(
    state: UniversalState, 
    components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    cursor = int(state.get("tool_cursor") or 0)
    if cursor >= len(plan.steps):
        return {
            "next_action": "generate",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    max_steps = min(HARD_MAX_STEPS, int(state.get("max_steps") or DEFAULT_MAX_STEPS))
    tool_results = list(state.get("tool_results") or [])
    if len(tool_results) >= max_steps:
        return {
            "next_action": "generate",
            "stop_reason": "max_steps_reached",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    step_call = plan.steps[cursor]
    tool_name = str(step_call.tool or "").strip()
    tool = get_tool(components.tools or {}, tool_name)
    tool_elapsed_ms = 0.0
    if tool is None:
        result = ToolResult(tool=tool_name, ok=False, error="tool_not_registered")
    else:
        payload = dict(step_call.input or {})
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

        # Apply adaptive budget to tool execution
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
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            result = ToolResult(tool=tool_name, ok=False, error=f"tool_error: {str(e)}")
        tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000.0

    # --- DIAGNOSTIC LOG (remove after investigation) ---
    if tool_name == "semantic_retrieval":
        _diag_chunks = list(result.metadata.get("chunks") or []) if result.metadata else []
        logger.warning(
            "DIAG_semantic_retrieval_result",
            ok=result.ok,
            error=result.error,
            chunk_count=len(_diag_chunks),
            output_keys=list((result.output or {}).keys()),
            metadata_keys=list((result.metadata or {}).keys()),
            tool_elapsed_ms=round(tool_elapsed_ms, 2),
        )
    # --- END DIAGNOSTIC ---

    updated_results = [*tool_results, result]
    trace_steps = list(state.get("reasoning_steps") or [])
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
        retrieval = result.metadata.get("retrieval")
        if chunks or summaries:
            updates["chunks"] = chunks
            updates["summaries"] = summaries
            updates["retrieved_documents"] = [*chunks, *summaries]
        if isinstance(retrieval, RetrievalDiagnostics):
            updates["retrieval"] = retrieval
    elif result.ok:
        memory = dict(state.get("working_memory") or {})
        memory[result.tool] = dict(result.output or {})
        updates["working_memory"] = memory
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    if tool_name:
        updates["tool_timings_ms"] = _append_tool_timing(
            state,
            tool=tool_name,
            elapsed_ms=tool_elapsed_ms,
        )
    return updates


async def reflect_node(state: UniversalState) -> dict[str, object]:
    t0 = time.perf_counter()
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
            "stage_timings_ms": _append_stage_timing(
                state, stage="reflect", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    cursor = int(state.get("tool_cursor") or 0)
    reflections = int(state.get("reflections") or 0)
    max_reflections = min(
        HARD_MAX_REFLECTIONS,
        int(state.get("max_reflections") or DEFAULT_MAX_REFLECTIONS),
    )
    plan_attempts = int(state.get("plan_attempts") or 1)
    tool_results = list(state.get("tool_results") or [])
    last = tool_results[-1] if tool_results else None
    trace_steps = list(state.get("reasoning_steps") or [])

    next_action = "generate"
    stop_reason = str(state.get("stop_reason") or "")
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
        updates["working_query"] = (
            str(state.get("user_query") or "")
            + f"\n\n[REPLAN_REASON] {reason[:RETRY_REASON_LIMIT]}"
        )
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="reflect", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    return updates


async def generator_node(
    state: UniversalState, 
    components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    plan = state.get("retrieval_plan")
    logger.warning(
        "DIAG_generator_entry",
        has_plan=isinstance(plan, RetrievalPlan),
        chunks_in_state=len(list(state.get("chunks") or [])),
        summaries_in_state=len(list(state.get("summaries") or [])),
    )
    if not isinstance(plan, RetrievalPlan):
        return {
            "stop_reason": "missing_retrieval_plan",
            "stage_timings_ms": _append_stage_timing(
                state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }
    chunks = cast(list[EvidenceItem], list(state.get("chunks") or []))
    summaries = cast(list[EvidenceItem], list(state.get("summaries") or []))

    working_memory = dict(state.get("working_memory") or {})
    expectation_data = working_memory.get("expectation_coverage")
    if isinstance(expectation_data, dict):
        covered = list(expectation_data.get("covered") or [])
        missing = list(expectation_data.get("missing") or [])
        ratio = expectation_data.get("coverage_ratio")
        synthetic_lines = [
            "[EXPECTATION_COVERAGE]",
            f"coverage_ratio={ratio}",
            f"covered={len(covered)}",
            f"missing={len(missing)}",
        ]
        for row in missing[:6]:
            if not isinstance(row, dict):
                continue
            eid = str(row.get("id") or "expectation")
            risk = str(row.get("missing_risk") or "").strip()
            reason = str(row.get("reason") or "").strip()
            synthetic_lines.append(f"- missing:{eid} risk={risk} reason={reason}")
        summaries = [
            *summaries,
            EvidenceItem(
                source="R999",
                content="\n".join(synthetic_lines),
                score=1.0,
                metadata={"row": {"content": "\n".join(synthetic_lines), "metadata": {}}},
            ),
        ]

    try:
        generator_timeout_ms = get_adaptive_timeout_ms(
            state,
            stage_default_ms=_timeout_ms_for_stage("generator"),
            headroom_ms=1000,
        )
        answer = await asyncio.wait_for(
            components.answer_generator.generate(
                query=str(state.get("user_query") or ""),
                scope_label=str(state.get("scope_label") or ""),
                plan=plan,
                chunks=chunks,
                summaries=summaries,
                agent_profile=state.get("agent_profile"),
            ),
            timeout=generator_timeout_ms / 1000.0,
        )
    except TimeoutError:
        return {
            "stop_reason": "generator_timeout",
            "stage_timings_ms": _append_stage_timing(
                state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }
    trace_steps = list(state.get("reasoning_steps") or [])
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="synthesis",
            description="synthesis_completed",
            output={
                "answer_preview": _clip_text(answer.text, limit=ANSWER_PREVIEW_LIMIT),
                "evidence_count": len(answer.evidence),
            },
        )
    )
    return {
        "generation": answer,
        "reasoning_steps": trace_steps,
        "stage_timings_ms": _append_stage_timing(
            state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
        ),
    }


async def citation_validate_node(
    state: UniversalState, 
    components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    answer = state.get("generation")
    plan = state.get("retrieval_plan")
    if not isinstance(answer, AnswerDraft) or not isinstance(plan, RetrievalPlan):
        return {
            "validation": ValidationResult(
                accepted=False, issues=["missing_generation_or_plan"]
            ),
            "stop_reason": "validation_failed",
            "stage_timings_ms": _append_stage_timing(
                state, stage="validation", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    allowed = list(state.get("allowed_tools") or [])
    if "citation_validator" in allowed:
        tool = get_tool(components.tools or {}, "citation_validator")
        if tool is not None:
            try:
                validation_timeout_ms = get_adaptive_timeout_ms(
                    state,
                    stage_default_ms=_timeout_ms_for_stage("validation"),
                    headroom_ms=200,
                )
                result = await asyncio.wait_for(
                    tool.run({}, state=dict(state), context=components._runtime_context()),
                    timeout=validation_timeout_ms / 1000.0,
                )
            except TimeoutError:
                result = ToolResult(tool="citation_validator", ok=False, error="tool_timeout")
            accepted = (
                bool(result.output.get("accepted"))
                if isinstance(result.output, dict)
                else bool(result.ok)
            )
            issues = (
                list(result.output.get("issues") or [])
                if isinstance(result.output, dict)
                else ([result.error] if result.error else [])
            )
            validation = ValidationResult(accepted=accepted, issues=issues)
        else:
            validation = components.validator.validate(
                answer, plan, str(state.get("user_query") or "")
            )
    else:
        validation = components.validator.validate(answer, plan, str(state.get("user_query") or ""))

    trace_steps = list(state.get("reasoning_steps") or [])
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="validation",
            tool="citation_validator" if "citation_validator" in allowed else None,
            description="validation_completed",
            output={"accepted": bool(validation.accepted), "issues": list(validation.issues)},
            ok=bool(validation.accepted),
        )
    )
    stop_reason = str(state.get("stop_reason") or "")
    if not stop_reason:
        stop_reason = "done" if validation.accepted else "validation_failed"
    return {
        "validation": validation,
        "reasoning_steps": trace_steps,
        "stop_reason": stop_reason,
        "stage_timings_ms": _append_stage_timing(
            state, stage="validation", elapsed_ms=(time.perf_counter() - t0) * 1000.0
        ),
    }


def route_after_planner(state: UniversalState) -> str:
    return "execute" if str(state.get("next_action") or "") == "execute" else "generate"


def route_after_reflect(state: UniversalState) -> str:
    next_action = str(state.get("next_action") or "")
    if next_action == "replan":
        return "replan"
    if next_action == "execute_tool":
        return "execute_tool"
    return "generate"
