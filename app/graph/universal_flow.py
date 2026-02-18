from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict, cast

import structlog
from langgraph.graph import END, START, StateGraph

from app.agent.application import (
    AnswerGeneratorPort,
    HandleQuestionCommand,
    HandleQuestionResult,
    RetrieverPort,
    ValidatorPort,
)
from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_EMPTY_RETRIEVAL,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_LOW_SCORE,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
)
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    QueryIntent,
    ReasoningPlan,
    ReasoningStep,
    RetrievalDiagnostics,
    RetrievalPlan,
    ToolResult,
    ValidationResult,
)
from app.agent.policies import classify_intent
from app.agent.tools import (
    AgentTool,
    ToolRuntimeContext,
    create_default_tools,
    get_tool,
    resolve_allowed_tools,
)
from app.cartridges.models import AgentProfile
from app.graph.nodes.universal_planner import build_universal_plan


logger = structlog.get_logger(__name__)

_RETRYABLE_REASONS = {
    RETRIEVAL_CODE_EMPTY_RETRIEVAL,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_LOW_SCORE,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
}

DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_REFLECTIONS = 2
MAX_PLAN_ATTEMPTS = 3
ANSWER_PREVIEW_LIMIT = 180
RETRY_REASON_LIMIT = 120
HARD_MAX_STEPS = 12
HARD_MAX_REFLECTIONS = 6


def _non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return default
        return max(0, parsed)
    if value is None:
        return default
    return default


def _append_stage_timing(
    state: UniversalState | dict[str, object],
    *,
    stage: str,
    elapsed_ms: float,
) -> dict[str, float]:
    current = state.get("stage_timings_ms")
    timings = dict(current) if isinstance(current, dict) else {}
    timings[stage] = round(float(timings.get(stage, 0.0)) + max(0.0, elapsed_ms), 2)
    return timings


def _append_tool_timing(
    state: UniversalState | dict[str, object],
    *,
    tool: str,
    elapsed_ms: float,
) -> dict[str, float]:
    current = state.get("tool_timings_ms")
    timings = dict(current) if isinstance(current, dict) else {}
    timings[tool] = round(float(timings.get(tool, 0.0)) + max(0.0, elapsed_ms), 2)
    return timings


def _is_retryable_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return text in _RETRYABLE_REASONS


def _extract_retry_signal_from_retrieval(state: UniversalState, last: ToolResult | None) -> str:
    if not isinstance(last, ToolResult):
        return ""
    if last.tool != "semantic_retrieval" or not last.ok:
        return ""

    output = dict(last.output or {})
    chunk_count = _non_negative_int(output.get("chunk_count"), default=0)
    summary_count = _non_negative_int(output.get("summary_count"), default=0)
    if chunk_count + summary_count <= 0:
        return RETRIEVAL_CODE_EMPTY_RETRIEVAL

    retrieval = state.get("retrieval")
    if not isinstance(retrieval, RetrievalDiagnostics):
        return ""

    scope_validation = dict(retrieval.scope_validation or {})
    if scope_validation.get("valid") is False:
        return RETRIEVAL_CODE_SCOPE_MISMATCH

    trace = dict(retrieval.trace or {})
    missing_scopes = trace.get("missing_scopes")
    if isinstance(missing_scopes, list) and missing_scopes:
        return RETRIEVAL_CODE_SCOPE_MISMATCH

    missing_clause_refs = trace.get("missing_clause_refs")
    if isinstance(missing_clause_refs, list) and missing_clause_refs:
        return RETRIEVAL_CODE_CLAUSE_MISSING

    error_codes_raw = trace.get("error_codes")
    error_codes = error_codes_raw if isinstance(error_codes_raw, list) else []
    for code in (
        RETRIEVAL_CODE_SCOPE_MISMATCH,
        RETRIEVAL_CODE_CLAUSE_MISSING,
        RETRIEVAL_CODE_LOW_SCORE,
        RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    ):
        if code in error_codes:
            return code
    return ""


class UniversalState(TypedDict):
    user_query: str
    working_query: str
    tenant_id: str
    collection_id: str | None
    user_id: str | None
    request_id: str | None
    correlation_id: str | None
    scope_label: str
    agent_profile: AgentProfile | None
    tool_results: list[ToolResult]
    tool_cursor: int
    plan_attempts: int
    reflections: int
    reasoning_steps: list[ReasoningStep]
    working_memory: dict[str, object]
    chunks: list[EvidenceItem]
    summaries: list[EvidenceItem]
    retrieved_documents: list[EvidenceItem]
    allowed_tools: NotRequired[list[str]]
    intent: NotRequired[object]
    retrieval_plan: NotRequired[RetrievalPlan]
    reasoning_plan: NotRequired[ReasoningPlan]
    max_steps: NotRequired[int]
    max_reflections: NotRequired[int]
    next_action: NotRequired[str]
    stop_reason: NotRequired[str]
    retrieval: NotRequired[RetrievalDiagnostics]
    generation: NotRequired[AnswerDraft]
    validation: NotRequired[ValidationResult]
    stage_timings_ms: NotRequired[dict[str, float]]
    tool_timings_ms: NotRequired[dict[str, float]]


def _clip_text(value: object, limit: int = 280) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            out[key] = _clip_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = {str(k): _clip_text(v) for k, v in value.items()}
        else:
            out[key] = _clip_text(value)
    return out


def _infer_expression_from_query(query: str) -> str:
    text = str(query or "")
    # Conservative extraction: only plain arithmetic expressions.
    match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\(?\d+(?:\.\d+)?\)?)+)", text)
    if not match:
        return ""
    return str(match.group(1)).strip()


@dataclass
class UniversalReasoningOrchestrator:
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, AgentTool] | None = None

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = create_default_tools()
        self._graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(UniversalState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("execute_tool", self._execute_tool_node)
        graph.add_node("reflect", self._reflect_node)
        graph.add_node("generator", self._generator_node)
        graph.add_node("citation_validate", self._citation_validate_node)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {"execute": "execute_tool", "generate": "generator"},
        )
        graph.add_edge("execute_tool", "reflect")
        graph.add_conditional_edges(
            "reflect",
            self._route_after_reflect,
            {
                "replan": "planner",
                "execute_tool": "execute_tool",
                "generate": "generator",
            },
        )
        graph.add_edge("generator", "citation_validate")
        graph.add_edge("citation_validate", END)
        return graph

    def _runtime_context(self) -> ToolRuntimeContext:
        return ToolRuntimeContext(
            retriever=self.retriever,
            answer_generator=self.answer_generator,
            validator=self.validator,
        )

    async def _planner_node(self, state: UniversalState) -> dict[str, object]:
        t0 = time.perf_counter()
        query = str(state.get("working_query") or state.get("user_query") or "").strip()
        profile = state.get("agent_profile")
        allowed_tools = resolve_allowed_tools(profile, self.tools or {})
        intent, retrieval_plan, reasoning_plan, trace_steps = build_universal_plan(
            query=query,
            profile=profile,
            allowed_tools=allowed_tools,
        )

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

    async def _execute_tool_node(self, state: UniversalState) -> dict[str, object]:
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
        tool = get_tool(self.tools or {}, tool_name)
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
            result = await tool.run(
                payload,
                state=dict(state),
                context=self._runtime_context(),
            )
            tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000.0

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
        if result.ok and result.tool == "semantic_retrieval":
            chunks = list(result.metadata.get("chunks") or [])
            summaries = list(result.metadata.get("summaries") or [])
            retrieval = result.metadata.get("retrieval")
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

    async def _reflect_node(self, state: UniversalState) -> dict[str, object]:
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

    async def _generator_node(self, state: UniversalState) -> dict[str, object]:
        t0 = time.perf_counter()
        plan = state.get("retrieval_plan")
        if not isinstance(plan, RetrievalPlan):
            return {
                "stop_reason": "missing_retrieval_plan",
                "stage_timings_ms": _append_stage_timing(
                    state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
                ),
            }
        answer = await self.answer_generator.generate(
            query=str(state.get("user_query") or ""),
            scope_label=str(state.get("scope_label") or ""),
            plan=plan,
            chunks=cast(list[EvidenceItem], list(state.get("chunks") or [])),
            summaries=cast(list[EvidenceItem], list(state.get("summaries") or [])),
            agent_profile=state.get("agent_profile"),
        )
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

    async def _citation_validate_node(self, state: UniversalState) -> dict[str, object]:
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
            tool = get_tool(self.tools or {}, "citation_validator")
            if tool is not None:
                result = await tool.run({}, state=dict(state), context=self._runtime_context())
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
                validation = self.validator.validate(
                    answer, plan, str(state.get("user_query") or "")
                )
        else:
            validation = self.validator.validate(answer, plan, str(state.get("user_query") or ""))

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

    @staticmethod
    def _route_after_planner(state: UniversalState) -> str:
        return "execute" if str(state.get("next_action") or "") == "execute" else "generate"

    @staticmethod
    def _route_after_reflect(state: UniversalState) -> str:
        next_action = str(state.get("next_action") or "")
        if next_action == "replan":
            return "replan"
        if next_action == "execute_tool":
            return "execute_tool"
        return "generate"

    def _build_reasoning_trace(self, state: UniversalState) -> dict[str, Any]:
        steps = [asdict(step) for step in list(state.get("reasoning_steps") or [])]
        tools_used = sorted(
            {
                str(step.get("tool") or "")
                for step in steps
                if isinstance(step, dict) and step.get("tool")
            }
        )
        accepted = None
        validation = state.get("validation")
        if isinstance(validation, ValidationResult):
            accepted = bool(validation.accepted)
        return {
            "engine": "universal_flow",
            "stop_reason": str(state.get("stop_reason") or "unknown"),
            "plan_attempts": int(state.get("plan_attempts") or 1),
            "reflections": int(state.get("reflections") or 0),
            "tools_used": tools_used,
            "steps": steps,
            "stage_timings_ms": dict(state.get("stage_timings_ms") or {}),
            "tool_timings_ms": dict(state.get("tool_timings_ms") or {}),
            "final_confidence": (1.0 if accepted else 0.45 if accepted is False else None),
        }

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        t_total = time.perf_counter()
        set_profile_context = getattr(self.retriever, "set_profile_context", None)
        if callable(set_profile_context):
            try:
                set_profile_context(
                    profile=cmd.agent_profile,
                    profile_resolution=cmd.profile_resolution,
                )
            except Exception:
                logger.warning("universal_flow_set_profile_context_failed", exc_info=True)

        initial_state: UniversalState = {
            "user_query": cmd.query,
            "working_query": cmd.query,
            "tenant_id": cmd.tenant_id,
            "collection_id": cmd.collection_id,
            "user_id": cmd.user_id,
            "request_id": cmd.request_id,
            "correlation_id": cmd.correlation_id,
            "scope_label": cmd.scope_label,
            "agent_profile": cmd.agent_profile,
            "tool_cursor": 0,
            "plan_attempts": 1,
            "reflections": 0,
            "max_steps": DEFAULT_MAX_STEPS,
            "max_reflections": DEFAULT_MAX_REFLECTIONS,
            "next_action": "generate",
            "stop_reason": "",
            "tool_results": [],
            "reasoning_steps": [],
            "working_memory": {},
            "chunks": [],
            "summaries": [],
            "retrieved_documents": [],
        }
        final_state = await self._graph.ainvoke(initial_state)
        stage_timings = dict(final_state.get("stage_timings_ms") or {})
        stage_timings["total"] = round((time.perf_counter() - t_total) * 1000.0, 2)
        final_state["stage_timings_ms"] = stage_timings

        intent = final_state.get("intent")
        retrieval_plan = final_state.get("retrieval_plan")
        answer = final_state.get("generation")
        validation = final_state.get("validation")
        retrieval = final_state.get("retrieval")
        if not isinstance(retrieval, RetrievalDiagnostics):
            retrieval = RetrievalDiagnostics(
                contract="legacy",
                strategy="langgraph_universal_flow",
                partial=False,
                trace={},
            )

        trace = dict(retrieval.trace or {})
        reasoning_trace = self._build_reasoning_trace(cast(UniversalState, final_state))
        trace_timings = dict(trace.get("timings_ms") or {})
        for key, value in dict(stage_timings).items():
            trace_timings[f"universal_{key}"] = value
        trace["timings_ms"] = trace_timings
        trace["reasoning_trace"] = reasoning_trace
        retrieval = RetrievalDiagnostics(
            contract=retrieval.contract,
            strategy="langgraph_universal_flow",
            partial=bool(retrieval.partial),
            trace=trace,
            scope_validation=dict(retrieval.scope_validation or {}),
        )

        if not isinstance(intent, QueryIntent):
            intent = classify_intent(cmd.query, profile=cmd.agent_profile)

        if not isinstance(answer, AnswerDraft):
            if isinstance(retrieval_plan, RetrievalPlan):
                mode = retrieval_plan.mode
            else:
                mode = "explicativa"
            answer = AnswerDraft(text="No tengo evidencia suficiente para responder.", mode=mode)  # type: ignore[arg-type]
        if not isinstance(validation, ValidationResult):
            if isinstance(retrieval_plan, RetrievalPlan):
                validation = self.validator.validate(answer, retrieval_plan, cmd.query)
            else:
                validation = ValidationResult(accepted=False, issues=["missing_retrieval_plan"])

        if not isinstance(retrieval_plan, RetrievalPlan):
            # Conservative fallback compatible with existing contracts.
            retrieval_plan = RetrievalPlan(
                mode=getattr(answer, "mode", "explicativa"),  # type: ignore[arg-type]
                chunk_k=30,
                chunk_fetch_k=120,
                summary_k=5,
            )

        return HandleQuestionResult(
            intent=intent,
            plan=retrieval_plan,
            answer=answer,
            validation=validation,
            retrieval=retrieval,
            clarification=None,
            reasoning_trace=reasoning_trace,
            engine="universal_flow",
        )
