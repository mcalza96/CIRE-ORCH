from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast

import structlog
from langgraph.graph import END, START, StateGraph

from app.agent.application import (
    AnswerGeneratorPort,
    HandleQuestionCommand,
    HandleQuestionResult,
    RetrieverPort,
    ValidatorPort,
)
from app.agent.models import (
    AnswerDraft,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.agent.policies import classify_intent
from app.agent.tools import ToolRuntimeContext, create_default_tools
from app.core.config import settings
from app.graph.universal.steps import (
    aggregate_subqueries_node,
    citation_validate_node,
    execute_tool_node,
    generator_node,
    planner_node,
    reflect_node,
    route_after_planner,
    route_after_reflect,
)
from app.graph.universal.state import UniversalState
from app.graph.universal.trace import build_reasoning_trace

logger = structlog.get_logger(__name__)


@dataclass
class UniversalReasoningOrchestrator:
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.tools is None:
            self.tools = create_default_tools()
        self._graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        orch = self  # capture for closures

        async def _planner(s: UniversalState) -> dict[str, object]:
            return await planner_node(s, orch)

        async def _execute_tool(s: UniversalState) -> dict[str, object]:
            return await execute_tool_node(s, orch)

        async def _generator(s: UniversalState) -> dict[str, object]:
            return await generator_node(s, orch)

        async def _aggregate_subqueries(s: UniversalState) -> dict[str, object]:
            return await aggregate_subqueries_node(s)

        async def _citation_validate(s: UniversalState) -> dict[str, object]:
            return await citation_validate_node(s, orch)

        graph = StateGraph(UniversalState)
        graph.add_node("planner", _planner)
        graph.add_node("execute_tool", _execute_tool)
        graph.add_node("reflect", reflect_node)
        graph.add_node("aggregate_subqueries", _aggregate_subqueries)
        graph.add_node("generator", _generator)
        graph.add_node("citation_validate", _citation_validate)

        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner",
            route_after_planner,
            {"execute": "execute_tool", "generate": "aggregate_subqueries"},
        )
        graph.add_edge("execute_tool", "reflect")
        graph.add_conditional_edges(
            "reflect",
            route_after_reflect,
            {
                "replan": "planner",
                "execute_tool": "execute_tool",
                "generate": "aggregate_subqueries",
            },
        )
        graph.add_edge("aggregate_subqueries", "generator")
        graph.add_edge("generator", "citation_validate")
        graph.add_edge("citation_validate", END)
        return graph

    def _runtime_context(self) -> ToolRuntimeContext:
        return ToolRuntimeContext(
            retriever=self.retriever,
            answer_generator=self.answer_generator,
            validator=self.validator,
        )

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
            "scope_label": (cmd.scope_label or ""),
            "agent_profile": cmd.agent_profile,
            "tool_results": [],
            "tool_cursor": 0,
            "plan_attempts": 1,
            "reflections": 0,
            "reasoning_steps": [],
            "working_memory": {},
            "chunks": [],
            "summaries": [],
            "retrieved_documents": [],
            "subquery_groups": [],
            "partial_answers": [],
            "flow_start_pc": time.perf_counter(),
        }
        total_timeout_ms = max(200, int(getattr(settings, "ORCH_TIMEOUT_TOTAL_MS", 60000) or 60000))
        try:
            final_state = await asyncio.wait_for(
                self._graph.ainvoke(initial_state),
                timeout=total_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            final_state = {
                **initial_state,
                "stop_reason": "orchestrator_timeout",
                "validation": ValidationResult(accepted=False, issues=["orchestrator_timeout"]),
                "stage_timings_ms": {"total": round((time.perf_counter() - t_total) * 1000.0, 2)},
            }

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
                contract="advanced",
                strategy="langgraph_universal_flow",
                partial=False,
                trace={},
            )

        trace = dict(retrieval.trace or {})
        reasoning_trace = build_reasoning_trace(cast(UniversalState, final_state))
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
            mode = (
                retrieval_plan.mode if isinstance(retrieval_plan, RetrievalPlan) else "explicativa"
            )
            answer = AnswerDraft(text="No tengo evidencia suficiente para responder.", mode=mode)

        if not isinstance(validation, ValidationResult):
            if isinstance(retrieval_plan, RetrievalPlan):
                validation = self.validator.validate(answer, retrieval_plan, cmd.query)
            else:
                validation = ValidationResult(accepted=False, issues=["missing_retrieval_plan"])

        if not isinstance(retrieval_plan, RetrievalPlan):
            retrieval_plan = RetrievalPlan(
                mode=getattr(answer, "mode", "explicativa"),
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
