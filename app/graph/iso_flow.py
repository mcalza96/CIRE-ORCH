from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TypedDict

import structlog
from langgraph.graph import END, START, StateGraph

from app.agent.application import (
    AnswerGeneratorPort,
    HandleQuestionCommand,
    HandleQuestionResult,
    RetrieverPort,
    ValidatorPort,
)
from app.agent.components.grading import looks_relevant_retrieval
from app.agent.components.parsing import build_retry_focus_query, extract_row_standard
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.agent.policies import build_retrieval_plan, classify_intent
from app.cartridges.models import AgentProfile
from app.core.config import settings


logger = structlog.get_logger(__name__)


class AgentState(TypedDict, total=False):
    user_query: str
    plan: RetrievalPlan
    retrieved_documents: list[EvidenceItem]
    generation: AnswerDraft
    retry_count: int
    tenant_id: str
    collection_id: str | None
    user_id: str | None
    request_id: str | None
    correlation_id: str | None
    scope_label: str
    agent_profile: AgentProfile | None
    working_query: str
    intent: QueryIntent
    chunks: list[EvidenceItem]
    summaries: list[EvidenceItem]
    retrieval: RetrievalDiagnostics
    validation: ValidationResult
    grade_reason: str
    next_step: str
    max_retries: int


@dataclass
class IsoFlowOrchestrator:
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    max_retries: int = 1

    def __post_init__(self) -> None:
        self._graph = self._build_graph().compile()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("retriever", self._retriever_node)
        graph.add_node("grader", self._grader_node)
        graph.add_node("generator", self._generator_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "grader")
        graph.add_conditional_edges(
            "grader",
            self._route_after_grader,
            {
                "retry": "planner",
                "generate": "generator",
            },
        )
        graph.add_edge("generator", END)
        return graph

    async def _planner_node(self, state: AgentState) -> AgentState:
        query = str(state.get("working_query") or state.get("user_query") or "").strip()
        profile = state.get("agent_profile")
        retry_count = int(state.get("retry_count") or 0)
        reason = str(state.get("grade_reason") or "")

        if retry_count > 0:
            prev_plan = state.get("plan")
            if isinstance(prev_plan, RetrievalPlan):
                query = build_retry_focus_query(
                    query=query,
                    plan=prev_plan,
                    reason=reason or "retry",
                )

                # Mode-specific retry strategy:
                # - literal: keep literal when empty, relax mode on mismatches/low quality.
                # - interpretive/comparative: keep mode and broaden via retry hints.
                if prev_plan.mode in {"literal_normativa", "literal_lista"}:
                    if reason in {"scope_mismatch", "clause_missing", "low_score"}:
                        fallback_mode = (
                            "comparativa"
                            if len(prev_plan.requested_standards) >= 2
                            else "explicativa"
                        )
                        intent = QueryIntent(
                            mode=fallback_mode,
                            rationale=f"retry_relax_from_{prev_plan.mode}",
                        )
                        plan = build_retrieval_plan(intent, query=query, profile=profile)
                        return {
                            "working_query": query,
                            "intent": intent,
                            "plan": plan,
                        }

        intent = classify_intent(query, profile=profile)
        plan = build_retrieval_plan(intent, query=query, profile=profile)
        return {
            "working_query": query,
            "intent": intent,
            "plan": plan,
        }

    async def _retriever_node(self, state: AgentState) -> AgentState:
        plan = state.get("plan")
        if not isinstance(plan, RetrievalPlan):
            raise RuntimeError("Planner must run before retriever")

        query = str(state.get("working_query") or state.get("user_query") or "")
        tenant_id = str(state.get("tenant_id") or "")
        collection_id = state.get("collection_id")
        user_id = state.get("user_id")
        request_id = state.get("request_id")
        correlation_id = state.get("correlation_id")

        chunks_task = asyncio.create_task(
            self.retriever.retrieve_chunks(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                plan=plan,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )
        )
        summaries_task = asyncio.create_task(
            self.retriever.retrieve_summaries(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                plan=plan,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )
        )
        chunks, summaries = await asyncio.gather(chunks_task, summaries_task)
        diagnostics = getattr(self.retriever, "last_retrieval_diagnostics", None)
        retrieval = (
            diagnostics
            if isinstance(diagnostics, RetrievalDiagnostics)
            else RetrievalDiagnostics(contract="legacy", strategy="graph", partial=False, trace={})
        )

        return {
            "chunks": list(chunks),
            "summaries": list(summaries),
            "retrieved_documents": [*chunks, *summaries],
            "retrieval": retrieval,
        }

    async def _grader_node(self, state: AgentState) -> AgentState:
        plan = state.get("plan")
        if not isinstance(plan, RetrievalPlan):
            return {"next_step": "generate", "grade_reason": "missing_plan"}

        documents = state.get("retrieved_documents") or []
        good, reason = looks_relevant_retrieval(
            documents,
            plan,
            query=str(state.get("working_query") or state.get("user_query") or ""),
        )
        retry_count = int(state.get("retry_count") or 0)
        max_retries = int(state.get("max_retries") or self.max_retries)

        if not good and retry_count < max_retries:
            logger.info(
                "iso_flow_retry_triggered",
                reason=reason,
                retry_count=retry_count,
                max_retries=max_retries,
            )
            return {
                "retry_count": retry_count + 1,
                "grade_reason": reason,
                "next_step": "retry",
            }

        return {
            "grade_reason": reason,
            "next_step": "generate",
        }

    async def _generator_node(self, state: AgentState) -> AgentState:
        plan = state.get("plan")
        if not isinstance(plan, RetrievalPlan):
            raise RuntimeError("Planner must run before generator")

        answer = await self.answer_generator.generate(
            query=str(state.get("user_query") or ""),
            scope_label=str(state.get("scope_label") or ""),
            plan=plan,
            chunks=list(state.get("chunks") or []),
            summaries=list(state.get("summaries") or []),
            agent_profile=state.get("agent_profile"),
        )
        validation = self.validator.validate(answer, plan, str(state.get("user_query") or ""))
        return {
            "generation": answer,
            "validation": validation,
        }

    @staticmethod
    def _route_after_grader(state: AgentState) -> str:
        return "retry" if str(state.get("next_step") or "") == "retry" else "generate"

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        set_profile_context = getattr(self.retriever, "set_profile_context", None)
        if callable(set_profile_context):
            try:
                set_profile_context(
                    profile=cmd.agent_profile,
                    profile_resolution=cmd.profile_resolution,
                )
            except Exception:
                logger.warning("iso_flow_set_profile_context_failed", exc_info=True)

        initial_state: AgentState = {
            "user_query": cmd.query,
            "working_query": cmd.query,
            "retry_count": 0,
            "max_retries": max(0, int(self.max_retries)),
            "tenant_id": cmd.tenant_id,
            "collection_id": cmd.collection_id,
            "user_id": cmd.user_id,
            "request_id": cmd.request_id,
            "correlation_id": cmd.correlation_id,
            "scope_label": cmd.scope_label,
            "agent_profile": cmd.agent_profile,
            "retrieved_documents": [],
            "chunks": [],
            "summaries": [],
        }

        final_state = await self._graph.ainvoke(initial_state)

        intent = final_state.get("intent")
        plan = final_state.get("plan")
        generation = final_state.get("generation")
        validation = final_state.get("validation")
        retrieval = final_state.get("retrieval")

        if not isinstance(intent, QueryIntent):
            intent = classify_intent(cmd.query, profile=cmd.agent_profile)
        if not isinstance(plan, RetrievalPlan):
            plan = build_retrieval_plan(intent, query=cmd.query, profile=cmd.agent_profile)
        if not isinstance(generation, AnswerDraft):
            generation = AnswerDraft(
                text="No tengo evidencia suficiente para responder.", mode=plan.mode
            )
        if not isinstance(validation, ValidationResult):
            validation = self.validator.validate(generation, plan, cmd.query)
        if not isinstance(retrieval, RetrievalDiagnostics):
            retrieval = RetrievalDiagnostics(
                contract="legacy", strategy="graph", partial=False, trace={}
            )

        trace = dict(retrieval.trace or {})
        trace["graph"] = {
            "name": "iso_flow",
            "retry_count": int(final_state.get("retry_count") or 0),
            "max_retries": int(final_state.get("max_retries") or self.max_retries),
            "grade_reason": str(final_state.get("grade_reason") or ""),
            "working_query": str(final_state.get("working_query") or cmd.query),
        }
        retrieval = RetrievalDiagnostics(
            contract=retrieval.contract,
            strategy="langgraph_iso_flow",
            partial=bool(retrieval.partial),
            trace=trace,
            scope_validation=dict(retrieval.scope_validation or {}),
        )

        return HandleQuestionResult(
            intent=intent,
            plan=plan,
            answer=generation,
            validation=validation,
            retrieval=retrieval,
            clarification=None,
        )
