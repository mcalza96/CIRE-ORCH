
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TypedDict, cast

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
from app.agent.components.parsing import build_retry_focus_query
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    QueryIntent,
    QueryMode,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.agent.policies import build_retrieval_plan, classify_intent
from app.cartridges.models import AgentProfile
from app.core.config import settings

# Policy Imports (SOLID Refactor)
from app.agent.policies.query_splitter import QuerySplitter
from app.agent.policies.scope_policy import ScopePolicy
from app.agent.policies.retry_policy import RetryPolicy
from app.agent.policies.query_analysis import (
    is_literal_query,
    is_list_literal_query,
    literal_force_eligible,
)
from app.agent.policies.graph_diagnostics import (
    calculate_graph_retry_reason,
    derive_graph_contract,
    derive_graph_gaps,
)


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
    graph_contract: dict[str, object]


@dataclass
class IsoFlowOrchestrator:
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    max_retries: int = 1
    
    # Injected Policies
    splitter: QuerySplitter = field(default_factory=QuerySplitter)
    scope_policy: ScopePolicy = field(default_factory=ScopePolicy)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

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
        original_query = str(state.get("user_query") or "").strip()
        profile = state.get("agent_profile")
        retry_count = int(state.get("retry_count") or 0)
        reason = str(state.get("grade_reason") or "")

        if retry_count > 0:
            prev_plan = state.get("plan")
            if isinstance(prev_plan, RetrievalPlan):
                # Use Graph Diagnostics policy
                contract = state.get("graph_contract")
                graph_gaps: list[str] = []
                if isinstance(contract, dict):
                    disconnected = contract.get("disconnected_entities")
                    if isinstance(disconnected, list) and disconnected:
                        graph_gaps.append(
                            "desconectadas=" + ", ".join(str(item) for item in disconnected)
                        )
                    missing_links = contract.get("missing_links")
                    if isinstance(missing_links, list) and missing_links:
                        graph_gaps.append(
                            "vinculos=" + "; ".join(str(item) for item in missing_links[:2])
                        )
                        
                query = build_retry_focus_query(
                    query=query,
                    plan=prev_plan,
                    reason=reason or "retry",
                    graph_gaps=graph_gaps,
                )

                # Use Retry Policy
                if literal_force_eligible(original_query) and prev_plan.mode in {"literal_normativa", "literal_lista"}:
                     return {
                        "working_query": query,
                        "intent": QueryIntent(
                            mode=prev_plan.mode,
                            rationale=f"retry_force_literal_{reason or 'retry'}",
                        ),
                        "plan": prev_plan,
                    }
                
                next_intent = self.retry_policy.determine_next_intent(
                    current_plan=prev_plan,
                    reason=reason,
                    profile=profile
                )
                
                if next_intent:
                    plan = build_retrieval_plan(next_intent, query=query, profile=profile)
                    return {
                        "working_query": query,
                        "intent": next_intent,
                        "plan": plan,
                    }

        intent = classify_intent(query, profile=profile)
        
        # Use Query Analysis Policy
        if literal_force_eligible(original_query):
            forced_mode = (
                "literal_lista" if is_list_literal_query(original_query) else "literal_normativa"
            )
            if intent.mode != forced_mode:
                intent = QueryIntent(mode=forced_mode, rationale="graph_literal_override")
                
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
        retrieval = state.get("retrieval")
        
        # Use Graph Diagnostics policy
        graph_contract: dict[str, object] = {
            "graph_used": False,
            "graph_strategy": "none",
            "anchor_count": 0,
            "anchor_entities": [],
            "graph_paths_count": 0,
            "community_hits": 0,
            "scope_mismatch_reason": "",
            "disconnected_entities": [],
            "missing_links": [],
        }
        
        if isinstance(retrieval, RetrievalDiagnostics) and isinstance(retrieval.trace, dict):
            graph_reason = calculate_graph_retry_reason(retrieval.trace)
            if graph_reason and reason == "ok":
                reason = graph_reason
                good = False
            
            base_contract = derive_graph_contract(retrieval.trace)
            graph_contract = derive_graph_gaps(
                query=str(state.get("working_query") or state.get("user_query") or ""),
                documents=documents,
                contract=base_contract,
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
                "graph_contract": graph_contract,
            }

        return {
            "grade_reason": reason,
            "next_step": "generate",
            "graph_contract": graph_contract,
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

    async def _execute_compound(self, cmd: HandleQuestionCommand, parts: list[str]) -> HandleQuestionResult:
        """
        Executes split parts of a compound query in parallel and synthesizes results.
        """
        # Execute each part in parallel using the graph runner
        tasks = []
        for part in parts:
            child_cmd = HandleQuestionCommand(
                query=part,
                tenant_id=cmd.tenant_id,
                collection_id=cmd.collection_id,
                scope_label=cmd.scope_label,
                user_id=cmd.user_id,
                agent_profile=cmd.agent_profile,
                profile_resolution=cmd.profile_resolution,
                request_id=cmd.request_id,
                correlation_id=cmd.correlation_id,
                split_depth=cmd.split_depth + 1,
            )
            tasks.append(self.execute(child_cmd))
            
        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results: list[HandleQuestionResult] = []
        
        for res in sub_results:
            if isinstance(res, HandleQuestionResult):
                if res.clarification is not None:
                     # If any child asks for clarification, bubble it up immediately
                     return res
                valid_results.append(res)
            else:
                logger.error("compound_query_part_failed", error=str(res))

        # Synthesis Logic
        sections = [
            f"**Seccion {idx + 1}**\n{result.answer.text}"
            for idx, result in enumerate(valid_results)
        ]
        combined_text = "\n\n".join(sections)
        
        # Combine Evidence
        combined_evidence: list[EvidenceItem] = []
        seen_ids: set[str] = set()
        for res in valid_results:
             for ev in res.answer.evidence:
                 # Simplistic dedupe
                 eid = str(ev.metadata.get("id") if ev.metadata else "") or ev.content[:32]
                 if eid not in seen_ids:
                     seen_ids.add(eid)
                     combined_evidence.append(ev)

        # Determine Combined Mode
        modes = [res.plan.mode for res in valid_results]
        combined_mode: QueryMode = "explicativa"
        if modes and all(str(m).startswith("literal") for m in modes):
            combined_mode = cast(QueryMode, modes[0])
        elif len(set(modes)) > 1:
            combined_mode = "comparativa"
        elif modes:
             combined_mode = cast(QueryMode, modes[0])

        combined_answer = AnswerDraft(
            text=combined_text,
            mode=combined_mode,
            evidence=combined_evidence,
        )
        
        # We don't have a plan for the combined result, so we synthesize one
        combined_plan = RetrievalPlan(
             mode=combined_mode,
             chunk_k=0, chunk_fetch_k=0, summary_k=0,
             require_literal_evidence=False,
             requested_standards=()
        )
        
        combined_validation = ValidationResult(accepted=True, issues=[])
        
        return HandleQuestionResult(
            intent=QueryIntent(mode=combined_mode),
            plan=combined_plan,
            answer=combined_answer,
            validation=combined_validation,
            retrieval=RetrievalDiagnostics(contract="virtual", strategy="compound", partial=False, trace={})
        )

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
        
        # 1. Compound Query Check (Policy-driven)
        if cmd.split_depth == 0:
            parts = self.splitter.split(cmd.query)
            if len(parts) >= 2:
                return await self._execute_compound(cmd, parts)

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
        contract_state = final_state.get("graph_contract")
        trace["graph_contract"] = (
            dict(contract_state)
            if isinstance(contract_state, dict)
            else derive_graph_contract(trace) 
        )
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
