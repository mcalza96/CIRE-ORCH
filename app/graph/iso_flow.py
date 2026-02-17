from __future__ import annotations

import asyncio
import re
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


def _is_literal_query(text: str) -> bool:
    lowered = (text or "").lower()
    markers = (
        "textualmente",
        "literal",
        "verbatim",
        "transcribe",
        "cita",
        "citas",
        "que exige",
        "qué exige",
    )
    if any(token in lowered for token in markers):
        return True
    return bool(re.search(r"\bcl(?:a|á)usula\s*\d+(?:\.\d+)+\b", lowered))


def _is_list_literal_query(text: str) -> bool:
    lowered = (text or "").lower()
    markers = ("enumera", "lista", "listado", "viñetas", "vinetas")
    return any(token in lowered for token in markers)


def _literal_force_eligible(text: str) -> bool:
    query = str(text or "").strip()
    if not _is_literal_query(query):
        return False
    if len(query) > 220:
        return False
    scopes = re.findall(r"\biso\s*[-:]?\s*\d{4,5}\b", query, flags=re.IGNORECASE)
    if len(scopes) >= 2:
        return False
    if query.count("?") >= 2:
        return False
    return True


def _graph_retry_reason(trace: dict[str, object]) -> str:
    if not isinstance(trace, dict):
        return ""
    rag_features = trace.get("rag_features")
    if isinstance(rag_features, dict):
        fallback_used = bool(rag_features.get("fallback_used", False))
        planner_multihop = bool(rag_features.get("planner_multihop", False))
        if fallback_used and not planner_multihop:
            return "graph_fallback_no_multihop"
    missing_scopes = trace.get("missing_scopes")
    if isinstance(missing_scopes, list) and missing_scopes:
        return "scope_mismatch"
    return ""


def _graph_contract(trace: dict[str, object]) -> dict[str, object]:
    if not isinstance(trace, dict):
        return {
            "graph_used": False,
            "graph_strategy": "none",
            "anchor_count": 0,
            "anchor_entities": [],
            "graph_paths_count": 0,
            "community_hits": 0,
        }

    rag_features_raw = trace.get("rag_features")
    rag_features = rag_features_raw if isinstance(rag_features_raw, dict) else {}
    strategy = "none"
    if bool(rag_features.get("planner_multihop", False)):
        strategy = "hybrid"
    elif bool(rag_features.get("planner_used", False)):
        strategy = "local"
    elif bool(rag_features.get("fallback_used", False)):
        strategy = "global"

    missing_scopes = trace.get("missing_scopes")
    scope_mismatch_reason = ""
    if isinstance(missing_scopes, list) and missing_scopes:
        scope_mismatch_reason = "missing_scopes"

    def _to_int(value: object) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return 0

    return {
        "graph_used": strategy != "none",
        "graph_strategy": strategy,
        "anchor_count": _to_int(trace.get("layer_0")),
        "anchor_entities": trace.get("anchor_entities")
        if isinstance(trace.get("anchor_entities"), list)
        else [],
        "graph_paths_count": _to_int(trace.get("layer_1")),
        "community_hits": _to_int(trace.get("layer_2")),
        "scope_mismatch_reason": scope_mismatch_reason,
    }


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


def _derive_graph_gaps(
    *,
    query: str,
    documents: list[EvidenceItem],
    contract: dict[str, object],
) -> dict[str, object]:
    requested = {
        f"ISO {match}"
        for match in re.findall(r"\biso\s*[-:]?\s*(\d{4,5})\b", query, flags=re.IGNORECASE)
    }
    found: set[str] = set()
    for item in documents:
        scope = extract_row_standard(item)
        if not scope:
            continue
        digits = "".join(ch for ch in scope if ch.isdigit())
        if digits:
            found.add(f"ISO {digits}")

    disconnected_entities = sorted(scope for scope in requested if scope not in found)
    clause_refs = sorted(set(re.findall(r"\b\d+(?:\.\d+)+\b", query or "")))
    missing_links = [
        f"{scope} -> clausula {clause}"
        for scope in disconnected_entities
        for clause in clause_refs[:2]
    ]

    out = dict(contract)
    out["disconnected_entities"] = disconnected_entities
    out["missing_links"] = missing_links[:4]
    return out


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
        original_query = str(state.get("user_query") or "").strip()
        profile = state.get("agent_profile")
        retry_count = int(state.get("retry_count") or 0)
        reason = str(state.get("grade_reason") or "")

        if retry_count > 0:
            prev_plan = state.get("plan")
            if isinstance(prev_plan, RetrievalPlan):
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

                # Mode-specific retry strategy:
                # - literal: keep literal when empty, relax mode on mismatches/low quality.
                # - interpretive/comparative: keep mode and broaden via retry hints.
                if prev_plan.mode in {"literal_normativa", "literal_lista"}:
                    if _literal_force_eligible(original_query):
                        return {
                            "working_query": query,
                            "intent": QueryIntent(
                                mode=prev_plan.mode,
                                rationale=f"retry_force_literal_{reason or 'retry'}",
                            ),
                            "plan": prev_plan,
                        }
                    if reason in {
                        "scope_mismatch",
                        "clause_missing",
                        "low_score",
                        "empty_retrieval",
                    }:
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
        if _literal_force_eligible(original_query):
            forced_mode = (
                "literal_lista" if _is_list_literal_query(original_query) else "literal_normativa"
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
            graph_reason = _graph_retry_reason(retrieval.trace)
            if graph_reason and reason == "ok":
                reason = graph_reason
                good = False
            graph_contract = _derive_graph_gaps(
                query=str(state.get("working_query") or state.get("user_query") or ""),
                documents=documents,
                contract=_graph_contract(retrieval.trace),
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
        contract_state = final_state.get("graph_contract")
        trace["graph_contract"] = (
            dict(contract_state)
            if isinstance(contract_state, dict)
            else _derive_graph_gaps(
                query=cmd.query,
                documents=list(final_state.get("retrieved_documents") or []),
                contract=_graph_contract(trace),
            )
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
