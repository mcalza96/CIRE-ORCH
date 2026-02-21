from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

import httpx
import structlog

from app.agent.error_codes import (
    RETRIEVAL_CODE_INVALID_RESPONSE,
    RETRIEVAL_CODE_TIMEOUT,
    RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE,
)
from app.agent.interfaces import (
    EmbeddingProvider,
    RerankingProvider,
    SubqueryPlanningContext,
    SubqueryPlanner,
)
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig
from app.agent.retrieval_planner import (
    mode_requires_literal_evidence,
    normalize_query_filters,
)
from app.infrastructure.config import settings
from app.domain.rag_schemas import RetrievalPlanPayload
from app.infrastructure.clients.rag_client import RagRetrievalContractClient
from app.agent.retrieval_strategies import (
    calculate_layer_stats,
    features_from_hybrid_trace,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievalExecutionContext:
    filters: dict[str, Any] | None
    query: str
    context_volume: str
    require_all_scopes: bool
    min_clause_refs_required: int
    k: int
    fetch_k: int
    timeout_comprehensive_ms: int


class RetrievalTrace(TypedDict, total=False):
    error_codes: list[str]
    search_hint_expansions: dict[str, Any]
    agent_profile_resolution: dict[str, Any]
    strategy_path: str
    timings_ms: dict[str, float]
    rag_features: dict[str, Any]
    collection_scope: dict[str, Any]
    scope_balance: dict[str, Any]
    empty_result_backstop: dict[str, Any]
    min_score_filter: dict[str, Any]


class RetrievalFlow:
    def __init__(
        self,
        contract_client: RagRetrievalContractClient,
        subquery_planner: SubqueryPlanner | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranking_provider: RerankingProvider | None = None,
        profile_context: AgentProfile | None = None,
        profile_resolution_context: dict[str, Any] | None = None,
    ):
        self.contract_client = contract_client
        self.subquery_planner = subquery_planner
        self.embedding_provider = embedding_provider
        self.reranking_provider = reranking_provider
        self.profile_context = profile_context
        self.profile_resolution_context = profile_resolution_context
        self.last_diagnostics: RetrievalDiagnostics | None = None

    def _mode_config(self, mode: str) -> QueryModeConfig | None:
        if self.profile_context is None:
            return None
        cfg = self.profile_context.query_modes.modes.get(str(mode or "").strip())
        return cfg if isinstance(cfg, QueryModeConfig) else None

    def _prepare_execution_context(
        self,
        *,
        query: str,
        plan: RetrievalPlan,
        validated_filters: dict[str, Any] | None,
    ) -> RetrievalExecutionContext:
        filters = normalize_query_filters(validated_filters)
        if filters is None:
            filters = normalize_query_filters(
                {"source_standards": list(plan.requested_standards)}
                if plan.requested_standards
                else None
            )

        resolved_query = str(query or "").strip()
        mode_cfg = self._mode_config(plan.mode)
        coverage_requirements = (
            dict(mode_cfg.coverage_requirements) if isinstance(mode_cfg, QueryModeConfig) else {}
        )

        require_all_scopes = bool(
            coverage_requirements.get(
                "require_all_requested_scopes",
                len(plan.requested_standards) >= 2,
            )
        )
        try:
            min_clause_refs_required = int(
                coverage_requirements.get(
                    "min_clause_refs",
                    1 if bool(plan.require_literal_evidence) else 0,
                )
            )
        except (TypeError, ValueError):
            min_clause_refs_required = 0
        min_clause_refs_required = max(0, min(6, min_clause_refs_required))

        literal_mode = mode_requires_literal_evidence(
            mode=plan.mode,
            profile=self.profile_context,
            explicit_flag=plan.require_literal_evidence,
        )
        cross_scope_mode = len(plan.requested_standards) >= 2 and not literal_mode
        context_volume = "high" if cross_scope_mode else "standard"
        k = max(1, int(plan.chunk_k))
        fetch_k = max(1, int(plan.chunk_fetch_k))

        timeout_comprehensive_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COMPREHENSIVE_MS", 28000) or 28000),
        )

        return RetrievalExecutionContext(
            filters=filters,
            query=resolved_query,
            context_volume=context_volume,
            require_all_scopes=require_all_scopes,
            min_clause_refs_required=min_clause_refs_required,
            k=k,
            fetch_k=fetch_k,
            timeout_comprehensive_ms=timeout_comprehensive_ms,
        )

    def _build_retrieval_policy_payload(self) -> dict[str, Any]:
        hints: list[dict[str, Any]] = []
        min_score: float | None = None
        if self.profile_context is not None:
            for hint in self.profile_context.retrieval.search_hints:
                term = str(hint.term or "").strip()
                expands = [
                    str(item or "").strip() for item in hint.expand_to if str(item or "").strip()
                ]
                if term and expands:
                    hints.append({"term": term, "expand_to": expands})
            try:
                min_score = float(self.profile_context.retrieval.min_score)
            except Exception:
                min_score = None
        payload: dict[str, Any] = {
            "search_hints": hints,
            "noise_reduction": True,
        }
        if min_score is not None:
            payload["min_score"] = min_score
        return payload

    async def _build_retrieval_plan_payload(
        self,
        *,
        query: str,
        plan: RetrievalPlan,
    ) -> dict[str, Any] | None:
        if self.subquery_planner is None:
            return None

        mode_cfg = self._mode_config(plan.mode)
        decomposition_policy = (
            dict(mode_cfg.decomposition_policy) if isinstance(mode_cfg, QueryModeConfig) else {}
        )
        try:
            max_queries = int(
                decomposition_policy.get("max_subqueries")
                or max(1, min(12, len(plan.requested_standards) + 1))
            )
        except (TypeError, ValueError):
            max_queries = max(1, min(12, len(plan.requested_standards) + 1))
        max_queries = max(1, min(12, max_queries))

        planning_context = SubqueryPlanningContext(
            query=str(query or "").strip(),
            requested_standards=tuple(plan.requested_standards),
            max_queries=max_queries,
            mode=plan.mode,
            require_literal_evidence=plan.require_literal_evidence,
            include_semantic_tail=True,
            profile=self.profile_context,
            decomposition_policy=decomposition_policy,
        )

        try:
            raw_subqueries = await self.subquery_planner.plan(planning_context)
        except Exception as exc:
            logger.warning("subquery_planner_failed", error=str(exc))
            return None

        normalized_subqueries: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_subqueries, start=1):
            if not isinstance(item, dict):
                continue
            sq_query = str(item.get("query") or "").strip()
            if not sq_query:
                continue

            raw_id = item.get("id")
            if isinstance(raw_id, int):
                sq_id = raw_id
            elif isinstance(raw_id, str) and raw_id.strip().isdigit():
                sq_id = int(raw_id.strip())
            else:
                sq_id = idx

            raw_dep = item.get("dependency_id")
            dep_id: int | None = None
            if isinstance(raw_dep, int):
                dep_id = raw_dep
            elif isinstance(raw_dep, str) and raw_dep.strip().isdigit():
                dep_id = int(raw_dep.strip())

            rels = item.get("target_relations")
            nodes = item.get("target_node_types")
            target_relations = (
                [str(v).strip() for v in rels if str(v).strip()] if isinstance(rels, list) else None
            )
            target_node_types = (
                [str(v).strip() for v in nodes if str(v).strip()]
                if isinstance(nodes, list)
                else None
            )

            normalized_subqueries.append(
                {
                    "id": sq_id,
                    "query": sq_query,
                    "dependency_id": dep_id,
                    "target_relations": target_relations,
                    "target_node_types": target_node_types,
                    "is_deep": bool(item.get("is_deep", False)),
                }
            )

        if not normalized_subqueries:
            return None

        execution_mode = (
            str(decomposition_policy.get("execution_mode") or "parallel").strip().lower()
        )
        if execution_mode not in {"parallel", "sequential"}:
            execution_mode = "parallel"

        payload = RetrievalPlanPayload.model_validate(
            {
                "is_multihop": len(normalized_subqueries) > 1,
                "execution_mode": execution_mode,
                "sub_queries": normalized_subqueries,
            }
        )
        return payload.model_dump(mode="python", exclude_none=True)

    @staticmethod
    def _build_budget_timeout_fn(
        *, total_budget_ms: int, started_at: float
    ) -> Callable[[int], int]:
        def budgeted_timeout(default_timeout_ms: int) -> int:
            elapsed = int((time.perf_counter() - started_at) * 1000)
            remaining = max(0, total_budget_ms - elapsed)
            if remaining <= 300:
                return 200
            return max(200, min(default_timeout_ms, remaining - 200))

        return budgeted_timeout

    def _finalize_diagnostics(
        self,
        *,
        items: list[dict[str, Any]],
        trace: dict[str, Any],
        timings_ms: dict[str, float],
        validated_scope_payload: dict[str, Any] | None,
        retrieval_plan_payload: dict[str, Any] | None,
    ) -> None:
        if isinstance(self.profile_resolution_context, dict):
            trace["agent_profile_resolution"] = dict(self.profile_resolution_context)
        trace.setdefault("strategy_path", "comprehensive")
        trace["timings_ms"] = {**dict(trace.get("timings_ms") or {}), **timings_ms}
        if isinstance(retrieval_plan_payload, dict):
            trace["subqueries"] = list(retrieval_plan_payload.get("sub_queries") or [])

        diag_trace_final = dict(trace)
        diag_trace_final.update(calculate_layer_stats([it for it in items if isinstance(it, dict)]))
        diag_trace_final["rag_features"] = features_from_hybrid_trace(trace)
        self.last_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="comprehensive",
            partial=False,
            trace=diag_trace_final,
            scope_validation=validated_scope_payload or {},
        )

    async def _execute_comprehensive_primary(
        self,
        *,
        tenant_id: str,
        collection_id: str | None,
        user_id: str | None,
        request_id: str | None,
        correlation_id: str | None,
        plan: RetrievalPlan,
        context: RetrievalExecutionContext,
        retrieval_plan_payload: dict[str, Any] | None,
        timings_ms: dict[str, float],
        budgeted_timeout_fn: Callable[[int], int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None, str | None]:
        comprehensive_payload, error_code, error_detail = await self._safe_execute(
            op_name="comprehensive_primary",
            timeout_ms=budgeted_timeout_fn(context.timeout_comprehensive_ms),
            operation=self.contract_client.comprehensive(
                query=context.query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
                context_volume=context.context_volume,
                k=context.k,
                fetch_k=context.fetch_k,
                filters=context.filters,
                rerank={"enabled": True},
                graph={"max_hops": 2},
                coverage_requirements={
                    "requested_standards": list(plan.requested_standards),
                    "require_all_scopes": context.require_all_scopes,
                    "min_clause_refs": context.min_clause_refs_required,
                },
                retrieval_policy=self._build_retrieval_policy_payload(),
                retrieval_plan=retrieval_plan_payload,
            ),
            timings_ms=timings_ms,
        )
        items = (
            comprehensive_payload.get("items") if isinstance(comprehensive_payload, dict) else []
        )
        trace = (
            comprehensive_payload.get("trace") if isinstance(comprehensive_payload, dict) else {}
        )
        if not isinstance(items, list):
            items = []
        if not isinstance(trace, dict):
            trace = {}
        return [it for it in items if isinstance(it, dict)], trace, error_code, error_detail

    async def execute(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        validated_filters: dict[str, Any] | None = None,
        validated_scope_payload: dict[str, Any] | None = None,
    ) -> list[EvidenceItem]:
        if not bool(getattr(settings, "ORCH_RETRIEVAL_COMPREHENSIVE_ENABLED", True)):
            raise RuntimeError("Comprehensive retrieval is required but disabled")

        execute_started_at = time.perf_counter()
        total_budget_ms = max(
            1000, int(getattr(settings, "ORCH_TIMEOUT_EXECUTE_TOOL_MS", 30000) or 30000)
        )
        budgeted_timeout = self._build_budget_timeout_fn(
            total_budget_ms=total_budget_ms,
            started_at=execute_started_at,
        )
        context = self._prepare_execution_context(
            query=query,
            plan=plan,
            validated_filters=validated_filters,
        )
        retrieval_plan_payload = await self._build_retrieval_plan_payload(
            query=context.query,
            plan=plan,
        )
        timings_ms: dict[str, float] = {}

        (
            items,
            trace,
            comprehensive_error_code,
            comprehensive_error_detail,
        ) = await self._execute_comprehensive_primary(
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
            plan=plan,
            context=context,
            retrieval_plan_payload=retrieval_plan_payload,
            timings_ms=timings_ms,
            budgeted_timeout_fn=budgeted_timeout,
        )
        if comprehensive_error_code:
            raise RuntimeError(
                f"comprehensive_retrieval_failed:{comprehensive_error_code}:{comprehensive_error_detail or ''}"
            )

        self._finalize_diagnostics(
            items=items,
            trace=trace,
            timings_ms=timings_ms,
            validated_scope_payload=validated_scope_payload,
            retrieval_plan_payload=retrieval_plan_payload,
        )
        return self._to_evidence(items)

    async def _safe_execute(
        self,
        *,
        op_name: str,
        timeout_ms: int,
        operation: Any,
        timings_ms: dict[str, float],
    ) -> tuple[dict[str, Any], str | None, str | None]:
        started = time.perf_counter()
        try:
            payload = await self._with_timeout(
                op_name=op_name,
                timeout_ms=timeout_ms,
                operation=operation,
            )
        except RuntimeError as exc:
            timings_ms[op_name] = round((time.perf_counter() - started) * 1000, 2)
            logger.warning("retrieval_strategy_failed", strategy=op_name, error=str(exc)[:160])
            return {}, RETRIEVAL_CODE_TIMEOUT, str(exc)[:160]
        except httpx.RequestError as exc:
            timings_ms[op_name] = round((time.perf_counter() - started) * 1000, 2)
            logger.warning("retrieval_strategy_failed", strategy=op_name, error=str(exc)[:160])
            return {}, RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE, str(exc)[:160]
        except Exception as exc:
            timings_ms[op_name] = round((time.perf_counter() - started) * 1000, 2)
            logger.warning("retrieval_strategy_failed", strategy=op_name, error=str(exc)[:160])
            return {}, RETRIEVAL_CODE_INVALID_RESPONSE, str(exc)[:160]

        timings_ms[op_name] = round((time.perf_counter() - started) * 1000, 2)
        if not isinstance(payload, dict):
            logger.warning("retrieval_strategy_invalid_payload", strategy=op_name)
            return {}, RETRIEVAL_CODE_INVALID_RESPONSE, "invalid_payload"
        return payload, None, None

    async def _with_timeout(
        self,
        *,
        op_name: str,
        timeout_ms: int,
        operation: Any,
    ) -> Any:
        try:
            return await asyncio.wait_for(operation, timeout=timeout_ms / 1000.0)
        except TimeoutError as exc:
            logger.warning("retrieval_budget_timeout", operation=op_name, timeout_ms=timeout_ms)
            raise RuntimeError(f"retrieval_timeout:{op_name}") from exc

    @staticmethod
    def _to_evidence(items: Any) -> list[EvidenceItem]:
        if not isinstance(items, list):
            return []
        out: list[EvidenceItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue

            raw_meta = item.get("metadata") or {}
            if isinstance(raw_meta, dict) and "row" in raw_meta:
                final_metadata = raw_meta
            else:
                final_metadata = {
                    "row": {
                        "content": content,
                        "metadata": raw_meta,
                        "similarity": item.get("score"),
                    }
                }

            out.append(
                EvidenceItem(
                    source=str(item.get("source") or "C1"),
                    content=content,
                    score=float(item.get("score") or 0.0),
                    metadata=final_metadata,
                )
            )
        return out
