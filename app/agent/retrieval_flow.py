from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog

from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    merge_error_codes,
)
from app.agent.interfaces import SubqueryPlanner, SubqueryPlanningContext
from app.agent.components.query_decomposer import HybridSubqueryPlanner
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig
from app.agent.retrieval_planner import (
    apply_search_hints,
    decide_multihop_fallback,
    extract_clause_refs,
    mode_requires_literal_evidence,
)
from app.agent.retrieval_sufficiency_evaluator import RetrievalSufficiencyEvaluator
from app.core.config import settings
from app.core.rag_retrieval_contract_client import RagRetrievalContractClient
from app.agent.retrieval_strategies import (
    calculate_layer_stats,
    features_from_hybrid_trace,
    find_missing_clause_refs,
    find_missing_scopes,
    reduce_structural_noise,
)

logger = structlog.get_logger(__name__)

# Re-define RETRIEVAL_CODE_LOW_SCORE if not available in error_codes but used in filter logic.
# Looking at http_adapters.py, it was used but not explicitly imported in the snippet I saw.
# Actually, it was used in _filter_items_by_min_score which I should also probably move or replicate.
RETRIEVAL_CODE_LOW_SCORE = "ORCH_RETRIEVAL_LOW_SCORE"

class RetrievalFlow:
    def __init__(
        self,
        contract_client: RagRetrievalContractClient,
        subquery_planner: SubqueryPlanner | None = None,
        profile_context: AgentProfile | None = None,
        profile_resolution_context: dict[str, Any] | None = None,
    ):
        self.contract_client = contract_client
        self.subquery_planner = subquery_planner or HybridSubqueryPlanner.from_settings()
        self.profile_context = profile_context
        self.profile_resolution_context = profile_resolution_context
        self.last_diagnostics: RetrievalDiagnostics | None = None

    def _profile_min_score(self) -> float | None:
        if self.profile_context is None:
            return None
        try:
            return float(self.profile_context.retrieval.min_score)
        except Exception:
            return None

    def _mode_config(self, mode: str) -> QueryModeConfig | None:
        if self.profile_context is None:
            return None
        cfg = self.profile_context.query_modes.modes.get(str(mode or "").strip())
        return cfg if isinstance(cfg, QueryModeConfig) else None

    def _filter_items_by_min_score(
        self,
        items: list[dict[str, Any]],
        *,
        trace_target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        threshold = self._profile_min_score()
        if threshold is None:
            return items

        kept: list[dict[str, Any]] = []
        dropped = 0
        scored_dropped: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            score_raw = item.get("score")
            if score_raw is None:
                score_raw = item.get("similarity")
            if score_raw is None:
                kept.append(item)
                continue
            score = float(score_raw or 0.0)
            if score >= threshold:
                kept.append(item)
            else:
                dropped += 1
                scored_dropped.append((score, item))
        
        logger.info(
            "min_score_filter_debug",
            threshold=threshold,
            kept=len(kept),
            dropped=dropped,
            top_dropped_score=scored_dropped[0][0] if scored_dropped else None
        )

        backstop_applied = False
        backstop_enabled = bool(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_ENABLED", False))
        backstop_top_n = max(1, int(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_TOP_N", 6) or 6))
        if not kept and dropped > 0 and backstop_enabled:
            top_n = backstop_top_n
            scored_dropped.sort(key=lambda pair: pair[0], reverse=True)
            kept = [item for _, item in scored_dropped[:top_n]]
            backstop_applied = bool(kept)

        if isinstance(trace_target, dict):
            trace_target["min_score_filter"] = {
                "threshold": threshold,
                "kept": len(kept),
                "dropped": dropped,
                "backstop_applied": backstop_applied,
                "backstop_top_n": (backstop_top_n if backstop_applied else 0),
            }
            if dropped > 0 and not kept:
                trace_target["error_codes"] = merge_error_codes(
                    trace_target.get("error_codes"),
                    [RETRIEVAL_CODE_LOW_SCORE],
                )
        return kept

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
        contract_client = self.contract_client
        filters = validated_filters
        if filters is None:
            filters = (
                {"source_standards": list(plan.requested_standards)}
                if plan.requested_standards
                else None
            )

        expanded_query, hint_trace = apply_search_hints(query, profile=self.profile_context)
        clause_refs = extract_clause_refs(query, profile=self.profile_context)
        multihop_hint = len(plan.requested_standards) >= 2 or len(clause_refs) >= 2
        mode_cfg = self._mode_config(plan.mode)
        
        coverage_requirements = (
            dict(mode_cfg.coverage_requirements) if isinstance(mode_cfg, QueryModeConfig) else {}
        )
        decomposition_policy = (
            dict(mode_cfg.decomposition_policy) if isinstance(mode_cfg, QueryModeConfig) else {}
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
        
        try:
            mode_max_subqueries = int(decomposition_policy.get("max_subqueries", 6))
        except (TypeError, ValueError):
            mode_max_subqueries = 6
        mode_max_subqueries = max(2, min(12, mode_max_subqueries))
        
        literal_mode = mode_requires_literal_evidence(
            mode=plan.mode,
            profile=self.profile_context,
            explicit_flag=plan.require_literal_evidence,
        )
        cross_scope_mode = len(plan.requested_standards) >= 2 and not literal_mode
        k_cap = 24 if cross_scope_mode else 18
        k = max(1, min(int(plan.chunk_k), k_cap))
        fetch_k = max(1, int(plan.chunk_fetch_k))
        semantic_tail_enabled = bool(
            getattr(settings, "ORCH_DETERMINISTIC_SUBQUERY_SEMANTIC_TAIL", False)
        )

        timings_ms: dict[str, float] = {}
        timeout_multi_query_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_MULTI_QUERY_MS", 1800) or 1800),
        )
        timeout_hybrid_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_HYBRID_MS", 1800) or 1800),
        )
        timeout_cov_repair_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COVERAGE_REPAIR_MS", 800) or 800),
        )

        async def build_subqueries(*, purpose: str = "primary") -> list[dict[str, Any]]:
            planner = self.subquery_planner or HybridSubqueryPlanner.from_settings()
            subqueries_local = await planner.plan(
                SubqueryPlanningContext(
                    query=query,
                    requested_standards=plan.requested_standards,
                    max_queries=mode_max_subqueries,
                    mode=plan.mode,
                    require_literal_evidence=plan.require_literal_evidence,
                    include_semantic_tail=semantic_tail_enabled,
                    profile=self.profile_context,
                    decomposition_policy=decomposition_policy,
                )
            )
            fallback_max = max(
                2,
                int(getattr(settings, "ORCH_MULTI_QUERY_FALLBACK_MAX_QUERIES", 3) or 3),
            )
            fallback_max = min(mode_max_subqueries, fallback_max)
            if purpose == "fallback" and len(subqueries_local) > fallback_max:
                subqueries_local = subqueries_local[:fallback_max]
            return subqueries_local

        # 1. Primary Strategy: Multi-Query (Optional)
        if settings.ORCH_MULTI_QUERY_PRIMARY and multihop_hint and int(plan.chunk_k or 0) > 0:
            merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
            subqueries = await build_subqueries(purpose="primary")
            t0 = time.perf_counter()
            mq_payload = await self._with_timeout(
                op_name="multi_query_primary",
                timeout_ms=timeout_multi_query_ms,
                operation=contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=subqueries,
                    merge=merge,
                ),
            )
            timings_ms["multi_query_primary"] = round((time.perf_counter() - t0) * 1000, 2)
            mq_items = mq_payload.get("items") if isinstance(mq_payload, dict) else []
            mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
            partial = bool(mq_payload.get("partial", False)) if isinstance(mq_payload, dict) else False
            subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
            diag_trace: dict[str, Any] = {
                "promoted": True,
                "reason": "complex_intent",
                "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                "subqueries": subq if isinstance(subq, list) else [],
                "timings_ms": dict(timings_ms),
            }

            if isinstance(mq_items, list) and len(mq_items) >= max(
                1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
            ):
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items = reduce_structural_noise(mq_items, query)
                diag_trace.update(calculate_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                if hint_trace.get("applied"):
                    diag_trace["search_hint_expansions"] = hint_trace
                if isinstance(self.profile_resolution_context, dict):
                    diag_trace["agent_profile_resolution"] = dict(self.profile_resolution_context)
                
                # Coverage gate
                mq_items = await self._execute_coverage_repair(
                    items=[it for it in mq_items if isinstance(it, dict)],
                    base_trace=diag_trace,
                    reason="multi_query_primary",
                    plan=plan,
                    require_all_scopes=require_all_scopes,
                    clause_refs=clause_refs,
                    min_clause_refs_required=min_clause_refs_required,
                    expanded_query=expanded_query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    timings_ms=timings_ms,
                    timeout_cov_repair_ms=timeout_cov_repair_ms,
                    k=k,
                )
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items = reduce_structural_noise(mq_items, query)

                self.last_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="multi_query_primary",
                    partial=partial,
                    trace=diag_trace,
                    scope_validation=validated_scope_payload or {},
                )
                return self._to_evidence(mq_items)

            if settings.ORCH_MULTI_QUERY_EVALUATOR and isinstance(mq_items, list) and mq_items:
                evaluator = RetrievalSufficiencyEvaluator()
                decision = await evaluator.evaluate(
                    query=query,
                    requested_standards=plan.requested_standards,
                    items=[it for it in mq_items if isinstance(it, dict)],
                    min_items=int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6),
                )
                if decision.sufficient:
                    diag_trace["evaluator_override"] = True
                    diag_trace["evaluator_reason"] = decision.reason
                    diag_trace.update(calculate_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                    mq_items = await self._execute_coverage_repair(
                        items=[it for it in mq_items if isinstance(it, dict)],
                        base_trace=diag_trace,
                        reason="multi_query_primary_evaluator",
                        plan=plan,
                        require_all_scopes=require_all_scopes,
                        clause_refs=clause_refs,
                        min_clause_refs_required=min_clause_refs_required,
                        expanded_query=expanded_query,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        timings_ms=timings_ms,
                        timeout_cov_repair_ms=timeout_cov_repair_ms,
                        k=k,
                    )
                    mq_items = self._filter_items_by_min_score(
                        [it for it in mq_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    mq_items = reduce_structural_noise(mq_items, query)
                    self.last_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="multi_query_primary_evaluator",
                        partial=partial,
                        trace=diag_trace,
                        scope_validation=validated_scope_payload or {},
                    )
                    return self._to_evidence(mq_items)

            if settings.ORCH_MULTI_QUERY_REFINE:
                step_back = {
                    "id": "step_back",
                    "query": f"principios generales y requisitos clave relacionados: {expanded_query}",
                    "k": None,
                    "fetch_k": None,
                    "filters": {"source_standards": list(plan.requested_standards)}
                    if plan.requested_standards
                    else None,
                }
                refined = (subqueries + [step_back])[: max(1, int(settings.ORCH_PLANNER_MAX_QUERIES or 5))]
                t1 = time.perf_counter()
                mq2 = await self._with_timeout(
                    op_name="multi_query_refine",
                    timeout_ms=timeout_multi_query_ms,
                    operation=contract_client.multi_query(
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        queries=refined,
                        merge=merge,
                    ),
                )
                timings_ms["multi_query_refine"] = round((time.perf_counter() - t1) * 1000, 2)
                mq2_items = mq2.get("items") if isinstance(mq2, dict) else []
                if isinstance(mq2_items, list) and len(mq2_items) >= max(
                    1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
                ):
                    mq2_items = self._filter_items_by_min_score(
                        [it for it in mq2_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    mq2_items = reduce_structural_noise(mq2_items, query)
                    diag_trace["refined"] = True
                    diag_trace["refine_reason"] = "insufficient_primary_multi_query"
                    diag_trace["timings_ms"] = dict(timings_ms)
                    diag_trace.update(
                        calculate_layer_stats([it for it in mq2_items if isinstance(it, dict)])
                    )
                    mq2_items = await self._execute_coverage_repair(
                        items=[it for it in mq2_items if isinstance(it, dict)],
                        base_trace=diag_trace,
                        reason="multi_query_refined",
                        plan=plan,
                        require_all_scopes=require_all_scopes,
                        clause_refs=clause_refs,
                        min_clause_refs_required=min_clause_refs_required,
                        expanded_query=expanded_query,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        timings_ms=timings_ms,
                        timeout_cov_repair_ms=timeout_cov_repair_ms,
                        k=k,
                    )
                    mq2_items = self._filter_items_by_min_score(
                        [it for it in mq2_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    self.last_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="multi_query_refined",
                        partial=bool(mq2.get("partial", False)) if isinstance(mq2, dict) else False,
                        trace=diag_trace,
                        scope_validation=validated_scope_payload or {},
                    )
                    return self._to_evidence(mq2_items)

        # 2. Strategy: Hybrid (Default)
        t_h = time.perf_counter()
        try:
            hybrid_payload = await self._with_timeout(
                op_name="hybrid",
                timeout_ms=timeout_hybrid_ms,
                operation=contract_client.hybrid(
                    query=expanded_query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    k=k,
                    fetch_k=fetch_k,
                    filters=filters,
                    rerank={"enabled": True},
                    graph={"max_hops": 2},
                ),
            )
        except Exception as hybrid_exc:
            logger.warning(
                "hybrid_retrieval_failed",
                error=str(hybrid_exc)[:120],
            )
            hybrid_payload = {}
        timings_ms["hybrid"] = round((time.perf_counter() - t_h) * 1000, 2)
        items = hybrid_payload.get("items") if isinstance(hybrid_payload, dict) else []
        trace = hybrid_payload.get("trace") if isinstance(hybrid_payload, dict) else {}
        if not isinstance(items, list):
            items = []
        if not isinstance(trace, dict):
            trace = {}
        items = self._filter_items_by_min_score(
            [it for it in items if isinstance(it, dict)],
            trace_target=trace,
        )
        if hint_trace.get("applied"):
            trace["search_hint_expansions"] = hint_trace
        if isinstance(self.profile_resolution_context, dict):
            trace["agent_profile_resolution"] = dict(self.profile_resolution_context)

        # 3. Multihop Fallback Decision
        if settings.ORCH_MULTIHOP_FALLBACK and multihop_hint and int(plan.chunk_k or 0) > 0:
            if (
                bool(getattr(settings, "EARLY_EXIT_COVERAGE_ENABLED", True))
                and len(plan.requested_standards) >= 2
            ):
                missing_before_fallback = find_missing_scopes(
                    items,
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                if not missing_before_fallback:
                    if isinstance(trace, dict):
                        trace["multi_query_fallback_skipped"] = "coverage_already_satisfied"
                    
                    diag_trace_local = {
                        "hybrid_trace": trace,
                        "timings_ms": dict(timings_ms),
                        "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                        "multi_query_fallback_skipped": "coverage_already_satisfied",
                    }
                    self.last_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="hybrid",
                        partial=False,
                        trace=diag_trace_local,
                        scope_validation=validated_scope_payload or {},
                    )
                    items2 = await self._execute_coverage_repair(
                        items=[it for it in items if isinstance(it, dict)],
                        base_trace=diag_trace_local,
                        reason="hybrid",
                        plan=plan,
                        require_all_scopes=require_all_scopes,
                        clause_refs=clause_refs,
                        min_clause_refs_required=min_clause_refs_required,
                        expanded_query=expanded_query,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        timings_ms=timings_ms,
                        timeout_cov_repair_ms=timeout_cov_repair_ms,
                        k=k,
                    )
                    items2 = self._filter_items_by_min_score(
                        [it for it in items2 if isinstance(it, dict)],
                        trace_target=diag_trace_local,
                    )
                    items2 = reduce_structural_noise(items2, query)
                    return self._to_evidence(items2)

            rows: list[dict[str, Any]] = []
            for it in items[: max(1, min(12, len(items)))]:
                if not isinstance(it, dict):
                    continue
                meta = it.get("metadata")
                row = meta.get("row") if isinstance(meta, dict) else None
                if isinstance(row, dict):
                    rows.append(row)

            decision = decide_multihop_fallback(
                query=query,
                requested_standards=plan.requested_standards,
                items=rows,
                hybrid_trace=trace,
                top_k=12,
            )
            if decision.needs_fallback:
                missing_before_fallback = find_missing_scopes(
                    items,
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                subqueries = await build_subqueries(purpose="fallback")
                merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
                t_mq = time.perf_counter()
                try:
                    mq_payload = await self._with_timeout(
                        op_name="multi_query_fallback",
                        timeout_ms=timeout_multi_query_ms,
                        operation=contract_client.multi_query(
                            tenant_id=tenant_id,
                            collection_id=collection_id,
                            user_id=user_id,
                            request_id=request_id,
                            correlation_id=correlation_id,
                            queries=subqueries,
                            merge=merge,
                        ),
                    )
                except Exception as mq_exc:
                    logger.warning(
                        "multi_query_fallback_failed_using_hybrid",
                        error=str(mq_exc)[:120],
                        hybrid_items=len(items),
                    )
                    timings_ms["multi_query_fallback"] = round((time.perf_counter() - t_mq) * 1000, 2)
                    # Graceful degradation: return hybrid items instead of nothing
                    items = self._filter_items_by_min_score(
                        [it for it in items if isinstance(it, dict)],
                        trace_target=trace,
                    )
                    items = reduce_structural_noise(items, query)
                    self.last_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="hybrid",
                        partial=True,
                        trace={
                            "hybrid_trace": trace,
                            "timings_ms": dict(timings_ms),
                            "multi_query_fallback_error": str(mq_exc)[:120],
                        },
                        scope_validation=validated_scope_payload or {},
                    )
                    return self._to_evidence(items)

                timings_ms["multi_query_fallback"] = round((time.perf_counter() - t_mq) * 1000, 2)
                mq_items_raw = mq_payload.get("items") if isinstance(mq_payload, dict) else []
                mq_items = mq_items_raw if isinstance(mq_items_raw, list) else []
                mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
                partial = bool(mq_payload.get("partial", False)) if isinstance(mq_payload, dict) else False
                subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
                
                diag_trace_fb: dict[str, Any] = {
                    "fallback_reason": decision.reason,
                    "error_codes": [decision.code or RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP],
                    "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                    "mode_policy": {
                        "require_all_requested_scopes": require_all_scopes,
                        "min_clause_refs": min_clause_refs_required,
                        "max_subqueries": mode_max_subqueries,
                    },
                    "hybrid_trace": trace,
                    "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                    "subqueries": subq if isinstance(subq, list) else [],
                    "timings_ms": dict(timings_ms),
                }
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace_fb,
                )
                mq_items = reduce_structural_noise(mq_items, query)
                missing_after_fallback = find_missing_scopes(
                    [it for it in mq_items if isinstance(it, dict)],
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                if (
                    bool(getattr(settings, "EARLY_EXIT_COVERAGE_ENABLED", True))
                    and len(plan.requested_standards) >= 2
                ):
                    if len(missing_after_fallback) >= len(missing_before_fallback):
                        diag_trace_fb["multi_query_fallback_early_exit"] = "no_coverage_improvement"
                        diag_trace_fb["missing_scopes_before"] = list(missing_before_fallback)
                        diag_trace_fb["missing_scopes_after"] = list(missing_after_fallback)
                        if missing_after_fallback:
                            diag_trace_fb["error_codes"] = merge_error_codes(
                                diag_trace_fb.get("error_codes"),
                                [RETRIEVAL_CODE_SCOPE_MISMATCH],
                            )
                        self.last_diagnostics = RetrievalDiagnostics(
                            contract="advanced",
                            strategy="multi_query",
                            partial=partial,
                            trace=diag_trace_fb,
                            scope_validation=validated_scope_payload or {},
                        )
                        return self._to_evidence(mq_items)
                
                diag_trace_fb["rag_features"] = features_from_hybrid_trace(trace)
                if hint_trace.get("applied"):
                    diag_trace_fb["search_hint_expansions"] = hint_trace
                if isinstance(self.profile_resolution_context, dict):
                    diag_trace_fb["agent_profile_resolution"] = dict(self.profile_resolution_context)
                if isinstance(mq_items, list):
                    diag_trace_fb.update(calculate_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                
                self.last_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="multi_query",
                    partial=partial,
                    trace=diag_trace_fb,
                    scope_validation=validated_scope_payload or {},
                )
                mq_items2 = await self._execute_coverage_repair(
                    items=[it for it in mq_items if isinstance(it, dict)],
                    base_trace=diag_trace_fb,
                    reason="multi_query_fallback",
                    plan=plan,
                    require_all_scopes=require_all_scopes,
                    clause_refs=clause_refs,
                    min_clause_refs_required=min_clause_refs_required,
                    expanded_query=expanded_query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    timings_ms=timings_ms,
                    timeout_cov_repair_ms=timeout_cov_repair_ms,
                    k=k,
                )
                mq_items2 = self._filter_items_by_min_score(
                    [it for it in mq_items2 if isinstance(it, dict)],
                    trace_target=diag_trace_fb,
                )
                mq_items2 = reduce_structural_noise(mq_items2, query)
                return self._to_evidence(mq_items2)

        # 4. Final Fallback: Hybrid with Coverage Repair
        diag_trace_final = {
            "hybrid_trace": trace,
            "timings_ms": dict(timings_ms),
            "deterministic_subquery_semantic_tail": semantic_tail_enabled,
            "mode_policy": {
                "require_all_requested_scopes": require_all_scopes,
                "min_clause_refs": min_clause_refs_required,
                "max_subqueries": mode_max_subqueries,
            },
        }
        self.last_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="hybrid",
            partial=False,
            trace=diag_trace_final,
            scope_validation=validated_scope_payload or {},
        )
        if isinstance(items, list):
            diag_trace_final.update(calculate_layer_stats([it for it in items if isinstance(it, dict)]))
            diag_trace_final["rag_features"] = features_from_hybrid_trace(trace)
            if hint_trace.get("applied"):
                diag_trace_final["search_hint_expansions"] = hint_trace
            if isinstance(self.profile_resolution_context, dict):
                diag_trace_final["agent_profile_resolution"] = dict(self.profile_resolution_context)
        
        items2 = await self._execute_coverage_repair(
            items=[it for it in items if isinstance(it, dict)],
            base_trace=diag_trace_final,
            reason="hybrid",
            plan=plan,
            require_all_scopes=require_all_scopes,
            clause_refs=clause_refs,
            min_clause_refs_required=min_clause_refs_required,
            expanded_query=expanded_query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
            timings_ms=timings_ms,
            timeout_cov_repair_ms=timeout_cov_repair_ms,
            k=k,
        )
        items2 = self._filter_items_by_min_score(
            [it for it in items2 if isinstance(it, dict)],
            trace_target=diag_trace_final,
        )
        items2 = reduce_structural_noise(items2, query)
        return self._to_evidence(items2)

    async def _execute_coverage_repair(
        self,
        *,
        items: list[dict[str, Any]],
        base_trace: dict[str, Any],
        reason: str,
        plan: RetrievalPlan,
        require_all_scopes: bool,
        clause_refs: list[str],
        min_clause_refs_required: int,
        expanded_query: str,
        tenant_id: str,
        collection_id: str | None,
        user_id: str | None,
        request_id: str | None,
        correlation_id: str | None,
        timings_ms: dict[str, float],
        timeout_cov_repair_ms: int,
        k: int,
    ) -> list[dict[str, Any]]:
        if not settings.ORCH_COVERAGE_GATE_ENABLED:
            return items
        
        missing_scopes = find_missing_scopes(
            items,
            plan.requested_standards,
            enforce=require_all_scopes,
        )
        missing_clauses = find_missing_clause_refs(
            items,
            clause_refs,
            min_required=min_clause_refs_required,
        )
        if not missing_scopes and not missing_clauses:
            base_trace["missing_scopes"] = []
            base_trace["missing_clause_refs"] = []
            return items
            
        cap = max(1, int(settings.ORCH_COVERAGE_GATE_MAX_MISSING or 2))
        missing_scopes = missing_scopes[:cap]
        missing_clauses = missing_clauses[:cap]

        focused: list[dict[str, Any]] = []
        for idx, scope in enumerate(missing_scopes, start=1):
            qtext = " ".join(part for part in [scope, *clause_refs[:3], expanded_query] if part).strip()
            focused.append({
                "id": f"scope_repair_{idx}",
                "query": qtext[:900],
                "k": None,
                "fetch_k": None,
                "filters": {"source_standard": scope},
            })
        for idx, clause in enumerate(missing_clauses, start=1):
            focused.append({
                "id": f"clause_repair_{idx}",
                "query": f"{expanded_query} clausula {clause}"[:900],
                "k": None,
                "fetch_k": None,
                "filters": {
                    **({"source_standards": list(plan.requested_standards)} if plan.requested_standards else {}),
                    "metadata": {"clause_id": clause},
                },
            })

        merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(18, max(12, k))}
        t_cov = time.perf_counter()
        
        try:
            cov_payload = await self._with_timeout(
                op_name="coverage_gate_multi_query",
                timeout_ms=timeout_cov_repair_ms,
                operation=self.contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=focused,
                    merge=merge,
                ),
            )
        except RuntimeError:
            # Coverage repair is best-effort: if it fails, return original items
            timings_ms["coverage_gate"] = round((time.perf_counter() - t_cov) * 1000, 2)
            base_trace["coverage_gate"] = {
                "trigger_reason": reason,
                "missing_scopes": missing_scopes,
                "missing_clause_refs": missing_clauses,
                "error": "coverage_gate_timeout",
            }
            return items
        timings_ms["coverage_gate"] = round((time.perf_counter() - t_cov) * 1000, 2)

        cov_items = cov_payload.get("items") if isinstance(cov_payload, dict) else []
        if not isinstance(cov_items, list) or not cov_items:
            base_trace["coverage_gate"] = {
                "trigger_reason": reason,
                "missing_scopes": missing_scopes,
                "missing_clause_refs": missing_clauses,
                "added_queries": [q.get("id") for q in focused],
                "final_missing_scopes": list(missing_scopes),
                "final_missing_clause_refs": list(missing_clauses),
            }
            base_trace["missing_scopes"] = list(missing_scopes)
            base_trace["missing_clause_refs"] = list(missing_clauses)
            codes = []
            if missing_scopes: codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
            if missing_clauses: codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
            base_trace["error_codes"] = merge_error_codes(base_trace.get("error_codes"), codes)
            return items

        # Merge and dedupe
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for it in [*items, *cov_items]:
            if not isinstance(it, dict): continue
            sid = str(it.get("source") or "")
            key = sid or str(len(seen))
            if key in seen: continue
            seen.add(key)
            merged.append(it)

        base_trace["coverage_gate"] = {
            "trigger_reason": reason,
            "missing_scopes": missing_scopes,
            "missing_clause_refs": missing_clauses,
            "added_queries": [q.get("id") for q in focused],
        }

        # Step-back pass
        remaining = find_missing_scopes(merged, plan.requested_standards, enforce=require_all_scopes)
        remaining_clauses = find_missing_clause_refs(merged, clause_refs, min_required=min_clause_refs_required)
        
        if (remaining or remaining_clauses) and settings.ORCH_COVERAGE_GATE_STEP_BACK:
            remaining = remaining[:cap]
            remaining_clauses = remaining_clauses[:cap]
            step_back_queries: list[dict[str, Any]] = []
            for idx, scope in enumerate(remaining, start=1):
                step_back_queries.append({
                    "id": f"scope_step_back_{idx}",
                    "query": f"principios generales y requisitos clave relacionados con: {expanded_query}",
                    "k": None,
                    "fetch_k": None,
                    "filters": {"source_standard": scope},
                })
            for idx, clause in enumerate(remaining_clauses, start=1):
                step_back_queries.append({
                    "id": f"clause_step_back_{idx}",
                    "query": f"principios generales y requisitos clave relacionados con: {expanded_query} clausula {clause}"[:900],
                    "k": None,
                    "fetch_k": None,
                    "filters": {
                        **({"source_standards": list(plan.requested_standards)} if plan.requested_standards else {}),
                        "metadata": {"clause_id": clause},
                    },
                })
            
            t_sb = time.perf_counter()
            sb_payload = await self._with_timeout(
                op_name="coverage_gate_step_back_multi_query",
                timeout_ms=timeout_cov_repair_ms,
                operation=self.contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=step_back_queries,
                    merge=merge,
                ),
            )
            timings_ms["coverage_gate_step_back"] = round((time.perf_counter() - t_sb) * 1000, 2)
            sb_items = sb_payload.get("items") if isinstance(sb_payload, dict) else []
            if isinstance(sb_items, list) and sb_items:
                for it in sb_items:
                    if not isinstance(it, dict): continue
                    sid = str(it.get("source") or "")
                    key = sid or str(len(seen))
                    if key in seen: continue
                    seen.add(key)
                    merged.append(it)
                if isinstance(base_trace.get("coverage_gate"), dict):
                    base_trace["coverage_gate"]["step_back_missing_scopes"] = remaining
                    base_trace["coverage_gate"]["step_back_missing_clause_refs"] = remaining_clauses
                    base_trace["coverage_gate"]["step_back_queries"] = [q.get("id") for q in step_back_queries]

        final_missing = find_missing_scopes(merged, plan.requested_standards, enforce=require_all_scopes)
        final_missing_clauses = find_missing_clause_refs(merged, clause_refs, min_required=min_clause_refs_required)
        
        if isinstance(base_trace.get("coverage_gate"), dict):
            base_trace["coverage_gate"]["final_missing_scopes"] = final_missing
            base_trace["coverage_gate"]["final_missing_clause_refs"] = final_missing_clauses
        
        base_trace["missing_scopes"] = list(final_missing)
        base_trace["missing_clause_refs"] = list(final_missing_clauses)
        codes = []
        if final_missing: codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
        if final_missing_clauses: codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
        base_trace["error_codes"] = merge_error_codes(base_trace.get("error_codes"), codes)
        
        return merged

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
