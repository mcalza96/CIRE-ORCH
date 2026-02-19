from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx
import structlog

from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_INVALID_RESPONSE,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    RETRIEVAL_CODE_TIMEOUT,
    RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE,
    merge_error_codes,
)
from app.agent.retrieval_filters import filter_items_by_min_score
from app.agent.interfaces import (
    EmbeddingProvider,
    RerankingProvider,
    SubqueryPlanner,
    SubqueryPlanningContext,
)
from app.agent.components.query_decomposer import HybridSubqueryPlanner
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig
from app.agent.retrieval_planner import (
    apply_search_hints,
    decide_multihop_fallback,
    extract_clause_refs,
    mode_requires_literal_evidence,
    normalize_query_filters,
)
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
        self.subquery_planner = subquery_planner or HybridSubqueryPlanner.from_settings()
        self.embedding_provider = embedding_provider
        self.reranking_provider = reranking_provider
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
        return filter_items_by_min_score(
            items,
            threshold=self._profile_min_score(),
            trace_target=trace_target,
        )

    @staticmethod
    def _safe_score(item: dict[str, Any]) -> float:
        if not isinstance(item, dict):
            return -1.0
        raw = item.get("score")
        if raw is None:
            raw = item.get("similarity")
        if raw is None:
            return -1.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return -1.0

    def _ensure_non_empty_candidates(
        self,
        *,
        raw_items: list[dict[str, Any]],
        filtered_items: list[dict[str, Any]],
        trace_target: dict[str, Any] | None,
        stage: str,
    ) -> list[dict[str, Any]]:
        if filtered_items or not raw_items:
            return filtered_items
        top_n = max(1, int(getattr(settings, "ORCH_EMPTY_RESULT_BACKSTOP_TOP_N", 8) or 8))
        rescued = sorted(raw_items, key=self._safe_score, reverse=True)[:top_n]
        if isinstance(trace_target, dict):
            trace_target["empty_result_backstop"] = {
                "stage": stage,
                "applied": True,
                "kept": len(rescued),
                "top_n": top_n,
            }
        return rescued

    @staticmethod
    def _item_collection_id(item: dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        direct = str(item.get("collection_id") or "").strip()
        if direct:
            return direct
        meta = item.get("metadata")
        if isinstance(meta, dict):
            meta_direct = str(meta.get("collection_id") or "").strip()
            if meta_direct:
                return meta_direct
            row = meta.get("row")
            if isinstance(row, dict):
                row_direct = str(row.get("collection_id") or "").strip()
                if row_direct:
                    return row_direct
                row_meta = row.get("metadata")
                if isinstance(row_meta, dict):
                    row_meta_direct = str(row_meta.get("collection_id") or "").strip()
                    if row_meta_direct:
                        return row_meta_direct
        return ""

    @staticmethod
    def _item_scope_tokens(item: dict[str, Any]) -> list[str]:
        if not isinstance(item, dict):
            return []

        tokens: list[str] = []

        def _push(value: Any) -> None:
            text = str(value or "").strip()
            if text:
                tokens.append(text.casefold())

        def _push_from_meta(meta: dict[str, Any]) -> None:
            _push(meta.get("source_standard"))
            _push(meta.get("standard"))
            standards = meta.get("source_standards")
            if isinstance(standards, list):
                for s in standards:
                    _push(s)

        meta = item.get("metadata")
        if isinstance(meta, dict):
            _push_from_meta(meta)
            row = meta.get("row")
            if isinstance(row, dict):
                _push(row.get("source_standard"))
                row_meta = row.get("metadata")
                if isinstance(row_meta, dict):
                    _push_from_meta(row_meta)

        direct = item.get("source_standard")
        _push(direct)
        return tokens

    @staticmethod
    def _stable_item_key(item: dict[str, Any]) -> str:
        source = str(item.get("source") or "")
        meta_raw = item.get("metadata")
        meta = meta_raw if isinstance(meta_raw, dict) else {}
        row_raw = meta.get("row")
        row = row_raw if isinstance(row_raw, dict) else {}
        row_meta_raw = row.get("metadata")
        row_meta = row_meta_raw if isinstance(row_meta_raw, dict) else {}
        doc_id = str(row.get("doc_id") or row_meta.get("doc_id") or "")
        chunk_id = str(
            row.get("chunk_id")
            or row.get("id")
            or row_meta.get("chunk_id")
            or row_meta.get("id")
            or ""
        )
        composite = "::".join(part for part in [doc_id, chunk_id, source] if part)
        return composite or source or str(hash(str(item)))

    @staticmethod
    def _scope_aliases(scope: str) -> tuple[str, ...]:
        value = str(scope or "").strip().upper()
        if not value:
            return ()
        aliases: list[str] = [value.casefold()]
        digits = [m for m in re.findall(r"\b\d{3,6}\b", value)]
        for digit in digits:
            aliases.append(str(digit).casefold())
        seen: set[str] = set()
        ordered: list[str] = []
        for alias in aliases:
            if alias in seen:
                continue
            seen.add(alias)
            ordered.append(alias)
        return tuple(ordered)

    @classmethod
    def _item_matches_scope(cls, item: dict[str, Any], scope: str) -> bool:
        aliases = cls._scope_aliases(scope)
        if not aliases:
            return False
        tokens = cls._item_scope_tokens(item)
        if not tokens:
            return False
        for token in tokens:
            for alias in aliases:
                if alias in token or token in alias:
                    return True
        return False

    @classmethod
    def _rebalance_scope_coverage(
        cls,
        *,
        items: list[dict[str, Any]],
        requested_standards: tuple[str, ...],
        top_k: int,
        trace_target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        requested = [
            str(scope or "").strip().upper() for scope in requested_standards if str(scope).strip()
        ]
        if len(requested) < 2:
            return items

        per_scope: dict[str, list[dict[str, Any]]] = {scope: [] for scope in requested}
        leftovers: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            matched = False
            for scope in requested:
                if cls._item_matches_scope(item, scope):
                    per_scope[scope].append(item)
                    matched = True
                    break
            if not matched:
                leftovers.append(item)

        selected: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def _append_unique(candidate: dict[str, Any]) -> None:
            key = cls._stable_item_key(candidate)
            if key in seen_keys:
                return
            seen_keys.add(key)
            selected.append(candidate)

        min_per_scope = 1
        for scope in requested:
            for candidate in per_scope.get(scope, [])[:min_per_scope]:
                _append_unique(candidate)

        cursor = {scope: min(len(per_scope.get(scope, [])), min_per_scope) for scope in requested}
        while len(selected) < max(1, top_k):
            progressed = False
            for scope in requested:
                bucket = per_scope.get(scope, [])
                idx = cursor.get(scope, 0)
                if idx >= len(bucket):
                    continue
                _append_unique(bucket[idx])
                cursor[scope] = idx + 1
                progressed = True
                if len(selected) >= max(1, top_k):
                    break
            if not progressed:
                break

        for candidate in leftovers:
            if len(selected) >= max(1, top_k):
                break
            _append_unique(candidate)

        if len(selected) < max(1, top_k):
            for candidate in items:
                if len(selected) >= max(1, top_k):
                    break
                if isinstance(candidate, dict):
                    _append_unique(candidate)

        if isinstance(trace_target, dict):
            trace_target["scope_balance"] = {
                "enabled": True,
                "requested_scopes": requested,
                "per_scope_counts_before": {
                    scope: len(per_scope.get(scope, [])) for scope in requested
                },
                "selected": len(selected),
                "top_k": int(max(1, top_k)),
            }

        return selected[: max(1, top_k)]

    @classmethod
    def _enforce_collection_scope(
        cls,
        items: list[dict[str, Any]],
        *,
        collection_id: str | None,
        requested_standards: tuple[str, ...] = (),
        trace_target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(trace_target, dict):
            trace_target["collection_scope"] = {
                "selected_collection_id": str(collection_id or "").strip(),
                "enforced": False,
                "reason": "collection_scope_filter_disabled",
                "requested_standards": list(requested_standards),
                "kept": len(items),
                "dropped": 0,
            }
        return [it for it in items if isinstance(it, dict)]

    @staticmethod
    def _contains_rate_limit_hint(payload: Any) -> bool:
        if isinstance(payload, str):
            text = payload.lower()
            return "rate_limit" in text or "429" in text
        if isinstance(payload, dict):
            return any(RetrievalFlow._contains_rate_limit_hint(v) for v in payload.values())
        if isinstance(payload, list):
            return any(RetrievalFlow._contains_rate_limit_hint(v) for v in payload)
        return False

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
        execute_started_at = time.perf_counter()
        total_budget_ms = max(
            1000, int(getattr(settings, "ORCH_TIMEOUT_EXECUTE_TOOL_MS", 30000) or 30000)
        )

        def remaining_budget_ms() -> int:
            elapsed = int((time.perf_counter() - execute_started_at) * 1000)
            return max(0, total_budget_ms - elapsed)

        def budgeted_timeout(default_timeout_ms: int) -> int:
            remaining = remaining_budget_ms()
            if remaining <= 300:
                return 200
            return max(200, min(default_timeout_ms, remaining - 200))

        filters = normalize_query_filters(validated_filters)
        if filters is None:
            filters = normalize_query_filters(
                {"source_standards": list(plan.requested_standards)}
                if plan.requested_standards
                else None
            )

        expanded_query, hint_trace = apply_search_hints(query, profile=self.profile_context)
        clause_refs = extract_clause_refs(query, profile=self.profile_context)
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
        global_max_subqueries = max(
            2,
            int(getattr(settings, "ORCH_PLANNER_MAX_QUERIES", 3) or 3),
        )
        mode_max_subqueries = max(2, min(mode_max_subqueries, global_max_subqueries))

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
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_MULTI_QUERY_MS", 25000) or 25000),
        )
        timeout_hybrid_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_HYBRID_MS", 25000) or 25000),
        )
        timeout_cov_repair_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COVERAGE_REPAIR_MS", 15000) or 15000),
        )

        async def build_subqueries(*, purpose: str = "primary") -> list[dict[str, Any]]:
            planner = self.subquery_planner or HybridSubqueryPlanner.from_settings()
            low_budget_cap = max(
                2,
                int(getattr(settings, "ORCH_RETRIEVAL_LOW_BUDGET_SUBQUERY_CAP", 2) or 2),
            )
            max_queries_for_plan = mode_max_subqueries
            if remaining_budget_ms() < 8000:
                scope_count = len(tuple(plan.requested_standards or ()))
                budget_floor = max(low_budget_cap, scope_count)
                max_queries_for_plan = min(max_queries_for_plan, budget_floor)
            subqueries_local = await planner.plan(
                SubqueryPlanningContext(
                    query=query,
                    requested_standards=plan.requested_standards,
                    max_queries=max_queries_for_plan,
                    mode=plan.mode,
                    require_literal_evidence=plan.require_literal_evidence,
                    include_semantic_tail=semantic_tail_enabled,
                    profile=self.profile_context,
                    decomposition_policy=decomposition_policy,
                )
            )
            seen_keys: set[str] = set()
            deduped: list[dict[str, Any]] = []
            for sq in subqueries_local:
                if not isinstance(sq, dict):
                    continue
                key = str(sq.get("id") or sq.get("query") or "").strip().lower()
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                raw_filters = sq.get("filters")
                normalized_filters = normalize_query_filters(
                    raw_filters if isinstance(raw_filters, dict) else None
                )
                if normalized_filters is None:
                    sq.pop("filters", None)
                else:
                    sq["filters"] = normalized_filters
                deduped.append(sq)

            subqueries_local = deduped
            fallback_max = max(
                2,
                int(getattr(settings, "ORCH_MULTI_QUERY_FALLBACK_MAX_QUERIES", 3) or 3),
            )
            fallback_max = max(fallback_max, len(tuple(plan.requested_standards or ())))
            fallback_max = min(mode_max_subqueries, fallback_max)
            if purpose == "fallback" and len(subqueries_local) > fallback_max:
                subqueries_local = subqueries_local[:fallback_max]
            return subqueries_local

        # 1) Hybrid first
        hybrid_payload, hybrid_error_code, hybrid_error_detail = await self._safe_execute(
            op_name="hybrid_primary",
            timeout_ms=budgeted_timeout(timeout_hybrid_ms),
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
            timings_ms=timings_ms,
        )
        items = hybrid_payload.get("items") if isinstance(hybrid_payload, dict) else []
        trace = hybrid_payload.get("trace") if isinstance(hybrid_payload, dict) else {}
        if not isinstance(items, list):
            items = []
        if not isinstance(trace, dict):
            trace = {}
        raw_hybrid_items = [it for it in items if isinstance(it, dict)]
        items = self._filter_items_by_min_score(
            raw_hybrid_items,
            trace_target=trace,
        )
        items = self._ensure_non_empty_candidates(
            raw_items=raw_hybrid_items,
            filtered_items=items,
            trace_target=trace,
            stage="hybrid_primary",
        )
        items = self._enforce_collection_scope(
            items,
            collection_id=collection_id,
            requested_standards=plan.requested_standards,
            trace_target=trace,
        )
        items = self._rebalance_scope_coverage(
            items=items,
            requested_standards=plan.requested_standards,
            top_k=max(12, k),
            trace_target=trace,
        )
        if hint_trace.get("applied"):
            trace["search_hint_expansions"] = hint_trace
        if hybrid_error_code:
            trace["error_codes"] = merge_error_codes(trace.get("error_codes"), [hybrid_error_code])
            trace["hybrid_primary_error"] = hybrid_error_detail
        if isinstance(self.profile_resolution_context, dict):
            trace["agent_profile_resolution"] = dict(self.profile_resolution_context)

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
        min_items = max(1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6))
        insufficient_reasons: list[str] = []
        if not items:
            insufficient_reasons.append("empty_hybrid")
        if len(items) < min_items:
            insufficient_reasons.append("low_hybrid_count")
        if missing_scopes:
            insufficient_reasons.append("missing_scopes")
        if missing_clauses:
            insufficient_reasons.append("missing_clause_refs")

        should_use_multi_query = bool(insufficient_reasons) and int(plan.chunk_k or 0) > 0
        fallback_code = RETRIEVAL_CODE_SCOPE_MISMATCH if missing_scopes else None

        if (
            not should_use_multi_query
            and settings.ORCH_MULTIHOP_FALLBACK
            and int(plan.chunk_k or 0) > 0
        ):
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
                should_use_multi_query = True
                insufficient_reasons.append(str(decision.reason or "hybrid_semantic_gap"))
                fallback_code = decision.code or RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP

        if should_use_multi_query:
            min_mq_budget_ms = max(
                500,
                int(getattr(settings, "ORCH_RETRIEVAL_MIN_MQ_BUDGET_MS", 1200) or 1200),
            )
            if remaining_budget_ms() < min_mq_budget_ms:
                should_use_multi_query = False
                trace["multi_query_fallback_skipped_by_budget"] = {
                    "remaining_budget_ms": remaining_budget_ms(),
                    "min_budget_ms": min_mq_budget_ms,
                }
                logger.warning(
                    "multi_query_fallback_skipped_by_budget",
                    remaining_budget_ms=remaining_budget_ms(),
                    min_budget_ms=min_mq_budget_ms,
                )

        if should_use_multi_query:
            subqueries = await build_subqueries(purpose="fallback")
            if self._contains_rate_limit_hint(hybrid_payload) or self._contains_rate_limit_hint(
                trace
            ):
                rate_limit_cap = max(
                    2,
                    int(getattr(settings, "ORCH_RETRIEVAL_RATE_LIMIT_SUBQUERY_CAP", 3) or 3),
                )
                subqueries = subqueries[:rate_limit_cap]
                trace["subquery_cap_reason"] = "rate_limit_hint"
                trace["subquery_cap_applied"] = rate_limit_cap
            if missing_scopes:
                for scope in missing_scopes:
                    subqueries.insert(
                        0,
                        {
                            "id": f"missing_scope_{scope[:24].replace(' ', '_').lower()}",
                            "query": f"{expanded_query} {scope}",
                            "k": None,
                            "fetch_k": None,
                            "filters": normalize_query_filters({"source_standard": scope}),
                        },
                    )
            if missing_clauses:
                for clause in missing_clauses[:2]:
                    clause_filters = normalize_query_filters(
                        {
                            "source_standard": plan.requested_standards[0],
                            "metadata": {"clause_id": clause},
                        }
                        if len(plan.requested_standards) == 1 and plan.requested_standards[0]
                        else {"source_standards": list(plan.requested_standards)}
                    )
                    subqueries.insert(
                        0,
                        {
                            "id": f"missing_clause_{str(clause).replace('.', '_')}",
                            "query": f"{expanded_query} clausula {clause}"[:900],
                            "k": None,
                            "fetch_k": None,
                            "filters": clause_filters,
                        },
                    )
            subqueries = subqueries[:mode_max_subqueries]

            merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
            mq_timeout_ms = budgeted_timeout(timeout_multi_query_ms)
            use_client_fanout = bool(
                getattr(settings, "ORCH_MULTI_QUERY_CLIENT_FANOUT_ENABLED", False)
            )
            if use_client_fanout:
                mq_payload, mq_error_code, mq_error_detail = await self._execute_multi_query_fanout(
                    op_name="multi_query_fallback",
                    timeout_ms=mq_timeout_ms,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=subqueries,
                    default_k=k,
                    default_fetch_k=fetch_k,
                    timings_ms=timings_ms,
                )
            else:
                mq_payload, mq_error_code, mq_error_detail = await self._safe_execute(
                    op_name="multi_query_fallback",
                    timeout_ms=mq_timeout_ms,
                    operation=contract_client.multi_query(
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        queries=subqueries,
                        merge=merge,
                    ),
                    timings_ms=timings_ms,
                )
            mq_items_raw = mq_payload.get("items") if isinstance(mq_payload, dict) else []
            mq_items = mq_items_raw if isinstance(mq_items_raw, list) else []
            mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
            partial = (
                bool(mq_payload.get("partial", False)) if isinstance(mq_payload, dict) else False
            )
            subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
            subq_groups = (
                mq_payload.get("subquery_groups") if isinstance(mq_payload, dict) else None
            )

            planned_subquery_by_id: dict[str, dict[str, Any]] = {}
            for planned in subqueries:
                if not isinstance(planned, dict):
                    continue
                sq_id = str(planned.get("id") or "").strip()
                if sq_id:
                    planned_subquery_by_id[sq_id] = planned

            if isinstance(subq, list):
                for entry in subq:
                    if not isinstance(entry, dict):
                        continue
                    sq_id = str(entry.get("id") or "").strip()
                    if not sq_id:
                        continue
                    if str(entry.get("query") or "").strip():
                        continue
                    planned = planned_subquery_by_id.get(sq_id)
                    if isinstance(planned, dict):
                        entry["query"] = str(planned.get("query") or "").strip()[:280]

            if mq_error_code:
                diag_trace_local = {
                    "strategy_path": "hybrid_then_multi_query_failed",
                    "multi_query_execution_mode": (
                        "client_fanout" if use_client_fanout else "contract_multi_query"
                    ),
                    "hybrid_trace": trace,
                    "timings_ms": dict(timings_ms),
                    "fallback_reason": insufficient_reasons,
                    "error_codes": merge_error_codes(
                        trace.get("error_codes"),
                        [mq_error_code],
                    ),
                    "multi_query_fallback_error": mq_error_detail,
                }
                if fallback_code:
                    diag_trace_local["error_codes"] = merge_error_codes(
                        diag_trace_local.get("error_codes"),
                        [fallback_code],
                    )
                self.last_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="hybrid",
                    partial=True,
                    trace=diag_trace_local,
                    scope_validation=validated_scope_payload or {},
                )
                coverage_timeout_ms, allow_step_back = self._coverage_budget(
                    default_timeout_ms=timeout_cov_repair_ms,
                    remaining_budget_ms=remaining_budget_ms(),
                )
                if coverage_timeout_ms is None:
                    diag_trace_local["coverage_gate_skipped_by_budget"] = {
                        "remaining_budget_ms": remaining_budget_ms(),
                    }
                    items = [it for it in items if isinstance(it, dict)]
                else:
                    items = await self._execute_coverage_repair(
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
                        timeout_cov_repair_ms=coverage_timeout_ms,
                        allow_step_back=allow_step_back,
                        k=k,
                    )
                raw_local_items = [it for it in items if isinstance(it, dict)]
                items = self._filter_items_by_min_score(
                    raw_local_items,
                    trace_target=diag_trace_local,
                )
                items = self._ensure_non_empty_candidates(
                    raw_items=raw_local_items,
                    filtered_items=items,
                    trace_target=diag_trace_local,
                    stage="hybrid_local_repair",
                )
                items = self._enforce_collection_scope(
                    items,
                    collection_id=collection_id,
                    requested_standards=plan.requested_standards,
                    trace_target=diag_trace_local,
                )
                items = reduce_structural_noise(items, query)
                items = self._rebalance_scope_coverage(
                    items=items,
                    requested_standards=plan.requested_standards,
                    top_k=max(12, k),
                    trace_target=diag_trace_local,
                )
                return self._to_evidence(items)

            raw_mq_items = [it for it in mq_items if isinstance(it, dict)]
            mq_items = self._filter_items_by_min_score(
                raw_mq_items,
                trace_target=mq_trace if isinstance(mq_trace, dict) else None,
            )
            mq_items = self._ensure_non_empty_candidates(
                raw_items=raw_mq_items,
                filtered_items=mq_items,
                trace_target=mq_trace if isinstance(mq_trace, dict) else None,
                stage="multi_query_fallback",
            )
            mq_items = self._enforce_collection_scope(
                mq_items,
                collection_id=collection_id,
                requested_standards=plan.requested_standards,
                trace_target=mq_trace if isinstance(mq_trace, dict) else None,
            )
            mq_items = reduce_structural_noise(mq_items, query)
            mq_items = self._rebalance_scope_coverage(
                items=mq_items,
                requested_standards=plan.requested_standards,
                top_k=max(12, k),
                trace_target=mq_trace if isinstance(mq_trace, dict) else None,
            )
            merged_items = self._merge_and_deduplicate(items, mq_items)

            diag_trace_fb: dict[str, Any] = {
                "strategy_path": "hybrid_then_multi_query",
                "fallback_reason": insufficient_reasons,
                "error_codes": merge_error_codes(
                    trace.get("error_codes"),
                    [fallback_code] if fallback_code else [],
                ),
                "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                "multi_query_execution_mode": (
                    "client_fanout" if use_client_fanout else "contract_multi_query"
                ),
                "mode_policy": {
                    "require_all_requested_scopes": require_all_scopes,
                    "min_clause_refs": min_clause_refs_required,
                    "max_subqueries": mode_max_subqueries,
                },
                "hybrid_trace": trace,
                "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                "subqueries": subq if isinstance(subq, list) else [],
                "subquery_groups": subq_groups if isinstance(subq_groups, list) else [],
                "timings_ms": dict(timings_ms),
                "rag_features": features_from_hybrid_trace(trace),
            }
            if hint_trace.get("applied"):
                diag_trace_fb["search_hint_expansions"] = hint_trace
            if isinstance(self.profile_resolution_context, dict):
                diag_trace_fb["agent_profile_resolution"] = dict(self.profile_resolution_context)
            if isinstance(merged_items, list):
                diag_trace_fb.update(
                    calculate_layer_stats([it for it in merged_items if isinstance(it, dict)])
                )

            coverage_timeout_ms, allow_step_back = self._coverage_budget(
                default_timeout_ms=timeout_cov_repair_ms,
                remaining_budget_ms=remaining_budget_ms(),
            )
            if coverage_timeout_ms is None:
                diag_trace_fb["coverage_gate_skipped_by_budget"] = {
                    "remaining_budget_ms": remaining_budget_ms(),
                }
                repaired_items = [it for it in merged_items if isinstance(it, dict)]
            else:
                repaired_items = await self._execute_coverage_repair(
                    items=[it for it in merged_items if isinstance(it, dict)],
                    base_trace=diag_trace_fb,
                    reason="hybrid_then_multi_query",
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
                    timeout_cov_repair_ms=coverage_timeout_ms,
                    allow_step_back=allow_step_back,
                    k=k,
                )
            raw_repaired_items = [it for it in repaired_items if isinstance(it, dict)]
            repaired_items = self._filter_items_by_min_score(
                raw_repaired_items,
                trace_target=diag_trace_fb,
            )
            repaired_items = self._ensure_non_empty_candidates(
                raw_items=raw_repaired_items,
                filtered_items=repaired_items,
                trace_target=diag_trace_fb,
                stage="multi_query_repair",
            )
            repaired_items = self._enforce_collection_scope(
                repaired_items,
                collection_id=collection_id,
                requested_standards=plan.requested_standards,
                trace_target=diag_trace_fb,
            )
            repaired_items = reduce_structural_noise(repaired_items, query)
            repaired_items = self._rebalance_scope_coverage(
                items=repaired_items,
                requested_standards=plan.requested_standards,
                top_k=max(12, k),
                trace_target=diag_trace_fb,
            )
            self.last_diagnostics = RetrievalDiagnostics(
                contract="advanced",
                strategy="multi_query",
                partial=partial,
                trace=diag_trace_fb,
                scope_validation=validated_scope_payload or {},
            )
            return self._to_evidence(repaired_items)

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
            diag_trace_final.update(
                calculate_layer_stats([it for it in items if isinstance(it, dict)])
            )
            diag_trace_final["rag_features"] = features_from_hybrid_trace(trace)
            if hint_trace.get("applied"):
                diag_trace_final["search_hint_expansions"] = hint_trace
            if isinstance(self.profile_resolution_context, dict):
                diag_trace_final["agent_profile_resolution"] = dict(self.profile_resolution_context)

        coverage_timeout_ms, allow_step_back = self._coverage_budget(
            default_timeout_ms=timeout_cov_repair_ms,
            remaining_budget_ms=remaining_budget_ms(),
        )
        if coverage_timeout_ms is None:
            diag_trace_final["coverage_gate_skipped_by_budget"] = {
                "remaining_budget_ms": remaining_budget_ms(),
            }
            items2 = [it for it in items if isinstance(it, dict)]
        else:
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
                timeout_cov_repair_ms=coverage_timeout_ms,
                allow_step_back=allow_step_back,
                k=k,
            )
        raw_items2 = [it for it in items2 if isinstance(it, dict)]
        items2 = self._filter_items_by_min_score(
            raw_items2,
            trace_target=diag_trace_final,
        )
        items2 = self._ensure_non_empty_candidates(
            raw_items=raw_items2,
            filtered_items=items2,
            trace_target=diag_trace_final,
            stage="hybrid_final",
        )
        items2 = self._enforce_collection_scope(
            items2,
            collection_id=collection_id,
            requested_standards=plan.requested_standards,
            trace_target=diag_trace_final,
        )
        items2 = reduce_structural_noise(items2, query)
        items2 = self._rebalance_scope_coverage(
            items=items2,
            requested_standards=plan.requested_standards,
            top_k=max(12, k),
            trace_target=diag_trace_final,
        )
        return self._to_evidence(items2)

    def _coverage_budget(
        self,
        *,
        default_timeout_ms: int,
        remaining_budget_ms: int,
    ) -> tuple[int | None, bool]:
        min_repair_budget_ms = max(
            400,
            int(getattr(settings, "ORCH_RETRIEVAL_MIN_REPAIR_BUDGET_MS", 800) or 800),
        )
        if remaining_budget_ms < min_repair_budget_ms:
            return None, False
        step_back_min_budget_ms = max(
            min_repair_budget_ms,
            int(getattr(settings, "ORCH_COVERAGE_GATE_STEP_BACK_MIN_BUDGET_MS", 1200) or 1200),
        )
        allow_step_back = remaining_budget_ms >= step_back_min_budget_ms
        timeout_ms = max(200, min(default_timeout_ms, remaining_budget_ms - 200))
        return timeout_ms, allow_step_back

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

    async def _execute_multi_query_fanout(
        self,
        *,
        op_name: str,
        timeout_ms: int,
        tenant_id: str,
        collection_id: str | None,
        user_id: str | None,
        request_id: str | None,
        correlation_id: str | None,
        queries: list[dict[str, Any]],
        default_k: int,
        default_fetch_k: int,
        timings_ms: dict[str, float],
    ) -> tuple[dict[str, Any], str | None, str | None]:
        max_parallel = max(
            1,
            int(getattr(settings, "ORCH_MULTI_QUERY_CLIENT_FANOUT_MAX_PARALLEL", 2) or 2),
        )
        per_query_timeout_ms = max(
            200,
            int(
                getattr(settings, "ORCH_MULTI_QUERY_CLIENT_FANOUT_PER_QUERY_TIMEOUT_MS", 8000)
                or 8000
            ),
        )
        per_query_timeout_ms = min(per_query_timeout_ms, timeout_ms)
        rerank_enabled = bool(
            getattr(settings, "ORCH_MULTI_QUERY_CLIENT_FANOUT_RERANK_ENABLED", False)
        )
        graph_max_hops = max(
            0,
            int(getattr(settings, "ORCH_MULTI_QUERY_CLIENT_FANOUT_GRAPH_MAX_HOPS", 2) or 2),
        )

        semaphore = asyncio.Semaphore(max_parallel)
        t0 = time.perf_counter()

        async def run_subquery(idx: int, subquery: dict[str, Any]) -> dict[str, Any]:
            sq_id = str(subquery.get("id") or f"q{idx + 1}")
            sq_query = str(subquery.get("query") or "").strip()
            if not sq_query:
                return {
                    "id": sq_id,
                    "query": "",
                    "ok": False,
                    "error_code": RETRIEVAL_CODE_INVALID_RESPONSE,
                    "error_detail": "empty_query",
                    "items": [],
                }

            try:
                sq_k = int(subquery.get("k") or default_k)
            except (TypeError, ValueError):
                sq_k = default_k
            try:
                sq_fetch_k = int(subquery.get("fetch_k") or default_fetch_k)
            except (TypeError, ValueError):
                sq_fetch_k = default_fetch_k
            sq_filters = normalize_query_filters(
                subquery.get("filters") if isinstance(subquery.get("filters"), dict) else None
            )

            async with semaphore:
                payload, error_code, error_detail = await self._safe_execute(
                    op_name=f"{op_name}_fanout_{idx + 1}",
                    timeout_ms=per_query_timeout_ms,
                    operation=self.contract_client.hybrid(
                        query=sq_query,
                        tenant_id=tenant_id,
                        collection_id=collection_id,
                        user_id=user_id,
                        request_id=request_id,
                        correlation_id=correlation_id,
                        k=max(1, sq_k),
                        fetch_k=max(1, sq_fetch_k),
                        filters=sq_filters,
                        rerank={"enabled": rerank_enabled},
                        graph={"max_hops": graph_max_hops},
                    ),
                    timings_ms=timings_ms,
                )

            sq_items_raw = payload.get("items") if isinstance(payload, dict) else []
            sq_items = sq_items_raw if isinstance(sq_items_raw, list) else []
            return {
                "id": sq_id,
                "query": sq_query,
                "ok": error_code is None,
                "error_code": error_code,
                "error_detail": error_detail,
                "items": [it for it in sq_items if isinstance(it, dict)],
            }

        tasks = [asyncio.create_task(run_subquery(idx, sq)) for idx, sq in enumerate(queries)]
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=max(0.2, timeout_ms / 1000.0),
            )
        except TimeoutError:
            for task in tasks:
                task.cancel()
            timings_ms[op_name] = round((time.perf_counter() - t0) * 1000, 2)
            return {}, RETRIEVAL_CODE_TIMEOUT, f"retrieval_timeout:{op_name}_fanout"

        timings_ms[op_name] = round((time.perf_counter() - t0) * 1000, 2)

        all_items: list[dict[str, Any]] = []
        subqueries_trace: list[dict[str, Any]] = []
        subquery_groups: list[dict[str, Any]] = []
        error_codes: list[str] = []
        first_error_code: str | None = None
        first_error_detail: str | None = None
        for result in results:
            sq_items = [it for it in result.get("items") or [] if isinstance(it, dict)]
            all_items.extend(sq_items)
            sq_query = str(result.get("query") or "").strip()
            sq_error_code = result.get("error_code")
            sq_error_detail = result.get("error_detail")
            if isinstance(sq_error_code, str) and sq_error_code:
                error_codes.append(sq_error_code)
                if first_error_code is None:
                    first_error_code = sq_error_code
                    first_error_detail = (
                        str(sq_error_detail)[:160] if isinstance(sq_error_detail, str) else None
                    )
            subqueries_trace.append(
                {
                    "id": str(result.get("id") or ""),
                    "query": sq_query[:280],
                    "ok": bool(result.get("ok")),
                    "item_count": len(sq_items),
                    "error_code": sq_error_code,
                }
            )
            subquery_groups.append(
                {
                    "id": str(result.get("id") or ""),
                    "query": sq_query[:520],
                    "ok": bool(result.get("ok")),
                    "error_code": sq_error_code,
                    "items": sq_items[:10],
                }
            )

        merged_items = self._merge_and_deduplicate([], all_items)
        has_success = any(bool(r.get("ok")) for r in results)
        partial = any(not bool(r.get("ok")) for r in results)
        payload: dict[str, Any] = {
            "items": merged_items,
            "partial": partial,
            "subqueries": subqueries_trace,
            "subquery_groups": subquery_groups,
            "trace": {
                "fanout": True,
                "max_parallel": max_parallel,
                "rerank_enabled": rerank_enabled,
                "graph_max_hops": graph_max_hops,
                "error_codes": merge_error_codes([], error_codes),
            },
        }
        if has_success:
            return payload, None, None
        return payload, first_error_code or RETRIEVAL_CODE_INVALID_RESPONSE, first_error_detail

    @staticmethod
    def _merge_and_deduplicate(
        base_items: list[dict[str, Any]],
        new_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()

        for candidate in [*base_items, *new_items]:
            if not isinstance(candidate, dict):
                continue
            key = RetrievalFlow._stable_item_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)

        return merged

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
        allow_step_back: bool,
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
        if len(plan.requested_standards) >= 2:
            cap = max(cap, len(plan.requested_standards))
        missing_scopes = missing_scopes[:cap]
        missing_clauses = missing_clauses[:cap]

        focused: list[dict[str, Any]] = []
        for idx, scope in enumerate(missing_scopes, start=1):
            qtext = " ".join(
                part for part in [scope, *clause_refs[:3], expanded_query] if part
            ).strip()
            focused.append(
                {
                    "id": f"scope_repair_{idx}",
                    "query": qtext[:900],
                    "k": None,
                    "fetch_k": None,
                    "filters": normalize_query_filters({"source_standard": scope}),
                }
            )
        for idx, clause in enumerate(missing_clauses, start=1):
            clause_filters = normalize_query_filters(
                {"source_standard": plan.requested_standards[0], "metadata": {"clause_id": clause}}
                if len(plan.requested_standards) == 1 and plan.requested_standards[0]
                else {"source_standards": list(plan.requested_standards)}
            )
            focused.append(
                {
                    "id": f"clause_repair_{idx}",
                    "query": f"{expanded_query} clausula {clause}"[:900],
                    "k": None,
                    "fetch_k": None,
                    "filters": clause_filters,
                }
            )

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
        except Exception as cov_exc:
            # Coverage repair is best-effort: if it fails, return original items
            timings_ms["coverage_gate"] = round((time.perf_counter() - t_cov) * 1000, 2)
            base_trace["coverage_gate"] = {
                "trigger_reason": reason,
                "missing_scopes": missing_scopes,
                "missing_clause_refs": missing_clauses,
                "error": str(cov_exc)[:160],
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
            if missing_scopes:
                codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
            if missing_clauses:
                codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
            base_trace["error_codes"] = merge_error_codes(base_trace.get("error_codes"), codes)
            return items

        # Merge and dedupe
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for it in [*items, *cov_items]:
            if not isinstance(it, dict):
                continue
            key = self._stable_item_key(it)
            if key in seen:
                continue
            seen.add(key)
            merged.append(it)

        base_trace["coverage_gate"] = {
            "trigger_reason": reason,
            "missing_scopes": missing_scopes,
            "missing_clause_refs": missing_clauses,
            "added_queries": [q.get("id") for q in focused],
        }

        # Step-back pass
        remaining = find_missing_scopes(
            merged, plan.requested_standards, enforce=require_all_scopes
        )
        remaining_clauses = find_missing_clause_refs(
            merged, clause_refs, min_required=min_clause_refs_required
        )

        if (
            (remaining or remaining_clauses)
            and settings.ORCH_COVERAGE_GATE_STEP_BACK
            and allow_step_back
        ):
            remaining = remaining[:cap]
            remaining_clauses = remaining_clauses[:cap]
            step_back_queries: list[dict[str, Any]] = []
            for idx, scope in enumerate(remaining, start=1):
                step_back_queries.append(
                    {
                        "id": f"scope_step_back_{idx}",
                        "query": f"principios generales y requisitos clave relacionados con: {expanded_query}",
                        "k": None,
                        "fetch_k": None,
                        "filters": normalize_query_filters({"source_standard": scope}),
                    }
                )
            for idx, clause in enumerate(remaining_clauses, start=1):
                clause_filters = normalize_query_filters(
                    {
                        "source_standard": plan.requested_standards[0],
                        "metadata": {"clause_id": clause},
                    }
                    if len(plan.requested_standards) == 1 and plan.requested_standards[0]
                    else {"source_standards": list(plan.requested_standards)}
                )
                step_back_queries.append(
                    {
                        "id": f"clause_step_back_{idx}",
                        "query": f"principios generales y requisitos clave relacionados con: {expanded_query} clausula {clause}"[
                            :900
                        ],
                        "k": None,
                        "fetch_k": None,
                        "filters": clause_filters,
                    }
                )
            t_sb = time.perf_counter()
            try:
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
            except Exception as step_back_exc:
                timings_ms["coverage_gate_step_back"] = round(
                    (time.perf_counter() - t_sb) * 1000, 2
                )
                logger.warning(
                    "coverage_gate_step_back_failed",
                    error=str(step_back_exc)[:160],
                    tenant_id=tenant_id,
                )
                if isinstance(base_trace.get("coverage_gate"), dict):
                    base_trace["coverage_gate"]["step_back_error"] = str(step_back_exc)[:160]
                    base_trace["coverage_gate"]["step_back_queries"] = [
                        q.get("id") for q in step_back_queries
                    ]
            else:
                timings_ms["coverage_gate_step_back"] = round(
                    (time.perf_counter() - t_sb) * 1000, 2
                )
                sb_items = sb_payload.get("items") if isinstance(sb_payload, dict) else []
                if isinstance(sb_items, list) and sb_items:
                    for it in sb_items:
                        if not isinstance(it, dict):
                            continue
                        key = self._stable_item_key(it)
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(it)
                    if isinstance(base_trace.get("coverage_gate"), dict):
                        base_trace["coverage_gate"]["step_back_missing_scopes"] = remaining
                        base_trace["coverage_gate"]["step_back_missing_clause_refs"] = (
                            remaining_clauses
                        )
                        base_trace["coverage_gate"]["step_back_queries"] = [
                            q.get("id") for q in step_back_queries
                        ]

        if (
            (remaining or remaining_clauses)
            and settings.ORCH_COVERAGE_GATE_STEP_BACK
            and not allow_step_back
        ):
            if isinstance(base_trace.get("coverage_gate"), dict):
                base_trace["coverage_gate"]["step_back_skipped_by_budget"] = True

        merged = self._enforce_collection_scope(
            [it for it in merged if isinstance(it, dict)],
            collection_id=collection_id,
            requested_standards=plan.requested_standards,
            trace_target=base_trace,
        )

        final_missing = find_missing_scopes(
            merged, plan.requested_standards, enforce=require_all_scopes
        )
        final_missing_clauses = find_missing_clause_refs(
            merged, clause_refs, min_required=min_clause_refs_required
        )

        if isinstance(base_trace.get("coverage_gate"), dict):
            base_trace["coverage_gate"]["final_missing_scopes"] = final_missing
            base_trace["coverage_gate"]["final_missing_clause_refs"] = final_missing_clauses

        base_trace["missing_scopes"] = list(final_missing)
        base_trace["missing_clause_refs"] = list(final_missing_clauses)
        codes = []
        if final_missing:
            codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
        if final_missing_clauses:
            codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
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
