from __future__ import annotations

import asyncio
import re
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
from app.agent.retrieval_filters import filter_items_by_min_score
from app.agent.interfaces import (
    EmbeddingProvider,
    RerankingProvider,
    SubqueryPlanner,
)
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig
from app.agent.retrieval_planner import (
    apply_search_hints,
    extract_clause_refs,
    mode_requires_literal_evidence,
    normalize_query_filters,
)
from app.core.config import settings
from app.core.rag_retrieval_contract_client import RagRetrievalContractClient
from app.agent.retrieval_strategies import (
    calculate_layer_stats,
    features_from_hybrid_trace,
    reduce_structural_noise,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetrievalExecutionContext:
    filters: dict[str, Any] | None
    expanded_query: str
    hint_trace: dict[str, Any]
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

        expanded_query, hint_trace = apply_search_hints(query, profile=self.profile_context)
        _ = extract_clause_refs(query, profile=self.profile_context)
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
        k_cap = 24 if cross_scope_mode else 18
        k = max(1, min(int(plan.chunk_k), k_cap))
        fetch_k = max(1, int(plan.chunk_fetch_k))

        timeout_comprehensive_ms = max(
            200,
            int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COMPREHENSIVE_MS", 28000) or 28000),
        )

        return RetrievalExecutionContext(
            filters=filters,
            expanded_query=expanded_query,
            hint_trace=hint_trace,
            require_all_scopes=require_all_scopes,
            min_clause_refs_required=min_clause_refs_required,
            k=k,
            fetch_k=fetch_k,
            timeout_comprehensive_ms=timeout_comprehensive_ms,
        )

    async def _execute_comprehensive_primary(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        user_id: str | None,
        request_id: str | None,
        correlation_id: str | None,
        plan: RetrievalPlan,
        context: RetrievalExecutionContext,
        timings_ms: dict[str, float],
        budgeted_timeout_fn: Callable[[int], int],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], str | None, str | None]:
        comprehensive_payload, error_code, error_detail = await self._safe_execute(
            op_name="comprehensive_primary",
            timeout_ms=budgeted_timeout_fn(context.timeout_comprehensive_ms),
            operation=self.contract_client.comprehensive(
                query=context.expanded_query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
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
        refined = self._refine_candidates(
            raw_items=[it for it in items if isinstance(it, dict)],
            trace_target=trace,
            stage="comprehensive_primary",
            query=query,
            collection_id=collection_id,
            requested_standards=plan.requested_standards,
            top_k=max(12, context.k),
            reduce_noise=True,
        )
        return refined, trace, error_code, error_detail

    def _refine_candidates(
        self,
        *,
        raw_items: list[dict[str, Any]],
        trace_target: dict[str, Any] | None,
        stage: str,
        query: str,
        collection_id: str | None,
        requested_standards: tuple[str, ...],
        top_k: int,
        reduce_noise: bool,
    ) -> list[dict[str, Any]]:
        items = self._filter_items_by_min_score(raw_items, trace_target=trace_target)
        items = self._ensure_non_empty_candidates(
            raw_items=raw_items,
            filtered_items=items,
            trace_target=trace_target,
            stage=stage,
        )
        items = self._enforce_collection_scope(
            items,
            collection_id=collection_id,
            requested_standards=requested_standards,
            trace_target=trace_target,
        )
        if reduce_noise:
            items = reduce_structural_noise(items, query)
        items = self._rebalance_scope_coverage(
            items=items,
            requested_standards=requested_standards,
            top_k=top_k,
            trace_target=trace_target,
        )
        return items

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

        def budgeted_timeout(default_timeout_ms: int) -> int:
            elapsed = int((time.perf_counter() - execute_started_at) * 1000)
            remaining = max(0, total_budget_ms - elapsed)
            if remaining <= 300:
                return 200
            return max(200, min(default_timeout_ms, remaining - 200))

        context = self._prepare_execution_context(
            query=query,
            plan=plan,
            validated_filters=validated_filters,
        )
        timings_ms: dict[str, float] = {}

        (
            items,
            trace,
            comprehensive_error_code,
            comprehensive_error_detail,
        ) = await self._execute_comprehensive_primary(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
            plan=plan,
            context=context,
            timings_ms=timings_ms,
            budgeted_timeout_fn=budgeted_timeout,
        )
        if comprehensive_error_code:
            raise RuntimeError(
                f"comprehensive_retrieval_failed:{comprehensive_error_code}:{comprehensive_error_detail or ''}"
            )

        if context.hint_trace.get("applied"):
            trace["search_hint_expansions"] = context.hint_trace
        if isinstance(self.profile_resolution_context, dict):
            trace["agent_profile_resolution"] = dict(self.profile_resolution_context)
        trace.setdefault("strategy_path", "comprehensive")
        trace["timings_ms"] = {**dict(trace.get("timings_ms") or {}), **timings_ms}

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
