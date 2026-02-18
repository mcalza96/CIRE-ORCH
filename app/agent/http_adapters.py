from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from app.agent.error_codes import (
    RETRIEVAL_CODE_LOW_SCORE,
    merge_error_codes,
)
from app.agent.retrieval_planner import (
    mode_requires_literal_evidence,
)
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.rag_retrieval_contract_client import (
    RagRetrievalContractClient,
)
from app.core.retrieval_metrics import retrieval_metrics_store

# New Strategy Imports
from app.agent.retrieval_flow import RetrievalFlow
from app.agent.interfaces import SubqueryPlanner
from app.agent.components.query_decomposer import HybridSubqueryPlanner
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig


logger = structlog.get_logger(__name__)


@dataclass
class RagEngineRetrieverAdapter:
    base_url: str | None = None
    timeout_seconds: float = 45.0
    backend_selector: RagBackendSelector | None = None
    contract_client: RagRetrievalContractClient | None = None
    http_client: httpx.AsyncClient | None = None
    subquery_planner: SubqueryPlanner | None = None

    # last diagnostics are read by the use case (duck typing).
    last_retrieval_diagnostics: RetrievalDiagnostics | None = None
    _validated_filters: dict[str, Any] | None = None
    _validated_scope_payload: dict[str, Any] | None = None
    _profile_context: AgentProfile | None = None
    _profile_resolution_context: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.backend_selector is None:
            self.backend_selector = RagBackendSelector(
                local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
                docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
                health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
                probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
                ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
                force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
            )

        if not settings.RAG_SERVICE_SECRET:
            logger.error("rag_service_secret_missing", status="error")
            raise RuntimeError("RAG_SERVICE_SECRET must be configured for production security")

        if self.contract_client is None:
            self.contract_client = RagRetrievalContractClient(
                timeout_seconds=self.timeout_seconds,
                backend_selector=self.backend_selector,
                http_client=self.http_client,
            )
        if self.subquery_planner is None:
            self.subquery_planner = HybridSubqueryPlanner.from_settings()

    def set_profile_context(
        self,
        *,
        profile: AgentProfile | None,
        profile_resolution: dict[str, Any] | None = None,
    ) -> None:
        self._profile_context = profile
        self._profile_resolution_context = (
            profile_resolution if isinstance(profile_resolution, dict) else None
        )

    def _profile_min_score(self) -> float | None:
        if self._profile_context is None:
            return None
        try:
            return float(self._profile_context.retrieval.min_score)
        except Exception:
            return None

    def _mode_config(self, mode: str) -> QueryModeConfig | None:
        if self._profile_context is None:
            return None
        cfg = self._profile_context.query_modes.modes.get(str(mode or "").strip())
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

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert self.contract_client is not None
        payload = await self.contract_client.validate_scope(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
            filters=filters,
        )
        if isinstance(payload, dict):
            self._validated_scope_payload = payload
        return payload if isinstance(payload, dict) else {}

    def apply_validated_scope(self, validated: dict[str, Any]) -> None:
        normalized = validated.get("normalized_scope") if isinstance(validated, dict) else None
        filters = normalized.get("filters") if isinstance(normalized, dict) else None
        self._validated_filters = filters if isinstance(filters, dict) else None
        self._validated_scope_payload = validated

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        return await self._retrieve_advanced(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            plan=plan,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
        )

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        if not settings.ORCH_RAPTOR_SUMMARIES_ENABLED:
            return []
        if int(plan.summary_k or 0) <= 0:
            return []

        # Optional: Advanced contract typically handles everything in hybrid/mq,
        # but if we keep specific summary calls, they should also go through advanced contract if available.
        # For now, following the plan to simplify and favor the advanced orchestration.
        return []

    async def _retrieve_advanced(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        assert self.contract_client is not None
        flow = RetrievalFlow(
            contract_client=self.contract_client,
            subquery_planner=self.subquery_planner,
            profile_context=self._profile_context,
            profile_resolution_context=self._profile_resolution_context,
        )
        items = await flow.execute(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            plan=plan,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
            validated_filters=self._validated_filters,
            validated_scope_payload=self._validated_scope_payload,
        )
        self.last_retrieval_diagnostics = flow.last_diagnostics
        return items

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
