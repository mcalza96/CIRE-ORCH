from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog
from app.clients.backend_selector import RagBackendSelector, RagProviderFactory
from app.core.config import settings
from app.core.rag_retrieval_contract_client import (
    RagRetrievalContractClient,
)

# New Strategy Imports
from app.agent.retrieval_flow import RetrievalFlow
from app.agent.interfaces import EmbeddingProvider, RerankingProvider, SubqueryPlanner
from app.agent.components.query_decomposer import HybridSubqueryPlanner
from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile


logger = structlog.get_logger(__name__)


@dataclass
class RagEngineRetrieverAdapter:
    base_url: str | None = None
    timeout_seconds: float = 45.0
    backend_selector: RagBackendSelector | None = None
    contract_client: RagRetrievalContractClient | None = None
    http_client: httpx.AsyncClient | None = None
    subquery_planner: SubqueryPlanner | None = None
    embedding_provider: EmbeddingProvider | None = None
    reranking_provider: RerankingProvider | None = None

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

        if self.embedding_provider is None:
            try:
                self.embedding_provider = RagProviderFactory.create_embedding_provider(
                    http_client=self.http_client
                )
            except Exception as exc:
                logger.warning("embedding_provider_not_initialized", error=str(exc)[:160])

        if self.reranking_provider is None:
            try:
                self.reranking_provider = RagProviderFactory.create_reranking_provider(
                    http_client=self.http_client
                )
            except Exception as exc:
                logger.warning("reranking_provider_not_initialized", error=str(exc)[:160])

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
            embedding_provider=self.embedding_provider,
            reranking_provider=self.reranking_provider,
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
