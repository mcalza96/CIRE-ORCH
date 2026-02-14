from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalPlan
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.retrieval_metrics import retrieval_metrics_store


logger = structlog.get_logger(__name__)


@dataclass
class RagEngineRetrieverAdapter:
    base_url: str | None = None
    timeout_seconds: float = 8.0
    backend_selector: RagBackendSelector | None = None

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

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        retrieval_metrics_store.record_request("chunks")
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "chunk_k": int(plan.chunk_k),
            "fetch_k": int(plan.chunk_fetch_k),
        }
        context_headers = {"X-Tenant-ID": tenant_id}
        if user_id:
            context_headers["X-User-ID"] = user_id
        try:
            data = await self._post_json(
                "/api/v1/debug/retrieval/chunks",
                payload,
                endpoint="chunks",
                extra_headers=context_headers,
            )
        except (httpx.RequestError, httpx.HTTPStatusError):
            retrieval_metrics_store.record_failure("chunks")
            raise
        retrieval_metrics_store.record_success("chunks")
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        retrieval_metrics_store.record_request("summaries")
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "summary_k": int(plan.summary_k),
        }
        context_headers = {"X-Tenant-ID": tenant_id}
        if user_id:
            context_headers["X-User-ID"] = user_id
        try:
            data = await self._post_json(
                "/api/v1/debug/retrieval/summaries",
                payload,
                endpoint="summaries",
                extra_headers=context_headers,
            )
            retrieval_metrics_store.record_success("summaries")
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            retrieval_metrics_store.record_failure("summaries")
            retrieval_metrics_store.record_degraded_response("summaries")
            logger.warning(
                "rag_retrieval_endpoint_degraded",
                endpoint="summaries",
                reason="returning_empty_summaries",
                error=str(exc),
            )
            return []
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        endpoint: str,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        selector = self.backend_selector
        if selector is None:
            base_url = str(self.base_url or "http://localhost:8000").rstrip("/")
            return await self._post_once(base_url=base_url, path=path, payload=payload, extra_headers=extra_headers)

        primary_backend = await selector.current_backend()
        primary_base_url = await selector.resolve_base_url()

        try:
            return await self._post_once(
                base_url=primary_base_url,
                path=path,
                payload=payload,
                extra_headers=extra_headers,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as primary_exc:
            if selector.is_forced():
                raise

            retryable_status = (
                isinstance(primary_exc, httpx.HTTPStatusError)
                and primary_exc.response is not None
                and primary_exc.response.status_code >= 500
            )
            if isinstance(primary_exc, httpx.HTTPStatusError) and not retryable_status:
                raise

            alternate_backend = selector.alternate_backend(primary_backend)
            alternate_base_url = selector.base_url_for(alternate_backend)
            retrieval_metrics_store.record_fallback_retry(endpoint)
            logger.warning(
                "rag_backend_fallback_retry",
                from_backend=primary_backend,
                to_backend=alternate_backend,
                path=path,
                error=str(primary_exc),
            )
            response = await self._post_once(
                base_url=alternate_base_url,
                path=path,
                payload=payload,
                extra_headers=extra_headers,
            )
            selector.set_backend(alternate_backend)
            return response

    async def _post_once(
        self,
        *,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + path
        headers = {
            "X-Service-Secret": settings.RAG_SERVICE_SECRET or "",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return dict(response.json())

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
            raw_candidate = item.get("metadata")
            if isinstance(raw_candidate, dict):
                raw_metadata: dict[str, Any] = raw_candidate
            else:
                raw_metadata = {}
            out.append(
                EvidenceItem(
                    source=str(item.get("source") or "C1"),
                    content=content,
                    score=float(item.get("score") or 0.0),
                    metadata={"row": {"content": content, "metadata": raw_metadata, "similarity": item.get("score")}},
                )
            )
        return out


@dataclass
class GroundedAnswerAdapter:
    service: GroundedAnswerService

    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
    ) -> AnswerDraft:
        del scope_label
        context_chunks = [item.content for item in [*chunks, *summaries] if item.content]
        text = await self.service.generate_answer(query=query, context_chunks=context_chunks)
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])
