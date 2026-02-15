from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.rag_contract_schemas import MergeOptions, MultiQueryRetrievalRequest, SubQueryRequest
from app.core.retrieval_metrics import retrieval_metrics_store


logger = structlog.get_logger(__name__)


class RagContractNotSupportedError(RuntimeError):
    pass


@dataclass
class RagRetrievalContractClient:
    timeout_seconds: float = 12.0
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
            raise RuntimeError("RAG_SERVICE_SECRET must be configured for production security")

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        collection_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval_metrics_store.record_request("validate_scope")
        payload: dict[str, Any] = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "filters": filters,
        }
        try:
            data = await self._post_json(
                "/api/v1/retrieval/validate-scope",
                payload,
                endpoint="validate_scope",
                tenant_id=tenant_id,
                user_id=user_id,
            )
            retrieval_metrics_store.record_success("validate_scope")
            return data
        except Exception:
            retrieval_metrics_store.record_failure("validate_scope")
            raise

    async def hybrid(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        collection_id: str | None = None,
        k: int = 12,
        fetch_k: int = 60,
        filters: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval_metrics_store.record_request("hybrid")
        payload: dict[str, Any] = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "k": int(k),
            "fetch_k": int(fetch_k),
            "filters": filters,
            "rerank": rerank,
            "graph": graph,
        }
        try:
            data = await self._post_json(
                "/api/v1/retrieval/hybrid",
                payload,
                endpoint="hybrid",
                tenant_id=tenant_id,
                user_id=user_id,
            )
            retrieval_metrics_store.record_success("hybrid")
            return data
        except Exception:
            retrieval_metrics_store.record_failure("hybrid")
            raise

    async def multi_query(
        self,
        *,
        tenant_id: str,
        user_id: str | None,
        collection_id: str | None,
        queries: list[dict[str, Any]],
        merge: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval_metrics_store.record_request("multi_query")

        parsed_queries: list[SubQueryRequest] = []
        for raw in queries:
            if not isinstance(raw, dict):
                continue
            try:
                parsed_queries.append(SubQueryRequest.model_validate(raw))
            except Exception:
                logger.warning(
                    "multi_query_subquery_invalid_dropped",
                    tenant_id=tenant_id,
                    raw_id=str(raw.get("id") or ""),
                )

        # Only RRF is supported by the RAG contract at the moment.
        merge_obj = MergeOptions()
        if isinstance(merge, dict):
            try:
                merge_obj = MergeOptions.model_validate(merge)
            except Exception:
                logger.warning("multi_query_merge_invalid_defaulted", tenant_id=tenant_id)

        request_obj = MultiQueryRetrievalRequest(
            tenant_id=tenant_id,
            collection_id=collection_id,
            queries=parsed_queries,
            merge=merge_obj,
        )
        payload = request_obj.model_dump(by_alias=True, exclude_none=True)
        try:
            data = await self._post_json(
                "/api/v1/retrieval/multi-query",
                payload,
                endpoint="multi_query",
                tenant_id=tenant_id,
                user_id=user_id,
            )
            retrieval_metrics_store.record_success("multi_query")
            return data
        except Exception:
            retrieval_metrics_store.record_failure("multi_query")
            raise

    async def explain(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        collection_id: str | None = None,
        top_n: int = 10,
        k: int = 12,
        fetch_k: int = 60,
        filters: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        retrieval_metrics_store.record_request("explain")
        payload: dict[str, Any] = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "top_n": int(top_n),
            "k": int(k),
            "fetch_k": int(fetch_k),
            "filters": filters,
            "rerank": rerank,
            "graph": graph,
        }
        try:
            data = await self._post_json(
                "/api/v1/retrieval/explain",
                payload,
                endpoint="explain",
                tenant_id=tenant_id,
                user_id=user_id,
            )
            retrieval_metrics_store.record_success("explain")
            return data
        except Exception:
            retrieval_metrics_store.record_failure("explain")
            raise

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        endpoint: str,
        tenant_id: str,
        user_id: str | None,
    ) -> dict[str, Any]:
        selector = self.backend_selector
        assert selector is not None

        primary_backend = await selector.current_backend()
        primary_base_url = await selector.resolve_base_url()
        try:
            return await self._post_once(
                base_url=primary_base_url,
                path=path,
                payload=payload,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        except httpx.HTTPStatusError as exc:
            # If contract is not deployed yet, allow caller to fall back to legacy.
            if exc.response is not None and exc.response.status_code == 404:
                raise RagContractNotSupportedError(
                    f"RAG contract endpoint not found: {path}"
                ) from exc
            raise
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
                tenant_id=tenant_id,
                user_id=user_id,
            )
            selector.set_backend(alternate_backend)
            return response

    async def _post_once(
        self,
        *,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        tenant_id: str,
        user_id: str | None,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + path
        headers: dict[str, str] = {
            "X-Service-Secret": settings.RAG_SERVICE_SECRET or "",
            "X-Tenant-ID": tenant_id,
            "Content-Type": "application/json",
        }
        if user_id:
            headers["X-User-ID"] = user_id
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict):
                return data
            return {"items": data}
