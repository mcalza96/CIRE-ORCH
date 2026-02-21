from __future__ import annotations

from contextlib import asynccontextmanager

from dataclasses import dataclass
import time
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
import structlog

from app.clients.backend_selector import RagBackendSelector
from app.infrastructure.config import settings
from app.infrastructure.metrics.retrieval import retrieval_metrics_store


logger = structlog.get_logger(__name__)


def _rag_http_timeout(timeout_seconds: float | None = None) -> httpx.Timeout:
    read_timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None and float(timeout_seconds) > 0
        else float(settings.RAG_HTTP_READ_TIMEOUT_SECONDS)
    )
    return httpx.Timeout(
        timeout=float(settings.RAG_HTTP_TIMEOUT_SECONDS),
        connect=float(settings.RAG_HTTP_CONNECT_TIMEOUT_SECONDS),
        read=read_timeout,
        write=float(settings.RAG_HTTP_WRITE_TIMEOUT_SECONDS),
        pool=float(settings.RAG_HTTP_POOL_TIMEOUT_SECONDS),
    )


def _rag_http_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=int(settings.RAG_HTTP_MAX_CONNECTIONS),
        max_keepalive_connections=int(settings.RAG_HTTP_MAX_KEEPALIVE_CONNECTIONS),
        keepalive_expiry=float(settings.RAG_HTTP_KEEPALIVE_EXPIRY_SECONDS),
    )


def build_rag_http_client(timeout_seconds: float | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_rag_http_timeout(timeout_seconds), limits=_rag_http_limits())


@dataclass
class RagRetrievalContractClient:
    timeout_seconds: float = 20.0
    backend_selector: RagBackendSelector | None = None
    http_client: httpx.AsyncClient | None = None
    _owns_http_client: bool = False

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

        if self.http_client is None:
            self.http_client = build_rag_http_client(timeout_seconds=self.timeout_seconds)
            self._owns_http_client = True

    @asynccontextmanager
    async def _record_metrics(self, endpoint: str) -> AsyncIterator[None]:
        retrieval_metrics_store.record_request(endpoint)
        try:
            yield
            retrieval_metrics_store.record_success(endpoint)
        except Exception:
            retrieval_metrics_store.record_failure(endpoint)
            raise

    async def aclose(self) -> None:
        if self._owns_http_client and self.http_client is not None:
            await self.http_client.aclose()
            self._owns_http_client = False

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        collection_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "filters": filters,
        }
        async with self._record_metrics("validate_scope"):
            return await self._dispatch(
                "/api/v1/retrieval/validate-scope",
                payload,
                endpoint="validate_scope",
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )

    async def comprehensive(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        collection_id: str | None = None,
        context_volume: str | None = None,
        k: int = 12,
        fetch_k: int = 60,
        filters: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
        coverage_requirements: dict[str, Any] | None = None,
        retrieval_policy: dict[str, Any] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "context_volume": context_volume,
            "k": int(k),
            "fetch_k": int(fetch_k),
            "filters": filters,
            "rerank": rerank,
            "graph": graph,
            "coverage_requirements": coverage_requirements,
            "retrieval_policy": retrieval_policy,
            "retrieval_plan": retrieval_plan,
        }
        async with self._record_metrics("comprehensive"):
            return await self._dispatch(
                "/api/v1/retrieval/comprehensive",
                payload,
                endpoint="comprehensive",
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )

    async def explain(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        collection_id: str | None = None,
        top_n: int = 10,
        k: int = 12,
        fetch_k: int = 60,
        filters: dict[str, Any] | None = None,
        rerank: dict[str, Any] | None = None,
        graph: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
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
        async with self._record_metrics("explain"):
            return await self._dispatch(
                "/api/v1/retrieval/explain",
                payload,
                endpoint="explain",
                tenant_id=tenant_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
            )

    async def _dispatch(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        endpoint: str,
        tenant_id: str,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
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
                request_id=request_id,
                correlation_id=correlation_id,
            )
        except httpx.HTTPStatusError as exc:
            # Non-retryable status codes (e.g. 400, 401, 403, 404) should raise immediately
            if exc.response is not None and exc.response.status_code < 500:
                raise
            # If fallback is forced, don't retry
            if selector.is_forced():
                raise
            return await self._handle_fallback(
                primary_backend, primary_base_url, path, payload, endpoint, 
                tenant_id, user_id, request_id, correlation_id, exc
            )
        except httpx.RequestError as exc:
            if selector.is_forced():
                raise
            return await self._handle_fallback(
                primary_backend, primary_base_url, path, payload, endpoint, 
                tenant_id, user_id, request_id, correlation_id, exc
            )

    async def _handle_fallback(
        self,
        primary_backend: str,
        primary_base_url: str,
        path: str,
        payload: dict[str, Any],
        endpoint: str,
        tenant_id: str,
        user_id: str | None,
        request_id: str | None,
        correlation_id: str | None,
        original_exc: Exception,
    ) -> dict[str, Any]:
        selector = self.backend_selector
        assert selector is not None
        
        alternate_backend = selector.alternate_backend(primary_backend)
        alternate_base_url = selector.base_url_for(alternate_backend)
        retrieval_metrics_store.record_fallback_retry(endpoint)
        
        logger.warning(
            "rag_backend_fallback_retry",
            from_backend=primary_backend,
            to_backend=alternate_backend,
            path=path,
            error=str(original_exc),
        )
        response = await self._post_once(
            base_url=alternate_base_url,
            path=path,
            payload=payload,
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
        )
        selector.set_backend(alternate_backend)
        return response


    def _build_headers(
        self, 
        tenant_id: str, 
        user_id: str | None, 
        request_id: str | None, 
        trace_id: str, 
        corr_id: str
    ) -> dict[str, str]:
        headers: dict[str, str] = {
            "X-Service-Secret": settings.RAG_SERVICE_SECRET or "",
            "X-Tenant-ID": tenant_id,
            "Content-Type": "application/json",
            "X-Trace-ID": trace_id,
            "X-Correlation-ID": corr_id,
        }
        if request_id:
            headers["X-Request-ID"] = str(request_id)
        if user_id:
            headers["X-User-ID"] = user_id
        return headers

    def _log_timeout(self, path: str, base_url: str, started_at: float, client: httpx.AsyncClient) -> None:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        timeout_cfg = client.timeout
        logger.warning(
            "rag_contract_request_timeout",
            endpoint=path,
            base_url=base_url,
            elapsed_ms=elapsed_ms,
            timeout_connect_s=float(timeout_cfg.connect or 0.0),
            timeout_read_s=float(timeout_cfg.read or 0.0),
            timeout_write_s=float(timeout_cfg.write or 0.0),
            timeout_pool_s=float(timeout_cfg.pool or 0.0),
        )

    async def _post_once(
        self,
        *,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        tenant_id: str,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + path
        trace_id = str(request_id or correlation_id or uuid4())
        corr_id = str(correlation_id or request_id or trace_id)
        
        headers = self._build_headers(tenant_id, user_id, request_id, trace_id, corr_id)
        
        client = self.http_client
        if client is None:
            client = build_rag_http_client(timeout_seconds=self.timeout_seconds)
            self.http_client = client
            self._owns_http_client = True

        started_at = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {"items": data}
        except httpx.TimeoutException:
            self._log_timeout(path, base_url, started_at, client)
            raise
