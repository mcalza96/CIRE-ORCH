from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import httpx
import structlog

from app.agent.types.interfaces import EmbeddingProvider, RerankingProvider
from app.infrastructure.config import settings
from app.infrastructure.providers.cohere_adapter import CohereAdapter
from app.infrastructure.providers.jina_adapter import JinaAdapter


logger = structlog.get_logger(__name__)

BackendName = str


@dataclass(frozen=True)
class BackendEndpoints:
    local_url: str
    docker_url: str


class RagBackendSelector:
    def __init__(
        self,
        *,
        local_url: str,
        docker_url: str,
        health_path: str = "/health",
        probe_timeout_ms: int = 300,
        ttl_seconds: int = 20,
        force_backend: str | None = None,
    ) -> None:
        self._endpoints = BackendEndpoints(
            local_url=local_url.rstrip("/"),
            docker_url=docker_url.rstrip("/"),
        )
        self._health_path = health_path if health_path.startswith("/") else f"/{health_path}"
        self._probe_timeout_seconds = max(0.05, int(probe_timeout_ms) / 1000.0)
        self._ttl_seconds = max(1, int(ttl_seconds))
        self._force_backend = self._normalize_backend(force_backend)

        self._cached_backend: BackendName | None = None
        self._cache_expires_at: float = 0.0

    async def resolve_base_url(self) -> str:
        backend = await self._resolve_backend()
        return self._url_for(backend)

    async def current_backend(self) -> BackendName:
        return await self._resolve_backend()

    def is_forced(self) -> bool:
        return self._force_backend is not None

    def force_backend(self) -> BackendName | None:
        return self._force_backend

    def alternate_backend(self, backend: BackendName) -> BackendName:
        return "docker" if backend == "local" else "local"

    def set_backend(self, backend: BackendName) -> None:
        normalized = self._normalize_backend(backend)
        if normalized is None:
            return
        self._cached_backend = normalized
        self._cache_expires_at = monotonic() + self._ttl_seconds

    def base_url_for(self, backend: BackendName) -> str:
        return self._url_for(backend)

    async def _resolve_backend(self) -> BackendName:
        if self._force_backend:
            return self._force_backend

        now = monotonic()
        if self._cached_backend and now < self._cache_expires_at:
            return self._cached_backend

        backend = await self._detect_backend()
        if backend != self._cached_backend:
            logger.info("rag_backend_selected", backend=backend)

        self._cached_backend = backend
        self._cache_expires_at = now + self._ttl_seconds
        return backend

    async def _detect_backend(self) -> BackendName:
        probe_url = self._endpoints.local_url + self._health_path
        timeout = httpx.Timeout(self._probe_timeout_seconds, connect=self._probe_timeout_seconds)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(probe_url)
                if response.status_code == 200:
                    return "local"
                logger.warning(
                    "rag_backend_probe_failed",
                    backend="local",
                    url=probe_url,
                    status_code=response.status_code,
                )
        except httpx.RequestError as exc:
            logger.warning(
                "rag_backend_probe_failed",
                backend="local",
                url=probe_url,
                error=str(exc),
            )
        return "docker"

    def _url_for(self, backend: BackendName) -> str:
        if backend == "local":
            return self._endpoints.local_url
        return self._endpoints.docker_url

    @staticmethod
    def _normalize_backend(value: str | None) -> BackendName | None:
        if not value:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"local", "docker"}:
            return normalized
        return None


class RagProviderFactory:
    @staticmethod
    def create_embedding_provider(
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> EmbeddingProvider:
        provider = str(settings.RAG_PROVIDER or "jina").strip().lower()
        if provider == "cohere":
            return CohereAdapter(
                api_key=str(settings.COHERE_API_KEY or ""), http_client=http_client
            )
        return JinaAdapter(api_key=str(settings.JINA_API_KEY or ""), http_client=http_client)

    @staticmethod
    def create_reranking_provider(
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> RerankingProvider:
        provider = str(settings.RAG_PROVIDER or "jina").strip().lower()
        if provider == "cohere":
            return CohereAdapter(
                api_key=str(settings.COHERE_API_KEY or ""), http_client=http_client
            )
        return JinaAdapter(api_key=str(settings.JINA_API_KEY or ""), http_client=http_client)
