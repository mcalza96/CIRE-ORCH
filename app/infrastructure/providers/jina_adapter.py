from __future__ import annotations

from typing import Any

import httpx
import structlog

from app.agent.interfaces import (
    EmbeddingProvider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    RerankingProvider,
)
from app.core.config import settings


logger = structlog.get_logger(__name__)


class JinaAdapter(EmbeddingProvider, RerankingProvider):
    def __init__(self, *, api_key: str, http_client: httpx.AsyncClient | None = None):
        self._api_key = str(api_key or "").strip()
        if not self._api_key:
            raise ProviderAuthError("JINA_API_KEY is required")
        self._embed_model = str(settings.JINA_EMBED_MODEL or "jina-embeddings-v3")
        self._rerank_model = str(settings.JINA_RERANK_MODEL or "jina-reranker-v2-base-multilingual")
        self._embed_url = str(settings.JINA_EMBED_URL or "https://api.jina.ai/v1/embeddings")
        self._rerank_url = str(settings.JINA_RERANK_URL or "https://api.jina.ai/v1/rerank")
        self._http_client = http_client
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self._embed_model,
            "task": "retrieval.query",
            "input": texts,
            "embedding_type": "float",
        }
        data = await self._post_json(url=self._embed_url, payload=payload, operation="jina_embed")
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ProviderError("invalid_response:jina_embed")
        ordered = sorted(
            (row for row in rows if isinstance(row, dict)), key=lambda x: int(x.get("index", 0))
        )
        return [list(map(float, row.get("embedding") or [])) for row in ordered]

    async def rerank(self, query: str, documents: list[str], top_n: int) -> list[dict[str, Any]]:
        if not query.strip() or not documents:
            return []
        payload = {
            "model": self._rerank_model,
            "query": query,
            "documents": documents,
            "top_n": max(1, min(top_n, len(documents))),
        }
        data = await self._post_json(url=self._rerank_url, payload=payload, operation="jina_rerank")
        rows = data.get("results") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise ProviderError("invalid_response:jina_rerank")
        mapped: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            if isinstance(idx, int):
                mapped.append(
                    {"index": idx, "relevance_score": float(row.get("relevance_score") or 0.0)}
                )
        return mapped

    async def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        operation: str,
    ) -> dict[str, Any]:
        client = await self._client()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        try:
            response = await client.post(url, json=payload, headers=headers)
            if response.status_code in {401, 403}:
                raise ProviderAuthError(f"provider_auth_error:{operation}")
            if response.status_code == 429:
                raise ProviderRateLimitError(f"provider_rate_limited:{operation}")
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ProviderError(f"invalid_response:{operation}")
            return data
        except ProviderError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("provider_timeout", provider="jina", operation=operation)
            raise ProviderError(f"provider_timeout:{operation}") from exc
        except httpx.RequestError as exc:
            logger.warning("provider_request_failed", provider="jina", operation=operation)
            raise ProviderError(f"provider_request_error:{operation}") from exc
        except ValueError as exc:
            raise ProviderError(f"invalid_response:{operation}") from exc

    async def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    timeout=float(settings.RAG_HTTP_TIMEOUT_SECONDS),
                    connect=float(settings.RAG_HTTP_CONNECT_TIMEOUT_SECONDS),
                    read=float(settings.RAG_HTTP_READ_TIMEOUT_SECONDS),
                    write=float(settings.RAG_HTTP_WRITE_TIMEOUT_SECONDS),
                    pool=float(settings.RAG_HTTP_POOL_TIMEOUT_SECONDS),
                )
            )
        return self._http_client
