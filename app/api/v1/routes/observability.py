from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.api.deps import UserContext, get_current_user
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.rag_retrieval_contract_client import build_rag_http_client
from app.security.tenant_authorizer import authorize_requested_tenant


logger = structlog.get_logger(__name__)
router = APIRouter(tags=["observability"])


def _selector() -> RagBackendSelector:
    return RagBackendSelector(
        local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
        docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
        health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
        probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
        ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
        force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
    )


def _s2s_headers(*, tenant_id: str, user_id: str) -> dict[str, str]:
    return {
        "X-Service-Secret": str(settings.RAG_SERVICE_SECRET or ""),
        "X-Tenant-ID": tenant_id,
        "X-User-ID": user_id,
    }


@asynccontextmanager
async def _rag_http_client_ctx(http_request: Request):
    shared_client = getattr(http_request.app.state, "rag_http_client", None)
    if isinstance(shared_client, httpx.AsyncClient):
        yield shared_client
        return

    client = build_rag_http_client()
    try:
        yield client
    finally:
        await client.aclose()


def _map_upstream_http_error(exc: httpx.HTTPStatusError, *, operation: str) -> HTTPException:
    upstream_status = exc.response.status_code if exc.response is not None else 502
    if upstream_status in {401, 403}:
        code = "RAG_UPSTREAM_AUTH_FAILED"
        message = f"RAG upstream auth failed during {operation}"
        raise HTTPException(
            status_code=502,
            detail={"code": code, "message": message, "upstream_status": upstream_status},
        ) from exc
    if upstream_status >= 500:
        code = "RAG_UPSTREAM_ERROR"
        message = f"RAG upstream error during {operation}"
        raise HTTPException(
            status_code=502,
            detail={"code": code, "message": message, "upstream_status": upstream_status},
        ) from exc
    raise HTTPException(
        status_code=upstream_status,
        detail={
            "code": "RAG_UPSTREAM_CLIENT_ERROR",
            "message": f"RAG rejected {operation} request",
        },
    ) from exc


async def _proxy_json_get(
    *,
    http_request: Request,
    path: str,
    params: dict[str, Any],
    tenant_id: str,
    user_id: str,
    operation: str,
) -> dict[str, Any]:
    base_url = await _selector().resolve_base_url()
    url = f"{base_url.rstrip('/')}{path}"
    try:
        async with _rag_http_client_ctx(http_request) as client:
            response = await client.get(
                url,
                params=params,
                headers=_s2s_headers(tenant_id=tenant_id, user_id=user_id),
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"items": payload}
    except httpx.HTTPStatusError as exc:
        raise _map_upstream_http_error(exc, operation=operation)
    except httpx.RequestError as exc:
        logger.warning(
            "observability_proxy_unreachable",
            operation=operation,
            tenant_id=tenant_id,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "RAG_UPSTREAM_UNREACHABLE",
                "message": f"RAG endpoint unreachable for {operation}",
            },
        ) from exc


@router.get("/batches/{batch_id}/progress")
async def get_batch_progress_proxy(
    batch_id: str,
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    current_user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    return await _proxy_json_get(
        http_request=http_request,
        path=f"/api/v1/ingestion/batches/{batch_id}/progress",
        params={"tenant_id": authorized_tenant},
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        operation="batch_progress",
    )


@router.get("/batches/{batch_id}/events")
async def get_batch_events_proxy(
    batch_id: str,
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    cursor: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    params: dict[str, Any] = {"tenant_id": authorized_tenant, "limit": limit}
    if cursor:
        params["cursor"] = cursor
    return await _proxy_json_get(
        http_request=http_request,
        path=f"/api/v1/ingestion/batches/{batch_id}/events",
        params=params,
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        operation="batch_events",
    )


@router.get("/batches/active")
async def list_active_batches_proxy(
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    limit: int = Query(default=10, ge=1, le=100),
    current_user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    return await _proxy_json_get(
        http_request=http_request,
        path="/api/v1/ingestion/batches/active",
        params={"tenant_id": authorized_tenant, "limit": limit},
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        operation="active_batches",
    )


async def _proxy_sse(
    *,
    batch_id: str,
    tenant_id: str,
    user_id: str,
    cursor: str | None,
    interval_ms: int,
) -> AsyncIterator[bytes]:
    selector = _selector()
    base_url = await selector.resolve_base_url()
    url = f"{base_url.rstrip('/')}/api/v1/ingestion/batches/{batch_id}/stream"
    params: dict[str, Any] = {"tenant_id": tenant_id, "interval_ms": interval_ms}
    if cursor:
        params["cursor"] = cursor

    timeout = httpx.Timeout(connect=5.0, read=None, write=10.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "GET",
            url,
            params=params,
            headers=_s2s_headers(tenant_id=tenant_id, user_id=user_id),
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                detail = f'event: error\ndata: {{"type":"error","status":{response.status_code},"message":"upstream stream failed"}}\n\n'
                yield detail.encode("utf-8")
                return
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


@router.get("/batches/{batch_id}/stream")
async def stream_batch_proxy(
    batch_id: str,
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    cursor: str | None = Query(default=None),
    interval_ms: int = Query(default=1500, ge=500, le=15000),
    current_user: UserContext = Depends(get_current_user),
) -> StreamingResponse:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)

    # Handshake to fail fast with proper HTTP status before opening stream.
    await _proxy_json_get(
        http_request=http_request,
        path=f"/api/v1/ingestion/batches/{batch_id}/progress",
        params={"tenant_id": authorized_tenant},
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        operation="batch_stream_handshake",
    )

    return StreamingResponse(
        _proxy_sse(
            batch_id=batch_id,
            tenant_id=authorized_tenant,
            user_id=current_user.user_id,
            cursor=cursor,
            interval_ms=interval_ms,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
