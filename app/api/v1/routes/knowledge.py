import asyncio
from contextlib import asynccontextmanager
import json
import time
from typing import Any, Dict, Optional

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.adapters import LiteralEvidenceValidator
from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.components import build_citation_bundle
from app.agent.errors import ScopeValidationError
from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.http_adapters import RagEngineRetrieverAdapter
from app.agent.answer_adapter import GroundedAnswerAdapter
from app.api.deps import UserContext, get_current_user
from app.cartridges.deps import resolve_agent_profile
from app.cartridges.loader import get_cartridge_loader
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.rag_retrieval_contract_client import build_rag_http_client
from app.core.scope_metrics import scope_metrics_store
from app.security.tenant_authorizer import (
    authorize_requested_tenant,
    fetch_tenant_names,
    resolve_allowed_tenants,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["knowledge"])


def _classify_orchestrator_error(exc: Exception) -> str:
    text = str(exc or "").strip().lower()
    if "orch_answer_generation_failed" in text:
        return "ORCH_ANSWER_GENERATION_FAILED"
    if "rag retrieval failed" in text:
        return "ORCH_RETRIEVAL_FAILED"
    if isinstance(exc, TimeoutError):
        return "ORCH_TIMEOUT"
    if isinstance(exc, ValueError):
        return "ORCH_INVALID_INPUT"
    return "ORCH_UNHANDLED_ERROR"


class OrchestratorQuestionRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None


def _sse(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


class OrchestratorValidateScopeRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class OrchestratorExplainRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None
    top_n: int = 10
    k: int = 12
    fetch_k: int = 60
    filters: Optional[Dict[str, Any]] = None


class TenantItem(BaseModel):
    id: str
    name: str


class TenantListResponse(BaseModel):
    items: list[TenantItem]


class CollectionItem(BaseModel):
    id: str
    name: str
    collection_key: str | None = None


class CollectionListResponse(BaseModel):
    items: list[CollectionItem]


class AgentProfileItem(BaseModel):
    id: str
    declared_profile_id: str
    version: str
    status: str
    description: str
    owner: str


class AgentProfileListResponse(BaseModel):
    items: list[AgentProfileItem]


class TenantProfileUpdateRequest(BaseModel):
    tenant_id: str
    profile_id: Optional[str] = None
    clear: bool = False


def _build_use_case(http_request: Request) -> HandleQuestionUseCase:
    shared_client = getattr(http_request.app.state, "rag_http_client", None)
    retriever = RagEngineRetrieverAdapter(http_client=shared_client)
    answer_generator = GroundedAnswerAdapter(service=GroundedAnswerService())
    validator = LiteralEvidenceValidator()
    return HandleQuestionUseCase(
        retriever=retriever,
        answer_generator=answer_generator,
        validator=validator,
    )


def _s2s_headers(
    tenant_id: str,
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Service-Secret": str(settings.RAG_SERVICE_SECRET or ""),
        "X-Tenant-ID": tenant_id,
    }
    if request_id:
        headers["X-Request-ID"] = request_id
        headers["X-Trace-ID"] = request_id
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id
    return headers


async def _fetch_collections_from_rag(
    tenant_id: str,
    client: httpx.AsyncClient,
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> list[CollectionItem]:
    selector = RagBackendSelector(
        local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
        docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
        health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
        probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
        ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
        force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
    )
    base_url = await selector.resolve_base_url()

    url = f"{base_url.rstrip('/')}/api/v1/ingestion/collections"
    params = {"tenant_id": tenant_id}
    response = await client.get(
        url,
        params=params,
        headers=_s2s_headers(
            tenant_id,
            request_id=request_id,
            correlation_id=correlation_id,
        ),
    )
    response.raise_for_status()
    payload = response.json()
    raw_items = payload if isinstance(payload, list) else payload.get("items", [])
    out: list[CollectionItem] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        ckey = str(item.get("collection_key") or "").strip() or None
        cname = str(item.get("name") or item.get("collection_name") or ckey or cid).strip()
        out.append(CollectionItem(id=cid, name=cname or cid, collection_key=ckey))
    return out


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


@router.post("/answer", response_model=Dict[str, Any])
async def answer_with_orchestrator(
    http_request: Request,
    request: OrchestratorQuestionRequest,
    current_user: UserContext = Depends(get_current_user),
    use_case: HandleQuestionUseCase = Depends(_build_use_case),
):
    try:
        authorized_tenant = await authorize_requested_tenant(
            http_request, current_user, request.tenant_id
        )
        req_id = str(
            http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
        ).strip()
        corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()

        resolved_profile = await resolve_agent_profile(
            tenant_id=authorized_tenant, request=http_request
        )
        agent_profile = resolved_profile.profile
        scope_metrics_store.record_request(authorized_tenant)

        command = HandleQuestionCommand(
            query=request.query,
            tenant_id=authorized_tenant,
            user_id=current_user.user_id,
            collection_id=request.collection_id,
            scope_label=f"tenant={authorized_tenant}",
            agent_profile=agent_profile,
            profile_resolution=resolved_profile.resolution.model_dump(),
            request_id=req_id or None,
            correlation_id=corr_id or None,
        )

        result = await use_case.execute(command)

        if result.clarification:
            scope_metrics_store.record_clarification(authorized_tenant)

        blocked = not result.validation.accepted and any(
            "Scope mismatch" in issue for issue in result.validation.issues
        )
        if blocked:
            scope_metrics_store.record_mismatch_detected(authorized_tenant)
            scope_metrics_store.record_mismatch_blocked(authorized_tenant)

        context_chunks = [item.content for item in result.answer.evidence]
        citations, citations_detailed, citation_quality = build_citation_bundle(
            answer_text=result.answer.text,
            evidence=result.answer.evidence,
            profile=agent_profile,
        )
        validation_accepted = bool(result.validation.accepted)
        clarification_present = bool(result.clarification)
        logger.info(
            "orchestrator_answer_summary",
            user_id=current_user.user_id,
            tenant_id=authorized_tenant,
            collection_id=request.collection_id,
            mode=result.plan.mode,
            context_chunks_count=len(context_chunks),
            citations_count=len(citations),
            validation_accepted=validation_accepted,
            clarification_present=clarification_present,
            citation_structured_ratio=citation_quality.get("structured_ratio"),
            citation_missing_standard_count=citation_quality.get("missing_standard_count"),
            citation_missing_clause_count=citation_quality.get("missing_clause_count"),
            hypothesis_markers=citation_quality.get("hypothesis_markers"),
            agent_profile_id=agent_profile.profile_id,
            agent_profile_version=agent_profile.version,
            agent_profile_source=resolved_profile.resolution.source,
            agent_profile_resolution_reason=resolved_profile.resolution.decision_reason,
        )
        if len(context_chunks) == 0:
            reason = "scope_validation_blocked" if blocked else "retrieval_empty"
            logger.info(
                "orchestrator_answer_empty_context",
                user_id=current_user.user_id,
                tenant_id=authorized_tenant,
                collection_id=request.collection_id,
                reason=reason,
            )

        return {
            "answer": result.answer.text,
            "mode": result.plan.mode,
            "engine": str(result.engine or "universal_flow"),
            "agent_profile": {
                "profile_id": agent_profile.profile_id,
                "version": agent_profile.version,
                "status": agent_profile.status,
                "resolution": resolved_profile.resolution.model_dump(),
            },
            "citations": citations,
            "citations_detailed": citations_detailed,
            "citation_quality": citation_quality,
            "context_chunks": context_chunks,
            "requested_scopes": list(result.plan.requested_standards),
            "retrieval_plan": {
                "promoted": bool((result.retrieval.trace or {}).get("promoted", False)),
                "reason": str(
                    (result.retrieval.trace or {}).get("reason")
                    or (result.retrieval.trace or {}).get("fallback_reason")
                    or ""
                ),
                "initial_mode": str((result.retrieval.trace or {}).get("initial_mode") or ""),
                "final_mode": str((result.retrieval.trace or {}).get("final_mode") or ""),
                "missing_scopes": list((result.retrieval.trace or {}).get("missing_scopes") or []),
                "fallback_blocked_by_literal_lock": bool(
                    (result.retrieval.trace or {}).get("fallback_blocked_by_literal_lock", False)
                ),
                "subqueries": list((result.retrieval.trace or {}).get("subqueries") or []),
                "timings_ms": dict((result.retrieval.trace or {}).get("timings_ms") or {}),
                "kernel_flags": {
                    "semantic_planner": bool(settings.ORCH_SEMANTIC_PLANNER),
                    "multi_query_primary": bool(settings.ORCH_MULTI_QUERY_PRIMARY),
                    "multi_query_refine": bool(settings.ORCH_MULTI_QUERY_REFINE),
                    "multi_query_evaluator": bool(settings.ORCH_MULTI_QUERY_EVALUATOR),
                },
            },
            "retrieval": {
                "contract": result.retrieval.contract,
                "strategy": result.retrieval.strategy,
                "partial": bool(result.retrieval.partial),
                "trace": dict(result.retrieval.trace or {}),
            },
            "scope_validation": dict(result.retrieval.scope_validation or {}),
            "clarification": (
                {
                    "question": result.clarification.question,
                    "options": list(result.clarification.options),
                }
                if result.clarification
                else None
            ),
            "validation": {
                "accepted": result.validation.accepted,
                "issues": list(result.validation.issues),
            },
            "reasoning_trace": dict(result.reasoning_trace or {}),
        }
    except ScopeValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "SCOPE_VALIDATION_FAILED",
                "message": exc.message,
                "details": {
                    "violations": exc.violations,
                    "warnings": exc.warnings or [],
                    "normalized_scope": exc.normalized_scope or {},
                    "query_scope": exc.query_scope or {},
                },
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        req_id = str(
            http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
        ).strip()
        corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
        error_code = _classify_orchestrator_error(exc)
        logger.error(
            "orchestrator_answer_failed",
            error_code=error_code,
            error=str(exc),
            request_id=req_id or None,
            correlation_id=corr_id or None,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": error_code,
                "message": "Orchestrator answer failed",
                "request_id": req_id or None,
                "correlation_id": corr_id or None,
            },
        )


@router.post("/answer/stream")
async def answer_with_orchestrator_stream(
    http_request: Request,
    request: OrchestratorQuestionRequest,
    current_user: UserContext = Depends(get_current_user),
    use_case: HandleQuestionUseCase = Depends(_build_use_case),
):
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    req_id = str(
        http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
    ).strip()
    corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
    resolved_profile = await resolve_agent_profile(
        tenant_id=authorized_tenant, request=http_request
    )
    agent_profile = resolved_profile.profile

    command = HandleQuestionCommand(
        query=request.query,
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        collection_id=request.collection_id,
        scope_label=f"tenant={authorized_tenant}",
        agent_profile=agent_profile,
        profile_resolution=resolved_profile.resolution.model_dump(),
        request_id=req_id or None,
        correlation_id=corr_id or None,
    )

    async def _event_stream():
        started = time.perf_counter()
        yield _sse(
            "status",
            {
                "type": "accepted",
                "tenant_id": authorized_tenant,
                "request_id": req_id or None,
                "correlation_id": corr_id or None,
            },
        )
        task = asyncio.create_task(use_case.execute(command))
        pulse = 0
        while not task.done():
            pulse += 1
            elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
            yield _sse(
                "status",
                {
                    "type": "working",
                    "phase": "retrieve_and_synthesize",
                    "elapsed_ms": elapsed_ms,
                    "pulse": pulse,
                },
            )
            await asyncio.sleep(0.4)

        try:
            result = await task
            citations, citations_detailed, citation_quality = build_citation_bundle(
                answer_text=result.answer.text,
                evidence=result.answer.evidence,
                profile=agent_profile,
            )
            payload = {
                "type": "final_answer",
                "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "answer": result.answer.text,
                "mode": result.plan.mode,
                "engine": str(result.engine or "universal_flow"),
                "citations": citations,
                "citations_detailed": citations_detailed,
                "citation_quality": citation_quality,
                "retrieval": {
                    "contract": result.retrieval.contract,
                    "strategy": result.retrieval.strategy,
                    "partial": bool(result.retrieval.partial),
                    "trace": dict(result.retrieval.trace or {}),
                },
                "validation": {
                    "accepted": result.validation.accepted,
                    "issues": list(result.validation.issues),
                },
            }
            yield _sse("result", payload)
            yield _sse("done", {"ok": True})
        except Exception as exc:
            error_code = _classify_orchestrator_error(exc)
            logger.error(
                "orchestrator_answer_stream_failed",
                error_code=error_code,
                error=str(exc),
                request_id=req_id or None,
                correlation_id=corr_id or None,
                exc_info=True,
            )
            yield _sse(
                "error",
                {
                    "code": error_code,
                    "message": "Orchestrator stream failed",
                    "request_id": req_id or None,
                    "correlation_id": corr_id or None,
                },
            )

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


@router.get("/tenants", response_model=TenantListResponse)
async def list_authorized_tenants(
    current_user: UserContext = Depends(get_current_user),
) -> TenantListResponse:
    tenant_ids = await resolve_allowed_tenants(current_user)
    names = await fetch_tenant_names(tenant_ids)
    items = [TenantItem(id=tid, name=names.get(tid) or tid) for tid in tenant_ids]
    return TenantListResponse(items=items)


@router.get("/collections", response_model=CollectionListResponse)
async def list_authorized_collections(
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    current_user: UserContext = Depends(get_current_user),
) -> CollectionListResponse:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    try:
        req_id = str(
            http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
        ).strip()
        corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
        async with _rag_http_client_ctx(http_request) as client:
            items = await _fetch_collections_from_rag(
                authorized_tenant,
                client,
                request_id=req_id or None,
                correlation_id=corr_id or None,
            )
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        logger.warning(
            "orchestrator_collection_proxy_failed",
            status=status,
            tenant_id=authorized_tenant,
        )
        code = "RAG_UPSTREAM_AUTH_FAILED" if status in {401, 403} else "RAG_UPSTREAM_ERROR"
        raise HTTPException(
            status_code=502,
            detail={
                "code": code,
                "message": "Failed to load collections from RAG",
                "upstream_status": status,
            },
        ) from exc
    except httpx.RequestError as exc:
        logger.warning(
            "orchestrator_collection_proxy_unreachable",
            tenant_id=authorized_tenant,
            error=str(exc),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "RAG_UPSTREAM_UNREACHABLE",
                "message": "RAG endpoint unreachable when listing collections",
            },
        ) from exc
    except Exception as exc:
        logger.warning(
            "orchestrator_collection_proxy_error", tenant_id=authorized_tenant, error=str(exc)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ORCH_COLLECTION_PROXY_ERROR",
                "message": "Unexpected collection proxy error",
            },
        ) from exc
    return CollectionListResponse(items=items)


@router.get("/agent-profiles", response_model=AgentProfileListResponse)
async def list_agent_profiles(
    current_user: UserContext = Depends(get_current_user),
) -> AgentProfileListResponse:
    del current_user
    loader = get_cartridge_loader()
    rows = loader.list_available_profile_entries()
    items = [AgentProfileItem.model_validate(row) for row in rows]
    return AgentProfileListResponse(items=items)


@router.get("/tenant-profile", response_model=Dict[str, Any])
async def get_tenant_profile_override(
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    current_user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    loader = get_cartridge_loader()
    resolved = await loader.resolve_for_tenant_async(tenant_id=authorized_tenant)
    return {
        "tenant_id": authorized_tenant,
        "dev_assignments_enabled": loader.dev_profile_assignments_enabled(),
        "override_profile_id": loader.get_dev_profile_override(authorized_tenant),
        "agent_profile": {
            "profile_id": resolved.profile.profile_id,
            "version": resolved.profile.version,
            "status": resolved.profile.status,
        },
        "resolution": resolved.resolution.model_dump(),
    }


@router.put("/tenant-profile", response_model=Dict[str, Any])
async def put_tenant_profile_override(
    http_request: Request,
    request: TenantProfileUpdateRequest,
    current_user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    loader = get_cartridge_loader()
    if not loader.dev_profile_assignments_enabled():
        logger.warning(
            "profile_dev_override_denied",
            tenant_id=authorized_tenant,
            requested_profile_id=request.profile_id,
            reason="feature_disabled",
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "DEV_PROFILE_ASSIGNMENTS_DISABLED",
                "message": "Dev profile assignments are disabled",
            },
        )

    should_clear = bool(request.clear) or not str(request.profile_id or "").strip()
    if should_clear:
        cleared = loader.clear_dev_profile_override(tenant_id=authorized_tenant)
        logger.info(
            "profile_dev_override_cleared",
            tenant_id=authorized_tenant,
            cleared=cleared,
            requested_profile_id=request.profile_id,
        )
    else:
        profile_id = str(request.profile_id or "").strip()
        try:
            loader.set_dev_profile_override(tenant_id=authorized_tenant, profile_id=profile_id)
        except ValueError:
            logger.warning(
                "profile_dev_override_denied",
                tenant_id=authorized_tenant,
                requested_profile_id=profile_id,
                reason="invalid_profile_id",
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_AGENT_PROFILE",
                    "message": f"Unknown or invalid profile_id: {profile_id}",
                },
            )
        logger.info(
            "profile_dev_override_set",
            tenant_id=authorized_tenant,
            applied_profile_id=profile_id,
        )

    resolved = await loader.resolve_for_tenant_async(tenant_id=authorized_tenant)
    return {
        "tenant_id": authorized_tenant,
        "dev_assignments_enabled": loader.dev_profile_assignments_enabled(),
        "override_profile_id": loader.get_dev_profile_override(authorized_tenant),
        "agent_profile": {
            "profile_id": resolved.profile.profile_id,
            "version": resolved.profile.version,
            "status": resolved.profile.status,
        },
        "resolution": resolved.resolution.model_dump(),
    }


@router.post("/validate-scope", response_model=Dict[str, Any])
async def validate_scope_proxy(
    http_request: Request,
    request: OrchestratorValidateScopeRequest,
    current_user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    selector = RagBackendSelector(
        local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
        docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
        health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
        probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
        ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
        force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
    )
    base_url = await selector.resolve_base_url()
    url = f"{base_url.rstrip('/')}/api/v1/retrieval/validate-scope"
    payload: Dict[str, Any] = {
        "query": request.query,
        "tenant_id": authorized_tenant,
        "collection_id": request.collection_id,
        "filters": request.filters,
    }
    req_id = str(
        http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
    ).strip()
    corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
    headers = _s2s_headers(
        authorized_tenant,
        request_id=req_id or None,
        correlation_id=corr_id or None,
    )
    headers["X-User-ID"] = current_user.user_id
    async with _rag_http_client_ctx(http_request) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, dict) else {"items": data}


@router.post("/explain-retrieval", response_model=Dict[str, Any])
async def explain_retrieval_proxy(
    http_request: Request,
    request: OrchestratorExplainRequest,
    current_user: UserContext = Depends(get_current_user),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    selector = RagBackendSelector(
        local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
        docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
        health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
        probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
        ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
        force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
    )
    base_url = await selector.resolve_base_url()
    url = f"{base_url.rstrip('/')}/api/v1/retrieval/explain"
    payload: Dict[str, Any] = {
        "query": request.query,
        "tenant_id": authorized_tenant,
        "collection_id": request.collection_id,
        "top_n": int(request.top_n),
        "k": int(request.k),
        "fetch_k": int(request.fetch_k),
        "filters": request.filters,
    }
    req_id = str(
        http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
    ).strip()
    corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
    headers = _s2s_headers(
        authorized_tenant,
        request_id=req_id or None,
        correlation_id=corr_id or None,
    )
    headers["X-User-ID"] = current_user.user_id
    async with _rag_http_client_ctx(http_request) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data if isinstance(data, dict) else {"items": data}


@router.get("/scope-health", response_model=Dict[str, Any])
async def scope_health(tenant_id: Optional[str] = Query(default=None)):
    return scope_metrics_store.snapshot(tenant_id=tenant_id)
