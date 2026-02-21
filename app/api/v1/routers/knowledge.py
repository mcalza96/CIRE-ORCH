import asyncio
from typing import Any, Dict, Optional
import time

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.agent.formatters.adapters import LiteralEvidenceValidator
from app.agent.engine import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.errors import ScopeValidationError
from app.agent.components.grounded_answer_service import GroundedAnswerService
from app.infrastructure.clients.http_adapters import RagEngineRetrieverAdapter
from app.agent.formatters.answer_adapter import GroundedAnswerAdapter
from app.api.v1.deps import UserContext, get_current_user
from app.profiles.deps import resolve_agent_profile
from app.api.v1.schemas.knowledge_schemas import (
    AgentProfileItem,
    AgentProfileListResponse,
    CollectionListResponse,
    DevTenantCreateRequest,
    DevTenantCreateResponse,
    OrchestratorExplainRequest,
    OrchestratorQuestionRequest,
    OrchestratorValidateScopeRequest,
    TenantItem,
    TenantListResponse,
    TenantProfileUpdateRequest,
)
from app.profiles.loader import get_profile_loader
from app.infrastructure.config import settings
from app.infrastructure.observability.logging_utils import compact_error, emit_event
from app.infrastructure.metrics.scope import scope_metrics_store
from app.api.v1.auth_guards import (
    authorize_requested_tenant,
    resolve_allowed_tenants,
)
from app.infrastructure.security.membership_repository import fetch_tenant_names
from app.infrastructure.clients.rag_client import RagRetrievalContractClient
from app.infrastructure.supabase.tenant_client import create_dev_tenant as supabase_create_dev_tenant
from app.api.v1.routers.helpers.knowledge_helpers import (
    THINKING_PHASES,
    classify_orchestrator_error,
    format_sse_event,
    map_collection_items,
    map_orchestrator_result,
)

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["knowledge"])


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


def _get_rag_client(http_request: Request) -> RagRetrievalContractClient:
    shared_client = getattr(http_request.app.state, "rag_http_client", None)
    return RagRetrievalContractClient(http_client=shared_client)



@router.post("/answer", response_model=Dict[str, Any])
async def answer_with_orchestrator(
    http_request: Request,
    request: OrchestratorQuestionRequest,
    current_user: UserContext = Depends(get_current_user),
    use_case: HandleQuestionUseCase = Depends(_build_use_case),
):
    started = time.perf_counter()
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
            clarification_context=(
                dict(request.clarification_context)
                if isinstance(request.clarification_context, dict)
                else None
            ),
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

        response_data = map_orchestrator_result(
            result=result,
            agent_profile=agent_profile,
            profile_resolution=resolved_profile.resolution.model_dump(),
        )

        # Inject kernel flags (settings dependent)
        response_data["retrieval_plan"]["kernel_flags"] = {
            "semantic_planner": bool(settings.ORCH_SEMANTIC_PLANNER),
            "multi_query_primary": bool(settings.ORCH_MULTI_QUERY_PRIMARY),
            "multi_query_refine": bool(settings.ORCH_MULTI_QUERY_REFINE),
            "multi_query_evaluator": bool(settings.ORCH_MULTI_QUERY_EVALUATOR),
        }

        # Logging and Summary metrics
        interaction_metrics = response_data.get("interaction") or {}
        citation_quality = response_data.get("citation_quality") or {}
        emit_event(
            logger,
            "orchestrator_answer_summary",
            user_id=current_user.user_id,
            tenant_id=authorized_tenant,
            collection_id=request.collection_id,
            mode=result.plan.mode,
            context_chunks_count=len(response_data.get("context_chunks", [])),
            citations_count=len(response_data.get("citations", [])),
            validation_accepted=bool(result.validation.accepted),
            clarification_present=bool(result.clarification),
            citation_structured_ratio=citation_quality.get("structured_ratio"),
            citation_missing_standard_count=citation_quality.get("missing_standard_count"),
            citation_missing_clause_count=citation_quality.get("missing_clause_count"),
            hypothesis_markers=citation_quality.get("hypothesis_markers"),
            clarification_model_used=interaction_metrics.get("clarification_model_used"),
            clarification_confidence=interaction_metrics.get("clarification_confidence"),
            slots_filled=interaction_metrics.get("slots_filled"),
            loop_prevented=interaction_metrics.get("loop_prevented"),
            agent_profile_id=agent_profile.profile_id,
            agent_profile_version=agent_profile.version,
            agent_profile_source=resolved_profile.resolution.source,
            agent_profile_resolution_reason=resolved_profile.resolution.decision_reason,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )

        return response_data
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
        error_code = classify_orchestrator_error(exc)
        emit_event(
            logger,
            "orchestrator_answer_failed",
            level="error",
            error_code=error_code,
            error=compact_error(exc),
            request_id=req_id or None,
            correlation_id=corr_id or None,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        if bool(settings.ORCH_LOG_EXC_INFO):
            logger.error("orchestrator_answer_failed_exc", exc_info=True)
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
    started = time.perf_counter()
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
        clarification_context=(
            dict(request.clarification_context)
            if isinstance(request.clarification_context, dict)
            else None
        ),
    )

    async def _event_stream():
        streaming_started = time.perf_counter()
        yield format_sse_event(
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
        emitted_phase_index = -1
        while not task.done():
            pulse += 1
            elapsed_ms = round((time.perf_counter() - streaming_started) * 1000.0, 2)
            elapsed_seconds = elapsed_ms / 1000.0
            
            # Thinking phases
            while emitted_phase_index + 1 < len(THINKING_PHASES) and elapsed_seconds >= float(
                THINKING_PHASES[emitted_phase_index + 1].get("at_seconds") or 0.0
            ):
                emitted_phase_index += 1
                phase_item = THINKING_PHASES[emitted_phase_index]
                yield format_sse_event(
                    "status",
                    {
                        "type": "thinking",
                        "phase": phase_item.get("phase"),
                        "label": phase_item.get("label"),
                        "step": emitted_phase_index + 1,
                        "total_steps": len(THINKING_PHASES),
                        "elapsed_ms": elapsed_ms,
                    },
                )

            yield format_sse_event(
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
            response_data = map_orchestrator_result(
                result=result,
                agent_profile=agent_profile,
                profile_resolution=resolved_profile.resolution.model_dump(),
            )
            
            # Injection for stream specific payload
            response_data["type"] = "clarification_required" if result.clarification else "final_answer"
            response_data["elapsed_ms"] = round((time.perf_counter() - streaming_started) * 1000.0, 2)
            response_data["context_chunks_count"] = len(response_data.get("context_chunks", []))

            # Logging
            interaction_metrics = response_data.get("interaction") or {}
            citation_quality = response_data.get("citation_quality") or {}
            emit_event(
                logger,
                "orchestrator_answer_stream_summary",
                user_id=current_user.user_id,
                tenant_id=authorized_tenant,
                collection_id=request.collection_id,
                mode=result.plan.mode,
                context_chunks_count=response_data["context_chunks_count"],
                citations_count=len(response_data.get("citations", [])),
                validation_accepted=bool(result.validation.accepted),
                clarification_present=bool(result.clarification),
                citation_structured_ratio=citation_quality.get("structured_ratio"),
                citation_missing_standard_count=citation_quality.get("missing_standard_count"),
                citation_missing_clause_count=citation_quality.get("missing_clause_count"),
                clarification_model_used=interaction_metrics.get("clarification_model_used"),
                clarification_confidence=interaction_metrics.get("clarification_confidence"),
                slots_filled=interaction_metrics.get("slots_filled"),
                loop_prevented=interaction_metrics.get("loop_prevented"),
                duration_ms=response_data["elapsed_ms"],
            )

            yield format_sse_event("result", response_data)
            yield format_sse_event("done", {"ok": True})
        except Exception as exc:
            error_code = classify_orchestrator_error(exc)
            emit_event(
                logger,
                "orchestrator_answer_stream_failed",
                level="error",
                error_code=error_code,
                error=compact_error(exc),
                request_id=req_id or None,
                correlation_id=corr_id or None,
                duration_ms=round((time.perf_counter() - streaming_started) * 1000.0, 2),
            )
            if bool(settings.ORCH_LOG_EXC_INFO):
                logger.error("orchestrator_answer_stream_failed_exc", exc_info=True)
            yield format_sse_event(
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


@router.post("/dev/tenants", response_model=DevTenantCreateResponse)
async def create_dev_tenant(
    request: Request,
    body: DevTenantCreateRequest,
    current_user: UserContext = Depends(get_current_user),
) -> DevTenantCreateResponse:
    if not bool(settings.ORCH_DEV_TENANT_CREATE_ENABLED):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "DEV_TENANT_CREATE_DISABLED",
                "message": "Dev tenant creation is disabled",
            },
        )

    tenant_name = str(body.name or "").strip()
    if not tenant_name:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "TENANT_NAME_REQUIRED",
                "message": "Tenant name is required",
            },
        )

    try:
        tenant_id, created_name = await supabase_create_dev_tenant(
            user_id=current_user.user_id,
            name=tenant_name,
        )
    except Exception as exc:
        emit_event(
            logger,
            "dev_tenant_create_failed",
            level="warning",
            user_id=current_user.user_id,
            error=compact_error(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "DEV_TENANT_CREATE_FAILED",
                "message": "Could not create tenant in dev mode",
            },
        ) from exc

    emit_event(
        logger,
        "dev_tenant_created",
        user_id=current_user.user_id,
        tenant_id=tenant_id,
        tenant_name=created_name,
        request_id=str(request.headers.get("X-Request-ID") or "").strip() or None,
    )
    return DevTenantCreateResponse(tenant_id=tenant_id, name=created_name)


@router.get("/collections", response_model=CollectionListResponse)
async def list_authorized_collections(
    http_request: Request,
    tenant_id: str = Query(..., min_length=1),
    current_user: UserContext = Depends(get_current_user),
    rag_client: RagRetrievalContractClient = Depends(_get_rag_client),
) -> CollectionListResponse:
    started = time.perf_counter()
    authorized_tenant = await authorize_requested_tenant(http_request, current_user, tenant_id)
    try:
        req_id = str(
            http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
        ).strip()
        corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()
        
        raw_items = await rag_client.list_collections(
            tenant_id=authorized_tenant,
            user_id=current_user.user_id,
            request_id=req_id or None,
            correlation_id=corr_id or None,
        )
        items = map_collection_items(raw_items)
        
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 502
        emit_event(
            logger,
            "orchestrator_collection_proxy_failed",
            level="warning",
            status=status,
            tenant_id=authorized_tenant,
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
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
        emit_event(
            logger,
            "orchestrator_collection_proxy_unreachable",
            level="warning",
            tenant_id=authorized_tenant,
            error=compact_error(exc),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "code": "RAG_UPSTREAM_UNREACHABLE",
                "message": "RAG endpoint unreachable when listing collections",
            },
        ) from exc
    except Exception as exc:
        emit_event(
            logger,
            "orchestrator_collection_proxy_error",
            level="warning",
            tenant_id=authorized_tenant,
            error=compact_error(exc),
            duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "ORCH_COLLECTION_PROXY_ERROR",
                "message": "Unexpected collection proxy error",
            },
        ) from exc
    emit_event(
        logger,
        "orchestrator_collection_proxy_ok",
        tenant_id=authorized_tenant,
        collections_count=len(items),
        duration_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )
    return CollectionListResponse(items=items)



@router.get("/agent-profiles", response_model=AgentProfileListResponse)
async def list_agent_profiles(
    current_user: UserContext = Depends(get_current_user),
) -> AgentProfileListResponse:
    del current_user
    loader = get_profile_loader()
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
    loader = get_profile_loader()
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
    loader = get_profile_loader()
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
    rag_client: RagRetrievalContractClient = Depends(_get_rag_client),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    req_id = str(
        http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
    ).strip()
    corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()

    data = await rag_client.validate_scope(
        query=request.query,
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        request_id=req_id or None,
        correlation_id=corr_id or None,
        collection_id=request.collection_id,
        filters=request.filters,
    )
    return data if isinstance(data, dict) else {"items": data}


@router.post("/explain-retrieval", response_model=Dict[str, Any])
async def explain_retrieval_proxy(
    http_request: Request,
    request: OrchestratorExplainRequest,
    current_user: UserContext = Depends(get_current_user),
    rag_client: RagRetrievalContractClient = Depends(_get_rag_client),
) -> Dict[str, Any]:
    authorized_tenant = await authorize_requested_tenant(
        http_request, current_user, request.tenant_id
    )
    req_id = str(
        http_request.headers.get("X-Request-ID") or http_request.headers.get("X-Trace-ID") or ""
    ).strip()
    corr_id = str(http_request.headers.get("X-Correlation-ID") or "").strip()

    data = await rag_client.explain(
        query=request.query,
        tenant_id=authorized_tenant,
        user_id=current_user.user_id,
        request_id=req_id or None,
        correlation_id=corr_id or None,
        collection_id=request.collection_id,
        top_n=int(request.top_n),
        k=int(request.k),
        fetch_k=int(request.fetch_k),
        filters=request.filters,
    )
    return data if isinstance(data, dict) else {"items": data}


@router.get("/scope-health", response_model=Dict[str, Any])
async def scope_health(tenant_id: Optional[str] = Query(default=None)):
    return scope_metrics_store.snapshot(tenant_id=tenant_id)

