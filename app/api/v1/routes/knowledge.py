from typing import Any, Dict, Optional

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.agent.adapters import LiteralEvidenceValidator
from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.http_adapters import GroundedAnswerAdapter, RagEngineRetrieverAdapter
from app.api.deps import UserContext, get_current_user
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.scope_metrics import scope_metrics_store
from app.security.tenant_authorizer import authorize_requested_tenant, fetch_tenant_names, resolve_allowed_tenants

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["knowledge"])


class OrchestratorQuestionRequest(BaseModel):
    query: str
    tenant_id: Optional[str] = None
    collection_id: Optional[str] = None


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


def _build_use_case() -> HandleQuestionUseCase:
    retriever = RagEngineRetrieverAdapter()
    answer_generator = GroundedAnswerAdapter(service=GroundedAnswerService())
    validator = LiteralEvidenceValidator()
    return HandleQuestionUseCase(
        retriever=retriever,
        answer_generator=answer_generator,
        validator=validator,
    )


def _s2s_headers(tenant_id: str) -> dict[str, str]:
    return {
        "X-Service-Secret": str(settings.RAG_SERVICE_SECRET or ""),
        "X-Tenant-ID": tenant_id,
    }


async def _fetch_collections_from_rag(tenant_id: str) -> list[CollectionItem]:
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(6.0, connect=2.0)) as client:
        response = await client.get(url, params=params, headers=_s2s_headers(tenant_id))
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


@router.post("/answer", response_model=Dict[str, Any])
async def answer_with_orchestrator(
    http_request: Request,
    request: OrchestratorQuestionRequest,
    current_user: UserContext = Depends(get_current_user),
    use_case: HandleQuestionUseCase = Depends(_build_use_case),
):
    try:
        authorized_tenant = await authorize_requested_tenant(http_request, current_user, request.tenant_id)
        scope_metrics_store.record_request(authorized_tenant)

        result = await use_case.execute(
            HandleQuestionCommand(
                query=request.query,
                tenant_id=authorized_tenant,
                user_id=current_user.user_id,
                collection_id=request.collection_id,
                scope_label=f"tenant={authorized_tenant}",
            )
        )

        if result.clarification:
            scope_metrics_store.record_clarification(authorized_tenant)

        blocked = (
            not result.validation.accepted
            and any("Scope mismatch" in issue for issue in result.validation.issues)
        )
        if blocked:
            scope_metrics_store.record_mismatch_detected(authorized_tenant)
            scope_metrics_store.record_mismatch_blocked(authorized_tenant)

        context_chunks = [item.content for item in result.answer.evidence]
        citations = [item.source for item in result.answer.evidence]
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
            "citations": citations,
            "context_chunks": context_chunks,
            "requested_scopes": list(result.plan.requested_standards),
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
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("orchestrator_answer_failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Orchestrator answer failed")


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
        items = await _fetch_collections_from_rag(authorized_tenant)
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "orchestrator_collection_proxy_failed",
            status=exc.response.status_code,
            tenant_id=authorized_tenant,
        )
        items = []
    except Exception as exc:
        logger.warning("orchestrator_collection_proxy_error", tenant_id=authorized_tenant, error=str(exc))
        items = []
    return CollectionListResponse(items=items)


@router.get("/scope-health", response_model=Dict[str, Any])
async def scope_health(tenant_id: Optional[str] = Query(default=None)):
    return scope_metrics_store.snapshot(tenant_id=tenant_id)
