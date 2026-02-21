from __future__ import annotations

import structlog
from fastapi import HTTPException, Request, status
from uuid import uuid4

from app.api.v1.deps import UserContext
from app.infrastructure.config import settings
from app.infrastructure.security.membership_repository import (
    fetch_membership_tenants,
)

logger = structlog.get_logger(__name__)

def _request_id(request: Request) -> str:
    return str(request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or uuid4())

def _http_error(request: Request, status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "request_id": _request_id(request),
        },
    )

def _normalize_tenant(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None

def _normalize_tenants(values: list[str]) -> list[str]:
    out: set[str] = set()
    for value in values:
        normalized = _normalize_tenant(value)
        if normalized:
            out.add(normalized)
    return sorted(out)

async def _resolve_allowed_tenants(user: UserContext) -> list[str]:
    claim_tenants = _normalize_tenants(list(user.tenant_ids))
    if claim_tenants:
        return claim_tenants
    return await fetch_membership_tenants(user.user_id)

async def resolve_allowed_tenants(user: UserContext) -> list[str]:
    """Resolves all tenants a user belongs to."""
    return await _resolve_allowed_tenants(user)

async def authorize_requested_tenant(
    request: Request,
    current_user: UserContext,
    requested_tenant: str | None,
) -> str:
    """
    FastAPI guard to authorize access to a specific tenant.
    Returns the authorized tenant_id or raises HTTPException.
    """
    requested = _normalize_tenant(requested_tenant)

    if not settings.ORCH_AUTH_REQUIRED:
        if not requested:
            raise _http_error(
                request,
                status.HTTP_400_BAD_REQUEST,
                "TENANT_REQUIRED",
                "tenant_id is required when ORCH_AUTH_REQUIRED is disabled",
            )
        return requested

    allowed_tenants = await _resolve_allowed_tenants(current_user)
    if not allowed_tenants:
        logger.warning("tenant_denied", user_id=current_user.user_id, reason="no_membership")
        raise _http_error(
            request,
            status.HTTP_403_FORBIDDEN,
            "TENANT_ACCESS_DENIED",
            "User has no tenant membership",
        )

    if requested:
        if requested not in allowed_tenants:
            logger.warning("tenant_denied", user_id=current_user.user_id, tenant_id=requested, reason="not_allowed")
            raise _http_error(
                request,
                status.HTTP_403_FORBIDDEN,
                "TENANT_ACCESS_DENIED",
                "User is not authorized for requested tenant",
            )
        return requested

    if len(allowed_tenants) == 1:
        return allowed_tenants[0]

    raise _http_error(
        request,
        status.HTTP_400_BAD_REQUEST,
        "TENANT_REQUIRED",
        "tenant_id is required for multi-tenant users",
    )
