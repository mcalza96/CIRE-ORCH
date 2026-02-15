from __future__ import annotations

from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx
import structlog
from fastapi import HTTPException, Request, status

from app.api.deps import UserContext
from app.core.config import settings


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


async def _query_membership_tenants(user_id: str, table: str, user_col: str, tenant_col: str) -> list[str]:
    rest_url = settings.resolved_supabase_rest_url
    service_role = str(settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if not rest_url or not service_role:
        return []

    url = f"{rest_url.rstrip('/')}/{quote(table, safe='')}"
    params = {
        "select": tenant_col,
        user_col: f"eq.{user_id}",
    }
    headers = {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
    except Exception as exc:
        logger.warning("tenant_membership_lookup_failed", user_id=user_id, error=str(exc))
        return []

    data: Any = response.json()
    if not isinstance(data, list):
        return []

    out: list[str] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        value = row.get(tenant_col)
        if isinstance(value, str) and value.strip():
            out.append(value.strip())
    return _normalize_tenants(out)


async def _fetch_membership_tenants(user_id: str) -> list[str]:
    table = settings.SUPABASE_MEMBERSHIPS_TABLE
    user_col = settings.SUPABASE_MEMBERSHIP_USER_COLUMN
    tenant_col = settings.SUPABASE_MEMBERSHIP_TENANT_COLUMN

    rows = await _query_membership_tenants(
        user_id=user_id,
        table=table,
        user_col=user_col,
        tenant_col=tenant_col,
    )
    if rows:
        return rows

    # Backward-compatible fallback for schemas that model tenant as institution_id.
    if tenant_col != "institution_id":
        institution_rows = await _query_membership_tenants(
            user_id=user_id,
            table=table,
            user_col=user_col,
            tenant_col="institution_id",
        )
        if institution_rows:
            logger.info(
                "tenant_membership_lookup_column_fallback",
                configured_column=tenant_col,
                fallback_column="institution_id",
                count=len(institution_rows),
            )
            return institution_rows

    if table == "tenant_memberships":
        fallback_rows = await _query_membership_tenants(
            user_id=user_id,
            table="memberships",
            user_col=user_col,
            tenant_col=tenant_col,
        )
        if fallback_rows:
            logger.info(
                "tenant_membership_lookup_fallback",
                configured_table=table,
                fallback_table="memberships",
                count=len(fallback_rows),
            )
            return fallback_rows
        if tenant_col != "institution_id":
            fallback_institution_rows = await _query_membership_tenants(
                user_id=user_id,
                table="memberships",
                user_col=user_col,
                tenant_col="institution_id",
            )
            if fallback_institution_rows:
                logger.info(
                    "tenant_membership_lookup_fallback",
                    configured_table=table,
                    fallback_table="memberships",
                    fallback_column="institution_id",
                    count=len(fallback_institution_rows),
                )
                return fallback_institution_rows
    return []


async def _resolve_allowed_tenants(user: UserContext) -> list[str]:
    claim_tenants = _normalize_tenants(list(user.tenant_ids))
    if claim_tenants:
        return claim_tenants
    return await _fetch_membership_tenants(user.user_id)


async def resolve_allowed_tenants(user: UserContext) -> list[str]:
    return await _resolve_allowed_tenants(user)


async def fetch_tenant_names(tenant_ids: list[str]) -> dict[str, str]:
    scoped = _normalize_tenants(tenant_ids)
    if not scoped:
        return {}

    rest_url = settings.resolved_supabase_rest_url
    service_role = str(settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if not rest_url or not service_role:
        return {}

    url = f"{rest_url.rstrip('/')}/institutions"
    csv_ids = ",".join(scoped)
    params = {"select": "id,name", "id": f"in.({csv_ids})"}
    headers = {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
    }
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
    except Exception as exc:
        logger.warning("tenant_names_lookup_failed", count=len(scoped), error=str(exc))
        return {}

    payload: Any = response.json()
    if not isinstance(payload, list):
        return {}

    names: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        tid = _normalize_tenant(item.get("id")) if isinstance(item.get("id"), str) else None
        tname = str(item.get("name") or "").strip()
        if tid and tname:
            names[tid] = tname
    return names


async def authorize_requested_tenant(
    request: Request,
    current_user: UserContext,
    requested_tenant: str | None,
) -> str:
    requested = _normalize_tenant(requested_tenant)

    if not settings.ORCH_AUTH_REQUIRED:
        if not requested:
            raise _http_error(
                request,
                status.HTTP_400_BAD_REQUEST,
                "TENANT_REQUIRED",
                "tenant_id is required when ORCH_AUTH_REQUIRED is disabled",
            )
        logger.info(
            "tenant_allowed",
            decision="tenant_allowed",
            user_id=current_user.user_id,
            tenant_id=requested,
            source="auth_bypass",
        )
        return requested

    allowed_tenants = await _resolve_allowed_tenants(current_user)
    if not allowed_tenants:
        logger.warning(
            "tenant_denied",
            decision="tenant_denied",
            user_id=current_user.user_id,
            reason="no_membership",
        )
        raise _http_error(
            request,
            status.HTTP_403_FORBIDDEN,
            "TENANT_ACCESS_DENIED",
            "User has no tenant membership",
        )

    if requested:
        if requested not in allowed_tenants:
            logger.warning(
                "tenant_denied",
                decision="tenant_denied",
                user_id=current_user.user_id,
                tenant_id=requested,
                reason="not_allowed",
            )
            raise _http_error(
                request,
                status.HTTP_403_FORBIDDEN,
                "TENANT_ACCESS_DENIED",
                "User is not authorized for requested tenant",
            )
        logger.info(
            "tenant_allowed",
            decision="tenant_allowed",
            user_id=current_user.user_id,
            tenant_id=requested,
            source="request",
        )
        return requested

    if len(allowed_tenants) == 1:
        selected = allowed_tenants[0]
        logger.info(
            "tenant_allowed",
            decision="tenant_allowed",
            user_id=current_user.user_id,
            tenant_id=selected,
            source="single_membership",
        )
        return selected

    raise _http_error(
        request,
        status.HTTP_400_BAD_REQUEST,
        "TENANT_REQUIRED",
        "tenant_id is required for multi-tenant users",
    )
