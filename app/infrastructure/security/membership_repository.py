from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import httpx
import structlog

from app.infrastructure.config import settings

logger = structlog.get_logger(__name__)

_MEMBERSHIP_CACHE: dict[str, tuple[float, list[str]]] = {}
_MEMBERSHIP_CACHE_TTL = 300  # 5 minutes

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

async def _internal_fetch_membership_tenants(user_id: str) -> list[str]:
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

    if tenant_col != "institution_id":
        institution_rows = await _query_membership_tenants(
            user_id=user_id,
            table=table,
            user_col=user_col,
            tenant_col="institution_id",
        )
        if institution_rows:
            return institution_rows

    if table == "tenant_memberships":
        fallback_rows = await _query_membership_tenants(
            user_id=user_id,
            table="memberships",
            user_col=user_col,
            tenant_col=tenant_col,
        )
        if fallback_rows:
            return fallback_rows
        if tenant_col != "institution_id":
            fallback_institution_rows = await _query_membership_tenants(
                user_id=user_id,
                table="memberships",
                user_col=user_col,
                tenant_col="institution_id",
            )
            if fallback_institution_rows:
                return fallback_institution_rows
    return []

async def fetch_membership_tenants(user_id: str) -> list[str]:
    now = time.time()
    if user_id in _MEMBERSHIP_CACHE:
        timestamp, cached_tenants = _MEMBERSHIP_CACHE[user_id]
        if now - timestamp < _MEMBERSHIP_CACHE_TTL:
            return cached_tenants

    tenants = await _internal_fetch_membership_tenants(user_id)
    _MEMBERSHIP_CACHE[user_id] = (time.time(), tenants)
    return tenants

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
        tid = _normalize_tenant(item.get("id"))
        tname = str(item.get("name") or "").strip()
        if tid and tname:
            names[tid] = tname
    return names
