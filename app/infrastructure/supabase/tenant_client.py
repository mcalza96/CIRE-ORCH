from __future__ import annotations

import httpx
import structlog
from uuid import uuid4
from typing import Any

from app.infrastructure.config import settings
from app.infrastructure.observability.logging_utils import emit_event

logger = structlog.get_logger(__name__)

def _supabase_rest_headers() -> dict[str, str]:
    service_role = str(settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if not service_role:
        raise RuntimeError("supabase_service_role_missing")
    return {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

async def create_dev_tenant(user_id: str, name: str) -> tuple[str, str]:
    """
    Creates a new tenant (institution) in Supabase and associates it with the given user.
    Only for development/testing purposes.
    """
    rest_url = str(settings.resolved_supabase_rest_url or "").strip()
    if not rest_url:
        raise RuntimeError("supabase_rest_url_missing")

    tenant_id = str(uuid4())
    tenant_name = str(name or "").strip()
    if not tenant_name:
        raise RuntimeError("tenant_name_required")

    headers = _supabase_rest_headers()
    institutions_url = f"{rest_url.rstrip('/')}/institutions"
    memberships_url = f"{rest_url.rstrip('/')}/{settings.SUPABASE_MEMBERSHIPS_TABLE}"

    institution_payload = {
        "id": tenant_id,
        "name": tenant_name,
    }
    membership_payload = {
        str(settings.SUPABASE_MEMBERSHIP_USER_COLUMN): str(user_id),
        str(settings.SUPABASE_MEMBERSHIP_TENANT_COLUMN): str(tenant_id),
    }

    async with httpx.AsyncClient(timeout=6.0) as client:
        response = await client.post(institutions_url, json=institution_payload, headers=headers)
        if response.status_code >= 400:
            raise RuntimeError(f"institution_insert_failed:{response.text}")

        membership_error: str | None = None
        membership_response = await client.post(
            memberships_url,
            json=membership_payload,
            headers=headers,
        )
        if membership_response.status_code >= 400:
            # Fallback for old schema where institution_id was hardcoded
            fallback_payload = {
                str(settings.SUPABASE_MEMBERSHIP_USER_COLUMN): str(user_id),
                "institution_id": str(tenant_id),
            }
            fallback_response = await client.post(
                memberships_url,
                json=fallback_payload,
                headers=headers,
            )
            if fallback_response.status_code >= 400:
                membership_error = (
                    "membership_insert_failed:"
                    f"{membership_response.text} || fallback={fallback_response.text}"
                )
        
        if membership_error:
            if bool(settings.ORCH_AUTH_REQUIRED):
                raise RuntimeError(membership_error)
            
            emit_event(
                logger,
                "dev_tenant_membership_skipped",
                level="warning",
                reason="auth_disabled_or_non_uuid_user",
                error=membership_error,
                tenant_id=tenant_id,
                user_id=user_id,
            )

    return tenant_id, tenant_name
