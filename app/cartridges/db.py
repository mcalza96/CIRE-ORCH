from __future__ import annotations

import json
from urllib.parse import quote

import httpx
import structlog

from app.cartridges.models import AgentProfile
from app.infrastructure.config import settings

logger = structlog.get_logger(__name__)


async def fetch_db_profile_async(tenant_id: str) -> tuple[AgentProfile | None, str]:
    """
    Fetches the tenant profile from the configured Supabase connection.
    Returns a tuple of (AgentProfile or None, resolution_reason).
    """
    if not bool(settings.ORCH_CARTRIDGE_DB_ENABLED):
        return None, "db_override_disabled"

    tenant = str(tenant_id or "").strip()
    if not tenant:
        return None, "empty_tenant_id"

    rest_url = settings.resolved_supabase_rest_url
    service_role = str(settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
    if not rest_url or not service_role:
        return None, "db_credentials_missing"

    table = str(settings.ORCH_CARTRIDGE_DB_TABLE or "tenant_configs").strip()
    tenant_col = str(settings.ORCH_CARTRIDGE_DB_TENANT_COLUMN or "tenant_id").strip()
    profile_col = str(settings.ORCH_CARTRIDGE_DB_PROFILE_COLUMN or "agent_profile").strip()
    profile_id_col = str(settings.ORCH_CARTRIDGE_DB_PROFILE_ID_COLUMN or "profile_id").strip()
    version_col = str(settings.ORCH_CARTRIDGE_DB_VERSION_COLUMN or "profile_version").strip()
    status_col = str(settings.ORCH_CARTRIDGE_DB_STATUS_COLUMN or "status").strip()
    updated_col = str(settings.ORCH_CARTRIDGE_DB_UPDATED_COLUMN or "updated_at").strip()

    select_columns = ",".join([profile_col, profile_id_col, version_col, status_col])
    url = f"{rest_url.rstrip('/')}/{quote(table, safe='')}"
    params = {
        "select": select_columns,
        tenant_col: f"eq.{tenant}",
        "order": f"{updated_col}.desc",
        "limit": "1",
    }
    headers = {
        "apikey": service_role,
        "Authorization": f"Bearer {service_role}",
    }
    timeout = float(settings.ORCH_CARTRIDGE_DB_TIMEOUT_SECONDS or 1.8)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        logger.warning("cartridge_db_lookup_failed", tenant_id=tenant, error=str(exc))
        return None, "db_lookup_failed"

    if not isinstance(payload, list) or not payload:
        return None, "db_profile_not_found"

    row = payload[0]
    if not isinstance(row, dict):
        return None, "db_profile_not_found"

    profile_payload = row.get(profile_col)
    if isinstance(profile_payload, str):
        try:
            profile_payload = json.loads(profile_payload)
        except Exception:
            profile_payload = None

    if not isinstance(profile_payload, dict):
        return None, "db_profile_not_found"

    profile_dict = dict(profile_payload)
    if (
        not isinstance(profile_dict.get("profile_id"), str)
        or not str(profile_dict.get("profile_id")).strip()
    ):
        db_profile_id = str(row.get(profile_id_col) or "").strip()
        profile_dict["profile_id"] = db_profile_id or f"tenant_{tenant}"

    db_version = str(row.get(version_col) or "").strip()
    if db_version and not profile_dict.get("version"):
        profile_dict["version"] = db_version

    db_status = str(row.get(status_col) or "").strip()
    if db_status and not profile_dict.get("status"):
        profile_dict["status"] = db_status

    try:
        profile = AgentProfile.model_validate(profile_dict)
    except Exception as exc:
        logger.warning("cartridge_db_invalid_profile", tenant_id=tenant, error=str(exc))
        return None, "db_profile_invalid"

    return profile, "db_profile_applied"
