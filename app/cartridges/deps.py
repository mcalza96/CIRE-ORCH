from __future__ import annotations

import json

from fastapi import Request

from app.cartridges.loader import get_cartridge_loader
from app.cartridges.models import AgentProfile
from app.core.config import settings


def _tenant_profile_whitelist() -> dict[str, set[str]]:
    raw = str(settings.ORCH_TENANT_PROFILE_WHITELIST or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, set[str]] = {}
    for tenant_id, profiles in payload.items():
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            continue
        values: set[str] = set()
        if isinstance(profiles, list):
            for item in profiles:
                if isinstance(item, str) and item.strip():
                    values.add(item.strip())
        if values:
            out[tenant_id.strip()] = values
    return out


def resolve_agent_profile(*, tenant_id: str | None, request: Request | None = None) -> AgentProfile:
    loader = get_cartridge_loader()
    explicit_profile_id: str | None = None
    header_name = str(settings.ORCH_AGENT_PROFILE_HEADER or "X-Agent-Profile").strip()
    if request is not None and header_name:
        explicit_profile_id = request.headers.get(header_name)

    if explicit_profile_id and tenant_id:
        whitelist = _tenant_profile_whitelist()
        allowed = whitelist.get(str(tenant_id).strip())
        if allowed is not None and explicit_profile_id.strip() not in allowed:
            explicit_profile_id = None

    return loader.load_for_tenant(tenant_id=tenant_id, explicit_profile_id=explicit_profile_id)
