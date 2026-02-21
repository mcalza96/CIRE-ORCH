from __future__ import annotations

from fastapi import Request

from app.profiles.loader import get_profile_loader
from app.profiles.models import ResolvedAgentProfile
from app.infrastructure.config import settings


async def resolve_agent_profile(
    *,
    tenant_id: str | None,
    request: Request | None = None,
) -> ResolvedAgentProfile:
    loader = get_profile_loader()
    explicit_profile_id: str | None = None
    header_name = str(settings.ORCH_AGENT_PROFILE_HEADER or "X-Agent-Profile").strip()
    if request is not None and header_name:
        explicit_profile_id = request.headers.get(header_name)

    return await loader.resolve_for_tenant_async(
        tenant_id=tenant_id,
        explicit_profile_id=explicit_profile_id,
    )
