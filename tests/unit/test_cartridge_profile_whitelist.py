import asyncio
from types import SimpleNamespace
from typing import Any, cast

from app.cartridges.deps import resolve_agent_profile


def test_resolve_agent_profile_blocks_unauthorized_header_profile(monkeypatch) -> None:
    from app.infrastructure.config import settings

    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor"]}',
    )

    request = SimpleNamespace(headers={"X-Agent-Profile": "legal_cl"})
    resolved = asyncio.run(resolve_agent_profile(tenant_id="tenant-iso", request=cast(Any, request)))
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.source == "tenant_map"


def test_resolve_agent_profile_allows_authorized_header_profile(monkeypatch) -> None:
    from app.infrastructure.config import settings

    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"base"}')
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor","base"]}',
    )

    request = SimpleNamespace(headers={"X-Agent-Profile": "iso_auditor"})
    resolved = asyncio.run(resolve_agent_profile(tenant_id="tenant-iso", request=cast(Any, request)))
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.source == "header"
