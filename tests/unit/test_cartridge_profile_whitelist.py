from types import SimpleNamespace

from app.cartridges.deps import resolve_agent_profile


def test_resolve_agent_profile_blocks_unauthorized_header_profile(monkeypatch) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor"]}',
    )

    request = SimpleNamespace(headers={"X-Agent-Profile": "legal_cl"})
    profile = resolve_agent_profile(tenant_id="tenant-iso", request=request)
    assert profile.profile_id == "iso_auditor"
