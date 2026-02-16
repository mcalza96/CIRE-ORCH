from app.cartridges.loader import get_cartridge_loader


def test_loader_falls_back_to_base_for_unknown_profile() -> None:
    get_cartridge_loader.cache_clear()
    loader = get_cartridge_loader()
    profile = loader.load_for_tenant(tenant_id="unknown-tenant")
    assert profile.profile_id == "base"


def test_loader_uses_tenant_profile_map(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    profile = loader.load_for_tenant(tenant_id="tenant-iso")
    assert profile.profile_id == "iso_auditor"
