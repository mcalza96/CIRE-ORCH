import asyncio

import pytest

from app.cartridges.dev_assignments import get_dev_profile_assignments_store
from app.cartridges.loader import CartridgeLoader, get_cartridge_loader
from app.cartridges.models import AgentProfile


def test_loader_falls_back_to_base_for_unknown_profile(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    loader = get_cartridge_loader()
    profile = asyncio.run(loader.load_for_tenant_async(tenant_id="unknown-tenant"))
    assert profile.profile_id == "base"


def test_loader_uses_tenant_profile_map(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    profile = asyncio.run(loader.load_for_tenant_async(tenant_id="tenant-iso"))
    assert profile.profile_id == "iso_auditor"


def test_loader_db_profile_has_priority_over_mapping(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_CARTRIDGE_DB_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    async def _fake_db_profile(_tenant_id: str):
        return AgentProfile(profile_id="tenant-private", version="2.0.0", status="active")

    monkeypatch.setattr(loader, "_fetch_db_profile_async", _fake_db_profile)
    profile = asyncio.run(loader.load_for_tenant_async(tenant_id="tenant-iso"))
    assert profile.profile_id == "tenant-private"


def test_loader_invalid_db_profile_falls_back_to_mapped(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_CARTRIDGE_DB_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    async def _fake_db_none(_tenant_id: str):
        return None

    monkeypatch.setattr(loader, "_fetch_db_profile_async", _fake_db_none)
    profile = asyncio.run(loader.load_for_tenant_async(tenant_id="tenant-iso"))
    assert profile.profile_id == "iso_auditor"


def test_loader_logs_unauthorized_header_override(monkeypatch) -> None:
    from app.cartridges import loader as loader_module
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor"]}',
    )
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')

    captured: dict[str, str] = {}

    def _fake_warning(event: str, **kwargs):
        captured["event"] = event
        captured["candidate"] = str(kwargs.get("requested_profile_id") or "")

    monkeypatch.setattr(loader_module.logger, "warning", _fake_warning)
    loader = get_cartridge_loader()

    resolved = loader.resolve_profile_id(tenant_id="tenant-iso", explicit_profile_id="legal_cl")
    assert resolved == "iso_auditor"
    assert captured.get("event") == "profile_header_override_denied"
    assert captured.get("candidate") == "legal_cl"


def test_loader_resolve_for_tenant_includes_resolution_metadata(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    resolved = asyncio.run(loader.resolve_for_tenant_async(tenant_id="tenant-iso"))
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.source == "tenant_map"
    assert resolved.resolution.applied_profile_id == "iso_auditor"


def test_loader_resolve_for_tenant_db_resolution_source(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_CARTRIDGE_DB_ENABLED", True)
    loader = get_cartridge_loader()

    async def _fake_db_profile(_tenant_id: str):
        return AgentProfile(profile_id="tenant-private", version="2.0.0", status="active")

    monkeypatch.setattr(loader, "_fetch_db_profile_async", _fake_db_profile)
    resolved = asyncio.run(loader.resolve_for_tenant_async(tenant_id="tenant-iso"))
    assert resolved.profile.profile_id == "tenant-private"
    assert resolved.resolution.source == "db"


def test_loader_resolution_reason_tracks_unauthorized_header_fallback(monkeypatch) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", False)
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor"]}',
    )
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"iso_auditor"}')
    loader = get_cartridge_loader()

    resolved = asyncio.run(
        loader.resolve_for_tenant_async(tenant_id="tenant-iso", explicit_profile_id="legal_cl")
    )
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.decision_reason == "unauthorized_header_override_fallback_tenant_profile_map_match"


def test_loader_validate_cartridge_files_strict_rejects_missing_v2_keys(tmp_path) -> None:
    invalid = tmp_path / "broken.yaml"
    invalid.write_text("profile_id: broken\nversion: 1.0.0\nstatus: active\n", encoding="utf-8")
    loader = CartridgeLoader(cartridges_dir=tmp_path)
    with pytest.raises(ValueError):
        loader.validate_cartridge_files_strict()


def test_loader_uses_dev_map_before_env_map(monkeypatch, tmp_path) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE", str(tmp_path / "assignments.json"))
    monkeypatch.setattr(settings, "ORCH_TENANT_PROFILE_MAP", '{"tenant-iso":"base"}')
    loader = get_cartridge_loader()
    loader.set_dev_profile_override(tenant_id="tenant-iso", profile_id="iso_auditor")

    resolved = asyncio.run(loader.resolve_for_tenant_async(tenant_id="tenant-iso"))
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.source == "dev_map"
    assert resolved.resolution.decision_reason == "dev_profile_map_match"


def test_loader_db_priority_over_dev_map(monkeypatch, tmp_path) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE", str(tmp_path / "assignments.json"))
    monkeypatch.setattr(settings, "ORCH_CARTRIDGE_DB_ENABLED", True)
    loader = get_cartridge_loader()
    loader.set_dev_profile_override(tenant_id="tenant-iso", profile_id="iso_auditor")

    async def _fake_db_profile(_tenant_id: str):
        return AgentProfile(profile_id="tenant-private", version="2.0.0", status="active")

    monkeypatch.setattr(loader, "_fetch_db_profile_async", _fake_db_profile)
    resolved = asyncio.run(loader.resolve_for_tenant_async(tenant_id="tenant-iso"))
    assert resolved.profile.profile_id == "tenant-private"
    assert resolved.resolution.source == "db"


def test_loader_authorized_header_has_priority_over_dev_map(monkeypatch, tmp_path) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE", str(tmp_path / "assignments.json"))
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor","legal_cl"]}',
    )
    loader = get_cartridge_loader()
    loader.set_dev_profile_override(tenant_id="tenant-iso", profile_id="iso_auditor")

    resolved = asyncio.run(
        loader.resolve_for_tenant_async(tenant_id="tenant-iso", explicit_profile_id="legal_cl")
    )
    assert resolved.profile.profile_id == "legal_cl"
    assert resolved.resolution.source == "header"


def test_loader_unauthorized_header_falls_back_to_dev_map(monkeypatch, tmp_path) -> None:
    from app.core.config import settings

    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE", str(tmp_path / "assignments.json"))
    monkeypatch.setattr(
        settings,
        "ORCH_TENANT_PROFILE_WHITELIST",
        '{"tenant-iso":["iso_auditor"]}',
    )
    loader = get_cartridge_loader()
    loader.set_dev_profile_override(tenant_id="tenant-iso", profile_id="iso_auditor")

    resolved = asyncio.run(
        loader.resolve_for_tenant_async(tenant_id="tenant-iso", explicit_profile_id="legal_cl")
    )
    assert resolved.profile.profile_id == "iso_auditor"
    assert resolved.resolution.source == "dev_map"
    assert resolved.resolution.decision_reason == "unauthorized_header_override_fallback_dev_profile_map_match"
