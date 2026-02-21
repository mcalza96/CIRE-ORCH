import pytest
from fastapi.testclient import TestClient

from app.api.deps import UserContext, get_current_user
from app.api.server import app
from app.cartridges.dev_assignments import get_dev_profile_assignments_store
from app.cartridges.loader import get_cartridge_loader
from app.infrastructure.config import settings


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    monkeypatch.setattr(settings, "ORCH_CARTRIDGE_DB_ENABLED", False)
    monkeypatch.setattr(settings, "ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED", True)
    monkeypatch.setattr(
        settings,
        "ORCH_DEV_PROFILE_ASSIGNMENTS_FILE",
        str(tmp_path / "tenant_profile_assignments.json"),
    )
    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )
    with TestClient(app) as api_client:
        yield api_client
    app.dependency_overrides.clear()
    get_cartridge_loader.cache_clear()
    get_dev_profile_assignments_store.cache_clear()


def test_agent_profiles_lists_yaml_profiles(client):
    response = client.get("/api/v1/knowledge/agent-profiles")
    assert response.status_code == 200
    payload = response.json()
    ids = {row["id"] for row in payload["items"]}
    assert "base" in ids
    assert "iso_auditor" in ids


def test_tenant_profile_put_set_and_get(client):
    put_response = client.put(
        "/api/v1/knowledge/tenant-profile",
        json={"tenant_id": "tenant-a", "profile_id": "iso_auditor"},
    )
    assert put_response.status_code == 200
    put_payload = put_response.json()
    assert put_payload["override_profile_id"] == "iso_auditor"
    assert put_payload["resolution"]["source"] == "dev_map"

    get_response = client.get("/api/v1/knowledge/tenant-profile?tenant_id=tenant-a")
    assert get_response.status_code == 200
    get_payload = get_response.json()
    assert get_payload["override_profile_id"] == "iso_auditor"
    assert get_payload["resolution"]["source"] == "dev_map"


def test_tenant_profile_put_clear_override(client):
    set_response = client.put(
        "/api/v1/knowledge/tenant-profile",
        json={"tenant_id": "tenant-a", "profile_id": "iso_auditor"},
    )
    assert set_response.status_code == 200

    clear_response = client.put(
        "/api/v1/knowledge/tenant-profile",
        json={"tenant_id": "tenant-a", "clear": True},
    )
    assert clear_response.status_code == 200
    payload = clear_response.json()
    assert payload["override_profile_id"] is None


def test_tenant_profile_put_rejects_invalid_profile(client):
    response = client.put(
        "/api/v1/knowledge/tenant-profile",
        json={"tenant_id": "tenant-a", "profile_id": "missing_profile"},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "INVALID_AGENT_PROFILE"


def test_tenant_profile_put_enforces_tenant_membership(client):
    response = client.put(
        "/api/v1/knowledge/tenant-profile",
        json={"tenant_id": "tenant-b", "profile_id": "iso_auditor"},
    )
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "TENANT_ACCESS_DENIED"
