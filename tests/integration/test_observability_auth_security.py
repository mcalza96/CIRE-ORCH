from fastapi.testclient import TestClient

from app.api.deps import UserContext, get_current_user
from app.api.server import app
from app.api.v1.routes import observability as obs_routes
from app.core.config import settings


def test_observability_denies_cross_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    client = TestClient(app)
    response = client.get("/api/v1/observability/batches/b1/progress?tenant_id=tenant-b")

    app.dependency_overrides.clear()
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "TENANT_ACCESS_DENIED"


def test_observability_progress_proxies_for_authorized_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    captured = {}

    async def _fake_proxy_json_get(**kwargs):
        captured.update(kwargs)
        return {"batch": {"id": "b1", "status": "processing"}, "observability": {"progress_percent": 42.0}}

    monkeypatch.setattr(obs_routes, "_proxy_json_get", _fake_proxy_json_get)

    client = TestClient(app)
    response = client.get("/api/v1/observability/batches/b1/progress?tenant_id=tenant-a")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["batch"]["id"] == "b1"
    assert captured["tenant_id"] == "tenant-a"
    assert captured["user_id"] == "user-1"
