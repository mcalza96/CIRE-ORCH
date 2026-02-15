from dataclasses import dataclass
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.agent.application import HandleQuestionResult
from app.agent.models import AnswerDraft, QueryIntent, RetrievalDiagnostics, RetrievalPlan, ValidationResult
from app.api.deps import UserContext, get_current_user
from app.api.server import app
from app.api.v1.routes import knowledge as knowledge_routes
from app.api.v1.routes.knowledge import _build_use_case
from app.core.config import settings


@dataclass
class _FakeUseCase:
    last_command: Any = None

    async def execute(self, cmd):
        self.last_command = cmd
        return HandleQuestionResult(
            intent=QueryIntent(mode="explicativa", rationale="test"),
            plan=RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=20, summary_k=2),
            answer=AnswerDraft(text="ok", mode="explicativa", evidence=[]),
            validation=ValidationResult(accepted=True, issues=[]),
            retrieval=RetrievalDiagnostics(contract="legacy", strategy="test", partial=False, trace={}, scope_validation={}),
            clarification=None,
        )


def test_answer_requires_authentication(monkeypatch):
    fake_use_case = _FakeUseCase()
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[_build_use_case] = lambda: fake_use_case

    client = TestClient(app)
    response = client.post(
        "/api/v1/knowledge/answer",
        json={"query": "q", "tenant_id": "tenant-a"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 401
    payload = response.json()
    assert payload["detail"]["code"] == "UNAUTHORIZED"


def test_answer_denies_cross_tenant_request(monkeypatch):
    fake_use_case = _FakeUseCase()
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[_build_use_case] = lambda: fake_use_case
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/knowledge/answer",
        json={"query": "q", "tenant_id": "tenant-b"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 403
    payload = response.json()
    assert payload["detail"]["code"] == "TENANT_ACCESS_DENIED"
    assert fake_use_case.last_command is None


def test_answer_uses_authorized_tenant_and_user_context(monkeypatch):
    fake_use_case = _FakeUseCase()
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[_build_use_case] = lambda: fake_use_case
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    client = TestClient(app)
    response = client.post(
        "/api/v1/knowledge/answer",
        json={"query": "q"},
    )

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert fake_use_case.last_command is not None
    assert fake_use_case.last_command.tenant_id == "tenant-a"
    assert fake_use_case.last_command.user_id == "user-1"


def test_tenants_endpoint_returns_authorized_tenants(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a", "tenant-b"],
    )

    async def _fake_resolve_allowed_tenants(_):
        return ["tenant-a", "tenant-b"]

    async def _fake_fetch_tenant_names(_):
        return {"tenant-a": "Tenant A", "tenant-b": "Tenant B"}

    monkeypatch.setattr(knowledge_routes, "resolve_allowed_tenants", _fake_resolve_allowed_tenants)
    monkeypatch.setattr(knowledge_routes, "fetch_tenant_names", _fake_fetch_tenant_names)

    client = TestClient(app)
    response = client.get("/api/v1/knowledge/tenants")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["items"] == [
        {"id": "tenant-a", "name": "Tenant A"},
        {"id": "tenant-b", "name": "Tenant B"},
    ]


def test_collections_endpoint_denies_cross_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    client = TestClient(app)
    response = client.get("/api/v1/knowledge/collections?tenant_id=tenant-b")

    app.dependency_overrides.clear()
    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "TENANT_ACCESS_DENIED"


def test_collections_endpoint_returns_items_for_authorized_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    async def _fake_fetch_collections(_tenant_id):
        return [
            knowledge_routes.CollectionItem(id="c1", name="ISO", collection_key="iso"),
            knowledge_routes.CollectionItem(id="c2", name="45001", collection_key="iso45001"),
        ]

    monkeypatch.setattr(knowledge_routes, "_fetch_collections_from_rag", _fake_fetch_collections)

    client = TestClient(app)
    response = client.get("/api/v1/knowledge/collections?tenant_id=tenant-a")

    app.dependency_overrides.clear()
    assert response.status_code == 200
    assert response.json()["items"] == [
        {"id": "c1", "name": "ISO", "collection_key": "iso"},
        {"id": "c2", "name": "45001", "collection_key": "iso45001"},
    ]


def test_collections_endpoint_surfaces_upstream_auth_failure(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    app.dependency_overrides[get_current_user] = lambda: UserContext(
        user_id="user-1",
        tenant_ids=["tenant-a"],
    )

    async def _fake_fetch_collections(_tenant_id):
        request = httpx.Request("GET", "http://rag:8000/api/v1/ingestion/collections")
        response = httpx.Response(401, request=request)
        raise httpx.HTTPStatusError("unauthorized", request=request, response=response)

    monkeypatch.setattr(knowledge_routes, "_fetch_collections_from_rag", _fake_fetch_collections)

    client = TestClient(app)
    response = client.get("/api/v1/knowledge/collections?tenant_id=tenant-a")

    app.dependency_overrides.clear()
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "RAG_UPSTREAM_AUTH_FAILED"
    assert response.json()["detail"]["upstream_status"] == 401
