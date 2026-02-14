import asyncio

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app.api.deps import UserContext
from app.core.config import settings
from app.security import tenant_authorizer


def _request() -> Request:
    return Request({"type": "http", "headers": []})


def test_authorize_allows_requested_tenant_from_claims(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    user = UserContext(user_id="u1", tenant_ids=["tenant-a"])
    tenant = asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, "tenant-a"))
    assert tenant == "tenant-a"


def test_authorize_denies_unlisted_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    user = UserContext(user_id="u1", tenant_ids=["tenant-a"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, "tenant-b"))
    assert exc.value.status_code == 403


def test_authorize_resolves_single_tenant_when_request_omits(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    user = UserContext(user_id="u1", tenant_ids=["tenant-a"])
    tenant = asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, None))
    assert tenant == "tenant-a"


def test_authorize_requires_explicit_tenant_for_multi_tenant(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    user = UserContext(user_id="u1", tenant_ids=["tenant-a", "tenant-b"])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, None))
    assert exc.value.status_code == 400


def test_authorize_queries_membership_when_claims_missing(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)

    async def _fake_fetch(_: str) -> list[str]:
        return ["tenant-db"]

    monkeypatch.setattr(tenant_authorizer, "_fetch_membership_tenants", _fake_fetch)
    user = UserContext(user_id="u1", tenant_ids=[])

    tenant = asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, None))
    assert tenant == "tenant-db"


def test_authorize_requires_tenant_when_auth_bypass(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", False)
    user = UserContext(user_id="local-dev", tenant_ids=[])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(tenant_authorizer.authorize_requested_tenant(_request(), user, None))
    assert exc.value.status_code == 400
