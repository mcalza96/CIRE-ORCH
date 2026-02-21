import asyncio

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request

from app.api import deps
from app.infrastructure.config import settings


def _request() -> Request:
    return Request({"type": "http", "headers": []})


def test_get_current_user_rejects_missing_token_when_required(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deps.get_current_user(_request(), None))
    assert exc.value.status_code == 401


def test_get_current_user_allows_local_bypass_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", False)
    user = asyncio.run(deps.get_current_user(_request(), None))
    assert user.user_id == "local-dev"
    assert user.roles == ["local_bypass"]


def test_get_current_user_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)

    def _raise(_: str):
        raise jwt.InvalidTokenError("invalid")

    monkeypatch.setattr(deps, "_decode_jwt_payload", _raise)

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad-token")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deps.get_current_user(_request(), creds))
    assert exc.value.status_code == 401


def test_get_current_user_returns_user_context(monkeypatch):
    monkeypatch.setattr(settings, "ORCH_AUTH_REQUIRED", True)

    payload = {
        "sub": "user-123",
        "email": "user@example.com",
        "tenant_ids": ["tenant-a", "tenant-b"],
        "roles": ["member"],
    }
    monkeypatch.setattr(deps, "_decode_jwt_payload", lambda _: payload)

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="good-token")
    user = asyncio.run(deps.get_current_user(_request(), creds))
    assert user.user_id == "user-123"
    assert user.email == "user@example.com"
    assert user.tenant_ids == ["tenant-a", "tenant-b"]
