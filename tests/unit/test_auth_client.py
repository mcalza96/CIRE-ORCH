import asyncio
import base64
import json
from pathlib import Path

import pytest

from app.core import auth_client
from app.core.auth_client import (
    AuthInteractiveRequired,
    SessionPaths,
    SessionToken,
    decode_jwt_exp,
    ensure_access_token,
    get_valid_access_token,
    load_session,
    login_with_password,
    refresh_session,
    save_session,
)


def _jwt(exp: int, sub: str = "user-1") -> str:
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none"}).encode()).decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps({"sub": sub, "exp": exp}).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


def _paths(tmp_path: Path) -> SessionPaths:
    return SessionPaths(session_dir=tmp_path, session_file=tmp_path / "session.json")


def test_load_session_returns_none_for_corrupt_file(tmp_path):
    paths = _paths(tmp_path)
    paths.session_dir.mkdir(parents=True, exist_ok=True)
    paths.session_file.write_text("{invalid", encoding="utf-8")
    assert load_session(paths) is None


def test_save_session_sets_0600_permissions(tmp_path):
    paths = _paths(tmp_path)
    token = SessionToken(access_token=_jwt(9999999999), refresh_token="r1")
    save_session(token, paths)
    mode = paths.session_file.stat().st_mode & 0o777
    assert mode == 0o600


def test_decode_jwt_exp_and_valid_token(tmp_path):
    now = 1_900_000_000
    token = _jwt(now + 3600)
    assert decode_jwt_exp(token) == now + 3600
    save_session(SessionToken(access_token=token, refresh_token="r1"), _paths(tmp_path))
    assert get_valid_access_token(_paths(tmp_path), skew_seconds=0) == token


def test_get_valid_access_token_rejects_expired(tmp_path):
    token = _jwt(1)
    save_session(SessionToken(access_token=token, refresh_token="r1"), _paths(tmp_path))
    assert get_valid_access_token(_paths(tmp_path), skew_seconds=0) is None


def test_ensure_access_token_non_interactive_requires_token(tmp_path):
    with pytest.raises(AuthInteractiveRequired):
        asyncio.run(
            ensure_access_token(
                interactive=False,
                paths=_paths(tmp_path),
                supabase_url="https://example.supabase.co",
                supabase_anon_key="anon",
            )
        )


def test_refresh_session_updates_session_file(monkeypatch, tmp_path):
    old = SessionToken(access_token=_jwt(1), refresh_token="refresh-1")
    paths = _paths(tmp_path)
    save_session(old, paths)

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "access_token": _jwt(4_000_000_000, "user-2"),
                "refresh_token": "refresh-2",
                "token_type": "bearer",
                "user": {"id": "user-2"},
            }

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(auth_client.httpx, "AsyncClient", _Client)

    refreshed = asyncio.run(
        refresh_session(
            supabase_url="https://example.supabase.co",
            supabase_anon_key="anon",
            paths=paths,
        )
    )
    assert refreshed is not None
    assert refreshed.refresh_token == "refresh-2"
    assert load_session(paths).refresh_token == "refresh-2"


def test_login_with_password_raises_on_http_error(monkeypatch, tmp_path):
    class _Response:
        status_code = 401
        text = "invalid"

        @staticmethod
        def json():
            return {"error": "invalid"}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def post(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(auth_client.httpx, "AsyncClient", _Client)

    with pytest.raises(auth_client.AuthLoginFailed):
        asyncio.run(
            login_with_password(
                email="u@example.com",
                password="pw",
                supabase_url="https://example.supabase.co",
                supabase_anon_key="anon",
                paths=_paths(tmp_path),
            )
        )
