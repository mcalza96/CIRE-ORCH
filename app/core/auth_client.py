from __future__ import annotations

import base64
import getpass
import json
import os
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from app.core.config import settings


PromptFn = Callable[[str, bool], str]


class AuthClientError(RuntimeError):
    """Base auth error."""


class AuthConfigError(AuthClientError):
    """Raised when auth client configuration is missing."""


class AuthInteractiveRequired(AuthClientError):
    """Raised when interactive input is required but disabled."""


class AuthLoginFailed(AuthClientError):
    """Raised when Supabase login fails."""


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    session_file: Path

    @staticmethod
    def from_env() -> "SessionPaths":
        session_dir_raw = str(os.getenv("CHAT_SESSION_DIR") or "~/.cire/orch")
        session_file_raw = str(os.getenv("CHAT_SESSION_FILE") or str(Path(session_dir_raw) / "session.json"))
        session_dir = Path(session_dir_raw).expanduser()
        session_file = Path(session_file_raw).expanduser()
        return SessionPaths(session_dir=session_dir, session_file=session_file)


@dataclass
class SessionToken:
    access_token: str
    refresh_token: str | None = None
    expires_at: int | None = None
    expires_in: int | None = None
    token_type: str | None = None
    user: dict[str, Any] | None = None

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "SessionToken":
        return SessionToken(
            access_token=str(payload.get("access_token") or "").strip(),
            refresh_token=str(payload.get("refresh_token") or "").strip() or None,
            expires_at=int(payload.get("expires_at")) if payload.get("expires_at") else None,
            expires_in=int(payload.get("expires_in")) if payload.get("expires_in") else None,
            token_type=str(payload.get("token_type") or "").strip() or None,
            user=payload.get("user") if isinstance(payload.get("user"), dict) else {},
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "expires_in": self.expires_in,
            "token_type": self.token_type,
            "user": self.user or {},
        }


def _supabase_url() -> str:
    value = str(settings.SUPABASE_URL or "").strip()
    if value:
        return value
    raise AuthConfigError("SUPABASE_URL is not configured")


def _supabase_anon_key() -> str:
    value = str(settings.SUPABASE_ANON_KEY or "").strip()
    if value:
        return value
    raise AuthConfigError("SUPABASE_ANON_KEY is not configured")


def load_session(paths: SessionPaths | None = None) -> SessionToken | None:
    target = paths or SessionPaths.from_env()
    if not target.session_file.exists():
        return None
    try:
        payload = json.loads(target.session_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    token = SessionToken.from_payload(payload)
    if not token.access_token:
        return None
    return token


def save_session(session: SessionToken, paths: SessionPaths | None = None) -> None:
    target = paths or SessionPaths.from_env()
    target.session_dir.mkdir(parents=True, exist_ok=True)
    target.session_file.write_text(json.dumps(session.to_json(), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.chmod(target.session_file, stat.S_IRUSR | stat.S_IWUSR)


def decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = str(token or "").split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    payload += "=" * ((4 - len(payload) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("utf-8")).decode("utf-8")
        parsed = json.loads(decoded)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def decode_jwt_exp(token: str) -> int | None:
    payload = decode_jwt_payload(token)
    if not payload:
        return None
    try:
        exp = int(payload.get("exp") or 0)
    except Exception:
        return None
    return exp or None


def get_valid_access_token(paths: SessionPaths | None = None, skew_seconds: int = 60) -> str | None:
    session = load_session(paths=paths)
    if not session:
        return None
    exp = decode_jwt_exp(session.access_token)
    if not exp:
        return None
    if exp <= int(time.time()) + int(skew_seconds):
        return None
    return session.access_token


async def refresh_session(
    *,
    supabase_url: str | None = None,
    supabase_anon_key: str | None = None,
    paths: SessionPaths | None = None,
) -> SessionToken | None:
    target = paths or SessionPaths.from_env()
    current = load_session(paths=target)
    if not current or not current.refresh_token:
        return None

    url = (supabase_url or _supabase_url()).rstrip("/") + "/auth/v1/token?grant_type=refresh_token"
    anon_key = supabase_anon_key or _supabase_anon_key()
    payload = {"refresh_token": current.refresh_token}

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        response = await client.post(
            url,
            json=payload,
            headers={"apikey": anon_key, "Content-Type": "application/json"},
        )

    if response.status_code < 200 or response.status_code >= 300:
        return None
    body = response.json()
    if not isinstance(body, dict):
        return None
    refreshed = SessionToken.from_payload(body)
    if not refreshed.access_token:
        return None
    save_session(refreshed, paths=target)
    return refreshed


async def login_with_password(
    email: str,
    password: str,
    *,
    supabase_url: str | None = None,
    supabase_anon_key: str | None = None,
    paths: SessionPaths | None = None,
) -> SessionToken:
    email_norm = str(email or "").strip()
    password_norm = str(password or "")
    if not email_norm or not password_norm:
        raise AuthLoginFailed("Email and password are required")

    url = (supabase_url or _supabase_url()).rstrip("/") + "/auth/v1/token?grant_type=password"
    anon_key = supabase_anon_key or _supabase_anon_key()

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        response = await client.post(
            url,
            json={"email": email_norm, "password": password_norm},
            headers={"apikey": anon_key, "Content-Type": "application/json"},
        )

    if response.status_code < 200 or response.status_code >= 300:
        details = response.text
        raise AuthLoginFailed(f"Supabase login failed (HTTP {response.status_code}): {details}")
    body = response.json()
    if not isinstance(body, dict):
        raise AuthLoginFailed("Invalid login payload from Supabase")
    session = SessionToken.from_payload(body)
    if not session.access_token:
        raise AuthLoginFailed("Supabase login response missing access_token")
    save_session(session, paths=paths)
    return session


def _default_prompt(label: str, secret: bool = False) -> str:
    if secret:
        return getpass.getpass(f"{label}: ")
    return input(f"{label}: ")


async def ensure_access_token(
    *,
    interactive: bool,
    prompt_fn: PromptFn | None = None,
    supabase_url: str | None = None,
    supabase_anon_key: str | None = None,
    paths: SessionPaths | None = None,
) -> str:
    target = paths or SessionPaths.from_env()
    token = get_valid_access_token(paths=target)
    if token:
        return token

    refreshed = await refresh_session(
        supabase_url=supabase_url,
        supabase_anon_key=supabase_anon_key,
        paths=target,
    )
    if refreshed and refreshed.access_token:
        return refreshed.access_token

    if not interactive:
        raise AuthInteractiveRequired("A valid access token is required in non-interactive mode")

    prompt = prompt_fn or _default_prompt
    try:
        url = supabase_url or _supabase_url()
        anon_key = supabase_anon_key or _supabase_anon_key()
    except AuthConfigError:
        raise

    while True:
        email = str(prompt("📧 Email", False) or "").strip()
        password = str(prompt("🔑 Password", True) or "")
        try:
            session = await login_with_password(
                email=email,
                password=password,
                supabase_url=url,
                supabase_anon_key=anon_key,
                paths=target,
            )
        except AuthLoginFailed as exc:
            print(f"❌ {exc}")
            print("Intenta nuevamente.")
            continue
        return session.access_token
