from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import uuid4

import httpx
import jwt
import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import settings


logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)


class UserContext(BaseModel):
    user_id: str
    email: str | None = None
    roles: list[str] = Field(default_factory=list)
    tenant_ids: list[str] = Field(default_factory=list)
    raw_claims: dict[str, Any] = Field(default_factory=dict)


def _request_id(request: Request) -> str:
    return str(request.headers.get("X-Request-ID") or request.headers.get("X-Correlation-ID") or uuid4())


@lru_cache(maxsize=4)
def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _extract_tenant_ids(claims: dict[str, Any]) -> list[str]:
    app_metadata = claims.get("app_metadata") if isinstance(claims.get("app_metadata"), dict) else {}
    user_metadata = claims.get("user_metadata") if isinstance(claims.get("user_metadata"), dict) else {}

    candidates: list[str] = []
    for source in (
        claims.get("tenant_id"),
        app_metadata.get("tenant_id"),
        app_metadata.get("organization_id"),
        user_metadata.get("tenant_id"),
    ):
        candidates.extend(_as_str_list(source))

    candidates.extend(_as_str_list(claims.get("tenant_ids")))
    candidates.extend(_as_str_list(app_metadata.get("tenant_ids")))
    candidates.extend(_as_str_list(user_metadata.get("tenant_ids")))

    organizations = claims.get("organizations")
    if isinstance(organizations, list):
        for org in organizations:
            if isinstance(org, dict):
                for key in ("tenant_id", "id", "organization_id"):
                    value = org.get(key)
                    if isinstance(value, str) and value.strip():
                        candidates.append(value.strip())

    return sorted(set(candidates))


def _extract_roles(claims: dict[str, Any]) -> list[str]:
    app_metadata = claims.get("app_metadata") if isinstance(claims.get("app_metadata"), dict) else {}
    role_candidates = [
        claims.get("role"),
        claims.get("roles"),
        app_metadata.get("role"),
        app_metadata.get("roles"),
    ]
    out: list[str] = []
    for candidate in role_candidates:
        out.extend(_as_str_list(candidate))
    return sorted(set(out))


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    jwks_url = settings.resolved_supabase_jwks_url
    if not jwks_url:
        raise RuntimeError("SUPABASE_URL or SUPABASE_JWKS_URL must be configured")

    jwks_client = _jwks_client(jwks_url)
    signing_key = jwks_client.get_signing_key_from_jwt(token).key
    verify_aud = bool(str(settings.SUPABASE_JWT_AUDIENCE or "").strip())
    payload = jwt.decode(
        token,
        signing_key,
        algorithms=["RS256"],
        audience=settings.SUPABASE_JWT_AUDIENCE if verify_aud else None,
        options={"verify_aud": verify_aud},
    )
    if not isinstance(payload, dict):
        raise jwt.InvalidTokenError("invalid_jwt_payload")
    return payload


async def _fetch_supabase_user_profile(token: str) -> dict[str, Any] | None:
    supabase_url = str(settings.SUPABASE_URL or "").strip()
    anon_key = str(settings.SUPABASE_ANON_KEY or "").strip()
    if not supabase_url or not anon_key:
        return None

    url = supabase_url.rstrip("/") + "/auth/v1/user"
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {token}",
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
    except Exception:
        return None

    data = response.json()
    if not isinstance(data, dict):
        return None

    user_id = str(data.get("id") or "").strip()
    if not user_id:
        return None

    # Normalize profile payload into JWT-like claims consumed by downstream code.
    app_metadata = data.get("app_metadata") if isinstance(data.get("app_metadata"), dict) else {}
    user_metadata = data.get("user_metadata") if isinstance(data.get("user_metadata"), dict) else {}
    role = data.get("role")
    claims: dict[str, Any] = {
        "sub": user_id,
        "email": data.get("email"),
        "role": role,
        "app_metadata": app_metadata,
        "user_metadata": user_metadata,
    }
    if isinstance(app_metadata.get("tenant_ids"), list):
        claims["tenant_ids"] = app_metadata.get("tenant_ids")
    elif isinstance(user_metadata.get("tenant_ids"), list):
        claims["tenant_ids"] = user_metadata.get("tenant_ids")
    return claims


def _http_error(request: Request, status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
            "request_id": _request_id(request),
        },
    )


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserContext:
    if not settings.ORCH_AUTH_REQUIRED and credentials is None:
        logger.info("auth_bypass", decision="auth_bypass", user_id="local-dev")
        return UserContext(user_id="local-dev", roles=["local_bypass"], tenant_ids=[])

    if credentials is None:
        logger.warning("auth_fail", decision="auth_fail", reason="missing_bearer")
        raise _http_error(
            request,
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHORIZED",
            "Missing bearer token",
        )

    token = str(credentials.credentials or "").strip()
    if not token:
        logger.warning("auth_fail", decision="auth_fail", reason="empty_bearer")
        raise _http_error(
            request,
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHORIZED",
            "Invalid bearer token",
        )

    try:
        claims = _decode_jwt_payload(token)
    except RuntimeError:
        logger.error("auth_fail", decision="auth_fail", reason="jwt_misconfigured")
        raise _http_error(
            request,
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "AUTH_MISCONFIGURED",
            "JWT validation is misconfigured",
        )
    except jwt.PyJWTError:
        claims = await _fetch_supabase_user_profile(token)
        if not claims:
            logger.warning("auth_fail", decision="auth_fail", reason="invalid_jwt")
            raise _http_error(
                request,
                status.HTTP_401_UNAUTHORIZED,
                "UNAUTHORIZED",
                "Invalid or expired token",
            )
        logger.info("auth_jwt_fallback", decision="auth_ok", source="supabase_user_profile")

    user_id = str(claims.get("sub") or "").strip()
    if not user_id:
        logger.warning("auth_fail", decision="auth_fail", reason="missing_sub")
        raise _http_error(
            request,
            status.HTTP_401_UNAUTHORIZED,
            "UNAUTHORIZED",
            "Token missing subject",
        )

    context = UserContext(
        user_id=user_id,
        email=str(claims.get("email")).strip() if claims.get("email") else None,
        roles=_extract_roles(claims),
        tenant_ids=_extract_tenant_ids(claims),
        raw_claims=claims,
    )
    logger.info(
        "auth_ok",
        decision="auth_ok",
        user_id=context.user_id,
        tenant_count=len(context.tenant_ids),
    )
    return context
