from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import structlog

from app.cartridges.builtin_profiles import BUILTIN_PROFILES
from app.cartridges.models import AgentProfile
from app.core.config import PROJECT_ROOT, settings


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


logger = structlog.get_logger(__name__)


def _default_cartridges_dir() -> Path:
    configured = str(settings.ORCH_CARTRIDGES_DIR or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (PROJECT_ROOT / "app" / "cartridges").resolve()


def _safe_load_yaml(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("yaml_unavailable")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid profile payload in {path}")
    return raw


def _tenant_profile_map() -> dict[str, str]:
    raw = settings.ORCH_TENANT_PROFILE_MAP
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        logger.warning("cartridge_map_invalid_json")
        return {}
    out: dict[str, str] = {}
    if not isinstance(payload, dict):
        return {}
    for tenant_id, profile_id in payload.items():
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            continue
        if not isinstance(profile_id, str) or not profile_id.strip():
            continue
        out[tenant_id.strip()] = profile_id.strip()
    return out


class CartridgeLoader:
    def __init__(self, cartridges_dir: Path | None = None) -> None:
        self._cartridges_dir = cartridges_dir or _default_cartridges_dir()
        self._cache: dict[str, AgentProfile] = {}

    @property
    def cartridges_dir(self) -> Path:
        return self._cartridges_dir

    def resolve_profile_id(
        self, tenant_id: str | None, explicit_profile_id: str | None = None
    ) -> str:
        tenant = str(tenant_id or "").strip()
        
        if explicit_profile_id and explicit_profile_id.strip():
            candidate = explicit_profile_id.strip()
            # If a whitelist exists for this tenant, enforce it.
            whitelist = _tenant_profile_whitelist().get(tenant)
            if whitelist is not None:
                if candidate in whitelist:
                    return candidate
                # If candidate not in whitelist, fallback to mapped or base.
                logger.warning("unauthorized_profile_header_ignored", tenant=tenant, candidate=candidate)
            else:
                # No whitelist defined => allow candidate (for dev/open systems)
                return candidate

        mapped = _tenant_profile_map().get(tenant)
        if mapped:
            return mapped

        if tenant:
            candidate = self._cartridges_dir / f"{tenant}.yaml"
            if candidate.exists():
                return tenant

        default_profile = str(settings.ORCH_DEFAULT_PROFILE_ID or "base").strip()
        return default_profile or "base"

    def load(self, profile_id: str) -> AgentProfile:
        normalized = str(profile_id or "").strip()
        if not normalized:
            normalized = "base"

        if normalized in self._cache:
            return self._cache[normalized]

        path = self._cartridges_dir / f"{normalized}.yaml"
        payload: dict | None = None

        if path.exists():
            try:
                payload = _safe_load_yaml(path)
            except Exception as exc:
                logger.warning(
                    "cartridge_yaml_load_failed",
                    profile_id=normalized,
                    path=str(path),
                    error=str(exc),
                )

        if payload is None:
            builtin_payload = BUILTIN_PROFILES.get(normalized)
            if builtin_payload is not None:
                payload = deepcopy(builtin_payload)

        if payload is None:
            if normalized != "base":
                logger.warning("cartridge_not_found_fallback_base", profile_id=normalized)
                return self.load("base")
            raise FileNotFoundError(f"Base cartridge not found: {path}")

        payload.setdefault("profile_id", normalized)

        profile = AgentProfile.model_validate(payload)
        self._cache[normalized] = profile
        return profile

    def load_for_tenant(
        self,
        *,
        tenant_id: str | None,
        explicit_profile_id: str | None = None,
    ) -> AgentProfile:
        profile_id = self.resolve_profile_id(
            tenant_id=tenant_id, explicit_profile_id=explicit_profile_id
        )
        return self.load(profile_id)


@lru_cache(maxsize=1)
def get_cartridge_loader() -> CartridgeLoader:
    return CartridgeLoader()
