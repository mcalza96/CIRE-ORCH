from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import httpx
import structlog

from app.cartridges.db import fetch_db_profile_async
from app.cartridges.dev_assignments import get_dev_profile_assignments_store
from app.cartridges.models import AgentProfile, ProfileResolution, ResolvedAgentProfile
from app.infrastructure.config import PROJECT_ROOT, settings


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


logger = structlog.get_logger(__name__)
_REQUIRED_TOP_LEVEL_KEYS = {
    "profile_id",
    "version",
    "status",
    "meta",
    "identity",
    "router",
    "retrieval",
    "validation",
    "synthesis",
}


@dataclass(frozen=True)
class ProfileChoice:
    candidate_id: str
    source: Literal["header", "dev_map", "tenant_map", "tenant_yaml", "base"]
    decision_reason: str
    requested_profile_id: str | None


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


def _validate_v2_payload(path: Path, payload: dict) -> None:
    missing = sorted(key for key in _REQUIRED_TOP_LEVEL_KEYS if key not in payload)
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(f"Cartridge {path.name} missing required keys: {missing_csv}")


def _deep_merge(base_dict: dict, override_dict: dict) -> dict:
    """Recursively merges override_dict into base_dict. Returns a new dict without deepcopy."""
    result = base_dict.copy()
    for k, v in override_dict.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        elif isinstance(v, dict):
            result[k] = _deep_merge({}, v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _parse_json_map(raw: str | None) -> dict[str, str]:
    normalized = str(raw or "").strip()
    if not normalized:
        return {}
    try:
        payload = json.loads(normalized)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            continue
        if not isinstance(value, str) or not value.strip():
            continue
        out[key.strip()] = value.strip()
    return out


@lru_cache(maxsize=1)
def _tenant_profile_map() -> dict[str, str]:
    app_env = str(getattr(settings, "APP_ENV", "development") or "development").strip().lower()
    is_production = app_env in {"prod", "production"}
    raw_map = str(settings.ORCH_TENANT_PROFILE_MAP or "").strip()
    if is_production and raw_map:
        raise RuntimeError("ORCH_TENANT_PROFILE_MAP is not allowed in production")

    parsed = _parse_json_map(settings.ORCH_TENANT_PROFILE_MAP)
    if parsed:
        return parsed

    if settings.ORCH_TENANT_PROFILE_MAP:
        logger.warning("cartridge_map_invalid_json")
    return {}


@lru_cache(maxsize=1)
def _tenant_profile_whitelist() -> dict[str, set[str]]:
    raw = str(settings.ORCH_TENANT_PROFILE_WHITELIST or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning("cartridge_whitelist_invalid_json")
        return {}

    if not isinstance(payload, dict):
        return {}

    out: dict[str, set[str]] = {}
    for tenant_id, profiles in payload.items():
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            continue
        if not isinstance(profiles, list):
            continue
        allowed = {str(item).strip() for item in profiles if str(item).strip()}
        if allowed:
            out[tenant_id.strip()] = allowed
    return out


class CartridgeLoader:
    """Resuelve cartuchos con estrategia en cascada:

    1) DB privada por tenant (si esta habilitada)
    2) Perfil explicito/header autorizado
    3) Override dev local (tenant->perfil)
    4) Mapeo tenant->perfil por env
    5) Cartucho en filesystem (tenant_id.yaml)
    6) Perfil por defecto (`base`)
    """

    def __init__(self, cartridges_dir: Path | None = None) -> None:
        self._cartridges_dir = cartridges_dir or _default_cartridges_dir()
        self._profile_cache: dict[str, AgentProfile] = {}
        self._tenant_db_cache: dict[str, tuple[float, AgentProfile]] = {}
        self._last_db_resolution_reason: str = "db_not_checked"

    @property
    def cartridges_dir(self) -> Path:
        return self._cartridges_dir

    def _is_profile_allowed_for_tenant(self, *, tenant_id: str, profile_id: str) -> bool:
        allowed = _tenant_profile_whitelist().get(tenant_id)
        if allowed is None:
            return True
        return profile_id in allowed

    def dev_profile_assignments_enabled(self) -> bool:
        return bool(settings.ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED)

    def _profile_yaml_path(self, profile_id: str) -> Path:
        normalized = str(profile_id or "").strip() or "base"
        return self._cartridges_dir / f"{normalized}.yaml"

    def profile_exists(self, profile_id: str) -> bool:
        path = self._profile_yaml_path(profile_id)
        if not path.exists():
            return False
        try:
            self.load(profile_id)
        except Exception:
            return False
        return True

    def list_available_profile_entries(self) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        if not self._cartridges_dir.exists():
            return entries
        for path in sorted(self._cartridges_dir.glob("*.yaml")):
            profile_id = str(path.stem or "").strip()
            if not profile_id:
                continue
            try:
                parsed = self.load(profile_id)
            except Exception as exc:
                logger.warning(
                    "cartridge_profile_discovery_skip_invalid",
                    path=str(path),
                    error=str(exc),
                )
                continue
            entries.append(
                {
                    "id": profile_id,
                    "declared_profile_id": parsed.profile_id,
                    "version": parsed.version,
                    "status": parsed.status,
                    "description": parsed.meta.description,
                    "owner": parsed.meta.owner,
                }
            )
        return entries

    def get_dev_profile_override(self, tenant_id: str | None) -> str | None:
        if not self.dev_profile_assignments_enabled():
            return None
        return get_dev_profile_assignments_store().get(tenant_id)

    def set_dev_profile_override(self, *, tenant_id: str, profile_id: str) -> None:
        if not self.dev_profile_assignments_enabled():
            raise RuntimeError("dev_profile_assignments_disabled")
        normalized_profile = str(profile_id or "").strip()
        if not self.profile_exists(normalized_profile):
            raise ValueError(f"profile_not_found:{normalized_profile}")
        get_dev_profile_assignments_store().set(tenant_id=tenant_id, profile_id=normalized_profile)

    def clear_dev_profile_override(self, *, tenant_id: str) -> bool:
        if not self.dev_profile_assignments_enabled():
            return False
        return get_dev_profile_assignments_store().clear(tenant_id)

    def snapshot_dev_profile_overrides(self) -> dict[str, str]:
        if not self.dev_profile_assignments_enabled():
            return {}
        return get_dev_profile_assignments_store().snapshot()

    async def _fetch_db_profile_async(self, tenant_id: str) -> AgentProfile | None:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            self._last_db_resolution_reason = "empty_tenant_id"
            return None

        ttl = int(settings.ORCH_CARTRIDGE_DB_CACHE_TTL_SECONDS or 60)
        now = time.time()
        cached = self._tenant_db_cache.get(tenant)
        if cached and (now - cached[0]) < max(1, ttl):
            self._last_db_resolution_reason = "db_profile_cache_hit"
            return cached[1]

        profile, reason = await fetch_db_profile_async(tenant)
        self._last_db_resolution_reason = reason
        
        if profile is not None:
            self._tenant_db_cache[tenant] = (now, profile)
        return profile

    def _resolve_profile_choice(
        self, tenant_id: str | None, explicit_profile_id: str | None = None
    ) -> ProfileChoice:
        tenant = str(tenant_id or "").strip()
        requested = str(explicit_profile_id or "").strip() or None
        fallback_prefix = ""

        if requested:
            if self._is_profile_allowed_for_tenant(tenant_id=tenant, profile_id=requested):
                logger.info("profile_header_override_allowed", tenant_id=tenant, requested_profile_id=requested)
                return ProfileChoice(candidate_id=requested, source="header", decision_reason="authorized_header_override", requested_profile_id=requested)
            
            logger.warning("profile_header_override_denied", tenant_id=tenant, requested_profile_id=requested, reason="tenant_whitelist_denied")
            fallback_prefix = "unauthorized_header_override_fallback_"

        candidates = [
            ("dev_map", self.get_dev_profile_override(tenant), f"{fallback_prefix}dev_profile_map_match"),
            ("tenant_map", _tenant_profile_map().get(tenant), f"{fallback_prefix}tenant_profile_map_match"),
        ]

        if tenant and (self._cartridges_dir / f"{tenant}.yaml").exists():
            candidates.append(("tenant_yaml", tenant, f"{fallback_prefix}tenant_yaml_found"))

        default_base = str(settings.ORCH_DEFAULT_PROFILE_ID or "base").strip() or "base"
        candidates.append(("base", default_base, f"{fallback_prefix}default_profile_fallback"))

        for source, candidate, reason in candidates:
            if candidate:
                return ProfileChoice(
                    candidate_id=candidate,
                    source=source,
                    decision_reason=reason,
                    requested_profile_id=requested,
                )
        
        return ProfileChoice(candidate_id="base", source="base", decision_reason="hard_fallback_base", requested_profile_id=requested)

    def resolve_profile_id(
        self, tenant_id: str | None, explicit_profile_id: str | None = None
    ) -> str:
        choice = self._resolve_profile_choice(
            tenant_id=tenant_id, explicit_profile_id=explicit_profile_id
        )
        return choice.candidate_id

    def load(self, profile_id: str) -> AgentProfile:
        normalized = str(profile_id or "").strip() or "base"

        if normalized in self._profile_cache:
            return self._profile_cache[normalized]

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
            if normalized != "base":
                logger.warning("cartridge_not_found_fallback_base", profile_id=normalized)
                return self.load("base")
            raise FileNotFoundError(f"Base cartridge not found: {path}")

        extends = str(payload.get("extends") or "").strip()
        if extends and extends != normalized:
            try:
                base_profile = self.load(extends)
                base_payload = base_profile.model_dump(exclude={"profile_id", "extends"})
                payload = _deep_merge(base_payload, payload)
            except Exception as exc:
                logger.warning(
                    "cartridge_inheritance_failed",
                    profile_id=normalized,
                    extends=extends,
                    error=str(exc),
                )

        try:
            _validate_v2_payload(path, payload)
        except Exception as exc:
            logger.warning(
                "cartridge_validation_failed",
                profile_id=normalized,
                error=str(exc),
            )

        payload.setdefault("profile_id", normalized)
        profile = AgentProfile.model_validate(payload)
        self._profile_cache[normalized] = profile
        return profile

    def validate_cartridge_files_strict(self) -> None:
        # Safety guard: env JSON map is dev-only and must never be active in production.
        _tenant_profile_map()

        if not self._cartridges_dir.exists():
            raise RuntimeError(f"Cartridges directory not found: {self._cartridges_dir}")

        yaml_paths = sorted(self._cartridges_dir.glob("*.yaml"))
        if not yaml_paths:
            raise RuntimeError(f"No cartridge YAML files found in {self._cartridges_dir}")

        for path in yaml_paths:
            payload = _safe_load_yaml(path)
            
            extends = str(payload.get("extends") or "").strip()
            if extends and extends != path.stem:
                base_profile = self.load(extends)
                base_payload = base_profile.model_dump(exclude={"profile_id", "extends"})
                payload = _deep_merge(base_payload, payload)
            
            _validate_v2_payload(path, payload)
            payload.setdefault("profile_id", str(path.stem))
            AgentProfile.model_validate(payload)

    async def load_for_tenant_async(
        self,
        *,
        tenant_id: str | None,
        explicit_profile_id: str | None = None,
    ) -> AgentProfile:
        tenant = str(tenant_id or "").strip()

        # Paso 1: DB privada por tenant (si esta habilitada)
        profile_from_db = await self._fetch_db_profile_async(tenant)
        if profile_from_db is not None:
            return profile_from_db

        # Paso 2..6: explicit/dev_map/map/filesystem/default
        profile_id = await asyncio.to_thread(
            self.resolve_profile_id,
            tenant_id=tenant,
            explicit_profile_id=explicit_profile_id
        )
        return await asyncio.to_thread(self.load, profile_id)

    async def resolve_for_tenant_async(
        self,
        *,
        tenant_id: str | None,
        explicit_profile_id: str | None = None,
    ) -> ResolvedAgentProfile:
        tenant = str(tenant_id or "").strip()
        requested = str(explicit_profile_id or "").strip() or None

        self._last_db_resolution_reason = "db_not_checked"
        profile_from_db = await self._fetch_db_profile_async(tenant)
        if profile_from_db is not None:
            resolution = ProfileResolution(
                source="db",
                requested_profile_id=requested,
                applied_profile_id=profile_from_db.profile_id,
                decision_reason=self._last_db_resolution_reason or "db_profile_applied",
            )
            logger.info(
                "profile_resolution_decision",
                tenant_id=tenant,
                source=resolution.source,
                requested_profile_id=resolution.requested_profile_id,
                applied_profile_id=resolution.applied_profile_id,
                decision_reason=resolution.decision_reason,
            )
            return ResolvedAgentProfile(profile=profile_from_db, resolution=resolution)

        choice = await asyncio.to_thread(
            self._resolve_profile_choice,
            tenant_id=tenant,
            explicit_profile_id=explicit_profile_id,
        )
        profile = await asyncio.to_thread(self.load, choice.candidate_id)
        resolved_source = choice.source
        resolved_reason = choice.decision_reason
        if profile.profile_id != choice.candidate_id:
            resolved_source = "base"
            resolved_reason = f"profile_not_found_fallback_base_from_{choice.source}"

        resolution = ProfileResolution(
            source=resolved_source,
            requested_profile_id=choice.requested_profile_id,
            applied_profile_id=profile.profile_id,
            decision_reason=resolved_reason,
        )
        logger.info(
            "profile_resolution_decision",
            tenant_id=tenant,
            source=resolution.source,
            requested_profile_id=resolution.requested_profile_id,
            applied_profile_id=resolution.applied_profile_id,
            decision_reason=resolution.decision_reason,
            db_reason=self._last_db_resolution_reason,
        )
        return ResolvedAgentProfile(profile=profile, resolution=resolution)



@lru_cache(maxsize=1)
def get_cartridge_loader() -> CartridgeLoader:
    return CartridgeLoader()
