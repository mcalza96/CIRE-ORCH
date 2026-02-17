from __future__ import annotations

import asyncio
import json
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import httpx
import structlog

from app.cartridges.dev_assignments import get_dev_profile_assignments_store
from app.cartridges.models import AgentProfile, ProfileResolution, ResolvedAgentProfile
from app.core.config import PROJECT_ROOT, settings


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


def _tenant_profile_map() -> dict[str, str]:
    parsed = _parse_json_map(settings.ORCH_TENANT_PROFILE_MAP)
    if parsed:
        return parsed

    if settings.ORCH_TENANT_PROFILE_MAP:
        logger.warning("cartridge_map_invalid_json")
    return {}


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
            payload = _safe_load_yaml(path)
            _validate_v2_payload(path, payload)
            AgentProfile.model_validate(payload)
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
                payload = _safe_load_yaml(path)
                _validate_v2_payload(path, payload)
                parsed = AgentProfile.model_validate(payload)
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
        if not bool(settings.ORCH_CARTRIDGE_DB_ENABLED):
            self._last_db_resolution_reason = "db_override_disabled"
            return None

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

        rest_url = settings.resolved_supabase_rest_url
        service_role = str(settings.SUPABASE_SERVICE_ROLE_KEY or "").strip()
        if not rest_url or not service_role:
            self._last_db_resolution_reason = "db_credentials_missing"
            return None

        table = str(settings.ORCH_CARTRIDGE_DB_TABLE or "tenant_configs").strip()
        tenant_col = str(settings.ORCH_CARTRIDGE_DB_TENANT_COLUMN or "tenant_id").strip()
        profile_col = str(settings.ORCH_CARTRIDGE_DB_PROFILE_COLUMN or "agent_profile").strip()
        profile_id_col = str(settings.ORCH_CARTRIDGE_DB_PROFILE_ID_COLUMN or "profile_id").strip()
        version_col = str(settings.ORCH_CARTRIDGE_DB_VERSION_COLUMN or "profile_version").strip()
        status_col = str(settings.ORCH_CARTRIDGE_DB_STATUS_COLUMN or "status").strip()
        updated_col = str(settings.ORCH_CARTRIDGE_DB_UPDATED_COLUMN or "updated_at").strip()

        select_columns = ",".join([profile_col, profile_id_col, version_col, status_col])
        url = f"{rest_url.rstrip('/')}/{quote(table, safe='')}"
        params = {
            "select": select_columns,
            tenant_col: f"eq.{tenant}",
            "order": f"{updated_col}.desc",
            "limit": "1",
        }
        headers = {
            "apikey": service_role,
            "Authorization": f"Bearer {service_role}",
        }
        timeout = float(settings.ORCH_CARTRIDGE_DB_TIMEOUT_SECONDS or 1.8)

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.warning("cartridge_db_lookup_failed", tenant_id=tenant, error=str(exc))
            self._last_db_resolution_reason = "db_lookup_failed"
            return None

        if not isinstance(payload, list) or not payload:
            self._last_db_resolution_reason = "db_profile_not_found"
            return None

        row = payload[0]
        if not isinstance(row, dict):
            self._last_db_resolution_reason = "db_profile_not_found"
            return None

        profile_payload = row.get(profile_col)
        if isinstance(profile_payload, str):
            try:
                profile_payload = json.loads(profile_payload)
            except Exception:
                profile_payload = None

        if not isinstance(profile_payload, dict):
            self._last_db_resolution_reason = "db_profile_not_found"
            return None

        profile_dict = dict(profile_payload)
        if (
            not isinstance(profile_dict.get("profile_id"), str)
            or not str(profile_dict.get("profile_id")).strip()
        ):
            db_profile_id = str(row.get(profile_id_col) or "").strip()
            profile_dict["profile_id"] = db_profile_id or f"tenant_{tenant}"

        db_version = str(row.get(version_col) or "").strip()
        if db_version and not profile_dict.get("version"):
            profile_dict["version"] = db_version

        db_status = str(row.get(status_col) or "").strip()
        if db_status and not profile_dict.get("status"):
            profile_dict["status"] = db_status

        try:
            profile = AgentProfile.model_validate(profile_dict)
        except Exception as exc:
            logger.warning("cartridge_db_invalid_profile", tenant_id=tenant, error=str(exc))
            self._last_db_resolution_reason = "db_profile_invalid"
            return None

        self._tenant_db_cache[tenant] = (now, profile)
        self._last_db_resolution_reason = "db_profile_applied"
        return profile

    def _resolve_profile_choice(
        self, tenant_id: str | None, explicit_profile_id: str | None = None
    ) -> tuple[str, str, str, str | None]:
        tenant = str(tenant_id or "").strip()
        requested = str(explicit_profile_id or "").strip() or None
        override_denied = False

        if requested:
            candidate = requested
            if self._is_profile_allowed_for_tenant(tenant_id=tenant, profile_id=candidate):
                logger.info(
                    "profile_header_override_allowed",
                    tenant_id=tenant,
                    requested_profile_id=candidate,
                )
                return candidate, "header", "authorized_header_override", requested
            logger.warning(
                "profile_header_override_denied",
                tenant_id=tenant,
                requested_profile_id=candidate,
                reason="tenant_whitelist_denied",
            )
            override_denied = True

        dev_mapped = self.get_dev_profile_override(tenant)
        if dev_mapped:
            if override_denied:
                return (
                    dev_mapped,
                    "dev_map",
                    "unauthorized_header_override_fallback_dev_map",
                    requested,
                )
            return dev_mapped, "dev_map", "dev_profile_map_match", requested

        mapped = _tenant_profile_map().get(tenant)
        if mapped:
            if override_denied:
                return (
                    mapped,
                    "tenant_map",
                    "unauthorized_header_override_fallback_tenant_map",
                    requested,
                )
            return mapped, "tenant_map", "tenant_profile_map_match", requested

        if tenant:
            candidate_path = self._cartridges_dir / f"{tenant}.yaml"
            if candidate_path.exists():
                if override_denied:
                    return (
                        tenant,
                        "tenant_yaml",
                        "unauthorized_header_override_fallback_tenant_yaml",
                        requested,
                    )
                return tenant, "tenant_yaml", "tenant_yaml_found", requested

        default_profile = str(settings.ORCH_DEFAULT_PROFILE_ID or "base").strip()
        if override_denied:
            return (
                default_profile or "base",
                "base",
                "unauthorized_header_override_fallback_base",
                requested,
            )
        return default_profile or "base", "base", "default_profile_fallback", requested

    def resolve_profile_id(
        self, tenant_id: str | None, explicit_profile_id: str | None = None
    ) -> str:
        profile_id, _, _, _ = self._resolve_profile_choice(
            tenant_id=tenant_id, explicit_profile_id=explicit_profile_id
        )
        return profile_id

    def load(self, profile_id: str) -> AgentProfile:
        normalized = str(profile_id or "").strip() or "base"

        if normalized in self._profile_cache:
            return self._profile_cache[normalized]

        path = self._cartridges_dir / f"{normalized}.yaml"
        payload: dict | None = None

        if path.exists():
            try:
                payload = _safe_load_yaml(path)
                _validate_v2_payload(path, payload)
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

        payload.setdefault("profile_id", normalized)
        profile = AgentProfile.model_validate(payload)
        self._profile_cache[normalized] = profile
        return profile

    def validate_cartridge_files_strict(self) -> None:
        if not self._cartridges_dir.exists():
            raise RuntimeError(f"Cartridges directory not found: {self._cartridges_dir}")

        yaml_paths = sorted(self._cartridges_dir.glob("*.yaml"))
        if not yaml_paths:
            raise RuntimeError(f"No cartridge YAML files found in {self._cartridges_dir}")

        for path in yaml_paths:
            payload = _safe_load_yaml(path)
            _validate_v2_payload(path, payload)
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
        profile_id = self.resolve_profile_id(
            tenant_id=tenant, explicit_profile_id=explicit_profile_id
        )
        return self.load(profile_id)

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

        candidate_id, source, decision_reason, requested_candidate = self._resolve_profile_choice(
            tenant_id=tenant,
            explicit_profile_id=explicit_profile_id,
        )
        profile = self.load(candidate_id)
        resolved_source = source
        resolved_reason = decision_reason
        if profile.profile_id != candidate_id:
            resolved_source = "base"
            resolved_reason = f"profile_not_found_fallback_base_from_{source}"

        resolution = ProfileResolution(
            source=resolved_source,
            requested_profile_id=requested_candidate,
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

    def load_for_tenant(
        self,
        *,
        tenant_id: str | None,
        explicit_profile_id: str | None = None,
    ) -> AgentProfile:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.load_for_tenant_async(
                    tenant_id=tenant_id,
                    explicit_profile_id=explicit_profile_id,
                )
            )
        raise RuntimeError("Use load_for_tenant_async inside async context")


@lru_cache(maxsize=1)
def get_cartridge_loader() -> CartridgeLoader:
    return CartridgeLoader()
