from __future__ import annotations

import json
import threading
from functools import lru_cache
from pathlib import Path

import structlog

from app.infrastructure.config import PROJECT_ROOT, settings


logger = structlog.get_logger(__name__)


def _resolve_store_path() -> Path:
    configured = str(settings.ORCH_DEV_PROFILE_ASSIGNMENTS_FILE or "").strip()
    if not configured:
        configured = ".state/tenant_profile_assignments.json"
    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = (PROJECT_ROOT / candidate).resolve()
    return candidate


def _normalize_assignments(payload: object) -> dict[str, str]:
    if not isinstance(payload, dict):
        return {}
    normalized: dict[str, str] = {}
    for tenant_id, profile_id in payload.items():
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            continue
        if not isinstance(profile_id, str) or not profile_id.strip():
            continue
        normalized[tenant_id.strip()] = profile_id.strip()
    return normalized


class DevProfileAssignmentsStore:
    """Dev-only profile assignment store persisted in a local JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or _resolve_store_path()
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    def _read_assignments(self) -> dict[str, str]:
        if not bool(settings.ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED):
            return {}
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "profile_dev_override_store_read_failed",
                path=str(self._path),
                error=str(exc),
            )
            return {}
        return _normalize_assignments(raw)

    def _write_assignments(self, assignments: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = json.dumps(assignments, ensure_ascii=True, indent=2) + "\n"
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self._path)

    def get(self, tenant_id: str | None) -> str | None:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            return None
        with self._lock:
            return self._read_assignments().get(tenant)

    def set(self, tenant_id: str, profile_id: str) -> None:
        if not bool(settings.ORCH_DEV_PROFILE_ASSIGNMENTS_ENABLED):
            raise RuntimeError("dev_profile_assignments_disabled")
        tenant = str(tenant_id or "").strip()
        profile = str(profile_id or "").strip()
        if not tenant or not profile:
            raise ValueError("tenant_id and profile_id are required")
        with self._lock:
            current = self._read_assignments()
            current[tenant] = profile
            self._write_assignments(current)

    def clear(self, tenant_id: str | None) -> bool:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            return False
        with self._lock:
            current = self._read_assignments()
            existed = tenant in current
            if existed:
                current.pop(tenant, None)
                self._write_assignments(current)
            return existed

    def snapshot(self) -> dict[str, str]:
        with self._lock:
            return dict(self._read_assignments())


@lru_cache(maxsize=1)
def get_dev_profile_assignments_store() -> DevProfileAssignmentsStore:
    return DevProfileAssignmentsStore()
