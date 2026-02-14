from __future__ import annotations

import json
import logging
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests

logger = logging.getLogger(__name__)

TENANT_HEADER_REQUIRED_CODE = "TENANT_HEADER_REQUIRED"
TENANT_MISMATCH_CODE = "TENANT_MISMATCH"


def user_message_for_tenant_error_code(code: str) -> str:
    normalized = str(code or "").upper()
    if normalized == TENANT_HEADER_REQUIRED_CODE:
        return "Selecciona una organización antes de continuar."
    if normalized == TENANT_MISMATCH_CODE:
        return "Conflicto de organización. Recarga la vista o vuelve a seleccionar tu organización."
    return "Error de organización. Verifica tu organización activa e inténtalo de nuevo."


def _tenant_fingerprint(tenant_id: Optional[str]) -> str:
    if not tenant_id:
        return "none"
    return sha256(tenant_id.encode("utf-8")).hexdigest()[:10]


def _sanitize_tenant_id(tenant_id: Optional[str]) -> Optional[str]:
    if tenant_id is None:
        return None
    normalized = str(tenant_id).strip()
    if not normalized:
        return None
    return normalized


@dataclass
class TenantContext:
    tenant_id: Optional[str] = None
    storage_path: Optional[Path] = None
    _loaded_from_storage: bool = False

    def __post_init__(self) -> None:
        self.storage_path = Path(self.storage_path) if self.storage_path else None
        if self.tenant_id is not None:
            normalized = _sanitize_tenant_id(self.tenant_id)
            if not normalized:
                raise ValueError("tenant_id must be a non-empty string")
            self.tenant_id = normalized
            self._persist()

    def _ensure_loaded(self) -> None:
        if self._loaded_from_storage:
            return
        self.reload()
        self._loaded_from_storage = True

    def get_tenant(self) -> Optional[str]:
        self._ensure_loaded()
        return self.tenant_id

    def set_tenant(self, tenant_id: str) -> None:
        normalized = _sanitize_tenant_id(tenant_id)
        if not normalized:
            raise ValueError("tenant_id must be a non-empty string")
        self.tenant_id = normalized
        self._loaded_from_storage = True
        self._persist()

    def clear_tenant(self) -> None:
        self.tenant_id = None
        self._loaded_from_storage = True
        self._persist()

    def reload(self) -> Optional[str]:
        if not self.storage_path or not self.storage_path.exists():
            return self.tenant_id
        try:
            raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(
                "tenant_context_storage_invalid",
                extra={"event": "tenant_context_storage_invalid", "storage_path": str(self.storage_path)},
            )
            return self.tenant_id

        raw_tenant = raw.get("tenant_id") if isinstance(raw, dict) else None
        normalized = _sanitize_tenant_id(raw_tenant)
        if raw_tenant is not None and not normalized:
            logger.warning(
                "tenant_context_storage_invalid_tenant",
                extra={"event": "tenant_context_storage_invalid_tenant", "storage_path": str(self.storage_path)},
            )
            return self.tenant_id

        self.tenant_id = normalized
        self._loaded_from_storage = True
        return self.tenant_id

    def _persist(self) -> None:
        if not self.storage_path:
            return
        payload = {"tenant_id": self.tenant_id}
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


@dataclass
class CireRagApiError(Exception):
    status: int
    code: str
    message: str
    details: Any
    request_id: str

    def __str__(self) -> str:
        return f"[{self.status}] {self.code}: {self.message} (request_id={self.request_id})"


@dataclass
class TenantSelectionRequiredError(Exception):
    message: str = "Selecciona una organización antes de continuar."

    def __str__(self) -> str:
        return self.message


@dataclass
class TenantMismatchLocalError(Exception):
    message: str = "Conflicto de organización. Recarga la vista o vuelve a seleccionar tu organización."

    def __str__(self) -> str:
        return self.message


@dataclass
class TenantProtocolError(Exception):
    status: int
    code: str
    message: str
    user_message: str
    request_id: str
    details: Any = None

    def __str__(self) -> str:
        return f"[{self.status}] {self.code}: {self.user_message} (request_id={self.request_id})"


def _build_auth_headers(
    api_key: Optional[str],
    default_headers: Optional[Dict[str, str]],
) -> Dict[str, str]:
    headers = dict(default_headers or {})
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _raise_from_http_error(status_code: int, response_text: str, response_headers: Dict[str, str], payload: Any) -> None:
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        error = {
            "code": "UNPARSEABLE_ERROR",
            "message": response_text,
            "details": None,
            "request_id": response_headers.get("X-Correlation-ID", "unknown"),
        }

    code = str(error.get("code") or "UNKNOWN_ERROR")
    message = str(error.get("message") or "Request failed")
    request_id = str(error.get("request_id") or response_headers.get("X-Correlation-ID") or "unknown")
    details = error.get("details")

    if code in {TENANT_HEADER_REQUIRED_CODE, TENANT_MISMATCH_CODE}:
        raise TenantProtocolError(
            status=status_code,
            code=code,
            message=message,
            user_message=user_message_for_tenant_error_code(code),
            request_id=request_id,
            details=details,
        )

    raise CireRagApiError(status=status_code, code=code, message=message, details=details, request_id=request_id)


class CireRagClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
        tenant_context: Optional[TenantContext] = None,
        tenant_storage_path: Optional[str | Path] = None,
    ):
        if tenant_context is not None and tenant_storage_path is not None:
            raise ValueError("Provide either tenant_context or tenant_storage_path, not both")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self.session = session or requests.Session()
        self.tenant_context = tenant_context or TenantContext(storage_path=Path(tenant_storage_path) if tenant_storage_path else None)

    def close(self) -> None:
        self.session.close()

    def set_tenant(self, tenant_id: str) -> None:
        self.tenant_context.set_tenant(tenant_id)

    def get_tenant(self) -> Optional[str]:
        return self.tenant_context.get_tenant()

    def clear_tenant(self) -> None:
        self.tenant_context.clear_tenant()

    def create_document(self, file_path: str | Path, metadata: Dict[str, Any] | str) -> Dict[str, Any]:
        path = Path(file_path)
        metadata_json = metadata if isinstance(metadata, str) else json.dumps(metadata)
        with path.open("rb") as fp:
            files = {"file": (path.name, fp)}
            data = {"metadata": metadata_json}
            return self._request("POST", "/documents", files=files, data=data, enforce_tenant=True)

    def list_documents(self, limit: int = 20) -> Dict[str, Any]:
        return self._request("GET", "/documents", params={"limit": limit}, enforce_tenant=True)

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}/status", enforce_tenant=True)

    def delete_document(self, document_id: str, purge_chunks: bool = True) -> Dict[str, Any]:
        return self._request(
            "DELETE",
            f"/documents/{document_id}",
            params={"purge_chunks": str(purge_chunks).lower()},
            enforce_tenant=True,
        )

    def create_chat_completion(
        self,
        message: str,
        tenant_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_chunks: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "message": message,
            "history": history or [],
            "max_context_chunks": max_context_chunks,
        }
        return self._request(
            "POST",
            "/chat/completions",
            json_body=payload,
            tenant_id=tenant_id,
            sync_tenant_body=True,
            enforce_tenant=True,
        )

    def submit_chat_feedback(self, interaction_id: str, rating: str, comment: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"interaction_id": interaction_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        return self._request("POST", "/chat/feedback", json_body=payload, enforce_tenant=True)

    def list_tenant_collections(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/management/collections",
            params={},
            tenant_id=tenant_id,
            sync_tenant_params=True,
            enforce_tenant=True,
        )

    def get_tenant_queue_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return self._request(
            "GET",
            "/management/queue/status",
            params={},
            tenant_id=tenant_id,
            sync_tenant_params=True,
            enforce_tenant=True,
        )

    def get_management_health(self) -> Dict[str, Any]:
        return self._request("GET", "/management/health", enforce_tenant=False)

    def _resolve_tenant(self, tenant_id: Optional[str], *, endpoint: str, enforce_tenant: bool) -> Optional[str]:
        explicit_tenant = _sanitize_tenant_id(tenant_id)
        context_tenant = self.tenant_context.get_tenant()
        if explicit_tenant and context_tenant and explicit_tenant != context_tenant:
            logger.warning(
                "tenant_mismatch_detected",
                extra={
                    "event": "tenant_mismatch_detected",
                    "endpoint": endpoint,
                    "tenant_hash": _tenant_fingerprint(context_tenant),
                    "status": "blocked_local_conflict",
                },
            )
            raise TenantMismatchLocalError()
        resolved = explicit_tenant or context_tenant
        if enforce_tenant and not resolved:
            logger.warning(
                "tenant_missing_blocked",
                extra={"event": "tenant_missing_blocked", "endpoint": endpoint, "status": "blocked"},
            )
            raise TenantSelectionRequiredError()
        return resolved

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        sync_tenant_body: bool = False,
        sync_tenant_params: bool = False,
        enforce_tenant: bool = True,
        _retry_on_mismatch: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{path}"
        resolved_tenant = self._resolve_tenant(tenant_id, endpoint=path, enforce_tenant=enforce_tenant)
        request_params = dict(params or {})
        request_json = dict(json_body or {})

        if sync_tenant_body and resolved_tenant:
            request_json["tenant_id"] = resolved_tenant
        if sync_tenant_params and resolved_tenant:
            request_params["tenant_id"] = resolved_tenant

        headers = _build_auth_headers(self.api_key, self.default_headers)
        if resolved_tenant:
            headers["X-Tenant-ID"] = resolved_tenant
        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=request_params or None,
            json=request_json or None,
            data=data,
            files=files,
            timeout=self.timeout_seconds,
        )

        if response.ok:
            return response.json()

        try:
            payload = response.json()
        except Exception:
            payload = None

        payload_error = payload.get("error") if isinstance(payload, dict) else None
        payload_code = str(payload_error.get("code") or "") if isinstance(payload_error, dict) else ""
        payload_request_id = (
            str(payload_error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown")
            if isinstance(payload_error, dict)
            else str(response.headers.get("X-Correlation-ID") or "unknown")
        )
        if (
            _retry_on_mismatch
            and response.status_code == 400
            and payload_code == TENANT_MISMATCH_CODE
            and files is None
            and self.tenant_context.storage_path is not None
        ):
            logger.warning(
                "tenant_mismatch_detected",
                extra={
                    "event": "tenant_mismatch_detected",
                    "endpoint": path,
                    "request_id": payload_request_id,
                    "status": response.status_code,
                    "tenant_hash": _tenant_fingerprint(resolved_tenant),
                },
            )
            previous = resolved_tenant
            reloaded = self.tenant_context.reload()
            if reloaded and reloaded != previous:
                return self._request(
                    method,
                    path,
                    params=request_params,
                    json_body=request_json,
                    data=data,
                    files=files,
                    tenant_id=reloaded,
                    sync_tenant_body=sync_tenant_body,
                    sync_tenant_params=sync_tenant_params,
                    enforce_tenant=enforce_tenant,
                    _retry_on_mismatch=False,
                )
        _raise_from_http_error(response.status_code, response.text, dict(response.headers), payload)


class AsyncCireRagClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
        tenant_context: Optional[TenantContext] = None,
        tenant_storage_path: Optional[str | Path] = None,
    ):
        if tenant_context is not None and tenant_storage_path is not None:
            raise ValueError("Provide either tenant_context or tenant_storage_path, not both")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self._managed_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self.tenant_context = tenant_context or TenantContext(storage_path=Path(tenant_storage_path) if tenant_storage_path else None)

    async def __aenter__(self) -> "AsyncCireRagClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._managed_client:
            await self.client.aclose()

    def set_tenant(self, tenant_id: str) -> None:
        self.tenant_context.set_tenant(tenant_id)

    def get_tenant(self) -> Optional[str]:
        return self.tenant_context.get_tenant()

    def clear_tenant(self) -> None:
        self.tenant_context.clear_tenant()

    async def create_document(self, file_path: str | Path, metadata: Dict[str, Any] | str) -> Dict[str, Any]:
        path = Path(file_path)
        metadata_json = metadata if isinstance(metadata, str) else json.dumps(metadata)
        file_bytes = path.read_bytes()
        files = {"file": (path.name, file_bytes)}
        data = {"metadata": metadata_json}
        return await self._request("POST", "/documents", files=files, data=data, enforce_tenant=True)

    async def list_documents(self, limit: int = 20) -> Dict[str, Any]:
        return await self._request("GET", "/documents", params={"limit": limit}, enforce_tenant=True)

    async def get_document_status(self, document_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/documents/{document_id}/status", enforce_tenant=True)

    async def delete_document(self, document_id: str, purge_chunks: bool = True) -> Dict[str, Any]:
        return await self._request(
            "DELETE",
            f"/documents/{document_id}",
            params={"purge_chunks": str(purge_chunks).lower()},
            enforce_tenant=True,
        )

    async def create_chat_completion(
        self,
        message: str,
        tenant_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_chunks: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "message": message,
            "history": history or [],
            "max_context_chunks": max_context_chunks,
        }
        return await self._request(
            "POST",
            "/chat/completions",
            json_body=payload,
            tenant_id=tenant_id,
            sync_tenant_body=True,
            enforce_tenant=True,
        )

    async def submit_chat_feedback(self, interaction_id: str, rating: str, comment: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"interaction_id": interaction_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        return await self._request("POST", "/chat/feedback", json_body=payload, enforce_tenant=True)

    async def list_tenant_collections(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/management/collections",
            params={},
            tenant_id=tenant_id,
            sync_tenant_params=True,
            enforce_tenant=True,
        )

    async def get_tenant_queue_status(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        return await self._request(
            "GET",
            "/management/queue/status",
            params={},
            tenant_id=tenant_id,
            sync_tenant_params=True,
            enforce_tenant=True,
        )

    async def get_management_health(self) -> Dict[str, Any]:
        return await self._request("GET", "/management/health", enforce_tenant=False)

    def _resolve_tenant(self, tenant_id: Optional[str], *, endpoint: str, enforce_tenant: bool) -> Optional[str]:
        explicit_tenant = _sanitize_tenant_id(tenant_id)
        context_tenant = self.tenant_context.get_tenant()
        if explicit_tenant and context_tenant and explicit_tenant != context_tenant:
            logger.warning(
                "tenant_mismatch_detected",
                extra={
                    "event": "tenant_mismatch_detected",
                    "endpoint": endpoint,
                    "tenant_hash": _tenant_fingerprint(context_tenant),
                    "status": "blocked_local_conflict",
                },
            )
            raise TenantMismatchLocalError()
        resolved = explicit_tenant or context_tenant
        if enforce_tenant and not resolved:
            logger.warning(
                "tenant_missing_blocked",
                extra={"event": "tenant_missing_blocked", "endpoint": endpoint, "status": "blocked"},
            )
            raise TenantSelectionRequiredError()
        return resolved

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
        sync_tenant_body: bool = False,
        sync_tenant_params: bool = False,
        enforce_tenant: bool = True,
        _retry_on_mismatch: bool = True,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{path}"
        resolved_tenant = self._resolve_tenant(tenant_id, endpoint=path, enforce_tenant=enforce_tenant)
        request_params = dict(params or {})
        request_json = dict(json_body or {})
        if sync_tenant_body and resolved_tenant:
            request_json["tenant_id"] = resolved_tenant
        if sync_tenant_params and resolved_tenant:
            request_params["tenant_id"] = resolved_tenant

        headers = _build_auth_headers(self.api_key, self.default_headers)
        if resolved_tenant:
            headers["X-Tenant-ID"] = resolved_tenant
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            params=request_params or None,
            json=request_json or None,
            data=data,
            files=files,
        )

        if response.is_success:
            return response.json()

        try:
            payload = response.json()
        except Exception:
            payload = None

        payload_error = payload.get("error") if isinstance(payload, dict) else None
        payload_code = str(payload_error.get("code") or "") if isinstance(payload_error, dict) else ""
        payload_request_id = (
            str(payload_error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown")
            if isinstance(payload_error, dict)
            else str(response.headers.get("X-Correlation-ID") or "unknown")
        )

        if (
            _retry_on_mismatch
            and response.status_code == 400
            and payload_code == TENANT_MISMATCH_CODE
            and files is None
            and self.tenant_context.storage_path is not None
        ):
            logger.warning(
                "tenant_mismatch_detected",
                extra={
                    "event": "tenant_mismatch_detected",
                    "endpoint": path,
                    "request_id": payload_request_id,
                    "status": response.status_code,
                    "tenant_hash": _tenant_fingerprint(resolved_tenant),
                },
            )
            previous = resolved_tenant
            reloaded = self.tenant_context.reload()
            if reloaded and reloaded != previous:
                return await self._request(
                    method,
                    path,
                    params=request_params,
                    json_body=request_json,
                    data=data,
                    files=files,
                    tenant_id=reloaded,
                    sync_tenant_body=sync_tenant_body,
                    sync_tenant_params=sync_tenant_params,
                    enforce_tenant=enforce_tenant,
                    _retry_on_mismatch=False,
                )
        _raise_from_http_error(response.status_code, response.text, dict(response.headers), payload)
