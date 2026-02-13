from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import requests


@dataclass
class CireRagApiError(Exception):
    status: int
    code: str
    message: str
    details: Any
    request_id: str

    def __str__(self) -> str:
        return f"[{self.status}] {self.code}: {self.message} (request_id={self.request_id})"


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

    raise CireRagApiError(
        status=status_code,
        code=str(error.get("code") or "UNKNOWN_ERROR"),
        message=str(error.get("message") or "Request failed"),
        details=error.get("details"),
        request_id=str(error.get("request_id") or response_headers.get("X-Correlation-ID") or "unknown"),
    )


class CireRagClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self.session = session or requests.Session()

    def close(self) -> None:
        self.session.close()

    def create_document(self, file_path: str | Path, metadata: Dict[str, Any] | str) -> Dict[str, Any]:
        path = Path(file_path)
        metadata_json = metadata if isinstance(metadata, str) else json.dumps(metadata)
        with path.open("rb") as fp:
            files = {"file": (path.name, fp)}
            data = {"metadata": metadata_json}
            return self._request("POST", "/documents", files=files, data=data)

    def list_documents(self, limit: int = 20) -> Dict[str, Any]:
        return self._request("GET", "/documents", params={"limit": limit})

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/documents/{document_id}/status")

    def delete_document(self, document_id: str, purge_chunks: bool = True) -> Dict[str, Any]:
        return self._request("DELETE", f"/documents/{document_id}", params={"purge_chunks": str(purge_chunks).lower()})

    def create_chat_completion(
        self,
        message: str,
        tenant_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_chunks: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "message": message,
            "tenant_id": tenant_id,
            "history": history or [],
            "max_context_chunks": max_context_chunks,
        }
        return self._request("POST", "/chat/completions", json_body=payload)

    def submit_chat_feedback(self, interaction_id: str, rating: str, comment: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"interaction_id": interaction_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        return self._request("POST", "/chat/feedback", json_body=payload)

    def list_tenant_collections(self, tenant_id: str) -> Dict[str, Any]:
        return self._request("GET", "/management/collections", params={"tenant_id": tenant_id})

    def get_tenant_queue_status(self, tenant_id: str) -> Dict[str, Any]:
        return self._request("GET", "/management/queue/status", params={"tenant_id": tenant_id})

    def get_management_health(self) -> Dict[str, Any]:
        return self._request("GET", "/management/health")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{path}"
        headers = _build_auth_headers(self.api_key, self.default_headers)
        response = self.session.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
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
        _raise_from_http_error(response.status_code, response.text, dict(response.headers), payload)


class AsyncCireRagClient:
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        default_headers: Optional[Dict[str, str]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers or {}
        self._managed_client = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def __aenter__(self) -> "AsyncCireRagClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._managed_client:
            await self.client.aclose()

    async def create_document(self, file_path: str | Path, metadata: Dict[str, Any] | str) -> Dict[str, Any]:
        path = Path(file_path)
        metadata_json = metadata if isinstance(metadata, str) else json.dumps(metadata)
        file_bytes = path.read_bytes()
        files = {"file": (path.name, file_bytes)}
        data = {"metadata": metadata_json}
        return await self._request("POST", "/documents", files=files, data=data)

    async def list_documents(self, limit: int = 20) -> Dict[str, Any]:
        return await self._request("GET", "/documents", params={"limit": limit})

    async def get_document_status(self, document_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/documents/{document_id}/status")

    async def delete_document(self, document_id: str, purge_chunks: bool = True) -> Dict[str, Any]:
        return await self._request("DELETE", f"/documents/{document_id}", params={"purge_chunks": str(purge_chunks).lower()})

    async def create_chat_completion(
        self,
        message: str,
        tenant_id: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_context_chunks: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "message": message,
            "tenant_id": tenant_id,
            "history": history or [],
            "max_context_chunks": max_context_chunks,
        }
        return await self._request("POST", "/chat/completions", json_body=payload)

    async def submit_chat_feedback(self, interaction_id: str, rating: str, comment: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"interaction_id": interaction_id, "rating": rating}
        if comment:
            payload["comment"] = comment
        return await self._request("POST", "/chat/feedback", json_body=payload)

    async def list_tenant_collections(self, tenant_id: str) -> Dict[str, Any]:
        return await self._request("GET", "/management/collections", params={"tenant_id": tenant_id})

    async def get_tenant_queue_status(self, tenant_id: str) -> Dict[str, Any]:
        return await self._request("GET", "/management/queue/status", params={"tenant_id": tenant_id})

    async def get_management_health(self) -> Dict[str, Any]:
        return await self._request("GET", "/management/health")

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1{path}"
        headers = _build_auth_headers(self.api_key, self.default_headers)
        response = await self.client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json_body,
            data=data,
            files=files,
        )

        if response.is_success:
            return response.json()

        try:
            payload = response.json()
        except Exception:
            payload = None
        _raise_from_http_error(response.status_code, response.text, dict(response.headers), payload)
