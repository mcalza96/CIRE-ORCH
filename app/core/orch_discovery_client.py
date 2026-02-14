from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class OrchestratorDiscoveryError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class Tenant:
    id: str
    name: str


@dataclass(frozen=True)
class Collection:
    id: str
    name: str
    collection_key: str | None = None


def _headers(token: str) -> dict[str, str]:
    value = str(token or "").strip()
    if not value:
        return {}
    return {"Authorization": f"Bearer {value}"}


def _raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    message = response.text
    raise OrchestratorDiscoveryError(
        f"Discovery request failed (HTTP {response.status_code}): {message}",
        status_code=response.status_code,
    )


async def list_authorized_tenants(base_url: str, token: str, timeout_seconds: float = 4.0) -> list[Tenant]:
    url = f"{base_url.rstrip('/')}/api/v1/knowledge/tenants"
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=2.5)) as client:
        response = await client.get(url, headers=_headers(token))
    _raise_for_status(response)

    payload: Any = response.json()
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []

    out: list[Tenant] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        tenant_id = str(item.get("id") or "").strip()
        if not tenant_id:
            continue
        name = str(item.get("name") or tenant_id).strip() or tenant_id
        out.append(Tenant(id=tenant_id, name=name))
    return out


async def list_authorized_collections(
    base_url: str,
    token: str,
    tenant_id: str,
    timeout_seconds: float = 4.0,
) -> list[Collection]:
    tenant = str(tenant_id or "").strip()
    if not tenant:
        return []

    url = f"{base_url.rstrip('/')}/api/v1/knowledge/collections"
    params = {"tenant_id": tenant}
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=2.5)) as client:
        response = await client.get(url, params=params, headers=_headers(token))
    _raise_for_status(response)

    payload: Any = response.json()
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return []

    out: list[Collection] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        collection_id = str(item.get("id") or "").strip()
        if not collection_id:
            continue
        collection_key = str(item.get("collection_key") or "").strip() or None
        name = str(item.get("name") or collection_key or collection_id).strip() or collection_id
        out.append(Collection(id=collection_id, name=name, collection_key=collection_key))
    return out
