from __future__ import annotations

import argparse
import httpx
import logging
from typing import Any

from app.infrastructure.clients.discovery_client import (
    OrchestratorDiscoveryError,
    list_authorized_collections,
    list_authorized_tenants,
)
from sdk.python.cire_rag_sdk import (
    TenantContext,
    TenantSelectionRequiredError,
)

logger = logging.getLogger(__name__)

def _prompt(message: str) -> str:
    return input(message).strip()

async def resolve_tenant(
    *,
    args: argparse.Namespace,
    tenant_context: TenantContext,
    access_token: str,
) -> str:
    if args.tenant_id:
        tenant_context.set_tenant(args.tenant_id)
        return args.tenant_id

    try:
        tenants = await list_authorized_tenants(args.orchestrator_url, access_token)
    except OrchestratorDiscoveryError as exc:
        if exc.status_code == 401:
            raise
        if args.non_interactive:
            raise RuntimeError(f"Tenant discovery failed: {exc}") from exc
        print(f"⚠️ No se pudieron cargar tenants autorizados ({exc}).")
        manual = _prompt("🏢 Tenant ID (manual): ")
        if not manual:
            raise TenantSelectionRequiredError()
        tenant_context.set_tenant(manual)
        return manual

    if not tenants:
        if args.non_interactive:
            raise RuntimeError("No authorized tenants found for current user")
        print("⚠️ No hay tenants autorizados en ORCH.")
        manual = _prompt("🏢 Tenant ID (manual): ")
        if not manual:
            raise TenantSelectionRequiredError()
        tenant_context.set_tenant(manual)
        return manual

    if len(tenants) == 1:
        tenant_context.set_tenant(tenants[0].id)
        print(f"🏢 Tenant auto-seleccionado: {tenants[0].name} ({tenants[0].id})")
        return tenants[0].id

    if args.non_interactive:
        raise RuntimeError("Multiple tenants available; pass --tenant-id in non-interactive mode")

    print("🏢 Tenants disponibles:")
    for idx, tenant in enumerate(tenants, start=1):
        print(f"  {idx}) {tenant.name} ({tenant.id})")
    print("  0) Ingresar manual")

    option = _prompt(f"📝 Selecciona Tenant [1-{len(tenants)}]: ")
    if option.isdigit():
        selected = int(option)
        if selected == 0:
            manual = _prompt("🏢 Tenant ID: ")
            if manual:
                tenant_context.set_tenant(manual)
                return manual
        if 1 <= selected <= len(tenants):
            tenant = tenants[selected - 1]
            tenant_context.set_tenant(tenant.id)
            return tenant.id

    manual = _prompt("🏢 Tenant ID: ")
    if not manual:
        raise TenantSelectionRequiredError()
    tenant_context.set_tenant(manual)
    return manual

async def resolve_collection(
    *,
    args: argparse.Namespace,
    tenant_id: str,
    access_token: str,
) -> tuple[str | None, str | None]:
    if args.collection_id:
        return args.collection_id, args.collection_name

    try:
        collections = await list_authorized_collections(
            args.orchestrator_url, access_token, tenant_id
        )
    except OrchestratorDiscoveryError as exc:
        if exc.status_code == 401:
            raise
        print(f"⚠️ No se pudieron cargar colecciones ({exc}).")
        return None, args.collection_name

    if not collections:
        return None, args.collection_name

    if args.non_interactive:
        return None, args.collection_name

    print("📁 Colecciones:")
    print("  0) Todas / Default")
    for idx, item in enumerate(collections, start=1):
        suffix = f" | key={item.collection_key}" if item.collection_key else ""
        print(f"  {idx}) {item.name}{suffix} | id={item.id}")

    option = _prompt(f"📝 Selecciona Colección [0-{len(collections)}]: ")
    if option.isdigit():
        selected = int(option)
        if 1 <= selected <= len(collections):
            col = collections[selected - 1]
            return col.id, col.name

    return None, args.collection_name

async def list_agent_profile_ids(orchestrator_url: str, access_token: str) -> list[str]:
    headers: dict[str, str] = {}
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            orchestrator_url.rstrip("/") + "/api/v1/knowledge/agent-profiles",
            headers=headers,
        )
        response.raise_for_status()
        payload = response.json()
    items = payload.get("items") if isinstance(payload, dict) else []
    out: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if pid:
                out.append(pid)
    return out

async def resolve_agent_profile(args: argparse.Namespace, access_token: str) -> str | None:
    explicit = str(getattr(args, "agent_profile", "") or "").strip()
    if explicit:
        return explicit
    if args.non_interactive:
        return None
    try:
        profile_ids = await list_agent_profile_ids(args.orchestrator_url, access_token)
    except Exception:
        return None
    if not profile_ids:
        return None
    print("🧩 Cartucho para esta sesion")
    print("  0) Automatico por tenant")
    for idx, pid in enumerate(profile_ids, start=1):
        print(f"  {idx}) {pid}")
    option = _prompt(f"📝 Selecciona cartucho [0-{len(profile_ids)}]: ")
    if option.isdigit():
        selected = int(option)
        if selected == 0:
            return None
        if 1 <= selected <= len(profile_ids):
            return profile_ids[selected - 1]
    return None
