from __future__ import annotations

import argparse
import httpx
import time
import logging
from typing import Any, Protocol

from app.infrastructure.clients.auth_client import decode_jwt_payload
from app.infrastructure.clients.discovery_client import list_authorized_collections, list_authorized_tenants
from sdk.python.cire_rag_sdk import TenantContext

logger = logging.getLogger(__name__)

class AnswerPoster(Protocol):
    async def __call__(
        self,
        *,
        client: httpx.AsyncClient,
        orchestrator_url: str,
        tenant_context: TenantContext,
        query: str,
        collection_id: str | None,
        agent_profile_id: str | None,
        access_token: str | None,
    ) -> dict[str, Any]:
        ...

def _short_token(token: str) -> str:
    if not token or len(token) < 12:
        return "invalid"
    return f"{token[:6]}...{token[-4:]}"

async def run_doctor(
    *,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tenant_context: TenantContext,
    access_token: str,
    collection_id: str | None,
    post_answer_fn: AnswerPoster,
) -> None:
    print("🩺 ORCH Doctor")
    print("=" * 60)

    payload = decode_jwt_payload(access_token) if access_token else None
    now = int(time.time())
    exp = None
    if isinstance(payload, dict) and payload.get("exp"):
        try:
            exp = int(payload.get("exp"))
        except Exception:
            exp = None
    sub = str(payload.get("sub") or "").strip() if isinstance(payload, dict) else ""
    print(f"auth_ok: {'yes' if bool(access_token) else 'no'}")
    print(f"token_fingerprint: {_short_token(access_token) if access_token else 'none'}")
    if sub:
        print(f"user_sub: {sub}")
    if exp:
        print(f"token_exp_unix: {exp}")
        print(f"token_expired: {'yes' if exp <= now else 'no'}")

    tenant_count = 0
    collection_count = 0
    try:
        tenants = await list_authorized_tenants(args.orchestrator_url, access_token)
        tenant_count = len(tenants)
        print(f"tenant_discovery_ok: yes")
    except Exception as exc:
        print(f"tenant_discovery_ok: no ({exc})")

    print(f"tenant_count: {tenant_count}")
    tenant_id = tenant_context.get_tenant()
    if tenant_id:
        print(f"selected_tenant: {tenant_id}")
    else:
        print("selected_tenant: none")

    if tenant_id:
        try:
            collections = await list_authorized_collections(
                args.orchestrator_url, access_token, tenant_id
            )
            collection_count = len(collections)
            print("collection_discovery_ok: yes")
        except Exception as exc:
            print(f"collection_discovery_ok: no ({exc})")
    else:
        print("collection_discovery_ok: skipped")
    print(f"collection_count: {collection_count}")

    if not tenant_id:
        print("retrieval_probe: skipped (tenant not resolved)")
        print("=" * 60)
        return

    try:
        result = await post_answer_fn(
            client=client,
            orchestrator_url=args.orchestrator_url,
            tenant_context=tenant_context,
            query=args.doctor_query,
            collection_id=collection_id,
            agent_profile_id=None,
            access_token=access_token,
        )
        mode = str(result.get("mode") or "unknown")
        context_chunks = (
            result.get("context_chunks") if isinstance(result.get("context_chunks"), list) else []
        )
        citations = result.get("citations") if isinstance(result.get("citations"), list) else []
        validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
        print("retrieval_probe_ok: yes")
        print(f"mode: {mode}")
        print(f"context_chunks_count: {len(context_chunks)}")
        print(f"citations_count: {len(citations)}")
        print(f"validation_accepted: {bool(validation.get('accepted', True))}")
    except Exception as exc:
        print(f"retrieval_probe_ok: no ({exc})")

    print("=" * 60)
