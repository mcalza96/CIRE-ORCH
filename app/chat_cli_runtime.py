"""HTTP-based chat CLI for split orchestrator architecture."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx

from app.core import cli_api, cli_args, cli_utils, diagnostics, discovery_utils
from app.core.auth_client import AuthClientError, ensure_access_token
from app.core.orch_discovery_client import OrchestratorDiscoveryError
from app.core.ui.chat_commands import dispatch_chat_command
from app.core.ui.chat_query_flow import handle_chat_query
from app.core.ui.chat_runtime_models import (
    ChatRuntimeContext,
    ChatSessionState,
    print_chat_banner,
)
from sdk.python.cire_rag_sdk import TenantContext


async def main(argv: list[str] | None = None) -> None:
    args = cli_args.parse_chat_args(argv)

    _require_orchestrator_health(args.orchestrator_url)
    access_token = await _resolve_access_token(args)
    tenant_context = _build_tenant_context(args)

    try:
        (
            tenant_id,
            collection_id,
            collection_name,
            agent_profile_id,
            access_token,
        ) = await _resolve_runtime_inputs(
            args=args,
            tenant_context=tenant_context,
            access_token=access_token,
        )
    except Exception as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)

    await _run_chat_session(
        args=args,
        tenant_context=tenant_context,
        tenant_id=tenant_id,
        collection_id=collection_id,
        collection_name=collection_name,
        access_token=access_token,
        agent_profile_id=agent_profile_id,
    )


def _require_orchestrator_health(orchestrator_url: str) -> None:
    try:
        cli_utils.require_orch_health(orchestrator_url)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)


async def _resolve_access_token(args: Any) -> str:
    access_token = str(args.access_token or "").strip()
    if access_token:
        return access_token
    try:
        return await ensure_access_token(interactive=not args.non_interactive)
    except AuthClientError as exc:
        if args.non_interactive:
            print(f"❌ {exc}")
            raise SystemExit(1)
        print(f"⚠️ No se pudo resolver token automáticamente: {exc}")
        print("⚠️ Continuando sin token (útil solo si ORCH_AUTH_REQUIRED=false).")
        return ""


def _build_tenant_context(args: Any) -> TenantContext:
    return TenantContext(
        tenant_id=args.tenant_id,
        storage_path=Path(args.tenant_storage_path) if args.tenant_storage_path else None,
    )


async def _resolve_runtime_inputs(
    *,
    args: Any,
    tenant_context: TenantContext,
    access_token: str,
) -> tuple[str, str | None, str | None, str | None, str]:
    try:
        tenant_id = await discovery_utils.resolve_tenant(
            args=args,
            tenant_context=tenant_context,
            access_token=access_token,
        )
        collection_id, collection_name = await discovery_utils.resolve_collection(
            args=args,
            tenant_id=tenant_id,
            access_token=access_token,
        )
        agent_profile_id = await discovery_utils.resolve_agent_profile(
            args=args,
            access_token=access_token,
        )
        return tenant_id, collection_id, collection_name, agent_profile_id, access_token
    except OrchestratorDiscoveryError as exc:
        if exc.status_code != 401:
            raise

    refreshed = await ensure_access_token(interactive=not args.non_interactive)
    tenant_id = await discovery_utils.resolve_tenant(
        args=args,
        tenant_context=tenant_context,
        access_token=refreshed,
    )
    collection_id, collection_name = await discovery_utils.resolve_collection(
        args=args,
        tenant_id=tenant_id,
        access_token=refreshed,
    )
    agent_profile_id = await discovery_utils.resolve_agent_profile(
        args=args,
        access_token=refreshed,
    )
    return tenant_id, collection_id, collection_name, agent_profile_id, refreshed


async def _run_chat_session(
    *,
    args: Any,
    tenant_context: TenantContext,
    tenant_id: str,
    collection_id: str | None,
    collection_name: str | None,
    access_token: str,
    agent_profile_id: str | None,
) -> None:
    timeout = httpx.Timeout(connect=5.0, read=float(args.timeout_seconds), write=20.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.doctor:
            await diagnostics.run_doctor(
                client=client,
                args=args,
                tenant_context=tenant_context,
                access_token=access_token,
                collection_id=collection_id,
                post_answer_fn=cli_api.post_answer,
            )
            return

        runtime = ChatRuntimeContext(
            args=args,
            client=client,
            tenant_context=tenant_context,
            tenant_id=tenant_id,
            collection_id=collection_id,
            access_token=access_token,
        )
        state = ChatSessionState(
            last_result={},
            last_query="",
            forced_mode=None,
            agent_profile_id=agent_profile_id,
        )
        print_chat_banner(runtime=runtime, collection_name=collection_name, state=state)
        await _run_chat_repl(runtime=runtime, state=state)


async def _run_chat_repl(*, runtime: ChatRuntimeContext, state: ChatSessionState) -> None:
    while True:
        query = cli_utils.prompt("❓ > ")
        normalized = query.lower().strip()
        if normalized in {"salir", "exit", "quit"}:
            print("Okey, ¡adiós! 👋")
            return
        if not query:
            continue
        if await dispatch_chat_command(query=query, runtime=runtime, state=state):
            continue
        await handle_chat_query(query=query, runtime=runtime, state=state)


if __name__ == "__main__":
    asyncio.run(main())
