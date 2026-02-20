from __future__ import annotations

from app.core import cli_api
from app.core.observability.ingestion_utils import show_ingestion_overview, watch_batch_stream
from app.core.observability.logging_utils import compact_error
from app.core.ui import renderers

from app.core.ui.chat_runtime_models import ChatRuntimeContext, ChatSessionState


async def dispatch_chat_command(
    *,
    query: str,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
) -> bool:
    normalized = query.lower().strip()

    exact_handlers = {
        "/ingestion": _handle_cmd_ingestion,
        "/trace": _handle_cmd_trace,
        "/citations": _handle_cmd_citations,
        "/profile": _handle_cmd_profile,
        "/explain": _handle_cmd_explain,
    }
    if normalized in exact_handlers:
        await exact_handlers[normalized](query=query, runtime=runtime, state=state)
        return True

    prefix_handlers = {
        "/watch": _handle_cmd_watch,
        "/cartridge": _handle_cmd_cartridge,
        "/mode": _handle_cmd_mode,
    }
    for prefix, handler in prefix_handlers.items():
        if normalized == prefix or normalized.startswith(prefix + " "):
            await handler(query=query, runtime=runtime, state=state)
            return True
    return False


async def _handle_cmd_ingestion(
    *,
    query: str,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
) -> None:
    _ = query, state
    await show_ingestion_overview(
        client=runtime.client,
        orchestrator_url=runtime.args.orchestrator_url,
        tenant_id=runtime.tenant_id,
        access_token=runtime.access_token,
    )


async def _handle_cmd_watch(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    _ = state
    raw = query.split(" ", 1)
    batch_id = raw[1].strip() if len(raw) > 1 else ""
    if not batch_id:
        print("❌ Uso: /watch <batch_id>")
        return
    await watch_batch_stream(
        client=runtime.client,
        orchestrator_url=runtime.args.orchestrator_url,
        tenant_id=runtime.tenant_id,
        access_token=runtime.access_token,
        batch_id=batch_id,
    )


async def _handle_cmd_trace(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    _ = query, runtime
    if state.last_result:
        renderers.print_trace(state.last_result)
        return
    print("ℹ️ No hay un resultado previo para mostrar trace.")


async def _handle_cmd_citations(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    _ = query, runtime
    if state.last_result:
        renderers.print_citations_only(state.last_result)
        return
    print("ℹ️ No hay un resultado previo para mostrar citas.")


async def _handle_cmd_profile(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    _ = query, runtime
    if state.last_result:
        renderers.print_profile_snapshot(state.last_result, forced_mode=state.forced_mode)
        return
    print("ℹ️ Aun no hay respuesta previa. Haz una consulta primero.")
    if state.forced_mode:
        print(f"   forced_mode={state.forced_mode}")


async def _handle_cmd_cartridge(
    *,
    query: str,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
) -> None:
    _ = runtime
    raw = query.split(" ", 1)
    arg = raw[1].strip() if len(raw) > 1 else ""
    if not arg:
        print(f"ℹ️ cartridge={state.agent_profile_id or '(automatico por tenant)'}")
        print("   Uso: /cartridge <profile_id> | /cartridge clear")
        return
    if arg.lower() in {"clear", "off", "none", "auto"}:
        state.agent_profile_id = None
        print("✅ cartridge en modo automatico por tenant")
        return
    state.agent_profile_id = arg.lower()
    print(f"✅ cartridge={state.agent_profile_id}")


async def _handle_cmd_mode(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    _ = runtime
    raw = query.split(" ", 1)
    arg = raw[1].strip() if len(raw) > 1 else ""
    if not arg:
        print(f"ℹ️ forced_mode={state.forced_mode or '(none)'}")
        print("   Uso: /mode <mode_name> | /mode clear")
        return
    if arg.lower() in {"clear", "off", "none"}:
        state.forced_mode = None
        print("✅ forced_mode desactivado")
        return
    state.forced_mode = arg.lower()
    print(f"✅ forced_mode={state.forced_mode}")


async def _handle_cmd_explain(
    *,
    query: str,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
) -> None:
    _ = query
    if not state.last_query:
        print("ℹ️ No hay una consulta previa para explicar.")
        return
    try:
        payload = await cli_api.post_explain(
            client=runtime.client,
            orchestrator_url=runtime.args.orchestrator_url,
            tenant_context=runtime.tenant_context,
            query=state.last_query,
            collection_id=runtime.collection_id,
            agent_profile_id=state.agent_profile_id,
            access_token=runtime.access_token,
        )
        renderers.print_explain(payload)
    except Exception as exc:
        print(f"❌ explain failed: {compact_error(exc)}")
