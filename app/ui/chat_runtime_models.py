from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from sdk.python.cire_rag_sdk import TenantContext


@dataclass
class ChatSessionState:
    last_result: dict[str, Any]
    last_query: str
    forced_mode: str | None
    agent_profile_id: str | None


@dataclass
class ChatRuntimeContext:
    args: Any
    client: httpx.AsyncClient
    tenant_context: TenantContext
    tenant_id: str
    collection_id: str | None
    access_token: str


def print_thinking_status(status: dict[str, Any], seen: set[str]) -> None:
    status_type = str(status.get("type") or "").strip().lower()
    phase = str(status.get("phase") or "").strip().lower()
    label = str(status.get("label") or "").strip()
    elapsed_ms = status.get("elapsed_ms")
    elapsed_s = (
        round(float(elapsed_ms) / 1000.0, 1) if isinstance(elapsed_ms, (int, float)) else None
    )

    if status_type == "thinking" and phase:
        if phase in seen:
            return
        seen.add(phase)
        suffix = f" [{elapsed_s}s]" if elapsed_s is not None else ""
        print(f"⏳ {label or phase}{suffix}")
        return

    if status_type == "working":
        pulse = status.get("pulse")
        if isinstance(pulse, int) and pulse % 5 == 0:
            suffix = f" {elapsed_s}s" if elapsed_s is not None else ""
            print(f"⏳ Procesando...{suffix}")


def print_chat_banner(
    *,
    runtime: ChatRuntimeContext,
    collection_name: str | None,
    state: ChatSessionState,
) -> None:
    scope = (
        collection_name or runtime.collection_id or runtime.args.collection_name or "todo el tenant"
    )
    print("🚀 Chat Q/A Orchestrator (split mode HTTP)")
    print(f"🏢 Tenant: {runtime.tenant_context.get_tenant() or '(sin seleccionar)'}")
    print(f"📁 Scope: {scope}")
    print(f"🌐 Orchestrator URL: {runtime.args.orchestrator_url}")
    print(f"🔐 Auth: {'Bearer token' if runtime.access_token else 'sin token'}")
    print(f"🧩 Perfil: {state.agent_profile_id or 'automatico por tenant'}")
    print("💡 Escribe tu pregunta (o 'salir')")
    print(
        "🔭 Comandos: /ingestion , /watch <batch_id> , /trace , /snapshot , /citations , /explain , /profile , /mode"
    )
