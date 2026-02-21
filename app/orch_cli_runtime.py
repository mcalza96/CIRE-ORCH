from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

from app import chat_cli_runtime, ing_cli_runtime
from app.infrastructure.clients.auth_client import AuthClientError, decode_jwt_exp, ensure_access_token
from app.infrastructure.clients.discovery_client import OrchestratorDiscoveryError, Tenant, list_authorized_tenants

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecContext:
    script: str
    argv: list[str]
    env: dict[str, str]


ExecFn = Callable[[str, list[str], dict[str, str]], None]


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_enabled(env: dict[str, str]) -> bool:
    return _is_truthy(env.get("ORCH_CLI_DEBUG"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _print_help() -> None:
    print("Uso:")
    print("  python orch_cli.py <comando> [args]")
    print("")
    print("Comandos:")
    print("  chat      Inicia chat interactivo contra ORCH")
    print("  ingest    Ejecuta cliente de ingesta portable")
    print("")
    print("Ejemplos:")
    print("  python orch_cli.py chat --doctor")
    print("  python orch_cli.py ingest --help")


def _resolve_orchestrator_url(env: dict[str, str]) -> str:
    return str(env.get("ORCH_URL") or env.get("ORCHESTRATOR_URL") or "http://localhost:8001").rstrip("/")


def _resolve_rag_url(env: dict[str, str]) -> str:
    forced = str(env.get("RAG_ENGINE_FORCE_BACKEND") or "").strip().lower()
    local_url = str(env.get("RAG_ENGINE_LOCAL_URL") or env.get("RAG_ENGINE_URL") or "http://localhost:8000")
    docker_url = str(env.get("RAG_ENGINE_DOCKER_URL") or "http://localhost:8000")

    if forced == "local":
        return local_url
    if forced == "docker":
        return docker_url

    if _rag_local_healthy(env=env, rag_url=local_url):
        return local_url
    return docker_url


def _rag_local_healthy(*, env: dict[str, str], rag_url: str) -> bool:
    health_path = str(env.get("RAG_ENGINE_HEALTH_PATH") or "/health")
    probe_timeout_ms = int(str(env.get("RAG_ENGINE_PROBE_TIMEOUT_MS") or "300"))
    timeout_seconds = max(float(probe_timeout_ms) / 1000.0, 0.05)

    try:
        response = httpx.get(f"{rag_url.rstrip('/')}{health_path}", timeout=timeout_seconds)
    except Exception:
        return False
    return 200 <= response.status_code < 300


def _require_orch_health(orchestrator_url: str) -> None:
    try:
        response = httpx.get(f"{orchestrator_url}/health", timeout=3.0)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {orchestrator_url}/health\n💡 Ejecuta ./stack.sh up"
        ) from exc

    if not (200 <= response.status_code < 300):
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {orchestrator_url}/health\n💡 Ejecuta ./stack.sh up"
        )


def _extract_internal_ingest_flags(args: list[str]) -> tuple[bool, list[str]]:
    non_interactive_flag = False
    passthrough: list[str] = []
    for arg in args:
        if arg == "--orch-non-interactive-auth":
            non_interactive_flag = True
            continue
        passthrough.append(arg)
    return non_interactive_flag, passthrough


def _has_help_flag(args: list[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in args)


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag for arg in args)


async def _resolve_tenant_from_orch(
    *,
    orchestrator_url: str,
    token: str,
    non_interactive: bool,
) -> tuple[Tenant | None, bool]:
    tenants: list[Tenant] = []
    unauthorized = False
    try:
        tenants = await list_authorized_tenants(orchestrator_url, token)
    except OrchestratorDiscoveryError as exc:
        unauthorized = exc.status_code == 401
        print(f"⚠️ Tenant discovery ORCH falló: {exc}")
    except Exception as exc:
        print(f"⚠️ Tenant discovery ORCH falló: {exc}")

    if not tenants and not unauthorized:
        try:
            response = httpx.get(
                f"{orchestrator_url.rstrip('/')}/api/v1/knowledge/tenants",
                headers={"Authorization": f"Bearer {token}"},
                timeout=4.0,
            )
            if 200 <= response.status_code < 300:
                payload = response.json()
                items = payload if isinstance(payload, list) else payload.get("items", [])
                parsed: list[Tenant] = []
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        tenant_id = str(item.get("id") or item.get("tenant_id") or "").strip()
                        if not tenant_id:
                            continue
                        name = str(item.get("name") or tenant_id).strip() or tenant_id
                        parsed.append(Tenant(id=tenant_id, name=name))
                tenants = parsed
            else:
                print(
                    "⚠️ Tenant discovery fallback HTTP falló "
                    f"(status={response.status_code})."
                )
                if response.status_code == 401:
                    unauthorized = True
        except Exception as exc:
            print(f"⚠️ Tenant discovery fallback falló: {exc}")

    if not tenants:
        return None, unauthorized
    if len(tenants) == 1:
        selected = tenants[0]
        print(f"🏢 Tenant auto-seleccionado: {selected.name} ({selected.id})")
        return selected, unauthorized
    if non_interactive:
        return None, unauthorized

    print("🏢 Tenants disponibles:")
    print("  0) Crear tenant nuevo (setup interactivo de ingesta)")
    for idx, tenant in enumerate(tenants, start=1):
        print(f"  {idx}) {tenant.name} ({tenant.id})")

    option = input(f"📝 Selecciona Tenant [0-{len(tenants)}]: ").strip()
    if not option:
        return None, unauthorized
    if option.isdigit():
        selected = int(option)
        if selected == 0:
            return None, unauthorized
        if 1 <= selected <= len(tenants):
            return tenants[selected - 1], unauthorized
    print("⚠️ Opción inválida. Continuando con setup interactivo de ingesta.")
    return None, unauthorized


async def _resolve_access_token(*, env: dict[str, str], non_interactive: bool) -> str:
    now = int(time.time())
    for key in ("ORCH_ACCESS_TOKEN", "SUPABASE_ACCESS_TOKEN", "AUTH_BEARER_TOKEN"):
        token = str(env.get(key) or "").strip()
        if not token:
            continue
        exp = decode_jwt_exp(token)
        if exp is not None and exp <= now + 60:
            continue
        return token

    try:
        return await ensure_access_token(interactive=not non_interactive)
    except AuthClientError as exc:
        raise RuntimeError(f"❌ {exc}") from exc


    return ExecContext(script="", argv=passthrough_args, env=merged_env)


async def _run_chat(args: list[str]) -> int:
    logger.info("cli_command_start", extra={"event": "cli_command_start", "command": "chat"})
    await chat_cli_runtime.main(args)
    logger.info("cli_command_end", extra={"event": "cli_command_end", "command": "chat"})
    return 0

async def _run_ingest(args: list[str], *, exec_fn: ExecFn = os.execvpe) -> int:
    logger.info("cli_command_start", extra={"event": "cli_command_start", "command": "ingest"})
    context = await build_ingest_exec_context(args)
    # Re-inyectamos el token al env si fue resuelto durante el build_context
    if context.env.get("ORCH_ACCESS_TOKEN"):
        os.environ["ORCH_ACCESS_TOKEN"] = context.env["ORCH_ACCESS_TOKEN"]
    if context.env.get("TENANT_ID"):
        os.environ["TENANT_ID"] = context.env["TENANT_ID"]
    
    await ing_cli_runtime.main(context.argv)
    logger.info("cli_command_end", extra={"event": "cli_command_end", "command": "ingest"})
    return 0


def _main(argv: list[str], *, exec_fn: ExecFn = os.execvpe) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        _print_help()
        return 0

    command = argv[0]
    args = argv[1:]

    if command == "chat":
        try:
            return asyncio.run(_run_chat(args))
        except Exception as exc:
            logger.exception("cli_command_fail", extra={"event": "cli_command_fail", "command": "chat"})
            print(str(exc))
            return 1

    if command == "ingest":
        try:
            return asyncio.run(_run_ingest(args, exec_fn=exec_fn))
        except Exception as exc:
            logger.exception("cli_command_fail", extra={"event": "cli_command_fail", "command": "ingest"})
            print(str(exc))
            return 1

    print(f"❌ Comando no reconocido: {command}")
    _print_help()
    return 2


def main(argv: list[str] | None = None) -> int:
    return _main(list(argv if argv is not None else sys.argv[1:]))
