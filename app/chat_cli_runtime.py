"""HTTP-based chat CLI for split orchestrator architecture."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.auth_client import (
    AuthClientError,
    ensure_access_token,
)
from app.core.observability.logging_utils import compact_error
from app.core.observability.ingestion_utils import (
    show_ingestion_overview,
    watch_batch_stream,
)
from app.core.orch_discovery_client import OrchestratorDiscoveryError
from app.core import diagnostics, cli_api, cli_args, cli_utils, discovery_utils
from app.core.ui import renderers
from sdk.python.cire_rag_sdk import (
    TenantContext,
    TenantProtocolError,
    TenantSelectionRequiredError,
)

logger = logging.getLogger(__name__)


def _print_thinking_status(status: dict[str, Any], seen: set[str]) -> None:
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


async def main(argv: list[str] | None = None) -> None:
    args = cli_args.parse_chat_args(argv)

    try:
        cli_utils.require_orch_health(args.orchestrator_url)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)

    access_token = str(args.access_token or "").strip()
    if not access_token:
        try:
            access_token = await ensure_access_token(interactive=not args.non_interactive)
        except AuthClientError as exc:
            if args.non_interactive:
                print(f"❌ {exc}")
                raise SystemExit(1)
            print(f"⚠️ No se pudo resolver token automáticamente: {exc}")
            print("⚠️ Continuando sin token (útil solo si ORCH_AUTH_REQUIRED=false).")
            access_token = ""

    tenant_context = TenantContext(
        tenant_id=args.tenant_id,
        storage_path=Path(args.tenant_storage_path) if args.tenant_storage_path else None,
    )

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
            args=args, access_token=access_token
        )
    except OrchestratorDiscoveryError as exc:
        if exc.status_code != 401:
            print(f"❌ {exc}")
            raise SystemExit(1)
        try:
            refreshed = await ensure_access_token(interactive=not args.non_interactive)
            access_token = refreshed
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
                args=args, access_token=access_token
            )
        except Exception as retry_exc:
            print(f"❌ {retry_exc}")
            raise SystemExit(1)
    except Exception as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)

    # The orchestrator may spend significant time in retrieval + reranking + synthesis.
    # Use a longer READ timeout by default so the CLI doesn't time out first.
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

        scope = collection_name or collection_id or args.collection_name or "todo el tenant"
        print("🚀 Chat Q/A Orchestrator (split mode HTTP)")
        print(f"🏢 Tenant: {tenant_context.get_tenant() or '(sin seleccionar)'}")
        print(f"📁 Scope: {scope}")
        print(f"🌐 Orchestrator URL: {args.orchestrator_url}")
        print(f"🔐 Auth: {'Bearer token' if access_token else 'sin token'}")
        print(f"🧩 Cartucho: {agent_profile_id or 'automatico por tenant'}")
        print("💡 Escribe tu pregunta (o 'salir')")
        print(
            "🔭 Comandos: /ingestion , /watch <batch_id> , /trace , /citations , /explain , /profile , /cartridge , /mode"
        )

        last_result: dict[str, Any] = {}
        last_query: str = ""
        forced_mode: str | None = None

        while True:
            query = cli_utils.prompt("❓ > ")
            if query.lower() in {"salir", "exit", "quit"}:
                print("Okey, ¡adiós! 👋")
                return
            if not query:
                continue
            if query.lower() == "/ingestion":
                await show_ingestion_overview(
                    client=client,
                    orchestrator_url=args.orchestrator_url,
                    tenant_id=tenant_id,
                    access_token=access_token,
                )
                continue
            if query.lower().startswith("/watch "):
                batch_id = query.split(" ", 1)[1].strip()
                if not batch_id:
                    print("❌ Uso: /watch <batch_id>")
                    continue
                await watch_batch_stream(
                    client=client,
                    orchestrator_url=args.orchestrator_url,
                    tenant_id=tenant_id,
                    access_token=access_token,
                    batch_id=batch_id,
                )
                continue
            if query.lower() == "/trace":
                if isinstance(last_result, dict) and last_result:
                    renderers.print_trace(last_result)
                else:
                    print("ℹ️ No hay un resultado previo para mostrar trace.")
                continue
            if query.lower() == "/citations":
                if isinstance(last_result, dict) and last_result:
                    renderers.print_citations_only(last_result)
                else:
                    print("ℹ️ No hay un resultado previo para mostrar citas.")
                continue
            if query.lower() == "/profile":
                if isinstance(last_result, dict) and last_result:
                    renderers.print_profile_snapshot(last_result, forced_mode=forced_mode)
                else:
                    print("ℹ️ Aun no hay respuesta previa. Haz una consulta primero.")
                    if forced_mode:
                        print(f"   forced_mode={forced_mode}")
                continue
            if query.lower().startswith("/cartridge"):
                raw = query.split(" ", 1)
                arg = raw[1].strip() if len(raw) > 1 else ""
                if not arg:
                    print(f"ℹ️ cartridge={agent_profile_id or '(automatico por tenant)'}")
                    print("   Uso: /cartridge <profile_id> | /cartridge clear")
                    continue
                if arg.lower() in {"clear", "off", "none", "auto"}:
                    agent_profile_id = None
                    print("✅ cartridge en modo automatico por tenant")
                    continue
                agent_profile_id = arg.lower()
                print(f"✅ cartridge={agent_profile_id}")
                continue
            if query.lower().startswith("/mode"):
                raw = query.split(" ", 1)
                arg = raw[1].strip() if len(raw) > 1 else ""
                if not arg:
                    print(f"ℹ️ forced_mode={forced_mode or '(none)'}")
                    print("   Uso: /mode <mode_name> | /mode clear")
                    continue
                if arg.lower() in {"clear", "off", "none"}:
                    forced_mode = None
                    print("✅ forced_mode desactivado")
                    continue
                forced_mode = arg.lower()
                print(f"✅ forced_mode={forced_mode}")
                continue
            if query.lower() == "/explain":
                if not last_query:
                    print("ℹ️ No hay una consulta previa para explicar.")
                    continue
                try:
                    payload = await cli_api.post_explain(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=last_query,
                        collection_id=collection_id,
                        agent_profile_id=agent_profile_id,
                        access_token=access_token,
                    )
                    renderers.print_explain(payload)
                except Exception as exc:
                    print(f"❌ explain failed: {compact_error(exc)}")
                continue

            try:
                t0 = time.perf_counter()
                effective_query = cli_utils.apply_mode_override(query, forced_mode)
                if args.no_thinking_stream:
                    result = await cli_api.post_answer(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=effective_query,
                        collection_id=collection_id,
                        agent_profile_id=agent_profile_id,
                        clarification_context=None,
                        access_token=access_token,
                    )
                else:
                    seen_phases: set[str] = set()
                    result = await cli_api.post_answer_stream(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=effective_query,
                        collection_id=collection_id,
                        agent_profile_id=agent_profile_id,
                        clarification_context=None,
                        access_token=access_token,
                        on_status=lambda st: _print_thinking_status(st, seen_phases),
                    )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                clarification = (
                    result.get("clarification")
                    if isinstance(result.get("clarification"), dict)
                    else None
                )
                clarification_query_base = effective_query
                if not clarification:
                    renderers.print_answer(result)
                    last_result = result if isinstance(result, dict) else {}
                    last_query = effective_query
                    if args.obs:
                        renderers.print_obs_answer(result, latency_ms)

                rounds = 0
                while clarification and rounds < 3:
                    question = str(clarification.get("question") or "").strip()
                    clar_kind = str(clarification.get("kind") or "clarification").strip()
                    clar_level = str(clarification.get("level") or "L2").strip()
                    missing_slots = (
                        clarification.get("missing_slots")
                        if isinstance(clarification.get("missing_slots"), list)
                        else []
                    )
                    expected_answer = str(clarification.get("expected_answer") or "").strip()
                    options = (
                        clarification.get("options")
                        if isinstance(clarification.get("options"), list)
                        else []
                    )
                    if question:
                        print(f"🧠 {clar_level}/{clar_kind}: " + question)
                    reply = ""
                    if options:
                        print("🧩 Opciones:")
                        for idx, opt in enumerate(options, start=1):
                            print(f"  {idx}) {opt}")
                        while True:
                            selected_raw = cli_utils.prompt(
                                f"📝 Selecciona opcion [1-{len(options)}]: "
                            )
                            if not selected_raw:
                                reply = ""
                                break
                            if selected_raw.isdigit():
                                selected = int(selected_raw)
                                if 1 <= selected <= len(options):
                                    reply = str(options[selected - 1])
                                    if reply.strip().lower() == "comparar multiples":
                                        explicit_scopes = cli_utils.prompt(
                                            "📝 Indica alcances exactos (ej: ISO 9001, ISO 14001): "
                                        )
                                        if explicit_scopes:
                                            reply = explicit_scopes
                                    break
                            print("⚠️ Opción inválida. Ingresa el número de una alternativa.")
                    else:
                        reply = cli_utils.prompt("📝 Aclaracion > ")
                    if (
                        "scope" in [str(slot).strip().lower() for slot in missing_slots]
                        and reply
                        and not cli_utils.looks_like_scope_answer(reply)
                    ):
                        hint = (
                            expected_answer
                            or "Si quieres, confirma con 'si' o indica normas exactas."
                        )
                        print(f"ℹ️ Tomo tu respuesta como criterio de comparacion. {hint}")
                        proposed_scopes = cli_utils.propose_scope_candidates(
                            clarification_query_base,
                            reply,
                        )
                        if proposed_scopes:
                            print("💡 Propuesta de alcances: " + ", ".join(proposed_scopes))
                            confirm = cli_utils.prompt("📝 ¿Confirmas esta propuesta? [s/N]: ")
                            if confirm.strip().lower() in {"s", "si", "sí", "y", "yes"}:
                                reply = ", ".join(proposed_scopes)
                                clarification_query_base = f"__requested_scopes__=[{'|'.join(proposed_scopes)}] {clarification_query_base}"
                            else:
                                manual = cli_utils.prompt(
                                    "📝 Indica alcances exactos separados por coma: "
                                )
                                if manual:
                                    reply = manual
                    if not reply:
                        break
                    clarification_context = cli_utils.build_clarification_context(
                        clarification=clarification,
                        answer_text=reply,
                        round_no=rounds + 1,
                    )
                    result = await cli_api.post_answer(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=clarification_query_base,
                        collection_id=collection_id,
                        agent_profile_id=agent_profile_id,
                        clarification_context=clarification_context,
                        access_token=access_token,
                    )
                    clarification = (
                        result.get("clarification")
                        if isinstance(result.get("clarification"), dict)
                        else None
                    )
                    if clarification is None:
                        renderers.print_answer(result)
                        last_result = result if isinstance(result, dict) else {}
                        last_query = clarification_query_base
                        if args.obs:
                            renderers.print_obs_answer(result, 0.0)
                    rounds += 1
            except TenantSelectionRequiredError as exc:
                print(f"❌ {exc}")
            except TenantProtocolError as exc:
                print(f"❌ {exc.user_message} (code={exc.code}, request_id={exc.request_id})")
                if args.debug:
                    print(f"   raw_message={exc.message}")
                    if exc.details:
                        print(f"   details={exc.details}")
            except httpx.ReadTimeout as exc:
                print(
                    "❌ Error: ReadTimeout (el backend puede seguir trabajando). "
                    f"Sugerencia: reintenta con --timeout-seconds {int(max(60, args.timeout_seconds))} "
                    "o revisa el trace en el servidor con el trace_id/correlation_id."
                )
                if args.debug:
                    renderers.print_debug_exception(exc)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                print(f"❌ Error HTTP {status}")
                if exc.response is not None:
                    renderers.print_debug_http_error(exc.response)
                if args.debug:
                    renderers.print_debug_exception(exc)
            except Exception as exc:
                msg = compact_error(exc)
                if not msg:
                    msg = "(sin mensaje)"
                print(f"❌ Error: {msg} [type={type(exc).__name__}]")
                if args.debug:
                    renderers.print_debug_exception(exc)


if __name__ == "__main__":
    asyncio.run(main())
