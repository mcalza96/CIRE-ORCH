from __future__ import annotations

import time
from typing import Any

import httpx

from app.ui.cli import api as cli_api, utils as cli_utils
from app.infrastructure.observability.logging_utils import compact_error
from app.ui import renderers
from app.ui.chat_runtime_models import (
    ChatRuntimeContext,
    ChatSessionState,
    print_thinking_status,
)
from sdk.python.cire_rag_sdk import (
    TenantProtocolError,
    TenantSelectionRequiredError,
)


async def handle_chat_query(
    *, query: str, runtime: ChatRuntimeContext, state: ChatSessionState
) -> None:
    try:
        t0 = time.perf_counter()
        effective_query = cli_utils.apply_mode_override(query, state.forced_mode)
        initial_result = await _post_answer(
            runtime=runtime,
            state=state,
            query=effective_query,
            clarification_context=None,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        final_result, final_query, answered = await _handle_clarification_loop(
            runtime=runtime,
            state=state,
            base_query=effective_query,
            initial_result=initial_result,
        )
        if not answered and _extract_clarification(final_result) is None:
            answered = True

        if answered:
            renderers.print_answer(final_result)
            state.last_result = final_result
            state.last_query = final_query
            if runtime.args.obs:
                renderers.print_obs_answer(final_result, latency_ms)
    except TenantSelectionRequiredError as exc:
        print(f"❌ {exc}")
    except TenantProtocolError as exc:
        print(f"❌ {exc.user_message} (code={exc.code}, request_id={exc.request_id})")
        if runtime.args.debug:
            print(f"   raw_message={exc.message}")
            if exc.details:
                print(f"   details={exc.details}")
    except httpx.ReadTimeout as exc:
        print(
            "❌ Error: ReadTimeout (el backend puede seguir trabajando). "
            f"Sugerencia: reintenta con --timeout-seconds {int(max(60, runtime.args.timeout_seconds))} "
            "o revisa el trace en el servidor con el trace_id/correlation_id."
        )
        if runtime.args.debug:
            renderers.print_debug_exception(exc)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else 0
        print(f"❌ Error HTTP {status}")
        if exc.response is not None:
            renderers.print_debug_http_error(exc.response)
        if runtime.args.debug:
            renderers.print_debug_exception(exc)
    except Exception as exc:
        msg = compact_error(exc) or "(sin mensaje)"
        print(f"❌ Error: {msg} [type={type(exc).__name__}]")
        if runtime.args.debug:
            renderers.print_debug_exception(exc)


async def _post_answer(
    *,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
    query: str,
    clarification_context: dict[str, object] | None,
) -> dict[str, Any]:
    if runtime.args.no_thinking_stream:
        result = await cli_api.post_answer(
            client=runtime.client,
            orchestrator_url=runtime.args.orchestrator_url,
            tenant_context=runtime.tenant_context,
            query=query,
            collection_id=runtime.collection_id,
            agent_profile_id=state.agent_profile_id,
            clarification_context=clarification_context,
            access_token=runtime.access_token,
        )
    else:
        seen_phases: set[str] = set()
        result = await cli_api.post_answer_stream(
            client=runtime.client,
            orchestrator_url=runtime.args.orchestrator_url,
            tenant_context=runtime.tenant_context,
            query=query,
            collection_id=runtime.collection_id,
            agent_profile_id=state.agent_profile_id,
            clarification_context=clarification_context,
            access_token=runtime.access_token,
            on_status=lambda st: print_thinking_status(st, seen_phases),
        )
    return result if isinstance(result, dict) else {}


def _extract_clarification(payload: dict[str, Any]) -> dict[str, Any] | None:
    clarification = payload.get("clarification")
    return clarification if isinstance(clarification, dict) else None


def _prompt_clarification_reply(
    *,
    clarification: dict[str, Any],
    clarification_query_base: str,
) -> tuple[str, str]:
    question = str(clarification.get("question") or "").strip()
    clar_kind = str(clarification.get("kind") or "clarification").strip()
    clar_level = str(clarification.get("level") or "L2").strip()
    missing_slots_raw = clarification.get("missing_slots")
    missing_slots = missing_slots_raw if isinstance(missing_slots_raw, list) else []
    expected_answer = str(clarification.get("expected_answer") or "").strip()
    options = clarification.get("options") if isinstance(clarification.get("options"), list) else []

    if question:
        print(f"🧠 {clar_level}/{clar_kind}: {question}")

    reply = ""
    if options:
        print("🧩 Opciones:")
        for idx, opt in enumerate(options, start=1):
            print(f"  {idx}) {opt}")
        while True:
            selected_raw = cli_utils.prompt(f"📝 Selecciona opcion [1-{len(options)}]: ")
            if not selected_raw:
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

    normalized_slots = [str(slot).strip().lower() for slot in missing_slots]
    updated_query = clarification_query_base
    if "scope" in normalized_slots and reply and not cli_utils.looks_like_scope_answer(reply):
        hint = expected_answer or "Si quieres, confirma con 'si' o indica normas exactas."
        print(f"ℹ️ Tomo tu respuesta como criterio de comparacion. {hint}")
        proposed_scopes = cli_utils.propose_scope_candidates(clarification_query_base, reply)
        if proposed_scopes:
            print("💡 Propuesta de alcances: " + ", ".join(proposed_scopes))
            confirm = cli_utils.prompt("📝 ¿Confirmas esta propuesta? [s/N]: ")
            if confirm.strip().lower() in {"s", "si", "sí", "y", "yes"}:
                reply = ", ".join(proposed_scopes)
                updated_query = (
                    f"__requested_scopes__=[{'|'.join(proposed_scopes)}] {clarification_query_base}"
                )
            else:
                manual = cli_utils.prompt("📝 Indica alcances exactos separados por coma: ")
                if manual:
                    reply = manual
    return reply, updated_query


async def _handle_clarification_loop(
    *,
    runtime: ChatRuntimeContext,
    state: ChatSessionState,
    base_query: str,
    initial_result: dict[str, Any],
) -> tuple[dict[str, Any], str, bool]:
    result = initial_result
    clarification = _extract_clarification(result)
    clarification_query_base = base_query
    rounds = 0

    while clarification and rounds < 3:
        reply, clarification_query_base = _prompt_clarification_reply(
            clarification=clarification,
            clarification_query_base=clarification_query_base,
        )
        if not reply:
            return result, base_query, False

        clarification_context = cli_utils.build_clarification_context(
            clarification=clarification,
            answer_text=reply,
            round_no=rounds + 1,
        )
        result = await _post_answer(
            runtime=runtime,
            state=state,
            query=clarification_query_base,
            clarification_context=clarification_context,
        )
        clarification = _extract_clarification(result)
        rounds += 1

    return result, clarification_query_base, clarification is None
