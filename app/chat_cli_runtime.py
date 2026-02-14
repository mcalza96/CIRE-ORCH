"""HTTP-based chat CLI for split orchestrator architecture."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from app.core.auth_client import (
    AuthClientError,
    decode_jwt_payload,
    ensure_access_token,
)
from app.core.orch_discovery_client import (
    OrchestratorDiscoveryError,
    list_authorized_collections,
    list_authorized_tenants,
)
from sdk.python.cire_rag_sdk import (
    TENANT_MISMATCH_CODE,
    TenantContext,
    TenantProtocolError,
    TenantSelectionRequiredError,
    user_message_for_tenant_error_code,
)

logger = logging.getLogger(__name__)
DOCTOR_DEFAULT_QUERY = "Que exige ISO 9001 en la clausula 7.5.3?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_orchestrator_url = (
        os.getenv("ORCH_URL")
        or os.getenv("ORCHESTRATOR_URL")
        or "http://localhost:8001"
    )
    parser = argparse.ArgumentParser(description="Q/A chat via Orchestrator API")
    parser.add_argument("--tenant-id", help="Institutional tenant id (optional if tenant storage is configured)")
    parser.add_argument(
        "--tenant-storage-path",
        help="Optional path to persisted tenant context JSON",
    )
    parser.add_argument("--collection-id", help="Collection id (optional)")
    parser.add_argument("--collection-name", help="Collection name (display only)")
    parser.add_argument(
        "--orchestrator-url",
        default=default_orchestrator_url,
        help="Base URL for orchestrator API",
    )
    parser.add_argument(
        "--access-token",
        default=(
            os.getenv("ORCH_ACCESS_TOKEN")
            or os.getenv("SUPABASE_ACCESS_TOKEN")
            or os.getenv("AUTH_BEARER_TOKEN")
            or ""
        ),
        help="Bearer token for orchestrator auth",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Fail if auth or tenant selection requires user input",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run auth/discovery/retrieval diagnosis and exit",
    )
    parser.add_argument(
        "--doctor-query",
        default=DOCTOR_DEFAULT_QUERY,
        help="Controlled query used by --doctor",
    )
    parser.add_argument(
        "--obs",
        action="store_true",
        help="Show compact observability diagnostics after each answer",
    )
    return parser.parse_args(argv)


def _rewrite_query_with_clarification(original_query: str, clarification_answer: str) -> str:
    text = (clarification_answer or "").strip()
    if not text:
        return original_query
    return (
        f"{original_query}\n\n"
        "__clarified_scope__=true "
        f"Aclaracion de alcance: {text}."
    )


def _parse_error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        data = None
    if isinstance(data, dict):
        if isinstance(data.get("error"), dict):
            return data["error"]
        if isinstance(data.get("detail"), dict):
            return data["detail"]
    return {}


def _short_token(token: str) -> str:
    if len(token) < 12:
        return token
    return f"{token[:6]}...{token[-4:]}"


def _prompt(message: str) -> str:
    return input(message).strip()


async def _post_answer(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_context: TenantContext,
    query: str,
    collection_id: str | None,
    access_token: str | None = None,
    retry_on_mismatch: bool = True,
) -> dict[str, Any]:
    resolved_tenant = tenant_context.get_tenant()
    if not resolved_tenant:
        logger.warning(
            "tenant_missing_blocked",
            extra={"event": "tenant_missing_blocked", "endpoint": "/api/v1/knowledge/answer", "status": "blocked"},
        )
        raise TenantSelectionRequiredError()

    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": resolved_tenant,
    }
    if collection_id:
        payload["collection_id"] = collection_id

    headers = {"X-Tenant-ID": resolved_tenant}
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = await client.post(
        orchestrator_url.rstrip("/") + "/api/v1/knowledge/answer",
        json=payload,
        headers=headers,
    )
    if response.status_code >= 400:
        error = _parse_error_payload(response)
        code = str(error.get("code") or "")
        request_id = str(error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown")
        if code == TENANT_MISMATCH_CODE and retry_on_mismatch and tenant_context.storage_path:
            logger.warning(
                "tenant_mismatch_detected",
                extra={
                    "event": "tenant_mismatch_detected",
                    "endpoint": "/api/v1/knowledge/answer",
                    "status": response.status_code,
                    "request_id": request_id,
                },
            )
            previous = resolved_tenant
            reloaded = tenant_context.reload()
            if reloaded and reloaded != previous:
                return await _post_answer(
                    client,
                    orchestrator_url,
                    tenant_context,
                    query,
                    collection_id,
                    access_token=access_token,
                    retry_on_mismatch=False,
                )
        if code:
            raise TenantProtocolError(
                status=response.status_code,
                code=code,
                message=str(error.get("message") or response.text),
                user_message=user_message_for_tenant_error_code(code),
                request_id=request_id,
                details=error.get("details"),
            )
        response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _print_answer(data: dict[str, Any]) -> None:
    answer = str(data.get("answer") or "").strip()
    mode = str(data.get("mode") or "").strip()
    citations = data.get("citations") if isinstance(data.get("citations"), list) else []
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    accepted = bool(validation.get("accepted", True))
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []

    print("\n" + "=" * 60)
    print(f"🤖 RESPUESTA ({mode or 'N/A'})")
    print("=" * 60)
    print(answer or "(sin respuesta)")
    if citations:
        print("\n📚 Citas: " + ", ".join(str(item) for item in citations[:10]))
    if not accepted and issues:
        print("⚠️ Validacion: " + "; ".join(str(issue) for issue in issues))
    print("=" * 60 + "\n")


def _obs_headers(access_token: str | None, tenant_id: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if tenant_id:
        headers["X-Tenant-ID"] = tenant_id
    return headers


def _print_obs_answer(result: dict[str, Any], latency_ms: float) -> None:
    mode = str(result.get("mode") or "unknown")
    context_chunks = result.get("context_chunks") if isinstance(result.get("context_chunks"), list) else []
    citations = result.get("citations") if isinstance(result.get("citations"), list) else []
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    accepted = bool(validation.get("accepted", True))
    print(
        "📈 obs:"
        f" mode={mode}"
        f" context_chunks={len(context_chunks)}"
        f" citations={len(citations)}"
        f" validation={accepted}"
        f" latency_ms={round(latency_ms, 2)}"
    )


async def _show_ingestion_overview(
    *,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tenant_context: TenantContext,
    access_token: str,
) -> None:
    tenant_id = tenant_context.get_tenant()
    if not tenant_id:
        print("❌ No hay tenant seleccionado.")
        return
    base = args.orchestrator_url.rstrip("/")
    headers = _obs_headers(access_token, tenant_id=tenant_id)
    try:
        active_resp = await client.get(
            f"{base}/api/v1/observability/batches/active",
            params={"tenant_id": tenant_id, "limit": 5},
            headers=headers,
        )
        active_resp.raise_for_status()
        active_payload = active_resp.json()
    except Exception as exc:
        print(f"❌ No se pudo obtener batches activos: {exc}")
        return

    queue_line = "n/a"

    items = active_payload.get("items") if isinstance(active_payload, dict) else []
    if not isinstance(items, list):
        items = []

    print("📡 Ingestion overview")
    print(f"   tenant={tenant_id}")
    print(f"   active_batches={len(items)}")
    print(f"   queue={queue_line}")
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        batch = item.get("batch") if isinstance(item.get("batch"), dict) else {}
        obs = item.get("observability") if isinstance(item.get("observability"), dict) else {}
        batch_id = str(batch.get("id") or batch.get("batch_id") or "unknown")
        status = str(batch.get("status") or "unknown")
        percent = float(obs.get("progress_percent") or 0.0)
        stage = str(obs.get("dominant_stage") or "OTHER")
        eta = int(obs.get("eta_seconds") or 0)
        print(f"   - {batch_id}: status={status} progress={percent}% stage={stage} eta={eta}s")


async def _watch_batch_stream(
    *,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tenant_context: TenantContext,
    access_token: str,
    batch_id: str,
) -> None:
    tenant_id = tenant_context.get_tenant()
    if not tenant_id:
        print("❌ No hay tenant seleccionado.")
        return
    base = args.orchestrator_url.rstrip("/")
    url = f"{base}/api/v1/observability/batches/{batch_id}/stream"
    headers = _obs_headers(access_token, tenant_id=tenant_id)
    params = {"tenant_id": tenant_id, "interval_ms": 1500}
    print(f"🔎 Watching batch {batch_id} ...")

    current_event = "message"
    try:
        async with client.stream("GET", url, params=params, headers=headers, timeout=None) as response:
            if response.status_code < 200 or response.status_code >= 300:
                text = await response.aread()
                print(f"❌ watch failed (HTTP {response.status_code}): {text.decode('utf-8', errors='ignore')[:300]}")
                return
            async for raw_line in response.aiter_lines():
                line = str(raw_line or "").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line.split(":", 1)[1].strip() or "message"
                    continue
                if not line.startswith("data:"):
                    continue
                payload_text = line.split(":", 1)[1].strip()
                try:
                    payload = json.loads(payload_text) if payload_text else {}
                except Exception:
                    continue
                if current_event == "snapshot":
                    progress = payload.get("progress") if isinstance(payload, dict) else {}
                    batch = progress.get("batch") if isinstance(progress, dict) and isinstance(progress.get("batch"), dict) else {}
                    obs = progress.get("observability") if isinstance(progress, dict) and isinstance(progress.get("observability"), dict) else {}
                    status = str(batch.get("status") or "unknown")
                    percent = float(obs.get("progress_percent") or 0.0)
                    stage = str(obs.get("dominant_stage") or "OTHER")
                    eta = int(obs.get("eta_seconds") or 0)
                    stalled = bool(obs.get("stalled", False))
                    print(f"📡 status={status} progress={percent}% stage={stage} eta={eta}s stalled={stalled}")
                elif current_event == "terminal":
                    status = str(payload.get("status") or "unknown")
                    print(f"✅ watch terminal: {status}")
                    return
                elif current_event == "error":
                    print(f"❌ watch error: {payload}")
                    return
    except Exception as exc:
        print(f"❌ stream watch error: {exc}")


async def _resolve_tenant(
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


async def _resolve_collection(
    *,
    args: argparse.Namespace,
    tenant_id: str,
    access_token: str,
) -> tuple[str | None, str | None]:
    if args.collection_id:
        return args.collection_id, args.collection_name

    try:
        collections = await list_authorized_collections(args.orchestrator_url, access_token, tenant_id)
    except OrchestratorDiscoveryError as exc:
        print(f"⚠️ No se pudieron cargar colecciones ({exc}).")
        return None, args.collection_name

    if not collections:
        return None, args.collection_name

    if args.non_interactive:
        return None, args.collection_name

    print("📁 Colecciones:")
    print("  0) Todas / Default")
    for idx, item in enumerate(collections, start=1):
        print(f"  {idx}) {item.name}")

    option = _prompt(f"📝 Selecciona Colección [0-{len(collections)}]: ")
    if option.isdigit():
        selected = int(option)
        if 1 <= selected <= len(collections):
            col = collections[selected - 1]
            return col.id, col.name

    return None, args.collection_name


async def _run_doctor(
    *,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    tenant_context: TenantContext,
    access_token: str,
    collection_id: str | None,
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
        tenants = []
        print(f"tenant_discovery_ok: no ({exc})")

    print(f"tenant_count: {tenant_count}")
    tenant_id = tenant_context.get_tenant()
    if tenant_id:
        print(f"selected_tenant: {tenant_id}")
    else:
        print("selected_tenant: none")

    if tenant_id:
        try:
            collections = await list_authorized_collections(args.orchestrator_url, access_token, tenant_id)
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
        result = await _post_answer(
            client=client,
            orchestrator_url=args.orchestrator_url,
            tenant_context=tenant_context,
            query=args.doctor_query,
            collection_id=collection_id,
            access_token=access_token,
        )
        mode = str(result.get("mode") or "unknown")
        context_chunks = result.get("context_chunks") if isinstance(result.get("context_chunks"), list) else []
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


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

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
        tenant_id = await _resolve_tenant(
            args=args,
            tenant_context=tenant_context,
            access_token=access_token,
        )
        collection_id, collection_name = await _resolve_collection(
            args=args,
            tenant_id=tenant_id,
            access_token=access_token,
        )
    except Exception as exc:
        print(f"❌ {exc}")
        raise SystemExit(1)

    timeout = httpx.Timeout(20.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if args.doctor:
            await _run_doctor(
                client=client,
                args=args,
                tenant_context=tenant_context,
                access_token=access_token,
                collection_id=collection_id,
            )
            return

        scope = collection_name or collection_id or args.collection_name or "todo el tenant"
        print("🚀 Chat Q/A Orchestrator (split mode HTTP)")
        print(f"🏢 Tenant: {tenant_context.get_tenant() or '(sin seleccionar)'}")
        print(f"📁 Scope: {scope}")
        print(f"🌐 Orchestrator URL: {args.orchestrator_url}")
        print(f"🔐 Auth: {'Bearer token' if access_token else 'sin token'}")
        print("💡 Escribe tu pregunta (o 'salir')")
        print("🔭 Comandos: /ingestion , /watch <batch_id>")

        while True:
            query = input("❓ > ").strip()
            if query.lower() in {"salir", "exit", "quit"}:
                print("Okey, ¡adiós! 👋")
                return
            if not query:
                continue
            if query.lower() == "/ingestion":
                await _show_ingestion_overview(
                    client=client,
                    args=args,
                    tenant_context=tenant_context,
                    access_token=access_token,
                )
                continue
            if query.lower().startswith("/watch "):
                batch_id = query.split(" ", 1)[1].strip()
                if not batch_id:
                    print("❌ Uso: /watch <batch_id>")
                    continue
                await _watch_batch_stream(
                    client=client,
                    args=args,
                    tenant_context=tenant_context,
                    access_token=access_token,
                    batch_id=batch_id,
                )
                continue

            try:
                t0 = time.perf_counter()
                result = await _post_answer(
                    client=client,
                    orchestrator_url=args.orchestrator_url,
                    tenant_context=tenant_context,
                    query=query,
                    collection_id=collection_id,
                    access_token=access_token,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                _print_answer(result)
                if args.obs:
                    _print_obs_answer(result, latency_ms)

                clarification = result.get("clarification") if isinstance(result.get("clarification"), dict) else None
                rounds = 0
                while clarification and rounds < 3:
                    question = str(clarification.get("question") or "").strip()
                    options = clarification.get("options") if isinstance(clarification.get("options"), list) else []
                    if question:
                        print("🧠 Clarificacion requerida: " + question)
                    if options:
                        print("🧩 Opciones: " + " | ".join(str(opt) for opt in options))
                    reply = input("📝 Aclaracion > ").strip()
                    if not reply:
                        break
                    clarified_query = _rewrite_query_with_clarification(query, reply)
                    result = await _post_answer(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=clarified_query,
                        collection_id=collection_id,
                        access_token=access_token,
                    )
                    _print_answer(result)
                    if args.obs:
                        _print_obs_answer(result, 0.0)
                    clarification = result.get("clarification") if isinstance(result.get("clarification"), dict) else None
                    rounds += 1
            except TenantSelectionRequiredError as exc:
                print(f"❌ {exc}")
            except TenantProtocolError as exc:
                print(f"❌ {exc.user_message} (code={exc.code}, request_id={exc.request_id})")
            except httpx.HTTPStatusError as exc:
                print(f"❌ Error HTTP {exc.response.status_code}: {exc.response.text}")
            except Exception as exc:
                print(f"❌ Error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
