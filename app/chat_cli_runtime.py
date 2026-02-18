"""HTTP-based chat CLI for split orchestrator architecture."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
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
DOCTOR_DEFAULT_QUERY = "Que exige la referencia 7.5.3 en este alcance?"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    default_orchestrator_url = (
        os.getenv("ORCH_URL") or os.getenv("ORCHESTRATOR_URL") or "http://localhost:8001"
    )
    parser = argparse.ArgumentParser(description="Q/A chat via Orchestrator API")
    parser.add_argument(
        "--tenant-id", help="Institutional tenant id (optional if tenant storage is configured)"
    )
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
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("ORCH_HTTP_READ_TIMEOUT_SECONDS") or 45.0),
        help="Read timeout for /knowledge/answer requests (default: env ORCH_HTTP_READ_TIMEOUT_SECONDS or 45)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print verbose error diagnostics (exception types, traces, HTTP payload snippets)",
    )
    return parser.parse_args(argv)


def _rewrite_query_with_clarification(original_query: str, clarification_answer: str) -> str:
    text = (clarification_answer or "").strip()
    if not text:
        return original_query
    lowered = text.lower().strip()
    if re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", lowered):
        return f"{original_query}\n\n__clarified_mode__={lowered} Aclaracion de modo: {text}."
    coverage_tag = ""
    if lowered in {"respuesta parcial", "aceptar respuesta parcial"}:
        coverage_tag = "__coverage__=partial "
    elif lowered in {"cobertura completa", "exigir cobertura completa"}:
        coverage_tag = "__coverage__=full "
    return (
        f"{original_query}\n\n__clarified_scope__=true {coverage_tag}Aclaracion de alcance: {text}."
    )


def _apply_mode_override(query: str, forced_mode: str | None) -> str:
    mode = str(forced_mode or "").strip()
    if not mode:
        return query
    lowered = mode.lower()
    if not re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", lowered):
        return query
    return f"__mode__={lowered} {query}".strip()


def _print_profile_snapshot(last_result: dict[str, Any], forced_mode: str | None = None) -> None:
    profile_resolution = (
        last_result.get("profile_resolution")
        if isinstance(last_result.get("profile_resolution"), dict)
        else {}
    )
    retrieval_plan = (
        last_result.get("retrieval_plan")
        if isinstance(last_result.get("retrieval_plan"), dict)
        else {}
    )
    retrieval = (
        last_result.get("retrieval") if isinstance(last_result.get("retrieval"), dict) else {}
    )
    trace = retrieval.get("trace") if isinstance(retrieval.get("trace"), dict) else {}
    print("🧩 Profile Snapshot")
    if profile_resolution:
        print(f"   source={profile_resolution.get('source')}")
        print(f"   applied_profile_id={profile_resolution.get('applied_profile_id')}")
        if profile_resolution.get("decision_reason"):
            print(f"   decision_reason={profile_resolution.get('decision_reason')}")
    if forced_mode:
        print(f"   forced_mode={forced_mode}")
    mode = str(last_result.get("mode") or retrieval_plan.get("mode") or "").strip()
    if mode:
        print(f"   active_mode={mode}")
    if retrieval_plan:
        print(
            "   retrieval_plan="
            f"chunk_k={retrieval_plan.get('chunk_k')} "
            f"fetch_k={retrieval_plan.get('chunk_fetch_k')} "
            f"summary_k={retrieval_plan.get('summary_k')} "
            f"literal={retrieval_plan.get('require_literal_evidence')}"
        )
    error_codes = trace.get("error_codes") if isinstance(trace.get("error_codes"), list) else []
    if error_codes:
        print("   error_codes=" + ", ".join(str(code) for code in error_codes))


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


def _print_debug_http_error(response: httpx.Response) -> None:
    try:
        body_text = response.text
    except Exception:
        body_text = ""

    cid = response.headers.get("X-Correlation-ID") or response.headers.get("x-correlation-id")
    ctype = response.headers.get("Content-Type") or response.headers.get("content-type")
    if cid:
        print(f"   correlation_id={cid}")
    if ctype:
        print(f"   content_type={ctype}")

    error = _parse_error_payload(response)
    if error:
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        request_id = str(error.get("request_id") or cid or "")
        if code:
            print(f"   code={code}")
        if request_id:
            print(f"   request_id={request_id}")
        if message:
            print(f"   message={message[:400]}")

    if body_text:
        snippet = body_text.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."
        print(f"   body={snippet}")


def _print_debug_exception(exc: BaseException) -> None:
    import traceback

    print(f"   exc_type={type(exc).__name__}")
    try:
        print(f"   exc_repr={exc!r}")
    except Exception:
        pass
    traceback.print_exception(exc)


def _short_token(token: str) -> str:
    if len(token) < 12:
        return token
    return f"{token[:6]}...{token[-4:]}"


def _prompt(message: str) -> str:
    return input(message).strip()


def _require_orch_health(orchestrator_url: str) -> None:
    base = orchestrator_url.rstrip("/")
    health_url = f"{base}/health"
    try:
        response = httpx.get(health_url, timeout=3.0)
    except Exception as exc:
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {health_url}\n💡 Ejecuta ./stack.sh up"
        ) from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(
            f"❌ Orchestrator API no disponible en {health_url}\n💡 Ejecuta ./stack.sh up"
        )


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
            extra={
                "event": "tenant_missing_blocked",
                "endpoint": "/api/v1/knowledge/answer",
                "status": "blocked",
            },
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
        request_id = str(
            error.get("request_id") or response.headers.get("X-Correlation-ID") or "unknown"
        )
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


async def _post_explain(
    client: httpx.AsyncClient,
    orchestrator_url: str,
    tenant_context: TenantContext,
    query: str,
    collection_id: str | None,
    access_token: str | None = None,
) -> dict[str, Any]:
    resolved_tenant = tenant_context.get_tenant()
    if not resolved_tenant:
        raise TenantSelectionRequiredError()

    payload: dict[str, Any] = {
        "query": query,
        "tenant_id": resolved_tenant,
        "collection_id": collection_id,
        "top_n": 10,
        "k": 12,
        "fetch_k": 60,
        "filters": None,
    }
    headers = {"X-Tenant-ID": resolved_tenant}
    token = str(access_token or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = await client.post(
        orchestrator_url.rstrip("/") + "/api/v1/knowledge/explain-retrieval",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, dict) else {}


def _print_trace(last_result: dict[str, Any]) -> None:
    retrieval = (
        last_result.get("retrieval") if isinstance(last_result.get("retrieval"), dict) else {}
    )
    retrieval_plan = (
        last_result.get("retrieval_plan")
        if isinstance(last_result.get("retrieval_plan"), dict)
        else {}
    )
    scope_validation = (
        last_result.get("scope_validation")
        if isinstance(last_result.get("scope_validation"), dict)
        else {}
    )
    print("🧾 Trace")
    print(f"   contract={retrieval.get('contract')}")
    print(f"   strategy={retrieval.get('strategy')}")
    print(f"   partial={bool(retrieval.get('partial', False))}")

    if retrieval_plan:
        promoted = bool(retrieval_plan.get("promoted", False))
        reason = str(retrieval_plan.get("reason") or "").strip()
        if promoted or reason:
            print(f"   plan_promoted={promoted}")
            if reason:
                print(f"   plan_reason={reason}")
        timings = (
            retrieval_plan.get("timings_ms")
            if isinstance(retrieval_plan.get("timings_ms"), dict)
            else {}
        )
        if timings:
            parts: list[str] = []
            for key, value in list(timings.items())[:10]:
                if isinstance(value, (int, float)):
                    parts.append(f"{key}={round(float(value), 2)}")
            if parts:
                print("   timings_ms=" + ", ".join(parts))
        flags = (
            retrieval_plan.get("kernel_flags")
            if isinstance(retrieval_plan.get("kernel_flags"), dict)
            else {}
        )
        if flags:
            enabled = ", ".join(k for k, v in flags.items() if bool(v))
            if enabled:
                print(f"   kernel_flags_enabled={enabled}")
        subq = (
            retrieval_plan.get("subqueries")
            if isinstance(retrieval_plan.get("subqueries"), list)
            else []
        )
        if subq:
            shown: list[str] = []
            for item in subq[:6]:
                if not isinstance(item, dict):
                    continue
                qid = str(item.get("id") or "")
                status = str(item.get("status") or "")
                latency = item.get("latency_ms")
                cnt = item.get("items_count")
                if qid:
                    frag = f"{qid}:{status}"
                    if isinstance(cnt, int):
                        frag += f" items={cnt}"
                    if isinstance(latency, (int, float)):
                        frag += f" {round(float(latency), 1)}ms"
                    shown.append(frag)
            if shown:
                print("   subqueries=" + " | ".join(shown))
    trace = retrieval.get("trace") if isinstance(retrieval.get("trace"), dict) else {}
    if trace:
        keys = ", ".join(sorted(trace.keys())[:12])
        print(f"   trace_keys={keys}")
        coverage_preference = str(trace.get("coverage_preference") or "").strip()
        if coverage_preference:
            print(f"   coverage_preference={coverage_preference}")
        missing_scopes = (
            trace.get("missing_scopes") if isinstance(trace.get("missing_scopes"), list) else []
        )
        if missing_scopes:
            print("   missing_scopes=" + ", ".join(str(x) for x in missing_scopes))
        layer_counts = (
            trace.get("layer_counts") if isinstance(trace.get("layer_counts"), dict) else {}
        )
        if layer_counts:
            compact = ", ".join(
                f"{k}={v}" for k, v in list(layer_counts.items())[:8] if isinstance(v, int)
            )
            if compact:
                print(f"   layer_counts={compact}")
        raptor = trace.get("raptor_summary_count")
        if isinstance(raptor, int) and raptor:
            print(f"   raptor_summary_count={raptor}")
        feats = trace.get("rag_features") if isinstance(trace.get("rag_features"), dict) else {}
        if feats:
            enabled = ", ".join(
                f"{k}={v}" for k, v in feats.items() if str(v) not in {"", "False", "0"}
            )
            if enabled:
                print(f"   rag_features={enabled}")
        attempts = trace.get("attempts") if isinstance(trace.get("attempts"), list) else []
        if attempts:
            last = attempts[-1] if isinstance(attempts[-1], dict) else {}
            action = str(last.get("action") or "")
            validation = last.get("validation") if isinstance(last.get("validation"), dict) else {}
            accepted = bool(validation.get("accepted", True))
            print(f"   attempts={len(attempts)} last_action={action} last_accepted={accepted}")
    query_scope = (
        scope_validation.get("query_scope")
        if isinstance(scope_validation.get("query_scope"), dict)
        else {}
    )
    if query_scope:
        requested = (
            query_scope.get("requested_standards")
            if isinstance(query_scope.get("requested_standards"), list)
            else []
        )
        if requested:
            print("   requested_standards=" + ", ".join(str(x) for x in requested))
        if bool(query_scope.get("requires_scope_clarification", False)):
            print("   requires_scope_clarification=true")


def _print_explain(payload: dict[str, Any]) -> None:
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
    print("🔎 Explain (top)")
    if trace:
        print(f"   engine_mode={trace.get('engine_mode')}")
        print(f"   planner_multihop={trace.get('planner_multihop')}")
        warnings = trace.get("warnings") if isinstance(trace.get("warnings"), list) else []
        if warnings:
            print("   warnings=" + " | ".join(str(w) for w in warnings[:4]))
    for idx, item in enumerate(items[:10], start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        score = item.get("score")
        explain = item.get("explain") if isinstance(item.get("explain"), dict) else {}
        comps = (
            explain.get("score_components")
            if isinstance(explain.get("score_components"), dict)
            else {}
        )
        final_score = comps.get("final_score")
        base_sim = comps.get("base_similarity")
        jina = comps.get("jina_relevance_score")
        penalized = comps.get("scope_penalized")
        print(
            f"   {idx}) {source} score={score} final={final_score} sim={base_sim} jina={jina} penalized={penalized}"
        )


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
        print("\n📚 Citas: " + ", ".join(str(item) for item in citations))
    citation_quality = (
        data.get("citation_quality") if isinstance(data.get("citation_quality"), dict) else {}
    )
    if citation_quality:
        print(
            "📏 Calidad citas: "
            f"structured={citation_quality.get('structured_count')}/{citation_quality.get('total')} "
            f"ratio={citation_quality.get('structured_ratio')} "
            f"noise={citation_quality.get('discarded_noise')}"
        )
    citation_details = (
        data.get("citations_detailed") if isinstance(data.get("citations_detailed"), list) else []
    )
    if citation_details:
        print("\n📚 Evidencias detalladas")
        for raw in citation_details:
            if not isinstance(raw, dict):
                continue
            cid = str(raw.get("id") or "").strip() or "?"
            standard = str(raw.get("standard") or "").strip() or "N/A"
            clause = str(raw.get("clause") or "").strip() or "N/A"
            snippet = str(raw.get("snippet") or "").strip()
            used = bool(raw.get("used_in_answer", False))
            marker = "*" if used else "-"
            rendered = str(raw.get("rendered") or "").strip()
            missing_fields = (
                raw.get("missing_fields") if isinstance(raw.get("missing_fields"), list) else []
            )
            noise = bool(raw.get("noise", False))
            line = (
                f"{marker} {rendered}"
                if rendered
                else f"{marker} {cid} | {standard} | clausula {clause}"
            )
            if not rendered and snippet:
                line += f' | "{snippet}"'
            if missing_fields:
                line += " | missing=" + ",".join(str(x) for x in missing_fields)
            if noise:
                line += " | noise=true"
            print(line)
    if not accepted and issues:
        print("⚠️ Validacion: " + "; ".join(str(issue) for issue in issues))
    if not citations and not (context_chunks := data.get("context_chunks")):
        print("💡 Sin evidencia recuperada. Prueba con otra colección o con '0) Todas / Default'.")
    _print_answer_diagnostics(data)
    print("=" * 60 + "\n")


def _extract_retrieval_warnings(data: dict[str, Any]) -> list[str]:
    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    retrieval_plan = (
        data.get("retrieval_plan") if isinstance(data.get("retrieval_plan"), dict) else {}
    )
    trace = retrieval.get("trace") if isinstance(retrieval.get("trace"), dict) else {}
    hybrid_trace = trace.get("hybrid_trace") if isinstance(trace.get("hybrid_trace"), dict) else {}

    out: list[str] = []
    warning = str(trace.get("warning") or "").strip()
    if warning:
        out.append(warning)
    for raw in trace.get("warnings") if isinstance(trace.get("warnings"), list) else []:
        text = str(raw or "").strip()
        if text:
            out.append(text)
    for raw in trace.get("warning_codes") if isinstance(trace.get("warning_codes"), list) else []:
        text = str(raw or "").strip()
        if text:
            out.append(text)
    for raw in (
        hybrid_trace.get("warnings") if isinstance(hybrid_trace.get("warnings"), list) else []
    ):
        text = str(raw or "").strip()
        if text:
            out.append(text)
    for raw in (
        hybrid_trace.get("warning_codes")
        if isinstance(hybrid_trace.get("warning_codes"), list)
        else []
    ):
        text = str(raw or "").strip()
        if text:
            out.append(text)
    plan_reason = str(retrieval_plan.get("reason") or "").strip()
    if plan_reason:
        out.append(plan_reason)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _print_citations_only(data: dict[str, Any]) -> None:
    citation_details = (
        data.get("citations_detailed") if isinstance(data.get("citations_detailed"), list) else []
    )
    citation_quality = (
        data.get("citation_quality") if isinstance(data.get("citation_quality"), dict) else {}
    )
    if citation_quality:
        print("📏 citation_quality=" + str(citation_quality))
    if not citation_details:
        print("ℹ️ No hay citas detalladas disponibles.")
        return
    print("📚 Citas detalladas (solo)")
    for raw in citation_details:
        if not isinstance(raw, dict):
            continue
        rendered = str(raw.get("rendered") or "").strip()
        if rendered:
            print("- " + rendered)
        else:
            cid = str(raw.get("id") or "?")
            print("- " + cid)


def _print_answer_diagnostics(data: dict[str, Any]) -> None:
    retrieval = data.get("retrieval") if isinstance(data.get("retrieval"), dict) else {}
    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    context_chunks = (
        data.get("context_chunks") if isinstance(data.get("context_chunks"), list) else []
    )
    citations = data.get("citations") if isinstance(data.get("citations"), list) else []
    issues = validation.get("issues") if isinstance(validation.get("issues"), list) else []
    accepted = bool(validation.get("accepted", True))

    degraded = (not accepted) or (len(context_chunks) == 0) or (len(citations) == 0)
    if not degraded:
        return

    contract = str(retrieval.get("contract") or "").strip() or "unknown"
    strategy = str(retrieval.get("strategy") or "").strip() or "unknown"
    warnings = _extract_retrieval_warnings(data)
    issues_text = " | ".join(str(x) for x in issues).lower()
    warning_text = " | ".join(warnings).lower()

    stage = "synthesis"
    reason = "respuesta sin evidencia suficiente o con validacion fallida"
    if len(context_chunks) == 0:
        stage = "retrieval"
        reason = "no se recuperaron chunks para la respuesta"
    if "source markers" in issues_text:
        stage = "synthesis"
        reason = "la generacion no incluyo marcadores C#/R# exigidos por validacion"
    if "scope mismatch" in issues_text:
        stage = "scope_validation"
        reason = "inconsistencia entre alcance consultado y evidencia/respuesta"
    if "advanced_contract_404_fallback_legacy" in warning_text:
        stage = "retrieval_contract"
        reason = "RAG no expone endpoints advanced y ORCH cayo a legacy"
    if (
        "pgrst202" in warning_text
        or "could not find the function" in warning_text
        or "hybrid_rpc_signature_mismatch_hnsw_ef_search" in warning_text
        or "hybrid_rpc_preflight_signature_mismatch" in warning_text
        or "hybrid_rpc_signature_mismatch_hnsw" in warning_text
    ):
        stage = "rag_sql_contract"
        reason = "desalineacion de firma RPC en Supabase (funcion/parametros)"

    print("🩺 Diagnostico")
    print(f"   stage={stage}")
    print(f"   reason={reason}")
    print(f"   retrieval={contract}/{strategy}")
    if warnings:
        top = " | ".join(warnings[:2])
        print(f"   warnings={top}")
    print("   next=/trace (detalle) | /explain (ranking de retrieval)")


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
    context_chunks = (
        result.get("context_chunks") if isinstance(result.get("context_chunks"), list) else []
    )
    citations = result.get("citations") if isinstance(result.get("citations"), list) else []
    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    accepted = bool(validation.get("accepted", True))
    retrieval = result.get("retrieval") if isinstance(result.get("retrieval"), dict) else {}
    retrieval_plan = (
        result.get("retrieval_plan") if isinstance(result.get("retrieval_plan"), dict) else {}
    )
    strategy = str(retrieval.get("strategy") or "unknown")
    contract = str(retrieval.get("contract") or "unknown")
    timings = (
        retrieval_plan.get("timings_ms")
        if isinstance(retrieval_plan.get("timings_ms"), dict)
        else {}
    )
    timings_compact = ""
    if timings:
        parts: list[str] = []
        for key, value in list(timings.items())[:4]:
            if isinstance(value, (int, float)):
                parts.append(f"{key}={round(float(value), 1)}")
        if parts:
            timings_compact = " timings_ms(" + ", ".join(parts) + ")"
    print(
        "📈 obs:"
        f" mode={mode}"
        f" retrieval={contract}/{strategy}"
        f" context_chunks={len(context_chunks)}"
        f" citations={len(citations)}"
        f" validation={accepted}"
        f" latency_ms={round(latency_ms, 2)}"
        f"{timings_compact}"
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
        async with client.stream(
            "GET", url, params=params, headers=headers, timeout=None
        ) as response:
            if response.status_code < 200 or response.status_code >= 300:
                text = await response.aread()
                print(
                    f"❌ watch failed (HTTP {response.status_code}): {text.decode('utf-8', errors='ignore')[:300]}"
                )
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
                    batch = (
                        progress.get("batch")
                        if isinstance(progress, dict) and isinstance(progress.get("batch"), dict)
                        else {}
                    )
                    obs = (
                        progress.get("observability")
                        if isinstance(progress, dict)
                        and isinstance(progress.get("observability"), dict)
                        else {}
                    )
                    status = str(batch.get("status") or "unknown")
                    percent = float(obs.get("progress_percent") or 0.0)
                    stage = str(obs.get("dominant_stage") or "OTHER")
                    eta = int(obs.get("eta_seconds") or 0)
                    stalled = bool(obs.get("stalled", False))
                    print(
                        f"📡 status={status} progress={percent}% stage={stage} eta={eta}s stalled={stalled}"
                    )
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


async def _resolve_collection(
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
        result = await _post_answer(
            client=client,
            orchestrator_url=args.orchestrator_url,
            tenant_context=tenant_context,
            query=args.doctor_query,
            collection_id=collection_id,
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


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        _require_orch_health(args.orchestrator_url)
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
    except OrchestratorDiscoveryError as exc:
        if exc.status_code != 401:
            print(f"❌ {exc}")
            raise SystemExit(1)
        try:
            refreshed = await ensure_access_token(interactive=not args.non_interactive)
            access_token = refreshed
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
        print(
            "🔭 Comandos: /ingestion , /watch <batch_id> , /trace , /citations , /explain , /profile , /mode"
        )

        last_result: dict[str, Any] = {}
        last_query: str = ""
        forced_mode: str | None = None

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
            if query.lower() == "/trace":
                if isinstance(last_result, dict) and last_result:
                    _print_trace(last_result)
                else:
                    print("ℹ️ No hay un resultado previo para mostrar trace.")
                continue
            if query.lower() == "/citations":
                if isinstance(last_result, dict) and last_result:
                    _print_citations_only(last_result)
                else:
                    print("ℹ️ No hay un resultado previo para mostrar citas.")
                continue
            if query.lower() == "/profile":
                if isinstance(last_result, dict) and last_result:
                    _print_profile_snapshot(last_result, forced_mode=forced_mode)
                else:
                    print("ℹ️ Aun no hay respuesta previa. Haz una consulta primero.")
                    if forced_mode:
                        print(f"   forced_mode={forced_mode}")
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
                if not re.fullmatch(r"[a-z][a-z0-9_:-]{1,63}", arg.lower()):
                    print("❌ mode invalido. Usa [a-z][a-z0-9_:-]")
                    continue
                forced_mode = arg.lower()
                print(f"✅ forced_mode={forced_mode}")
                continue
            if query.lower() == "/explain":
                if not last_query:
                    print("ℹ️ No hay una consulta previa para explicar.")
                    continue
                try:
                    payload = await _post_explain(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=last_query,
                        collection_id=collection_id,
                        access_token=access_token,
                    )
                    _print_explain(payload)
                except Exception as exc:
                    print(f"❌ explain failed: {exc}")
                continue

            try:
                t0 = time.perf_counter()
                result = await _post_answer(
                    client=client,
                    orchestrator_url=args.orchestrator_url,
                    tenant_context=tenant_context,
                    query=_apply_mode_override(query, forced_mode),
                    collection_id=collection_id,
                    access_token=access_token,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                _print_answer(result)
                last_result = result if isinstance(result, dict) else {}
                last_query = _apply_mode_override(query, forced_mode)
                if args.obs:
                    _print_obs_answer(result, latency_ms)

                clarification = (
                    result.get("clarification")
                    if isinstance(result.get("clarification"), dict)
                    else None
                )
                rounds = 0
                while clarification and rounds < 3:
                    question = str(clarification.get("question") or "").strip()
                    options = (
                        clarification.get("options")
                        if isinstance(clarification.get("options"), list)
                        else []
                    )
                    if question:
                        print("🧠 Clarificacion requerida: " + question)
                    reply = ""
                    if options:
                        print("🧩 Opciones:")
                        for idx, opt in enumerate(options, start=1):
                            print(f"  {idx}) {opt}")
                        while True:
                            selected_raw = _prompt(f"📝 Selecciona opcion [1-{len(options)}]: ")
                            if not selected_raw:
                                reply = ""
                                break
                            if selected_raw.isdigit():
                                selected = int(selected_raw)
                                if 1 <= selected <= len(options):
                                    reply = str(options[selected - 1])
                                    break
                            print("⚠️ Opción inválida. Ingresa el número de una alternativa.")
                    else:
                        reply = _prompt("📝 Aclaracion > ")
                    if not reply:
                        break
                    clarified_query = _rewrite_query_with_clarification(query, reply)
                    result = await _post_answer(
                        client=client,
                        orchestrator_url=args.orchestrator_url,
                        tenant_context=tenant_context,
                        query=_apply_mode_override(clarified_query, forced_mode),
                        collection_id=collection_id,
                        access_token=access_token,
                    )
                    _print_answer(result)
                    last_result = result if isinstance(result, dict) else {}
                    last_query = _apply_mode_override(clarified_query, forced_mode)
                    if args.obs:
                        _print_obs_answer(result, 0.0)
                    clarification = (
                        result.get("clarification")
                        if isinstance(result.get("clarification"), dict)
                        else None
                    )
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
                    _print_debug_exception(exc)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                print(f"❌ Error HTTP {status}")
                if exc.response is not None:
                    _print_debug_http_error(exc.response)
                if args.debug:
                    _print_debug_exception(exc)
            except Exception as exc:
                msg = str(exc).strip()
                if not msg:
                    msg = "(sin mensaje)"
                print(f"❌ Error: {msg} [type={type(exc).__name__}]")
                if args.debug:
                    _print_debug_exception(exc)


if __name__ == "__main__":
    asyncio.run(main())
