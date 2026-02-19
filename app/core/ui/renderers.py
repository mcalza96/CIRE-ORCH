from __future__ import annotations

import httpx
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

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

def print_profile_snapshot(last_result: dict[str, Any], forced_mode: str | None = None) -> None:
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

def print_debug_http_error(response: httpx.Response) -> None:
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

def print_debug_exception(exc: BaseException) -> None:
    import traceback
    print(f"   exc_type={type(exc).__name__}")
    try:
        print(f"   exc_repr={exc!r}")
    except Exception:
        pass
    traceback.print_exception(exc)

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

def print_answer_diagnostics(data: dict[str, Any]) -> None:
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

def print_trace(last_result: dict[str, Any]) -> None:
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

def print_explain(payload: dict[str, Any]) -> None:
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

def print_answer(data: dict[str, Any]) -> None:
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
    if not citations and not (data.get("context_chunks")):
        print("💡 Sin evidencia recuperada. Prueba con otra colección o con '0) Todas / Default'.")
    print_answer_diagnostics(data)
    print("=" * 60 + "\n")

def print_citations_only(data: dict[str, Any]) -> None:
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

def print_obs_answer(result: dict[str, Any], latency_ms: float) -> None:
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
