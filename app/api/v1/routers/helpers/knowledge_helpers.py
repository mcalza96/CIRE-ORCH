from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from app.agent.engine import HandleQuestionResult
from app.profiles.models import AgentProfile
from app.agent.components import build_citation_bundle
from app.api.v1.schemas.knowledge_schemas import CollectionItem

THINKING_PHASES: list[dict[str, Any]] = [
    {
        "at_seconds": 0.4,
        "phase": "intent_analysis",
        "label": "Analizando intencion y alcance",
    },
    {
        "at_seconds": 1.2,
        "phase": "retrieval",
        "label": "Recuperando clausulas y evidencia",
    },
    {
        "at_seconds": 2.4,
        "phase": "graph_reasoning",
        "label": "Navegando relaciones semanticas",
    },
    {
        "at_seconds": 3.6,
        "phase": "coverage_validation",
        "label": "Validando suficiencia de evidencia",
    },
    {
        "at_seconds": 4.6,
        "phase": "synthesis",
        "label": "Generando respuesta final",
    },
]

def format_sse_event(event: str, payload: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")

def classify_orchestrator_error(exc: Exception) -> str:
    text = str(exc or "").strip().lower()
    if "orch_answer_generation_failed" in text:
        return "ORCH_ANSWER_GENERATION_FAILED"
    if "rag retrieval failed" in text:
        return "ORCH_RETRIEVAL_FAILED"
    if isinstance(exc, TimeoutError):
        return "ORCH_TIMEOUT"
    if isinstance(exc, ValueError):
        return "ORCH_INVALID_INPUT"
    return "ORCH_UNHANDLED_ERROR"

def map_orchestrator_result(
    result: HandleQuestionResult,
    agent_profile: AgentProfile,
    profile_resolution: dict[str, Any],
) -> dict[str, Any]:
    """
    Maps the internal HandleQuestionResult to the API response format.
    """
    context_chunks = [item.content for item in result.answer.evidence]
    citations, citations_detailed, citation_quality = build_citation_bundle(
        answer_text=result.answer.text,
        evidence=result.answer.evidence,
        profile=agent_profile,
        requested_scopes=tuple(result.plan.requested_standards or ()),
    )
    
    interaction_metrics = dict((result.retrieval.trace or {}).get("interaction_metrics") or {})
    
    return {
        "answer": result.answer.text,
        "mode": result.plan.mode,
        "engine": str(result.engine or "universal_flow"),
        "agent_profile": {
            "profile_id": agent_profile.profile_id,
            "version": agent_profile.version,
            "status": agent_profile.status,
            "resolution": profile_resolution,
        },
        "citations": citations,
        "citations_detailed": citations_detailed,
        "citation_quality": citation_quality,
        "context_chunks": context_chunks,
        "requested_scopes": list(result.plan.requested_standards),
        "retrieval_plan": {
            "promoted": bool((result.retrieval.trace or {}).get("promoted", False)),
            "reason": str(
                (result.retrieval.trace or {}).get("reason")
                or (result.retrieval.trace or {}).get("fallback_reason")
                or ""
            ),
            "initial_mode": str((result.retrieval.trace or {}).get("initial_mode") or ""),
            "final_mode": str((result.retrieval.trace or {}).get("final_mode") or ""),
            "missing_scopes": list((result.retrieval.trace or {}).get("missing_scopes") or []),
            "fallback_blocked_by_literal_lock": bool(
                (result.retrieval.trace or {}).get("fallback_blocked_by_literal_lock", False)
            ),
            "subqueries": list((result.retrieval.trace or {}).get("subqueries") or []),
            "timings_ms": dict((result.retrieval.trace or {}).get("timings_ms") or {}),
        },
        "retrieval": {
            "contract": result.retrieval.contract,
            "strategy": result.retrieval.strategy,
            "partial": bool(result.retrieval.partial),
            "trace": dict(result.retrieval.trace or {}),
        },
        "interaction": interaction_metrics,
        "scope_validation": dict(result.retrieval.scope_validation or {}),
        "clarification": (
            {
                "question": result.clarification.question,
                "options": list(result.clarification.options),
                "kind": str(result.clarification.kind or "clarification"),
                "level": str(result.clarification.level or "L2"),
                "missing_slots": list(
                    ((result.retrieval.trace or {}).get("clarification_request") or {}).get(
                        "missing_slots"
                    )
                    or []
                ),
                "expected_answer": str(
                    ((result.retrieval.trace or {}).get("clarification_request") or {}).get(
                        "expected_answer"
                    )
                    or ""
                ),
            }
            if result.clarification
            else None
        ),
        "validation": {
            "accepted": result.validation.accepted,
            "issues": list(result.validation.issues),
        },
        "reasoning_trace": dict(result.reasoning_trace or {}),
    }

def map_collection_items(raw_items: list[dict[str, Any]]) -> list[CollectionItem]:
    """
    Maps raw RAG API collection items to the CollectionItem schema.
    """
    out: list[CollectionItem] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            continue
        ckey = str(item.get("collection_key") or "").strip() or None
        cname = str(item.get("name") or item.get("collection_name") or ckey or cid).strip()
        out.append(CollectionItem(id=cid, name=cname or cid, collection_key=ckey))
    return out
