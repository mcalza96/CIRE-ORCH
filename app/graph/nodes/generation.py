from __future__ import annotations

import asyncio
from typing import Any, cast

import structlog

from app.agent.types.models import (
    AnswerDraft,
    EvidenceItem,
    ReasoningStep,
    RetrievalDiagnostics,
    RetrievalPlan,
    ToolResult,
    ValidationResult,
)
from app.agent.tools import get_tool
from app.profiles.models import AgentProfile
from app.infrastructure.config import settings
from app.graph.logic.logic import _query_mode_aggregation_mode
from app.graph.state import ANSWER_PREVIEW_LIMIT, UniversalState
from app.graph.logic.utils import (
    _clip_text,
    _keyword_overlap_score,
    _timeout_ms_for_stage,
    get_adaptive_timeout_ms,
    state_get_dict,
    state_get_list,
    state_get_str,
    track_node_timing,
)

from .types import OrchestratorComponents

logger = structlog.get_logger(__name__)


@track_node_timing("subquery_aggregate")
async def aggregate_subqueries_node(
    state: UniversalState, components: OrchestratorComponents | None = None
) -> dict[str, object]:
    enabled_by_settings = bool(getattr(settings, "ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", False))
    aggregation_mode = _query_mode_aggregation_mode(state)
    enabled = enabled_by_settings or aggregation_mode == "grouped_map_reduce"
    if not enabled:
        return {}

    retrieval = state.get("retrieval")
    trace = retrieval.trace if isinstance(retrieval, RetrievalDiagnostics) else {}
    subqueries_from_trace = trace.get("subqueries") if isinstance(trace, dict) else []
    raw_groups = state.get("subquery_groups")
    subquery_groups_list: list[Any] = raw_groups if isinstance(raw_groups, list) else []
    chunks = cast(list[EvidenceItem], list(state.get("chunks") or []))

    max_subqueries = max(
        1,
        int(getattr(settings, "ORCH_SUBQUERY_MAP_MAX_SUBQUERIES", 8) or 8),
    )
    max_items_per_subquery = max(
        1,
        int(getattr(settings, "ORCH_SUBQUERY_MAP_ITEMS_PER_SUBQUERY", 5) or 5),
    )

    groups: list[dict[str, Any]] = [
        group for group in subquery_groups_list if isinstance(group, dict)
    ][:max_subqueries]

    if not groups and isinstance(subqueries_from_trace, list):
        groups = [item for item in subqueries_from_trace if isinstance(item, dict)][:max_subqueries]

    if not groups:
        return {}

    async def _summarize_one(
        comp: OrchestratorComponents,
        query: str,
        evidence: list[EvidenceItem],
        profile: AgentProfile | None,
    ) -> str:
        sub_plan = RetrievalPlan(
            mode="concisa_y_directa",
            chunk_k=len(evidence),
            chunk_fetch_k=len(evidence),
            summary_k=0,
            require_literal_evidence=False,
        )
        draft = await comp.answer_generator.generate(
            query=f"[SUBCONSULTA: {query}]\nResume la respuesta basandote SOLO en los fragmentos proporcionados.",
            scope_label="",
            plan=sub_plan,
            chunks=evidence,
            summaries=[],
            agent_profile=profile,
        )
        return draft.text

    partial_answers: list[dict[str, Any]] = []
    summarization_jobs: list[tuple[int, str, list[EvidenceItem]]] = []
    agent_profile = cast(AgentProfile | None, state.get("agent_profile"))

    for idx, group in enumerate(groups, start=1):
        sq_id = str(group.get("id") or f"q{idx}").strip() or f"q{idx}"
        sq_query = str(group.get("query") or "").strip()
        raw_items = [it for it in list(group.get("items") or []) if isinstance(it, dict)]

        candidates: list[EvidenceItem] = []
        if raw_items:
            for i, item in enumerate(raw_items[:max_items_per_subquery]):
                content = str(item.get("content") or "").strip()
                if not content:
                    continue
                source = str(item.get("source") or f"C{i + 1}").strip() or f"C{i + 1}"
                score_raw = item.get("score")
                if score_raw is None:
                    score_raw = item.get("similarity")
                try:
                    score = float(score_raw or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                candidates.append(
                    EvidenceItem(
                        source=source, content=content, score=score, metadata={"row": item}
                    )
                )
        elif sq_query and chunks:
            ranked = sorted(
                chunks,
                key=lambda item: (
                    _keyword_overlap_score(sq_query, item.content),
                    float(item.score or 0.0),
                ),
                reverse=True,
            )
            candidates = ranked[:max_items_per_subquery]

        if not candidates:
            partial_answers.append(
                {
                    "id": sq_id,
                    "query": sq_query,
                    "status": "no_evidence",
                    "evidence_sources": [],
                    "summary": "Sin evidencia suficiente para esta subconsulta.",
                }
            )
            continue

        partial_answers.append(
            {
                "id": sq_id,
                "query": sq_query,
                "status": "ok",
                "evidence_sources": [str(ev.source or "") for ev in candidates],
                "summary": "Resumen pendiente",
            }
        )

        partial_idx = len(partial_answers) - 1
        if components and components.answer_generator:
            summarization_jobs.append((partial_idx, sq_query, candidates))
        else:
            snippets = [
                f"{ev.source}: {_clip_text(ev.content, limit=220)}"
                for ev in candidates[:2]
                if str(ev.content or "").strip()
            ]
            partial_answers[partial_idx]["summary"] = (
                " | ".join(snippets) if snippets else "Evidencia recuperada."
            )

    if summarization_jobs and components and components.answer_generator:
        summarize_coros = [
            _summarize_one(components, sq_query, candidates, agent_profile)
            for _, sq_query, candidates in summarization_jobs
        ]
        summarize_results = await asyncio.gather(*summarize_coros, return_exceptions=True)
        for (partial_idx, _, candidates), result in zip(summarization_jobs, summarize_results):
            if isinstance(result, Exception):
                logger.error("subquery_summarization_failed", error=str(result))
                snippets = [
                    f"{ev.source}: {_clip_text(ev.content, limit=220)}"
                    for ev in candidates[:2]
                    if str(ev.content or "").strip()
                ]
                summary_text = " | ".join(snippets) if snippets else "Evidencia recuperada."
            else:
                summary_text = str(result).strip() or "Evidencia recuperada."
            partial_answers[partial_idx]["summary"] = summary_text

    return {
        "partial_answers": partial_answers,
    }


@track_node_timing("generator")
async def generator_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    plan = state.get("retrieval_plan")
    logger.warning(
        "DIAG_generator_entry",
        has_plan=isinstance(plan, RetrievalPlan),
        chunks_in_state=len(list(state.get("chunks") or [])),
        summaries_in_state=len(list(state.get("summaries") or [])),
    )
    if not isinstance(plan, RetrievalPlan):
        return {
            "stop_reason": "missing_retrieval_plan",
        }
    chunks = cast(list[EvidenceItem], list(state.get("chunks") or []))
    summaries = cast(list[EvidenceItem], list(state.get("summaries") or []))
    working_memory = state_get_dict(state, "working_memory")
    raw_partials = state.get("partial_answers")
    partial_answers_list: list[Any] = raw_partials if isinstance(raw_partials, list) else []

    try:
        generator_timeout_ms = get_adaptive_timeout_ms(
            state,
            stage_default_ms=_timeout_ms_for_stage("generator"),
            headroom_ms=1000,
        )
        answer = await asyncio.wait_for(
            components.answer_generator.generate(
                query=state_get_str(state, "user_query", ""),
                scope_label=state_get_str(state, "scope_label", ""),
                plan=plan,
                chunks=chunks,
                summaries=summaries,
                working_memory=working_memory,
                partial_answers=partial_answers_list,
                agent_profile=state.get("agent_profile"),
            ),
            timeout=generator_timeout_ms / 1000.0,
        )
    except TimeoutError:
        return {
            "stop_reason": "generator_timeout",
        }
    trace_steps = state_get_list(state, "reasoning_steps")
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="synthesis",
            description="synthesis_completed",
            output={
                "answer_preview": _clip_text(answer.text, limit=ANSWER_PREVIEW_LIMIT),
                "evidence_count": len(answer.evidence),
                "partial_answers_count": len(partial_answers_list),
            },
        )
    )
    return {
        "generation": answer,
        "reasoning_steps": trace_steps,
    }


@track_node_timing("validation")
async def citation_validate_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    answer = state.get("generation")
    plan = state.get("retrieval_plan")
    if not isinstance(answer, AnswerDraft) or not isinstance(plan, RetrievalPlan):
        return {
            "validation": ValidationResult(accepted=False, issues=["missing_generation_or_plan"]),
            "stop_reason": "validation_failed",
        }

    allowed = state_get_list(state, "allowed_tools")
    if "citation_validator" in allowed:
        tool = get_tool(components.tools or {}, "citation_validator")
        if tool is not None:
            try:
                validation_timeout_ms = get_adaptive_timeout_ms(
                    state,
                    stage_default_ms=_timeout_ms_for_stage("validation"),
                    headroom_ms=200,
                )
                result = await asyncio.wait_for(
                    tool.run({}, state=dict(state), context=components._runtime_context()),
                    timeout=validation_timeout_ms / 1000.0,
                )
            except TimeoutError:
                result = ToolResult(tool="citation_validator", ok=False, error="tool_timeout")
            accepted = (
                bool(result.output.get("accepted"))
                if isinstance(result.output, dict)
                else bool(result.ok)
            )
            issues = (
                list(result.output.get("issues") or [])
                if isinstance(result.output, dict)
                else ([result.error] if result.error else [])
            )
            validation = ValidationResult(accepted=accepted, issues=issues)
        else:
            validation = components.validator.validate(
                answer, plan, str(state.get("user_query") or "")
            )
    else:
        validation = components.validator.validate(answer, plan, str(state.get("user_query") or ""))

    trace_steps = state_get_list(state, "reasoning_steps")
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="validation",
            tool="citation_validator" if "citation_validator" in allowed else None,
            description="validation_completed",
            output={"accepted": bool(validation.accepted), "issues": list(validation.issues)},
            ok=bool(validation.accepted),
        )
    )
    stop_reason = state_get_str(state, "stop_reason", "")
    if not stop_reason:
        stop_reason = "done" if validation.accepted else "validation_failed"
    return {
        "validation": validation,
        "reasoning_steps": trace_steps,
        "stop_reason": stop_reason,
    }
