from __future__ import annotations

import asyncio
import re
import time
from typing import Any, cast, Protocol

import structlog

from app.agent.application import AnswerGeneratorPort, RetrieverPort, ValidatorPort
from app.agent.error_codes import RETRIEVAL_CODE_EMPTY_RETRIEVAL
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    ReasoningPlan,
    ReasoningStep,
    RetrievalDiagnostics,
    RetrievalPlan,
    ToolResult,
    ValidationResult,
)
from app.agent.tools import AgentTool, ToolRuntimeContext, get_tool, resolve_allowed_tools
from app.cartridges.models import AgentProfile
from app.core.config import settings
from app.graph.universal.planning import build_universal_plan
from app.graph.universal.state import (
    ANSWER_PREVIEW_LIMIT,
    DEFAULT_MAX_REFLECTIONS,
    DEFAULT_MAX_STEPS,
    HARD_MAX_REFLECTIONS,
    HARD_MAX_STEPS,
    MAX_PLAN_ATTEMPTS,
    RETRY_REASON_LIMIT,
    UniversalState,
)
from app.graph.universal.utils import (
    _append_stage_timing,
    _append_tool_timing,
    _clip_text,
    _effective_execute_tool_timeout_ms,
    _sanitize_payload,
    _timeout_ms_for_stage,
    get_adaptive_timeout_ms,
)
from app.graph.universal.logic import (
    _extract_retry_signal_from_retrieval,
    _infer_expression_from_query,
    _is_retryable_reason,
)

logger = structlog.get_logger(__name__)


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-Z0-9áéíóúñÁÉÍÓÚÑ]{3,}", str(text or "").lower())
        if token
    }


def _keyword_overlap_score(query: str, content: str) -> int:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0
    c_tokens = _tokenize(content)
    return sum(1 for token in q_tokens if token in c_tokens)


def _query_mode_aggregation_mode(state: UniversalState) -> str:
    profile = state.get("agent_profile")
    plan = state.get("retrieval_plan")
    if not isinstance(profile, AgentProfile) or not isinstance(plan, RetrievalPlan):
        return ""
    mode_cfg = profile.query_modes.modes.get(str(plan.mode or "").strip())
    if mode_cfg is None:
        return ""
    policy = getattr(mode_cfg, "decomposition_policy", None)
    if not isinstance(policy, dict):
        return ""
    return str(policy.get("subquery_aggregation_mode") or "").strip().lower()


class OrchestratorComponents(Protocol):
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, AgentTool] | None

    def _runtime_context(self) -> ToolRuntimeContext: ...


async def planner_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    query = str(state.get("working_query") or state.get("user_query") or "").strip()
    profile = state.get("agent_profile")
    allowed_tools = resolve_allowed_tools(profile, components.tools or {})

    # Use adaptive timeout for planner
    base_planner_ms = int(getattr(settings, "ORCH_TIMEOUT_CLASSIFY_MS", 2000) or 2000) + int(
        getattr(settings, "ORCH_TIMEOUT_PLAN_MS", 3000) or 3000
    )
    planner_timeout_ms = get_adaptive_timeout_ms(
        state, stage_default_ms=base_planner_ms, headroom_ms=3000
    )

    try:
        intent, retrieval_plan, reasoning_plan, trace_steps = await asyncio.wait_for(
            asyncio.to_thread(
                build_universal_plan,
                query=query,
                profile=profile,
                allowed_tools=allowed_tools,
            ),
            timeout=planner_timeout_ms / 1000.0,
        )
    except TimeoutError:
        return {
            "next_action": "generate",
            "stop_reason": "planner_timeout",
            "stage_timings_ms": _append_stage_timing(
                state, stage="planner", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    max_steps = DEFAULT_MAX_STEPS
    max_reflections = DEFAULT_MAX_REFLECTIONS
    if isinstance(profile, AgentProfile):
        max_steps = min(HARD_MAX_STEPS, int(profile.capabilities.reasoning_budget.max_steps))
        max_reflections = min(
            HARD_MAX_REFLECTIONS,
            int(profile.capabilities.reasoning_budget.max_reflections),
        )
    reasoning_plan = ReasoningPlan(
        goal=reasoning_plan.goal,
        steps=list(reasoning_plan.steps[: max(1, max_steps)]),
        complexity=reasoning_plan.complexity,
    )
    existing_steps = list(state.get("reasoning_steps") or [])
    updates: dict[str, object] = {
        "intent": intent,
        "retrieval_plan": retrieval_plan,
        "reasoning_plan": reasoning_plan,
        "allowed_tools": allowed_tools,
        "max_steps": max_steps,
        "max_reflections": max_reflections,
        "tool_cursor": 0,
        "reasoning_steps": [*existing_steps, *trace_steps],
        "next_action": ("execute" if reasoning_plan.steps else "generate"),
    }
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="planner", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    return updates


async def execute_tool_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    cursor = int(state.get("tool_cursor") or 0)
    if cursor >= len(plan.steps):
        return {
            "next_action": "generate",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    max_steps = min(HARD_MAX_STEPS, int(state.get("max_steps") or DEFAULT_MAX_STEPS))
    tool_results = list(state.get("tool_results") or [])
    if len(tool_results) >= max_steps:
        return {
            "next_action": "generate",
            "stop_reason": "max_steps_reached",
            "stage_timings_ms": _append_stage_timing(
                state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    step_call = plan.steps[cursor]
    tool_name = str(step_call.tool or "").strip()
    tool = get_tool(components.tools or {}, tool_name)
    tool_elapsed_ms = 0.0
    if tool is None:
        result = ToolResult(tool=tool_name, ok=False, error="tool_not_registered")
    else:
        payload = dict(step_call.input or {})
        # ── Piping: inject previous tool output so tools can chain ──
        if cursor > 0 and tool_results:
            prev = tool_results[-1]
            if isinstance(prev, ToolResult) and prev.ok:
                if prev.output:
                    payload.setdefault("previous_tool_output", dict(prev.output))
                if prev.metadata:
                    payload.setdefault("previous_tool_metadata", dict(prev.metadata))
        if tool_name == "python_calculator" and not payload.get("expression"):
            inferred = _infer_expression_from_query(str(state.get("working_query") or ""))
            if inferred:
                payload["expression"] = inferred
        t_tool = time.perf_counter()
        tool_timeout_ms = _effective_execute_tool_timeout_ms(tool_name)
        profile = state.get("agent_profile")
        if isinstance(profile, AgentProfile):
            policy = profile.capabilities.tool_policies.get(tool_name)
            if policy is not None:
                tool_timeout_ms = max(20, min(5000, int(policy.timeout_ms)))

        # Apply adaptive budget to tool execution
        tool_timeout_ms = get_adaptive_timeout_ms(
            state, stage_default_ms=tool_timeout_ms, headroom_ms=2800
        )

        try:
            result = await asyncio.wait_for(
                tool.run(
                    payload,
                    state=dict(state),
                    context=components._runtime_context(),
                ),
                timeout=tool_timeout_ms / 1000.0,
            )
        except TimeoutError:
            result = ToolResult(tool=tool_name, ok=False, error="tool_timeout")
        except Exception as e:
            logger.error("tool_execution_failed", tool=tool_name, error=str(e))
            result = ToolResult(tool=tool_name, ok=False, error=f"tool_error: {str(e)}")
        tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000.0

    # --- DIAGNOSTIC LOG (remove after investigation) ---
    if tool_name == "semantic_retrieval":
        _diag_chunks = list(result.metadata.get("chunks") or []) if result.metadata else []
        logger.warning(
            "DIAG_semantic_retrieval_result",
            ok=result.ok,
            error=result.error,
            chunk_count=len(_diag_chunks),
            output_keys=list((result.output or {}).keys()),
            metadata_keys=list((result.metadata or {}).keys()),
            tool_elapsed_ms=round(tool_elapsed_ms, 2),
        )
    # --- END DIAGNOSTIC ---

    updated_results = [*tool_results, result]
    trace_steps = list(state.get("reasoning_steps") or [])
    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="tool",
            tool=tool_name,
            description=step_call.rationale or "tool_execution",
            input=_sanitize_payload(dict(step_call.input or {})),
            output={
                **_sanitize_payload(dict(result.output or {})),
                "duration_ms": round(tool_elapsed_ms, 2),
            },
            ok=bool(result.ok),
            error=result.error,
        )
    )

    updates: dict[str, object] = {
        "tool_results": updated_results,
        "tool_cursor": cursor + 1,
        "reasoning_steps": trace_steps,
    }
    if result.tool == "semantic_retrieval" and result.metadata:
        chunks = list(result.metadata.get("chunks") or [])
        summaries = list(result.metadata.get("summaries") or [])
        subquery_groups = list(result.metadata.get("subquery_groups") or [])
        retrieval = result.metadata.get("retrieval")
        if chunks or summaries:
            updates["chunks"] = chunks
            updates["summaries"] = summaries
            updates["retrieved_documents"] = [*chunks, *summaries]
        if subquery_groups:
            updates["subquery_groups"] = [
                group for group in subquery_groups if isinstance(group, dict)
            ]
        if isinstance(retrieval, RetrievalDiagnostics):
            updates["retrieval"] = retrieval
    elif result.ok:
        memory = dict(state.get("working_memory") or {})
        memory[result.tool] = dict(result.output or {})
        updates["working_memory"] = memory
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="execute_tool", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    if tool_name:
        updates["tool_timings_ms"] = _append_tool_timing(
            state,
            tool=tool_name,
            elapsed_ms=tool_elapsed_ms,
        )
    return updates


async def reflect_node(state: UniversalState) -> dict[str, object]:
    t0 = time.perf_counter()
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
            "stage_timings_ms": _append_stage_timing(
                state, stage="reflect", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    cursor = int(state.get("tool_cursor") or 0)
    reflections = int(state.get("reflections") or 0)
    max_reflections = min(
        HARD_MAX_REFLECTIONS,
        int(state.get("max_reflections") or DEFAULT_MAX_REFLECTIONS),
    )
    plan_attempts = int(state.get("plan_attempts") or 1)
    tool_results = list(state.get("tool_results") or [])
    last = tool_results[-1] if tool_results else None
    trace_steps = list(state.get("reasoning_steps") or [])

    next_action = "generate"
    stop_reason = str(state.get("stop_reason") or "")
    retry_reason = ""
    retryable = False
    if isinstance(last, ToolResult) and not last.ok:
        retry_reason = str(last.error or "")
        retryable = _is_retryable_reason(retry_reason)
        if retryable and reflections < max_reflections and plan_attempts < MAX_PLAN_ATTEMPTS:
            reflections += 1
            plan_attempts += 1
            next_action = "replan"
        else:
            next_action = "generate"
            if not stop_reason:
                stop_reason = (
                    "tool_error_unrecoverable" if retryable else "tool_error_non_retryable"
                )
    elif isinstance(last, ToolResult) and last.ok and cursor >= len(plan.steps):
        retry_reason = _extract_retry_signal_from_retrieval(state, last)
        retryable = _is_retryable_reason(retry_reason)
        if retryable and reflections < max_reflections and plan_attempts < MAX_PLAN_ATTEMPTS:
            reflections += 1
            plan_attempts += 1
            next_action = "replan"
    elif cursor < len(plan.steps):
        next_action = "execute_tool"
    else:
        next_action = "generate"

    trace_steps.append(
        ReasoningStep(
            index=len(trace_steps) + 1,
            type="reflection",
            description="reflection_decision",
            output={
                "next_action": next_action,
                "plan_attempts": plan_attempts,
                "reflections": reflections,
                "last_tool_ok": bool(last.ok) if isinstance(last, ToolResult) else True,
                "retryable": retryable,
                "retry_reason": (retry_reason[:RETRY_REASON_LIMIT] if retry_reason else ""),
            },
            ok=True,
        )
    )

    logger.warning(
        "DIAG_reflect_decision",
        next_action=next_action,
        retry_reason=retry_reason[:80] if retry_reason else "",
        retryable=retryable,
        reflections=reflections,
        plan_attempts=plan_attempts,
        cursor=cursor,
        plan_steps=len(plan.steps),
        last_tool_ok=bool(last.ok) if isinstance(last, ToolResult) else None,
        last_tool_name=last.tool if isinstance(last, ToolResult) else None,
    )
    updates: dict[str, object] = {
        "next_action": next_action,
        "plan_attempts": plan_attempts,
        "reflections": reflections,
        "reasoning_steps": trace_steps,
    }
    if stop_reason:
        updates["stop_reason"] = stop_reason
    if next_action == "replan":
        reason = retry_reason or (str(last.error) if isinstance(last, ToolResult) else "retry")
        updates["working_query"] = (
            str(state.get("user_query") or "")
            + f"\n\n[REPLAN_REASON] {reason[:RETRY_REASON_LIMIT]}"
        )
    updates["stage_timings_ms"] = _append_stage_timing(
        state, stage="reflect", elapsed_ms=(time.perf_counter() - t0) * 1000.0
    )
    return updates


async def aggregate_subqueries_node(state: UniversalState) -> dict[str, object]:
    t0 = time.perf_counter()
    enabled_by_settings = bool(getattr(settings, "ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", False))
    aggregation_mode = _query_mode_aggregation_mode(state)
    enabled = enabled_by_settings or aggregation_mode == "grouped_map_reduce"
    if not enabled:
        return {
            "stage_timings_ms": _append_stage_timing(
                state, stage="subquery_aggregate", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            )
        }

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
        return {
            "stage_timings_ms": _append_stage_timing(
                state, stage="subquery_aggregate", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            )
        }

    partial_answers: list[dict[str, Any]] = []
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

        snippets = [
            f"{ev.source}: {_clip_text(ev.content, limit=220)}"
            for ev in candidates[:2]
            if str(ev.content or "").strip()
        ]
        partial_answers.append(
            {
                "id": sq_id,
                "query": sq_query,
                "status": "ok",
                "evidence_sources": [str(ev.source or "") for ev in candidates],
                "summary": " | ".join(snippets) if snippets else "Evidencia recuperada.",
            }
        )

    return {
        "partial_answers": partial_answers,
        "stage_timings_ms": _append_stage_timing(
            state, stage="subquery_aggregate", elapsed_ms=(time.perf_counter() - t0) * 1000.0
        ),
    }



# ── Working-memory → EvidenceItem converters ──────────────────────────


def _format_expectation_coverage(data: dict[str, object]) -> list[str]:
    """Format expectation_coverage output (preserves legacy R999 behaviour)."""
    covered = list(data.get("covered") or [])
    missing = list(data.get("missing") or [])
    ratio = data.get("coverage_ratio")
    lines = [
        "[EXPECTATION_COVERAGE]",
        f"coverage_ratio={ratio}",
        f"covered={len(covered)}",
        f"missing={len(missing)}",
    ]
    for row in missing[:6]:
        if not isinstance(row, dict):
            continue
        eid = str(row.get("id") or "expectation")
        risk = str(row.get("missing_risk") or "").strip()
        reason = str(row.get("reason") or "").strip()
        lines.append(f"- missing:{eid} risk={risk} reason={reason}")
    return lines


def _format_logical_comparison(data: dict[str, object]) -> list[str]:
    """Format logical_comparison output as a markdown comparison table."""
    lines = ["[LOGICAL_COMPARISON]"]
    topic = str(data.get("topic") or "").strip()
    if topic:
        lines.append(f"topic={topic}")
    md = str(data.get("comparison_markdown") or "").strip()
    if md:
        lines.append(md)
    else:
        rows = list(data.get("rows") or [])
        if rows:
            lines.append("| Scope | Evidence |")
            lines.append("|---|---|")
            for row in rows[:12]:
                if isinstance(row, dict):
                    lines.append(f"| {row.get('scope', '')} | {row.get('evidence', '')} |")
    # LLM-enriched fields
    llm_analysis = str(data.get("llm_analysis") or "").strip()
    if llm_analysis:
        lines.append(f"analysis={llm_analysis}")
    for key in ("commonalities", "differences", "gaps"):
        items = data.get(key)
        if isinstance(items, list) and items:
            lines.append(f"{key}={', '.join(str(i) for i in items[:6])}")
    return lines


def _format_structural_extraction(data: dict[str, object]) -> list[str]:
    """Format structural_extraction output as a record listing."""
    lines = ["[STRUCTURAL_EXTRACTION]"]
    schema_def = str(data.get("schema_definition") or "").strip()
    if schema_def:
        lines.append(f"schema={schema_def}")
    records = list(data.get("records") or [])
    lines.append(f"record_count={len(records)}")
    for rec in records[:20]:
        if isinstance(rec, dict):
            label = str(rec.get("label") or "").strip()
            value = rec.get("value", "")
            unit = str(rec.get("unit") or "").strip()
            src = str(rec.get("source") or "").strip()
            suffix = f" [{src}]" if src else ""
            lines.append(f"- {label}: {value} {unit}{suffix}")
    # LLM-enriched: tables
    tables = data.get("tables")
    if isinstance(tables, list) and tables:
        lines.append(f"tables_count={len(tables)}")
        for tbl in tables[:5]:
            if isinstance(tbl, dict):
                title = str(tbl.get("title") or "").strip()
                lines.append(f"table: {title}")
    # LLM-enriched: key-value pairs
    kvs = data.get("key_values")
    if isinstance(kvs, list) and kvs:
        for kv in kvs[:10]:
            if isinstance(kv, dict):
                lines.append(f"kv: {kv.get('key', '')}={kv.get('value', '')}")
    return lines


def _format_generic_tool(tool_name: str, data: dict[str, object]) -> list[str]:
    """Fallback formatter for unknown tools."""
    lines = [f"[TOOL_{tool_name.upper()}]"]
    for key, value in list(data.items())[:10]:
        text = str(value or "")
        if len(text) > 200:
            text = text[:200] + "..."
        lines.append(f"{key}={text}")
    return lines


_TOOL_FORMATTERS: dict[str, object] = {
    "expectation_coverage": _format_expectation_coverage,
    "logical_comparison": _format_logical_comparison,
    "structural_extraction": _format_structural_extraction,
}


def _working_memory_to_evidence(
    working_memory: dict[str, object],
) -> list[EvidenceItem]:
    """Convert all working_memory tool outputs into synthetic EvidenceItems."""
    items: list[EvidenceItem] = []
    for tool_name, tool_output in working_memory.items():
        if not isinstance(tool_output, dict):
            continue
        formatter = _TOOL_FORMATTERS.get(tool_name, _format_generic_tool)
        if formatter is _format_generic_tool:
            lines = _format_generic_tool(tool_name, tool_output)
        else:
            lines = formatter(tool_output)  # type: ignore[operator]
        if lines:
            items.append(
                EvidenceItem(
                    source=f"TOOL_{tool_name.upper()}",
                    content="\n".join(lines),
                    score=1.0,
                    metadata={"synthetic": True, "tool": tool_name},
                )
            )
    return items


async def generator_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
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
            "stage_timings_ms": _append_stage_timing(
                state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }
    chunks = cast(list[EvidenceItem], list(state.get("chunks") or []))
    summaries = cast(list[EvidenceItem], list(state.get("summaries") or []))
    raw_partials = state.get("partial_answers")
    partial_answers_list: list[Any] = raw_partials if isinstance(raw_partials, list) else []

    if partial_answers_list:
        synthesized_summaries: list[EvidenceItem] = []
        for idx, partial in enumerate(partial_answers_list[:8], start=1):
            if not isinstance(partial, dict):
                continue
            sq_id = str(partial.get("id") or f"q{idx}").strip() or f"q{idx}"
            sq_query = str(partial.get("query") or "").strip()
            status = str(partial.get("status") or "unknown").strip()
            summary = str(partial.get("summary") or "").strip()
            raw_sources = partial.get("evidence_sources")
            sources_list: list[Any] = raw_sources if isinstance(raw_sources, list) else []
            source_list = ", ".join(str(s) for s in sources_list if str(s).strip())
            content = (
                f"[SUBQUERY {sq_id}] query={sq_query}\n"
                f"status={status}\n"
                f"sources={source_list or 'none'}\n"
                f"summary={summary or 'Sin resumen parcial.'}"
            )
            synthesized_summaries.append(
                EvidenceItem(
                    source=f"RMAP{idx}",
                    content=content,
                    score=1.0,
                    metadata={"row": {"content": content, "metadata": {"subquery_id": sq_id}}},
                )
            )
        if synthesized_summaries:
            summaries = [*synthesized_summaries, *summaries]

    working_memory = dict(state.get("working_memory") or {})
    synthetic_evidence = _working_memory_to_evidence(working_memory)
    if synthetic_evidence:
        summaries = [*summaries, *synthetic_evidence]

    try:
        generator_timeout_ms = get_adaptive_timeout_ms(
            state,
            stage_default_ms=_timeout_ms_for_stage("generator"),
            headroom_ms=1000,
        )
        answer = await asyncio.wait_for(
            components.answer_generator.generate(
                query=str(state.get("user_query") or ""),
                scope_label=str(state.get("scope_label") or ""),
                plan=plan,
                chunks=chunks,
                summaries=summaries,
                agent_profile=state.get("agent_profile"),
            ),
            timeout=generator_timeout_ms / 1000.0,
        )
    except TimeoutError:
        return {
            "stop_reason": "generator_timeout",
            "stage_timings_ms": _append_stage_timing(
                state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }
    trace_steps = list(state.get("reasoning_steps") or [])
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
        "stage_timings_ms": _append_stage_timing(
            state, stage="generator", elapsed_ms=(time.perf_counter() - t0) * 1000.0
        ),
    }


async def citation_validate_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    t0 = time.perf_counter()
    answer = state.get("generation")
    plan = state.get("retrieval_plan")
    if not isinstance(answer, AnswerDraft) or not isinstance(plan, RetrievalPlan):
        return {
            "validation": ValidationResult(accepted=False, issues=["missing_generation_or_plan"]),
            "stop_reason": "validation_failed",
            "stage_timings_ms": _append_stage_timing(
                state, stage="validation", elapsed_ms=(time.perf_counter() - t0) * 1000.0
            ),
        }

    allowed = list(state.get("allowed_tools") or [])
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

    trace_steps = list(state.get("reasoning_steps") or [])
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
    stop_reason = str(state.get("stop_reason") or "")
    if not stop_reason:
        stop_reason = "done" if validation.accepted else "validation_failed"
    return {
        "validation": validation,
        "reasoning_steps": trace_steps,
        "stop_reason": stop_reason,
        "stage_timings_ms": _append_stage_timing(
            state, stage="validation", elapsed_ms=(time.perf_counter() - t0) * 1000.0
        ),
    }


def route_after_planner(state: UniversalState) -> str:
    return "execute" if str(state.get("next_action") or "") == "execute" else "generate"


def route_after_reflect(state: UniversalState) -> str:
    next_action = str(state.get("next_action") or "")
    if next_action == "replan":
        return "replan"
    if next_action == "execute_tool":
        return "execute_tool"
    return "generate"
