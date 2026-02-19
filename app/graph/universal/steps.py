from __future__ import annotations

import asyncio
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
from app.graph.universal.interaction import decide_interaction
from app.graph.universal.clarification_llm import build_clarification_with_llm
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
    state_get_list,
    state_get_dict,
    state_get_int,
    state_get_str,
    track_node_timing,
    _append_stage_timing,
    _append_tool_timing,
    _clip_text,
    _effective_execute_tool_timeout_ms,
    _keyword_overlap_score,
    _sanitize_payload,
    _timeout_ms_for_stage,
    get_adaptive_timeout_ms,
)
from app.graph.universal.logic import (
    _extract_retry_signal_from_retrieval,
    _infer_expression_from_query,
    _is_retryable_reason,
    _query_mode_aggregation_mode,
)

logger = structlog.get_logger(__name__)


class OrchestratorComponents(Protocol):
    retriever: RetrieverPort
    answer_generator: AnswerGeneratorPort
    validator: ValidatorPort
    tools: dict[str, AgentTool] | None

    def _runtime_context(self) -> ToolRuntimeContext: ...


@track_node_timing("planner")
async def planner_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
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
    existing_steps = state_get_list(state, "reasoning_steps")
    prior_interruptions = state_get_int(state, "interaction_interruptions", 0)
    
    clarification_context = state.get("clarification_context")
    if isinstance(clarification_context, dict):
        ans_text = str(clarification_context.get("answer_text") or "").strip()
        req_scopes = clarification_context.get("requested_scopes")
        missing_slots = clarification_context.get("missing_slots")
        if ans_text and isinstance(missing_slots, list) and missing_slots:
            from app.graph.universal.clarification_llm import extract_clarification_slots_with_llm
            extracted_slots = await extract_clarification_slots_with_llm(
                clarification_text=ans_text,
                original_query=query,
                missing_slots=missing_slots,
            )
            if isinstance(extracted_slots, dict):
                for slot_name, slot_values in extracted_slots.items():
                    if not isinstance(slot_values, list) or not slot_values:
                        continue
                    # Map the conceptual 'scope' slot to the internal key 'requested_scopes'
                    storage_key = "requested_scopes" if slot_name == "scope" else slot_name
                    clarification_context[storage_key] = slot_values

    interaction = decide_interaction(
        query=query,
        intent=intent,
        retrieval_plan=retrieval_plan,
        reasoning_plan=reasoning_plan,
        profile=profile if isinstance(profile, AgentProfile) else None,
        prior_interruptions=prior_interruptions,
        clarification_context=clarification_context,
    )

    if isinstance(clarification_context, dict):
        context_scopes = clarification_context.get("requested_scopes")
        if isinstance(context_scopes, list) and context_scopes:
            from dataclasses import replace
            merged = list(retrieval_plan.requested_standards)
            for s in context_scopes:
                if s not in merged:
                    merged.append(s)
            retrieval_plan = replace(retrieval_plan, requested_standards=tuple(merged))

        plan_feedback = clarification_context.get("plan_feedback")
        if isinstance(plan_feedback, str) and plan_feedback.strip() and prior_interruptions > 0:
            from app.graph.universal.clarification_llm import rewrite_plan_with_feedback_llm
            from app.graph.universal.planning import default_tool_input
            from app.agent.models import ToolCall
            from dataclasses import replace
            
            replan_payload = await rewrite_plan_with_feedback_llm(
                current_tools=[call.tool for call in reasoning_plan.steps],
                feedback=plan_feedback.replace("_", " "),  # Decode safe string format
                allowed_tools=allowed_tools
            )
            if isinstance(replan_payload, dict):
                new_tools_raw = replan_payload.get("new_plan")
                if isinstance(new_tools_raw, list) and new_tools_raw:
                    dynamic_inputs = replan_payload.get("dynamic_inputs") or {}
                    new_steps: list[ToolCall] = []
                    for tool_name in new_tools_raw:
                        tool_str = str(tool_name).strip()
                        if tool_str in allowed_tools:
                            base_input = default_tool_input(tool_str, query, str(intent.mode))
                            dyn_in = dynamic_inputs.get(tool_str)
                            if isinstance(dyn_in, dict):
                                import json
                                for k, v in dyn_in.items():
                                    base_input[str(k)] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                            new_steps.append(ToolCall(tool=tool_str, input=base_input, rationale="user_feedback_override"))
                    if new_steps:
                        reasoning_plan = replace(reasoning_plan, steps=new_steps)

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
        "interaction_level": interaction.level,
        "interaction_metrics": dict(interaction.metrics),
        "interaction_interruptions": prior_interruptions,
    }

    if interaction.needs_interrupt:
        interrupt_question = interaction.question
        interrupt_options = list(interaction.options)
        interaction_metrics = dict(interaction.metrics)
        interaction_metrics.setdefault("clarification_model_used", "heuristic")
        interaction_metrics.setdefault("clarification_confidence", 0.0)
        interaction_metrics.setdefault("slots_filled", 0)
        interaction_metrics.setdefault("loop_prevented", False)
        interaction_metrics.setdefault("clarification_expected_answer", "")

        if interaction.kind == "clarification":
            llm_payload = None
            # Do not override deterministic guided reprompts (anti-loop path).
            if not bool(interaction.metrics.get("guided_reprompt", False)):
                llm_payload = await build_clarification_with_llm(
                    query=query,
                    current_question=interaction.question,
                    current_options=interaction.options,
                    missing_slots=list(interaction.missing_slots),
                    scope_candidates=interaction.scope_candidates,
                    interaction_metrics=interaction_metrics,
                )
            if isinstance(llm_payload, dict):
                llm_question = str(llm_payload.get("question") or "").strip()
                llm_options_raw = llm_payload.get("options")
                llm_options = (
                    [str(opt).strip() for opt in llm_options_raw if str(opt).strip()]
                    if isinstance(llm_options_raw, list)
                    else []
                )
                if llm_question:
                    interrupt_question = llm_question
                if llm_options:
                    interrupt_options = llm_options
                interaction_metrics["clarification_expected_answer"] = str(
                    llm_payload.get("expected_answer") or ""
                ).strip()
                interaction_metrics["clarification_model_used"] = str(
                    llm_payload.get("model") or "llm"
                )
                interaction_metrics["clarification_confidence"] = float(
                    llm_payload.get("confidence") or 0.0
                )
        updates["next_action"] = "interrupt"
        updates["stop_reason"] = f"awaiting_{interaction.kind}"
        updates["interaction_metrics"] = interaction_metrics
        updates["clarification_request"] = {
            "question": interrupt_question,
            "options": interrupt_options,
            "kind": interaction.kind,
            "level": interaction.level,
            "missing_slots": list(interaction.missing_slots),
            "expected_answer": str(interaction_metrics.get("clarification_expected_answer") or ""),
        }
        updates["interaction_interruptions"] = prior_interruptions + 1
        updates["reasoning_steps"] = [
            *existing_steps,
            *trace_steps,
            ReasoningStep(
                index=len(existing_steps) + len(trace_steps) + 1,
                type="plan",
                description="interaction_interrupt",
                output={
                    "level": interaction.level,
                    "kind": interaction.kind,
                    "metrics": dict(interaction.metrics),
                },
            ),
        ]

    return updates


@track_node_timing("execute_tool")
async def execute_tool_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
        }

    cursor = state_get_int(state, "tool_cursor", 0)
    if cursor >= len(plan.steps):
        return {
            "next_action": "generate",
        }

    max_steps = min(HARD_MAX_STEPS, int(state.get("max_steps") or DEFAULT_MAX_STEPS))
    tool_results = state_get_list(state, "tool_results")
    if len(tool_results) >= max_steps:
        return {
            "next_action": "generate",
            "stop_reason": "max_steps_reached",
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

        # ── Full Context Piping ──
        working_memory = dict(state_get_dict(state, "working_memory"))
        if working_memory:
            payload["working_memory"] = working_memory

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
    trace_steps = state_get_list(state, "reasoning_steps")
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
        memory = dict(state_get_dict(state, "working_memory"))
        memory[result.tool] = dict(result.output or {})
        updates["working_memory"] = memory
    if tool_name:
        updates["tool_timings_ms"] = _append_tool_timing(
            state,
            tool=tool_name,
            elapsed_ms=tool_elapsed_ms,
        )
    return updates


@track_node_timing("reflect")
async def reflect_node(state: UniversalState) -> dict[str, object]:
    plan = state.get("reasoning_plan")
    if not isinstance(plan, ReasoningPlan):
        return {
            "next_action": "generate",
            "stop_reason": "missing_plan",
        }

    cursor = state_get_int(state, "tool_cursor", 0)
    reflections = state_get_int(state, "reflections", 0)
    max_reflections = min(
        HARD_MAX_REFLECTIONS,
        int(state.get("max_reflections") or DEFAULT_MAX_REFLECTIONS),
    )
    plan_attempts = int(state.get("plan_attempts") or 1)
    tool_results = state_get_list(state, "tool_results")
    last = tool_results[-1] if tool_results else None
    trace_steps = state_get_list(state, "reasoning_steps")

    next_action = "generate"
    stop_reason = state_get_str(state, "stop_reason", "")
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
    return updates


@track_node_timing("subquery_aggregate")
async def aggregate_subqueries_node(
    state: UniversalState, components: OrchestratorComponents | None = None
) -> dict[str, object]:
    enabled_by_settings = bool(getattr(settings, "ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", False))
    aggregation_mode = _query_mode_aggregation_mode(state)
    enabled = enabled_by_settings or aggregation_mode == "grouped_map_reduce"
    if not enabled:
        return {
            
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
            
        }

    async def _summarize_one(
        comp: OrchestratorComponents,
        q: str,
        ev: list[EvidenceItem],
        prof: AgentProfile | None,
    ) -> str:
        sub_plan = RetrievalPlan(
            mode="concisa_y_directa",
            chunk_k=len(ev),
            chunk_fetch_k=len(ev),
            summary_k=0,
            require_literal_evidence=False,
        )
        draft = await comp.answer_generator.generate(
            query=f"[SUBCONSULTA: {q}]\nResume la respuesta basandote SOLO en los fragmentos proporcionados.",
            scope_label="",
            plan=sub_plan,
            chunks=ev,
            summaries=[],
            agent_profile=prof,
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
            # Empty fallback logic remains same
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


# ── Working-memory → EvidenceItem converters ──────────────────────────


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

    # [MODIFIED] Removed _working_memory_to_evidence conversion.
    # working_memory is now passed directly to the generator as structured context.

    try:
        generator_timeout_ms = get_adaptive_timeout_ms(
            state,
            stage_default_ms=_timeout_ms_for_stage("generator"),
            headroom_ms=1000,
        )
        answer = await asyncio.wait_for(
            components.answer_generator.generate(
                query = state_get_str(state, "user_query", ""),
                scope_label = state_get_str(state, "scope_label", ""),
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


def route_after_planner(state: UniversalState) -> str:
    next_action = state_get_str(state, "next_action", "")
    if next_action == "interrupt":
        return "interrupt"
    return "execute" if next_action == "execute" else "generate"


def route_after_reflect(state: UniversalState) -> str:
    next_action = state_get_str(state, "next_action", "")
    if next_action == "replan":
        return "replan"
    if next_action == "execute_tool":
        return "execute_tool"
    return "generate"
