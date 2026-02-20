from __future__ import annotations

import asyncio

from app.agent.models import ReasoningPlan, ReasoningStep
from app.agent.tools import resolve_allowed_tools
from app.cartridges.models import AgentProfile
from app.core.config import settings
from app.graph.universal.clarification_llm import build_clarification_with_llm
from app.graph.universal.interaction import decide_interaction
from app.graph.universal.planning import build_universal_plan, default_tool_input
from app.graph.universal.state import (
    DEFAULT_MAX_REFLECTIONS,
    DEFAULT_MAX_STEPS,
    HARD_MAX_REFLECTIONS,
    HARD_MAX_STEPS,
    UniversalState,
)
from app.graph.universal.utils import (
    get_adaptive_timeout_ms,
    state_get_int,
    state_get_list,
    track_node_timing,
)

from .types import OrchestratorComponents


@track_node_timing("planner")
async def planner_node(
    state: UniversalState, components: OrchestratorComponents
) -> dict[str, object]:
    query = str(state.get("working_query") or state.get("user_query") or "").strip()
    profile = state.get("agent_profile")
    allowed_tools = resolve_allowed_tools(profile, components.tools or {})

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
            for scope in context_scopes:
                if scope not in merged:
                    merged.append(scope)
            retrieval_plan = replace(retrieval_plan, requested_standards=tuple(merged))

        plan_feedback = clarification_context.get("plan_feedback")
        if isinstance(plan_feedback, str) and plan_feedback.strip() and prior_interruptions > 0:
            from app.agent.models import ToolCall
            from app.graph.universal.clarification_llm import rewrite_plan_with_feedback_llm
            from dataclasses import replace
            import json

            replan_payload = await rewrite_plan_with_feedback_llm(
                current_tools=[call.tool for call in reasoning_plan.steps],
                feedback=plan_feedback.replace("_", " "),
                allowed_tools=allowed_tools,
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
                                for key, value in dyn_in.items():
                                    base_input[str(key)] = (
                                        json.dumps(value)
                                        if isinstance(value, (dict, list))
                                        else str(value)
                                    )
                            new_steps.append(
                                ToolCall(
                                    tool=tool_str,
                                    input=base_input,
                                    rationale="user_feedback_override",
                                )
                            )
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
        "tool_results": [],
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
