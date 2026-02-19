from __future__ import annotations

import re
from functools import lru_cache

from app.agent.models import (
    QueryIntent,
    ReasoningPlan,
    ReasoningStep,
    RetrievalPlan,
    ToolCall,
)
from app.agent.policies import build_retrieval_plan, classify_intent
from app.cartridges.models import AgentProfile, QueryModeConfig


_ARITHMETIC_PATTERN = re.compile(r"\d+\s*[\+\-\*/]\s*\d+")
_CLAUSE_REFERENCE_PATTERN = re.compile(r"\b\d+(?:\.\d+)+\b")


@lru_cache(maxsize=64)
def _compile_patterns(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        value = str(pattern or "").strip()
        if not value:
            continue
        try:
            compiled.append(re.compile(value, re.IGNORECASE))
        except re.error:
            continue
    return tuple(compiled)


def _profile_patterns(profile: AgentProfile | None, field_name: str) -> tuple[re.Pattern[str], ...]:
    if profile is None:
        return ()
    raw = getattr(profile.router, field_name, [])
    if not isinstance(raw, list):
        return ()
    return _compile_patterns(tuple(str(item) for item in raw if str(item).strip()))


def _matches_any(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _count_reference_matches(text: str, profile: AgentProfile | None) -> int:
    profile_refs = _profile_patterns(profile, "reference_patterns")
    if profile_refs:
        return sum(len(pattern.findall(text)) for pattern in profile_refs)
    return len(_CLAUSE_REFERENCE_PATTERN.findall(text))


def _is_complex_query(query: str, intent: QueryIntent, profile: AgentProfile | None) -> bool:
    text = query or ""
    if intent.mode in {"comparativa"}:
        return True
    if _matches_any(text, _profile_patterns(profile, "complexity_patterns")):
        return True
    if _count_reference_matches(text, profile) >= 2:
        return True
    return False


def _needs_extraction(query: str, profile: AgentProfile | None) -> bool:
    return _matches_any(query or "", _profile_patterns(profile, "extraction_patterns"))


def _needs_calculation(query: str, profile: AgentProfile | None) -> bool:
    text = query or ""
    has_math = bool(_ARITHMETIC_PATTERN.search(text))
    procedural_math = _matches_any(text, _profile_patterns(profile, "calculation_patterns"))
    return has_math or procedural_math


def default_tool_input(tool: str, query: str, mode: str) -> dict[str, str]:
    if tool == "semantic_retrieval":
        return {"query": query}
    if tool == "logical_comparison":
        return {"topic": query}
    if tool == "structural_extraction":
        return {"schema_definition": "entity, value, unit"}
    if tool == "expectation_coverage":
        return {"mode": mode}
    return {}


def build_universal_plan(
    *,
    query: str,
    profile: AgentProfile | None,
    allowed_tools: list[str],
) -> tuple[QueryIntent, RetrievalPlan, ReasoningPlan, list[ReasoningStep]]:
    intent = classify_intent(query, profile=profile)
    retrieval_plan = build_retrieval_plan(intent, query=query, profile=profile)
    complexity = "complex" if _is_complex_query(query, intent, profile) else "simple"
    allowed_tool_set = set(allowed_tools)
    mode_tool_hints: set[str] = set()
    mode_execution_plan: list[str] = []
    if profile is not None:
        mode_config = profile.query_modes.modes.get(str(intent.mode))
        if isinstance(mode_config, QueryModeConfig):
            mode_tool_hints = set(mode_config.tool_hints)
            mode_execution_plan = [
                str(tool) for tool in mode_config.execution_plan if str(tool).strip()
            ]
    steps: list[ToolCall] = []

    def _append_unique_step(tool: str, rationale: str) -> None:
        if tool not in allowed_tool_set:
            return
        if any(item.tool == tool for item in steps):
            return
        steps.append(
            ToolCall(
                tool=tool,
                input=default_tool_input(tool, query, str(intent.mode)),
                rationale=rationale,
            )
        )

    if mode_execution_plan:
        seen: set[str] = set()
        for tool in mode_execution_plan:
            if tool in seen or tool not in allowed_tool_set:
                continue
            seen.add(tool)
            _append_unique_step(tool, "mode_execution_plan")

    if not steps and "semantic_retrieval" in allowed_tool_set:
        _append_unique_step("semantic_retrieval", "retrieve_grounding")

    if (
        complexity == "complex"
        and "logical_comparison" in allowed_tool_set
        and "logical_comparison" in mode_tool_hints
    ):
        _append_unique_step("logical_comparison", "cross_scope_relation")

    if "structural_extraction" in allowed_tool_set and _needs_extraction(query, profile):
        _append_unique_step("structural_extraction", "extract_structured_data")

    if "python_calculator" in allowed_tool_set and _needs_calculation(query, profile):
        _append_unique_step("python_calculator", "deterministic_numeric_check")

    if not steps and "semantic_retrieval" in allowed_tool_set:
        _append_unique_step("semantic_retrieval", "default_retrieval")

    trace_steps = [
        ReasoningStep(
            index=1,
            type="plan",
            description="universal_plan_generated",
            output={
                "intent_mode": intent.mode,
                "complexity": complexity,
                "tool_sequence": [item.tool for item in steps],
            },
        )
    ]
    return (
        intent,
        retrieval_plan,
        ReasoningPlan(goal=query, steps=steps, complexity=complexity),  # type: ignore[arg-type]
        trace_steps,
    )
