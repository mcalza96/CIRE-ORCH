from __future__ import annotations

import re

from app.agent.models import (
    QueryIntent,
    ReasoningPlan,
    ReasoningStep,
    RetrievalPlan,
    ToolCall,
)
from app.agent.policies import build_retrieval_plan, classify_intent
from app.cartridges.models import AgentProfile


MAX_SIMPLE_QUERY_LENGTH = 180

_ARITHMETIC_PATTERN = re.compile(r"\d+\s*[\+\-\*/]\s*\d+")
_COMPLEX_HINT_PATTERN = re.compile(
    r"\b(?:analiza|relacion|relación|impact|compara)\b",
    re.IGNORECASE,
)
_EXTRACTION_HINT_PATTERN = re.compile(
    r"\b(?:extrae|extraer|estructura|json|tabla|reactivo|insumo|cantidad|bom)\b",
    re.IGNORECASE,
)
_CALCULATION_HINT_PATTERN = re.compile(
    r"\b(?:calcula|calcular|cuanto|cuánto|formula|lote|muestras)\b",
    re.IGNORECASE,
)


def _is_complex_query(query: str, intent: QueryIntent) -> bool:
    text = query or ""
    if intent.mode in {"comparativa"}:
        return True
    if len(text) > MAX_SIMPLE_QUERY_LENGTH:
        return True
    if _COMPLEX_HINT_PATTERN.search(text):
        return True
    return False


def _needs_extraction(query: str) -> bool:
    return bool(_EXTRACTION_HINT_PATTERN.search(query or ""))


def _needs_calculation(query: str) -> bool:
    text = query or ""
    has_math = bool(_ARITHMETIC_PATTERN.search(text))
    procedural_math = bool(_CALCULATION_HINT_PATTERN.search(text))
    return has_math or procedural_math


def build_universal_plan(
    *,
    query: str,
    profile: AgentProfile | None,
    allowed_tools: list[str],
) -> tuple[QueryIntent, RetrievalPlan, ReasoningPlan, list[ReasoningStep]]:
    intent = classify_intent(query, profile=profile)
    retrieval_plan = build_retrieval_plan(intent, query=query, profile=profile)
    complexity = "complex" if _is_complex_query(query, intent) else "simple"
    allowed_tool_set = set(allowed_tools)
    steps: list[ToolCall] = []

    if "semantic_retrieval" in allowed_tool_set:
        steps.append(
            ToolCall(
                tool="semantic_retrieval",
                input={"query": query},
                rationale="retrieve_grounding",
            )
        )

    if (
        complexity == "complex"
        and "logical_comparison" in allowed_tool_set
        and intent.mode == "comparativa"
    ):
        steps.append(
            ToolCall(
                tool="logical_comparison",
                input={"topic": query},
                rationale="cross_scope_relation",
            )
        )

    if "structural_extraction" in allowed_tool_set and _needs_extraction(query):
        steps.append(
            ToolCall(
                tool="structural_extraction",
                input={"schema_definition": "entity, value, unit"},
                rationale="extract_structured_data",
            )
        )

    if "python_calculator" in allowed_tool_set and _needs_calculation(query):
        steps.append(
            ToolCall(
                tool="python_calculator",
                input={},
                rationale="deterministic_numeric_check",
            )
        )

    if not steps and "semantic_retrieval" in allowed_tool_set:
        steps = [
            ToolCall(
                tool="semantic_retrieval", input={"query": query}, rationale="default_retrieval"
            )
        ]

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
