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


def _is_complex_query(query: str, intent: QueryIntent) -> bool:
    text = (query or "").lower()
    if intent.mode in {"comparativa"}:
        return True
    if len(text) > 180:
        return True
    if any(token in text for token in ("analiza", "relacion", "relación", "impact", "compara")):
        return True
    return False


def _needs_extraction(query: str) -> bool:
    text = (query or "").lower()
    return any(
        token in text
        for token in (
            "extrae",
            "extraer",
            "estructura",
            "json",
            "tabla",
            "reactivo",
            "insumo",
            "cantidad",
            "bom",
        )
    )


def _needs_calculation(query: str) -> bool:
    text = (query or "").lower()
    has_math = bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text))
    procedural_math = any(
        token in text
        for token in (
            "calcula",
            "calcular",
            "cuanto",
            "cuánto",
            "formula",
            "lote",
            "muestras",
        )
    )
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
    steps: list[ToolCall] = []

    if "semantic_retrieval" in allowed_tools:
        steps.append(
            ToolCall(
                tool="semantic_retrieval",
                input={"query": query},
                rationale="retrieve_grounding",
            )
        )

    if complexity == "complex" and "logical_comparison" in allowed_tools and intent.mode == "comparativa":
        steps.append(
            ToolCall(
                tool="logical_comparison",
                input={"topic": query},
                rationale="cross_scope_relation",
            )
        )

    if "structural_extraction" in allowed_tools and _needs_extraction(query):
        steps.append(
            ToolCall(
                tool="structural_extraction",
                input={"schema_definition": "entity, value, unit"},
                rationale="extract_structured_data",
            )
        )

    if "python_calculator" in allowed_tools and _needs_calculation(query):
        steps.append(
            ToolCall(
                tool="python_calculator",
                input={},
                rationale="deterministic_numeric_check",
            )
        )

    if not steps and "semantic_retrieval" in allowed_tools:
        steps = [ToolCall(tool="semantic_retrieval", input={"query": query}, rationale="default_retrieval")]

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
