from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.agent.types.models import AnswerDraft, ValidationResult
from app.infrastructure.config import settings
from app.graph.state import UniversalState
from app.graph.logic.logic import _count_section_markers
from app.graph.logic.utils import _effective_execute_tool_timeout_ms


def build_reasoning_trace(state: UniversalState) -> dict[str, Any]:
    steps = [asdict(step) for step in list(state.get("reasoning_steps") or [])]
    tools_used = sorted(
        {
            str(step.get("tool") or "")
            for step in steps
            if isinstance(step, dict) and step.get("tool")
        }
    )
    accepted = None
    validation = state.get("validation")
    if isinstance(validation, ValidationResult):
        accepted = bool(validation.accepted)
    generation = state.get("generation")
    answer_text = generation.text if isinstance(generation, AnswerDraft) else ""
    response_sections = [
        "hechos citados",
        "inferencias",
        "brechas",
        "recomendaciones",
        "confianza y supuestos",
    ]
    expectation_coverage_ratio = None
    missing_expectations = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("tool") or "") != "expectation_coverage":
            continue
        output = step.get("output")
        if not isinstance(output, dict):
            continue
        if "coverage_ratio" in output:
            expectation_coverage_ratio = output.get("coverage_ratio")
        missing_raw = output.get("missing")
        if isinstance(missing_raw, list):
            missing_expectations = len(missing_raw)
            
    return {
        "engine": "universal_flow",
        "stop_reason": str(state.get("stop_reason") or "unknown"),
        "plan_attempts": int(state.get("plan_attempts") or 1),
        "reflections": int(state.get("reflections") or 0),
        "tools_used": tools_used,
        "steps": steps,
        "stage_timings_ms": dict(state.get("stage_timings_ms") or {}),
        "tool_timings_ms": dict(state.get("tool_timings_ms") or {}),
        "stage_budgets_ms": {
            "planner": int(getattr(settings, "ORCH_TIMEOUT_PLAN_MS", 3000) or 3000)
            + int(getattr(settings, "ORCH_TIMEOUT_CLASSIFY_MS", 2000) or 2000),
            "execute_tool": _effective_execute_tool_timeout_ms("semantic_retrieval"),
            "generator": int(getattr(settings, "ORCH_TIMEOUT_GENERATE_MS", 15000) or 15000),
            "validation": int(getattr(settings, "ORCH_TIMEOUT_VALIDATE_MS", 5000) or 5000),
            "total": int(getattr(settings, "ORCH_TIMEOUT_TOTAL_MS", 60000) or 60000),
            "is_adaptive": True,
        },
        "final_confidence": (1.0 if accepted else 0.45 if accepted is False else None),
        "response_sections_detected": _count_section_markers(answer_text, response_sections),
        "expectation_coverage_ratio": expectation_coverage_ratio,
        "missing_expectations": missing_expectations,
    }
