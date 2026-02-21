from __future__ import annotations

import re
import time
from typing import Any

from app.infrastructure.config import settings
from app.graph.universal.state import UniversalState


def _non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return default
        return max(0, parsed)
    if value is None:
        return default
    return default


def _append_stage_timing(
    state: UniversalState | dict[str, object],
    *,
    stage: str,
    elapsed_ms: float,
) -> dict[str, float]:
    current = state.get("stage_timings_ms")
    timings = dict(current) if isinstance(current, dict) else {}
    timings[stage] = round(float(timings.get(stage, 0.0)) + max(0.0, elapsed_ms), 2)
    return timings


def _append_tool_timing(
    state: UniversalState | dict[str, object],
    *,
    tool: str,
    elapsed_ms: float,
) -> dict[str, float]:
    current = state.get("tool_timings_ms")
    timings = dict(current) if isinstance(current, dict) else {}
    timings[tool] = round(float(timings.get(tool, 0.0)) + max(0.0, elapsed_ms), 2)
    return timings


def _timeout_ms_for_stage(stage: str) -> int:
    mapping = {
        "planner": int(getattr(settings, "ORCH_TIMEOUT_PLAN_MS", 3000) or 3000),
        "execute_tool": int(getattr(settings, "ORCH_TIMEOUT_EXECUTE_TOOL_MS", 30000) or 30000),
        "generator": int(getattr(settings, "ORCH_TIMEOUT_GENERATE_MS", 15000) or 15000),
        "validation": int(getattr(settings, "ORCH_TIMEOUT_VALIDATE_MS", 5000) or 5000),
    }
    return max(25, mapping.get(stage, 1000))


def _effective_execute_tool_timeout_ms(tool_name: str) -> int:
    base_timeout_ms = _timeout_ms_for_stage("execute_tool")
    if str(tool_name or "").strip() != "semantic_retrieval":
        return base_timeout_ms

    # Legacy contract does not need extended timeout
    if getattr(settings, "ORCH_RETRIEVAL_CONTRACT", "advanced") == "legacy":
        return base_timeout_ms

    retrieval_timeout_ms = max(
        int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_HYBRID_MS", 25000) or 25000),
        int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_MULTI_QUERY_MS", 25000) or 25000),
        int(getattr(settings, "ORCH_TIMEOUT_RETRIEVAL_COVERAGE_REPAIR_MS", 15000) or 15000),
    )

    total_timeout_ms = int(getattr(settings, "ORCH_TIMEOUT_TOTAL_MS", 60000) or 60000)
    planner_timeout_ms = int(getattr(settings, "ORCH_TIMEOUT_PLAN_MS", 3000) or 3000) + int(
        getattr(settings, "ORCH_TIMEOUT_CLASSIFY_MS", 2000) or 2000
    )
    generator_timeout_ms = int(getattr(settings, "ORCH_TIMEOUT_GENERATE_MS", 15000) or 15000)
    validation_timeout_ms = int(getattr(settings, "ORCH_TIMEOUT_VALIDATE_MS", 5000) or 5000)
    minimum_tail_headroom_ms = max(
        400,
        planner_timeout_ms + generator_timeout_ms + validation_timeout_ms + 300,
    )
    max_timeout_by_total_ms = max(base_timeout_ms, total_timeout_ms - minimum_tail_headroom_ms)
    return max(base_timeout_ms, min(retrieval_timeout_ms, max_timeout_by_total_ms))


def _clip_text(value: object, limit: int = 280) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _sanitize_payload(payload: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            out[key] = _clip_text(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, dict):
            out[key] = {str(k): _clip_text(v) for k, v in value.items()}
        else:
            out[key] = _clip_text(value)
    return out


def get_adaptive_timeout_ms(
    state: UniversalState,
    stage_default_ms: int,
    headroom_ms: int = 500,
) -> int:
    """Calculate the remaining time for the current stage, ensuring we stay within total budget."""
    start_pc = state.get("flow_start_pc")
    if start_pc is None:
        return stage_default_ms

    total_budget_ms = int(getattr(settings, "ORCH_TIMEOUT_TOTAL_MS", 150000) or 150000)
    elapsed_ms = (time.perf_counter() - start_pc) * 1000.0
    remaining_ms = total_budget_ms - elapsed_ms - headroom_ms

    # We return the lesser of the stage default or the actual remaining time
    return max(25, int(min(float(stage_default_ms), remaining_ms)))


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


# --- Safe State Getters ---

def state_get_list(state: UniversalState, key: str, default: list[Any] | None = None) -> list[Any]:
    val = state.get(key)
    if isinstance(val, list):
        return val
    return default if default is not None else []


def state_get_dict(state: UniversalState, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    val = state.get(key)
    if isinstance(val, dict):
        return val
    return default if default is not None else {}


def state_get_int(state: UniversalState, key: str, default: int = 0) -> int:
    val = state.get(key)
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def state_get_str(state: UniversalState, key: str, default: str = "") -> str:
    val = state.get(key)
    return str(val).strip() if val is not None else default


def state_get_float(state: UniversalState, key: str, default: float = 0.0) -> float:
    val = state.get(key)
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


# --- Node Timing Decorator ---

import functools

def track_node_timing(stage_name: str):
    """
    Decorator for LangGraph nodes to automatically track their execution time.
    Instead of manually calling `_append_stage_timing` at every return, this handles it.
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(state: UniversalState, *args, **kwargs) -> dict[str, Any]:
            t0 = time.perf_counter()
            result = await func(state, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            
            if isinstance(result, dict):
                # We inject the stage object
                result["stage_timings_ms"] = _append_stage_timing(
                    state, stage=stage_name, elapsed_ms=elapsed_ms
                )
            return result
        return async_wrapper
    return decorator

