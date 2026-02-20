from __future__ import annotations

from app.graph.universal.state import MAX_PLAN_ATTEMPTS, UniversalState
from app.graph.universal.utils import state_get_str


def route_after_planner(state: UniversalState) -> str:
    next_action = state_get_str(state, "next_action", "")
    if next_action == "interrupt":
        return "interrupt"
    return "execute" if next_action == "execute" else "generate"


def route_after_reflect(state: UniversalState) -> str:
    next_action = state_get_str(state, "next_action", "")
    # Hard safety cap: never allow more replans than MAX_PLAN_ATTEMPTS.
    # Timeouts must NOT be the only mechanism that breaks the loop.
    plan_attempts = int(state.get("plan_attempts") or 1)
    if next_action == "replan" and plan_attempts < MAX_PLAN_ATTEMPTS:
        return "replan"
    if next_action == "execute_tool":
        return "execute_tool"
    return "generate"
