from __future__ import annotations

from typing import Any, NotRequired, TypedDict

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
from app.cartridges.models import AgentProfile

# Constants moved from universal_flow.py
DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_REFLECTIONS = 2
MAX_PLAN_ATTEMPTS = 3
ANSWER_PREVIEW_LIMIT = 180
RETRY_REASON_LIMIT = 120
HARD_MAX_STEPS = 12
HARD_MAX_REFLECTIONS = 6


class UniversalState(TypedDict):
    user_query: str
    working_query: str
    tenant_id: str
    collection_id: str | None
    user_id: str | None
    request_id: str | None
    correlation_id: str | None
    scope_label: str
    agent_profile: AgentProfile | None
    tool_results: list[ToolResult]
    tool_cursor: int
    plan_attempts: int
    reflections: int
    reasoning_steps: list[ReasoningStep]
    working_memory: dict[str, object]
    chunks: list[EvidenceItem]
    summaries: list[EvidenceItem]
    retrieved_documents: list[EvidenceItem]
    subquery_groups: NotRequired[list[dict[str, Any]]]
    partial_answers: NotRequired[list[dict[str, Any]]]
    allowed_tools: NotRequired[list[str]]
    intent: NotRequired[object]
    retrieval_plan: NotRequired[RetrievalPlan]
    reasoning_plan: NotRequired[ReasoningPlan]
    max_steps: NotRequired[int]
    max_reflections: NotRequired[int]
    next_action: NotRequired[str]
    stop_reason: NotRequired[str]
    retrieval: NotRequired[RetrievalDiagnostics]
    generation: NotRequired[AnswerDraft]
    validation: NotRequired[ValidationResult]
    stage_timings_ms: NotRequired[dict[str, float]]
    tool_timings_ms: NotRequired[dict[str, float]]
    flow_start_pc: NotRequired[float]
