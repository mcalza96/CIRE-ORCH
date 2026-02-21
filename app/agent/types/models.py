from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


QueryMode = str


@dataclass(frozen=True)
class QueryIntent:
    mode: QueryMode
    rationale: str = ""


@dataclass(frozen=True)
class RetrievalPlan:
    mode: QueryMode
    chunk_k: int
    chunk_fetch_k: int
    summary_k: int
    require_literal_evidence: bool = False
    allow_inference: bool = True
    response_contract: str | None = None
    requested_standards: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    source: str
    content: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerDraft:
    text: str
    mode: QueryMode
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ClarificationRequest:
    question: str
    options: tuple[str, ...] = ()
    kind: str = "clarification"
    level: str = "L2"


@dataclass(frozen=True)
class RetrievalDiagnostics:
    contract: Literal["legacy", "advanced"]
    strategy: str = "legacy"
    partial: bool = False
    trace: dict[str, Any] = field(default_factory=dict)
    scope_validation: dict[str, Any] = field(default_factory=dict)


ReasoningStepType = Literal["plan", "tool", "reflection", "synthesis", "validation"]


@dataclass(frozen=True)
class ToolCall:
    tool: str
    input: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


@dataclass(frozen=True)
class ToolResult:
    tool: str
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True)
class ReasoningStep:
    index: int
    type: ReasoningStepType
    description: str
    tool: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str = ""


@dataclass(frozen=True)
class ReasoningPlan:
    goal: str
    steps: list[ToolCall] = field(default_factory=list)
    complexity: Literal["simple", "complex"] = "simple"


@dataclass(frozen=True)
class ReasoningTrace:
    engine: str = "universal_flow"
    stop_reason: str = "unknown"
    plan_attempts: int = 1
    reflections: int = 0
    tools_used: list[str] = field(default_factory=list)
    final_confidence: float | None = None
    steps: list[ReasoningStep] = field(default_factory=list)
