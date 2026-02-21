from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from app.agent.types.models import (
    AnswerDraft,
    ClarificationRequest,
    EvidenceItem,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.profiles.models import AgentProfile
from app.agent.policies import (  # compatibility exports for existing tests/imports
    build_retrieval_plan,
    classify_intent,
    classify_intent_with_trace,
    detect_conflict_objectives,
    detect_scope_candidates,
    extract_requested_scopes,
    has_clause_reference,
    suggest_scope_candidates,
)


logger = structlog.get_logger(__name__)


class RetrieverPort(Protocol):
    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]: ...

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]: ...

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def apply_validated_scope(self, validated: dict[str, Any]) -> None: ...


class AnswerGeneratorPort(Protocol):
    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
        working_memory: dict[str, Any] | None = None,
        partial_answers: list[dict[str, Any]] | None = None,
        agent_profile: AgentProfile | None = None,
    ) -> AnswerDraft: ...


class ValidatorPort(Protocol):
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str) -> ValidationResult: ...


@dataclass(frozen=True)
class HandleQuestionCommand:
    query: str
    tenant_id: str
    collection_id: str | None
    scope_label: str
    user_id: str | None = None
    agent_profile: AgentProfile | None = None
    profile_resolution: dict[str, Any] | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    clarification_context: dict[str, Any] | None = None
    split_depth: int = 0


@dataclass(frozen=True)
class HandleQuestionResult:
    intent: QueryIntent
    plan: RetrievalPlan
    answer: AnswerDraft
    validation: ValidationResult
    retrieval: RetrievalDiagnostics
    clarification: ClarificationRequest | None = None
    reasoning_trace: dict[str, Any] | None = None
    engine: str = "universal_flow"


class HandleQuestionUseCase:
    """Entry point for question handling, delegating to the Universal Reasoning Orchestrator."""

    def __init__(
        self,
        retriever: RetrieverPort,
        answer_generator: AnswerGeneratorPort,
        validator: ValidatorPort,
    ):
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._validator = validator
        self._orchestrator: Any | None = None

    def _get_orchestrator(self) -> Any:
        if self._orchestrator is None:
            from app.graph.universal.flow import UniversalReasoningOrchestrator

            self._orchestrator = UniversalReasoningOrchestrator(
                retriever=self._retriever,
                answer_generator=self._answer_generator,
                validator=self._validator,
            )
            logger.info(
                "orchestrator_instance_initialized",
                engine="universal_flow",
            )
        return self._orchestrator

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        runner = self._get_orchestrator()
        return await runner.execute(cmd)
