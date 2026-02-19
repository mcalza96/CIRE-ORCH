from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from app.agent.models import (
    AnswerDraft,
    ClarificationRequest,
    EvidenceItem,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.cartridges.models import AgentProfile
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
        from app.agent.policies.query_splitter import QuerySplitter

        self._splitter = QuerySplitter()

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

    async def _execute_compound(
        self, cmd: HandleQuestionCommand, parts: list[str]
    ) -> HandleQuestionResult:
        """Executes split parts of a compound query in parallel and synthesizes results."""
        tasks = []
        for part in parts:
            child_cmd = HandleQuestionCommand(
                query=part,
                tenant_id=cmd.tenant_id,
                collection_id=cmd.collection_id,
                scope_label=cmd.scope_label,
                user_id=cmd.user_id,
                agent_profile=cmd.agent_profile,
                profile_resolution=cmd.profile_resolution,
                request_id=cmd.request_id,
                correlation_id=cmd.correlation_id,
                clarification_context=cmd.clarification_context,
                split_depth=cmd.split_depth + 1,
            )
            tasks.append(self.execute(child_cmd))

        from app.agent.models import AnswerDraft, QueryIntent, ValidationResult

        sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results: list[HandleQuestionResult] = []

        for res in sub_results:
            if isinstance(res, HandleQuestionResult):
                if res.clarification is not None:
                    return res
                valid_results.append(res)
            else:
                logger.error("compound_query_part_failed", error=str(res))

        if not valid_results:
            # Fallback for total failure
            runner = self._get_orchestrator()
            return await runner.execute(cmd)

        sections = [
            f"**Sección {idx + 1}**\n{result.answer.text}"
            for idx, result in enumerate(valid_results)
        ]
        combined_text = "\n\n".join(sections)

        from app.agent.models import EvidenceItem

        combined_evidence: list[EvidenceItem] = []
        seen_ids: set[str] = set()
        for res in valid_results:
            for ev in res.answer.evidence:
                eid = str(ev.metadata.get("id") if ev.metadata else "") or ev.content[:32]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    combined_evidence.append(ev)

        # Determine mode: if all are same, use it; otherwise fallback to profile default mode.
        modes = [res.plan.mode for res in valid_results]
        fallback_mode = (
            str(cmd.agent_profile.query_modes.default_mode).strip()
            if cmd.agent_profile is not None and cmd.agent_profile.query_modes.default_mode
            else "default"
        )
        mode = modes[0] if len(set(modes)) == 1 else fallback_mode

        combined_answer = AnswerDraft(
            text=combined_text,
            mode=mode,
            evidence=combined_evidence,
        )

        return HandleQuestionResult(
            intent=QueryIntent(mode=mode, rationale="compound_synthesis"),
            plan=valid_results[0].plan,  # Representative plan
            answer=combined_answer,
            validation=ValidationResult(accepted=True, issues=[]),
            retrieval=RetrievalDiagnostics(
                contract="legacy", strategy="compound", partial=False, trace={}
            ),
        )

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        # Handle splitting for top-level queries
        if cmd.split_depth == 0:
            parts = self._splitter.split(cmd.query)
            if len(parts) >= 2:
                return await self._execute_compound(cmd, parts)

        runner = self._get_orchestrator()
        return await runner.execute(cmd)
