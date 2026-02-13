from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.agent.models import (
    AnswerDraft,
    ClarificationRequest,
    EvidenceItem,
    QueryIntent,
    RetrievalPlan,
    ValidationResult,
)


class RetrieverPort(Protocol):
    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
    ) -> list[EvidenceItem]: ...

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
    ) -> list[EvidenceItem]: ...


class AnswerGeneratorPort(Protocol):
    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
    ) -> AnswerDraft: ...


class ValidatorPort(Protocol):
    def validate(self, draft: AnswerDraft, plan: RetrievalPlan, query: str) -> ValidationResult: ...
from app.agent.policies import (
    build_retrieval_plan,
    classify_intent,
    detect_conflict_objectives,
    detect_scope_candidates,
    suggest_scope_candidates,
)


@dataclass(frozen=True)
class HandleQuestionCommand:
    query: str
    tenant_id: str
    collection_id: str | None
    scope_label: str


@dataclass(frozen=True)
class HandleQuestionResult:
    intent: QueryIntent
    plan: RetrievalPlan
    answer: AnswerDraft
    validation: ValidationResult
    clarification: ClarificationRequest | None = None


class HandleQuestionUseCase:
    """Application orchestrator for Q/A Orchestrator."""

    def __init__(
        self,
        retriever: RetrieverPort,
        answer_generator: AnswerGeneratorPort,
        validator: ValidatorPort,
    ):
        self._retriever = retriever
        self._answer_generator = answer_generator
        self._validator = validator

    async def execute(self, cmd: HandleQuestionCommand) -> HandleQuestionResult:
        normalized_query = (cmd.query or "").lower()
        has_user_clarification = (
            "aclaración de alcance:" in normalized_query
            or "aclaracion de alcance:" in normalized_query
            or "modo preferido en sesión:" in normalized_query
            or "modo preferido en sesion:" in normalized_query
            or "__clarified_scope__=true" in normalized_query
        )

        intent = classify_intent(cmd.query)
        plan = build_retrieval_plan(intent, query=cmd.query)
        detected_scopes = detect_scope_candidates(cmd.query)
        conflict_mode = detect_conflict_objectives(cmd.query)

        if conflict_mode and has_user_clarification:
            plan = RetrievalPlan(
                mode="explicativa",
                chunk_k=max(plan.chunk_k, 35),
                chunk_fetch_k=max(plan.chunk_fetch_k, 140),
                summary_k=max(plan.summary_k, 4),
                require_literal_evidence=False,
                requested_standards=plan.requested_standards,
            )

        if not has_user_clarification and conflict_mode and len(detected_scopes) >= 2:
            clarification = ClarificationRequest(
                question=(
                    "Detecté conflicto entre integridad de evidencia y confidencialidad del denunciante "
                    f"en un escenario multinorma ({', '.join(detected_scopes)}). "
                    "¿Priorizo un análisis conservador de protección al denunciante y no represalia, "
                    "o un análisis forense estricto centrado en trazabilidad documental?"
                ),
                options=(
                    "Protección al denunciante",
                    "Forense de trazabilidad",
                    "Balanceado trinorma",
                ),
            )
            answer = AnswerDraft(text=clarification.question, mode=plan.mode, evidence=[])
            validation = ValidationResult(accepted=True, issues=[])
            return HandleQuestionResult(
                intent=intent,
                plan=plan,
                answer=answer,
                validation=validation,
                clarification=clarification,
            )

        if (
            not has_user_clarification
            and intent.mode == "explicativa"
            and len(detected_scopes) >= 2
            and len(plan.requested_standards) < 2
        ):
            clarification = ClarificationRequest(
                question=(
                    "Detecté señales de múltiples normas ("
                    + ", ".join(detected_scopes)
                    + "). ¿Quieres análisis integrado trinorma o limitarlo a una norma específica?"
                ),
                options=("Análisis integrado trinorma", *detected_scopes),
            )
            answer = AnswerDraft(text=clarification.question, mode=plan.mode, evidence=[])
            validation = ValidationResult(accepted=True, issues=[])
            return HandleQuestionResult(
                intent=intent,
                plan=plan,
                answer=answer,
                validation=validation,
                clarification=clarification,
            )

        if intent.mode == "ambigua_scope":
            options = suggest_scope_candidates(cmd.query)
            suggestion = ", ".join(options[:3])
            clarification = (
                "Necesito desambiguar el alcance antes de responder con trazabilidad. "
                f"Indica la norma objetivo (sugeridas: {suggestion})."
            )
            answer = AnswerDraft(text=clarification, mode=plan.mode, evidence=[])
            validation = ValidationResult(accepted=True, issues=[])
            return HandleQuestionResult(
                intent=intent,
                plan=plan,
                answer=answer,
                validation=validation,
            )

        chunks = await self._retriever.retrieve_chunks(
            query=cmd.query,
            tenant_id=cmd.tenant_id,
            collection_id=cmd.collection_id,
            plan=plan,
        )
        summaries = await self._retriever.retrieve_summaries(
            query=cmd.query,
            tenant_id=cmd.tenant_id,
            collection_id=cmd.collection_id,
            plan=plan,
        )

        answer = await self._answer_generator.generate(
            query=cmd.query,
            scope_label=cmd.scope_label,
            plan=plan,
            chunks=chunks,
            summaries=summaries,
        )
        validation = self._validator.validate(answer, plan, cmd.query)
        if not validation.accepted and any("Scope mismatch" in issue for issue in validation.issues):
            if len(detected_scopes) >= 2:
                clarification = ClarificationRequest(
                    question=(
                        "La consulta parece cruzar múltiples normas ("
                        + ", ".join(detected_scopes)
                        + "). ¿Confirmas análisis integrado o prefieres restringir el alcance?"
                    ),
                    options=("Análisis integrado trinorma", *detected_scopes),
                )
                answer = AnswerDraft(text=clarification.question, mode=plan.mode, evidence=answer.evidence)
                validation = ValidationResult(accepted=True, issues=[])
                return HandleQuestionResult(
                    intent=intent,
                    plan=plan,
                    answer=answer,
                    validation=validation,
                    clarification=clarification,
                )

            answer = AnswerDraft(
                text=(
                    "⚠️ Respuesta bloqueada por inconsistencia de ámbito entre la pregunta y las fuentes recuperadas. "
                    "Reformula indicando explícitamente la norma objetivo (por ejemplo: ISO 9001)."
                ),
                mode=plan.mode,
                evidence=answer.evidence,
            )

        return HandleQuestionResult(
            intent=intent,
            plan=plan,
            answer=answer,
            validation=validation,
        )
