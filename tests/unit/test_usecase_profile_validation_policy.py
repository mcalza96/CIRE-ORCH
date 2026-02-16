import pytest

from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.cartridges.models import AgentProfile, ValidationPolicy


class _Retriever:
    def __init__(self) -> None:
        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="hybrid",
            partial=False,
            trace={},
            scope_validation={},
        )

    async def validate_scope(self, **kwargs):
        return {"valid": True, "normalized_scope": {"filters": {}}, "query_scope": {}}

    def apply_validated_scope(self, validated):
        return None

    def set_profile_context(self, **kwargs):
        return None

    async def retrieve_chunks(self, query, tenant_id, collection_id, plan: RetrievalPlan, user_id=None, request_id=None, correlation_id=None):
        return [
            EvidenceItem(source="C1", content="evidencia", score=0.99, metadata={"row": {"metadata": {}}})
        ]

    async def retrieve_summaries(self, query, tenant_id, collection_id, plan: RetrievalPlan, user_id=None, request_id=None, correlation_id=None):
        return []


class _AnswerGen:
    async def generate(self, query, scope_label, plan, chunks, summaries, agent_profile=None):
        return AnswerDraft(text="Respuesta con sueldos. Fuente(C1)", mode=plan.mode, evidence=[*chunks, *summaries])


class _Validator:
    def validate(self, draft, plan, query):
        return ValidationResult(accepted=True, issues=[])


@pytest.mark.asyncio
async def test_usecase_applies_forbidden_concepts_policy(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ORCH_RETRIEVAL_CONTRACT", "advanced")

    profile = AgentProfile(
        profile_id="policy",
        validation=ValidationPolicy(
            require_citations=True,
            forbidden_concepts=["sueldos"],
            fallback_message="No hay evidencia",
        ),
    )

    use_case = HandleQuestionUseCase(_Retriever(), _AnswerGen(), _Validator())
    result = await use_case.execute(
        HandleQuestionCommand(
            query="analiza riesgos",
            tenant_id="tenant-1",
            collection_id=None,
            scope_label="tenant=tenant-1",
            agent_profile=profile,
        )
    )

    assert result.validation.accepted is False
    assert any("forbidden concept" in issue for issue in result.validation.issues)
