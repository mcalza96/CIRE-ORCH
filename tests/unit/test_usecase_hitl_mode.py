import pytest

from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.models import (
    AnswerDraft,
    ClarificationRequest,
    EvidenceItem,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)


class _EmptyRetriever:
    def __init__(self) -> None:
        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="hybrid",
            partial=False,
            trace={"timings_ms": {"hybrid": 1.0}},
            scope_validation={},
        )

    async def validate_scope(self, **kwargs):
        return {"valid": True, "normalized_scope": {"filters": {}}, "query_scope": {}}

    def apply_validated_scope(self, validated):
        return None

    async def retrieve_chunks(
        self, query, tenant_id, collection_id, plan: RetrievalPlan, user_id=None
    ):
        return []

    async def retrieve_summaries(
        self, query, tenant_id, collection_id, plan: RetrievalPlan, user_id=None
    ):
        return []


class _NoopAnswerGen:
    async def generate(self, query, scope_label, plan, chunks, summaries, agent_profile=None):
        return AnswerDraft(
            text="No tengo informacion suficiente en el contexto para responder.",
            mode=plan.mode,
            evidence=[],
        )


class _NoopValidator:
    def validate(self, draft, plan, query):
        return ValidationResult(accepted=False, issues=["No retrieval evidence"])


@pytest.mark.asyncio
async def test_hitl_mode_clarification_on_low_confidence_and_empty(monkeypatch):
    from app.core.config import settings
    import app.agent.policies as policies

    monkeypatch.setattr(settings, "ORCH_RETRIEVAL_CONTRACT", "advanced")
    monkeypatch.setattr(settings, "ORCH_MODE_HITL_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_MODE_LOW_CONFIDENCE_THRESHOLD", 0.9)
    monkeypatch.setattr(settings, "ORCH_MODE_AUTORETRY_ENABLED", False)

    def _fake_classify_with_trace(query: str, profile=None):
        return (
            policies.QueryIntent(mode="explicativa", rationale="test"),
            {"version": "v2", "mode": "explicativa", "confidence": 0.1, "reasons": ["low_signal"]},
        )

    monkeypatch.setattr(policies, "classify_intent_with_trace", _fake_classify_with_trace)

    use_case = HandleQuestionUseCase(_EmptyRetriever(), _NoopAnswerGen(), _NoopValidator())
    result = await use_case.execute(
        HandleQuestionCommand(
            query="Pregunta ambigua de tipo de respuesta.",
            tenant_id="t",
            user_id=None,
            collection_id=None,
            scope_label="tenant=t",
        )
    )

    assert isinstance(result.clarification, ClarificationRequest)
    assert "modo" in result.clarification.question.lower()
