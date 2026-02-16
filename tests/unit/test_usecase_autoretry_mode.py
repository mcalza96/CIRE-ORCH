import pytest

from app.agent.application import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)


class _FakeRetriever:
    def __init__(self) -> None:
        self.last_retrieval_diagnostics = None

    async def validate_scope(self, **kwargs):
        return {"valid": True, "normalized_scope": {"filters": {}}, "query_scope": {}}

    def apply_validated_scope(self, validated):
        return None

    async def retrieve_chunks(
        self,
        query,
        tenant_id,
        collection_id,
        plan: RetrievalPlan,
        user_id=None,
        request_id=None,
        correlation_id=None,
    ):
        # Simulate collapse for literal modes, success for others.
        if plan.mode in {"literal_normativa", "literal_lista"}:
            self.last_retrieval_diagnostics = RetrievalDiagnostics(
                contract="advanced",
                strategy="hybrid",
                partial=False,
                trace={"timings_ms": {"hybrid": 1.0}},
                scope_validation={},
            )
            return []
        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="multi_query_primary",
            partial=False,
            trace={"timings_ms": {"multi_query_primary": 1.0}},
            scope_validation={},
        )
        return [
            EvidenceItem(
                source="C1",
                content="evidence",
                score=0.9,
                metadata={"row": {"metadata": {"source_standard": "X"}}},
            )
        ]

    async def retrieve_summaries(
        self,
        query,
        tenant_id,
        collection_id,
        plan: RetrievalPlan,
        user_id=None,
        request_id=None,
        correlation_id=None,
    ):
        return []


class _FakeAnswerGen:
    async def generate(self, query, scope_label, plan, chunks, summaries, agent_profile=None):
        if not chunks and not summaries:
            return AnswerDraft(
                text="No tengo informacion suficiente en el contexto para responder.",
                mode=plan.mode,
                evidence=[],
            )
        return AnswerDraft(text="OK", mode=plan.mode, evidence=[*chunks, *summaries])


class _FakeValidator:
    def validate(self, draft, plan, query):
        return ValidationResult(
            accepted=bool(draft.evidence),
            issues=[] if draft.evidence else ["No retrieval evidence"],
        )


@pytest.mark.asyncio
async def test_autoretry_switches_off_literal_when_empty_retrieval(monkeypatch):
    # Force contract advanced and enable autoretry.
    from app.core.config import settings

    monkeypatch.setattr(settings, "ORCH_RETRIEVAL_CONTRACT", "advanced")
    monkeypatch.setattr(settings, "ORCH_MODE_AUTORETRY_ENABLED", True)
    monkeypatch.setattr(settings, "ORCH_MODE_AUTORETRY_MAX_ATTEMPTS", 2)

    # Force initial intent to literal by using explicit override.
    cmd = HandleQuestionCommand(
        query="__mode__=literal_normativa Analice como impacta... ISO 9001 9.1.2 ISO 14001 9.1.1 5.3 5.4",
        tenant_id="t",
        user_id=None,
        collection_id=None,
        scope_label="tenant=t",
    )
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGen(), _FakeValidator())
    result = await use_case.execute(cmd)

    assert result.plan.mode in {"comparativa", "explicativa"}
    assert result.answer.text == "OK"
    trace = result.retrieval.trace
    assert isinstance(trace, dict)
    attempts = trace.get("attempts")
    assert isinstance(attempts, list)
