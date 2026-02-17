from dataclasses import dataclass, field

import pytest

from app.agent.application import HandleQuestionCommand
from app.agent.models import (
    AnswerDraft,
    EvidenceItem,
    QueryIntent,
    RetrievalDiagnostics,
    RetrievalPlan,
    ValidationResult,
)
from app.graph.iso_flow import IsoFlowOrchestrator


@dataclass
class _RecorderRetriever:
    chunks_runs: list[list[EvidenceItem]]
    summaries_runs: list[list[EvidenceItem]] | None = None
    calls: int = 0
    plans_seen: list[RetrievalPlan] = field(default_factory=list)
    queries_seen: list[str] = field(default_factory=list)
    last_retrieval_diagnostics: RetrievalDiagnostics | None = None

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ):
        del tenant_id, collection_id, user_id, request_id, correlation_id
        self.calls += 1
        self.plans_seen.append(plan)
        self.queries_seen.append(str(query or ""))
        idx = min(self.calls - 1, len(self.chunks_runs) - 1)
        return self.chunks_runs[idx]

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ):
        del query, tenant_id, collection_id, plan, user_id, request_id, correlation_id
        if not self.summaries_runs:
            return []
        idx = min(self.calls - 1, len(self.summaries_runs) - 1)
        return self.summaries_runs[idx]

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
        filters: dict | None = None,
    ) -> dict:
        del query, tenant_id, collection_id, plan, user_id, request_id, correlation_id, filters
        return {"valid": True, "normalized_scope": {"filters": {}}, "query_scope": {}}

    def apply_validated_scope(self, validated: dict) -> None:
        del validated


@dataclass
class _AnswerGen:
    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
        agent_profile=None,
    ):
        del query, scope_label, summaries, agent_profile
        return AnswerDraft(text=f"ok mode={plan.mode}", mode=plan.mode, evidence=chunks)


@dataclass
class _Validator:
    def validate(self, *args, **kwargs):
        return ValidationResult(accepted=True, issues=[])


@pytest.mark.asyncio
async def test_graph_retries_when_first_retrieval_is_empty():
    retriever = _RecorderRetriever(
        chunks_runs=[[], [EvidenceItem(source="C1", content="ISO 9001 clause 9.3 text")]],
    )
    flow = IsoFlowOrchestrator(
        retriever=retriever,
        answer_generator=_AnswerGen(),
        validator=_Validator(),
        max_retries=1,
    )

    result = await flow.execute(
        HandleQuestionCommand(
            query="Que exige textualmente ISO 9001 9.3?",
            tenant_id="t1",
            collection_id=None,
            scope_label="tenant=t1",
        )
    )

    assert retriever.calls == 2
    assert result.retrieval.trace["graph"]["retry_count"] == 1
    assert result.answer.text.startswith("ok mode=")


@pytest.mark.asyncio
async def test_graph_skips_retry_when_retrieval_is_relevant():
    retriever = _RecorderRetriever(
        chunks_runs=[[EvidenceItem(source="C1", content="ISO 9001 clause 9.3 text")]],
    )
    flow = IsoFlowOrchestrator(
        retriever=retriever,
        answer_generator=_AnswerGen(),
        validator=_Validator(),
        max_retries=1,
    )

    result = await flow.execute(
        HandleQuestionCommand(
            query="Que exige textualmente ISO 9001 9.3?",
            tenant_id="t1",
            collection_id=None,
            scope_label="tenant=t1",
        )
    )

    assert retriever.calls == 1
    assert result.retrieval.trace["graph"]["retry_count"] == 0


@pytest.mark.asyncio
async def test_graph_preserves_literal_mode_on_scope_mismatch_retry(monkeypatch):
    import app.graph.iso_flow as flow_mod

    monkeypatch.setattr(
        flow_mod,
        "classify_intent",
        lambda query, profile=None: QueryIntent(mode="literal_normativa", rationale="test"),
    )
    monkeypatch.setattr(
        flow_mod,
        "build_retrieval_plan",
        lambda intent, query="", profile=None: RetrievalPlan(
            mode=intent.mode,
            chunk_k=10,
            chunk_fetch_k=40,
            summary_k=2,
            require_literal_evidence=(intent.mode == "literal_normativa"),
            requested_standards=("ISO 9001",),
        ),
    )

    wrong_scope = EvidenceItem(
        source="C1",
        content="ISO 45001 clause 6.1.3 text",
        metadata={"row": {"metadata": {"source_standard": "ISO 45001"}}},
    )
    recovered = EvidenceItem(
        source="C2",
        content="ISO 9001 9.3 revision por la direccion",
        metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
    )
    retriever = _RecorderRetriever(chunks_runs=[[wrong_scope], [recovered]])
    flow = IsoFlowOrchestrator(
        retriever=retriever,
        answer_generator=_AnswerGen(),
        validator=_Validator(),
        max_retries=1,
    )

    await flow.execute(
        HandleQuestionCommand(
            query="Que exige textualmente ISO 9001 9.3?",
            tenant_id="t1",
            collection_id=None,
            scope_label="tenant=t1",
        )
    )

    assert len(retriever.plans_seen) >= 2
    assert retriever.plans_seen[0].mode == "literal_normativa"
    assert retriever.plans_seen[1].mode == "literal_normativa"
    assert "[GRAPH_GAPS]" in retriever.queries_seen[1]


@pytest.mark.asyncio
async def test_graph_comparativa_requires_multi_scope_before_generating(monkeypatch):
    import app.graph.iso_flow as flow_mod

    monkeypatch.setattr(
        flow_mod,
        "classify_intent",
        lambda query, profile=None: QueryIntent(mode="comparativa", rationale="test"),
    )
    monkeypatch.setattr(
        flow_mod,
        "build_retrieval_plan",
        lambda intent, query="", profile=None: RetrievalPlan(
            mode="comparativa",
            chunk_k=10,
            chunk_fetch_k=40,
            summary_k=2,
            require_literal_evidence=False,
            requested_standards=("ISO 9001", "ISO 45001"),
        ),
    )

    one_scope = EvidenceItem(
        source="C1",
        content="hallazgo 8.1.3",
        metadata={"row": {"metadata": {"source_standard": "ISO 45001"}}},
    )
    two_scope = EvidenceItem(
        source="C2",
        content="hallazgo 6.3",
        metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
    )
    retriever = _RecorderRetriever(chunks_runs=[[one_scope], [one_scope, two_scope]])
    flow = IsoFlowOrchestrator(
        retriever=retriever,
        answer_generator=_AnswerGen(),
        validator=_Validator(),
        max_retries=1,
    )

    result = await flow.execute(
        HandleQuestionCommand(
            query="Compara ISO 9001 6.3 frente a ISO 45001 8.1.3",
            tenant_id="t1",
            collection_id=None,
            scope_label="tenant=t1",
        )
    )

    assert retriever.calls == 2
    assert result.retrieval.trace["graph"]["grade_reason"] in {"ok", "scope_mismatch"}


@pytest.mark.asyncio
async def test_graph_respects_max_retries_zero():
    retriever = _RecorderRetriever(chunks_runs=[[]])
    flow = IsoFlowOrchestrator(
        retriever=retriever,
        answer_generator=_AnswerGen(),
        validator=_Validator(),
        max_retries=0,
    )

    result = await flow.execute(
        HandleQuestionCommand(
            query="Que exige textualmente ISO 9001 9.3?",
            tenant_id="t1",
            collection_id=None,
            scope_label="tenant=t1",
        )
    )

    assert retriever.calls == 1
    assert result.retrieval.trace["graph"]["retry_count"] == 0
