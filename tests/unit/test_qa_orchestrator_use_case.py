from dataclasses import dataclass

import pytest

from app.agent.engine import HandleQuestionCommand, HandleQuestionUseCase
from app.agent.types.models import AnswerDraft, EvidenceItem, RetrievalPlan, ValidationResult
from app.profiles.models import AgentProfile, CapabilitiesPolicy


@dataclass
class _FakeRetriever:
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
        del query, tenant_id, collection_id, plan, user_id, request_id, correlation_id
        return [EvidenceItem(source="C1", content="ISO 9001 9.3 evidencia")]

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
        return [EvidenceItem(source="R1", content="Resumen")]

    async def validate_scope(self, **kwargs):
        del kwargs
        return {"valid": True, "normalized_scope": {"filters": {}}, "query_scope": {}}

    def apply_validated_scope(self, validated):
        del validated


@dataclass
class _FakeAnswerGenerator:
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
        return AnswerDraft(text="respuesta", mode=plan.mode, evidence=chunks)


@dataclass
class _FakeValidator:
    def validate(self, *args, **kwargs):
        del args, kwargs
        return ValidationResult(accepted=True, issues=[])


@pytest.mark.asyncio
async def test_use_case_delegates_to_graph_and_returns_result():
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    result = await use_case.execute(
        HandleQuestionCommand(
            query="Que exige ISO 9001 en 9.3?",
            tenant_id="t1",
            collection_id="c1",
            scope_label="s1",
        )
    )
    assert result.answer.text == "respuesta"
    assert result.retrieval.strategy == "langgraph_universal_flow"
    assert result.engine == "universal_flow"
    assert isinstance(result.reasoning_trace, dict)
    assert isinstance(result.reasoning_trace.get("stage_timings_ms"), dict)
    assert isinstance((result.retrieval.trace or {}).get("timings_ms"), dict)
    assert "universal_total" in dict((result.retrieval.trace or {}).get("timings_ms") or {})


@pytest.mark.asyncio
async def test_use_case_initializes_graph_runner_once():
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    await use_case.execute(
        HandleQuestionCommand(
            query="Q1",
            tenant_id="t1",
            collection_id="c1",
            scope_label="s1",
        )
    )
    first_runner = use_case._runner
    await use_case.execute(
        HandleQuestionCommand(
            query="Q2",
            tenant_id="t1",
            collection_id="c1",
            scope_label="s1",
        )
    )
    assert use_case._runner is first_runner


@pytest.mark.asyncio
async def test_use_case_preserves_handle_question_contract():
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    cmd = HandleQuestionCommand(
        query="Compara ISO 9001 vs 14001",
        tenant_id="t1",
        collection_id="c1",
        scope_label="s1",
        profile_resolution={"source": "base"},
    )
    result = await use_case.execute(cmd)
    assert result.plan.mode in {
        "literal_normativa",
        "literal_lista",
        "explicativa",
        "comparativa",
        "ambigua_scope",
    }
    assert isinstance(result.validation.accepted, bool)


@pytest.mark.asyncio
async def test_use_case_uses_universal_flow_for_high_reasoning_profile():
    profile = AgentProfile(
        profile_id="iso_auditor",
        capabilities=CapabilitiesPolicy(
            reasoning_level="high",
            allowed_tools=["semantic_retrieval", "citation_validator"],
        ),
    )
    use_case = HandleQuestionUseCase(_FakeRetriever(), _FakeAnswerGenerator(), _FakeValidator())
    result = await use_case.execute(
        HandleQuestionCommand(
            query="Que exige ISO 9001 en 9.3?",
            tenant_id="t1",
            collection_id="c1",
            scope_label="s1",
            agent_profile=profile,
        )
    )
    assert result.engine == "universal_flow"
    assert result.retrieval.strategy == "langgraph_universal_flow"
    assert isinstance(result.reasoning_trace, dict)
