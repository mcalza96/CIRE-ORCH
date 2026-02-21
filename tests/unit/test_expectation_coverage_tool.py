import pytest

from app.agent.types.models import EvidenceItem, RetrievalPlan
from app.agent.tools.base import ToolRuntimeContext
from app.agent.tools.expectation_coverage import ExpectationCoverageTool
from app.profiles.models import AgentProfile, ExpectationRule


class _DummyRetriever:
    async def retrieve_chunks(self, *args, **kwargs):  # pragma: no cover
        return []

    async def retrieve_summaries(self, *args, **kwargs):  # pragma: no cover
        return []


class _DummyAnswerGenerator:
    async def generate(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("unused")


class _DummyValidator:
    def validate(self, *args, **kwargs):  # pragma: no cover
        raise RuntimeError("unused")


@pytest.mark.asyncio
async def test_expectation_coverage_reports_missing_and_covered() -> None:
    tool = ExpectationCoverageTool()
    profile = AgentProfile(
        profile_id="iso",
        expectations=[
            ExpectationRule(
                id="ok",
                description="Programa de auditoria",
                scopes=["ISO 9001"],
                clause_refs=["9.2"],
                required_evidence_markers=["programa de auditoria"],
                applies_to_modes=["gap_analysis"],
            ),
            ExpectationRule(
                id="missing",
                description="Revision por la direccion",
                scopes=["ISO 9001"],
                clause_refs=["9.3"],
                required_evidence_markers=["revision por la direccion"],
                applies_to_modes=["gap_analysis"],
                missing_risk="No cierre de ciclo",
                severity="high",
            ),
        ],
    )
    state = {
        "agent_profile": profile,
        "retrieval_plan": RetrievalPlan(
            mode="gap_analysis",
            chunk_k=20,
            chunk_fetch_k=100,
            summary_k=4,
        ),
        "retrieved_documents": [
            EvidenceItem(
                source="C1",
                content="El programa de auditoria interna se ejecuta anualmente en clausula 9.2.",
                metadata={
                    "row": {
                        "content": "El programa de auditoria interna se ejecuta anualmente en clausula 9.2.",
                        "metadata": {"source_standard": "ISO 9001", "clause_id": "9.2"},
                    }
                },
            )
        ],
    }

    result = await tool.run(
        {},
        state=state,
        context=ToolRuntimeContext(
            retriever=_DummyRetriever(),
            answer_generator=_DummyAnswerGenerator(),
            validator=_DummyValidator(),
        ),
    )

    assert result.ok is True
    assert result.output["total_expectations"] == 2
    assert len(result.output["covered"]) == 1
    assert len(result.output["missing"]) == 1
