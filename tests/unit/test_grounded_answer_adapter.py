import asyncio

from app.agent.formatters.answer_adapter import GroundedAnswerAdapter
from app.agent.types.models import EvidenceItem, RetrievalPlan
from app.profiles.models import AgentProfile


class _FakeGroundedService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate_answer(
        self,
        query: str,
        context_chunks: list[str],
        *,
        mode: str,
        require_literal_evidence: bool,
        structured_context: str | None = None,
        max_chunks: int = 10,
        agent_profile: AgentProfile | None = None,
    ) -> str:  # type: ignore[override]
        self.calls.append(
            {
                "query": query,
                "context_chunks": list(context_chunks),
                "mode": mode,
                "require_literal_evidence": require_literal_evidence,
                "max_chunks": max_chunks,
                "agent_profile": agent_profile,
                "structured_context": structured_context,
            }
        )
        # Return a text that satisfies the validator expectations.
        return "Hallazgo trazable. Fuente(C1)"


def test_grounded_answer_adapter_passes_labeled_context() -> None:
    service = _FakeGroundedService()
    adapter = GroundedAnswerAdapter(service=service)  # type: ignore[arg-type]

    plan = RetrievalPlan(
        mode="literal_normativa",
        chunk_k=8,
        chunk_fetch_k=20,
        summary_k=3,
        require_literal_evidence=True,
        requested_standards=("ISO 9001",),
    )
    chunks = [
        EvidenceItem(source="C1", content="Texto de evidencia 1.", metadata={"row": {}}),
        EvidenceItem(source="C2", content="Texto de evidencia 2.", metadata={"row": {}}),
    ]
    summaries = [EvidenceItem(source="R1", content="Resumen 1.", metadata={"row": {}})]

    draft = asyncio.run(
        adapter.generate(
            query="Que exige ISO 9001?",
            scope_label="ISO 9001",
            plan=plan,
            chunks=chunks,
            summaries=summaries,
            working_memory={"tool": "result"},
            partial_answers=[],
            agent_profile=AgentProfile(profile_id="test-profile"),
        )
    )

    assert draft.text
    assert service.calls, "Expected adapter to call service.generate_answer()"
    call = service.calls[0]
    assert call["mode"] == "literal_normativa"
    assert call["require_literal_evidence"] is True
    assert call["agent_profile"] is not None

    context = "\n".join(call["context_chunks"])
    assert "[R1]" in context
    assert "[C1]" in context
    assert "[C2]" in context

    assert call["structured_context"] is not None
    assert "WORKING_MEMORY" in call["structured_context"]


def test_grounded_answer_adapter_balances_cross_scope_context() -> None:
    service = _FakeGroundedService()
    adapter = GroundedAnswerAdapter(service=service)  # type: ignore[arg-type]

    plan = RetrievalPlan(
        mode="cross_scope_analysis",
        chunk_k=35,
        chunk_fetch_k=140,
        summary_k=0,
        require_literal_evidence=False,
        requested_standards=("ISO 9001", "ISO 14001", "ISO 45001"),
    )
    chunks = [
        EvidenceItem(
            source="C1",
            content="ISO 9001 requisito 1",
            metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
        ),
        EvidenceItem(
            source="C2",
            content="ISO 9001 requisito 2",
            metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
        ),
        EvidenceItem(
            source="C3",
            content="ISO 9001 requisito 3",
            metadata={"row": {"metadata": {"source_standard": "ISO 9001"}}},
        ),
        EvidenceItem(
            source="C4",
            content="ISO 14001 requisito ambiental",
            metadata={"row": {"metadata": {"source_standard": "ISO 14001"}}},
        ),
        EvidenceItem(
            source="C5",
            content="ISO 45001 requisito SST",
            metadata={"row": {"metadata": {"source_standard": "ISO 45001"}}},
        ),
    ]

    draft = asyncio.run(
        adapter.generate(
            query="compara objetivos principales",
            scope_label="ISO",
            plan=plan,
            chunks=chunks,
            summaries=[],
            working_memory=None,
            partial_answers=None,
            agent_profile=AgentProfile(profile_id="test-profile"),
        )
    )

    assert draft.text
    assert service.calls
    context = "\n".join(service.calls[0]["context_chunks"])
    assert "ISO 9001" in context
    assert "ISO 14001" in context
    assert "ISO 45001" in context
