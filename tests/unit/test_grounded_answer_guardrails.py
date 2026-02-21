import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.agent.http_adapters import GroundedAnswerAdapter
from app.agent.models import RetrievalPlan, EvidenceItem, QueryMode
from typing import cast


@pytest.fixture
def mock_service():
    service = MagicMock()
    service.generate_answer = AsyncMock(return_value="Mocked Answer")
    return service


@pytest.fixture
def adapter(mock_service):
    return GroundedAnswerAdapter(service=mock_service)


def create_retrieval_plan(mode="comparativa", standards=(), require_literal=False):
    return RetrievalPlan(
        mode=cast(QueryMode, mode),
        chunk_k=12,
        chunk_fetch_k=60,
        summary_k=5,
        require_literal_evidence=require_literal,
        requested_standards=standards,
    )


@pytest.mark.asyncio
async def test_generate_best_effort_recency_ordering(adapter, mock_service):
    plan = create_retrieval_plan(mode="comparativa")

    # Evidence with different timestamps
    chunks = [
        EvidenceItem(
            source="C1",
            content="old",
            metadata={"row": {"metadata": {"updated_at": "2020-01-01T00:00:00Z"}}},
        ),
        EvidenceItem(
            source="C2",
            content="new",
            metadata={"row": {"metadata": {"updated_at": "2024-01-01T00:00:00Z"}}},
        ),
    ]

    await adapter.generate(query="q", scope_label="s", plan=plan, chunks=chunks, summaries=[])

    # Check what was passed to service.generate_answer
    args, kwargs = mock_service.generate_answer.call_args
    context_chunks = kwargs["context_chunks"]

    # C2 should come before C1 because it's newer
    assert "C2" in context_chunks[0]
    assert "C1" in context_chunks[1]


@pytest.mark.asyncio
async def test_generate_adds_traceability_note_when_bridge_missing(adapter, mock_service):
    plan = create_retrieval_plan(
        mode="comparativa", standards=("ISO 9001", "ISO 14001"), require_literal=False
    )

    # Two chunks, each mentioning only one standard
    chunks = [
        EvidenceItem(
            source="C1",
            content="About ISO 9001 requirements",
            metadata={
                "row": {"content": "About ISO 9001", "metadata": {"source_standard": "ISO 9001"}}
            },
        ),
        EvidenceItem(
            source="C2",
            content="About ISO 14001 requirements",
            metadata={
                "row": {"content": "About ISO 14001", "metadata": {"source_standard": "ISO 14001"}}
            },
        ),
    ]

    # Answer mentions both
    mock_service.generate_answer = AsyncMock(
        return_value="The relation between ISO 9001 and ISO 14001 is..."
    )

    result = await adapter.generate(
        query="comparing standards", scope_label="s", plan=plan, chunks=chunks, summaries=[]
    )

    assert "Nota de trazabilidad" in result.text
    assert "no hay un fragmento unico que las vincule" in result.text


@pytest.mark.asyncio
async def test_generate_no_note_if_bridge_chunk_exists(adapter, mock_service):
    plan = create_retrieval_plan(mode="comparativa", standards=("ISO 9001", "ISO 14001"))

    # One chunk mentioning both
    chunks = [
        EvidenceItem(
            source="C1",
            content="ISO 9001 and ISO 14001 together",
            metadata={"row": {"content": "ISO 9001 and ISO 14001", "metadata": {}}},
        ),
    ]

    mock_service.generate_answer = AsyncMock(
        return_value="Answer mentioning ISO 9001 and ISO 14001"
    )

    result = await adapter.generate(
        query="q", scope_label="s", plan=plan, chunks=chunks, summaries=[]
    )

    assert "Nota de trazabilidad" not in result.text


@pytest.mark.asyncio
async def test_generate_no_note_if_already_disclosed(adapter, mock_service):
    plan = create_retrieval_plan(mode="comparativa", standards=("ISO 9001", "ISO 14001"))

    chunks = [
        EvidenceItem(
            source="C1",
            content="ISO 9001",
            metadata={"row": {"content": "ISO 9001", "metadata": {}}},
        ),
        EvidenceItem(
            source="C2",
            content="ISO 14001",
            metadata={"row": {"content": "ISO 14001", "metadata": {}}},
        ),
    ]

    # Answer already contains disclosure keyword
    mock_service.generate_answer = AsyncMock(
        return_value="Basado en una inferencia de ISO 9001 y ISO 14001..."
    )

    result = await adapter.generate(
        query="q", scope_label="s", plan=plan, chunks=chunks, summaries=[]
    )

    assert "Nota de trazabilidad" not in result.text


@pytest.mark.asyncio
async def test_generate_defense_in_depth_appends_references_if_missing(adapter, mock_service):
    plan = create_retrieval_plan(mode="comparativa", require_literal=True)

    chunks = [EvidenceItem(source="C1", content="text")]

    # Provider ignores instructions and returns no [C1] markers
    mock_service.generate_answer = AsyncMock(return_value="Direct answer without citations")

    result = await adapter.generate(
        query="q", scope_label="s", plan=plan, chunks=chunks, summaries=[]
    )

    assert "Referencias revisadas: C1" in result.text


@pytest.mark.asyncio
async def test_generate_comparativa_no_evidence_reports_each_scope(adapter, mock_service):
    plan = create_retrieval_plan(
        mode="comparativa", standards=("ISO 9001", "ISO 14001", "ISO 45001")
    )
    mock_service.generate_answer = AsyncMock(
        return_value="No encontrado explicitamente en el contexto recuperado"
    )

    result = await adapter.generate(query="q", scope_label="s", plan=plan, chunks=[], summaries=[])

    assert "**ISO 9001**" in result.text
    assert "**ISO 14001**" in result.text
    assert "**ISO 45001**" in result.text


@pytest.mark.asyncio
async def test_generate_comparativa_with_evidence_replaces_generic_fallback(adapter, mock_service):
    plan = create_retrieval_plan(mode="comparativa", standards=("ISO 9001", "ISO 14001"))
    chunks = [
        EvidenceItem(
            source="C1", content="Texto de evidencia ISO 9001", metadata={"row": {"metadata": {}}}
        ),
        EvidenceItem(
            source="C2", content="Texto de evidencia ISO 14001", metadata={"row": {"metadata": {}}}
        ),
    ]
    mock_service.generate_answer = AsyncMock(
        return_value="No encontrado explicitamente en el contexto recuperado"
    )

    result = await adapter.generate(
        query="q", scope_label="s", plan=plan, chunks=chunks, summaries=[]
    )

    assert "No encontrado explicitamente" not in result.text
    assert "Fuente (C1)" in result.text
