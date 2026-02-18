from __future__ import annotations

import asyncio

from app.agent.models import EvidenceItem, RetrievalPlan
from app.agent.tools.base import ToolRuntimeContext
from app.agent.tools.semantic_retrieval import SemanticRetrievalTool


class _DummyRetriever:
    def __init__(self) -> None:
        self.chunk_calls = 0
        self.summary_calls = 0

    async def retrieve_chunks(self, *args, **kwargs) -> list[EvidenceItem]:
        del args, kwargs
        self.chunk_calls += 1
        return [EvidenceItem(source="C1", content="chunk")]

    async def retrieve_summaries(self, *args, **kwargs) -> list[EvidenceItem]:
        del args, kwargs
        self.summary_calls += 1
        return [EvidenceItem(source="R1", content="summary")]


class _DummyAnswerGenerator:
    async def generate(self, *args, **kwargs):
        del args, kwargs
        return None


class _DummyValidator:
    def validate(self, *args, **kwargs):
        del args, kwargs
        return None


def _run_tool(plan: RetrievalPlan) -> tuple[object, _DummyRetriever]:
    retriever = _DummyRetriever()
    tool = SemanticRetrievalTool()
    context = ToolRuntimeContext(
        retriever=retriever,
        answer_generator=_DummyAnswerGenerator(),
        validator=_DummyValidator(),
    )
    state = {
        "working_query": "Que dice ISO 9001",
        "tenant_id": "t1",
        "collection_id": None,
        "retrieval_plan": plan,
    }
    result = asyncio.run(tool.run({}, state=state, context=context))
    return result, retriever


def test_semantic_retrieval_skips_summaries_when_summary_k_is_zero() -> None:
    result, retriever = _run_tool(
        RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=25, summary_k=0)
    )
    assert retriever.chunk_calls == 1
    assert retriever.summary_calls == 0
    assert result.ok is True
    assert result.output.get("parallel") is False
    assert result.output.get("summary_count") == 0
    assert "chunks_only" in dict(result.metadata.get("timings_ms") or {})


def test_semantic_retrieval_parallel_fetch_when_chunks_and_summaries_enabled() -> None:
    result, retriever = _run_tool(
        RetrievalPlan(mode="explicativa", chunk_k=5, chunk_fetch_k=25, summary_k=2)
    )
    assert retriever.chunk_calls == 1
    assert retriever.summary_calls == 1
    assert result.ok is True
    assert result.output.get("parallel") is True
    assert result.output.get("chunk_count") == 1
    assert result.output.get("summary_count") == 1
    assert "parallel_retrieval" in dict(result.metadata.get("timings_ms") or {})
