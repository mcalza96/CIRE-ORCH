from __future__ import annotations

import pytest

from app.agent.models import AnswerDraft, EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.profiles.models import AgentProfile
from app.infrastructure.config import settings
from app.graph.universal.steps import aggregate_subqueries_node


class MockAnswerGenerator:
    async def generate(self, *args, **kwargs) -> AnswerDraft:
        return AnswerDraft(
            text="Resumen generado por LLM",
            mode="concisa_y_directa",
            evidence=[]
        )


class MockComponents:
    answer_generator = MockAnswerGenerator()
    retriever = None
    validator = None
    tools = None
    def _runtime_context(self): return None



@pytest.mark.asyncio
async def test_aggregate_subqueries_from_group_items(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", True)
    state = {
        "retrieval": RetrievalDiagnostics(contract="advanced", strategy="multi_query", trace={}),
        "subquery_groups": [
            {
                "id": "q1",
                "query": "introduccion del libro",
                "items": [
                    {
                        "source": "C1",
                        "content": "La introduccion presenta el objetivo del libro y su alcance.",
                        "score": 0.88,
                    }
                ],
            }
        ],
        "chunks": [],
    }

    updates = await aggregate_subqueries_node(state, MockComponents())

    partials = updates.get("partial_answers")
    assert isinstance(partials, list)
    assert len(partials) == 1
    assert partials[0].get("status") == "ok"
    assert "C1" in list(partials[0].get("evidence_sources") or [])
    assert partials[0].get("summary") == "Resumen generado por LLM"

@pytest.mark.asyncio
async def test_aggregate_subqueries_assigns_chunks_when_group_items_missing(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ORCH_SUBQUERY_GROUPED_MAP_REDUCE_ENABLED", True)
    state = {
        "retrieval": RetrievalDiagnostics(contract="advanced", strategy="multi_query", trace={}),
        "subquery_groups": [{"id": "q2", "query": "introduccion"}],
        "chunks": [
            EvidenceItem(
                source="C5", content="Introduccion a la matematica y lenguaje formal.", score=0.72
            ),
            EvidenceItem(source="C8", content="Tabla de contenido del capitulo 2.", score=0.1),
        ],
    }

    updates = await aggregate_subqueries_node(state, MockComponents())

    partials = updates.get("partial_answers")
    assert isinstance(partials, list)
    assert len(partials) == 1
    assert "C5" in list(partials[0].get("evidence_sources") or [])
    assert partials[0].get("summary") == "Resumen generado por LLM"
