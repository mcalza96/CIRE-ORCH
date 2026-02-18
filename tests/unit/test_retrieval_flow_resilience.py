from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agent.models import RetrievalPlan
from app.agent.retrieval_flow import RetrievalFlow


class _FakePlanner:
    async def plan(self, context):
        del context
        return [{"id": "q1", "query": "subquery"}]


class _TwoQueryPlanner:
    async def plan(self, context):
        del context
        return [
            {"id": "q1", "query": "subquery 1", "filters": {"source_standard": "ISO 9001"}},
            {"id": "q2", "query": "subquery 2", "filters": {"source_standard": "ISO 14001"}},
        ]


def _plan() -> RetrievalPlan:
    return RetrievalPlan(
        mode="comparativa",
        chunk_k=12,
        chunk_fetch_k=60,
        summary_k=0,
        require_literal_evidence=False,
        requested_standards=("ISO 9001", "ISO 14001"),
    )


def _set_retrieval_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_PRIMARY", True)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_MIN_ITEMS", 3)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_REFINE", True)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_EVALUATOR", False)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTIHOP_FALLBACK", False)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_ENABLED", False)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_STEP_BACK", True)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_MAX_MISSING", 2)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_TOP_N", 12)
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_DETERMINISTIC_SUBQUERY_SEMANTIC_TAIL", False
    )
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MIN_SCORE_BACKSTOP_ENABLED", False)
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_TIMEOUT_RETRIEVAL_MULTI_QUERY_MS", 5000
    )
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_TIMEOUT_RETRIEVAL_HYBRID_MS", 5000)
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_TIMEOUT_RETRIEVAL_COVERAGE_REPAIR_MS", 5000
    )
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_CLIENT_FANOUT_ENABLED", False
    )


@pytest.mark.asyncio
async def test_primary_multi_query_timeout_falls_back_to_hybrid(monkeypatch: pytest.MonkeyPatch):
    _set_retrieval_defaults(monkeypatch)

    contract_client = AsyncMock()
    contract_client.multi_query = AsyncMock(
        side_effect=RuntimeError("retrieval_timeout:multi_query_primary")
    )
    contract_client.hybrid = AsyncMock(
        return_value={
            "items": [{"content": "hybrid ok", "source": "H1", "score": 0.9}],
            "trace": {"hybrid": True},
        }
    )

    flow = RetrievalFlow(contract_client=contract_client, subquery_planner=_FakePlanner())
    items = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert len(items) == 1
    assert items[0].content == "hybrid ok"
    assert flow.last_diagnostics is not None
    assert flow.last_diagnostics.strategy == "hybrid"
    assert "multi_query_fallback_error" in flow.last_diagnostics.trace


@pytest.mark.asyncio
async def test_multi_query_success_merges_with_hybrid(monkeypatch: pytest.MonkeyPatch):
    _set_retrieval_defaults(monkeypatch)

    contract_client = AsyncMock()
    contract_client.multi_query = AsyncMock(
        return_value={"items": [{"content": "p1", "source": "M1", "score": 0.6}]}
    )
    contract_client.hybrid = AsyncMock(
        return_value={
            "items": [{"content": "hybrid after refine fail", "source": "H2", "score": 0.8}],
            "trace": {"hybrid": True},
        }
    )

    flow = RetrievalFlow(contract_client=contract_client, subquery_planner=_FakePlanner())
    items = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert len(items) >= 1
    assert any(it.content == "hybrid after refine fail" for it in items)
    assert any(it.content == "p1" for it in items)
    assert flow.last_diagnostics is not None
    assert flow.last_diagnostics.strategy == "multi_query"


@pytest.mark.asyncio
async def test_coverage_step_back_error_is_best_effort(monkeypatch: pytest.MonkeyPatch):
    _set_retrieval_defaults(monkeypatch)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_PRIMARY", False)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_ENABLED", True)

    contract_client = AsyncMock()
    contract_client.hybrid = AsyncMock(
        return_value={
            "items": [
                {
                    "content": "item base",
                    "source": "B1",
                    "score": 0.7,
                    "metadata": {"row": {"source_standard": "ISO 9001", "content": "item base"}},
                }
            ],
            "trace": {},
        }
    )
    contract_client.multi_query = AsyncMock(
        side_effect=[
            {
                "items": [
                    {
                        "content": "repair item",
                        "source": "R1",
                        "score": 0.6,
                        "metadata": {
                            "row": {"source_standard": "ISO 9001", "content": "repair item"}
                        },
                    }
                ]
            },
            RuntimeError("retrieval_timeout:coverage_gate_step_back_multi_query"),
        ]
    )

    flow = RetrievalFlow(contract_client=contract_client, subquery_planner=_FakePlanner())
    items = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert len(items) >= 1
    assert flow.last_diagnostics is not None
    assert flow.last_diagnostics.strategy == "multi_query"
    coverage_gate = flow.last_diagnostics.trace.get("coverage_gate", {})
    assert "error" in coverage_gate or "step_back_error" in coverage_gate


@pytest.mark.asyncio
async def test_multi_query_fallback_skips_when_budget_too_low(monkeypatch: pytest.MonkeyPatch):
    _set_retrieval_defaults(monkeypatch)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_ENABLED", False)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_TIMEOUT_EXECUTE_TOOL_MS", 1000)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_RETRIEVAL_MIN_MQ_BUDGET_MS", 5_000)

    contract_client = AsyncMock()
    contract_client.hybrid = AsyncMock(return_value={"items": [], "trace": {}})
    contract_client.multi_query = AsyncMock(return_value={"items": [{"content": "mq"}]})

    flow = RetrievalFlow(contract_client=contract_client, subquery_planner=_FakePlanner())
    items = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert items == []
    assert contract_client.multi_query.await_count == 0
    assert flow.last_diagnostics is not None
    hybrid_trace = flow.last_diagnostics.trace.get("hybrid_trace", {})
    assert "multi_query_fallback_skipped_by_budget" in hybrid_trace


@pytest.mark.asyncio
async def test_multi_query_client_fanout_uses_parallel_hybrid_calls(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_retrieval_defaults(monkeypatch)
    monkeypatch.setattr("app.agent.retrieval_flow.settings.ORCH_COVERAGE_GATE_ENABLED", False)
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_CLIENT_FANOUT_ENABLED", True
    )
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_CLIENT_FANOUT_MAX_PARALLEL", 2
    )
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_CLIENT_FANOUT_PER_QUERY_TIMEOUT_MS",
        3000,
    )
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_MULTI_QUERY_CLIENT_FANOUT_RERANK_ENABLED", False
    )

    contract_client = AsyncMock()
    contract_client.hybrid = AsyncMock(
        side_effect=[
            {"items": [], "trace": {}},
            {"items": [{"content": "fanout iso 9001", "source": "F1", "score": 0.7}]},
            {"items": [{"content": "fanout iso 14001", "source": "F2", "score": 0.69}]},
        ]
    )
    contract_client.multi_query = AsyncMock(return_value={"items": [{"content": "should_not_run"}]})

    flow = RetrievalFlow(contract_client=contract_client, subquery_planner=_TwoQueryPlanner())
    items = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert len(items) == 2
    assert {it.content for it in items} == {"fanout iso 9001", "fanout iso 14001"}
    assert contract_client.multi_query.await_count == 0
    assert flow.last_diagnostics is not None
    assert flow.last_diagnostics.trace.get("multi_query_execution_mode") == "client_fanout"
    mq_trace = flow.last_diagnostics.trace.get("multi_query_trace", {})
    assert bool(mq_trace.get("fanout")) is True
