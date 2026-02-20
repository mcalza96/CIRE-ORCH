from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agent.models import RetrievalPlan
from app.agent.retrieval_flow import RetrievalFlow


def _plan() -> RetrievalPlan:
    return RetrievalPlan(
        mode="comparativa",
        chunk_k=12,
        chunk_fetch_k=60,
        summary_k=0,
        require_literal_evidence=False,
        requested_standards=("ISO 9001", "ISO 14001"),
    )


@pytest.mark.asyncio
async def test_execute_uses_comprehensive_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_RETRIEVAL_COMPREHENSIVE_ENABLED", True
    )

    contract_client = AsyncMock()
    contract_client.comprehensive = AsyncMock(
        return_value={
            "items": [{"content": "ok", "source": "C1", "score": 0.9, "metadata": {}}],
            "trace": {"score_space": "mixed"},
        }
    )

    flow = RetrievalFlow(contract_client=contract_client)
    out = await flow.execute(
        query="compara iso 9001 vs iso 14001",
        tenant_id="t1",
        collection_id=None,
        plan=_plan(),
        user_id="u1",
    )

    assert len(out) == 1
    assert out[0].content == "ok"
    assert flow.last_diagnostics is not None
    assert flow.last_diagnostics.strategy == "comprehensive"
    contract_client.comprehensive.assert_awaited_once()
    kwargs = contract_client.comprehensive.await_args.kwargs
    assert kwargs["query"] == "compara iso 9001 vs iso 14001"
    policy = kwargs.get("retrieval_policy")
    assert isinstance(policy, dict)
    assert "min_score" in policy
    assert policy.get("noise_reduction") is True


@pytest.mark.asyncio
async def test_execute_fails_when_comprehensive_endpoint_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.agent.retrieval_flow.settings.ORCH_RETRIEVAL_COMPREHENSIVE_ENABLED", True
    )

    contract_client = AsyncMock()
    contract_client.comprehensive = AsyncMock(side_effect=RuntimeError("404 not found"))

    flow = RetrievalFlow(contract_client=contract_client)

    with pytest.raises(RuntimeError, match="comprehensive_retrieval_failed"):
        await flow.execute(
            query="compara iso 9001 vs iso 14001",
            tenant_id="t1",
            collection_id=None,
            plan=_plan(),
            user_id="u1",
        )
