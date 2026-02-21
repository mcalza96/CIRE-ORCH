from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.agent.components.query_decomposer import HybridSubqueryPlanner, LLMSubqueryPlanner
from app.agent.types.interfaces import SubqueryPlanningContext, SubqueryPlanner


@dataclass
class _StaticPlanner(SubqueryPlanner):
    items: list[dict]

    async def plan(self, context: SubqueryPlanningContext) -> list[dict]:
        del context
        return list(self.items)


def _ctx(*, enabled: bool) -> SubqueryPlanningContext:
    return SubqueryPlanningContext(
        query="analiza impacto entre ISO 9001 e ISO 14001",
        requested_standards=("ISO 9001", "ISO 14001"),
        max_queries=6,
        decomposition_policy={"light_llm_enabled": enabled},
    )


def test_hybrid_decomposer_skips_llm_when_mode_disabled() -> None:
    hybrid = HybridSubqueryPlanner(
        deterministic=_StaticPlanner(items=[{"id": "d1", "query": "det"}]),
        llm=_StaticPlanner(items=[{"id": "l1", "query": "llm"}]),
    )
    out = asyncio.run(hybrid.plan(_ctx(enabled=False)))
    assert [item["id"] for item in out] == ["d1"]


def test_hybrid_decomposer_merges_llm_for_complex_queries() -> None:
    hybrid = HybridSubqueryPlanner(
        deterministic=_StaticPlanner(items=[{"id": "d1", "query": "det"}]),
        llm=_StaticPlanner(items=[{"id": "l1", "query": "llm"}]),
    )
    out = asyncio.run(hybrid.plan(_ctx(enabled=True)))
    ids = {item["id"] for item in out}
    assert {"d1", "l1"}.issubset(ids)
    assert any(item_id.startswith("scope_") for item_id in ids)


def test_llm_decomposer_fails_safe_on_provider_error() -> None:
    planner = LLMSubqueryPlanner(timeout_ms=50)

    class _BrokenClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    del kwargs
                    raise RuntimeError("provider_error")

    planner._client = _BrokenClient()  # type: ignore[attr-defined]
    out = asyncio.run(planner.plan(_ctx(enabled=True)))
    assert out == []
