from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.cartridges.models import AgentProfile


@dataclass(frozen=True)
class SubqueryPlanningContext:
    query: str
    requested_standards: tuple[str, ...] = ()
    max_queries: int = 6
    mode: str | None = None
    require_literal_evidence: bool | None = None
    include_semantic_tail: bool = True
    profile: AgentProfile | None = None
    decomposition_policy: dict[str, Any] = field(default_factory=dict)


class SubqueryPlanner(Protocol):
    async def plan(self, context: SubqueryPlanningContext) -> list[dict[str, Any]]: ...
