from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.agent.interfaces import SubqueryPlanningContext, SubqueryPlanner
from app.agent.retrieval_planner import build_deterministic_subqueries, extract_clause_refs
from app.infrastructure.config import settings
from app.domain.rag_schemas import SubQueryRequest


logger = structlog.get_logger(__name__)


def _extract_scope_filters(item: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(item, dict):
        return ()
    raw_filters = item.get("filters")
    if not isinstance(raw_filters, dict):
        return ()
    one = str(raw_filters.get("source_standard") or "").strip()
    if one:
        return (one.upper(),)
    many = raw_filters.get("source_standards")
    if not isinstance(many, list):
        return ()
    out: list[str] = []
    for value in many:
        text = str(value or "").strip().upper()
        if text:
            out.append(text)
    return tuple(out)


def _ensure_scope_coverage(
    *,
    context: SubqueryPlanningContext,
    subqueries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requested = [
        str(scope or "").strip().upper()
        for scope in context.requested_standards
        if str(scope).strip()
    ]
    if len(requested) < 2:
        return subqueries[: max(1, context.max_queries)]

    present_scopes: set[str] = set()
    for item in subqueries:
        for scope in _extract_scope_filters(item):
            present_scopes.add(scope)

    missing = [scope for scope in requested if scope not in present_scopes]
    if not missing:
        return subqueries[: max(1, context.max_queries)]

    fillers = build_deterministic_subqueries(
        query=context.query,
        requested_standards=tuple(missing),
        max_queries=len(missing),
        mode=context.mode,
        require_literal_evidence=context.require_literal_evidence,
        include_semantic_tail=context.include_semantic_tail,
        profile=context.profile,
    )

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*subqueries, *fillers]:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "").strip() or str(item.get("query") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)

    max_queries = max(1, context.max_queries)
    if len(merged) <= max_queries:
        return merged

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for scope in requested:
        for item in merged:
            scope_filters = _extract_scope_filters(item)
            if scope not in scope_filters:
                continue
            key = str(item.get("id") or "").strip() or str(item.get("query") or "").strip().lower()
            if not key or key in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(key)
            break

    for item in merged:
        if len(selected) >= max_queries:
            break
        key = str(item.get("id") or "").strip() or str(item.get("query") or "").strip().lower()
        if not key or key in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(key)

    return selected[:max_queries]


class SubqueryPlanPayload(BaseModel):
    subqueries: list[SubQueryRequest] = Field(default_factory=list)


@dataclass
class DeterministicSubqueryPlanner(SubqueryPlanner):
    async def plan(self, context: SubqueryPlanningContext) -> list[dict[str, Any]]:
        planned = build_deterministic_subqueries(
            query=context.query,
            requested_standards=context.requested_standards,
            max_queries=context.max_queries,
            mode=context.mode,
            require_literal_evidence=context.require_literal_evidence,
            include_semantic_tail=context.include_semantic_tail,
            profile=context.profile,
        )
        return _ensure_scope_coverage(context=context, subqueries=planned)


@dataclass
class LLMSubqueryPlanner(SubqueryPlanner):
    timeout_ms: int = 600

    def __post_init__(self) -> None:
        self._client = (
            AsyncOpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1",
            )
            if settings.GROQ_API_KEY
            else None
        )

    async def plan(self, context: SubqueryPlanningContext) -> list[dict[str, Any]]:
        if self._client is None:
            logger.warning("light_planner_disabled_missing_key")
            return []

        model = (
            settings.ORCH_LIGHT_PLANNER_MODEL
            or settings.ORCH_PLANNER_MODEL
            or settings.GROQ_MODEL_ORCHESTRATION
        )

        system = (
            "You are a retrieval subquery planner. "
            'Return JSON only with {"subqueries": [...]}. No extra text.'
        )
        user = (
            f"Query: {context.query}\n"
            f"Requested standards: {', '.join(context.requested_standards) if context.requested_standards else '(none)'}\n"
            f"Max subqueries: {context.max_queries}\n"
            "Constraints: each subquery item must contain id, query, optional filters."
        )

        try:
            timeout = max(0.1, float(self.timeout_ms) / 1000.0)
            completion = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ),
                timeout=timeout,
            )
            raw = (completion.choices[0].message.content or "").strip()
            payload = SubqueryPlanPayload.model_validate(json.loads(raw))
            out = [item.model_dump(by_alias=True, exclude_none=True) for item in payload.subqueries]
            return out[: max(1, context.max_queries)]
        except Exception as exc:
            logger.warning("light_planner_failed_fallback", error=str(exc))
            return []


@dataclass
class HybridSubqueryPlanner(SubqueryPlanner):
    deterministic: SubqueryPlanner
    llm: SubqueryPlanner | None = None

    @classmethod
    def from_settings(cls) -> "HybridSubqueryPlanner":
        det = DeterministicSubqueryPlanner()
        llm: SubqueryPlanner | None = None
        if bool(getattr(settings, "ORCH_LIGHT_PLANNER_ENABLED", False)):
            llm = LLMSubqueryPlanner(timeout_ms=int(settings.ORCH_LIGHT_PLANNER_TIMEOUT_MS or 600))
        return cls(deterministic=det, llm=llm)

    def _is_complex(self, context: SubqueryPlanningContext) -> bool:
        query = str(context.query or "").lower()
        clause_refs = extract_clause_refs(context.query, profile=context.profile)
        high_entropy_tokens = (
            "impacto",
            "relacion",
            "relación",
            "difer",
            "versus",
            "vs",
            "interaccion",
            "interacción",
            "por que",
        )
        if len(context.requested_standards) >= 2:
            return True
        if len(clause_refs) >= 2:
            return True
        return any(token in query for token in high_entropy_tokens)

    async def plan(self, context: SubqueryPlanningContext) -> list[dict[str, Any]]:
        deterministic = await self.deterministic.plan(context)

        if self.llm is None:
            return deterministic

        mode_enabled = bool(context.decomposition_policy.get("light_llm_enabled", False))
        if not mode_enabled:
            return deterministic

        needs_llm = (not deterministic) or self._is_complex(context)
        if not needs_llm:
            return deterministic

        llm_subqueries = await self.llm.plan(context)
        if not llm_subqueries:
            return deterministic

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in [*deterministic, *llm_subqueries]:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id") or "").strip() or str(item.get("query") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

        covered = _ensure_scope_coverage(context=context, subqueries=merged)
        return covered[: max(1, context.max_queries)]
