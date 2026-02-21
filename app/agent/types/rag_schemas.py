from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TimeRangeFilter(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: Literal["created_at", "updated_at"]
    from_: str | None = Field(default=None, alias="from")
    to: str | None = None


class ScopeFilters(BaseModel):
    model_config = ConfigDict(extra="ignore")

    metadata: dict[str, Any] | None = None
    time_range: TimeRangeFilter | None = None
    source_standard: str | None = None
    source_standards: list[str] | None = None


class PlannedSubQueryPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int
    query: str
    dependency_id: int | None = None
    target_relations: list[str] | None = None
    target_node_types: list[str] | None = None
    is_deep: bool = False


class RetrievalPlanPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    is_multihop: bool = False
    execution_mode: Literal["parallel", "sequential"] = "parallel"
    sub_queries: list[PlannedSubQueryPayload] = Field(default_factory=list)
    fallback_reason: str | None = None


class SubQueryRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    query: str
    k: int | None = None
    fetch_k: int | None = None
    filters: ScopeFilters | None = None


class MergeOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # RAG contract currently supports RRF. Keep type strict to avoid 422s.
    strategy: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1, le=500)
    top_k: int = Field(default=12, ge=1, le=100)


class MultiQueryRetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tenant_id: str
    collection_id: str | None = None
    queries: list[SubQueryRequest]
    merge: MergeOptions = Field(default_factory=MergeOptions)
