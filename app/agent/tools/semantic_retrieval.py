from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.agent.models import EvidenceItem, RetrievalDiagnostics, RetrievalPlan, ToolResult
from app.agent.tools.base import ToolRuntimeContext


@dataclass(frozen=True)
class SemanticRetrievalTool:
    name: str = "semantic_retrieval"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        query = str(
            payload.get("query") or state.get("working_query") or state.get("user_query") or ""
        )
        tenant_id = str(state.get("tenant_id") or "")
        collection_id = state.get("collection_id")
        user_id = state.get("user_id")
        request_id = state.get("request_id")
        correlation_id = state.get("correlation_id")
        plan = state.get("retrieval_plan")
        if not isinstance(plan, RetrievalPlan):
            return ToolResult(
                tool=self.name,
                ok=False,
                error="missing_retrieval_plan",
            )

        collection = collection_id if isinstance(collection_id, str) else None
        user = user_id if isinstance(user_id, str) else None
        req_id = request_id if isinstance(request_id, str) else None
        corr_id = correlation_id if isinstance(correlation_id, str) else None

        timings_ms: dict[str, float] = {}
        chunks: list[EvidenceItem] = []
        summaries: list[EvidenceItem] = []

        should_fetch_chunks = int(plan.chunk_k or 0) > 0
        should_fetch_summaries = int(plan.summary_k or 0) > 0

        if should_fetch_chunks and should_fetch_summaries:
            t0 = time.perf_counter()
            chunks, summaries = await asyncio.gather(
                context.retriever.retrieve_chunks(
                    query=query,
                    tenant_id=tenant_id,
                    collection_id=collection,
                    plan=plan,
                    user_id=user,
                    request_id=req_id,
                    correlation_id=corr_id,
                ),
                context.retriever.retrieve_summaries(
                    query=query,
                    tenant_id=tenant_id,
                    collection_id=collection,
                    plan=plan,
                    user_id=user,
                    request_id=req_id,
                    correlation_id=corr_id,
                ),
            )
            timings_ms["parallel_retrieval"] = round((time.perf_counter() - t0) * 1000.0, 2)
        elif should_fetch_chunks:
            t0 = time.perf_counter()
            chunks = await context.retriever.retrieve_chunks(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection,
                plan=plan,
                user_id=user,
                request_id=req_id,
                correlation_id=corr_id,
            )
            timings_ms["chunks_only"] = round((time.perf_counter() - t0) * 1000.0, 2)
        elif should_fetch_summaries:
            t0 = time.perf_counter()
            summaries = await context.retriever.retrieve_summaries(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection,
                plan=plan,
                user_id=user,
                request_id=req_id,
                correlation_id=corr_id,
            )
            timings_ms["summaries_only"] = round((time.perf_counter() - t0) * 1000.0, 2)

        diagnostics = getattr(context.retriever, "last_retrieval_diagnostics", None)
        retrieval = (
            diagnostics
            if isinstance(diagnostics, RetrievalDiagnostics)
            else RetrievalDiagnostics(contract="legacy", strategy="tool_semantic_retrieval")
        )
        evidence: list[EvidenceItem] = [*list(chunks), *list(summaries)]
        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "chunk_count": len(chunks),
                "summary_count": len(summaries),
                "strategy": retrieval.strategy,
                "contract": retrieval.contract,
                "partial": bool(retrieval.partial),
                "parallel": should_fetch_chunks and should_fetch_summaries,
            },
            metadata={
                "retrieval": retrieval,
                "chunks": list(chunks),
                "summaries": list(summaries),
                "timings_ms": timings_ms,
            },
            evidence=evidence,
        )
