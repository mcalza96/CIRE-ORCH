from __future__ import annotations

import asyncio
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
        query = str(payload.get("query") or state.get("working_query") or state.get("user_query") or "")
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

        chunks_task = asyncio.create_task(
            context.retriever.retrieve_chunks(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id if isinstance(collection_id, str) else None,
                plan=plan,
                user_id=user_id if isinstance(user_id, str) else None,
                request_id=request_id if isinstance(request_id, str) else None,
                correlation_id=correlation_id if isinstance(correlation_id, str) else None,
            )
        )
        summaries_task = asyncio.create_task(
            context.retriever.retrieve_summaries(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id if isinstance(collection_id, str) else None,
                plan=plan,
                user_id=user_id if isinstance(user_id, str) else None,
                request_id=request_id if isinstance(request_id, str) else None,
                correlation_id=correlation_id if isinstance(correlation_id, str) else None,
            )
        )
        chunks, summaries = await asyncio.gather(chunks_task, summaries_task)
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
            },
            metadata={
                "retrieval": retrieval,
                "chunks": list(chunks),
                "summaries": list(summaries),
            },
            evidence=evidence,
        )
