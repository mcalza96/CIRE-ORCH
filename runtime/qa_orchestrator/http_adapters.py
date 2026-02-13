from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from runtime.qa_orchestrator.grounded_answer_service import GroundedAnswerService
from runtime.qa_orchestrator.models import AnswerDraft, EvidenceItem, RetrievalPlan


@dataclass
class RagEngineRetrieverAdapter:
    base_url: str
    timeout_seconds: float = 8.0

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
    ) -> list[EvidenceItem]:
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "chunk_k": int(plan.chunk_k),
            "fetch_k": int(plan.chunk_fetch_k),
        }
        data = await self._post_json("/api/v1/retrieval/chunks", payload)
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
    ) -> list[EvidenceItem]:
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "summary_k": int(plan.summary_k),
        }
        data = await self._post_json("/api/v1/retrieval/summaries", payload)
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url.rstrip("/") + path
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return dict(response.json())

    @staticmethod
    def _to_evidence(items: Any) -> list[EvidenceItem]:
        if not isinstance(items, list):
            return []
        out: list[EvidenceItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            raw_candidate = item.get("metadata")
            if isinstance(raw_candidate, dict):
                raw_metadata: dict[str, Any] = raw_candidate
            else:
                raw_metadata = {}
            out.append(
                EvidenceItem(
                    source=str(item.get("source") or "C1"),
                    content=content,
                    score=float(item.get("score") or 0.0),
                    metadata={"row": {"content": content, "metadata": raw_metadata, "similarity": item.get("score")}},
                )
            )
        return out


@dataclass
class GroundedAnswerAdapter:
    service: GroundedAnswerService

    async def generate(
        self,
        query: str,
        scope_label: str,
        plan: RetrievalPlan,
        chunks: list[EvidenceItem],
        summaries: list[EvidenceItem],
    ) -> AnswerDraft:
        del scope_label
        context_chunks = [item.content for item in [*chunks, *summaries] if item.content]
        text = await self.service.generate_answer(query=query, context_chunks=context_chunks)
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])
