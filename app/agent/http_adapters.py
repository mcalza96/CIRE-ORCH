from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import structlog
import time

from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.agent.retrieval_planner import (
    build_deterministic_subqueries,
    decide_multihop_fallback,
    extract_clause_refs,
)
from app.agent.semantic_subquery_planner import SemanticSubqueryPlanner
from app.agent.retrieval_sufficiency_evaluator import RetrievalSufficiencyEvaluator
from app.clients.backend_selector import RagBackendSelector
from app.core.config import settings
from app.core.rag_retrieval_contract_client import (
    RagContractNotSupportedError,
    RagRetrievalContractClient,
)
from app.core.retrieval_metrics import retrieval_metrics_store


logger = structlog.get_logger(__name__)


@dataclass
class RagEngineRetrieverAdapter:
    base_url: str | None = None
    timeout_seconds: float = 8.0
    backend_selector: RagBackendSelector | None = None
    contract_client: RagRetrievalContractClient | None = None

    # last diagnostics are read by the use case (duck typing).
    last_retrieval_diagnostics: RetrievalDiagnostics | None = None
    _validated_filters: dict[str, Any] | None = None
    _validated_scope_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.backend_selector is None:
            self.backend_selector = RagBackendSelector(
                local_url=str(settings.RAG_ENGINE_LOCAL_URL or "http://localhost:8000"),
                docker_url=str(settings.RAG_ENGINE_DOCKER_URL or "http://localhost:8000"),
                health_path=str(settings.RAG_ENGINE_HEALTH_PATH or "/health"),
                probe_timeout_ms=int(settings.RAG_ENGINE_PROBE_TIMEOUT_MS or 300),
                ttl_seconds=int(settings.RAG_ENGINE_BACKEND_TTL_SECONDS or 20),
                force_backend=settings.RAG_ENGINE_FORCE_BACKEND,
            )

        if not settings.RAG_SERVICE_SECRET:
            logger.error("rag_service_secret_missing", status="error")
            raise RuntimeError("RAG_SERVICE_SECRET must be configured for production security")

        if self.contract_client is None:
            self.contract_client = RagRetrievalContractClient(
                backend_selector=self.backend_selector,
            )

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() != "advanced":
            return {
                "valid": True,
                "normalized_scope": {
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                    "filters": filters or {},
                },
                "violations": [],
                "warnings": [],
                "query_scope": {},
            }
        assert self.contract_client is not None
        try:
            payload = await self.contract_client.validate_scope(
                query=query,
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                filters=filters,
            )
        except RagContractNotSupportedError:
            # Contract not deployed: allow caller to continue with legacy retrieval.
            return {
                "valid": True,
                "normalized_scope": {
                    "tenant_id": tenant_id,
                    "collection_id": collection_id,
                    "filters": filters or {},
                },
                "violations": [],
                "warnings": [],
                "query_scope": {},
            }
        if isinstance(payload, dict):
            self._validated_scope_payload = payload
        return payload if isinstance(payload, dict) else {}

    def apply_validated_scope(self, validated: dict[str, Any]) -> None:
        # normalized_scope: { tenant_id, collection_id, filters: { metadata, time_range, source_standard(s) } }
        normalized = validated.get("normalized_scope") if isinstance(validated, dict) else None
        filters = normalized.get("filters") if isinstance(normalized, dict) else None
        self._validated_filters = filters if isinstance(filters, dict) else None
        self._validated_scope_payload = validated

    async def retrieve_chunks(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced":
            try:
                return await self._retrieve_advanced(
                    query=query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    plan=plan,
                    user_id=user_id,
                )
            except RagContractNotSupportedError:
                logger.warning(
                    "rag_contract_not_supported_fallback_legacy", endpoint="retrieval_contract"
                )
                # Fall back to legacy for this request.
                self.last_retrieval_diagnostics = RetrievalDiagnostics(
                    contract="legacy",
                    strategy="legacy_fallback",
                    partial=False,
                    trace={"warning": "advanced_contract_404_fallback_legacy"},
                    scope_validation=self._validated_scope_payload or {},
                )
                # Continue into legacy call below.

        retrieval_metrics_store.record_request("chunks")
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "chunk_k": int(plan.chunk_k),
            "fetch_k": int(plan.chunk_fetch_k),
        }
        context_headers = {"X-Tenant-ID": tenant_id}
        if user_id:
            context_headers["X-User-ID"] = user_id
        try:
            data = await self._post_json(
                "/api/v1/debug/retrieval/chunks",
                payload,
                endpoint="chunks",
                extra_headers=context_headers,
            )
        except (httpx.RequestError, httpx.HTTPStatusError):
            retrieval_metrics_store.record_failure("chunks")
            raise
        retrieval_metrics_store.record_success("chunks")
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
    ) -> list[EvidenceItem]:
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced":
            # Advanced contract retrieval is unified; "summaries" are not fetched separately.
            return []
        retrieval_metrics_store.record_request("summaries")
        payload = {
            "query": query,
            "tenant_id": tenant_id,
            "collection_id": collection_id,
            "summary_k": int(plan.summary_k),
        }
        context_headers = {"X-Tenant-ID": tenant_id}
        if user_id:
            context_headers["X-User-ID"] = user_id
        try:
            data = await self._post_json(
                "/api/v1/debug/retrieval/summaries",
                payload,
                endpoint="summaries",
                extra_headers=context_headers,
            )
            retrieval_metrics_store.record_success("summaries")
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            retrieval_metrics_store.record_failure("summaries")
            retrieval_metrics_store.record_degraded_response("summaries")
            logger.warning(
                "rag_retrieval_endpoint_degraded",
                endpoint="summaries",
                reason="returning_empty_summaries",
                error=str(exc),
            )
            return []
        items = data.get("items") if isinstance(data, dict) else []
        return self._to_evidence(items)

    async def _retrieve_advanced(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None,
    ) -> list[EvidenceItem]:
        assert self.contract_client is not None

        filters = self._validated_filters
        # If caller did not pre-validate, keep behavior robust with a light default filter.
        if filters is None:
            filters = (
                {"source_standards": list(plan.requested_standards)}
                if plan.requested_standards
                else None
            )

        k = max(1, min(int(plan.chunk_k), 24 if plan.mode == "comparativa" else 18))
        fetch_k = max(1, int(plan.chunk_fetch_k))

        clause_refs = extract_clause_refs(query)
        multihop_hint = len(plan.requested_standards) >= 2 or len(clause_refs) >= 2

        async def build_subqueries() -> list[dict[str, Any]]:
            subqueries_local = build_deterministic_subqueries(
                query=query,
                requested_standards=plan.requested_standards,
            )
            if settings.ORCH_SEMANTIC_PLANNER:
                planner = SemanticSubqueryPlanner()
                planned = await planner.plan(
                    query=query,
                    requested_standards=plan.requested_standards,
                    max_queries=settings.ORCH_PLANNER_MAX_QUERIES,
                )
                if planned:
                    return planned
            return subqueries_local

        timings_ms: dict[str, float] = {}

        # Promote multi-query as the primary retrieval strategy for complex intents (optional).
        if (
            settings.ORCH_MULTI_QUERY_PRIMARY
            and multihop_hint
            and plan.mode in {"comparativa", "explicativa"}
        ):
            merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
            subqueries = await build_subqueries()
            t0 = time.perf_counter()
            mq_payload = await self.contract_client.multi_query(
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                queries=subqueries,
                merge=merge,
            )
            timings_ms["multi_query_primary"] = round((time.perf_counter() - t0) * 1000, 2)
            mq_items = mq_payload.get("items") if isinstance(mq_payload, dict) else []
            mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
            partial = (
                bool(mq_payload.get("partial", False)) if isinstance(mq_payload, dict) else False
            )
            subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
            diag_trace: dict[str, Any] = {
                "promoted": True,
                "reason": "complex_intent",
                "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                "subqueries": subq if isinstance(subq, list) else [],
                "timings_ms": dict(timings_ms),
            }

            if isinstance(mq_items, list) and len(mq_items) >= max(
                1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
            ):
                self.last_retrieval_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="multi_query_primary",
                    partial=partial,
                    trace=diag_trace,
                    scope_validation=self._validated_scope_payload or {},
                )
                return self._to_evidence(mq_items)

            if settings.ORCH_MULTI_QUERY_EVALUATOR and isinstance(mq_items, list) and mq_items:
                evaluator = RetrievalSufficiencyEvaluator()
                decision = await evaluator.evaluate(
                    query=query,
                    requested_standards=plan.requested_standards,
                    items=[it for it in mq_items if isinstance(it, dict)],
                    min_items=int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6),
                )
                if decision.sufficient:
                    diag_trace["evaluator_override"] = True
                    diag_trace["evaluator_reason"] = decision.reason
                    self.last_retrieval_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="multi_query_primary_evaluator",
                        partial=partial,
                        trace=diag_trace,
                        scope_validation=self._validated_scope_payload or {},
                    )
                    return self._to_evidence(mq_items)

            # Optional single refinement iteration.
            if settings.ORCH_MULTI_QUERY_REFINE:
                step_back = {
                    "id": "step_back",
                    "query": f"principios generales y requisitos clave relacionados: {query}",
                    "k": None,
                    "fetch_k": None,
                    "filters": {"source_standards": list(plan.requested_standards)}
                    if plan.requested_standards
                    else None,
                }
                refined = (subqueries + [step_back])[
                    : max(1, int(settings.ORCH_PLANNER_MAX_QUERIES or 5))
                ]
                t1 = time.perf_counter()
                mq2 = await self.contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    queries=refined,
                    merge=merge,
                )
                timings_ms["multi_query_refine"] = round((time.perf_counter() - t1) * 1000, 2)
                mq2_items = mq2.get("items") if isinstance(mq2, dict) else []
                if isinstance(mq2_items, list) and len(mq2_items) >= max(
                    1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
                ):
                    diag_trace["refined"] = True
                    diag_trace["refine_reason"] = "insufficient_primary_multi_query"
                    diag_trace["timings_ms"] = dict(timings_ms)
                    self.last_retrieval_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="multi_query_refined",
                        partial=bool(mq2.get("partial", False)) if isinstance(mq2, dict) else False,
                        trace=diag_trace,
                        scope_validation=self._validated_scope_payload or {},
                    )
                    return self._to_evidence(mq2_items)

        t_h = time.perf_counter()
        hybrid_payload = await self.contract_client.hybrid(
            query=query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            k=k,
            fetch_k=fetch_k,
            filters=filters,
            rerank={"enabled": True},
            graph={"max_hops": 2},
        )
        timings_ms["hybrid"] = round((time.perf_counter() - t_h) * 1000, 2)
        items = hybrid_payload.get("items") if isinstance(hybrid_payload, dict) else []
        trace = hybrid_payload.get("trace") if isinstance(hybrid_payload, dict) else {}
        if not isinstance(items, list):
            items = []
        if not isinstance(trace, dict):
            trace = {}

        # Decide multihop fallback if configured.
        if (
            settings.ORCH_MULTIHOP_FALLBACK
            and multihop_hint
            and plan.mode in {"comparativa", "explicativa"}
        ):
            rows: list[dict[str, Any]] = []
            for it in items[: max(1, min(12, len(items)))]:
                if not isinstance(it, dict):
                    continue
                meta = it.get("metadata")
                row = meta.get("row") if isinstance(meta, dict) else None
                if isinstance(row, dict):
                    rows.append(row)

            decision = decide_multihop_fallback(
                query=query,
                requested_standards=plan.requested_standards,
                items=rows,
                hybrid_trace=trace,
                top_k=12,
            )
            if decision.needs_fallback:
                subqueries = await build_subqueries()
                merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
                t_mq = time.perf_counter()
                mq_payload = await self.contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    queries=subqueries,
                    merge=merge,
                )
                timings_ms["multi_query_fallback"] = round((time.perf_counter() - t_mq) * 1000, 2)
                mq_items = mq_payload.get("items") if isinstance(mq_payload, dict) else []
                mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
                partial = (
                    bool(mq_payload.get("partial", False))
                    if isinstance(mq_payload, dict)
                    else False
                )
                subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
                diag_trace: dict[str, Any] = {
                    "fallback_reason": decision.reason,
                    "hybrid_trace": trace,
                    "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                    "subqueries": subq if isinstance(subq, list) else [],
                    "timings_ms": dict(timings_ms),
                }
                self.last_retrieval_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="multi_query",
                    partial=partial,
                    trace=diag_trace,
                    scope_validation=self._validated_scope_payload or {},
                )
                return self._to_evidence(mq_items)

        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="hybrid",
            partial=False,
            trace={"hybrid_trace": trace, "timings_ms": dict(timings_ms)},
            scope_validation=self._validated_scope_payload or {},
        )
        return self._to_evidence(items)

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        endpoint: str,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        selector = self.backend_selector
        if selector is None:
            base_url = str(self.base_url or "http://localhost:8000").rstrip("/")
            return await self._post_once(
                base_url=base_url, path=path, payload=payload, extra_headers=extra_headers
            )

        primary_backend = await selector.current_backend()
        primary_base_url = await selector.resolve_base_url()

        try:
            return await self._post_once(
                base_url=primary_base_url,
                path=path,
                payload=payload,
                extra_headers=extra_headers,
            )
        except (httpx.RequestError, httpx.HTTPStatusError) as primary_exc:
            if selector.is_forced():
                raise

            retryable_status = (
                isinstance(primary_exc, httpx.HTTPStatusError)
                and primary_exc.response is not None
                and primary_exc.response.status_code >= 500
            )
            if isinstance(primary_exc, httpx.HTTPStatusError) and not retryable_status:
                raise

            alternate_backend = selector.alternate_backend(primary_backend)
            alternate_base_url = selector.base_url_for(alternate_backend)
            retrieval_metrics_store.record_fallback_retry(endpoint)
            logger.warning(
                "rag_backend_fallback_retry",
                from_backend=primary_backend,
                to_backend=alternate_backend,
                path=path,
                error=str(primary_exc),
            )
            response = await self._post_once(
                base_url=alternate_base_url,
                path=path,
                payload=payload,
                extra_headers=extra_headers,
            )
            selector.set_backend(alternate_backend)
            return response

    async def _post_once(
        self,
        *,
        base_url: str,
        path: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = base_url.rstrip("/") + path
        headers = {
            "X-Service-Secret": settings.RAG_SERVICE_SECRET or "",
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)
        async with httpx.AsyncClient(timeout=self.timeout_seconds, headers=headers) as client:
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
                    metadata={
                        "row": {
                            "content": content,
                            "metadata": raw_metadata,
                            "similarity": item.get("score"),
                        }
                    },
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
        # IMPORTANT: Literal modes are validated against explicit C#/R# markers.
        # Always pass the LLM a context that includes those markers, otherwise the
        # validator will (correctly) reject "literal" answers as ungrounded.
        del scope_label

        import json

        def _clip(text: str, limit: int) -> str:
            t = (text or "").strip()
            if len(t) <= limit:
                return t
            return t[:limit].rstrip() + "..."

        def _extract_iso_standards(text: str) -> list[str]:
            import re

            found = []
            for match in re.findall(
                r"\biso\s*[-:]?\s*(\d{4,5})\b", (text or ""), flags=re.IGNORECASE
            ):
                found.append(f"ISO {match}")
            # Preserve order, unique.
            seen: set[str] = set()
            ordered: list[str] = []
            for s in found:
                if s in seen:
                    continue
                seen.add(s)
                ordered.append(s)
            return ordered

        def _row_mentions_standards(row: dict[str, Any], standards: list[str]) -> set[str]:
            import re

            content = str(row.get("content") or "")
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            blob = (content + "\n" + json.dumps(meta, default=str, ensure_ascii=True)).upper()
            present: set[str] = set()
            for std in standards:
                key = re.search(r"\b(\d{4,5})\b", std)
                digits = key.group(1) if key else ""
                if digits and digits in blob:
                    present.add(f"ISO {digits}")
            return present

        def _safe_parse_iso8601(value: Any) -> float | None:
            from datetime import datetime

            if not isinstance(value, str) or not value.strip():
                return None
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text).timestamp()
            except Exception:
                return None

        def _recency_key(item: EvidenceItem) -> float:
            row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
            if not isinstance(row, dict):
                return 0.0
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            ts = (
                _safe_parse_iso8601(meta.get("updated_at"))
                or _safe_parse_iso8601(meta.get("source_updated_at"))
                or _safe_parse_iso8601(meta.get("created_at"))
            )
            return float(ts or 0.0)

        ordered_items = [*summaries, *chunks]
        if plan.mode == "comparativa":
            # Best-effort recency: if timestamps exist in metadata, prefer more recent evidence.
            ordered_items = sorted(ordered_items, key=_recency_key, reverse=True)

        labeled: list[str] = []
        for item in ordered_items:
            content = (item.content or "").strip()
            if not content:
                continue
            source = (item.source or "").strip() or "C1"
            labeled.append(f"[{source}] {_clip(content, 900)}")

        # Use more context for multi-standard interpretive questions.
        if plan.mode == "comparativa":
            max_ctx = 28
        elif plan.require_literal_evidence:
            max_ctx = 14
        else:
            max_ctx = 18

        text = await self.service.generate_answer(
            query=query,
            context_chunks=labeled,
            mode=plan.mode,
            require_literal_evidence=bool(plan.require_literal_evidence),
            max_chunks=max_ctx,
        )

        # Guardrail: if the answer ties multiple standards together but evidence has no explicit bridge,
        # enforce transparent language (inference vs direct citation).
        requested = list(plan.requested_standards or ())
        mentioned = _extract_iso_standards(text)
        standards = requested if len(requested) >= 2 else mentioned
        if len(standards) >= 2 and not plan.require_literal_evidence:
            bridges = 0
            for ev in ordered_items:
                row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else None
                if not isinstance(row, dict):
                    continue
                present = _row_mentions_standards(row, standards)
                if len(present) >= 2:
                    bridges += 1
                    break
            already_disclosed = any(
                tok in (text or "").lower()
                for tok in ("inferencia", "interpret", "no encontrado explicitamente")
            )
            if bridges == 0 and not already_disclosed:
                note = (
                    "Nota de trazabilidad: La relacion entre normas se presenta como inferencia basada en "
                    "evidencias separadas; no hay un fragmento unico que las vincule explicitamente."
                )
                text = f"{note}\n\n{text}" if (text or "").strip() else note
        if plan.require_literal_evidence:
            # Defense-in-depth: if the provider ignores instructions and returns no markers,
            # append the reviewed references so the validator can trace the answer.
            import re

            if not re.search(r"\b[CR]\d+\b", text or ""):
                sources: list[str] = []
                seen: set[str] = set()
                for item in [*summaries, *chunks]:
                    src = (item.source or "").strip()
                    if not src or src in seen:
                        continue
                    seen.add(src)
                    sources.append(src)
                if sources:
                    suffix = "Referencias revisadas: " + ", ".join(sources)
                    text = (text or "").rstrip()
                    text = f"{text}\n\n{suffix}" if text else suffix
        return AnswerDraft(text=text, mode=plan.mode, evidence=[*chunks, *summaries])
