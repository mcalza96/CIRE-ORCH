from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import httpx
import structlog
import time

from app.agent.grounded_answer_service import GroundedAnswerService
from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_LOW_SCORE,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    merge_error_codes,
)
from app.agent.models import AnswerDraft, EvidenceItem, RetrievalDiagnostics, RetrievalPlan
from app.cartridges.models import AgentProfile, QueryModeConfig
from app.agent.retrieval_planner import (
    apply_search_hints,
    build_deterministic_subqueries,
    decide_multihop_fallback,
    extract_clause_refs,
    mode_requires_literal_evidence,
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
    _profile_context: AgentProfile | None = None
    _profile_resolution_context: dict[str, Any] | None = None

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

    def set_profile_context(
        self,
        *,
        profile: AgentProfile | None,
        profile_resolution: dict[str, Any] | None = None,
    ) -> None:
        self._profile_context = profile
        self._profile_resolution_context = (
            profile_resolution if isinstance(profile_resolution, dict) else None
        )

    def _profile_min_score(self) -> float | None:
        if self._profile_context is None:
            return None
        try:
            return float(self._profile_context.retrieval.min_score)
        except Exception:
            return None

    def _mode_config(self, mode: str) -> QueryModeConfig | None:
        if self._profile_context is None:
            return None
        cfg = self._profile_context.query_modes.modes.get(str(mode or "").strip())
        return cfg if isinstance(cfg, QueryModeConfig) else None

    def _filter_items_by_min_score(
        self,
        items: list[dict[str, Any]],
        *,
        trace_target: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        threshold = self._profile_min_score()
        if threshold is None:
            return items

        kept: list[dict[str, Any]] = []
        dropped = 0
        scored_dropped: list[tuple[float, dict[str, Any]]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            score_raw = item.get("score")
            if score_raw is None:
                score_raw = item.get("similarity")
            if score_raw is None:
                kept.append(item)
                continue
            score = float(score_raw or 0.0)
            if score >= threshold:
                kept.append(item)
            else:
                dropped += 1
                scored_dropped.append((score, item))

        backstop_applied = False
        backstop_enabled = bool(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_ENABLED", False))
        backstop_top_n = max(1, int(getattr(settings, "ORCH_MIN_SCORE_BACKSTOP_TOP_N", 6) or 6))
        if not kept and dropped > 0 and backstop_enabled:
            top_n = backstop_top_n
            scored_dropped.sort(key=lambda pair: pair[0], reverse=True)
            kept = [item for _, item in scored_dropped[:top_n]]
            backstop_applied = bool(kept)

        if isinstance(trace_target, dict):
            trace_target["min_score_filter"] = {
                "threshold": threshold,
                "kept": len(kept),
                "dropped": dropped,
                "backstop_applied": backstop_applied,
                "backstop_top_n": (backstop_top_n if backstop_applied else 0),
            }
            if dropped > 0 and not kept:
                trace_target["error_codes"] = merge_error_codes(
                    trace_target.get("error_codes"),
                    [RETRIEVAL_CODE_LOW_SCORE],
                )
        return kept

    async def validate_scope(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
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
                request_id=request_id,
                correlation_id=correlation_id,
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
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced":
            try:
                return await self._retrieve_advanced(
                    query=query,
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    plan=plan,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
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
                    trace={
                        "warning": "advanced_contract_404_fallback_legacy",
                        "agent_profile_resolution": dict(self._profile_resolution_context or {}),
                    },
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
        if request_id:
            context_headers["X-Request-ID"] = request_id
        if correlation_id:
            context_headers["X-Correlation-ID"] = correlation_id
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
        if isinstance(items, list):
            items = self._filter_items_by_min_score([it for it in items if isinstance(it, dict)])
        return self._to_evidence(items)

    async def retrieve_summaries(
        self,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        if str(settings.ORCH_RETRIEVAL_CONTRACT or "").lower() == "advanced":
            # Advanced contract does not expose summaries. Optionally hit the debug endpoint
            # to keep RAPTOR wired-in (agnostic feature; bounded by summary_k).
            if not settings.ORCH_RAPTOR_SUMMARIES_ENABLED:
                return []
            if int(plan.summary_k or 0) <= 0:
                return []
            # Continue into debug endpoint call below.
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
        if request_id:
            context_headers["X-Request-ID"] = request_id
        if correlation_id:
            context_headers["X-Correlation-ID"] = correlation_id
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
        if isinstance(items, list):
            items = self._filter_items_by_min_score([it for it in items if isinstance(it, dict)])
        return self._to_evidence(items)

    async def _retrieve_advanced(
        self,
        *,
        query: str,
        tenant_id: str,
        collection_id: str | None,
        plan: RetrievalPlan,
        user_id: str | None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> list[EvidenceItem]:
        assert self.contract_client is not None
        contract_client = self.contract_client

        filters = self._validated_filters
        # If caller did not pre-validate, keep behavior robust with a light default filter.
        if filters is None:
            filters = (
                {"source_standards": list(plan.requested_standards)}
                if plan.requested_standards
                else None
            )

        expanded_query, hint_trace = apply_search_hints(query, profile=self._profile_context)
        clause_refs = extract_clause_refs(query, profile=self._profile_context)
        multihop_hint = len(plan.requested_standards) >= 2 or len(clause_refs) >= 2
        mode_cfg = self._mode_config(plan.mode)
        coverage_requirements = (
            dict(mode_cfg.coverage_requirements) if isinstance(mode_cfg, QueryModeConfig) else {}
        )
        decomposition_policy = (
            dict(mode_cfg.decomposition_policy) if isinstance(mode_cfg, QueryModeConfig) else {}
        )
        require_all_scopes = bool(
            coverage_requirements.get(
                "require_all_requested_scopes",
                len(plan.requested_standards) >= 2,
            )
        )
        try:
            min_clause_refs_required = int(
                coverage_requirements.get(
                    "min_clause_refs",
                    1 if bool(plan.require_literal_evidence) else 0,
                )
            )
        except (TypeError, ValueError):
            min_clause_refs_required = 0
        min_clause_refs_required = max(0, min(6, min_clause_refs_required))
        try:
            mode_max_subqueries = int(decomposition_policy.get("max_subqueries", 6))
        except (TypeError, ValueError):
            mode_max_subqueries = 6
        mode_max_subqueries = max(2, min(12, mode_max_subqueries))
        literal_mode = mode_requires_literal_evidence(
            mode=plan.mode,
            profile=self._profile_context,
            explicit_flag=plan.require_literal_evidence,
        )
        cross_scope_mode = len(plan.requested_standards) >= 2 and not literal_mode
        k_cap = 24 if cross_scope_mode else 18
        k = max(1, min(int(plan.chunk_k), k_cap))
        fetch_k = max(1, int(plan.chunk_fetch_k))
        semantic_tail_enabled = bool(
            getattr(settings, "ORCH_DETERMINISTIC_SUBQUERY_SEMANTIC_TAIL", False)
        )

        def _layer_stats(raw_items: list[dict[str, Any]]) -> dict[str, Any]:
            counts: dict[str, int] = {}
            raptor = 0
            for it in raw_items:
                meta_raw = it.get("metadata")
                meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
                row_raw = meta.get("row")
                row = row_raw if isinstance(row_raw, dict) else None
                if not isinstance(row, dict):
                    continue
                layer = str(row.get("source_layer") or "").strip() or "unknown"
                counts[layer] = counts.get(layer, 0) + 1
                row_meta_raw = row.get("metadata")
                row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
                if bool(row_meta.get("is_raptor_summary", False)):
                    raptor += 1
            return {"layer_counts": counts, "raptor_summary_count": raptor}

        def _features_from_hybrid_trace(trace: Any) -> dict[str, Any]:
            if not isinstance(trace, dict):
                return {}
            return {
                "engine_mode": str(trace.get("engine_mode") or ""),
                "planner_used": bool(trace.get("planner_used", False)),
                "planner_multihop": bool(trace.get("planner_multihop", False)),
                "fallback_used": bool(trace.get("fallback_used", False)),
            }

        def _extract_row_scope(row: dict[str, Any]) -> str:
            meta = row.get("metadata")
            if isinstance(meta, dict):
                for key in ("source_standard", "scope", "standard"):
                    value = meta.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip().upper()
            value2 = row.get("source_standard")
            if isinstance(value2, str) and value2.strip():
                return value2.strip().upper()
            return ""

        def _missing_scopes(
            items: list[dict[str, Any]],
            requested: tuple[str, ...],
            *,
            enforce: bool,
        ) -> list[str]:
            if not enforce:
                return []
            if len(requested) < 2:
                return []
            top_n = max(1, int(settings.ORCH_COVERAGE_GATE_TOP_N or 12))
            present: set[str] = set()
            for it in items[:top_n]:
                meta_raw = it.get("metadata")
                meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
                row_raw = meta.get("row")
                row = row_raw if isinstance(row_raw, dict) else None
                if not isinstance(row, dict):
                    continue
                scope = _extract_row_scope(row)
                if scope:
                    present.add(scope)
            req_upper = [s.upper() for s in requested if s]
            return [s for s in req_upper if s not in present]

        def _row_matches_clause_ref(row: dict[str, Any], clause_ref: str) -> bool:
            clause = str(clause_ref or "").strip()
            if not clause:
                return False
            clause_re = re.compile(rf"\b{re.escape(clause)}(?:\.\d+)*\b")
            content = str(row.get("content") or "")
            if clause_re.search(content):
                return True
            row_meta_raw = row.get("metadata")
            row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
            for key in ("clause_id", "clause_ref", "clause"):
                value = row_meta.get(key)
                if isinstance(value, str) and value.strip():
                    val = value.strip()
                    if val == clause or val.startswith(f"{clause}."):
                        return True
            refs = row_meta.get("clause_refs")
            if isinstance(refs, list):
                for value in refs:
                    if isinstance(value, str) and value.strip():
                        val = value.strip()
                        if val == clause or val.startswith(f"{clause}."):
                            return True
            return False

        def _missing_clause_refs(
            items: list[dict[str, Any]],
            refs: list[str],
            *,
            min_required: int,
        ) -> list[str]:
            if min_required <= 0:
                return []
            required = [str(ref).strip() for ref in refs if str(ref).strip()]
            if not required:
                return []
            top_n = max(1, int(settings.ORCH_COVERAGE_GATE_TOP_N or 12))
            present: set[str] = set()
            for it in items[:top_n]:
                meta_raw = it.get("metadata")
                meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
                row_raw = meta.get("row")
                row = row_raw if isinstance(row_raw, dict) else None
                if not isinstance(row, dict):
                    continue
                for ref in required:
                    if ref in present:
                        continue
                    if _row_matches_clause_ref(row, ref):
                        present.add(ref)
            if len(present) >= min_required:
                return []
            missing = [ref for ref in required if ref not in present]
            shortfall = max(0, min_required - len(present))
            return missing[:shortfall] if shortfall else missing

        def _looks_structural_toc(item: dict[str, Any]) -> bool:
            content = str(item.get("content") or "").strip().lower()
            if not content:
                return True
            meta_raw = item.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            row_raw = meta.get("row")
            row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
            row_meta_raw = row.get("metadata")
            row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
            title = str(row_meta.get("title") or "").lower()
            hay = f"{title}\n{content}"
            if any(token in hay for token in ("indice", "índice", "tabla de contenido")):
                return True
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if len(lines) >= 3:
                short_lines = [ln for ln in lines[:8] if len(ln) <= 80]
                dotted_or_page = sum(
                    1 for ln in short_lines if "..." in ln or ln.rstrip().split(" ")[-1].isdigit()
                )
                if dotted_or_page >= 3:
                    return True
            return False

        def _looks_editorial_front_matter(item: dict[str, Any]) -> bool:
            content = str(item.get("content") or "").strip().lower()
            meta_raw = item.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            row_raw = meta.get("row")
            row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
            row_meta_raw = row.get("metadata")
            row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
            title = str(row_meta.get("title") or "").strip().lower()
            heading = (
                str(row_meta.get("heading") or row_meta.get("section_title") or "").strip().lower()
            )
            source_type = str(row.get("source_type") or "").strip().lower()

            hay = "\n".join(part for part in [title, heading, content[:600]] if part)
            if any(
                token in hay
                for token in (
                    "prologo",
                    "prólogo",
                    "preface",
                    "foreword",
                    "copyright",
                    "isbn",
                    "ics",
                    "committee",
                    "comite",
                    "comité",
                    "translation",
                    "traduccion",
                    "traducción",
                    "quinta edicion",
                    "fifth edition",
                    "anula y sustituye",
                    "iso/tc",
                    "secretaria central",
                    "published in switzerland",
                )
            ):
                return True

            if source_type in {"front_matter", "preface", "metadata"}:
                return True

            shortish = 20 <= len(content) <= 520
            institutional_markers = sum(
                1
                for token in ("tc", "sc", "committee", "comite", "sttf", "copyright", "edition")
                if token in hay
            )
            return shortish and institutional_markers >= 2

        def _reduce_structural_noise(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            q = str(query or "").lower()
            if any(token in q for token in ("indice", "índice", "tabla de contenido")):
                return items
            if any(token in q for token in ("prologo", "prólogo", "preface", "foreword")):
                return items

            toc_items: list[dict[str, Any]] = []
            editorial_items: list[dict[str, Any]] = []
            body_items: list[dict[str, Any]] = []
            for item in items:
                if _looks_structural_toc(item):
                    toc_items.append(item)
                    continue
                if _looks_editorial_front_matter(item):
                    editorial_items.append(item)
                    continue
                body_items.append(item)

            if not body_items:
                return items
            if not toc_items and not editorial_items:
                return items

            # Keep the best body evidence first, then a tiny structural tail for transparency.
            return body_items + editorial_items[:1] + toc_items[:1]

        async def _coverage_repair(
            *,
            items: list[dict[str, Any]],
            base_trace: dict[str, Any],
            reason: str,
        ) -> list[dict[str, Any]]:
            if not settings.ORCH_COVERAGE_GATE_ENABLED:
                return items
            missing_scopes = _missing_scopes(
                items,
                plan.requested_standards,
                enforce=require_all_scopes,
            )
            missing_clauses = _missing_clause_refs(
                items,
                clause_refs,
                min_required=min_clause_refs_required,
            )
            if not missing_scopes and not missing_clauses:
                base_trace["missing_scopes"] = []
                base_trace["missing_clause_refs"] = []
                return items
            cap = max(1, int(settings.ORCH_COVERAGE_GATE_MAX_MISSING or 2))
            missing_scopes = missing_scopes[:cap]
            missing_clauses = missing_clauses[:cap]

            # Build focused subqueries per missing scope and clause refs.
            focused: list[dict[str, Any]] = []
            for idx, scope in enumerate(missing_scopes, start=1):
                qtext = " ".join(
                    part for part in [scope, *clause_refs[:3], expanded_query] if part
                ).strip()
                focused.append(
                    {
                        "id": f"scope_repair_{idx}",
                        "query": qtext[:900],
                        "k": None,
                        "fetch_k": None,
                        "filters": {"source_standard": scope},
                    }
                )
            for idx, clause in enumerate(missing_clauses, start=1):
                focused.append(
                    {
                        "id": f"clause_repair_{idx}",
                        "query": f"{expanded_query} clausula {clause}"[:900],
                        "k": None,
                        "fetch_k": None,
                        "filters": {
                            **(
                                {"source_standards": list(plan.requested_standards)}
                                if plan.requested_standards
                                else {}
                            ),
                            "metadata": {"clause_id": clause},
                        },
                    }
                )

            merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(18, max(12, k))}
            t_cov = time.perf_counter()
            cov_payload = await contract_client.multi_query(
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
                queries=focused,
                merge=merge,
            )
            timings_ms["coverage_gate"] = round((time.perf_counter() - t_cov) * 1000, 2)

            cov_items = cov_payload.get("items") if isinstance(cov_payload, dict) else []
            if not isinstance(cov_items, list) or not cov_items:
                base_trace["coverage_gate"] = {
                    "trigger_reason": reason,
                    "missing_scopes": missing_scopes,
                    "missing_clause_refs": missing_clauses,
                    "added_queries": [q.get("id") for q in focused],
                    "final_missing_scopes": list(missing_scopes),
                    "final_missing_clause_refs": list(missing_clauses),
                }
                base_trace["missing_scopes"] = list(missing_scopes)
                base_trace["missing_clause_refs"] = list(missing_clauses)
                codes: list[str] = []
                if missing_scopes:
                    codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
                if missing_clauses:
                    codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
                base_trace["error_codes"] = merge_error_codes(base_trace.get("error_codes"), codes)
                return items

            # Merge and dedupe by source id.
            merged: list[dict[str, Any]] = []
            seen: set[str] = set()
            for it in [*items, *cov_items]:
                if not isinstance(it, dict):
                    continue
                sid = str(it.get("source") or "")
                key = sid or str(len(seen))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(it)

            base_trace["coverage_gate"] = {
                "trigger_reason": reason,
                "missing_scopes": missing_scopes,
                "missing_clause_refs": missing_clauses,
                "added_queries": [q.get("id") for q in focused],
            }

            # If still missing after focused queries, optionally try a step-back pass per missing scope.
            remaining = _missing_scopes(
                merged,
                plan.requested_standards,
                enforce=require_all_scopes,
            )
            remaining_clauses = _missing_clause_refs(
                merged,
                clause_refs,
                min_required=min_clause_refs_required,
            )
            if (remaining or remaining_clauses) and settings.ORCH_COVERAGE_GATE_STEP_BACK:
                remaining = remaining[:cap]
                remaining_clauses = remaining_clauses[:cap]
                step_back_queries: list[dict[str, Any]] = []
                for idx, scope in enumerate(remaining, start=1):
                    step_back_queries.append(
                        {
                            "id": f"scope_step_back_{idx}",
                            "query": (
                                "principios generales y requisitos clave relacionados con: "
                                f"{expanded_query}"
                            ),
                            "k": None,
                            "fetch_k": None,
                            "filters": {"source_standard": scope},
                        }
                    )
                for idx, clause in enumerate(remaining_clauses, start=1):
                    step_back_queries.append(
                        {
                            "id": f"clause_step_back_{idx}",
                            "query": (
                                "principios generales y requisitos clave relacionados con: "
                                f"{expanded_query} clausula {clause}"
                            )[:900],
                            "k": None,
                            "fetch_k": None,
                            "filters": {
                                **(
                                    {"source_standards": list(plan.requested_standards)}
                                    if plan.requested_standards
                                    else {}
                                ),
                                "metadata": {"clause_id": clause},
                            },
                        }
                    )
                t_sb = time.perf_counter()
                sb_payload = await contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=step_back_queries,
                    merge=merge,
                )
                timings_ms["coverage_gate_step_back"] = round(
                    (time.perf_counter() - t_sb) * 1000, 2
                )
                sb_items = sb_payload.get("items") if isinstance(sb_payload, dict) else []
                if isinstance(sb_items, list) and sb_items:
                    for it in sb_items:
                        if not isinstance(it, dict):
                            continue
                        sid = str(it.get("source") or "")
                        key = sid or str(len(seen))
                        if key in seen:
                            continue
                        seen.add(key)
                        merged.append(it)
                    if isinstance(base_trace.get("coverage_gate"), dict):
                        base_trace["coverage_gate"]["step_back_missing_scopes"] = remaining
                        base_trace["coverage_gate"]["step_back_missing_clause_refs"] = (
                            remaining_clauses
                        )
                        base_trace["coverage_gate"]["step_back_queries"] = [
                            q.get("id") for q in step_back_queries
                        ]

            final_missing = _missing_scopes(
                merged,
                plan.requested_standards,
                enforce=require_all_scopes,
            )
            final_missing_clauses = _missing_clause_refs(
                merged,
                clause_refs,
                min_required=min_clause_refs_required,
            )
            if isinstance(base_trace.get("coverage_gate"), dict):
                base_trace["coverage_gate"]["final_missing_scopes"] = final_missing
                base_trace["coverage_gate"]["final_missing_clause_refs"] = final_missing_clauses
            base_trace["missing_scopes"] = list(final_missing)
            base_trace["missing_clause_refs"] = list(final_missing_clauses)
            codes: list[str] = []
            if final_missing:
                codes.append(RETRIEVAL_CODE_SCOPE_MISMATCH)
            if final_missing_clauses:
                codes.append(RETRIEVAL_CODE_CLAUSE_MISSING)
            base_trace["error_codes"] = merge_error_codes(base_trace.get("error_codes"), codes)
            return merged

        async def build_subqueries(*, purpose: str = "primary") -> list[dict[str, Any]]:
            subqueries_local = build_deterministic_subqueries(
                query=query,
                requested_standards=plan.requested_standards,
                max_queries=mode_max_subqueries,
                mode=plan.mode,
                require_literal_evidence=plan.require_literal_evidence,
                include_semantic_tail=semantic_tail_enabled,
                profile=self._profile_context,
            )
            fallback_max = max(
                2,
                int(getattr(settings, "ORCH_MULTI_QUERY_FALLBACK_MAX_QUERIES", 3) or 3),
            )
            fallback_max = min(mode_max_subqueries, fallback_max)
            if purpose == "fallback" and len(subqueries_local) > fallback_max:
                subqueries_local = subqueries_local[:fallback_max]
            if settings.ORCH_SEMANTIC_PLANNER:
                planner = SemanticSubqueryPlanner()
                planned = await planner.plan(
                    query=query,
                    requested_standards=plan.requested_standards,
                    max_queries=settings.ORCH_PLANNER_MAX_QUERIES,
                    search_hints=(
                        [item.model_dump() for item in self._profile_context.retrieval.search_hints]
                        if self._profile_context is not None
                        else None
                    ),
                )
                if planned:
                    if purpose == "fallback" and len(planned) > fallback_max:
                        return planned[:fallback_max]
                    return planned
            return subqueries_local

        timings_ms: dict[str, float] = {}

        # Promote multi-query as the primary retrieval strategy for complex intents (optional).
        if settings.ORCH_MULTI_QUERY_PRIMARY and multihop_hint and int(plan.chunk_k or 0) > 0:
            merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
            subqueries = await build_subqueries(purpose="primary")
            t0 = time.perf_counter()
            mq_payload = await contract_client.multi_query(
                tenant_id=tenant_id,
                collection_id=collection_id,
                user_id=user_id,
                request_id=request_id,
                correlation_id=correlation_id,
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
                "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                "subqueries": subq if isinstance(subq, list) else [],
                "timings_ms": dict(timings_ms),
            }

            if isinstance(mq_items, list) and len(mq_items) >= max(
                1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
            ):
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items = _reduce_structural_noise(mq_items)
                diag_trace.update(_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                if hint_trace.get("applied"):
                    diag_trace["search_hint_expansions"] = hint_trace
                if isinstance(self._profile_resolution_context, dict):
                    diag_trace["agent_profile_resolution"] = dict(self._profile_resolution_context)
                # Coverage gate (agnostic) before returning.
                mq_items = await _coverage_repair(
                    items=[it for it in mq_items if isinstance(it, dict)],
                    base_trace=diag_trace,
                    reason="multi_query_primary",
                )
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items = _reduce_structural_noise(mq_items)

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
                    diag_trace.update(_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                    mq_items = await _coverage_repair(
                        items=[it for it in mq_items if isinstance(it, dict)],
                        base_trace=diag_trace,
                        reason="multi_query_primary_evaluator",
                    )
                    mq_items = self._filter_items_by_min_score(
                        [it for it in mq_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    mq_items = _reduce_structural_noise(mq_items)
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
                    "query": f"principios generales y requisitos clave relacionados: {expanded_query}",
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
                mq2 = await contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=refined,
                    merge=merge,
                )
                timings_ms["multi_query_refine"] = round((time.perf_counter() - t1) * 1000, 2)
                mq2_items = mq2.get("items") if isinstance(mq2, dict) else []
                if isinstance(mq2_items, list) and len(mq2_items) >= max(
                    1, int(settings.ORCH_MULTI_QUERY_MIN_ITEMS or 6)
                ):
                    mq2_items = self._filter_items_by_min_score(
                        [it for it in mq2_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    mq2_items = _reduce_structural_noise(mq2_items)
                    diag_trace["refined"] = True
                    diag_trace["refine_reason"] = "insufficient_primary_multi_query"
                    diag_trace["timings_ms"] = dict(timings_ms)
                    diag_trace.update(
                        _layer_stats([it for it in mq2_items if isinstance(it, dict)])
                    )
                    mq2_items = await _coverage_repair(
                        items=[it for it in mq2_items if isinstance(it, dict)],
                        base_trace=diag_trace,
                        reason="multi_query_refined",
                    )
                    mq2_items = self._filter_items_by_min_score(
                        [it for it in mq2_items if isinstance(it, dict)],
                        trace_target=diag_trace,
                    )
                    self.last_retrieval_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="multi_query_refined",
                        partial=bool(mq2.get("partial", False)) if isinstance(mq2, dict) else False,
                        trace=diag_trace,
                        scope_validation=self._validated_scope_payload or {},
                    )
                    return self._to_evidence(mq2_items)

        t_h = time.perf_counter()
        hybrid_payload = await contract_client.hybrid(
            query=expanded_query,
            tenant_id=tenant_id,
            collection_id=collection_id,
            user_id=user_id,
            request_id=request_id,
            correlation_id=correlation_id,
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
        items = self._filter_items_by_min_score(
            [it for it in items if isinstance(it, dict)],
            trace_target=trace,
        )
        if hint_trace.get("applied"):
            trace["search_hint_expansions"] = hint_trace
        if isinstance(self._profile_resolution_context, dict):
            trace["agent_profile_resolution"] = dict(self._profile_resolution_context)

        # Decide multihop fallback if configured.
        if settings.ORCH_MULTIHOP_FALLBACK and multihop_hint and int(plan.chunk_k or 0) > 0:
            if (
                bool(getattr(settings, "EARLY_EXIT_COVERAGE_ENABLED", True))
                and len(plan.requested_standards) >= 2
            ):
                missing_before_fallback = _missing_scopes(
                    items,
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                if not missing_before_fallback:
                    if isinstance(trace, dict):
                        trace["multi_query_fallback_skipped"] = "coverage_already_satisfied"
                        trace["multi_query_fallback_missing_scopes"] = []
                    self.last_retrieval_diagnostics = RetrievalDiagnostics(
                        contract="advanced",
                        strategy="hybrid",
                        partial=False,
                        trace={
                            "hybrid_trace": trace,
                            "timings_ms": dict(timings_ms),
                            "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                            "multi_query_fallback_skipped": "coverage_already_satisfied",
                        },
                        scope_validation=self._validated_scope_payload or {},
                    )
                    items2 = await _coverage_repair(
                        items=[it for it in items if isinstance(it, dict)],
                        base_trace=self.last_retrieval_diagnostics.trace
                        if isinstance(self.last_retrieval_diagnostics.trace, dict)
                        else {},
                        reason="hybrid",
                    )
                    items2 = self._filter_items_by_min_score(
                        [it for it in items2 if isinstance(it, dict)],
                        trace_target=self.last_retrieval_diagnostics.trace
                        if isinstance(self.last_retrieval_diagnostics.trace, dict)
                        else {},
                    )
                    items2 = _reduce_structural_noise(items2)
                    return self._to_evidence(items2)

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
                missing_before_fallback = _missing_scopes(
                    items,
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                subqueries = await build_subqueries(purpose="fallback")
                merge = {"strategy": "rrf", "rrf_k": 60, "top_k": min(16, max(12, k))}
                t_mq = time.perf_counter()
                mq_payload = await contract_client.multi_query(
                    tenant_id=tenant_id,
                    collection_id=collection_id,
                    user_id=user_id,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    queries=subqueries,
                    merge=merge,
                )
                timings_ms["multi_query_fallback"] = round((time.perf_counter() - t_mq) * 1000, 2)
                mq_items_raw = mq_payload.get("items") if isinstance(mq_payload, dict) else []
                mq_items = mq_items_raw if isinstance(mq_items_raw, list) else []
                mq_trace = mq_payload.get("trace") if isinstance(mq_payload, dict) else {}
                partial = (
                    bool(mq_payload.get("partial", False))
                    if isinstance(mq_payload, dict)
                    else False
                )
                subq = mq_payload.get("subqueries") if isinstance(mq_payload, dict) else None
                diag_trace: dict[str, Any] = {
                    "fallback_reason": decision.reason,
                    "error_codes": [decision.code or RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP],
                    "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                    "mode_policy": {
                        "require_all_requested_scopes": require_all_scopes,
                        "min_clause_refs": min_clause_refs_required,
                        "max_subqueries": mode_max_subqueries,
                    },
                    "hybrid_trace": trace,
                    "multi_query_trace": mq_trace if isinstance(mq_trace, dict) else {},
                    "subqueries": subq if isinstance(subq, list) else [],
                    "timings_ms": dict(timings_ms),
                }
                mq_items = self._filter_items_by_min_score(
                    [it for it in mq_items if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items = _reduce_structural_noise(mq_items)
                missing_after_fallback = _missing_scopes(
                    [it for it in mq_items if isinstance(it, dict)],
                    plan.requested_standards,
                    enforce=require_all_scopes,
                )
                if (
                    bool(getattr(settings, "EARLY_EXIT_COVERAGE_ENABLED", True))
                    and len(plan.requested_standards) >= 2
                ):
                    if len(missing_after_fallback) >= len(missing_before_fallback):
                        diag_trace["multi_query_fallback_early_exit"] = "no_coverage_improvement"
                        diag_trace["missing_scopes_before"] = list(missing_before_fallback)
                        diag_trace["missing_scopes_after"] = list(missing_after_fallback)
                        if missing_after_fallback:
                            diag_trace["error_codes"] = merge_error_codes(
                                diag_trace.get("error_codes"),
                                [RETRIEVAL_CODE_SCOPE_MISMATCH],
                            )
                        self.last_retrieval_diagnostics = RetrievalDiagnostics(
                            contract="advanced",
                            strategy="multi_query",
                            partial=partial,
                            trace=diag_trace,
                            scope_validation=self._validated_scope_payload or {},
                        )
                        return self._to_evidence(mq_items)
                diag_trace["rag_features"] = _features_from_hybrid_trace(trace)
                if hint_trace.get("applied"):
                    diag_trace["search_hint_expansions"] = hint_trace
                if isinstance(self._profile_resolution_context, dict):
                    diag_trace["agent_profile_resolution"] = dict(self._profile_resolution_context)
                if isinstance(mq_items, list):
                    diag_trace.update(_layer_stats([it for it in mq_items if isinstance(it, dict)]))
                self.last_retrieval_diagnostics = RetrievalDiagnostics(
                    contract="advanced",
                    strategy="multi_query",
                    partial=partial,
                    trace=diag_trace,
                    scope_validation=self._validated_scope_payload or {},
                )
                mq_items2 = await _coverage_repair(
                    items=[it for it in mq_items if isinstance(it, dict)],
                    base_trace=diag_trace,
                    reason="multi_query_fallback",
                )
                mq_items2 = self._filter_items_by_min_score(
                    [it for it in mq_items2 if isinstance(it, dict)],
                    trace_target=diag_trace,
                )
                mq_items2 = _reduce_structural_noise(mq_items2)
                return self._to_evidence(mq_items2)

        self.last_retrieval_diagnostics = RetrievalDiagnostics(
            contract="advanced",
            strategy="hybrid",
            partial=False,
            trace={
                "hybrid_trace": trace,
                "timings_ms": dict(timings_ms),
                "deterministic_subquery_semantic_tail": semantic_tail_enabled,
                "mode_policy": {
                    "require_all_requested_scopes": require_all_scopes,
                    "min_clause_refs": min_clause_refs_required,
                    "max_subqueries": mode_max_subqueries,
                },
            },
            scope_validation=self._validated_scope_payload or {},
        )
        if isinstance(items, list) and isinstance(self.last_retrieval_diagnostics.trace, dict):
            self.last_retrieval_diagnostics.trace.update(
                _layer_stats([it for it in items if isinstance(it, dict)])
            )
            self.last_retrieval_diagnostics.trace["rag_features"] = _features_from_hybrid_trace(
                trace
            )
            if hint_trace.get("applied"):
                self.last_retrieval_diagnostics.trace["search_hint_expansions"] = hint_trace
            if isinstance(self._profile_resolution_context, dict):
                self.last_retrieval_diagnostics.trace["agent_profile_resolution"] = dict(
                    self._profile_resolution_context
                )
        items2 = await _coverage_repair(
            items=[it for it in items if isinstance(it, dict)],
            base_trace=self.last_retrieval_diagnostics.trace
            if isinstance(self.last_retrieval_diagnostics.trace, dict)
            else {},
            reason="hybrid",
        )
        items2 = self._filter_items_by_min_score(
            [it for it in items2 if isinstance(it, dict)],
            trace_target=self.last_retrieval_diagnostics.trace
            if isinstance(self.last_retrieval_diagnostics.trace, dict)
            else {},
        )
        items2 = _reduce_structural_noise(items2)
        return self._to_evidence(items2)

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
        agent_profile: AgentProfile | None = None,
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

        def _extract_scope_labels(text: str, candidates: list[str]) -> list[str]:
            source = (text or "").lower()
            seen: set[str] = set()
            ordered: list[str] = []
            for candidate in candidates:
                label = (candidate or "").strip()
                if not label:
                    continue
                key = label.lower()
                if key in source and key not in seen:
                    seen.add(key)
                    ordered.append(label)
            return ordered

        def _row_mentions_scopes(row: dict[str, Any], scope_labels: list[str]) -> set[str]:
            content = str(row.get("content") or "")
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            blob = (content + "\n" + json.dumps(meta, default=str, ensure_ascii=True)).lower()
            present: set[str] = set()
            for label in scope_labels:
                key = (label or "").strip().lower()
                if key and key in blob:
                    present.add(label)
            return present

        def _extract_clause_refs(text: str) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for match in re.findall(r"\b\d+(?:\.\d+)+\b", text or ""):
                value = str(match).strip()
                if not value or value in seen:
                    continue
                seen.add(value)
                ordered.append(value)
            return ordered

        def _clause_match(requested: str, candidate: str) -> bool:
            req = str(requested or "").strip()
            cand = str(candidate or "").strip()
            if not req or not cand:
                return False
            return cand == req or cand.startswith(f"{req}.")

        def _row_matches_clause(item: EvidenceItem, clause_refs: list[str]) -> bool:
            if not clause_refs:
                return False
            row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
            if not isinstance(row, dict):
                return False
            content = str(row.get("content") or "")
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            values: list[str] = []
            for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
                val = str(meta.get(key) or "").strip()
                if val:
                    values.append(val)
            refs_raw = meta.get("clause_refs")
            if isinstance(refs_raw, list):
                values.extend(
                    str(v).strip() for v in refs_raw if isinstance(v, str) and str(v).strip()
                )
            for ref in clause_refs:
                if re.search(rf"\b{re.escape(ref)}(?:\.\d+)*\b", content):
                    return True
                if any(_clause_match(ref, candidate) for candidate in values):
                    return True
            return False

        def _snippet(text: str, limit: int = 240) -> str:
            raw = " ".join((text or "").split())
            if len(raw) <= limit:
                return raw
            return raw[:limit].rstrip() + "..."

        def _extract_subquestions(text: str) -> list[str]:
            raw = str(text or "").strip()
            if not raw:
                return []
            parts = re.split(r"\?+|\n+", raw)
            out: list[str] = []
            seen: set[str] = set()
            for part in parts:
                candidate = " ".join(part.split()).strip(" .:-")
                if len(candidate) < 18:
                    continue
                key = candidate.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(candidate)
            return out

        def _row_clause_label(item: EvidenceItem) -> str:
            row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
            if not isinstance(row, dict):
                return ""
            meta_raw = row.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            for key in ("clause_id", "clause_ref", "clause", "clause_anchor"):
                value = str(meta.get(key) or "").strip()
                if value:
                    return value
            return ""

        def _render_literal_rows(items: list[EvidenceItem], *, max_rows: int) -> str:
            rows: list[str] = []
            seen: set[str] = set()
            for item in items:
                src = str(item.source or "").strip()
                if not src or src in seen:
                    continue
                seen.add(src)
                clause = _row_clause_label(item)
                label = f"Clausula {clause}" if clause else "Afirmacion"
                rows.append(
                    f'{len(rows) + 1}) {label} | "{_snippet(item.content)}" | Fuente ({src})'
                )
                if len(rows) >= max_rows:
                    break
            return "\n".join(rows)

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
        cross_scope_mode = len(plan.requested_standards) >= 2 and not plan.require_literal_evidence
        if cross_scope_mode:
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
        if cross_scope_mode:
            max_ctx = 28
        elif plan.require_literal_evidence:
            max_ctx = 14
        else:
            max_ctx = 18

        clause_refs = _extract_clause_refs(query)
        clause_items = [item for item in ordered_items if _row_matches_clause(item, clause_refs)]
        literal_min_items = 2 if plan.require_literal_evidence and len(clause_items) >= 2 else 1

        query_for_generation = query
        if plan.require_literal_evidence and literal_min_items > 1:
            query_for_generation = (
                query
                + "\n\n[INSTRUCCION INTERNA] Si hay evidencia suficiente, responde con al menos 2 "
                "afirmaciones literales distintas de la misma clausula, cada una con su fuente C#/R#."
            )
        elif not plan.require_literal_evidence:
            subquestions = _extract_subquestions(query)
            if len(subquestions) >= 2:
                numbered = "\n".join(
                    f"{idx + 1}) {item}" for idx, item in enumerate(subquestions[:4])
                )
                query_for_generation = (
                    query + "\n\n[INSTRUCCION INTERNA] La consulta contiene multiples preguntas. "
                    "Responde cada una con subtitulo propio, en el mismo orden, y cita evidencia por seccion:\n"
                    + numbered
                )

        text = await self.service.generate_answer(
            query=query_for_generation,
            context_chunks=labeled,
            agent_profile=agent_profile,
            mode=plan.mode,
            require_literal_evidence=bool(plan.require_literal_evidence),
            max_chunks=max_ctx,
        )

        if not ordered_items and cross_scope_mode:
            lines = [
                f"**{scope}**: No encontrado explicitamente en el contexto recuperado."
                for scope in plan.requested_standards
            ]
            text = "\n\n".join(lines)

        # Guardrail: if the answer ties multiple scopes together but evidence has no explicit bridge,
        # enforce transparent language (inference vs direct citation).
        requested_scopes = list(plan.requested_standards or ())
        candidate_scopes: list[str] = []
        if agent_profile is not None:
            candidate_scopes.extend(list(agent_profile.router.scope_hints.keys()))
            candidate_scopes.extend(list(agent_profile.domain_entities))
        candidate_scopes.extend(requested_scopes)

        mentioned_scopes = _extract_scope_labels(text, candidate_scopes)
        scopes = requested_scopes if len(requested_scopes) >= 2 else mentioned_scopes
        if len(scopes) >= 2 and not plan.require_literal_evidence:
            bridges = 0
            for ev in ordered_items:
                row = ev.metadata.get("row") if isinstance(ev.metadata, dict) else None
                if not isinstance(row, dict):
                    continue
                present = _row_mentions_scopes(row, scopes)
                if len(present) >= 2:
                    bridges += 1
                    break
            already_disclosed = any(
                tok in (text or "").lower()
                for tok in ("inferencia", "interpret", "no encontrado explicitamente")
            )
            if bridges == 0 and not already_disclosed:
                note = (
                    "Nota de trazabilidad: La relacion entre fuentes se presenta como inferencia basada en "
                    "evidencias separadas; no hay un fragmento unico que las vincule explicitamente."
                )
                text = f"{note}\n\n{text}" if (text or "").strip() else note
        if plan.require_literal_evidence:
            # Defense-in-depth: if the provider ignores instructions and returns no markers,
            # append the reviewed references so the validator can trace the answer.
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

            # Ensure literal outputs include at least two grounded rows when available.
            markers = set(re.findall(r"\b[CR]\d+\b", text or ""))
            if literal_min_items > 1 and len(markers) < 2 and len(clause_items) >= 2:
                text = _render_literal_rows(clause_items, max_rows=2)

            low_text = (text or "").strip().lower()
            looks_fallback = low_text.startswith(
                "no encontrado explicitamente"
            ) or low_text.startswith("no encuentro evidencia suficiente")
            if looks_fallback and ordered_items:
                preferred = clause_items if clause_items else ordered_items
                max_rows = 3 if len(clause_refs) == 0 else 2
                rendered = _render_literal_rows(preferred, max_rows=max_rows)
                if rendered:
                    text = rendered
        else:
            low_text = (text or "").strip().lower()
            looks_fallback = low_text.startswith(
                "no encontrado explicitamente"
            ) or low_text.startswith("no encuentro evidencia suficiente")
            if looks_fallback and ordered_items:
                rendered = _render_literal_rows(ordered_items, max_rows=3)
                if rendered:
                    text = rendered

        # Transversal citation contract: when evidence exists, ensure explicit markers C#/R#.
        if [*chunks, *summaries] and not re.search(r"\b[CR]\d+\b", text or ""):
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
