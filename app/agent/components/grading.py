from __future__ import annotations

import re
from typing import Any

from app.agent.components.parsing import extract_row_standard
from app.agent.models import EvidenceItem, RetrievalPlan
from app.core.config import settings


def looks_relevant_retrieval(
    documents: list[EvidenceItem],
    plan: RetrievalPlan,
    *,
    query: str,
) -> tuple[bool, str]:
    if not documents:
        return False, "empty_retrieval"

    contentful = [doc for doc in documents if str(doc.content or "").strip()]
    if not contentful:
        return False, "empty_content"

    if plan.requested_standards:
        requested = [scope.upper() for scope in plan.requested_standards if scope]
        matched = set()
        for doc in contentful:
            row_standard = extract_row_standard(doc)
            if not row_standard:
                continue
            for scope in requested:
                if scope in row_standard or row_standard in scope:
                    matched.add(scope)
        if len(requested) >= 2 and len(matched) < 2:
            return False, "scope_mismatch"
        if not matched:
            return False, "scope_mismatch"

    query_clause_refs = re.findall(r"\b\d+(?:\.\d+)+\b", query or "")
    if bool(plan.require_literal_evidence) and query_clause_refs:
        found_clause_anchor = False
        for doc in contentful:
            row = doc.metadata.get("row") if isinstance(doc.metadata, dict) else None
            if not isinstance(row, dict):
                continue
            row_dict: dict[str, Any] = row
            meta_raw = row_dict.get("metadata")
            meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
            clause_meta = [
                str(meta["clause_id"] if "clause_id" in meta else "").strip(),
                str(meta["clause_ref"] if "clause_ref" in meta else "").strip(),
                str(meta["clause"] if "clause" in meta else "").strip(),
            ]
            if any(value for value in clause_meta):
                found_clause_anchor = True
                break
        if not found_clause_anchor:
            content_blob = "\n".join(doc.content for doc in contentful)
            if not any(
                re.search(rf"\b{re.escape(clause)}(?:\.\d+)*\b", content_blob)
                for clause in query_clause_refs
            ):
                return False, "clause_missing"

    scored = [float(doc.score) for doc in contentful if doc.score is not None]
    min_avg_score = float(getattr(settings, "ORCH_GRAPH_MIN_AVG_SCORE", 0.12) or 0.12)
    has_meaningful_score_signal = bool(scored) and any(score > 0.0 for score in scored)
    if has_meaningful_score_signal and (sum(scored) / max(1, len(scored))) < min_avg_score:
        return False, "low_score"

    return True, "ok"
