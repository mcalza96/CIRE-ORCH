from __future__ import annotations

import re
from typing import Any

from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_LOW_SCORE,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    RETRIEVAL_CODE_EMPTY_RETRIEVAL,
    RETRIEVAL_CODE_TIMEOUT,
    RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE,
)
from app.agent.models import RetrievalDiagnostics, ToolResult
from app.graph.universal.state import UniversalState
from app.graph.universal.utils import _non_negative_int

_RETRYABLE_REASONS = {
    RETRIEVAL_CODE_EMPTY_RETRIEVAL,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_LOW_SCORE,
    RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
    RETRIEVAL_CODE_TIMEOUT,
    RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE,
}


def _is_retryable_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    return text in _RETRYABLE_REASONS


def _extract_retry_signal_from_retrieval(state: UniversalState, last: ToolResult | None) -> str:
    if not isinstance(last, ToolResult):
        return ""
    if last.tool != "semantic_retrieval" or not last.ok:
        return ""

    output = dict(last.output or {})
    chunk_count = _non_negative_int(output.get("chunk_count"), default=0)
    summary_count = _non_negative_int(output.get("summary_count"), default=0)
    if chunk_count + summary_count <= 0:
        return RETRIEVAL_CODE_EMPTY_RETRIEVAL

    retrieval = state.get("retrieval")
    if not isinstance(retrieval, RetrievalDiagnostics):
        return ""

    scope_validation = dict(retrieval.scope_validation or {})
    if scope_validation.get("valid") is False:
        return RETRIEVAL_CODE_SCOPE_MISMATCH

    trace = dict(retrieval.trace or {})
    missing_scopes = trace.get("missing_scopes")
    if isinstance(missing_scopes, list) and missing_scopes:
        return RETRIEVAL_CODE_SCOPE_MISMATCH

    missing_clause_refs = trace.get("missing_clause_refs")
    if isinstance(missing_clause_refs, list) and missing_clause_refs:
        return RETRIEVAL_CODE_CLAUSE_MISSING

    error_codes_raw = trace.get("error_codes")
    error_codes = error_codes_raw if isinstance(error_codes_raw, list) else []
    for code in (
        RETRIEVAL_CODE_SCOPE_MISMATCH,
        RETRIEVAL_CODE_CLAUSE_MISSING,
        RETRIEVAL_CODE_LOW_SCORE,
        RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP,
        RETRIEVAL_CODE_TIMEOUT,
        RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE,
    ):
        if code in error_codes:
            return code
    return ""


def _infer_expression_from_query(query: str) -> str:
    text = str(query or "")
    # Conservative extraction: only plain arithmetic expressions.
    match = re.search(r"(\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\(?\d+(?:\.\d+)?\)?)+)", text)
    if not match:
        return ""
    return str(match.group(1)).strip()


def _count_section_markers(text: str, markers: list[str]) -> int:
    lowered = str(text or "").lower()
    return sum(1 for marker in markers if str(marker or "").strip().lower() in lowered)
