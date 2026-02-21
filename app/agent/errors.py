from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScopeValidationError(Exception):
    message: str
    violations: list[dict[str, Any]]
    warnings: list[dict[str, Any]] | None = None
    normalized_scope: dict[str, Any] | None = None
    query_scope: dict[str, Any] | None = None


# --- Error Codes (merged from error_codes.py) ---
RETRIEVAL_CODE_EMPTY_RETRIEVAL = "empty_retrieval"
RETRIEVAL_CODE_SCOPE_MISMATCH = "scope_mismatch"
RETRIEVAL_CODE_CLAUSE_MISSING = "clause_missing"
RETRIEVAL_CODE_LOW_SCORE = "low_score"
RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP = "graph_fallback_no_multihop"
RETRIEVAL_CODE_TIMEOUT = "timeout"
RETRIEVAL_CODE_UPSTREAM_UNAVAILABLE = "upstream_unavailable"
RETRIEVAL_CODE_INVALID_RESPONSE = "invalid_response"
RETRIEVAL_CODE_RETRIEVAL_TIMEOUT = "retrieval_timeout"
RETRIEVAL_CODE_RETRIEVAL_UPSTREAM_UNAVAILABLE = "retrieval_upstream_unavailable"
RETRIEVAL_CODE_RETRIEVAL_INVALID_RESPONSE = "retrieval_invalid_response"


def merge_error_codes(*groups: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, list):
            continue
        for raw in group:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out
