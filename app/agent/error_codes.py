from __future__ import annotations

RETRIEVAL_CODE_EMPTY_RETRIEVAL = "empty_retrieval"
RETRIEVAL_CODE_SCOPE_MISMATCH = "scope_mismatch"
RETRIEVAL_CODE_CLAUSE_MISSING = "clause_missing"
RETRIEVAL_CODE_LOW_SCORE = "low_score"
RETRIEVAL_CODE_GRAPH_FALLBACK_NO_MULTIHOP = "graph_fallback_no_multihop"


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
