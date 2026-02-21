
import re

from app.agent.types.models import EvidenceItem
from app.agent.components.parsing import extract_row_standard

def calculate_graph_retry_reason(trace: dict[str, object]) -> str:
    if not isinstance(trace, dict):
        return ""
    rag_features = trace.get("rag_features")
    if isinstance(rag_features, dict):
        fallback_used = bool(rag_features.get("fallback_used", False))
        planner_multihop = bool(rag_features.get("planner_multihop", False))
        if fallback_used and not planner_multihop:
            return "graph_fallback_no_multihop"
    missing_scopes = trace.get("missing_scopes")
    if isinstance(missing_scopes, list) and missing_scopes:
        return "scope_mismatch"
    return ""


def derive_graph_contract(trace: dict[str, object]) -> dict[str, object]:
    if not isinstance(trace, dict):
        return {
            "graph_used": False,
            "graph_strategy": "none",
            "anchor_count": 0,
            "anchor_entities": [],
            "graph_paths_count": 0,
            "community_hits": 0,
        }

    rag_features_raw = trace.get("rag_features")
    rag_features = rag_features_raw if isinstance(rag_features_raw, dict) else {}
    strategy = "none"
    if bool(rag_features.get("planner_multihop", False)):
        strategy = "hybrid"
    elif bool(rag_features.get("planner_used", False)):
        strategy = "local"
    elif bool(rag_features.get("fallback_used", False)):
        strategy = "global"

    missing_scopes = trace.get("missing_scopes")
    scope_mismatch_reason = ""
    if isinstance(missing_scopes, list) and missing_scopes:
        scope_mismatch_reason = "missing_scopes"

    def _to_int(value: object) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return 0

    return {
        "graph_used": strategy != "none",
        "graph_strategy": strategy,
        "anchor_count": _to_int(trace.get("layer_0")),
        "anchor_entities": trace.get("anchor_entities")
        if isinstance(trace.get("anchor_entities"), list)
        else [],
        "graph_paths_count": _to_int(trace.get("layer_1")),
        "community_hits": _to_int(trace.get("layer_2")),
        "scope_mismatch_reason": scope_mismatch_reason,
    }

def derive_graph_gaps(
    *,
    query: str,
    documents: list[EvidenceItem],
    contract: dict[str, object],
) -> dict[str, object]:
    requested = {
        f"ISO {match}"
        for match in re.findall(r"\biso\s*[-:]?\s*(\d{4,5})\b", query, flags=re.IGNORECASE)
    }
    found: set[str] = set()
    for item in documents:
        scope = extract_row_standard(item)
        if not scope:
            continue
        digits = "".join(ch for ch in scope if ch.isdigit())
        if digits:
            found.add(f"ISO {digits}")

    disconnected_entities = sorted(scope for scope in requested if scope not in found)
    clause_refs = sorted(set(re.findall(r"\b\d+(?:\.\d+)+\b", query or "")))
    missing_links = [
        f"{scope} -> clausula {clause}"
        for scope in disconnected_entities
        for clause in clause_refs[:2]
    ]

    out = dict(contract)
    out["disconnected_entities"] = disconnected_entities
    out["missing_links"] = missing_links[:4]
    return out
