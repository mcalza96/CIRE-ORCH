from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agent.policies import extract_requested_standards


_CLAUSE_RE = re.compile(r"\b\d+(?:\.\d+)+\b")


def extract_clause_refs(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CLAUSE_RE.findall(text or ""):
        if match in seen:
            continue
        seen.add(match)
        ordered.append(match)
    return ordered


def _standard_key(standard: str) -> str:
    # "ISO 45001" -> "45001"
    m = re.search(r"\b(\d{4,5})\b", standard or "")
    return m.group(1) if m else (standard or "").strip()


def _clause_near_standard(query: str, standard: str) -> str | None:
    text = query or ""
    key = _standard_key(standard)
    if not key:
        return None
    m = re.search(rf"\bISO\s*[-:]?\s*{re.escape(key)}\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    window = text[m.end() : m.end() + 90]
    clause = _CLAUSE_RE.search(window)
    return clause.group(0) if clause else None


def _row_standard(row: dict[str, Any]) -> str | None:
    meta = row.get("metadata")
    if isinstance(meta, dict):
        std = meta.get("source_standard") or meta.get("scope") or meta.get("standard")
        if isinstance(std, str) and std.strip():
            return std.strip()
    std2 = row.get("source_standard")
    if isinstance(std2, str) and std2.strip():
        return std2.strip()
    return None


def _row_mentions_clause(row: dict[str, Any], clause: str) -> bool:
    content = str(row.get("content") or "")
    if clause and clause in content:
        return True
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("clause_id", "clause_ref", "clause"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip() == clause:
                return True
        refs = meta.get("clause_refs")
        if isinstance(refs, list) and any(
            isinstance(item, str) and item.strip() == clause for item in refs
        ):
            return True
    return False


@dataclass(frozen=True)
class CoverageDecision:
    needs_fallback: bool
    reason: str = ""


def decide_multihop_fallback(
    *,
    query: str,
    requested_standards: tuple[str, ...],
    items: list[dict[str, Any]],
    hybrid_trace: dict[str, Any] | None,
    top_k: int = 12,
) -> CoverageDecision:
    """Decide whether to switch from hybrid to multi-query for better balanced evidence."""
    clause_refs = extract_clause_refs(query)
    top = items[: max(1, int(top_k))]

    present_standards: set[str] = set()
    for row in top:
        std = _row_standard(row)
        if std:
            present_standards.add(std.upper())

    req_upper = [s.upper() for s in requested_standards if s]
    if len(req_upper) >= 2:
        missing = [s for s in req_upper if s not in present_standards]
        if missing:
            planner_multihop = bool((hybrid_trace or {}).get("planner_multihop", False))
            # If planner already did multihop, missing standards can still happen. Require fallback only
            # when planner did not do multihop (or trace missing) to avoid redundant calls.
            if not planner_multihop:
                return CoverageDecision(
                    needs_fallback=True,
                    reason=f"missing_standards_in_topk: {', '.join(missing[:3])}",
                )

    if clause_refs:
        missing_clauses: list[str] = []
        for clause in clause_refs:
            if not any(_row_mentions_clause(row, clause) for row in top):
                missing_clauses.append(clause)
        if missing_clauses:
            planner_multihop = bool((hybrid_trace or {}).get("planner_multihop", False))
            if not planner_multihop:
                return CoverageDecision(
                    needs_fallback=True,
                    reason=f"missing_clause_refs_in_topk: {', '.join(missing_clauses[:3])}",
                )

    return CoverageDecision(needs_fallback=False, reason="coverage_ok")


def build_initial_scope_filters(
    *, plan_requested: tuple[str, ...], mode: str, query: str
) -> dict[str, Any] | None:
    requested = tuple(plan_requested) or extract_requested_standards(query)
    filters: dict[str, Any] = {}
    if requested:
        filters["source_standards"] = list(requested)

    # Only narrow to clause_id for strict literal extraction; interpretive modes keep recall.
    if mode in {"literal_normativa", "literal_lista"}:
        clause_refs = extract_clause_refs(query)
        if clause_refs:
            filters["metadata"] = {"clause_id": clause_refs[0], "clause_refs": clause_refs}

    return filters or None


def build_deterministic_subqueries(
    *,
    query: str,
    requested_standards: tuple[str, ...],
    max_queries: int = 6,
) -> list[dict[str, Any]]:
    clause_refs = extract_clause_refs(query)
    out: list[dict[str, Any]] = []

    # Per-standard subqueries (bounded).
    for standard in requested_standards[:3]:
        clause = _clause_near_standard(query, standard)
        key = _standard_key(standard).lower() or "iso"
        qtext = f"{standard} {clause or ''} " + " ".join(clause_refs[:3])
        qtext = " ".join(part for part in qtext.split() if part).strip()
        filters: dict[str, Any] = {"source_standard": standard}
        if clause:
            filters["metadata"] = {"clause_id": clause}
        out.append(
            {
                "id": f"iso{key}_{(clause or 'general').replace('.', '_')}",
                "query": qtext,
                "k": None,
                "fetch_k": None,
                "filters": filters,
            }
        )
        if len(out) >= max_queries:
            return out

    # Bridge/documentation impact query.
    shared_filters: dict[str, Any] | None = (
        {"source_standards": list(requested_standards)} if requested_standards else None
    )

    if len(out) < max_queries:
        out.append(
            {
                "id": "bridge_contexto",
                "query": f"{query} impacto documental evidencia registros cumplimiento riesgos",
                "k": None,
                "fetch_k": None,
                "filters": shared_filters,
            }
        )

    if len(out) < max_queries:
        out.append(
            {
                "id": "step_back",
                "query": f"principios generales y requisitos clave relacionados: {query}",
                "k": None,
                "fetch_k": None,
                "filters": shared_filters,
            }
        )

    return out[:max_queries]
