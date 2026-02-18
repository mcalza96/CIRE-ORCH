from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
)
from app.cartridges.models import AgentProfile
from app.agent.policies import extract_requested_scopes


_CLAUSE_RE = re.compile(r"\b\d+(?:\.\d+)+\b")


def _clause_ref_matches(requested: str, candidate: str) -> bool:
    req = str(requested or "").strip()
    cand = str(candidate or "").strip()
    if not req or not cand:
        return False
    return cand == req or cand.startswith(f"{req}.")


def normalize_query_filters(raw_filters: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(raw_filters, dict):
        return None

    out: dict[str, Any] = {}

    source_standard = raw_filters.get("source_standard")
    if isinstance(source_standard, str) and source_standard.strip():
        out["source_standard"] = source_standard.strip()

    source_standards = raw_filters.get("source_standards")
    if not out.get("source_standard") and isinstance(source_standards, list):
        cleaned = [str(item).strip() for item in source_standards if str(item).strip()]
        if cleaned:
            out["source_standards"] = cleaned

    metadata_raw = raw_filters.get("metadata")
    if not isinstance(metadata_raw, dict):
        nested_filters = raw_filters.get("filters")
        metadata_raw = nested_filters if isinstance(nested_filters, dict) else None

    if isinstance(metadata_raw, dict):
        clause_id = metadata_raw.get("clause_id")
        if isinstance(clause_id, str) and clause_id.strip():
            out["metadata"] = {"clause_id": clause_id.strip()}

    if "source_standard" in out and "source_standards" in out:
        out.pop("source_standards", None)

    return out or None


def apply_search_hints(
    query: str,
    profile: AgentProfile | None = None,
) -> tuple[str, dict[str, Any]]:
    text = str(query or "").strip()
    if not text or profile is None or not profile.retrieval.search_hints:
        return text, {"applied": False, "applied_hints": [], "expanded_terms": []}

    lower_text = text.lower()
    expanded_terms: list[str] = []
    applied: list[dict[str, Any]] = []

    for hint in profile.retrieval.search_hints:
        term = str(hint.term or "").strip()
        if not term:
            continue
        term_lower = term.lower()
        if term_lower not in lower_text:
            continue
        additions: list[str] = []
        for item in (str(v or "").strip() for v in hint.expand_to):
            if not item:
                continue
            if item.lower() in lower_text or item in expanded_terms:
                continue
            # Avoid injecting numeric clause refs from semantic hints.
            # Clause filters must come from explicit user text.
            if _CLAUSE_RE.fullmatch(item):
                continue
            additions.append(item)
        if not additions:
            continue
        expanded_terms.extend(additions)
        applied.append(
            {
                "term": term,
                "expand_to": additions,
            }
        )

    if not expanded_terms:
        return text, {"applied": False, "applied_hints": [], "expanded_terms": []}

    expanded_query = f"{text} {' '.join(expanded_terms)}".strip()
    return expanded_query, {
        "applied": True,
        "applied_hints": applied,
        "expanded_terms": expanded_terms,
    }


def extract_clause_refs(text: str, profile: AgentProfile | None = None) -> list[str]:
    patterns = profile.router.reference_patterns if profile is not None else []
    compiled: list[re.Pattern[str]] = []
    for expr in patterns:
        try:
            compiled.append(re.compile(expr, flags=re.IGNORECASE))
        except re.error:
            continue

    if not compiled:
        compiled = [_CLAUSE_RE]

    seen: set[str] = set()
    ordered: list[str] = []
    for pattern in compiled:
        for match in pattern.findall(text or ""):
            value = match if isinstance(match, str) else str(match)
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return ordered


def mode_requires_literal_evidence(
    *,
    mode: str | None,
    profile: AgentProfile | None,
    explicit_flag: bool | None = None,
) -> bool:
    if explicit_flag is not None:
        return bool(explicit_flag)
    mode_name = str(mode or "").strip()
    if not mode_name:
        return False
    if profile is not None:
        mode_cfg = profile.query_modes.modes.get(mode_name)
        if mode_cfg is not None:
            return bool(mode_cfg.require_literal_evidence)
        retrieval_cfg = profile.retrieval.by_mode.get(mode_name)
        if retrieval_cfg is not None:
            return bool(retrieval_cfg.require_literal_evidence)
    return False


def _standard_key(standard: str) -> str:
    # "scope label 45001" -> "45001"
    m = re.search(r"\b(\d{4,5})\b", standard or "")
    return m.group(1) if m else (standard or "").strip()


def _clause_near_standard(query: str, standard: str) -> str | None:
    text = query or ""
    key = _standard_key(standard)
    if not key:
        return None
    m = re.search(rf"\b{re.escape(standard)}\b", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(rf"\b{re.escape(key)}\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    window = text[m.end() : m.end() + 90]
    clause = _CLAUSE_RE.search(window)
    return clause.group(0) if clause else None


def _semantic_tail_for_standard(
    *,
    effective_query: str,
    standard: str,
    clause_refs: list[str],
) -> str:
    text = str(effective_query or "").strip()
    if not text:
        return ""

    cleaned = text
    cleaned = re.sub(rf"\b{re.escape(standard)}\b", " ", cleaned, flags=re.IGNORECASE)

    key = _standard_key(standard)
    if key:
        cleaned = re.sub(
            rf"\biso\s*[-:]?\s*{re.escape(key)}(?::\s*\d{{4}})?\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(rf"\b{re.escape(key)}\b", " ", cleaned, flags=re.IGNORECASE)

    for clause in clause_refs[:8]:
        value = str(clause or "").strip()
        if not value:
            continue
        cleaned = re.sub(
            rf"\bcl(?:a|á)usula\s*{re.escape(value)}\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(rf"\b{re.escape(value)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"[()\[\],;:]+", " ", cleaned)
    return " ".join(part for part in cleaned.split() if part).strip()


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
    clause_norm = str(clause or "").strip()
    if not clause_norm:
        return False
    content = str(row.get("content") or "")
    clause_re = re.compile(rf"\b{re.escape(clause_norm)}(?:\.\d+)*\b")
    if clause_re.search(content):
        return True
    meta = row.get("metadata")
    if isinstance(meta, dict):
        for key in ("clause_id", "clause_ref", "clause"):
            value = meta.get(key)
            if isinstance(value, str) and _clause_ref_matches(clause_norm, value):
                return True
        refs = meta.get("clause_refs")
        if isinstance(refs, list) and any(
            isinstance(item, str) and _clause_ref_matches(clause_norm, item) for item in refs
        ):
            return True
    return False


@dataclass(frozen=True)
class CoverageDecision:
    needs_fallback: bool
    reason: str = ""
    code: str = ""


def decide_multihop_fallback(
    *,
    query: str,
    requested_standards: tuple[str, ...],
    items: list[dict[str, Any]],
    hybrid_trace: dict[str, Any] | None,
    top_k: int = 12,
    profile: AgentProfile | None = None,
) -> CoverageDecision:
    """Decide whether to switch from hybrid to multi-query for better balanced evidence."""
    clause_refs = extract_clause_refs(query, profile=profile)
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
                    code=RETRIEVAL_CODE_SCOPE_MISMATCH,
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
                    code=RETRIEVAL_CODE_CLAUSE_MISSING,
                )

    return CoverageDecision(needs_fallback=False, reason="coverage_ok", code="")


def build_initial_scope_filters(
    *,
    plan_requested: tuple[str, ...],
    mode: str,
    query: str,
    profile: AgentProfile | None = None,
    require_literal_evidence: bool | None = None,
) -> dict[str, Any] | None:
    raw_query = str(query or "").strip()
    effective_query, _ = apply_search_hints(raw_query, profile=profile)
    requested = tuple(plan_requested) or extract_requested_scopes(effective_query, profile=profile)
    filters: dict[str, Any] = {}
    if requested:
        filters["source_standards"] = list(requested)

    # Only narrow to clause_id for strict literal extraction AND explicit user references.
    # CRITICAL DAY 2 FIX: Omit clause_id if multiple standards are requested to avoid cross-standard collision.
    if len(requested) <= 1 and mode_requires_literal_evidence(
        mode=mode,
        profile=profile,
        explicit_flag=require_literal_evidence,
    ):
        clause_refs = extract_clause_refs(raw_query, profile=profile)
        # If the user asked multiple questions in one turn, avoid over-constraining
        # retrieval to a single clause (e.g. "9.3 ..." + a second question).
        split_parts = [
            part
            for part in re.split(r"\?+|\n+", raw_query)
            if isinstance(part, str) and len(" ".join(part.split()).strip()) >= 18
        ]
        is_compound_query = len(split_parts) >= 2
        if clause_refs and not is_compound_query:
            filters["metadata"] = {"clause_id": clause_refs[0]}

    return filters or None


def build_deterministic_subqueries(
    *,
    query: str,
    requested_standards: tuple[str, ...],
    max_queries: int = 4,
    mode: str | None = None,
    require_literal_evidence: bool | None = None,
    include_semantic_tail: bool = True,
    profile: AgentProfile | None = None,
) -> list[dict[str, Any]]:
    raw_query = str(query or "").strip()
    effective_query, _ = apply_search_hints(raw_query, profile=profile)
    clause_refs = extract_clause_refs(raw_query, profile=profile)
    out: list[dict[str, Any]] = []
    used_clause_refs: set[str] = set()
    clause_by_standard: dict[str, str] = {}
    for standard in requested_standards:
        maybe_clause = _clause_near_standard(raw_query, standard)
        if maybe_clause:
            clause_by_standard[str(standard)] = str(maybe_clause)

    # Per-standard subqueries (bounded).
    for standard in requested_standards[:3]:
        clause = _clause_near_standard(raw_query, standard)
        key = _standard_key(standard).lower() or "scope"
        if include_semantic_tail:
            semantic_tail = _semantic_tail_for_standard(
                effective_query=effective_query,
                standard=standard,
                clause_refs=clause_refs,
            )
            tail = semantic_tail or " ".join(clause_refs[:3]) or effective_query[:500]
        else:
            tail = " ".join(clause_refs[:3])
        qtext = f"{standard} {clause or ''} {tail or ''}"
        qtext = " ".join(part for part in qtext.split() if part).strip()
        qtext = qtext[:900].rstrip()
        filters: dict[str, Any] = {"source_standard": standard}
        if clause:
            filters["metadata"] = {"clause_id": clause}
            used_clause_refs.add(clause)
        normalized_filters = normalize_query_filters(filters)
        out.append(
            {
                "id": f"scope_{key}_{(clause or 'general').replace('.', '_')}",
                "query": qtext,
                "k": None,
                "fetch_k": None,
                "filters": normalized_filters,
            }
        )
        if len(out) >= max_queries:
            return out

    # Clause-centric decomposition: force one focused query per missing clause reference.
    # This improves retrieval for dense prompts that cite several clauses at once.
    shared_filters: dict[str, Any] | None = normalize_query_filters(
        {"source_standards": list(requested_standards)} if requested_standards else None
    )
    for clause in clause_refs[:4]:
        clause_norm = str(clause or "").strip()
        if not clause_norm or clause_norm in used_clause_refs:
            continue
        if len(out) >= max_queries:
            return out

        mapped_standard: str | None = None
        for standard in requested_standards:
            near_clause = clause_by_standard.get(str(standard))
            if near_clause and _clause_ref_matches(clause_norm, near_clause):
                mapped_standard = str(standard)
                break

        clause_filters: dict[str, Any] | None
        if mapped_standard:
            clause_filters = normalize_query_filters(
                {
                    "source_standard": mapped_standard,
                    "metadata": {"clause_id": clause_norm},
                }
            )
        elif len(requested_standards) == 1 and requested_standards[0]:
            clause_filters = normalize_query_filters(
                {
                    "source_standard": requested_standards[0],
                    "metadata": {"clause_id": clause_norm},
                }
            )
        else:
            # Ambiguous clause in cross-standard prompts: avoid hard clause filter
            # that can over-constrain unrelated standards.
            clause_filters = shared_filters

        out.append(
            {
                "id": f"clause_{clause_norm.replace('.', '_')}",
                "query": f"{effective_query} clausula {clause_norm}"[:900],
                "k": None,
                "fetch_k": None,
                "filters": clause_filters,
            }
        )

    literal_mode = mode_requires_literal_evidence(
        mode=mode,
        profile=profile,
        explicit_flag=require_literal_evidence,
    )

    # Bridge/documentation impact query.
    if len(out) < max_queries and (not literal_mode or not out):
        out.append(
            {
                "id": "bridge_contexto",
                "query": f"{effective_query} impacto documental evidencia registros cumplimiento riesgos",
                "k": None,
                "fetch_k": None,
                "filters": shared_filters,
            }
        )

    # In literal modes, step-back is reserved for downstream coverage repair, not as a primary query.
    if len(out) < max_queries and not literal_mode:
        out.append(
            {
                "id": "step_back",
                "query": f"principios generales y requisitos clave relacionados: {effective_query}",
                "k": None,
                "fetch_k": None,
                "filters": shared_filters,
            }
        )

    return out[:max_queries]
