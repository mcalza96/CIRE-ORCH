from __future__ import annotations

import re
from typing import Any
from app.agent.error_codes import (
    RETRIEVAL_CODE_CLAUSE_MISSING,
    RETRIEVAL_CODE_SCOPE_MISMATCH,
)
from app.core.config import settings


_INTRO_QUERY_TOKENS = (
    "introduccion",
    "introducción",
    "prefacio",
    "preface",
    "prologo",
    "prólogo",
    "foreword",
)

_TOC_TOKENS = (
    "indice",
    "índice",
    "tabla de contenido",
    "table of contents",
    "table of content",
    "toc",
)


def _item_text_haystack(item: dict[str, Any]) -> str:
    content = str(item.get("content") or "").strip().lower()
    meta_raw = item.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    row_raw = meta.get("row")
    row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
    row_meta_raw = row.get("metadata")
    row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
    title = str(row_meta.get("title") or "").strip().lower()
    heading = str(row_meta.get("heading") or row_meta.get("section_title") or "").strip().lower()
    return "\n".join(part for part in (title, heading, content[:1200]) if part)


def _matches_intro_intent(item: dict[str, Any]) -> bool:
    hay = _item_text_haystack(item)
    return any(token in hay for token in _INTRO_QUERY_TOKENS)


def calculate_layer_stats(raw_items: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    raptor = 0
    for it in raw_items:
        meta_raw = it.get("metadata")
        meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}

        # Resilient row lookup: Check row key first, then fallback to flat meta
        row_raw = meta.get("row")
        row = row_raw if isinstance(row_raw, dict) else meta

        layer = str(row.get("source_layer") or "").strip() or "unknown"
        counts[layer] = counts.get(layer, 0) + 1

        # Resilient metadata lookup within row
        row_meta_raw = row.get("metadata")
        row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else row

        if bool(row_meta.get("is_raptor_summary", False)) or row.get("source_layer") == "raptor":
            raptor += 1
    return {"layer_counts": counts, "raptor_summary_count": raptor}


def features_from_hybrid_trace(trace: Any) -> dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    return {
        "engine_mode": str(trace.get("engine_mode") or ""),
        "planner_used": bool(trace.get("planner_used", False)),
        "planner_multihop": bool(trace.get("planner_multihop", False)),
        "fallback_used": bool(trace.get("fallback_used", False)),
    }


def extract_row_scope(row: dict[str, Any]) -> str:
    """
    Ultra-resilient Scope Extraction.
    Exhaustively searches for 'source_standard', 'scope', or 'standard'
    across ALL known nesting patterns from database to API response.
    """
    meta = row.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    nested_row = row.get("row")
    nested_row = nested_row if isinstance(nested_row, dict) else {}

    search_targets = [
        row,  # 1. API item root
        meta,  # 2. item.metadata (JSONB or full row)
        meta.get("metadata", {}),  # 3. item.metadata.metadata (JSONB inner)
        nested_row,  # 4. item.row (orchestrator wrap)
        nested_row.get("metadata", {}),  # 5. item.row.metadata
    ]

    # Also check for row inside metadata (double nesting)
    meta_row = meta.get("row")
    if isinstance(meta_row, dict):
        search_targets.append(meta_row)
        inner = meta_row.get("metadata")
        if isinstance(inner, dict):
            search_targets.append(inner)

    keys = ("source_standard", "scope", "standard")
    for target in search_targets:
        if not isinstance(target, dict):
            continue
        for key in keys:
            val = target.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip().upper()

    return "UNKNOWN"


def find_missing_scopes(
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
        scope = extract_row_scope(row)
        if scope:
            present.add(scope)
    req_upper = [s.upper() for s in requested if s]
    return [s for s in req_upper if s not in present]


def row_matches_clause_ref(row: dict[str, Any], clause_ref: str) -> bool:
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


def find_missing_clause_refs(
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
            if row_matches_clause_ref(row, ref):
                present.add(ref)
    if len(present) >= min_required:
        return []
    missing = [ref for ref in required if ref not in present]
    shortfall = max(0, min_required - len(present))
    return missing[:shortfall] if shortfall else missing


def looks_structural_toc(item: dict[str, Any]) -> bool:
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
    if any(token in hay for token in _TOC_TOKENS):
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


def looks_editorial_front_matter(item: dict[str, Any]) -> bool:
    content = str(item.get("content") or "").strip().lower()
    meta_raw = item.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    row_raw = meta.get("row")
    row: dict[str, Any] = row_raw if isinstance(row_raw, dict) else {}
    row_meta_raw = row.get("metadata")
    row_meta: dict[str, Any] = row_meta_raw if isinstance(row_meta_raw, dict) else {}
    title = str(row_meta.get("title") or "").strip().lower()
    heading = str(row_meta.get("heading") or row_meta.get("section_title") or "").strip().lower()
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


def reduce_structural_noise(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    q = str(query or "").lower()
    if any(token in q for token in _TOC_TOKENS):
        return items
    if any(token in q for token in ("prologo", "prólogo", "preface", "foreword")):
        return items

    toc_items: list[dict[str, Any]] = []
    editorial_items: list[dict[str, Any]] = []
    body_items: list[dict[str, Any]] = []
    for item in items:
        if looks_structural_toc(item):
            toc_items.append(item)
            continue
        if looks_editorial_front_matter(item):
            editorial_items.append(item)
            continue
        body_items.append(item)

    if not body_items:
        intro_intent = any(token in q for token in _INTRO_QUERY_TOKENS)
        if intro_intent:
            intro_editorial = [item for item in editorial_items if _matches_intro_intent(item)]
            intro_toc = [item for item in toc_items if _matches_intro_intent(item)]
            if intro_editorial:
                return intro_editorial[:4] + intro_toc[:1]
            if editorial_items:
                return editorial_items[:4] + toc_items[:1]
            # Last resort: avoid sending a wide TOC-only pack for intro-like queries.
            return items[: min(6, len(items))]
        return items
    if not toc_items and not editorial_items:
        return items

    # Keep the best body evidence first, then a tiny structural tail for transparency.
    return body_items + editorial_items[:1] + toc_items[:1]
