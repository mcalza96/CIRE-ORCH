from __future__ import annotations

import re
from typing import Any

from app.agent.models import EvidenceItem, RetrievalPlan


def extract_row_standard(item: EvidenceItem) -> str:
    row = item.metadata.get("row") if isinstance(item.metadata, dict) else None
    if not isinstance(row, dict):
        return ""
    row_dict: dict[str, Any] = row
    meta_raw = row_dict.get("metadata")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    candidates = [
        meta["source_standard"] if "source_standard" in meta else None,
        meta["standard"] if "standard" in meta else None,
        meta["scope"] if "scope" in meta else None,
        row_dict.get("source_standard"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def build_retry_focus_query(*, query: str, plan: RetrievalPlan, reason: str) -> str:
    clause_refs = re.findall(r"\b\d+(?:\.\d+)+\b", query or "")
    standards = [scope for scope in plan.requested_standards if scope]
    hints: list[str] = []
    if standards:
        hints.append("alcance=" + ", ".join(standards))
    if clause_refs:
        hints.append("clausulas=" + ", ".join(clause_refs[:4]))
    hints.append(f"motivo_retry={reason}")
    return f"{query}\n\n[RETRY_FOCUS] {' | '.join(hints)}"
