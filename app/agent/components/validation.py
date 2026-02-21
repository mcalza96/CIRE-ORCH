from __future__ import annotations

from dataclasses import dataclass

from app.agent.components.parsing import extract_row_standard
from app.agent.types.models import EvidenceItem


@dataclass(frozen=True)
class ValidationSignals:
    scope_answer_mismatch: bool
    scope_evidence_mismatch: bool
    clause_mismatch: bool
    missing_citations: bool
    no_evidence: bool


def classify_validation_issues(issues: list[str]) -> ValidationSignals:
    lowered = [str(issue or "").lower() for issue in issues]
    return ValidationSignals(
        scope_answer_mismatch=any("answer mentions" in item for item in lowered),
        scope_evidence_mismatch=any("evidence includes" in item for item in lowered),
        clause_mismatch=any("literal clause mismatch" in item for item in lowered),
        missing_citations=any("explicit source markers" in item for item in lowered),
        no_evidence=any("no retrieval evidence" in item for item in lowered),
    )


def filter_evidence_by_standards(
    evidence: list[EvidenceItem],
    *,
    allowed_standards: tuple[str, ...],
) -> list[EvidenceItem]:
    if not allowed_standards:
        return evidence
    allowed_upper = {s.upper() for s in allowed_standards if s}
    out: list[EvidenceItem] = []
    for item in evidence:
        std = extract_row_standard(item)
        if not std:
            out.append(item)
            continue
        if any(target in std for target in allowed_upper):
            out.append(item)
    return out


def split_evidence_by_source_prefix(
    evidence: list[EvidenceItem],
) -> tuple[list[EvidenceItem], list[EvidenceItem]]:
    chunks: list[EvidenceItem] = []
    summaries: list[EvidenceItem] = []
    
    for it in evidence:
        # 1. Try metadata inspection (Modern UUID mode)
        meta = it.metadata.get("row", {}) if isinstance(it.metadata, dict) else {}
        row_meta = meta.get("metadata", {}) if isinstance(meta, dict) else {}
        f_source = str(row_meta.get("fusion_source") or "").lower()
        
        if f_source == "raptor":
            summaries.append(it)
            continue
        if f_source in ("chunks", "graph"):
            chunks.append(it)
            continue
            
        # 2. Fallback to legacy prefix check
        src = str(it.source or "").upper()
        if src.startswith("C"):
            chunks.append(it)
        elif src.startswith("R"):
            summaries.append(it)
        else:
            chunks.append(it) # default
            
    return chunks, summaries
